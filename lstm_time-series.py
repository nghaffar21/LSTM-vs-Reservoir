""" Environmental Time-Series Forecasting with PyTorch Lightning
Using features Wind Speed, Cloud Cover, and Pressure we try to predict target variables Temperature and Humidity.
Dataset: https://www.kaggle.com/datasets/muthuj7/weather-dataset """

import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc
import math
import matplotlib

import pandas as pd
import numpy as np
from tqdm import tqdm
from tqdm.notebook import tqdm
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta
from torchmetrics.regression import WeightedMeanAbsolutePercentageError
from torch.utils.tensorboard import SummaryWriter
import torch.cuda

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pytorch_fabric")

import mlflow
from mlflow.pytorch import MlflowModelCheckpointCallback

import optuna

import yaml

import joblib

tqdm.pandas()

#%matplotlib inline
#%config InlineBackend.figure_format='retina'

sns.set_theme(style='whitegrid', palette='muted', font_scale=1.2)

HAPPY_COLORS_PALETTE = ["#01BEFE", "#FFDD00", "#FF7D00", "#FF006D", "#93D30C", "#8F00FF"]

sns.set_palette(sns.color_palette(HAPPY_COLORS_PALETTE))

rcParams['figure.figsize'] = 14, 10

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# Load the dataset
df = pd.read_csv('weatherHistory.csv')
"""
# Set the hyperparameters and model architecture in a YAML file
with open("hyperparams.yml", "r") as f:
    config = yaml.safe_load(f)

TRAIN_SPLIT = config["data"]["train_split"]
VAL_SPLIT = config["data"]["val_split"]
SEQ_LENGTH = config["data"]["seq_length"]

N_HIDDEN = config["model"]["n_hidden"]
N_LAYERS = config["model"]["n_layers"]
DROPOUT = config["model"]["dropout"]

N_EPOCHS = config["training"]["n_epochs"]
BATCH_SIZE = config["training"]["batch_size"]
LEARNING_RATE = config["training"]["learning_rate"]
num_workers_train = config["training"]["num_workers_train"]
num_workers_val = config["training"]["num_workers_val"]
num_workers_test = config["training"]["num_workers_test"]

early_stopping_patience = config["callbacks"]["early_stopping_patience"] 
"""
# Rename the column "Loud Cover" to "Cloud Cover" and "Formatted Date" to "date"
df = df.rename(columns={"Formatted Date": "date"})
df = df.rename(columns={"Loud Cover": "Cloud Cover"})

# Convert datetime column to datetime object
df['date'] = pd.to_datetime(df['date'], utc=True)

# Sort by date
df = df.sort_values('date').reset_index(drop=True)
# print(df.head(60))

#print(df.head())
#print(df.shape)
#print(df.columns)

# Check for missing values A: None
# print(df.isnull().sum())

# Drop columns that won't be used
features_df = df.drop(['Summary', 'Daily Summary', 'Precip Type', 'Daily Summary', 'Wind Bearing (degrees)', 'Visibility (km)', 'Apparent Temperature (C)'], axis=1)

# How many days of each month are recorded? Answer: All days.
""" for j in range(1, 13):
    count = 0
    for i in range(len(df)):
        if(df['date'][i].month == j and df['date'][i].year == 2016):
            count += 1
    print("days of each month captured: ", count / 24) """

# Are there any gaps in the time series? A: Yes, there are 74807 4hrs gaps.
# These gaps could be investigated further, but for now we will just note their existence.
""" deltas = df['date'].diff()[1:]
gaps = deltas[deltas > timedelta(hours=1)]
print(f"Number of gaps: {len(gaps)}")
if len(gaps) > 0:
    print("\nGaps found:")
    print(gaps)
    print(f"\nLargest gap: {gaps.min()}") """

# Divide the date into cyclical features using sine and cosine encoding to capture the cyclical patterns of the data.
# Hour of Day: Because weather is cyclical over a day
# Day of Week: In urban areas, human activity can affect weather patterns(e.g. weekends might have different pollution levels)
# Month of Year: To capture seasonal variations in weather

features_df.insert(0, 'Hour of Day Sine', np.sin(2 * np.pi * features_df['date'].dt.hour / 24))
features_df.insert(1, 'Hour of Day Cosine', np.cos(2 * np.pi * features_df['date'].dt.hour / 24))

features_df.insert(2, 'Day of Week Sine', np.sin(2 * np.pi * features_df['date'].dt.dayofweek / 7))
features_df.insert(3, 'Day of Week Cosine', np.cos(2 * np.pi * features_df['date'].dt.dayofweek / 7))

features_df.insert(4, 'Month of Year Sine', np.sin(2 * np.pi * (features_df['date'].dt.month - 1) / 12))
features_df.insert(5, 'Month of Year Cosine', np.cos(2 * np.pi * (features_df['date'].dt.month - 1) / 12))

features_df = features_df.drop(['date'], axis=1) # Drop the date column as we no longer need it

# Define PyTorch Dataset
class WeatherDataset(Dataset):
    def __init__(self, sequences):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence, label_temp, label_humidity = self.sequences[idx]

        return dict(
            sequence=torch.Tensor(sequence.to_numpy()),
            label_temp=torch.tensor(label_temp).float(),
            label_humidity=torch.tensor(label_humidity).float()
        )
    
class WeatherDataModule(pl.LightningDataModule):
    def __init__(self, batch_size, train_sequences, test_sequences, val_sequences=None):
        super().__init__()
        self.train_sequences = train_sequences

        self.val_sequences = None
        if val_sequences is not None: self.val_sequences = val_sequences

        self.test_sequences = test_sequences
        self.batch_size = batch_size

    def setup(self, stage=None):
        self.train_dataset = WeatherDataset(self.train_sequences)
        if self.val_sequences is not None: self.val_dataset = WeatherDataset(self.val_sequences)
        self.test_dataset = WeatherDataset(self.test_sequences)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=False, num_workers=7,  pin_memory=True, prefetch_factor=4)
    
    def val_dataloader(self):
        if self.val_sequences is not None: return DataLoader(self.val_dataset, batch_size = self.batch_size, shuffle=False, num_workers=1, pin_memory=True, prefetch_factor=4)
        return DataLoader([], batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size = self.batch_size, shuffle=False, num_workers=1, pin_memory=True, prefetch_factor=4)
    
class WeatherPredictionModel(nn.Module):
    def __init__(self, n_features, n_hidden, n_layers, n_dropout):
        super(WeatherPredictionModel, self).__init__()
    
        self.n_hidden = n_hidden

        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=n_hidden,
            num_layers=n_layers,
            batch_first=True,
            dropout=n_dropout
        )

        self.temp_head = nn.Linear(in_features=n_hidden, out_features=1)
        self.humidity_head = nn.Linear(in_features=n_hidden, out_features=1)

    def forward(self, x):
        self.lstm.flatten_parameters()

        _, (hidden, _) = self.lstm(x)
        last_time_step = hidden[-1]  # Shape: (batch_size, n_hidden)
        
        temp_pred = self.temp_head(last_time_step)  # Shape: (batch_size, 1)
        humidity_pred = self.humidity_head(last_time_step)  # Shape: (batch_size, 1)
        
        # Concatenate along dimension 1 to get (batch_size, 2)
        return torch.cat([temp_pred, humidity_pred], dim=1)

class WeatherPredictor(pl.LightningModule):
    def __init__(self, n_features: int, n_hidden, n_layers, n_dropout, n_learning_rate):
        super().__init__()
        self.model = WeatherPredictionModel(n_features, n_hidden, n_layers, n_dropout)
        self.criterion = nn.MSELoss()
        self.learning_rate = n_learning_rate

    def forward(self, x, labels=None):
        output = self.model(x)  # Shape: (batch_size, 2)
        loss = 0
        if labels is not None:
            loss = self.criterion(output, labels)  # labels should be (batch_size, 2)
        return loss, output

    def training_step(self, batch, batch_idx):
        sequences = batch['sequence']  # Shape: (batch_size, seq_len, n_features)
        
        # Assume your dataset returns separate temperature and humidity labels
        label_temp = batch['label_temp']  # Shape: (batch_size,)
        label_humidity = batch['label_humidity']  # Shape: (batch_size,)
        
        # Stack them into (batch_size, 2)
        labels = torch.stack([label_temp, label_humidity], dim=1)  # Shape: (batch_size, 2)

        loss, outputs = self(sequences, labels)  # outputs: (batch_size, 2)
        
        # Optional: Log individual losses for monitoring
        #preds_temp = outputs[:, 0]  # Shape: (batch_size,)
        #preds_humidity = outputs[:, 1]  # Shape: (batch_size,)
        
        #loss_temp = self.criterion(preds_temp, label_temp)
        #loss_humidity = self.criterion(preds_humidity, label_humidity)
        
        self.log('train_loss', loss, prog_bar=True, logger=True)
        #self.log('train_loss_temp', loss_temp, logger=True)
        #self.log('train_loss_humidity', loss_humidity, logger=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        sequences = batch['sequence']
        label_temp = batch['label_temp']
        label_humidity = batch['label_humidity']
        labels = torch.stack([label_temp, label_humidity], dim=1)

        loss, outputs = self(sequences, labels)
        
        self.log('val_loss', loss, prog_bar=True, logger=True)
        
        return loss
    
    def test_step(self, batch, batch_idx):
        sequences = batch['sequence']
        label_temp = batch['label_temp']
        label_humidity = batch['label_humidity']
        labels = torch.stack([label_temp, label_humidity], dim=1)

        loss, outputs = self(sequences, labels)
        
        self.log('test_loss', loss, prog_bar=True, logger=True)
        
        return loss
    
    def configure_optimizers(self):
        return optim.AdamW(self.parameters(), lr=self.learning_rate)

# Separate the training, validation, and test sets
def split_data():

    # Separate the training, validation, and test sets
    train_size = int(len(features_df) * 0.7)
    val_size = int(len(features_df) * 0.15)

    train_df, val_df, test_df = features_df[:train_size], features_df[train_size:train_size+val_size], features_df[train_size+val_size:]

    #print(train_df.shape, val_df.shape, test_df.shape)
    #print(features_df.shape)
    return train_df, val_df, test_df

# Scale the data using MinMaxScaler
def scale_data(train_df, val_df, test_df):

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler = scaler.fit(train_df)

    train_df = pd.DataFrame(scaler.transform(train_df), index = train_df.index, columns=train_df.columns)
    val_df = pd.DataFrame(scaler.transform(val_df), index = val_df.index, columns=val_df.columns)
    test_df = pd.DataFrame(scaler.transform(test_df), index = test_df.index, columns=test_df.columns)

    # print("Some train_df temp values: ", train_df['Temperature (C)'].head(10))
    # print("Some val_df temp values: ", val_df['Temperature (C)'].head(10))

    return scaler, train_df, val_df, test_df

# Create Data Sequences
#def create_data_sequences(train_df, val_df, test_df):
def create_sequences(data, seq_length):
    sequences = []
    
    for i in range(len(data) - seq_length):

        sequence = data.iloc[i:(i+seq_length)]

        label_position = i+seq_length
        label_temp = data.iloc[label_position]['Temperature (C)']
        label_humidity = data.iloc[label_position]['Humidity']

        sequences.append((sequence, label_temp, label_humidity))
        
    return sequences

#SEQ_LENGTH = 24 # Using past 24 hours to predict the next hour

#train_sequences = create_sequences(train_df, SEQ_LENGTH)
#val_sequences = create_sequences(val_df, SEQ_LENGTH)
#test_sequences = create_sequences(test_df, SEQ_LENGTH)

#print(train_sequences[0][0])  # First sequence input
#print(train_sequences[0][1])  # First sequence temperature label
#print(train_sequences[0][2])  # First sequence humidity label

#print("train_sequences.shape = ", train_sequences[0][0].shape)
#print("train_df.shape = ", train_df.shape)

#return train_sequences, val_sequences, test_sequences

# Fit the model
def fit_model(model, data_module, train_df, trial_number, sanity_check = -1):

    has_validation = data_module.val_sequences is not None

    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints",
        filename=f"trial_{trial_number}",
        save_top_k=1,
        verbose=True,
        monitor="val_loss",
        mode="min"
    )

    # Experiment with MLflow
    #mlflow.set_tracking_uri("sqlite:///mlflow.db")
    #mlflow.set_experiment("my-first-experiment")

    # Print connection information
    #print(f"MLflow Tracking URI: {mlflow.get_tracking_uri()}")
    #print(f"Active Experiment: {mlflow.get_experiment_by_name('my-first-experiment')}")

    # Test logging
    #with mlflow.start_run():
    #    mlflow.log_param("test_param", "test_value")
    #    print("✓ Successfully connected to MLflow!")

    #print("✓ MLflow connected. Test logging removed to avoid nested run conflicts.")

    # mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000

    # Enable autologging for pytorch
    mlflow.pytorch.autolog(log_models=True)
    #mflow_checkpoint_callback = MlflowModelCheckpointCallback()  # What does this do?

    # TensorBoard Logger
    logger = TensorBoardLogger("lightning_logs", name="temperature-humidity")

    # To see the results of tensorboard: tensorboard --logdir=lightning_logs

    callbacks = []

    if has_validation:
        early_stopping_callback = EarlyStopping(monitor="val_loss", patience=12)
        checkpoint_callback = ModelCheckpoint(
            dirpath="checkpoints",
            filename=f"trial_{trial_number}",
            save_top_k=1,
            verbose=True,
            monitor="val_loss",
            mode="min",
        )
        callbacks = [early_stopping_callback, checkpoint_callback]
    else:
        # No validation: only save final checkpoint
        checkpoint_callback = ModelCheckpoint(
            dirpath="checkpoints",
            filename=f"trial_{trial_number}",
            save_top_k=1,
            verbose=True,
            monitor=None  # <-- IMPORTANT
        )
        callbacks = [checkpoint_callback]

    callbacks.append(MlflowModelCheckpointCallback())

    trainer = pl.Trainer(
        logger=logger,
        callbacks=callbacks, #mflow_checkpoint_callback
        max_epochs=1,
        devices=1,
        accelerator='gpu',
        num_sanity_val_steps=sanity_check
    )

    #trainer.fit(model, data_module) # Done only once to train and save the model

    torch.serialization.add_safe_globals([pl.callbacks.model_checkpoint.ModelCheckpoint]) # What is this for?

    trained_model = WeatherPredictor.load_from_checkpoint(
        checkpoint_path=f"checkpoints/trial_{trial_number}.ckpt",
        n_features=train_df.shape[1],
        n_hidden=model.model.n_hidden,
        n_layers=model.model.lstm.num_layers,
        n_dropout=model.model.lstm.dropout,
        n_learning_rate=model.learning_rate,
        map_location="cuda"
    ).cuda()

    return trained_model

def descale_data(scaler, train_df, predictions_temp, labels_temp, predictions_humidity, labels_humidity):

    # Descale the predictions and labels
    def inverse_single_feature(scaler, scaled_values, feature_index):
        dummy = np.zeros((len(scaled_values), scaler.scale_.shape[0]))
        dummy[:, feature_index] = scaled_values
        inversed = scaler.inverse_transform(dummy)
        return inversed[:, feature_index]

    temp_index = train_df.columns.get_loc("Temperature (C)")
    humidity_index = train_df.columns.get_loc("Humidity")

    predictions_temp_descaled = inverse_single_feature(scaler, predictions_temp, temp_index)
    labels_temp_descaled = inverse_single_feature(scaler, labels_temp, temp_index)

    predictions_humidity_descaled = inverse_single_feature(scaler, predictions_humidity, humidity_index)
    labels_humidity_descaled = inverse_single_feature(scaler, labels_humidity, humidity_index)

    return predictions_temp_descaled, labels_temp_descaled, predictions_humidity_descaled, labels_humidity_descaled

def calculate_error(predictions_temp_descaled, labels_temp_descaled, predictions_humidity_descaled, labels_humidity_descaled):

    # Calculate RMSE for Temperature and Humidity
    def rmse(predictions, targets):
        return np.sqrt(np.mean((predictions - targets) ** 2))

    rmse_temp = rmse(predictions_temp_descaled, labels_temp_descaled)
    rmse_humidity = rmse(predictions_humidity_descaled, labels_humidity_descaled)

    #print(f"Test RMSE - Temperature: {rmse_temp:.2f} C")
    #print(f"Test RMSE - Humidity: {rmse_humidity:.2f} %")

    # Calculate MAPE for Temperature and Humidity
    wmape = WeightedMeanAbsolutePercentageError()

    wmape_temp = wmape(torch.tensor(predictions_temp_descaled), torch.tensor(labels_temp_descaled))
    wmape_humidity = wmape(torch.tensor(predictions_humidity_descaled), torch.tensor(labels_humidity_descaled))
    #print(f"Test WMAPE - Temperature: {wmape_temp:.2f} %")
    #print(f"Test WMAPE - Humidity: {wmape_humidity:.2f} %")

    return rmse_temp, rmse_humidity, wmape_temp, wmape_humidity

# Optimize hyperparameters using Optuna and MLflow
def objective(trial):
    with mlflow.start_run(nested=True, run_name=f"trial_{trial.number}") as child_run:

        data_seq_length = trial.suggest_categorical("seq_length", [12, 24])

        lstm_n_hidden = trial.suggest_categorical("n_hidden", [32, 64, 128, 256])
        lstm_n_layers = trial.suggest_categorical("n_layers", [1, 2, 3])
        lstm_dropout = trial.suggest_float("dropout", 0.0, 0.5)

        lstm_batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128])
        lstm_learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)

        params = {
            "seq_length": data_seq_length,
            "n_hidden": lstm_n_hidden,
            "n_layers": lstm_n_layers,
            "dropout": lstm_dropout,
            "batch_size": lstm_batch_size,
            "learning_rate": lstm_learning_rate,
        }

        # Log current trial's parameters
        mlflow.log_params(params)

        # 1. Split the training, validation, and test sets
        train_df, val_df, test_df = split_data()

        # 2. Scale the data
        scaler, train_df, val_df, test_df = scale_data(train_df, val_df, test_df)

        # 3. Create data sequences
        train_sequences = create_sequences(train_df, data_seq_length)
        val_sequences = create_sequences(val_df, data_seq_length)
        test_sequences = create_sequences(test_df, data_seq_length)

        # 4. Create DataModule
        data_module = WeatherDataModule(lstm_batch_size, train_sequences, test_sequences, val_sequences)
        data_module.setup()

        train_dataset = WeatherDataset(train_sequences)

        # 5. Define model
        model = WeatherPredictor(n_features=train_df.shape[1], n_hidden=lstm_n_hidden, n_layers=lstm_n_layers, 
                                 n_dropout=lstm_dropout, n_learning_rate=lstm_learning_rate)

        # 6. Fit the model
        trained_model = fit_model(model, data_module, train_df, trial.number)

        # 7. Evaluate on validation set
        val_dataset = WeatherDataset(val_sequences)

        predictions_temp = []
        labels_temp = []
        predictions_humidity = []
        labels_humidity = []

        for item in val_dataset:
            sequence = item["sequence"].unsqueeze(0).cuda()  # Add batch dimension and move to GPU
            label_temp = item["label_temp"]
            label_humidity = item["label_humidity"]

            _, output = trained_model(sequence)
            
            # Extract temperature and humidity predictions
            pred_temp = output[0, 0].item()  # First prediction (temperature)
            pred_humidity = output[0, 1].item()  # Second prediction (humidity)
            
            predictions_temp.append(pred_temp)
            predictions_humidity.append(pred_humidity)
            labels_temp.append(label_temp.item())
            labels_humidity.append(label_humidity.item())

        # 8. Descale the predictions and labels
        predictions_temp_descaled, labels_temp_descaled, predictions_humidity_descaled, labels_humidity_descaled = descale_data(
            scaler, train_df, predictions_temp, labels_temp, predictions_humidity, labels_humidity
        )

        # 9. Calculate error metrics
        _, _, wmape_temp, wmape_humidity = calculate_error(
            predictions_temp_descaled, labels_temp_descaled, predictions_humidity_descaled, labels_humidity_descaled
        )

        error = (wmape_temp + wmape_humidity)/2.0  # Combine errors for optimization

        # Log current trial's error metric
        mlflow.log_metrics({"error": error})

        # Log the model file
        """mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=readout
        )"""
        
        # Make it easy to retrieve the best-performing child run later
        trial.set_user_attr("run_id", child_run.info.run_id)
        return error
    
# Create a parent run that contains all child runs for different trials
mlflow.set_experiment("reservoir_optuna_experiment")
with mlflow.start_run(run_name="study") as run:
    # Log the experiment settings
    n_trials = 3
    mlflow.log_param("n_trials", n_trials)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)

    # Log the best trial and its run ID
    mlflow.log_params(study.best_trial.params)
    mlflow.log_metrics({"best_error": study.best_value})
    if best_run_id := study.best_trial.user_attrs.get("run_id"):
        mlflow.log_param("best_child_run_id", best_run_id)

def train_final_and_test(best_params):

    # 1. Split data
    train_df, val_df, test_df = split_data()
    
    # 2. Combine TRAIN + VAL because hyperparams are now fixed
    full_train_df = pd.concat([train_df, val_df]).reset_index(drop=True)
    
    # 3. Scale again using full_train_df
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(full_train_df)

    # save the scaler
    joblib.dump(scaler, 'scaler.pkl')

    full_train_df = pd.DataFrame(scaler.transform(full_train_df),
                                 columns=full_train_df.columns)
    test_df = pd.DataFrame(scaler.transform(test_df),
                           columns=test_df.columns)

    # 4. Create data sequences
    X_train = create_sequences(full_train_df, best_params["seq_length"])
    X_test = create_sequences(test_df, best_params["seq_length"])

    # 5. Create DataModule
    data_module = WeatherDataModule(best_params["batch_size"], X_train, X_test, val_sequences=None)
    data_module.setup()

    train_dataset = WeatherDataset(X_train)

    # 5. Define model
    model = WeatherPredictor(n_features=train_df.shape[1], n_hidden=best_params["n_hidden"],
                             n_layers=best_params["n_layers"],
                             n_dropout=best_params["dropout"],
                             n_learning_rate=best_params["learning_rate"])

    # 6. Fit the model
    #trained_model = fit_model(model, data_module, full_train_df, 1000, sanity_check = 0)

    # 7. Evaluate on test set
    test_dataset = WeatherDataset(X_test)

    predictions_temp = []
    labels_temp = []
    predictions_humidity = []
    labels_humidity = []

    for item in test_dataset:
        sequence = item["sequence"].unsqueeze(0).cuda()  # Add batch dimension and move to GPU
        label_temp = item["label_temp"]
        label_humidity = item["label_humidity"]

        _, output = trained_model(sequence)
        
        # Extract temperature and humidity predictions
        pred_temp = output[0, 0].item()  # First prediction (temperature)
        pred_humidity = output[0, 1].item()  # Second prediction (humidity)
        
        predictions_temp.append(pred_temp)
        predictions_humidity.append(pred_humidity)
        labels_temp.append(label_temp.item())
        labels_humidity.append(label_humidity.item())

    # 8. Descale the predictions and labels
    predictions_temp_descaled, labels_temp_descaled, predictions_humidity_descaled, labels_humidity_descaled = descale_data(
        scaler, train_df, predictions_temp, labels_temp, predictions_humidity, labels_humidity
    )

    # 9. Calculate error metrics
    rmse_temp, rmse_hum, wmape_temp, wmape_hum = calculate_error(
        predictions_temp_descaled, labels_temp_descaled, predictions_humidity_descaled, labels_humidity_descaled
    )

    return rmse_temp, rmse_hum, wmape_temp, wmape_hum, trained_model

# Retrieve best hyperparameters
best_params = study.best_trial.params

# Retrain best model on full train + val, test on test set
rmse_temp, rmse_hum, wmape_temp, wmape_hum, final_model = train_final_and_test(best_params)

# Save the final model
#torch.save(final_model.model.state_dict(), "weather_lstm_state_dict.pt")

# Save the model with ONNX as a cross-platform format
# ===== ONNX EXPORT (FOR PYNQ) =====
final_model.model.eval()
final_model.model.cpu()

SEQ_LENGTH = best_params["seq_length"]
N_FEATURES = features_df.shape[1]

dummy_input = torch.randn(
    1,                # batch size
    SEQ_LENGTH,       # sequence length
    N_FEATURES,       # number of features
    dtype=torch.float32
)

"""torch.onnx.export(
    final_model.model,
    dummy_input,
    "weather_lstm.onnx",
    input_names=["input"],
    output_names=["output"],
    opset_version=11,
    do_constant_folding=True
)

print("ONNX model exported: weather_lstm.onnx") """

# Log final model and metrics to MLflow
with mlflow.start_run(run_name="final_model"):
    mlflow.log_params(best_params)
    mlflow.log_metrics({
        "test_rmse_temp": rmse_temp,
        "test_rmse_humidity": rmse_hum,
        "test_wmape_temp": wmape_temp,
        "test_wmape_humidity": wmape_hum,
    })

print("Best hyperparameters:", best_params)

print(f"Test RMSE - Temperature: {rmse_temp:.2f} C")
print(f"Test RMSE - Humidity: {rmse_hum:.2f} %")

print(f"Test WMAPE - Temperature: {wmape_temp:.2f}")
print(f"Test WMAPE - Humidity: {wmape_hum:.2f}")

# Create lag features
# I choose lags of 1, 2, 3, 6, 12, 24 hours
""" lags = [1, 2, 3, 6, 12, 24]
for lag in lags:
    df[f'lag_{lag}'] = df['Temperature (C)'].shift(lag)
#print(df.head(25)) """
#df = df.fillna(0)  # Fill NaN values created by lagging with 0

# Add a change column for each lag column
""" for lag in lags:
    df[f'change_lag_{lag}'] = df['Temperature (C)'] - df[f'lag_{lag}']
    df[f'change_lag_{lag}'] = df['Humidity'] - df[f'lag_{lag}'] """
    
#N_EPOCHS = 8
#BATCH_SIZE = 64

"""for item in train_dataset:
    print(item["sequence"].shape)
    print(item["label_temp"].shape)
    print(item["label_humidity"].shape)
    print(item["label_temp"])
    break"""

"""i = 0
for item in data_module.train_dataloader():
print(item["sequence"].shape)
print(item["label_temp"].shape)
print(item["label_humidity"].shape)
break"""

#print("len(prediction_temp): ", len(predictions_temp))
#print("len(test_df): ", len(test_df) - SEQ_LENGTH)

# Plot Predictions vs Actuals for Temperature

#breakpoint()

"""
test_data = df[train_size+val_size:]
test_sequences_data = test_data.iloc[SEQ_LENGTH:]
# print(len(test_sequences_data), len(test_sequences))

dates = matplotlib.dates.date2num(test_sequences_data.date.tolist())
plt.plot_date(dates, predictions_temp_descaled, '-', label='Predicted Temperature')
plt.plot_date(dates, labels_temp_descaled, '-', label='Actual Temperature')
plt.xlabel('Date')
plt.ylabel('Temperature (C)')
plt.title('Temperature Prediction vs Actual')
plt.legend()
plt.show()

dates = matplotlib.dates.date2num(test_sequences_data.date.tolist())
plt.plot_date(dates, predictions_humidity_descaled, '-', label='Predicted Humidity')
plt.plot_date(dates, labels_humidity_descaled, '-', label='Actual Humidity')
plt.xlabel('Date')
plt.ylabel('Humidity (%)')
plt.title('Humidity Prediction vs Actual')
plt.legend()
plt.show()

# Hmm... looks a bit too good to be true. Let's plot all the values
# Plot all the values to see how orderly is the pattern
"""

"""
temp = []
humidity = []
for item in train_dataset:
    label_temp = item["label_temp"]
    label_humidity = item["label_humidity"]
    temp.append(label_temp.item())
    humidity.append(label_humidity.item())

train_data = df[:train_size]
train_sequences_data = train_data.iloc[SEQ_LENGTH:]

dates = matplotlib.dates.date2num(train_sequences_data.date.tolist())
plt.plot_date(dates, temp, '-', label='Actual Temperature')
plt.xlabel('Date')
plt.ylabel('Temperature (C)')
plt.title('All Temperature Values')
plt.legend()
# plt.show()

dates = matplotlib.dates.date2num(train_sequences_data.date.tolist())
plt.plot_date(dates, humidity, '-', label='Actual Humidity')
plt.xlabel('Date')
plt.ylabel('Humidity (%)')
plt.title('All Humidity Values')
plt.legend()
# plt.show()


# Let's look at the plot of the prediction for just a few days

dates = matplotlib.dates.date2num(test_sequences_data.date.tolist()[SEQ_LENGTH + 120 * 24:SEQ_LENGTH + 123 * 24])
plt.plot_date(dates, predictions_temp_descaled[120 * 24:123 * 24], '-', label='Predicted Temperature')
plt.plot_date(dates, labels_temp_descaled[120 * 24:123 * 24], '-', label='Actual Temperature')
plt.xlabel('Date')
plt.ylabel('Temperature (C)')
plt.title('Temperature Prediction vs Actual')
plt.legend()
plt.show()
"""