""" Reservoir Time-Series Forecasting with ReservoirPy
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
#import pytorch_lightning as pl
#from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
#from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta
#from torchmetrics.regression import WeightedMeanAbsolutePercentageError
#from torch.utils.tensorboard import SummaryWriter

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
from torchmetrics.regression import WeightedMeanAbsolutePercentageError

from joblib import dump

import torch.cuda

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pytorch_fabric")

import mlflow
#from mlflow.pytorch import MlflowModelCheckpointCallback

import optuna
import sklearn

import yaml

import reservoirpy as rpy
from reservoirpy.nodes import Reservoir, Ridge
from reservoirpy import ESN

tqdm.pandas()

#%matplotlib inline
#%config InlineBackend.figure_format='retina'

sns.set_theme(style='whitegrid', palette='muted', font_scale=1.2)

HAPPY_COLORS_PALETTE = ["#01BEFE", "#FFDD00", "#FF7D00", "#FF006D", "#93D30C", "#8F00FF"]

sns.set_palette(sns.color_palette(HAPPY_COLORS_PALETTE))

rcParams['figure.figsize'] = 14, 10
#register_matplotlib_converters()

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# Load the dataset
df = pd.read_csv('weatherHistory.csv')

print(df.head())

# Rename the column "Loud Cover" to "Cloud Cover" and "Formatted Date" to "date"
df = df.rename(columns={"Formatted Date": "date"})
df = df.rename(columns={"Loud Cover": "Cloud Cover"})

# Convert datetime column to datetime object
df['date'] = pd.to_datetime(df['date'], utc=True)

# Sort by date
df = df.sort_values('date').reset_index(drop=True)

# Drop columns that won't be used
features_df = df.drop(['Summary', 'Daily Summary', 'Precip Type', 'Daily Summary', 'Visibility (km)', 'Apparent Temperature (C)'], axis=1)

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

# Create a Shifted feature for Temperature and Humidity
features_df['Temperature_Shift'] = features_df['Temperature (C)'].shift(-1)
features_df['Humidity_Shift'] = features_df['Humidity'].shift(-1)
features_df = features_df.fillna(0)

# Separate the training, validation, and test sets
def split_data():

    train_size = int(len(features_df) * 0.7)
    val_size = int(len(features_df) * 0.15)

    train_df, val_df, test_df = features_df[:train_size], features_df[train_size:train_size+val_size], features_df[train_size+val_size:]
    return train_df, val_df, test_df

# Scale the data using MinMaxScaler
def scale_data(train_df, val_df, test_df):

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler = scaler.fit(train_df)

    train_df = pd.DataFrame(scaler.transform(train_df), index = train_df.index, columns=train_df.columns)
    val_df = pd.DataFrame(scaler.transform(val_df), index = val_df.index, columns=val_df.columns)
    test_df = pd.DataFrame(scaler.transform(test_df), index = test_df.index, columns=test_df.columns)

    return scaler, train_df, val_df, test_df

def create_reservoir(units, rc_lr, rc_sr, rc_ridge):

    # Create the Reservoir and the Ridge nodes
    reservoir = Reservoir(units, lr = rc_lr, sr = rc_sr) # Previous values: 500, lr = 0.5, sr = 0.9
    ridge = Ridge(ridge=rc_ridge)

    esn_model = ESN(reservoir, ridge)
    return esn_model

#  scaled_df: DataFrame containing scaled data including real target shifts
#  y: numpy array of scaled predictions of the target shifts
def descale_data(scaled_df, y, scaler):

    # Get the indices of Temperature and Humidity columns in the scaler
    temp_col_idx = scaled_df.columns.get_loc('Temperature (C)')
    temp_shift_col_idx = scaled_df.columns.get_loc('Temperature_Shift')

    hum_col_idx = scaled_df.columns.get_loc('Humidity')
    hum_shift_col_idx = scaled_df.columns.get_loc('Humidity_Shift')

    # Create dummy arrays with the same shape as the scaler expects
    # (scaler was fitted on all columns, so we need to provide all columns)
    def descale(scaled_values, scaler, col_idx):
        """
        Descale temperature values back to original scale
        scaled_values: array of scaled temperature values
        scaler: the fitted MinMaxScaler object
        col_idx: index of the temperature column in the original dataframe
        """
        # Create a dummy array with zeros for all features
        dummy = np.zeros((len(scaled_values), len(scaler.scale_)))
        # Place the scaled values in the correct column
        dummy[:, col_idx] = scaled_values.flatten()
        # Inverse transform
        descaled = scaler.inverse_transform(dummy)
        # Return only the temperature column
        return descaled[:, col_idx].reshape(-1, 1)

    # Descale predictions and actual values
    real_future_temp = np.array(scaled_df["Temperature_Shift"]).reshape(-1,1)
    real_future_temp_descaled = descale(real_future_temp, scaler, temp_shift_col_idx)

    real_future_hum = np.array(scaled_df["Humidity_Shift"]).reshape(-1,1)
    real_future_hum_descaled = descale(real_future_hum, scaler, hum_shift_col_idx)

    y_test_descaled_temp = descale(y[:, 0], scaler, temp_shift_col_idx)
    y_test_descaled_hum  = descale(y[:, 1], scaler, hum_shift_col_idx)

    return real_future_temp_descaled, real_future_hum_descaled, y_test_descaled_temp, y_test_descaled_hum

def calculate_error(y_test_descaled_temp, real_future_temp_descaled, y_test_descaled_hum, real_future_hum_descaled):
    # Calculate RMSE for Temperature and Humidity
    def rmse(predictions, targets):
        return np.sqrt(np.mean((predictions - targets) ** 2))

    rmse_temp = rmse(y_test_descaled_temp, real_future_temp_descaled)
    rmse_hum = rmse(y_test_descaled_hum, real_future_hum_descaled)

    #print(f"Test RMSE - Temperature: {rmse_temp:.2f} C")
    #print(f"Test RMSE - Humidity: {rmse_hum:.2f} %")

    # Calculate MAPE for Temperature
    wmape = WeightedMeanAbsolutePercentageError()

    wmape_temp = wmape(torch.tensor(y_test_descaled_temp), torch.tensor(real_future_temp_descaled))
    wmape_hum = wmape(torch.tensor(y_test_descaled_hum), torch.tensor(real_future_hum_descaled))

    #print(f"Test WMAPE - Temperature: {wmape_temp:.2f} %")
    #print(f"Test WMAPE - Humidity: {wmape_hum:.2f} %")

    return rmse_temp, rmse_hum, wmape_temp, wmape_hum

# Hyperparameter Optimization
def objective(trial):
    # Setting nested=True will create a child run under the parent run.
    with mlflow.start_run(nested=True, run_name=f"trial_{trial.number}") as child_run:

        # Reservoir Parameters
        rc_units = trial.suggest_int("units", 50, 1000)
        rc_lr    = trial.suggest_float("lr", 0.1, 1.0)
        rc_sr    = trial.suggest_float("sr", 0.1, 1.0)
        rc_ridge = trial.suggest_float("ridge", 0.0, 5.0)

        params = {
            "units": rc_units,
            "lr": rc_lr,
            "sr": rc_sr,
            "ridge": rc_ridge
        }

        # Log current trial's parameters
        mlflow.log_params(params)

        # Split the training, validation, and test sets
        train_df, val_df, test_df = split_data()

        # Scale the data using MinMaxScaler
        scaler, train_df, val_df, test_df = scale_data(train_df, val_df, test_df)

        # Create training, validation, and test sets with nd arrays
        t_df = train_df.drop(['Temperature_Shift', 'Humidity_Shift'], axis=1)
        X_train = np.array(t_df)

        y_train = np.array(train_df[["Temperature_Shift", "Humidity_Shift"]])

        t_df = val_df.drop(['Temperature_Shift', 'Humidity_Shift'], axis=1)
        X_val = np.array(t_df)

        # Create the Reservoire Model
        esn_model = create_reservoir(rc_units, rc_lr, rc_sr, rc_ridge)

        # Train the model
        readout = esn_model.fit(X_train, y_train, warmup=10)

        #print(reservoir.initialized, readout.initialized)
        
        # Run the model on the test data
        y_val = esn_model.run(X_val)

        # Descale data
        real_future_temp_descaled, real_future_hum_descaled, y_test_descaled_temp, y_test_descaled_hum = descale_data(val_df, y_val, scaler)

        # Calculate error
        (_, _, wmape_temp, wmape_hum) = calculate_error(y_test_descaled_temp, real_future_temp_descaled, y_test_descaled_hum, real_future_hum_descaled)
        error = (wmape_temp + wmape_hum) / 2.0

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

"""
# Register the best model using the model URI
mlflow.register_model(
    model_uri="runs:/d0210c58afff4737a306a2fbc5f1ff8d/model",
    name="ReservoirPy_Weather_Forecasting_Model"
) """

def plot_results(y_test_descaled_temp, real_future_temp_descaled,
                 y_test_descaled_hum, real_future_hum_descaled):

    # Find the index where the test set starts in the original dataframe
    train_size = int(len(features_df) * 0.7)
    val_size = int(len(features_df) * 0.15)
    test_data = df[train_size+val_size:]

    # Plot the results for temperature
    dates = matplotlib.dates.date2num(test_data.date.tolist())
    plt.plot(dates, y_test_descaled_temp, '-', label='Predicted Temperature')
    plt.plot(dates, real_future_temp_descaled, '-', label='Actual Temperature')
    plt.title("Temperature and its future (Original Scale)")
    plt.xlabel("$t$")
    plt.ylabel("Temperature (C)")
    plt.legend()
    plt.show()

    # Plot the results for humidity
    dates = matplotlib.dates.date2num(test_data.date.tolist())
    plt.plot(dates, y_test_descaled_hum, '-', label='Predicted Humidity')
    plt.plot(dates, real_future_hum_descaled, '-', label='Actual Humidity')
    plt.title("Humidity and its future (Original Scale)")
    plt.xlabel("$t$")
    plt.ylabel("Humidity")
    plt.legend()
    plt.show()

def train_final_and_test(best_params):

    # 1. Split data
    train_df, val_df, test_df = split_data()
    
    # 2. Combine TRAIN + VAL because hyperparams are now fixed
    full_train_df = pd.concat([train_df, val_df]).reset_index(drop=True)
    
    # 3. Scale again using full_train_df
    scaler = MinMaxScaler()
    scaler.fit(full_train_df)

    full_train_df = pd.DataFrame(scaler.transform(full_train_df),
                                 columns=full_train_df.columns)
    test_df = pd.DataFrame(scaler.transform(test_df),
                           columns=test_df.columns)

    # 4. Prepare train arrays
    X_train = full_train_df.drop(['Temperature_Shift', 'Humidity_Shift'], axis=1).values
    y_train = full_train_df[['Temperature_Shift','Humidity_Shift']].values

    X_test = test_df.drop(['Temperature_Shift','Humidity_Shift'], axis=1).values

    # 5. Build ESN with best hyperparameters
    esn = create_reservoir(
        best_params["units"],
        best_params["lr"],
        best_params["sr"],
        best_params["ridge"]
    )

    # 6. Train on full training data
    esn.fit(X_train, y_train, warmup=10)

    # 7. Predict on test set
    y_test_pred = esn.run(X_test)

    # 8. Descale predictions and actual test targets
    (real_temp, real_hum,
     pred_temp, pred_hum) = descale_data(test_df, y_test_pred, scaler)

    # 9. Compute test error
    rmse_temp, rmse_hum, wmape_temp, wmape_hum = calculate_error(pred_temp, real_temp, pred_hum, real_hum)

    # 10. Plot results
    plot_results(pred_temp, real_temp, pred_hum, real_hum)

    return rmse_temp, rmse_hum, wmape_temp, wmape_hum, esn

# Retrieve best hyperparameters
best_params = study.best_trial.params

# Retrain best model on full train + val, test on test set
rmse_temp, rmse_hum, wmape_temp, wmape_hum, final_model = train_final_and_test(best_params)

# Save the final model
#torch.save(final_model.state_dict(), 'my_reservoir_model.pth')

print("Best hyperparameters:", best_params)

print(f"Test RMSE - Temperature: {rmse_temp:.2f} C")
print(f"Test RMSE - Humidity: {rmse_hum:.2f} %")

print(f"Test WMAPE - Temperature: {wmape_temp:.2f}")
print(f"Test WMAPE - Humidity: {wmape_hum:.2f}")