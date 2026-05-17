# LSTM vs Reservoir Computing

This repository compares two machine learning approaches for environmental time-series forecasting:

- **LSTM (Long Short-Term Memory)**
- **Reservoir Computing (RC)**

The objective of this project is to evaluate both models in terms of:

- Prediction accuracy
- Training and inference time
- Computational efficiency

Both models are trained on the same dataset and optimized under equivalent conditions to ensure a fair comparison.

---

# Project Overview

Reservoir Computing is theoretically known for being highly efficient in terms of computational cost and training speed while still achieving strong predictive performance in time-series tasks. However, compared to LSTM networks, RC has not yet been as extensively validated in industrial environments.

This project was developed to empirically evaluate whether Reservoir Computing can outperform LSTM in a real-world environmental forecasting scenario.

Both implementations predict:

- Temperature for the next hour
- Humidity for the next hour

using historical weather data.

---

# Dataset

The dataset used for both models is:

https://www.kaggle.com/datasets/muthuj7/weather-dataset

It contains approximately 10 years of hourly weather measurements collected in Hungary.

---

# Experimental Setup

The dataset was divided into:

- **70% Training**
- **15% Validation**
- **15% Testing**

The workflow for both models was identical:

1. Train the model using the training split
2. Tune hyperparameters using the validation split
3. Evaluate final performance on the test split

The following metrics were compared:

- RMSE (Root Mean Square Error)
- Total training + inference time

---

# Technologies Used

## LSTM Implementation

Main libraries:

- PyTorch
- PyTorch Lightning

## Reservoir Computing Implementation

Main library:

- ReservoirPy

## Hyperparameter Optimization

Both models were optimized using:

- MLflow
- Optuna

This ensured that both approaches operated under optimized conditions, making the comparison as fair as possible.

---

# Results

## LSTM

- **Temperature Test RMSE:** 1.76 °C
- **Humidity Test RMSE:** 3.70 %
- **Training + Inference Time:** ~30 minutes

<img width="1139" height="55" alt="image" src="https://github.com/user-attachments/assets/46f35ac6-374c-4854-8365-b6c01903c24e" />

---

## Reservoir Computing

- **Temperature Test RMSE:** 0.84 °C
- **Humidity Test RMSE:** 0.05 %
- **Training + Inference Time:** 1 minute 13 seconds

<img width="829" height="144" alt="image" src="https://github.com/user-attachments/assets/0541eed3-2f0d-431c-a9bf-99bd8a6b874f" />

---

# Prediction Graphs

The following graphs illustrate the prediction accuracy achieved by the Reservoir Computing model:

![Graph 1](images/graph1.png)

![Graph 2](images/graph2.png)

---

# Analysis

One of the most common metrics for evaluating machine learning models is **RMSE (Root Mean Square Error)**.

RMSE preserves the original unit of the predicted variable and heavily penalizes large prediction errors, making it especially useful for forecasting tasks.

The results clearly show that Reservoir Computing:

- Achieves lower prediction error than LSTM
- Requires dramatically less training and inference time
- Provides significantly better computational efficiency

While the LSTM implementation required nearly 30 minutes to complete training and inference, Reservoir Computing completed the same process in just over one minute.

---

# Conclusion

The experimental results confirm the theoretical expectations surrounding Reservoir Computing for environmental time-series prediction.

Under the same dataset and optimization conditions, Reservoir Computing outperformed LSTM in both:

- Accuracy
- Execution speed

For this reason, Reservoir Computing was selected as the final model for the broader project this repository belongs to.
