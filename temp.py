import numpy as np
import pandas as pd
import math
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns

def scale_range(x, input_range, target_range):
    x_min, x_max = input_range
    x_scaled = (x - x_min) / (x_max - x_min)
    x_scaled = x_scaled * (target_range[1] - target_range[0]) + target_range[0]
    return x_scaled

def train_test_split_linear_regression(stocks):
    feature = stocks[['Open']].values
    label = stocks[['Close']].values

    scaler_feature = MinMaxScaler(feature_range=(-1, 1))
    scaler_label = MinMaxScaler(feature_range=(-1, 1))

    feature_scaled = scaler_feature.fit_transform(feature)
    label_scaled = scaler_label.fit_transform(label)

    split = int(math.floor(len(stocks) * 0.315))
    
    X_train_lr = feature_scaled[:-split]
    X_test_lr = feature_scaled[-split:]
    y_train_lr = label_scaled[:-split]
    y_test_lr = label_scaled[-split:]

    return X_train_lr, X_test_lr, y_train_lr, y_test_lr, scaler_label

# Load stock data
df = pd.read_csv('data.csv', parse_dates=['Date'], index_col='Date')

# Linear Regression Model
X_train_lr, X_test_lr, y_train_lr, y_test_lr, scaler_label = train_test_split_linear_regression(df)
linear_regression_model = LinearRegression()
linear_regression_model.fit(X_train_lr, y_train_lr)

# Predict and evaluate the model
y_pred_lr = linear_regression_model.predict(X_test_lr)

# Inverse transform the predictions and true values to original scale
y_test_lr_original = scaler_label.inverse_transform(y_test_lr)
y_pred_lr_original = scaler_label.inverse_transform(y_pred_lr)

# Calculate metrics
mse_lr = mean_squared_error(y_test_lr_original, y_pred_lr_original)
mae_lr = mean_absolute_error(y_test_lr_original, y_pred_lr_original)
r2_lr = r2_score(y_test_lr_original, y_pred_lr_original)

# Print results
print(f'Linear Regression Model Evaluation:')
print(f'Mean Squared Error (MSE): {mse_lr:.4f}')
print(f'Mean Absolute Error (MAE): {mae_lr:.4f}')
print(f'R-squared (R2) Score: {r2_lr:.4f}')

# Plot predictions vs actual values
plt.figure(figsize=(14, 7))

# Plot the predictions and actual values
plt.subplot(2, 1, 1)
plt.plot(df.index[-len(y_test_lr_original):], y_test_lr_original, color='blue', label='Actual Prices')
plt.plot(df.index[-len(y_test_lr_original):], y_pred_lr_original, color='red', linestyle='--', label='Predicted Prices')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.title('Stock Price Prediction vs Actual Prices')
plt.legend()
plt.grid(False)

# Plot metrics
metrics = {'MSE': mse_lr, 'MAE': mae_lr, 'R2 Score': r2_lr}
plt.subplot(2, 1, 2)
sns.barplot(x=list(metrics.keys()), y=list(metrics.values()), palette='viridis')
plt.xlabel('Metric')
plt.ylabel('Value')
plt.title('Model Performance Metrics')
plt.grid(False)

plt.tight_layout()
plt.show()
