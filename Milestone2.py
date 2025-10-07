# forecasting.py - Milestone 2 (Prophet + LSTM + Model Comparison)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pickle, os, warnings
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler


warnings.filterwarnings("ignore")

# -----------------------------
# 1. Load Cleaned Data
# -----------------------------
df = pd.read_csv("cleaned_retail_data.csv")  # update path if needed

# Ensure Date is datetime
df["Date"] = pd.to_datetime(df["Date"])

os.makedirs("models", exist_ok=True)
os.makedirs("data", exist_ok=True)

# -----------------------------
# 2. Helper — LSTM Training
# -----------------------------
def train_lstm(series, n_lags=7, epochs=10):
    """Train simple LSTM on 1D series"""
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(series.values.reshape(-1, 1))

    X, y = [], []
    for i in range(len(scaled) - n_lags):
        X.append(scaled[i:i + n_lags, 0])
        y.append(scaled[i + n_lags, 0])

    X, y = np.array(X), np.array(y)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    model = Sequential([
        LSTM(50, activation='relu', input_shape=(n_lags, 1)),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=epochs, verbose=0)

    return model, scaler

def forecast_lstm(model, scaler, series, steps=30, n_lags=7):
    data = scaler.transform(series.values.reshape(-1, 1)).flatten().tolist()
    preds = []

    for _ in range(steps):
        x_input = np.array(data[-n_lags:]).reshape((1, n_lags, 1))
        yhat = model.predict(x_input, verbose=0)
        data.append(yhat[0][0])
        preds.append(yhat[0][0])

    preds = scaler.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()
    return preds

# -----------------------------
# 3. Forecast for All Products
# -----------------------------
forecast_list = []
all_products = df["Product ID"].unique()

for product_name in all_products:
    print(f"🔄 Training Prophet & LSTM for {product_name}...")

    product_df = df[df['Product ID'] == product_name][["Date", 'Units Sold']]

    # ---------- Prophet ----------
    prophet_df = product_df.rename(columns={"Date": 'ds', 'Units Sold': 'y'})
    model_prophet = Prophet(yearly_seasonality=True, weekly_seasonality=True)
    model_prophet.fit(prophet_df)
    future = model_prophet.make_future_dataframe(periods=30)
    forecast_p = model_prophet.predict(future)
    yhat_prophet = forecast_p['yhat'][-30:]

    # ---------- LSTM ----------
    sales_series = product_df.set_index("Date")['Units Sold']
    train_size = int(len(sales_series) * 0.8)
    train_series = sales_series.iloc[:train_size]

    lstm_model, scaler = train_lstm(train_series)
    yhat_lstm = forecast_lstm(lstm_model, scaler, sales_series, steps=30)

    # ---------- Evaluation ----------
    actual = sales_series[-30:] if len(sales_series) >= 30 else sales_series

    mae_prophet = mean_absolute_error(actual, yhat_prophet[:len(actual)])
    rmse_prophet = np.sqrt(mean_squared_error(actual, yhat_prophet[:len(actual)]))

    mae_lstm = mean_absolute_error(actual, yhat_lstm[:len(actual)])
    rmse_lstm = np.sqrt(mean_squared_error(actual, yhat_lstm[:len(actual)]))

    print(f"Prophet MAE:{mae_prophet:.2f} RMSE:{rmse_prophet:.2f}")
    print(f"LSTM MAE:{mae_lstm:.2f} RMSE:{rmse_lstm:.2f}")

    # ---------- Choose Best ----------
    if rmse_lstm < rmse_prophet:
        print("✅ LSTM better, saving LSTM predictions.")
        best_forecast = yhat_lstm
        with open(f"models/lstm_model_{product_name}.pkl", "wb") as f:
            pickle.dump((lstm_model.to_json(), scaler.get_params()), f)
    else:
        print("✅ Prophet better, saving Prophet predictions.")
        best_forecast = yhat_prophet
        with open(f"models/prophet_model_{product_name}.pkl", "wb") as f:
            pickle.dump(model_prophet, f)

    # ---------- Save Forecast ----------
    forecast_dates = pd.date_range(
        start=product_df["Date"].max() + pd.Timedelta(days=1),
        periods=30
    )

    temp = pd.DataFrame({
        'date': forecast_dates,
        'forecast_best': best_forecast,
        'Product ID': product_name
    })

    forecast_list.append(temp)

# -----------------------------
# 4. Combine and Save
# -----------------------------
forecast_all = pd.concat(forecast_list)
forecast_all.to_csv("data/forecast_results.csv", index=False)

print("\n✅ Forecast results saved with Prophet & LSTM comparison in data/forecast_results.csv")
