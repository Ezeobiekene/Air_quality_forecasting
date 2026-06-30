import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    r2_score,
)


def create_features(df, target_col="pm2_5", horizon=3):
    df = df.copy()

    # 1. Create Target (Predicting 3 hours into the future)
    df["target"] = df[target_col].shift(-horizon)

    # 2. Extract Time-Based Features (Cyclic Encoding)
    df["hour"] = df.index.hour
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["day_of_week"] = df.index.dayofweek

    # 3. Use CURRENT values as features (Perfectly valid since we know them at prediction time)
    df["pm25_current"] = df[target_col]
    df["pm10_current"] = df["pm10"]
    df["ozone_current"] = df["ozone"]
    df["no2_current"] = df["no2"]

    # 4. Past History Lags relative to the current moment
    df["pm25_lag_1h"] = df[target_col].shift(1)
    df["pm25_lag_2h"] = df[target_col].shift(2)
    df["pm25_lag_24h"] = df[target_col].shift(24)

    # 5. Rolling window of the last 6 hours leading up to right now
    df["pm25_rolling_mean_6h"] = df[target_col].rolling(window=6).mean()

    df.dropna(inplace=True)
    return df


def train_pipeline():
    df = pd.read_csv("raw_air_quality.csv", index_col=0, parse_dates=True)
    featured_df = create_features(df)

    split_idx = int(len(featured_df) * 0.8)
    train_df = featured_df.iloc[:split_idx]
    test_df = featured_df.iloc[split_idx:]

    feature_cols = [
        "hour_sin",
        "hour_cos",
        "day_of_week",
        "pm25_current",
        "pm10_current",
        "ozone_current",
        "no2_current",
        "pm25_lag_1h",
        "pm25_lag_2h",
        "pm25_lag_24h",
        "pm25_rolling_mean_6h",
    ]

    X_train, y_train = train_df[feature_cols], train_df["target"]
    X_test, y_test = test_df[feature_cols], test_df["target"]

    model = LGBMRegressor(
        n_estimators=100, max_depth=5, learning_rate=0.05, random_state=42
    )
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    mape = mean_absolute_percentage_error(y_test, predictions)
    accuracy_percentage = (1 - mape) * 100

    print("\n=========================================")
    print("       MODEL EVALUATION REPORT           ")
    print("=========================================")
    print(f"Mean Absolute Error (MAE):    {mae:.2f} µg/m³")
    print(f"R² Score (Variance Explained): {r2:.2f}")
    print(f"Model Error Rate:             {mape * 100:.2f}%")
    print(f"Model Accuracy Rating:        {accuracy_percentage:.2f}%")
    print("=========================================\n")

    joblib.dump(model, "air_quality_model.pkl")
    joblib.dump(feature_cols, "model_features.pkl")
    print("Model trained and saved successfully.")


if __name__ == "__main__":
    train_pipeline()