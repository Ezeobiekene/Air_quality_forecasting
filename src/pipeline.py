import pandas as pd
import requests

def fetch_historical_air_quality(
    lat: float, lon: float, start_date: str, end_date: str
):
    """Fetches a custom window of historical air quality data (e.g., 2 years)

    using explicit start and end dates.
    Dates must be in 'YYYY-MM-DD' format.
    """
    url = "https://air-quality-api.open-meteo.com/v1/air-quality"

    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "pm2_5,pm10,ozone,nitrogen_dioxide",
        "start_date": start_date,
        "end_date": end_date,
        "timezone": "auto",
    }

    print(f"Requesting data from {start_date} to {end_date}...")
    response = requests.get(url, params=params)

    if response.status_code != 200:
        raise Exception(f"API Error: {response.status_code} - {response.text}")

    data = response.json()
    hourly_data = data["hourly"]

    df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(hourly_data["time"]),
            "pm2_5": hourly_data["pm2_5"],
            "pm10": hourly_data["pm10"],
            "ozone": hourly_data["ozone"],
            "no2": hourly_data["nitrogen_dioxide"],
        }
    )

    df.set_index("timestamp", inplace=True)
    return df


if __name__ == "__main__":
    # Query a clean 2-year range up to the current date (June 2026)
    df_air = fetch_historical_air_quality(
        lat=29.76, lon=-95.36, start_date="2024-06-01", end_date="2026-06-29"
    )

    print(f"Ingestion Successful! Total hourly rows: {len(df_air)}")
    print(df_air.head())

    # Overwrite your local CSV data store with the comprehensive dataset
    df_air.to_csv("raw_air_quality.csv")
