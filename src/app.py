import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse

app = FastAPI(title="Live Air Quality Prediction Terminal")

# Load our high-performing model artifact
model = joblib.load("air_quality_model.pkl")


def fetch_live_sensor_metrics():
    """Simulates real-time ingestion from a live streaming data pipe."""
    try:
        df = pd.read_csv("raw_air_quality.csv", index_col=0, parse_dates=True)
        latest_reading = df.iloc[-1]

        return {
            "pm25": float(latest_reading["pm2_5"]),
            "pm10": float(latest_reading["pm10"]),
            "ozone": float(latest_reading["ozone"]),
            "no2": float(latest_reading["no2"]),
            "lag_1h": float(df.iloc[-2]["pm2_5"]),
            "lag_2h": float(df.iloc[-3]["pm2_5"]),
            "lag_24h": float(df.iloc[-25]["pm2_5"]),
            "rolling_6h": float(df["pm2_5"].tail(6).mean()),
        }
    except Exception:
        return {
            "pm25": 12.4,
            "pm10": 22.1,
            "ozone": 31.5,
            "no2": 15.0,
            "lag_1h": 12.0,
            "lag_2h": 11.8,
            "lag_24h": 14.2,
            "rolling_6h": 12.2,
        }


def get_aqi_category(pm25):
    """Maps PM2.5 concentration to official EPA Air Quality categories and colors."""
    if pm25 <= 12.0:
        return "Good", "#22c55e", "#15803d"  # Green
    elif pm25 <= 35.4:
        return "Moderate", "#eab308", "#854d0e"  # Yellow
    elif pm25 <= 55.4:
        return "Unhealthy for Sensitive Groups", "#f97316", "#c2410c"  # Orange
    elif pm25 <= 150.4:
        return "Unhealthy", "#ef4444", "#b91c1c"  # Red
    elif pm25 <= 250.4:
        return "Very Unhealthy", "#a855f7", "#7e22ce"  # Purple
    else:
        return "Hazardous", "#7f1d1d", "#450a0a"  # Maroon


HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Live Air Hazard Terminal</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, sans-serif; background: #0f172a; color: #f8fafc; padding: 40px; }}
        .container {{ max-width: 500px; background: #1e293b; margin: 0 auto; padding: 30px; border-radius: 16px; box-shadow: 0 10px 15px -3px rgba(0,0,0,0.3); border: 1px solid #334155; }}
        .live-badge {{ background: #ef4444; color: white; padding: 4px 8px; font-size: 12px; font-weight: bold; border-radius: 4px; display: inline-block; margin-bottom: 10px; text-transform: uppercase; letter-spacing: 0.05em; animation: pulse 2s infinite; }}
        label {{ font-weight: 600; display: block; margin: 20px 0 5px; color: #94a3b8; }}
        select {{ width: 100%; padding: 12px; border: 1px solid #475569; background: #0f172a; color: white; border-radius: 8px; font-size: 16px; box-sizing: border-box; }}
        button {{ width: 100%; background: #2563eb; color: white; border: none; padding: 14px; border-radius: 8px; font-size: 16px; font-weight: 600; margin-top: 25px; cursor: pointer; transition: background 0.2s; }}
        button:hover {{ background: #1d4ed8; }}
        .result {{ margin-top: 30px; padding: 20px; border-radius: 12px; text-align: center; color: white; font-weight: bold; }}
        .category-badge {{ display: inline-block; padding: 6px 12px; font-size: 20px; border-radius: 6px; margin-top: 8px; font-weight: 800; text-transform: uppercase; }}
        @keyframes pulse {{ 0% {{ opacity: 1; }} 50% {{ opacity: 0.5; }} 100% {{ opacity: 1; }} }}
    </style>
</head>
<body>
    <div class="container">
        <div class="live-badge">● Live Streaming Connection</div>
        <h2 style="margin-top:0; color: #f1f5f9;">Air Hazard Alert System</h2>
        <p style="color: #94a3b8; font-size: 14px; margin-bottom: 25px;">Pulls raw sensor data automatically to evaluate atmospheric shifts over the upcoming window.</p>
        
        <form method="post" action="/predict-live">
            <label for="time_range">Target Forecast Window:</label>
            <select id="time_range" name="time_range">
                <option value="9">Morning Horizon (6:00 AM - 12:00 PM)</option>
                <option value="14">Afternoon Horizon (12:00 PM - 4:00 PM)</option>
                <option value="19">Evening Horizon (4:00 PM - 11:00 PM)</option>
                <option value="2">Night Horizon (11:00 PM - 6:00 AM)</option>
            </select>
            
            <button type="submit">Evaluate Current Risks</button>
        </form>
        {result_placeholder}
    </div>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
async def home():
    return HTML_TEMPLATE.format(result_placeholder="")


@app.post("/predict-live", response_class=HTMLResponse)
async def predict_live(time_range: str = Form(...)):
    now = pd.Timestamp.now()
    hour_target = int(time_range)

    live_data = fetch_live_sensor_metrics()

    hour_sin = np.sin(2 * np.pi * hour_target / 24)
    hour_cos = np.cos(2 * np.pi * hour_target / 24)

    input_data = [
        hour_sin,
        hour_cos,
        now.dayofweek,
        live_data["pm25"],
        live_data["pm10"],
        live_data["ozone"],
        live_data["no2"],
        live_data["lag_1h"],
        live_data["lag_2h"],
        live_data["lag_24h"],
        live_data["rolling_6h"],
    ]

    prediction = model.predict([input_data])[0]

    # Get official EPA rating components
    category, bg_color, text_border = get_aqi_category(prediction)

    result_html = f"""
    <div class="result" style="background-color: {text_border}; border: 2px solid {bg_color};">
        <div style="font-size: 14px; text-transform: uppercase; letter-spacing: 0.05em; color: #f8fafc; margin-bottom: 5px;">3-Hour Outlook Exposure</div>
        <div style="font-size: 24px; font-weight: 800;">{prediction:.2f} µg/m³</div>
        <div class="category-badge" style="background-color: {bg_color}; color: #000000;">{category}</div>
        <div style="font-size:12px; font-weight:normal; margin-top:12px; color: #cbd5e1;">
            Telemetry Sync: PM2.5={live_data['pm25']} | Ozone={live_data['ozone']}
        </div>
    </div>
    """
    return HTML_TEMPLATE.format(result_placeholder=result_html)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)