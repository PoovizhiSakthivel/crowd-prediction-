from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
import random

app = Flask(__name__)

# ── Load model artifacts ───────────────────────────────────────────
model    = joblib.load("model/crowd_model.pkl")
encoders = joblib.load("model/encoders.pkl")
meta     = joblib.load("model/meta.pkl")

CROWD_LABELS = {0: "Low", 1: "Medium", 2: "High", 3: "Very High"}
CROWD_COLORS = {
    "Low":       "#22c55e",
    "Medium":    "#f59e0b",
    "High":      "#ef4444",
    "Very High": "#7c3aed",
}

DAY_MAP = {
    "Monday": 0, "Tuesday": 1, "Wednesday": 2,
    "Thursday": 3, "Friday": 4, "Saturday": 5, "Sunday": 6
}

# ── Weather simulation (fallback if no API) ────────────────────────
def simulate_weather(date_str):
    try:
        dt    = datetime.strptime(date_str, "%Y-%m-%d")
        month = dt.month
        if month in [10, 11, 12]:
            w = random.choices(["Rainy", "Cloudy", "Sunny"], weights=[50, 30, 20])[0]
        elif month in [6, 7, 8, 9]:
            w = random.choices(["Rainy", "Cloudy", "Sunny"], weights=[35, 40, 25])[0]
        else:
            w = random.choices(["Sunny", "Cloudy", "Rainy"], weights=[65, 25, 10])[0]
        base_temp = {1:27,2:29,3:32,4:34,5:35,
                     6:33,7:31,8:30,9:31,10:30,11:28,12:27}
        temp = base_temp.get(month, 30) + random.randint(-2, 2)
        return w, temp
    except Exception:
        return "Sunny", 30


# ── Real weather via OpenWeatherMap (optional) ─────────────────────
OWM_API_KEY = "2dedba9187cab674896db3b70e02c77f"   # paste your key here to enable real weather

def get_weather(date_str, city):
    if not OWM_API_KEY:
        return simulate_weather(date_str)
    try:
        import requests
        target = datetime.strptime(date_str, "%Y-%m-%d").date()
        today  = datetime.today().date()
        delta  = (target - today).days
        if 0 <= delta <= 5:
            url    = "https://api.openweathermap.org/data/2.5/forecast"
            params = {"q": city+",IN", "appid": OWM_API_KEY, "units": "metric"}
            res    = requests.get(url, params=params, timeout=5)
            data   = res.json()
            if data.get("cod") != "200":
                raise ValueError("API error")
            target_str = date_str + " 12:00:00"
            entry = min(data["list"], key=lambda x: abs(
                (datetime.strptime(x["dt_txt"], "%Y-%m-%d %H:%M:%S")
                 - datetime.strptime(target_str, "%Y-%m-%d %H:%M:%S")).total_seconds()
            ))
            temp  = round(entry["main"]["temp"])
            desc  = entry["weather"][0]["main"]
            wmap  = {"Rain":"Rainy","Drizzle":"Rainy","Thunderstorm":"Rainy",
                     "Clouds":"Cloudy","Mist":"Cloudy","Haze":"Cloudy",
                     "Clear":"Sunny","Smoke":"Cloudy","Fog":"Cloudy"}
            return wmap.get(desc, "Sunny"), temp
        else:
            return simulate_weather(date_str)
    except Exception as e:
        print(f"Weather API error: {e} — using simulation")
        return simulate_weather(date_str)


@app.route("/")
def index():
    return render_template(
        "index.html",
        locations=meta["locations"],
        destinations=meta["destinations"],
    )


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        date_str    = data.get("date", "")
        time_str    = data.get("time", "12:00")
        location    = data.get("location", "")
        destination = data.get("destination", "")

        # ── Derive day from date ───────────────────────────────────
        dt      = datetime.strptime(date_str, "%Y-%m-%d")
        day_str = dt.strftime("%A")
        day_num = DAY_MAP[day_str]
        month   = dt.month
        dom     = dt.day
        hour    = int(time_str.split(":")[0])

        # ── Weather ────────────────────────────────────────────────
        weather, temperature = get_weather(date_str, location)

        # ── Encode categoricals ────────────────────────────────────
        def safe_encode(enc, val):
            classes = list(enc.classes_)
            return int(enc.transform([val])[0]) if val in classes else 0

        loc_enc  = safe_encode(encoders["Location"],    location)
        dest_enc = safe_encode(encoders["Destination"], destination)
        wea_enc  = safe_encode(encoders["Weather"],     weather)

        # ── Engineered features ────────────────────────────────────
        is_weekend   = 1 if day_num >= 5 else 0
        is_peak_hour = 1 if hour in [7,8,9,17,18,19,20] else 0
        is_saturday  = 1 if day_num == 5 else 0
        is_sunday    = 1 if day_num == 6 else 0
        is_rainy     = 1 if weather == "Rainy" else 0
        hot_day      = 1 if temperature > 33 else 0

        hour_sin  = np.sin(2*np.pi*hour/24)
        hour_cos  = np.cos(2*np.pi*hour/24)
        month_sin = np.sin(2*np.pi*month/12)
        month_cos = np.cos(2*np.pi*month/12)
        day_sin   = np.sin(2*np.pi*day_num/7)
        day_cos   = np.cos(2*np.pi*day_num/7)

        weekend_peak  = is_weekend * is_peak_hour
        weekend_rainy = is_weekend * is_rainy
        sat_peak      = is_saturday * is_peak_hour
        sun_peak      = is_sunday   * is_peak_hour

        # ── Build feature DataFrame (must match FEATURES order) ────
        FEATURES = meta["features"]
        X = pd.DataFrame([[
            hour, day_num, month, dom,
            loc_enc, dest_enc, wea_enc, temperature,
            is_weekend, is_peak_hour, is_saturday, is_sunday,
            is_rainy, hot_day,
            hour_sin, hour_cos, month_sin, month_cos, day_sin, day_cos,
            weekend_peak, weekend_rainy, sat_peak, sun_peak
        ]], columns=FEATURES)

        # ── Predict ────────────────────────────────────────────────
        crowd_idx  = int(model.predict(X)[0])
        crowd_lvl  = CROWD_LABELS.get(crowd_idx, "Low")
        crowd_color= CROWD_COLORS.get(crowd_lvl, "#22c55e")

        # ── Confidence from tree vote distribution ─────────────────
        proba      = model.predict_proba(X)[0]
        confidence = round(float(proba[crowd_idx]) * 100, 1)

        return jsonify({
            "success":      True,
            "date":         date_str,
            "day":          day_str,
            "time":         time_str,
            "location":     location,
            "destination":  destination,
            "weather":      weather,
            "temperature":  temperature,
            "crowd_level":  crowd_lvl,
            "crowd_color":  crowd_color,
            "confidence":   confidence,
            "crowd_index":  crowd_idx,
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


@app.route("/meta")
def get_meta():
    return jsonify({
        "locations":    meta["locations"],
        "destinations": meta["destinations"],
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000)