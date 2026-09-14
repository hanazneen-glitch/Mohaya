import pickle
from datetime import datetime
from typing import Dict, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sklearn.ensemble import RandomForestRegressor


# ============================================================
# 1. Traffic Model
# ============================================================

MODEL_PATH = "traffic_model.pkl"


def train_traffic_model():
    dates = pd.date_range(
        start="2026-01-01",
        end="2026-12-31 23:00:00",
        freq="h"
    )

    df = pd.DataFrame({"timestamp": dates})

    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["is_weekend"] = df["day_of_week"].apply(
        lambda x: 1 if x in [4, 5] else 0
    )

    def simulate_traffic(row):
        base_cars = 40

        if 16 <= row["hour"] <= 21:
            base_cars += 80
        elif 7 <= row["hour"] <= 10:
            base_cars += 40
        elif 1 <= row["hour"] <= 5:
            base_cars -= 25

        if row["is_weekend"]:
            base_cars += 50

        cars = max(
            5,
            int(base_cars + np.random.normal(0, 10))
        )

        wait_minutes = int(
            cars * 0.4 + np.random.normal(0, 3)
        )

        return max(5, wait_minutes)

    df["wait_minutes"] = df.apply(
        simulate_traffic,
        axis=1
    )

    X = df[
        [
            "hour",
            "day_of_week",
            "is_weekend"
        ]
    ]

    y = df["wait_minutes"]

    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42
    )

    model.fit(X, y)

    return model


# Try to load the trained model.
# If it doesn't exist, train it automatically.
try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

except FileNotFoundError:
    model = train_traffic_model()

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)


# ============================================================
# 2. FastAPI
# ============================================================

app = FastAPI(
    title="Mohaya Core API",
    version="1.0.0"
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# 3. Request Model
# ============================================================

class ReadinessRequest(BaseModel):
    destination: str

    crossing: Optional[str] = None
    nationality: Optional[str] = None

    document_type: Optional[str] = None
    document_number: Optional[str] = None
    date_of_birth: Optional[str] = None
    document_issue_date: Optional[str] = None
    document_expiry_date: Optional[str] = None

    travel_date: str
    arrival_time: str

    travel_mode: str = "car"

    vehicle_type: Optional[str] = None
    vehicle_ownership: Optional[str] = None

    documents: Dict[str, Optional[str]] = {}


# ============================================================
# 4. Readiness Evaluation
# ============================================================

def evaluate_readiness(
    travel_date_str: str,
    documents: dict,
    travel_mode: str = "car"
):
    travel_date = datetime.strptime(
        travel_date_str,
        "%Y-%m-%d"
    ).date()

    if travel_mode.lower() == "car":

        doc_rules = {
            "passport": {
                "weight": 30,
                "label": "جواز السفر / الهوية الوطنية"
            },

            "insurance": {
                "weight": 25,
                "label": "تأمين المركبة"
            },

            "license": {
                "weight": 25,
                "label": "رخصة القيادة"
            },

            "vehicle_inspection": {
                "weight": 20,
                "label": "الفحص الدوري للمركبة"
            }
        }

    else:

        doc_rules = {
            "passport": {
                "weight": 100,
                "label": "جواز السفر / الهوية الوطنية"
            }
        }

    total_score = 0

    missing_docs = []

    doc_details = {}

    for key, rule in doc_rules.items():

        expiry_str = documents.get(key)

        label = rule["label"]

        weight = rule["weight"]

        if expiry_str:

            try:
                exp_date = datetime.strptime(
                    expiry_str,
                    "%Y-%m-%d"
                ).date()

                if exp_date >= travel_date:

                    total_score += weight

                    doc_details[label] = "سارية"

                    continue

            except ValueError:
                pass

        doc_details[label] = "منتهية أو ناقصة"

        missing_docs.append(
            f"{label} منتهية أو غير متوفرة"
        )

    if total_score >= 85:

        level = "جاهزية عالية"

    elif total_score >= 60:

        level = "جاهزية متوسطة"

    else:

        level = "جاهزية منخفضة"

    requirements_total = len(doc_details)

    requirements_met = sum(
        1
        for status in doc_details.values()
        if status == "سارية"
    )

    return (
        total_score,
        level,
        missing_docs,
        doc_details,
        requirements_total,
        requirements_met
    )


# ============================================================
# 5. Traffic Prediction
# ============================================================

def predict_traffic_and_best_time(
    travel_date_str: str,
    arrival_hour: int
):

    dt = datetime.strptime(
        travel_date_str,
        "%Y-%m-%d"
    )

    day_of_week = dt.weekday()

    is_weekend = (
        1 if day_of_week in [4, 5]
        else 0
    )

    day_features = [
        [
            h,
            day_of_week,
            is_weekend
        ]
        for h in range(24)
    ]

    predictions = (
        model.predict(day_features)
        .astype(int)
        .tolist()
    )

    wait_time_minutes = predictions[
        arrival_hour
    ]

    if wait_time_minutes > 45:

        traffic_level = "high"

    elif wait_time_minutes > 25:

        traffic_level = "medium"

    else:

        traffic_level = "low"

    start = max(
        0,
        arrival_hour - 3
    )

    end = min(
        23,
        arrival_hour + 3
    )

    best_hour = min(
        range(start, end + 1),
        key=lambda h: predictions[h]
    )

    recommended_time = (
        f"{best_hour:02d}:00-"
        f"{(best_hour + 1):02d}:00"
    )

    return (
        traffic_level,
        wait_time_minutes,
        recommended_time,
        predictions
    )


# ============================================================
# 6. Recommendation
# ============================================================

def generate_recommendation(
    missing_docs: list,
    traffic_level: str,
    recommended_time: str,
    arrival_time_str: str
):

    recs = []

    if missing_docs:

        recs.append(
            "أكمل تجديد المتطلبات التالية: "
            f"({' ، '.join(missing_docs)})"
        )

    else:

        recs.append(
            "جميع وثائقك مكتملة وجاهزة للعبور"
        )

    if traffic_level == "high":

        recs.append(
            f"يتزامن وصولك عند الساعة "
            f"{arrival_time_str} مع ذروة ازدحام مرتفعة، "
            f"ويُفضل العبور بين "
            f"{recommended_time} لتفادي التأخير"
        )

    elif traffic_level == "medium":

        recs.append(
            f"حركة السير متوسطة عند الساعة "
            f"{arrival_time_str}، "
            f"وبإمكانك اختيار الفترة "
            f"{recommended_time} لعبور أسرع"
        )

    else:

        recs.append(
            f"توقيت وصولك عند الساعة "
            f"{arrival_time_str} ممتاز "
            f"وحركة السير خفيفة عند المنفذ"
        )

    return ". ".join(recs)


# ============================================================
# 7. API Endpoint
# ============================================================

@app.post("/api/check-readiness")
def check_readiness(data: ReadinessRequest):

    arrival_hour = int(
        data.arrival_time.split(":")[0]
    )

    (
        score,
        level,
        missing,
        doc_status,
        requirements_total,
        requirements_met
    ) = evaluate_readiness(
        data.travel_date,
        data.documents,
        data.travel_mode
    )

    (
        traffic_level,
        wait_minutes,
        recommended_time,
        hourly_forecast
    ) = predict_traffic_and_best_time(
        data.travel_date,
        arrival_hour
    )

    recommendation = generate_recommendation(
        missing,
        traffic_level,
        recommended_time,
        data.arrival_time
    )

    return {

        "readiness_score": score,

        "readiness_level": level,

        "traffic_level": traffic_level,

        "expected_wait_minutes": wait_minutes,

        "recommended_time": recommended_time,

        "missing_requirements": missing,

        "document_status": doc_status,

        "hourly_forecast": hourly_forecast,

        "recommendation": recommendation,

        "requirements_total": requirements_total,

        "requirements_met": requirements_met
    }


# ============================================================
# 8. Health Check
# ============================================================

@app.get("/")
def root():

    return {
        "status": "ok",
        "service": "Mohaya Core API"
    }


@app.get("/health")
def health():

    return {
        "status": "healthy"
    }


# ============================================================
# Local development
# ============================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000
    )
