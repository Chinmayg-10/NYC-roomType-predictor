
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Initialize FastAPI
app = FastAPI(title="NYC Airbnb Prediction API")

# CORS configuration (development)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Feature columns: must match the model's training data
COLUMNS = [
    "latitude",
    "longitude",
    "price",
    "minimum_nights",
    "number_of_reviews",
    "reviews_per_month",
    "calculated_host_listings_count",
    "availability_365",
    "neighbourhood_group",
    "neighbourhood",
]

# Load the saved, fitted ML pipeline
MODEL_PATH = Path(__file__).resolve().parent / "Model_pipeline_compressed.pkl"

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

model = joblib.load(MODEL_PATH)


# Validate incoming request data
class Features(BaseModel):
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)
    price: float = Field(..., gt=0)
    minimum_nights: int = Field(..., ge=1, le=365)
    number_of_reviews: int = Field(..., ge=0)
    reviews_per_month: float = Field(..., ge=0)
    calculated_host_listings_count: int = Field(..., ge=0)
    availability_365: int = Field(..., ge=0, le=365)
    neighbourhood_group: str = Field(..., min_length=1)
    neighbourhood: str = Field(..., min_length=1)


@app.get("/")
def greet():
    return {"message": "NYC Airbnb Prediction API is running"}


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict")
def predict(features: Features):
    try:
        # Convert input into a one-row DataFrame
        row = pd.DataFrame(
            [features.model_dump()],
            columns=COLUMNS
        )

        # Make prediction using the complete fitted pipeline
        prediction = model.predict(row)[0]

        # Calculate class probabilities when supported
        probabilities = None
        if hasattr(model, "predict_proba"):
            probabilities = [
                float(p) for p in model.predict_proba(row)[0]
            ]

        return {
            "predicted_room_type": str(prediction),
            "probabilities": probabilities
        }

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(exc)}"
        ) from exc
