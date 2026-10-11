# ============================================================
# FastAPI application for UFC fight prediction.
# ============================================================

from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

import mlflow
import mlflow.sklearn
import pandas as pd


print("\n" + "=" * 70)
print("PHASE 11 — UFC PREDICTION API")
print("=" * 70)


# ============================================================
# 11.1 — PROJECT & MLflow CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(
    r"C:\Users\pc\Downloads\PROJET DS"
)

MLFLOW_DB_PATH = (
    PROJECT_ROOT / "mlflow.db"
)


if not MLFLOW_DB_PATH.exists():

    raise FileNotFoundError(
        f"MLflow database not found: {MLFLOW_DB_PATH}"
    )


MLFLOW_DB = (
    "sqlite:///"
    + str(MLFLOW_DB_PATH).replace("\\", "/")
)


mlflow.set_tracking_uri(
    MLFLOW_DB
)


# IMPORTANT:
# This is the successful MLflow run from Phase 10.

RUN_ID = (
    "e1ef1e839fab49218e1126e342558333"
)


MODEL_URI = (
    f"runs:/{RUN_ID}/random_forest_model"
)


print("\n11.1 — MLflow Configuration")
print("-" * 70)

print(
    f"Project root : {PROJECT_ROOT}"
)

print(
    f"MLflow DB    : {MLFLOW_DB_PATH}"
)

print(
    f"Database exists: {MLFLOW_DB_PATH.exists()}"
)

print(
    f"Run ID       : {RUN_ID}"
)

print(
    f"Model URI    : {MODEL_URI}"
)


# ============================================================
# 11.2 — LOAD MLflow MODEL
# ============================================================

print("\n11.2 — Loading MLflow Model")
print("-" * 70)


try:

    model = mlflow.sklearn.load_model(
        MODEL_URI
    )

    print(
        "Model loaded successfully."
    )

    print(
        f"Run ID: {RUN_ID}"
    )


except Exception as error:

    print(
        "ERROR: Unable to load MLflow model."
    )

    print(
        error
    )

    model = None


# ============================================================
# 11.3 — FASTAPI APPLICATION
# ============================================================

app = FastAPI(

    title="UFC Fight Prediction API",

    description=(
        "Machine Learning API for predicting "
        "UFC fight outcomes using pre-fight data."
    ),

    version="1.0.0"
)


# ============================================================
# 11.4 — INPUT SCHEMA
# ============================================================

class FightPredictionRequest(BaseModel):

    # --------------------------------------------------------
    # Fighter 1
    # --------------------------------------------------------

    f_1_previous_wins: int = Field(
        ge=0
    )

    f_1_previous_losses: int = Field(
        ge=0
    )

    f_1_previous_draws: int = Field(
        ge=0
    )

    f_1_previous_fights: int = Field(
        ge=0
    )

    f_1_height: float | None = Field(
        default=None,
        gt=0
    )

    f_1_fighter_weight_lbs: float | None = Field(
        default=None,
        gt=0
    )

    f_1_fighter_reach_cm: float | None = Field(
        default=None,
        gt=0
    )

    f_1_fighter_stance: str | None = None


    # --------------------------------------------------------
    # Fighter 2
    # --------------------------------------------------------

    f_2_previous_wins: int = Field(
        ge=0
    )

    f_2_previous_losses: int = Field(
        ge=0
    )

    f_2_previous_draws: int = Field(
        ge=0
    )

    f_2_previous_fights: int = Field(
        ge=0
    )

    f_2_height: float | None = Field(
        default=None,
        gt=0
    )

    f_2_fighter_weight_lbs: float | None = Field(
        default=None,
        gt=0
    )

    f_2_fighter_reach_cm: float | None = Field(
        default=None,
        gt=0
    )

    f_2_fighter_stance: str | None = None


    # --------------------------------------------------------
    # Fight Context
    # --------------------------------------------------------

    weight_class: str

    gender: str

    num_rounds: int = Field(
        gt=0,
        le=5
    )

    title_fight: bool


# ============================================================
# 11.5 — FEATURE ENGINEERING
# ============================================================

def build_features(
    request: FightPredictionRequest
):

    if request.f_1_previous_fights > 0:

        win_rate_1 = (
            request.f_1_previous_wins
            / request.f_1_previous_fights
        )

    else:

        win_rate_1 = 0.0


    if request.f_2_previous_fights > 0:

        win_rate_2 = (
            request.f_2_previous_wins
            / request.f_2_previous_fights
        )

    else:

        win_rate_2 = 0.0


    win_rate_diff = (
        win_rate_1
        - win_rate_2
    )


    experience_diff = (
        request.f_1_previous_fights
        - request.f_2_previous_fights
    )


    wins_diff = (
        request.f_1_previous_wins
        - request.f_2_previous_wins
    )


    losses_diff = (
        request.f_1_previous_losses
        - request.f_2_previous_losses
    )


    draws_diff = (
        request.f_1_previous_draws
        - request.f_2_previous_draws
    )


    features = {

        "f_1_previous_wins":
            request.f_1_previous_wins,

        "f_1_previous_losses":
            request.f_1_previous_losses,

        "f_1_previous_draws":
            request.f_1_previous_draws,

        "f_1_previous_fights":
            request.f_1_previous_fights,


        "f_2_previous_wins":
            request.f_2_previous_wins,

        "f_2_previous_losses":
            request.f_2_previous_losses,

        "f_2_previous_draws":
            request.f_2_previous_draws,

        "f_2_previous_fights":
            request.f_2_previous_fights,


        "win_rate_diff":
            win_rate_diff,

        "experience_diff":
            experience_diff,

        "wins_diff":
            wins_diff,

        "losses_diff":
            losses_diff,

        "draws_diff":
            draws_diff,


        "f_1_fighter_height_cm":
            request.f_1_height,

        "f_1_fighter_weight_lbs":
            request.f_1_fighter_weight_lbs,

        "f_1_fighter_reach_cm":
            request.f_1_fighter_reach_cm,


        "f_2_fighter_height_cm":
            request.f_2_height,

        "f_2_fighter_weight_lbs":
            request.f_2_fighter_weight_lbs,

        "f_2_fighter_reach_cm":
            request.f_2_fighter_reach_cm,


        "weight_class":
            request.weight_class,

        "gender":
            request.gender,

        "num_rounds":
            request.num_rounds,

        "title_fight":
            request.title_fight,


        "f_1_fighter_stance":
            request.f_1_fighter_stance,

        "f_2_fighter_stance":
            request.f_2_fighter_stance
    }


    return features


# ============================================================
# 11.6 — ROOT ENDPOINT
# ============================================================

@app.get("/")
def root():

    return {

        "project":
            "UFC Fight Prediction",

        "status":
            "online",

        "model":
            "Random Forest",

        "mlflow_run_id":
            RUN_ID,

        "version":
            "1.0.0"
    }


# ============================================================
# 11.7 — HEALTH CHECK
# ============================================================

@app.get("/health")
def health():

    if model is None:

        return {

            "status":
                "error",

            "model_loaded":
                False
        }


    return {

        "status":
            "healthy",

        "model_loaded":
            True,

        "model":
            "Random Forest",

        "mlflow_run_id":
            RUN_ID
    }


# ============================================================
# 11.8 — PREDICTION ENDPOINT
# ============================================================

@app.post("/predict")
def predict(
    request: FightPredictionRequest
):

    if model is None:

        raise HTTPException(

            status_code=500,

            detail=(
                "Prediction model is not loaded."
            )
        )


    try:

        features = build_features(
            request
        )


        input_data = pd.DataFrame(
            [features]
        )


        probabilities = (
            model.predict_proba(
                input_data
            )[0]
        )


        fighter_1_probability = float(
            probabilities[1]
        )


        fighter_2_probability = float(
            probabilities[0]
        )


        if (
            fighter_1_probability
            >= fighter_2_probability
        ):

            predicted_winner = "fighter_1"

        else:

            predicted_winner = "fighter_2"


        confidence = max(
            fighter_1_probability,
            fighter_2_probability
        )


        return {

            "predicted_winner":
                predicted_winner,

            "fighter_1_probability":
                round(
                    fighter_1_probability,
                    4
                ),

            "fighter_2_probability":
                round(
                    fighter_2_probability,
                    4
                ),

            "confidence":
                round(
                    confidence,
                    4
                )
        }


    except Exception as error:

        raise HTTPException(

            status_code=500,

            detail=str(error)
        )


# ============================================================
# 11.9 — API INFORMATION
# ============================================================

print("\n11.9 — API Ready")
print("-" * 70)

print(
    "Run with:"
)

print(
    "uvicorn api.main:app --reload"
)

print(
    "Swagger UI:"
)

print(
    "http://127.0.0.1:8000/docs"
)

print("=" * 70)
