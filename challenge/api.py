"""FastAPI service exposing the SCL flight-delay model.

Endpoints
---------
* ``GET  /health``  — liveness probe.
* ``POST /predict`` — batch delay prediction for a list of flights.

Input is validated with Pydantic; any invalid field (unknown airline, bad flight
type, month out of range) yields **HTTP 400** (the challenge tests assert 400,
not FastAPI's default 422 — see the exception handler below).
"""

from __future__ import annotations

import fastapi
import pandas as pd
from fastapi import status
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, validator

from challenge.model import DelayModel

# Airlines seen in the training data — the only OPERA values we accept.
KNOWN_OPERA = frozenset(
    {
        "Aerolineas Argentinas",
        "Aeromexico",
        "Air Canada",
        "Air France",
        "Alitalia",
        "American Airlines",
        "Austral",
        "Avianca",
        "British Airways",
        "Copa Air",
        "Delta Air",
        "Gol Trans",
        "Grupo LATAM",
        "Iberia",
        "JetSmart SPA",
        "K.L.M.",
        "Lacsa",
        "Latin American Wings",
        "Oceanair Linhas Aereas",
        "Plus Ultra Lineas Aereas",
        "Qantas Airways",
        "Sky Airline",
        "United Airlines",
    }
)
VALID_TIPOVUELO = frozenset({"I", "N"})

app = fastapi.FastAPI(title="SCL Flight Delay Prediction API", version="1.0.0")

# Load the champion model once at start-up (artifact is loaded in __init__).
model = DelayModel()


class Flight(BaseModel):
    """A single flight to score."""

    OPERA: str
    TIPOVUELO: str
    MES: int

    @validator("MES")
    def _month_in_range(cls, value: int) -> int:
        if not 1 <= value <= 12:
            raise ValueError("MES must be between 1 and 12")
        return value

    @validator("TIPOVUELO")
    def _known_flight_type(cls, value: str) -> str:
        if value not in VALID_TIPOVUELO:
            raise ValueError("TIPOVUELO must be 'I' (international) or 'N' (national)")
        return value

    @validator("OPERA")
    def _known_airline(cls, value: str) -> str:
        if value not in KNOWN_OPERA:
            raise ValueError("OPERA is not a known airline")
        return value


class PredictRequest(BaseModel):
    """Batch of flights to predict."""

    flights: list[Flight]


class PredictResponse(BaseModel):
    predict: list[int]


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: fastapi.Request, exc: RequestValidationError
) -> JSONResponse:
    """Return 400 (not the default 422) for invalid input, as the tests expect."""
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content=jsonable_encoder({"detail": exc.errors()}),
    )


@app.get("/health", status_code=200)
async def get_health() -> dict:
    return {"status": "OK"}


@app.post("/predict", status_code=200, response_model=PredictResponse)
async def post_predict(request: PredictRequest) -> PredictResponse:
    data = pd.DataFrame([flight.dict() for flight in request.flights])
    features = model.preprocess(data)
    return PredictResponse(predict=model.predict(features))
