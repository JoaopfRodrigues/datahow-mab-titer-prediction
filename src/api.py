"""REST API for mAb titer prediction.

Implements the inference server per the OpenAPI spec
(inference_server_spec.yml). Two endpoints:
  - GET  /health  -- service health check
  - POST /predict -- run inference on bioprocess experiment data
"""

import logging
import math
import os
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import numpy as np
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, field_validator, model_validator

from src.features import W_COLS, X_COLS, Z_COLS, extract_features_from_payload

logger = logging.getLogger(__name__)

MODEL_PATH = Path(
    os.environ.get("MODEL_PATH", str(Path(__file__).parent.parent / "models" / "titer_model.pkl"))
)


def _load_model(path: Path = MODEL_PATH) -> dict | None:
    if not path.exists():
        logger.error("Model file not found at %s", path)
        return None
    try:
        return joblib.load(path)
    except Exception:
        logger.exception("Failed to load model from %s", path)
        return None


# --- Pydantic DTOs ---


class PredictRequest(BaseModel):
    timestamps: list[float]
    values: dict[str, list[float]]

    @field_validator("timestamps")
    @classmethod
    def timestamps_not_empty(cls, v: list[float]) -> list[float]:
        if len(v) < 2:
            raise ValueError("timestamps must have at least 2 entries")
        if any(math.isnan(x) or math.isinf(x) for x in v):
            raise ValueError("timestamps must not contain NaN or Inf")
        if any(b <= a for a, b in zip(v, v[1:])):
            raise ValueError("timestamps must be strictly monotonically increasing")
        if v[0] < 0:
            raise ValueError("timestamps must not be negative")
        return v

    @field_validator("values")
    @classmethod
    def values_has_required_keys(cls, v: dict[str, list[float]]) -> dict[str, list[float]]:
        for key, arr in v.items():
            if any(math.isnan(x) or math.isinf(x) for x in arr):
                raise ValueError(f"Variable '{key}' contains non-finite values")
        missing = [k for k in Z_COLS + W_COLS + X_COLS if k not in v]
        if missing:
            raise ValueError(f"Missing required variables: {missing}")
        return v

    @model_validator(mode="after")
    def validate_array_lengths(self) -> "PredictRequest":
        n_ts = len(self.timestamps)
        for key, arr in self.values.items():
            if key.startswith("Z:"):
                if len(arr) != 1:
                    raise ValueError(f"Z: variable '{key}' must have exactly 1 value, got {len(arr)}")
            elif len(arr) != n_ts:
                raise ValueError(f"Variable '{key}' has {len(arr)} values but {n_ts} timestamps")

        if "Z:ExpDuration" in self.values:
            declared = self.values["Z:ExpDuration"][0]
            actual = self.timestamps[-1]
            if abs(declared - actual) > 0.5:
                raise ValueError(
                    f"Z:ExpDuration ({declared}) does not match last timestamp ({actual})"
                )
        return self


class PredictResponse(BaseModel):
    predicted_titer: float
    confidence_lower: float
    confidence_upper: float


class HealthResponse(BaseModel):
    status: str


# --- FastAPI application ---


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model_bundle = _load_model()
    yield


app = FastAPI(
    title="Prediction API",
    description="REST API for bioprocess model inference.",
    version="1.0.0",
    lifespan=lifespan,
)


def get_model_bundle(request: Request) -> dict | None:
    return getattr(request.app.state, "model_bundle", None)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Map Pydantic validation errors to 400 per the OpenAPI spec."""
    messages = "; ".join(e.get("msg", str(e)) for e in exc.errors())
    return JSONResponse(status_code=400, content={"detail": messages})


@app.get(
    "/health",
    response_model=HealthResponse,
    responses={200: {"description": "Service is healthy"}},
)
def health(bundle: dict | None = Depends(get_model_bundle)) -> HealthResponse:
    if bundle is None or "model" not in bundle:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return HealthResponse(status="ok")


@app.post(
    "/predict",
    response_model=PredictResponse,
    responses={
        400: {"description": "Invalid request payload"},
        500: {"description": "Internal server error"},
    },
)
def predict(
    payload: PredictRequest,
    bundle: dict | None = Depends(get_model_bundle),
) -> PredictResponse:
    if bundle is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        features = extract_features_from_payload(payload.timestamps, payload.values)
        feature_names = bundle["feature_names"]
        feature_vector = np.array([[features[name] for name in feature_names]])
        log_target = bundle["log_target"]

        prediction = bundle["model"].predict(feature_vector)[0]
        if log_target:
            prediction = np.exp(prediction)

        if not math.isfinite(prediction):
            logger.warning("Non-finite prediction: %s", prediction)
            raise HTTPException(status_code=500, detail="Internal prediction error")

        bootstrap_models = bundle.get("bootstrap_models", [])
        if bootstrap_models:
            boot_preds = np.array([m.predict(feature_vector)[0] for m in bootstrap_models])
            if log_target:
                boot_preds = np.exp(boot_preds)
            boot_preds = boot_preds[np.isfinite(boot_preds)]
            if len(boot_preds) == 0:
                ci_lower = float(prediction)
                ci_upper = float(prediction)
            else:
                ci_lower = float(np.percentile(boot_preds, 5))
                ci_upper = float(np.percentile(boot_preds, 95))
        else:
            ci_lower = float(prediction)
            ci_upper = float(prediction)
    except HTTPException:
        raise
    except (ValueError, KeyError):
        logger.exception("Feature extraction failed")
        raise HTTPException(status_code=500, detail="Internal prediction error")
    except Exception:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail="Internal prediction error")

    titer = round(float(prediction), 2)
    logger.info("Predicted titer: %.2f (CI: %.2f - %.2f)", titer, ci_lower, ci_upper)
    return PredictResponse(
        predicted_titer=titer,
        confidence_lower=round(ci_lower, 2),
        confidence_upper=round(ci_upper, 2),
    )
