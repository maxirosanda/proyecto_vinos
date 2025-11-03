# app.py
# === ML API simplificada sin métricas ===
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict
from typing import List, Dict
import joblib, json, time
import numpy as np
from pathlib import Path

APP_VERSION = "0.1.3"
ARTIFACTS_DIR = Path("artifacts")
MODEL_CANDIDATES = [ARTIFACTS_DIR / "model.joblib", ARTIFACTS_DIR / "modelo_vino.joblib"]
COLUMNS_PATH = ARTIFACTS_DIR / "feature_columns.json"

_model = None
_columns: List[str] = []

def _load_artifacts_lazy():
    global _model, _columns
    if _model is None:
        model_path = next((p for p in MODEL_CANDIDATES if p.exists()), None)
        if model_path is None:
            raise HTTPException(status_code=500, detail="Modelo no encontrado en artifacts/.")
        _model = joblib.load(model_path)

        if not COLUMNS_PATH.exists():
            raise HTTPException(status_code=500, detail="feature_columns.json no encontrado.")
        raw = json.loads(COLUMNS_PATH.read_text(encoding="utf-8"))
        if isinstance(raw, list):
            _columns = raw
        elif isinstance(raw, dict) and "feature_columns" in raw:
            _columns = raw["feature_columns"]
        else:
            raise HTTPException(status_code=500, detail="Formato inválido en feature_columns.json")

def _to_vector(sample: Dict[str, float]) -> np.ndarray:
    missing = [c for c in _columns if c not in sample]
    if missing:
        raise HTTPException(status_code=400, detail=f"Faltan columnas requeridas: {missing}")
    extra = [k for k in sample.keys() if k not in _columns]
    if extra:
        raise HTTPException(status_code=400, detail=f"Campos no permitidos: {extra}")
    for k, v in sample.items():
        if not isinstance(v, (int, float)):
            raise HTTPException(status_code=400, detail=f"'{k}' debe ser numérico.")
        if v < 0:
            raise HTTPException(status_code=400, detail=f"'{k}' no puede ser negativo.")
    return np.array([float(sample[c]) for c in _columns], dtype=float)

class Sample(BaseModel):
    model_config = ConfigDict(extra="forbid")
    features: Dict[str, float]

class PredictionOut(BaseModel):
    prediction: str
    latency_ms: float

class BatchOut(BaseModel):
    predictions: List[str]
    latency_ms: float
    count: int

app = FastAPI(title="ML API", version=APP_VERSION)

@app.get("/health")
def health():
    _load_artifacts_lazy()
    return {"status": "ok", "version": APP_VERSION, "n_features": len(_columns)}

@app.post("/predict", response_model=PredictionOut)
def predict(sample: Sample):
    _load_artifacts_lazy()
    try:
        start = time.perf_counter()
        x = _to_vector(sample.features).reshape(1, -1)
        pred = _model.predict(x).tolist()[0]
        latency_ms = round((time.perf_counter() - start) * 1000.0, 3)
        return PredictionOut(prediction=str(pred), latency_ms=latency_ms)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error de predicción: {e}")

@app.post("/predict-batch", response_model=BatchOut)
def predict_batch(samples: List[Sample]):
    _load_artifacts_lazy()
    if not samples:
        raise HTTPException(status_code=400, detail="La lista de muestras está vacía.")
    try:
        start = time.perf_counter()
        X = np.vstack([_to_vector(s.features) for s in samples])
        preds = _model.predict(X).tolist()
        latency_ms = round((time.perf_counter() - start) * 1000.0, 3)
        return BatchOut(
            predictions=[str(p) for p in preds],
            latency_ms=latency_ms,
            count=len(preds)
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error de predicción batch: {e}")
