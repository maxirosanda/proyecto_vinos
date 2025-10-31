
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field
from typing import List, Dict, Any
import joblib, json, time
import numpy as np
from pathlib import Path

APP_VERSION = "0.1.0"
ARTIFACTS_DIR = Path("artifacts")
# Admitimos ambos nombres por si guardaste con "modelo_vino.joblib"
MODEL_CANDIDATES = [ARTIFACTS_DIR / "model.joblib", ARTIFACTS_DIR / "modelo_vino.joblib"]
COLUMNS_PATH = ARTIFACTS_DIR / "feature_columns.json"
METRICS_PATH = ARTIFACTS_DIR / "metrics.json"  # opcional

# Carga perezosa (on-demand)
_model = None
_columns: List[str] = []
_metrics: Dict[str, Any] = {}

def _load_artifacts_lazy():
    global _model, _columns, _metrics
    if _model is None:
        # modelo
        model_path = next((p for p in MODEL_CANDIDATES if p.exists()), None)
        if model_path is None:
            raise HTTPException(status_code=500, detail="Modelo no encontrado en artifacts/. Entrena y exporta primero.")
        try:
            _model = joblib.load(model_path)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error cargando modelo: {e}")

        # columnas
        if not COLUMNS_PATH.exists():
            raise HTTPException(status_code=500, detail="feature_columns.json no encontrado en artifacts/.")
        try:
            raw = json.loads(COLUMNS_PATH.read_text(encoding="utf-8"))
            # Soporta formato lista o dict {"feature_columns": [...]}
            if isinstance(raw, list):
                _columns = raw
            elif isinstance(raw, dict) and "feature_columns" in raw and isinstance(raw["feature_columns"], list):
                _columns = raw["feature_columns"]
            else:
                raise ValueError("Formato inválido en feature_columns.json")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error leyendo columnas: {e}")

        # métricas (opcional)
        if METRICS_PATH.exists():
            try:
                _metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
            except Exception:
                _metrics = {}

def _to_vector(sample: Dict[str, float]) -> np.ndarray:
    """Valida claves/tipos y respeta el orden de _columns."""
    missing = [c for c in _columns if c not in sample]
    if missing:
        raise HTTPException(status_code=400, detail=f"Faltan columnas requeridas: {missing}")
    extra = [k for k in sample.keys() if k not in _columns]
    if extra:
        raise HTTPException(status_code=400, detail=f"Campos no permitidos: {extra}")
    try:
        # Validación numérica básica + no-negatividad (ajustable)
        for k, v in sample.items():
            if not isinstance(v, (int, float)):
                raise TypeError(f"'{k}' debe ser numérico (int/float).")
            if v < 0:
                raise ValueError(f"'{k}' no puede ser negativo.")
        return np.array([float(sample[c]) for c in _columns], dtype=float)
    except (TypeError, ValueError) as e:
        raise HTTPException(status_code=400, detail=str(e))

# ===== Esquemas Pydantic (v2) =====
class Sample(BaseModel):
    """Un registro con todas las features como dict[str,float]."""
    model_config = ConfigDict(extra="forbid")

    features: Dict[str, float] = Field(
        ...,
        description="Mapa feature->valor numérico",
        example={
            "fixed acidity": 7.4,
            "volatile acidity": 0.70,
            "citric acid": 0.00,
            "residual sugar": 1.9,
            "chlorides": 0.076,
            "total sulfur dioxide": 34.0,
            "density": 0.9978,
            "pH": 3.51,
            "sulphates": 0.56,
            "alcohol": 9.4
        }
    )


class PredictionOut(BaseModel):
    prediction: int | float
    proba: List[float] | None = None
    latency_ms: float
    warnings: List[str] = []

class BatchOut(BaseModel):
    predictions: List[int | float]
    proba: List[List[float]] | None = None
    latency_ms: float
    count: int
    warnings: List[str] = []

app = FastAPI(title="ML API", version=APP_VERSION)

@app.get("/health")
def health():
    _load_artifacts_lazy()
    return {
        "status": "ok",
        "version": APP_VERSION,
        "n_features": len(_columns),
        "metrics": _metrics or {"note": "sin metrics.json"},
    }

@app.post("/predict", response_model=PredictionOut)
def predict(sample: Sample):
    _load_artifacts_lazy()
    try:
        start = time.perf_counter()
        x = _to_vector(sample.features).reshape(1, -1)
        pred = _model.predict(x).tolist()[0]
        proba = _model.predict_proba(x).tolist()[0] if hasattr(_model, "predict_proba") else None
        latency_ms = round((time.perf_counter() - start) * 1000.0, 3)
        return PredictionOut(prediction=pred, proba=proba, latency_ms=latency_ms, warnings=[])
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
        proba = _model.predict_proba(X).tolist() if hasattr(_model, "predict_proba") else None
        latency_ms = round((time.perf_counter() - start) * 1000.0, 3)
        return BatchOut(predictions=preds, proba=proba, latency_ms=latency_ms, count=len(preds), warnings=[])
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error de predicción batch: {e}")
