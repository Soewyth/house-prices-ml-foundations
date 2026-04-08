from __future__ import annotations

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from house_prices_ml_foundations.config.paths import get_paths, get_project_root
from house_prices_ml_foundations.features.build import make_features

app = FastAPI()

# Load model at startup

root_dir = get_project_root()
paths = get_paths(root_dir=root_dir)
model_path = paths["models"] / "champion.joblib"


try:
    model = joblib.load(model_path)
    model_loaded = True # flag
except Exception as e:
    print(f"Error loading model : {e}")
    model_loaded = False # flag
    model = None # Placeholder to avoid NameError


class PredictRequest(BaseModel):
    records: list[dict]

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded" : model_loaded}
    
@app.post("/predict")
def predict(request: PredictRequest):
    if not model_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
        
    try:
        df = pd.DataFrame(request.records)
        X = make_features(df=df, return_target=False)
        preds = model.predict(X)
        return {"predictions" : preds.tolist()}
    except KeyError as e:
        raise HTTPException(status_code=400, detail=f"Missing required column: {str(e)}")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid value: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")
        


















# Retour JSON = { "predictions": [..] }
# Indices “pro”
# Charger le modèle au démarrage (lazy-load ok mais pas à chaque requête)
# Chemin modèle : outputs/models/champion.joblib via get_paths(get_project_root())
# Gérer erreurs : si colonnes manquantes → HTTP 400 avec message clair

# ✅ Tu me renvoies :

# la sortie uvicorn ... qui démarre
# un curl (ou httpie) sur /health
# un curl sur /predict (2 lignes de test)