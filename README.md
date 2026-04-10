# House Prices — ML Foundations

Price regression of housing sales (Kaggle House Prices).  
Stack : Python 3.12 · scikit-learn · FastAPI · MLflow · Docker.

---

## Prerequisites

- Docker + Docker Compose
- **or** Python 3.10+

---

## Launch in 5 minutes (Docker)

```bash
make up
```

| Service   | URL                        |
| --------- | -------------------------- |
| API docs  | http://localhost:8000/docs |
| MLflow UI | http://localhost:5000      |

Check that the containers are `healthy` with:

```bash
docker compose ps
```

Stop the containers with:

```bash
make down
```

---

## Launch locally (without Docker)

```bash
make setup          # create the venv and install dependencies
make test           # run pytest
make lint           # ruff check + format
```

```bash
source .venv/bin/activate
```

for Linux/MacOS or

```bash
.venv\Scripts\activate
```

for Windows.

---

## Example API call

```bash
curl -X POST http://localhost:8000/predict \
-H "Content-Type: application/json" \
--data @src/house_prices_ml_foundations/api/examples/predict_payload.json
```

Expected response (JSON):

```json
{"predictions": [values...]}
{"predictions": [values...]}
```

---

## Structure

```
scripts/   # ML pipeline (00 → 14)
src/       # Python package (features, models, api, mlops)
outputs/   # models, reports, figures, mlruns
datasets/  # raw Kaggle data
```
