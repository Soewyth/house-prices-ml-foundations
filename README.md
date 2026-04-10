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
```

---

## Structure

```
scripts/   # ML pipeline (00 → 14)
src/       # Python package (features, models, api, mlops)
outputs/   # models, reports, figures, mlruns
datasets/  # raw Kaggle data
```

---

## Full setup from scratch

Datasets and model artifacts are not committed (Kaggle data + generated files).  
Follow these steps to reproduce everything from zero.

**1. Clone & install**
```bash
git clone <repo-url> && cd house-prices-ml-foundations
make setup
source .venv/bin/activate
```

**2. Download Kaggle data**  
Download [House Prices](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data) and place the files in:
```
datasets/raw/train.csv
datasets/raw/test.csv
datasets/raw/data_description.txt
```

**3. Start the stack (MLflow must be running before training scripts)**
```bash
make up
```

**4. Run the ML pipeline**
```bash
python scripts/00_check_data.py
python scripts/01_baseline_model.py
python scripts/02_model_comparison.py
python scripts/03_tune_rf.py
python scripts/04_rf_final_holdout.py
python scripts/05_make_submission_rf.py
python scripts/06_generate_report.py
python scripts/12_error_analysis.py
python scripts/13_error_analysis_plots.py
python scripts/10_train_champion_and_export.py   # trains + exports champion + logs to MLflow
```

**5. Check results**
- MLflow runs: http://localhost:5000
- API docs: http://localhost:8000/docs

---
## WARNING 
Use the `make up` command to start the stack and not `docker compose up` directly. Make up ensure to inject the UID and GID of the current user in the MLflow container to avoid permission issues when MLflow creates files (models, artifacts) on the host machine. If you start the stack with `docker compose up` directly, you might encounter permission issues with MLflow artifacts created on the host machine.
## Proof / Repro

| Check     | Command                                                                                                                                                    | Expected                 |
| --------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------ |
| Tests     | `make test`                                                                                                                                                | 8 passed                 |
| Stack up  | `make up`                                                                                                                                                  | API + MLflow running     |
| Healthy   | `docker compose ps`                                                                                                                                        | both services healthy    |
| MLflow UI | http://localhost:5000                                                                                                                                      | runs visible             |
| API docs  | http://localhost:8000/docs                                                                                                                                 | Swagger UI               |
| Predict   | `curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" --data @src/house_prices_ml_foundations/api/examples/predict_payload.json` | `{"predictions": [...]}` |
