# Try FastAPI

## Get /health :
```bash
curl "URL":"PORT"/health`
```
## Post /predict :
```bash
curl -X POST http://localhost:8000/predict \
-H "Content-Type: application/json" \
--data @src/house_prices_ml_foundations/api/examples/predict_payload.json
```