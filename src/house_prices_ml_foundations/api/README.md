# Try FastAPI

## Get /health :
```bash
curl "URL":"PORT"/health`
```
## Post /predict :
```bash
curl -X POST "URL":"PORT"/predict \
-H "Content-Type: application/json" \
--data @src/house_prices_ml_foundations/api/examples/predict_payload.json
```