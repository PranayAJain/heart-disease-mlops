# Heart Disease Risk Prediction - End-to-End MLOps Pipeline

**MLOps (AIMLCZG523) - Assignment 1** | Pranay Jain (2025AE05025)

A complete, reproducible MLOps pipeline for predicting heart-disease risk from 13 clinical
features (UCI Heart Disease, Cleveland subset): data acquisition, EDA, model training with
experiment tracking, packaging, CI/CD, containerised serving, Kubernetes deployment and monitoring.

```
UCI data -> clean + EDA -> sklearn Pipeline (impute/scale/one-hot + hr_pct_max)
         -> GridSearchCV x 3 models (LogReg, RF, XGBoost) -> MLflow tracking + registry
         -> packaged model (joblib / MLflow) -> FastAPI -> Docker -> Kubernetes (minikube)
         -> Prometheus metrics + Grafana dashboard + JSON request logs
GitHub Actions: lint -> 24 unit tests -> train -> package -> quality gate -> artifacts
```

## Results

| Model | CV ROC-AUC | CV Accuracy | CV Recall | Test ROC-AUC |
|---|---|---|---|---|
| **Logistic Regression (selected)** | **0.908 +/- 0.019** | 0.847 | 0.783 | 0.960 |
| Random Forest | 0.900 +/- 0.023 | 0.818 | 0.764 | 0.953 |
| XGBoost | 0.889 +/- 0.020 | 0.818 | 0.765 | 0.959 |

5-fold stratified CV on the 80 % training split; selection by CV ROC-AUC (not test score).

## Repository layout

| Path | Purpose |
|---|---|
| `scripts/download_data.py` | Downloads the UCI dataset into `data/raw/` |
| `src/heart/data.py` | Cleaning: `?` -> NaN, binary target, dtypes, duplicates, range checks |
| `src/heart/features.py` | Preprocessing pipeline + engineered feature `hr_pct_max` |
| `src/heart/train.py` | GridSearchCV + cross-validation for 3 models |
| `src/heart/tracking.py` | MLflow logging (params, metrics, plots, model, nested runs) |
| `src/heart/package.py` | Final model -> joblib + MLflow format + metadata + registry |
| `src/heart/predict.py` | Inference helper shared by API, tests and scripts |
| `src/heart/api.py` | FastAPI service (`/predict`, `/health`, `/metrics`, ...) |
| `notebooks/01_eda.ipynb` | Exploratory data analysis (figures in `reports/figures/`) |
| `tests/` | 24 pytest tests (data, features, model, API) |
| `.github/workflows/ci.yml` | CI/CD pipeline |
| `Dockerfile`, `requirements-serve.txt` | Serving image |
| `k8s/` | Kubernetes Deployment (2 replicas, probes) + NodePort Service |
| `monitoring/` | Prometheus config + Grafana datasource/dashboard provisioning |
| `screenshots/` | Evidence for every task |

## 1. Setup

```bash
git clone https://github.com/PranayAJain/heart-disease-mlops.git
cd heart-disease-mlops
conda create -y -n heart-mlops python=3.11 && conda activate heart-mlops
pip install -r requirements.txt
pip install -e .
```

## 2. Data, EDA, training, packaging

```bash
python scripts/download_data.py          # UCI download -> data/raw/
python -m heart.data                     # clean -> data/processed/heart_clean.csv
jupyter notebook notebooks/01_eda.ipynb  # EDA (Run All)
python -m heart.train                    # tune + compare 3 models, log to MLflow
python -m heart.package                  # final model -> models/
python scripts/verify_model.py           # reload + reproducibility check
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000   # http://127.0.0.1:5000
```

## 3. Tests and lint

```bash
flake8 src scripts tests
pytest -v                                # 24 tests
```

## 4. Docker

```bash
docker build -t heart-api:1.0 .
docker run -d --name heart-api -p 8000:8000 heart-api:1.0
curl localhost:8000/health
curl -X POST localhost:8000/predict -H "Content-Type: application/json" \
     -d @samples/patient_high_risk.json
# -> {"prediction":1,"label":"heart disease","probability_disease":0.996,"confidence":0.996,...}
```

Interactive API docs: http://localhost:8000/docs

## 5. Kubernetes (minikube)

```bash
minikube start --driver=docker
alias kubectl="minikube kubectl --"
minikube image load heart-api:1.0
kubectl apply -f k8s/
kubectl rollout status deployment/heart-api
URL=$(minikube service heart-api --url)
curl -X POST $URL/predict -H "Content-Type: application/json" -d @samples/patient_low_risk.json
```

## 6. Monitoring (Prometheus + Grafana)

```bash
docker network create monitoring
docker network connect monitoring heart-api
docker create --name prometheus --network monitoring -p 9090:9090 prom/prometheus:v2.55.1
docker cp monitoring/prometheus.yml prometheus:/etc/prometheus/prometheus.yml
docker start prometheus
docker create --name grafana --network monitoring -p 3000:3000 \
  -e GF_SECURITY_ADMIN_PASSWORD=admin -e GF_AUTH_ANONYMOUS_ENABLED=true \
  -e GF_AUTH_ANONYMOUS_ORG_ROLE=Viewer grafana/grafana:11.3.0
docker cp monitoring/grafana/provisioning/. grafana:/etc/grafana/provisioning/
docker start grafana
./scripts/generate_traffic.sh http://localhost:8000 300 &
```

- Prometheus: http://localhost:9090 (target `heart-api` scraped every 5 s)
- Grafana: http://localhost:3000 -> Dashboards -> MLOps -> *Heart Disease API - Monitoring*
- Request logs: `docker logs heart-api` (one JSON line per request / prediction)

## API

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Liveness / readiness, model version |
| GET | `/model-info` | Model kind, params, CV and test metrics |
| POST | `/predict` | One patient -> prediction, probability, confidence |
| POST | `/predict/batch` | Up to 100 patients |
| GET | `/metrics` | Prometheus metrics |
| GET | `/docs` | Swagger UI |

Input fields are validated (types, allowed codes, physiological ranges); invalid input returns
HTTP 422. `ca` and `thal` are optional and imputed by the pipeline when missing.
