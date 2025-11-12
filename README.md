# 🌸 IRIS Classification API - MLOps Pipeline with Auto-Scaling

[![CI/CD Pipeline](https://github.com/YOUR_USERNAME/YOUR_REPO/actions/workflows/cd-deploy-gke.yml/badge.svg)](https://github.com/YOUR_USERNAME/YOUR_REPO/actions)
[![Python 3.9](https://img.shields.io/badge/python-3.9-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104.1-green.svg)](https://fastapi.tiangolo.com/)
[![MLflow](https://img.shields.io/badge/MLflow-2.8.0-orange.svg)](https://mlflow.org/)

A production-ready machine learning pipeline for IRIS flower classification with automated CI/CD, model versioning, and Kubernetes auto-scaling capabilities.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [MLOps Pipeline](#mlops-pipeline)
- [Stress Testing & Auto-Scaling](#stress-testing--auto-scaling)
- [API Documentation](#api-documentation)
- [Monitoring & Observability](#monitoring--observability)
- [Development](#development)
- [Deployment](#deployment)
- [Contributing](#contributing)

---

## 🎯 Overview

This project implements a complete MLOps pipeline for IRIS flower classification, featuring:

- **Model Training**: Decision tree classifier trained on the IRIS dataset
- **Version Control**: DVC for data versioning, MLflow for model registry
- **CI/CD**: Automated testing and deployment via GitHub Actions
- **Cloud Deployment**: Google Kubernetes Engine (GKE) with auto-scaling
- **Monitoring**: Cloud Logging and structured logging for observability
- **Stress Testing**: Automated load testing to validate auto-scaling behavior

### 🎓 Educational Purpose

This project demonstrates:
- ✅ End-to-end MLOps best practices
- ✅ Model versioning and lifecycle management
- ✅ Containerized deployment with Docker
- ✅ Kubernetes orchestration and auto-scaling
- ✅ Production-grade monitoring and logging
- ✅ Automated stress testing and performance validation

---

## ✨ Features

### ML Pipeline
- 🤖 **Scikit-learn Model**: Decision tree classifier for IRIS species prediction
- 📊 **Data Versioning**: DVC tracks datasets in Google Cloud Storage
- 🔄 **Model Registry**: MLflow manages model versions with aliases (dev, stg, prod)
- ✅ **Quality Gates**: Automated performance thresholds (85% dev, 90% prod)

### CI/CD Pipeline
- 🔁 **Multi-branch Strategy**: 
  - `dev` → Train, validate, promote to staging
  - `main` → Production deployment to GKE
- 🐳 **Docker**: Containerized application with MLflow model integration
- 🚀 **Automated Deployment**: GitHub Actions orchestrates entire pipeline
- 🧪 **Smoke Tests**: Post-deployment validation

### Infrastructure
- ☸️ **Kubernetes (GKE)**: Production-grade orchestration
- 📈 **Horizontal Pod Autoscaler**: Auto-scaling from 1 to 3 pods based on CPU (50% threshold)
- 🌐 **Load Balancer**: Exposes API via external IP
- 🔍 **Cloud Logging**: Structured JSON logs for all requests
- 📊 **Cloud Monitoring**: Resource metrics and alerting

### API
- ⚡ **FastAPI**: Modern, fast web framework
- 📝 **Interactive Docs**: Auto-generated Swagger UI at `/docs`
- 🔒 **Input Validation**: Pydantic models ensure data quality
- 🎯 **Health Checks**: Kubernetes-compatible probes

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         GitHub Repository                        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │  dev branch  │  │ main branch  │  │  workflows   │          │
│  │  (Training)  │  │ (Production) │  │  (CI/CD)     │          │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘          │
└─────────┼──────────────────┼──────────────────┼─────────────────┘
          │                  │                  │
          ▼                  ▼                  ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│  CI - Dev       │ │  CD - Deploy    │ │ Stress Test     │
│  - Train model  │ │  - Build image  │ │ - Load testing  │
│  - Evaluate     │ │  - Push to GAR  │ │ - Auto-scaling  │
│  - Promote      │ │  - Deploy GKE   │ │ - Reporting     │
└────────┬────────┘ └────────┬────────┘ └────────┬────────┘
         │                   │                   │
         ▼                   ▼                   ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Google Cloud Platform                       │
│                                                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │   MLflow     │  │     DVC      │  │     GAR      │         │
│  │   Registry   │  │   Storage    │  │   (Images)   │         │
│  └──────────────┘  └──────────────┘  └──────┬───────┘         │
│                                               │                  │
│  ┌────────────────────────────────────────────┼────────────┐   │
│  │            Google Kubernetes Engine (GKE)  │            │   │
│  │                                            ▼            │   │
│  │  ┌──────────────────────────────────────────────────┐  │   │
│  │  │              IRIS API Deployment                 │  │   │
│  │  │  ┌──────┐  ┌──────┐  ┌──────┐                  │  │   │
│  │  │  │ Pod1 │  │ Pod2 │  │ Pod3 │ ◄── HPA (1-3)    │  │   │
│  │  │  └──┬───┘  └──┬───┘  └──┬───┘                  │  │   │
│  │  └─────┼─────────┼─────────┼──────────────────────┘  │   │
│  │        └─────────┴─────────┘                          │   │
│  │                  │                                     │   │
│  │                  ▼                                     │   │
│  │         ┌────────────────┐                            │   │
│  │         │ Load Balancer  │                            │   │
│  │         └────────┬───────┘                            │   │
│  └──────────────────┼────────────────────────────────────┘   │
│                     │                                          │
│  ┌──────────────────┼────────────────────────────────────┐   │
│  │   Observability  ▼                                    │   │
│  │  ┌─────────────────┐  ┌─────────────────┐           │   │
│  │  │  Cloud Logging  │  │ Cloud Monitoring│           │   │
│  │  └─────────────────┘  └─────────────────┘           │   │
│  └───────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────┘
                            │
                            ▼
                    ┌───────────────┐
                    │     Users     │
                    │ (API Clients) │
                    └───────────────┘
```

---

## 🚀 Quick Start

### Prerequisites

- Google Cloud Platform account
- GitHub account with repository
- Docker installed locally
- Python 3.9+

### 1. Clone Repository

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
cd YOUR_REPO
```

### 2. Set Up GCP

```bash
# Set your project ID
export PROJECT_ID="your-gcp-project-id"

# Enable required APIs
gcloud services enable \
  container.googleapis.com \
  logging.googleapis.com \
  monitoring.googleapis.com \
  cloudtrace.googleapis.com \
  artifactregistry.googleapis.com

# Create service account
gcloud iam service-accounts create mlops-deployer \
  --display-name="MLOps Pipeline Deployer"

# Grant permissions
gcloud projects add-iam-policy-binding ${PROJECT_ID} \
  --member="serviceAccount:mlops-deployer@${PROJECT_ID}.iam.gserviceaccount.com" \
  --role="roles/container.admin"

gcloud projects add-iam-policy-binding ${PROJECT_ID} \
  --member="serviceAccount:mlops-deployer@${PROJECT_ID}.iam.gserviceaccount.com" \
  --role="roles/storage.admin"

# Create and download key
gcloud iam service-accounts keys create key.json \
  --iam-account=mlops-deployer@${PROJECT_ID}.iam.gserviceaccount.com
```

### 3. Configure GitHub Secrets

Add these secrets to your GitHub repository (Settings → Secrets and variables → Actions):

```
GCP_PROJECT_ID          # Your GCP project ID
GCP_SA_KEY              # Contents of key.json (entire JSON)
MLFLOW_TRACKING_URI     # Your MLflow server URI
```

### 4. Deploy

```bash
# Push to main branch to trigger deployment
git add .
git commit -m "Initial deployment"
git push origin main

# Monitor deployment
# Go to: GitHub → Actions → Watch the workflow
```

### 5. Test the API

```bash
# Get external IP (wait ~10 minutes for deployment)
kubectl get service iris-api-service

# Test health endpoint
curl http://EXTERNAL_IP/health

# Make a prediction
curl -X POST "http://EXTERNAL_IP/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "sepal_length": 5.1,
    "sepal_width": 3.5,
    "petal_length": 1.4,
    "petal_width": 0.2
  }'

# Response:
# {
#   "prediction": "setosa",
#   "confidence": 0.98,
#   "probabilities": {
#     "setosa": 0.98,
#     "versicolor": 0.01,
#     "virginica": 0.01
#   },
#   "model_version": "iris-classifier",
#   "timestamp": "2024-01-15T10:30:00.123Z",
#   "request_id": "1705318200-1"
# }
```

---

## 📁 Project Structure

```
.
├── .github/
│   └── workflows/
│       ├── cd-deploy-gke.yml        # Production deployment pipeline
│       ├── ci-dev.yml               # Dev branch CI pipeline
│       ├── ci-main.yml              # Main branch CI pipeline
│       └── stress-test.yml          # Auto-scaling stress tests
│
├── kubernetes/
│   ├── deployment.yaml              # Kubernetes deployment manifest
│   ├── service.yaml                 # LoadBalancer service
│   └── hpa.yaml                     # Horizontal Pod Autoscaler (1-3 pods)
│
├── src/
│   ├── data_processing.py           # Data loading and validation
│   ├── model_training.py            # Model training with MLflow
│   └── dvc_operations.py            # DVC operations
│
├── iris-dvc-pipeline/
│   ├── v1_data.csv                  # IRIS dataset (DVC tracked)
│   ├── v1_data.csv.dvc              # DVC metadata
│   └── metrics.txt                  # Model evaluation metrics
│
├── app.py                           # FastAPI application (enhanced logging)
├── main.py                          # ML pipeline orchestrator
├── Dockerfile                       # Container image definition
├── requirements.txt                 # Python dependencies
│
├── post.lua                         # wrk load testing script
├── run_stress_tests.sh              # Interactive stress testing script
│
├── STRESS_TESTING.md                # Comprehensive stress testing guide
├── QUICK_REFERENCE.md               # Command cheat sheet
└── README.md                        # This file
```

---

## 🔄 MLOps Pipeline

### Branch Strategy

```
dev branch          main branch
    │                   │
    ▼                   ▼
┌────────┐         ┌────────┐
│ Train  │         │Validate│
│ Model  │         │ Stg    │
└───┬────┘         │ Model  │
    │              └───┬────┘
    ▼                  │
┌────────┐             │
│Evaluate│             │
│ 85%+   │             │
└───┬────┘             │
    │                  │
    ▼                  ▼
┌────────┐         ┌────────┐
│Promote │         │Promote │
│to @stg │         │to @prod│
└────────┘         └───┬────┘
                       │
                       ▼
                   ┌────────┐
                   │ Deploy │
                   │to GKE  │
                   └────────┘
```

### Model Lifecycle

1. **Development (`dev` branch)**
   - Train model on latest data
   - Evaluate against 85% accuracy threshold
   - Promote to `@stg` alias if successful
   - Automated via `ci-dev.yml`

2. **Staging Validation (`main` PR)**
   - Load model from `@stg` alias
   - Validate against 90% accuracy threshold
   - Generate performance report
   - Automated via `ci-main.yml`

3. **Production (`main` branch)**
   - Promote `@stg` → `@prod` alias
   - Build Docker image with model
   - Deploy to GKE cluster
   - Run smoke tests
   - Automated via `cd-deploy-gke.yml`

### Data Versioning

```bash
# DVC tracks data in Google Cloud Storage
dvc remote add -d gcsremote gs://your-bucket/iris-pipeline
dvc add iris-dvc-pipeline/v1_data.csv
dvc push

# Model versions managed by MLflow
mlflow models serve -m "models:/iris-classifier@prod"
```

---

## 🧪 Stress Testing & Auto-Scaling

### Overview

This project includes comprehensive stress testing to validate Kubernetes auto-scaling behavior and identify performance bottlenecks.

### Configuration

- **Min Pods**: 1
- **Max Pods**: 3
- **CPU Threshold**: 50%
- **Scale Up**: Immediate
- **Scale Down**: 60s stabilization

### Running Stress Tests

#### Option 1: Automated (GitHub Actions)

```bash
# Trigger workflow
GitHub → Actions → "Stress Test - Auto Scaling Validation" → Run workflow
```

#### Option 2: Manual (GCP Cloud Shell)

```bash
# Clone repo and run interactive script
git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
cd YOUR_REPO
chmod +x run_stress_tests.sh
./run_stress_tests.sh
```

### Test Scenarios

| Test | Load | Expected Pods | Purpose |
|------|------|---------------|---------|
| **Test 1** | 1000 req, 50 conn | 1 | Baseline performance |
| **Test 2** | 1000 req, 100 conn | 1→2 | Trigger auto-scaling |
| **Test 3** | 2000 req, 150 conn | 2→3 | Maximum capacity |
| **Test 4** | 2000 req, 150 conn | 1 (forced) | Demonstrate bottleneck |

### Expected Results

```
Test 1 (Baseline):
- Throughput: ~2000 req/sec
- Latency P50: ~25ms
- Latency P99: ~80ms

Test 3 (Scaled to 3 pods):
- Throughput: ~5500 req/sec (2.75x improvement)
- Latency P50: ~20ms
- Latency P99: ~60ms

Test 4 (Bottleneck):
- Throughput: ~1800 req/sec (degraded)
- Latency P50: ~60ms (2.4x worse)
- Latency P99: ~200ms (2.5x worse)
```

### 📊 Observing Auto-Scaling

```bash
# Real-time HPA monitoring
watch -n 2 'kubectl get hpa iris-api-hpa'

# Watch pod scaling
watch -n 2 'kubectl get pods -l app=iris-api'

# Resource usage
kubectl top pods -l app=iris-api

# View scaling events
kubectl describe hpa iris-api-hpa
```

**Detailed Guide**: See [STRESS_TESTING.md](./STRESS_TESTING.md)

---

## 📡 API Documentation

### Base URL

```
http://EXTERNAL_IP
```

### Endpoints

#### GET `/`
Root endpoint with API information

**Response:**
```json
{
  "message": "IRIS Classification API",
  "version": "1.0.0",
  "model_loaded": true,
  "endpoints": {
    "health": "/health",
    "predict": "/predict (POST)",
    "docs": "/docs"
  }
}
```

#### GET `/health`
Health check endpoint for Kubernetes probes

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_info": {
    "source": "mlflow_alias",
    "model_name": "iris-classifier",
    "alias": "champion"
  }
}
```

#### POST `/predict`
Predict IRIS species from features

**Request Body:**
```json
{
  "sepal_length": 5.1,
  "sepal_width": 3.5,
  "petal_length": 1.4,
  "petal_width": 0.2
}
```

**Response:**
```json
{
  "prediction": "setosa",
  "confidence": 0.98,
  "probabilities": {
    "setosa": 0.98,
    "versicolor": 0.01,
    "virginica": 0.01
  },
  "model_version": "iris-classifier",
  "timestamp": "2024-01-15T10:30:00.123Z",
  "request_id": "1705318200-1"
}
```

#### POST `/batch-predict`
Batch prediction (up to 100 samples)

**Request Body:**
```json
[
  {
    "sepal_length": 5.1,
    "sepal_width": 3.5,
    "petal_length": 1.4,
    "petal_width": 0.2
  },
  {
    "sepal_length": 6.3,
    "sepal_width": 2.5,
    "petal_length": 4.9,
    "petal_width": 1.5
  }
]
```

#### GET `/docs`
Interactive Swagger UI documentation

#### GET `/model-info`
Get detailed model information

**Interactive Docs**: Visit `http://EXTERNAL_IP/docs` for full API documentation

---

## 🔍 Monitoring & Observability

### Cloud Logging

#### Access Logs

```
GCP Console → Logging → Logs Explorer
```

#### Query Filter
```
resource.type="k8s_container"
resource.labels.cluster_name="iris-api-cluster"
resource.labels.namespace_name="default"
labels.k8s-pod/app="iris-api"
```

#### Structured Log Fields

```json
{
  "severity": "INFO",
  "message": "Prediction successful",
  "timestamp": "2024-01-15T10:30:00.123Z",
  "component": "iris-api",
  "request_id": "1705318200-1",
  "latency_ms": 45.23,
  "prediction": "setosa",
  "confidence": 0.98,
  "endpoint": "/predict"
}
```

#### CLI Queries

```bash
# Recent logs
gcloud logging read \
  "resource.type=k8s_container AND resource.labels.cluster_name=iris-api-cluster" \
  --limit 50 \
  --format json

# Error logs only
gcloud logging read \
  "resource.type=k8s_container AND severity>=ERROR" \
  --limit 50

# High latency requests
gcloud logging read \
  "jsonPayload.latency_ms>100" \
  --limit 20
```

### Cloud Monitoring

```
GCP Console → Monitoring → Metrics Explorer
```

**Key Metrics:**
- CPU utilization per pod
- Memory usage
- Request count
- Request latency (from logs)
- Error rate

### Kubernetes Metrics

```bash
# Pod metrics
kubectl top pods -l app=iris-api

# Node metrics
kubectl top nodes

# HPA status
kubectl get hpa iris-api-hpa

# Events
kubectl get events --sort-by='.lastTimestamp'
```

---

## 💻 Development

### Local Development

#### 1. Install Dependencies

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

#### 2. Train Model Locally

```bash
python main.py \
  --data-path iris-dvc-pipeline/v1_data.csv \
  --singleparameter-tuning \
  --metrics-path iris-dvc-pipeline/metrics.txt
```

#### 3. Run API Locally

```bash
python app.py

# Or with uvicorn
uvicorn app:app --reload --port 8080
```

#### 4. Test Locally

```bash
# Health check
curl http://localhost:8080/health

# Prediction
curl -X POST "http://localhost:8080/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "sepal_length": 5.1,
    "sepal_width": 3.5,
    "petal_length": 1.4,
    "petal_width": 0.2
  }'
```

### Docker Development

```bash
# Build image
docker build -t iris-classifier:local .

# Run container
docker run -p 8080:8080 iris-classifier:local

# Test
curl http://localhost:8080/health
```

### Testing Changes

```bash
# Create feature branch
git checkout -b feature/my-feature

# Make changes and test locally

# Push to dev branch for CI
git checkout dev
git merge feature/my-feature
git push origin dev

# Monitor GitHub Actions for results
```

---

## 🚀 Deployment

### Prerequisites

- GCP project with billing enabled
- GitHub repository with secrets configured
- MLflow server running (for model registry)

### Deployment Process

1. **Code Changes** → Push to `main`
2. **CI/CD Triggers** → GitHub Actions workflow starts
3. **Build Phase**:
   - Pull MLflow model artifacts
   - Build Docker image
   - Push to Artifact Registry
   - Run image tests
4. **Deploy Phase**:
   - Create/verify GKE cluster
   - Deploy Kubernetes resources
   - Wait for rollout
   - Run smoke tests
5. **Validation**:
   - Get external IP
   - Test endpoints
   - Generate deployment report

### Manual Deployment

```bash
# Connect to cluster
gcloud container clusters get-credentials iris-api-cluster \
  --zone=us-central1-a \
  --project=${PROJECT_ID}

# Apply manifests
kubectl apply -f kubernetes/deployment.yaml
kubectl apply -f kubernetes/service.yaml
kubectl apply -f kubernetes/hpa.yaml

# Check status
kubectl get all -l app=iris-api

# Get external IP
kubectl get service iris-api-service
```

### Rollback

```bash
# View rollout history
kubectl rollout history deployment iris-api

# Rollback to previous version
kubectl rollout undo deployment iris-api

# Rollback to specific revision
kubectl rollout undo deployment iris-api --to-revision=2
```

---

## 🧹 Cleanup

### Delete Deployment (Keep Cluster)

```bash
kubectl delete -f kubernetes/deployment.yaml
kubectl delete -f kubernetes/service.yaml
kubectl delete -f kubernetes/hpa.yaml
```

### Delete Entire Cluster

```bash
gcloud container clusters delete iris-api-cluster \
  --zone=us-central1-a \
  --project=${PROJECT_ID} \
  --quiet
```

### Delete Artifact Registry Images

```bash
gcloud artifacts docker images list \
  us-central1-docker.pkg.dev/${PROJECT_ID}/iris-api

gcloud artifacts docker images delete \
  us-central1-docker.pkg.dev/${PROJECT_ID}/iris-api/iris-classifier:TAG \
  --quiet
```

---

## 🤝 Contributing

### Development Workflow

1. Fork the repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open Pull Request

### Code Style

- Follow PEP 8 for Python code
- Use type hints where appropriate
- Add docstrings to all functions
- Write unit tests for new features

### Pull Request Process

1. Update README.md with details of changes
2. Update requirements.txt if dependencies added
3. Ensure all tests pass
4. Request review from maintainers

---

## 📚 Additional Resources

- **Detailed Guides**:
  - [Stress Testing Guide](./STRESS_TESTING.md)
  - [Quick Reference](./QUICK_REFERENCE.md)

- **External Documentation**:
  - [FastAPI Documentation](https://fastapi.tiangolo.com/)
  - [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
  - [Kubernetes HPA](https://kubernetes.io/docs/tasks/run-application/horizontal-pod-autoscale/)
  - [GKE Documentation](https://cloud.google.com/kubernetes-engine/docs)
  - [wrk Documentation](https://github.com/wg/wrk)

---

## 📊 Performance Benchmarks

### API Performance

| Metric | Value |
|--------|-------|
| Response Time (P50) | ~20ms |
| Response Time (P99) | ~80ms |
| Throughput (1 pod) | ~2000 req/sec |
| Throughput (3 pods) | ~5500 req/sec |
| Cold Start Time | ~15s |

### Resource Usage

| Component | CPU | Memory | Disk |
|-----------|-----|--------|------|
| API Pod | 250m (request) | 256Mi (request) | - |
| API Pod | 500m (limit) | 512Mi (limit) | - |
| Model Size | - | ~50MB | 50MB |

---

## 🐛 Troubleshooting

### Common Issues

#### External IP Pending
```bash
# Check service
kubectl describe service iris-api-service

# If stuck, recreate
kubectl delete service iris-api-service
kubectl apply -f kubernetes/service.yaml
```

#### HPA Not Scaling
```bash
# Check metrics server
kubectl get deployment metrics-server -n kube-system

# Check HPA conditions
kubectl describe hpa iris-api-hpa

# Verify resource requests are set
kubectl get deployment iris-api -o yaml | grep -A 5 resources
```

#### Pod CrashLoopBackOff
```bash
# Check pod logs
kubectl logs -l app=iris-api --tail=100

# Check pod events
kubectl describe pods -l app=iris-api

# Common causes:
# - Model not loaded (check MLflow connection)
# - Port already in use
# - Resource limits too low
```

#### Model Not Loading
```bash
# Check MLflow environment variables
kubectl get deployment iris-api -o yaml | grep -A 3 env

# Check if fallback model is being used
kubectl logs -l app=iris-api | grep "fallback"

# Verify MLflow registry
curl -X GET "${MLFLOW_TRACKING_URI}/api/2.0/mlflow/registered-models/get?name=iris-classifier"
```

</div>
