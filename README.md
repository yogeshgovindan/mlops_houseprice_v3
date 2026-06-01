# 🏠 House Price Prediction MLOps System (V3)

An end-to-end production-style MLOps system for house price prediction using Machine Learning, FastAPI, MLflow, Docker, Azure, Drift Detection, and Automatic Retraining.

---

## 🚀 Project Highlights

This project demonstrates a **real-world MLOps lifecycle**, including:

✅ Data ingestion pipeline
✅ Feature engineering pipeline
✅ Model training pipeline
✅ MLflow experiment tracking
✅ Model versioning
✅ FastAPI prediction API
✅ Swagger documentation
✅ Docker containerization
✅ Azure deployment
✅ Prediction monitoring
✅ Drift detection
✅ Automatic retraining trigger
✅ Self-healing ML system
✅ Monitoring dashboard using Streamlit
✅ Retraining history tracking

---

````markdown id="a6"
## 🏗️ System Architecture


flowchart TD

A[Housing Dataset] --> B[Data Ingestion]

B --> C[Feature Engineering Pipeline]

C --> D[Model Training]

D --> E[MLflow Tracking]

D --> F[Model Versioning]

F --> G[FastAPI Prediction API]

G --> H[Prediction Logging]

H --> I[Drift Detection]

I --> J{Drift High?}

J -- Yes --> K[Auto Retraining]

K --> D

J -- No --> L[Continue Serving]

G --> M[Streamlit Monitoring Dashboard]

M --> N[Prediction Analytics]

M --> O[Retraining History]

M --> P[Drift Monitoring]
```



---

## 🛠️ Tech Stack

### Machine Learning

* Python
* Scikit-learn
* Random Forest Regressor

### MLOps

* MLflow
* FastAPI
* Docker
* GitHub Actions
* Azure App Service
* Azure Container Registry (ACR)

### Monitoring

* Streamlit
* Drift Detection
* Prediction Logging
* Auto Retraining

---

## 📂 Project Structure

```text
mlops_houseprice_v3/

├── app.py
├── dashboard/
│   └── app.py
│
├── artifacts/
│   ├── model.pkl
│   ├── model_v1.pkl
│   └── model_v2.pkl
│
├── data/
│   └── housing.csv
│
├── logs/
│   ├── predictions.json
│   └── retraining_logs.json
│
├── src/
│   ├── features/
│   ├── monitoring/
│   ├── pipelines/
│   ├── model_registry/
│   └── utils/
│
└── README.md
```

---

## 📈 Features

### 1. FastAPI Prediction API

Real-time house price prediction using FastAPI.

### 2. Drift Detection

Automatically monitors feature drift between training and live data.

### 3. Self-Healing ML System

If drift exceeds threshold:

```text
Auto retraining triggers
```

New model version gets created automatically.

### 4. Monitoring Dashboard

Real-time monitoring dashboard with:

* Prediction statistics
* Drift status
* Prediction charts
* Retraining history
* Model version tracking

---

## ▶️ Run Locally

### Create virtual environment

```bash
python -m venv venv
```

### Activate environment

```bash
venv\Scripts\activate
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run API

```bash
uvicorn app:app --reload
```

### Run Dashboard

```bash
streamlit run dashboard/app.py
```

---

## 📌 Future Enhancements

* Champion–Challenger model system
* Scheduled retraining
* Model rollback
* Advanced drift metrics
* Kubernetes deployment

---

## 👨‍💻 Author

**Yogesh Govindan**
Data Scientist | MLOps | Data Scientist
