
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

