# Churn Prediction API

End-to-end ML project: Data -> Cleaning -> Features -> Model -> FastAPI -> Docker -> Cloud.

## Setup (Windows)
1. python -m venv venv
2. venv\Scripts\activate
3. pip install -r requirements.txt
4. Kaggle se "Telco Customer Churn" dataset download karo
   aur data/raw/telco_churn.csv naam se rakho.
5. python src/01_explore_data.py


Customer Churn Prediction API
An end-to-end machine learning project: a churn prediction model served through a FastAPI service, packaged with Docker, and deployed to the cloud (Render).
Live demo (interactive docs): https://churn-prediction-api-mml3.onrender.com/docs
> The demo runs on a free tier and sleeps after 15 minutes of inactivity. The first request after a pause can take about a minute to wake it up. Open `/health` first, then `/docs`.
---
What it does
Send the details of a telecom customer and get back:
whether the customer is likely to churn (`true` / `false`)
the churn probability (0 to 1)
a risk level (`Low`, `Medium`, `High`)
Example response:
```json
{
  "churn_prediction": true,
  "churn_probability": 0.8475,
  "risk_level": "High"
}
```
Architecture
```
Dataset (Telco Customer Churn, 7,043 customers)
   |
   v
Data cleaning        src/clean_data.py
   |
   v
Feature engineering  src/features.py
   |
   v
Model training       src/train.py  ->  models/model.joblib
   |
   v
FastAPI service      app/main.py
   |
   v
Docker image         Dockerfile
   |
   v
Cloud deployment     Render (from this GitHub repo)
```
Model and results
Data: the IBM Telco Customer Churn dataset (7,043 customers, 21 columns, about 26.5% churn).
Cleaning: `TotalCharges` is stored as text with 11 blank values (all brand-new customers with `tenure = 0`), so they are converted to numbers and set to 0. `customerID` is dropped. The target is encoded as 0/1.
Features added: `tenure_group`, `num_addon_services`, `is_month_to_month`, `is_auto_pay`, `avg_monthly_charge`.
Preprocessing: `StandardScaler` for numeric columns and `OneHotEncoder(handle_unknown="ignore")` for categorical columns, both inside a single scikit-learn `Pipeline` that is saved together with the model.
Class imbalance: handled with `class_weight="balanced"`.
Models compared: Logistic Regression and Random Forest, selected by ROC-AUC on a stratified 80/20 split.
Results on the held-out test set (1,409 customers), Random Forest:
Metric	Value
ROC-AUC	0.842
Recall (churners caught)	0.797
Precision	0.530
F1	0.637
Accuracy	0.759
A model that always predicts "no churn" reaches 73.5% accuracy but catches zero churners, so accuracy alone is misleading here. Recall and ROC-AUC are the metrics that matter. Logistic Regression scored almost the same ROC-AUC (0.842). Exact numbers can vary slightly with library versions.
API endpoints
Method	Path	Description
GET	`/health`	Health check
POST	`/predict`	Predict churn for one customer
GET	`/docs`	Interactive Swagger UI
Inputs are validated with Pydantic. Invalid values (for example an unknown `Contract` type or a non-numeric `tenure`) return a `422` error before reaching the model.
Example request body:
```json
{
  "gender": "Female",
  "SeniorCitizen": 0,
  "Partner": "No",
  "Dependents": "No",
  "tenure": 2,
  "PhoneService": "Yes",
  "MultipleLines": "No",
  "InternetService": "Fiber optic",
  "OnlineSecurity": "No",
  "OnlineBackup": "No",
  "DeviceProtection": "No",
  "TechSupport": "No",
  "StreamingTV": "No",
  "StreamingMovies": "No",
  "Contract": "Month-to-month",
  "PaperlessBilling": "Yes",
  "PaymentMethod": "Electronic check",
  "MonthlyCharges": 70.7,
  "TotalCharges": 151.65
}
```
Project structure
```
churn-prediction-api/
├── app/
│   ├── main.py            # FastAPI app (/health, /predict)
│   └── schemas.py         # Pydantic request/response models
├── src/
│   ├── 01_explore_data.py # Data exploration
│   ├── clean_data.py      # Cleaning
│   ├── features.py        # Feature engineering (shared by training and the API)
│   └── train.py           # Training, evaluation, model saving
├── models/
│   └── model.joblib       # Trained scikit-learn pipeline
├── Dockerfile
├── requirements.txt       # Pinned versions
└── README.md
```
The raw dataset is not included in the repository. To retrain, download "Telco Customer Churn" from Kaggle and save it as `data/raw/telco_churn.csv`.
Run locally
```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

# Optional: retrain (needs the dataset in data/raw/telco_churn.csv)
python src/clean_data.py
python src/train.py

# Start the API
uvicorn app.main:app --reload
```
Then open http://127.0.0.1:8000/docs
Run with Docker
```powershell
docker build -t churn-api .
docker run -p 8000:8000 churn-api
```
Then open http://localhost:8000/docs
Design decisions
One feature function for training and serving. `add_features()` is used by both `train.py` and the API, which avoids training-serving skew.
Preprocessing lives inside the saved pipeline. The API passes raw customer fields straight to the model.
Pinned dependency versions. The model is loaded with the same scikit-learn version it was trained with.
Model loaded once at startup, not on every request.
Port from environment. The container listens on `$PORT` if set, otherwise 8000, so it runs the same locally and on Render.
Limitations and future improvements
No authentication yet. Adding an API key would be the first step for real use.
The decision threshold is fixed at 0.5. It could be tuned to the business cost of missing a churner versus offering an unnecessary discount.
No automated tests or CI pipeline.
No monitoring or drift detection.
Trained on a single public sample dataset, so it is a demonstration of the full workflow rather than a production model.
Deployment is manual (no auto-deploy on push).
Tech stack
Python, pandas, NumPy, scikit-learn, joblib, FastAPI, Uvicorn, Pydantic, Docker, GitHub, Render
Dataset
IBM Telco Customer Churn sample dataset, available on Kaggle.
