# joblib: saved model wapas load karne ke liye
import joblib
# pandas: user ka data DataFrame mein badalne ke liye
import pandas as pd
from pathlib import Path
from fastapi import FastAPI

# Step 3 ka function: training mein jo features bane the, API mein bhi wahi banenge
from src.features import add_features
from app.schemas import CustomerData, PredictionResponse

# Model ka path is file ki jagah se nikalte hain (app/main.py -> parent.parent = project root).
# Isse API kahin se bhi chalao (Docker, cloud) path galat nahi hoga
MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "model.joblib"

# Model ko sirf EK baar, API start hote hi load karte hain (har request pe nahi, warna slow hoga)
model = joblib.load(MODEL_PATH)

# Is probability ya usse upar ho toh "churn hoga" bolenge
CHURN_THRESHOLD = 0.5

# FastAPI app banaya. title/description /docs page pe dikhte hain
app = FastAPI(
    title="Customer Churn Prediction API",
    description="Customer ki details bhejo, churn ki probability wapas milegi.",
    version="1.0.0",
)


def get_risk_level(probability):
    """Probability ko Low / Medium / High mein badlo, taaki business team asaani se samjhe."""
    if probability < 0.3:
        return "Low"
    if probability < 0.6:
        return "Medium"
    return "High"


@app.get("/health")
def health():
    """Check karne ke liye ki API zinda hai. Cloud isi ko baar-baar ping karta hai."""
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(customer: CustomerData):
    """Ek customer ka data lo, churn prediction wapas do."""

    # 1. Pydantic object ko dictionary banao, phir 1 row ka DataFrame
    # (list ke andar dict = ek row)
    df = pd.DataFrame([customer.model_dump()])

    # 2. Training jaise hi naye features jodo
    df = add_features(df)

    # 3. Model se churn ki probability nikalo. [0, 1] = pehli row, class 1 (churn) ki probability
    probability = float(model.predict_proba(df)[0, 1])

    # 4. Jawab bana ke bhejo
    return PredictionResponse(
        churn_prediction=probability >= CHURN_THRESHOLD,
        churn_probability=round(probability, 4),
        risk_level=get_risk_level(probability),
    )