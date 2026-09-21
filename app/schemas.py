# Literal: field mein sirf ginti ki fixed values allowed karne ke liye
from typing import Literal
# BaseModel: pydantic ka base class, Field: extra rules (min/max) lagane ke liye
from pydantic import BaseModel, Field


class CustomerData(BaseModel):
    """Ek customer ki details, jo API ko input mein chahiye."""

    gender: Literal["Male", "Female"]
    SeniorCitizen: Literal[0, 1]                 # 1 = senior citizen, 0 = nahi
    Partner: Literal["Yes", "No"]
    Dependents: Literal["Yes", "No"]
    tenure: int = Field(ge=0, le=72)             # kitne mahine se customer hai (0 se 72)
    PhoneService: Literal["Yes", "No"]
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: Literal["Yes", "No", "No internet service"]
    OnlineBackup: Literal["Yes", "No", "No internet service"]
    DeviceProtection: Literal["Yes", "No", "No internet service"]
    TechSupport: Literal["Yes", "No", "No internet service"]
    StreamingTV: Literal["Yes", "No", "No internet service"]
    StreamingMovies: Literal["Yes", "No", "No internet service"]
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: Literal["Yes", "No"]
    PaymentMethod: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ]
    MonthlyCharges: float = Field(ge=0)          # negative nahi ho sakta
    TotalCharges: float = Field(ge=0)            # naye customer ke liye 0

    # /docs page pe ye example pehle se bhara hua dikhega
    model_config = {
        "json_schema_extra": {
            "example": {
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
                "TotalCharges": 151.65,
            }
        }
    }


class PredictionResponse(BaseModel):
    """API jo jawab wapas degi, uska format."""

    churn_prediction: bool        # True = customer jaane wala hai
    churn_probability: float      # 0 se 1 ke beech
    risk_level: Literal["Low", "Medium", "High"]