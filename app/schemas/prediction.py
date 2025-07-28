# app/schemas/prediction.py
from pydantic import BaseModel, Field
from typing import Optional, Dict

# Online Payment Schema
class OnlinePaymentInput(BaseModel):
    step: int = Field(..., description="Transaction step/time")
    type: str = Field(..., description="Transaction type (PAYMENT, TRANSFER, CASH_OUT, DEBIT, CASH_IN)")
    amount: float = Field(..., description="Transaction amount")
    oldbalanceOrg: float = Field(..., description="Sender's balance before transaction")
    newbalanceOrig: float = Field(..., description="Sender's balance after transaction")
    oldbalanceDest: float = Field(..., description="Receiver's balance before transaction")
    newbalanceDest: float = Field(..., description="Receiver's balance after transaction")
    
    class Config:
        json_schema_extra = {
            "example": {
                "step": 1,
                "type": "PAYMENT",
                "amount": 9839.64,
                "oldbalanceOrg": 170136.0,
                "newbalanceOrig": 160296.36,
                "oldbalanceDest": 0.0,
                "newbalanceDest": 0.0
            }
        }

# Credit Card Schema
class CreditCardInput(BaseModel):
    merchant: str = Field(..., description="Merchant name")
    category: str = Field(..., description="Transaction category")
    amt: float = Field(..., description="Transaction amount")
    city: str = Field(..., description="City where transaction occurred")
    state: str = Field(..., description="State where transaction occurred")
    lat: float = Field(..., description="Latitude of transaction")
    long: float = Field(..., description="Longitude of transaction")
    city_pop: int = Field(..., description="City population")
    job: str = Field(..., description="Cardholder job")
    merch_lat: float = Field(..., description="Merchant latitude")
    merch_long: float = Field(..., description="Merchant longitude")
    
    class Config:
        json_schema_extra = {
            "example": {
                "merchant": "fraud_Rippin, Kub and Mann",
                "category": "misc_net",
                "amt": 4.97,
                "city": "Malton",
                "state": "OH",
                "lat": 39.9459,
                "long": -82.1661,
                "city_pop": 35,
                "job": "Psychologist, counselling",
                "merch_lat": 39.9459,
                "merch_long": -82.1661
            }
        }

# Universal Response Schema
class FraudDetectionResponse(BaseModel):
    model_type: str = Field(..., description="Model type used (online-payment or credit-card)")
    is_fraud: bool = Field(..., description="Fraud prediction (True/False)")
    fraud_probability: float = Field(..., description="Probability of fraud (0.0-1.0)")
    confidence_level: str = Field(..., description="Confidence level (low, medium, high)")
    risk_score: str = Field(..., description="Risk assessment (LOW, MEDIUM, HIGH)")
    transaction_amount: float = Field(..., description="Transaction amount")
    features_used: dict = Field(..., description="Features used for prediction")
    
    class Config:
        json_schema_extra = {
            "example": {
                "model_type": "credit-card",
                "is_fraud": False,
                "fraud_probability": 0.1234,
                "confidence_level": "high",
                "risk_score": "LOW",
                "transaction_amount": 4.97,
                "features_used": {
                    "amt": 4.97,
                    "merchant_encoded": 245
                }
            }
        }

# Health Response
class HealthResponse(BaseModel):
    status: str = Field(..., description="Service status")
    models_loaded: Dict[str, bool] = Field(..., description="Status of loaded models")
    message: str = Field(..., description="Status message")
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "models_loaded": {
                    "online-payment": True,
                    "credit-card": True
                },
                "message": "All services are healthy"
            }
        }
