# app/services/fraud_service.py
from app.models.model_loader import MultiModelLoader  # ✅ FIX
from app.utils.preprocessing import (
    preprocess_online_payment_data, validate_online_payment_data, get_online_payment_feature_names,  # ✅ FIX function names
    preprocess_credit_card_data, validate_credit_card_data, get_credit_card_feature_names
)
from app.schemas.prediction import (
    OnlinePaymentInput, CreditCardInput, FraudDetectionResponse  # ✅ FIX: Gunakan OnlinePaymentInput
)
from fastapi import HTTPException
import logging
import numpy as np

logger = logging.getLogger(__name__)

class FraudDetectionService:
    def __init__(self, model_loader: MultiModelLoader):  # ✅ FIX: MultiModelLoader
        self.model_loader = model_loader
    
    async def predict_online_payment_fraud(self, data: OnlinePaymentInput) -> FraudDetectionResponse:  # ✅ FIX
        """Prediksi fraud untuk online payment"""
        try:
            if not validate_online_payment_data(data.dict()):  # ✅ FIX function name
                raise HTTPException(status_code=400, detail="Invalid online payment data")
            
            # ✅ Get label encoder dari MultiModelLoader
            label_encoder = self.model_loader.get_label_encoders("online-payment")
            
            features = preprocess_online_payment_data(data.dict(), label_encoder)  # ✅ Pass encoder
            feature_names = get_online_payment_feature_names()  # ✅ FIX function name
            
            # Validasi feature count
            model = self.model_loader.get_model("online-payment")
            expected_features = model.n_features_in_
            
            if len(features) != expected_features:
                raise ValueError(f"Feature count mismatch: generated {len(features)}, model expects {expected_features}")
            
            # Prediksi
            prediction = self.model_loader.predict("online-payment", [features])[0]
            prediction = bool(prediction)
            
            # Probabilitas
            probability = 0.0
            probabilities = self.model_loader.predict_proba("online-payment", [features])
            if probabilities is not None:
                probability = float(probabilities[0][1])
            
            # Risk assessment
            confidence_level, risk_score = self._assess_risk(probability)
            
            # Feature dictionary
            features_dict = {}
            for name, value in zip(feature_names, features):
                if isinstance(value, np.number):
                    features_dict[name] = value.item()
                else:
                    features_dict[name] = value
            
            return FraudDetectionResponse(
                model_type="online-payment",
                is_fraud=prediction,
                fraud_probability=probability,
                confidence_level=confidence_level,
                risk_score=risk_score,
                transaction_amount=float(data.amount),
                features_used=features_dict
            )
            
        except Exception as e:
            logger.error(f"Online payment prediction error: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Online payment prediction failed: {str(e)}")
    
    async def predict_credit_card_fraud(self, data: CreditCardInput) -> FraudDetectionResponse:
        """Prediksi fraud untuk credit card"""
        try:
            if not validate_credit_card_data(data.dict()):
                raise HTTPException(status_code=400, detail="Invalid credit card data")
            
            # ✅ Get label encoders dari MultiModelLoader
            label_encoders = self.model_loader.get_label_encoders("credit-card")
            
            features = preprocess_credit_card_data(data.dict(), label_encoders)  # ✅ Pass encoders
            feature_names = get_credit_card_feature_names()
            
            # Validasi feature count
            model = self.model_loader.get_model("credit-card")
            expected_features = model.n_features_in_
            
            if len(features) != expected_features:
                raise ValueError(f"Feature count mismatch: generated {len(features)}, model expects {expected_features}")
            
            # Prediksi
            prediction = self.model_loader.predict("credit-card", [features])[0]
            prediction = bool(prediction)
            
            # Probabilitas
            probability = 0.0
            probabilities = self.model_loader.predict_proba("credit-card", [features])
            if probabilities is not None:
                probability = float(probabilities[0][1])
            
            # Risk assessment
            confidence_level, risk_score = self._assess_risk(probability)
            
            # Feature dictionary
            features_dict = {}
            for name, value in zip(feature_names, features):
                if isinstance(value, np.number):
                    features_dict[name] = value.item()
                else:
                    features_dict[name] = value
            
            return FraudDetectionResponse(
                model_type="credit-card",
                is_fraud=prediction,
                fraud_probability=probability,
                confidence_level=confidence_level,
                risk_score=risk_score,
                transaction_amount=float(data.amt),
                features_used=features_dict
            )
            
        except Exception as e:
            logger.error(f"Credit card prediction error: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Credit card prediction failed: {str(e)}")
    
    def _assess_risk(self, probability: float) -> tuple:
        """Risk assessment berdasarkan probabilitas"""
        if probability > 0.8:
            return "high", "HIGH"
        elif probability > 0.6:
            return "medium", "MEDIUM"
        elif probability > 0.4:
            return "medium", "MEDIUM"
        else:
            return "high", "LOW"
