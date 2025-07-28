# app/utils/preprocessing.py
import numpy as np
from typing import List, Dict, Any
import joblib
import os
import logging

logger = logging.getLogger(__name__)

# ✅ Online Payment Preprocessing - Dengan Parameter Label Encoder
def preprocess_online_payment_data(data: Dict, label_encoder: Any = None) -> List[float]:
    """Preprocessing untuk online payment dengan label encoder dari pkl"""
    
    # Fallback mapping
    fallback_mapping = {
        'PAYMENT': 0, 'TRANSFER': 1, 'CASH_OUT': 2, 'DEBIT': 3, 'CASH_IN': 4
    }
    
    try:
        if label_encoder and hasattr(label_encoder, 'transform'):
            # Gunakan sklearn LabelEncoder dari pkl
            type_encoded = label_encoder.transform([data['type']])[0]
        elif label_encoder and isinstance(label_encoder, dict):
            # Gunakan dictionary mapping
            type_encoded = label_encoder.get(data['type'], 0)
        else:
            # Fallback ke mapping default
            type_encoded = fallback_mapping.get(data['type'], 0)
            
    except ValueError:
        # Handle unknown transaction type
        logger.warning(f"Unknown transaction type: {data['type']}, using fallback")
        type_encoded = fallback_mapping.get(data['type'], 0)
    except Exception as e:
        logger.error(f"Error encoding transaction type: {e}")
        type_encoded = fallback_mapping.get(data['type'], 0)
    
    # Feature engineering
    diffOrig = data['oldbalanceOrg'] - data['newbalanceOrig'] + data['amount']
    diffDest = data['newbalanceDest'] - data['oldbalanceDest'] - data['amount']
    
    features = [
        data['step'],
        data['amount'],
        int(type_encoded),
        diffOrig,
        diffDest
    ]
    
    return features

def validate_online_payment_data(data: Dict) -> bool:  # ✅ Rename function
    """Validasi untuk online payment"""
    required_fields = ['step', 'type', 'amount', 'oldbalanceOrg', 
                      'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']
    
    for field in required_fields:
        if field not in data:
            return False
    
    valid_types = ['PAYMENT', 'TRANSFER', 'CASH_OUT', 'DEBIT', 'CASH_IN']
    if data['type'] not in valid_types:
        return False
    
    numeric_fields = ['step', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 
                     'oldbalanceDest', 'newbalanceDest']
    
    for field in numeric_fields:
        if not isinstance(data[field], (int, float)) or data[field] < 0:
            return False
    
    return True

def get_online_payment_feature_names() -> List[str]:  # ✅ Rename function
    """Feature names untuk online payment"""
    return ['step', 'amount', 'type_encoded', 'diffOrig', 'diffDest']

# ✅ Credit Card Preprocessing - Dengan Parameter Label Encoders PKL
def preprocess_credit_card_data(data: Dict, label_encoders: Dict = None) -> List[float]:
    """Preprocessing untuk credit card dengan label encoders pkl"""
    features = []
    
    # Add numerical features first (6 features)
    numerical_features = ['amt', 'lat', 'long', 'city_pop', 'merch_lat', 'merch_long']
    for feature in numerical_features:
        features.append(float(data[feature]))
    
    # Add encoded categorical features (5 features) - MENGGUNAKAN PKL ENCODERS
    categorical_features = ['merchant', 'category', 'city', 'state', 'job']
    
    if label_encoders and isinstance(label_encoders, dict):
        for feature in categorical_features:
            if feature in label_encoders:
                encoder = label_encoders[feature]
                try:
                    # Transform menggunakan fitted encoder dari pkl
                    encoded_value = encoder.transform([str(data[feature])])[0]
                    features.append(int(encoded_value))
                    logger.debug(f"Encoded {feature}: '{data[feature]}' -> {encoded_value}")
                    
                except ValueError:
                    # Handle unknown category
                    logger.warning(f"Unknown value for {feature}: '{data[feature]}', using 0")
                    features.append(0)
                except Exception as e:
                    logger.error(f"Error encoding {feature}: {e}")
                    features.append(0)
            else:
                logger.warning(f"No encoder found for {feature}")
                features.append(0)
    else:
        # Fallback ke hash-based encoding jika pkl encoders tidak tersedia
        logger.warning("Label encoders not available, using hash-based fallback")
        for feature in categorical_features:
            try:
                category_value = str(data[feature]).strip().lower()
                encoded_value = abs(hash(category_value)) % 1000
                features.append(encoded_value)
            except Exception as e:
                logger.warning(f"Error hash encoding {feature}: {e}")
                features.append(0)
    
    return features

def validate_credit_card_data(data: Dict) -> bool:
    """Validasi untuk credit card"""
    required_fields = ['merchant', 'category', 'amt', 'city', 'state', 
                      'lat', 'long', 'city_pop', 'job', 'merch_lat', 'merch_long']
    
    for field in required_fields:
        if field not in data:
            return False
    
    numeric_fields = ['amt', 'lat', 'long', 'city_pop', 'merch_lat', 'merch_long']
    for field in numeric_fields:
        if not isinstance(data[field], (int, float)):
            return False
    
    if data['amt'] <= 0:
        return False
    
    string_fields = ['merchant', 'category', 'city', 'state', 'job']
    for field in string_fields:
        if not isinstance(data[field], str) or not data[field].strip():
            return False
    
    return True

def get_credit_card_feature_names() -> List[str]:
    """Feature names untuk credit card"""
    return ['amt', 'lat', 'long', 'city_pop', 'merch_lat', 'merch_long',
            'merchant_encoded', 'category_encoded', 'city_encoded', 
            'state_encoded', 'job_encoded']

# ✅ Keep legacy functions for backward compatibility
def load_label_encoder():
    """Legacy function - Load label encoder"""
    encoder_path = "app/models/ml_models/label_encoder.pkl"
    if os.path.exists(encoder_path):
        return joblib.load(encoder_path)
    else:
        return {
            'PAYMENT': 0, 'TRANSFER': 1, 'CASH_OUT': 2, 'DEBIT': 3, 'CASH_IN': 4
        }

def preprocess_transaction_data(data: Dict) -> List[float]:
    """Legacy function - redirect to new function"""
    le = load_label_encoder()
    return preprocess_online_payment_data(data, le)

def get_feature_names():
    """Legacy function - redirect to new function"""
    return get_online_payment_feature_names()

def validate_transaction_data(data: Dict) -> bool:
    """Legacy function - redirect to new function"""
    return validate_online_payment_data(data)
