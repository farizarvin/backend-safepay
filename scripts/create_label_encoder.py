import pandas as pd
from sklearn.preprocessing import LabelEncoder
import joblib
import os

def create_and_save_label_encoder():
    """Create label encoder for transaction types"""
    le = LabelEncoder()
    
    # Fit with all possible transaction types
    transaction_types = ['PAYMENT', 'TRANSFER', 'CASH_OUT', 'DEBIT', 'CASH_IN']
    le.fit(transaction_types)
    
    # Create directory if not exists
    os.makedirs("app/models/ml_models", exist_ok=True)
    
    # Save label encoder
    joblib.dump(le, 'app/models/ml_models/label_encoder.pkl')
    
    print("✅ Label encoder created and saved!")
    print("Transaction types mapping:")
    for type_name in transaction_types:
        print(f"  {type_name}: {le.transform([type_name])[0]}")

if __name__ == "__main__":
    create_and_save_label_encoder()
