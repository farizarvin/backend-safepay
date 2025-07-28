# app/main.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging

from app.config import settings
from app.models.model_loader import MultiModelLoader  
from app.services.fraud_service import FraudDetectionService
from app.schemas.prediction import (
    OnlinePaymentInput, CreditCardInput, FraudDetectionResponse, HealthResponse  
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables
model_loader = None
fraud_service = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global model_loader, fraud_service
    
    logger.info("🚀 Starting Multi-Model Fraud Detection API...")
    
    # Initialize model loader
    model_loader = MultiModelLoader() 
    
    # ✅ FIX: Load models dengan encoder paths
    models_config = {
        "online-payment": {
            "model_path": "app/models/ml_models/online_payment.pkl",
            "encoder_path": "app/models/ml_models/online_payment_label_encoder.pkl"  # Encoder online payment existing
        },
        "credit-card": {
            "model_path": "app/models/ml_models/credit_card.pkl", 
            "encoder_path": "app/models/ml_models/credit_card_label_encoders.pkl"  # File yang Anda buat
        }
    }
    
    loaded_models = []
    for model_name, config in models_config.items():
        success = model_loader.load_model(
            model_name=model_name,
            model_path=config["model_path"],
            encoder_path=config["encoder_path"]
        )
        
        if success:
            loaded_models.append(model_name)
            logger.info(f"✅ {model_name} model and encoder loaded successfully")
        else:
            logger.error(f"❌ Failed to load {model_name} model")
    
    if loaded_models:
        fraud_service = FraudDetectionService(model_loader)
        logger.info(f"✅ Fraud service initialized with models: {loaded_models}")
    else:
        logger.error("❌ No models loaded, service unavailable")
        fraud_service = None
    
    yield
    
    # Shutdown
    logger.info("🔄 Shutting down Multi-Model Fraud Detection API...")

app = FastAPI(
    title="Multi-Model Fraud Detection API",
    version="2.0.0",  
    description="API untuk deteksi fraud dengan multiple machine learning models",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000", 
        "http://127.0.0.1:3000",
        "http://localhost:3001",
        "*"  # For development only
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
@app.get("/", response_model=HealthResponse)
async def root():
    models_status = model_loader.get_loaded_models_status() if model_loader else {}
    return HealthResponse(
        status="running",
        models_loaded=models_status,
        message="Multi-Model Fraud Detection API is running"
    )

@app.get("/health", response_model=HealthResponse)
async def health_check():
    if not model_loader:
        return HealthResponse(
            status="unhealthy",
            models_loaded={},
            message="Model loader not initialized"
        )
    
    models_status = model_loader.get_loaded_models_status()
    all_loaded = any(models_status.values())
    
    return HealthResponse(
        status="healthy" if all_loaded else "unhealthy",
        models_loaded=models_status,
        message="Service is healthy" if all_loaded else "No models loaded"
    )

# ✅ FIX: Gunakan OnlinePaymentInput
@app.post("/predict/online-payment", response_model=FraudDetectionResponse)
async def predict_online_payment_fraud(data: OnlinePaymentInput):
    """Prediksi fraud untuk online payment"""
    if not fraud_service:
        raise HTTPException(status_code=503, detail="Fraud detection service not available")
    
    if not model_loader.is_model_loaded("online-payment"):
        raise HTTPException(status_code=503, detail="Online payment model not loaded")
    
    return await fraud_service.predict_online_payment_fraud(data)

@app.post("/predict/credit-card", response_model=FraudDetectionResponse)
async def predict_credit_card_fraud(data: CreditCardInput):
    """Prediksi fraud untuk credit card"""
    if not fraud_service:
        raise HTTPException(status_code=503, detail="Fraud detection service not available")
    
    if not model_loader.is_model_loaded("credit-card"):
        raise HTTPException(status_code=503, detail="Credit card model not loaded")
    
    return await fraud_service.predict_credit_card_fraud(data)

# ✅ FIX: Legacy endpoint
@app.post("/predict", response_model=FraudDetectionResponse)
async def predict_fraud_legacy(data: OnlinePaymentInput):
    """Legacy endpoint - redirect ke online payment"""
    return await predict_online_payment_fraud(data)

@app.get("/models/status")
async def get_models_status():
    """Status semua model dan encoders"""
    if not model_loader:
        return {"error": "Model loader not initialized"}
    
    models_status = model_loader.get_loaded_models_status()
    
    return {
        "models_loaded": models_status,
        "available_endpoints": {
            "online-payment": "/predict/online-payment" if models_status.get("online-payment", False) else "Model not loaded",
            "credit-card": "/predict/credit-card" if models_status.get("credit-card", False) else "Model not loaded",
            "legacy": "/predict"
        },
        "encoders_status": {
            model: "Available" if model_loader.get_label_encoders(model) else "Not available" 
            for model in models_status.keys()
        }
    }
