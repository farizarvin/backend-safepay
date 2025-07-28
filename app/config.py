# app/config.py
from pydantic_settings import BaseSettings
from functools import lru_cache
import os

class Settings(BaseSettings):
    # Application settings
    APP_NAME: str = "Fraud Detection API"
    VERSION: str = "1.0.0"
    DEBUG: bool = True
    
    # Model settings
    MODEL_PATH: str = "app/models/model.pkl"
    
    # API settings
    API_V1_PREFIX: str = "/api/v1"
    
    class Config:
        env_file = ".env"
        case_sensitive = True

@lru_cache()
def get_settings():
    return Settings()

settings = get_settings()
