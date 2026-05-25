# models/__init__.py
"""
Models package initialization.
Contains the trained Random Forest model for resume readiness prediction.
"""

import os
import joblib

MODEL_PATH = os.path.join(os.path.dirname(__file__), "resume_rf_model.pkl")

def load_model():
    """Load the trained model from disk."""
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return None

def model_exists():
    """Check if model file exists."""
    return os.path.exists(MODEL_PATH)