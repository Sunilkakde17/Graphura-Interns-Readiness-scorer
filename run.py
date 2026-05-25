# run.py - Entry point script

import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Check if model exists, if not train it
from train_model import train_and_save_model

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("GRAPHURA INTERN PORTFOLIO READINESS SCORER")
    print("=" * 60)
    
    # Check if model exists
    model_path = "models/resume_rf_model.pkl"
    if not os.path.exists(model_path):
        print("\n⚠️ Model not found. Training new model...\n")
        train_and_save_model()
    
    # Import and run the app
    from app import app
    app.run(debug=True, host="0.0.0.0", port=5000)