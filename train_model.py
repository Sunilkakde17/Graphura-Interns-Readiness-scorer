# train_model.py - Place in project root
"""
Model training script for the Intern Portfolio Readiness Predictor.
This script trains the Random Forest model using the dataset and saves it for use in the Flask app.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os
import warnings
warnings.filterwarnings('ignore')


def train_and_save_model(data_path="data/Graphura_Intern_Portfolio_ML_Dataset.xlsx", 
                         model_path="models/resume_rf_model.pkl"):
    """
    Train the Random Forest model and save it to disk.
    
    Args:
        data_path (str): Path to the Excel dataset
        model_path (str): Path where to save the trained model
    
    Returns:
        tuple: (model, feature_columns, label_encoder)
    """
    print("=" * 60)
    print("INTERN PORTFOLIO READINESS MODEL TRAINING")
    print("=" * 60)
    
    # Check if data file exists
    if not os.path.exists(data_path):
        print(f"Error: Data file not found at {data_path}")
        print("Creating sample data for training...")
        df = create_sample_training_data()
    else:
        # Load the dataset
        print(f"Loading data from {data_path}...")
        try:
            # Try to load from ML-Ready Features sheet first
            df = pd.read_excel(data_path, sheet_name="4. ML-Ready Features")
            print("Loaded ML-Ready Features sheet")
        except:
            # Fallback to main sheet
            df = pd.read_excel(data_path)
            # Set column names from first row if needed
            if df.iloc[0, 0] == "Intern_ID":
                df.columns = df.iloc[0]
                df = df[1:]
                df.reset_index(drop=True, inplace=True)
            print("Loaded main dataset")
    
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()[:10]}...")
    
    # Select features for training (matching your notebook)
    selected_columns = [
        'Role', 'Department',
        'phone_present', 'email_present', 'linkedin_present', 'github_present',
        'summary_present', 'projects_section_present', 'skills_section_present',
        'experience_section_present', 'certifications_section_present',
        'achievements_section_present', 'extracurricular_section_present',
        'num_projects', 'role_relevant_project_count',
        'prog_lang_count', 'framework_count', 'database_count', 'tool_count',
        'soft_skill_count',
        'total_experiences', 'exp_years',
        'certification_count', 'achievement_count',
        'readiness_label'
    ]
    
    # Filter available columns
    available_cols = [col for col in selected_columns if col in df.columns]
    df_filtered = df[available_cols].copy()
    print(f"Using {len(available_cols)} features")
    
    # Handle missing values
    df_filtered = df_filtered.fillna(0)
    
    # Separate features and target
    y = df_filtered['readiness_label']
    X = df_filtered.drop(columns=['readiness_label'])
    
    # Encode categorical features
    le_dict = {}
    for col in X.select_dtypes(include='object').columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        le_dict[col] = le
    
    # Encode target
    target_encoder = LabelEncoder()
    y_encoded = target_encoder.fit_transform(y)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42
    )
    
    print(f"Training samples: {len(X_train)}, Testing samples: {len(X_test)}")
    
    # Train Random Forest model (using your exact parameters)
    print("\nTraining Random Forest Classifier...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        random_state=42
    )
    model.fit(X_train, y_train)
    
    # Evaluate
    train_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test)
    print(f"\nTraining Accuracy: {train_acc:.4f}")
    print(f"Testing Accuracy: {test_acc:.4f}")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print("\nTop 10 Important Features:")
    print(feature_importance.head(10).to_string(index=False))
    
    # Save the model and encoders
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    model_data = {
        'model': model,
        'feature_columns': X.columns.tolist(),
        'label_encoder': target_encoder,
        'categorical_encoders': le_dict,
        'training_accuracy': train_acc,
        'testing_accuracy': test_acc,
        'feature_importance': feature_importance.to_dict('records')
    }
    
    joblib.dump(model_data, model_path)
    print(f"\n✅ Model saved successfully to {model_path}")
    print(f"   Model accuracy: {test_acc:.2%}")
    
    return model, X.columns.tolist(), target_encoder


def create_sample_training_data():
    """Create sample training data if real data is not available."""
    np.random.seed(42)
    n_samples = 200
    
    data = {
        'Intern_ID': [f'GRP{i:03d}' for i in range(1, n_samples + 1)],
        'Name': [f'Intern_{i}' for i in range(1, n_samples + 1)],
        'Role': np.random.choice(['Backend Developer', 'Frontend Developer', 'Full Stack Developer', 
                                  'Data Science & Analytics', 'MERN Stack Developer', 'Sales', 
                                  'Digital Marketing', 'Content Creator'], n_samples),
        'Department': np.random.choice(['Technical', 'Non-Technical'], n_samples, p=[0.7, 0.3]),
        'phone_present': np.random.choice([0, 1], n_samples, p=[0.1, 0.9]),
        'email_present': np.random.choice([0, 1], n_samples, p=[0.05, 0.95]),
        'linkedin_present': np.random.choice([0, 1], n_samples, p=[0.2, 0.8]),
        'github_present': np.random.choice([0, 1], n_samples, p=[0.3, 0.7]),
        'summary_present': np.random.choice([0, 1], n_samples, p=[0.1, 0.9]),
        'projects_section_present': np.random.choice([0, 1], n_samples, p=[0.1, 0.9]),
        'skills_section_present': np.random.choice([0, 1], n_samples, p=[0.1, 0.9]),
        'experience_section_present': np.random.choice([0, 1], n_samples, p=[0.2, 0.8]),
        'certifications_section_present': np.random.choice([0, 1], n_samples, p=[0.3, 0.7]),
        'achievements_section_present': np.random.choice([0, 1], n_samples, p=[0.6, 0.4]),
        'extracurricular_section_present': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
        'num_projects': np.random.randint(1, 7, n_samples),
        'role_relevant_project_count': np.random.randint(1, 5, n_samples),
        'prog_lang_count': np.random.randint(0, 6, n_samples),
        'framework_count': np.random.randint(0, 8, n_samples),
        'database_count': np.random.randint(0, 4, n_samples),
        'tool_count': np.random.randint(0, 10, n_samples),
        'soft_skill_count': np.random.randint(0, 6, n_samples),
        'total_experiences': np.random.randint(0, 5, n_samples),
        'exp_years': np.random.exponential(2, n_samples).round(1),
        'certification_count': np.random.randint(0, 6, n_samples),
        'achievement_count': np.random.randint(0, 4, n_samples),
    }
    
    df = pd.DataFrame(data)
    
    # Calculate portfolio score (synthetic)
    df['portfolio_score_100'] = (
        df['num_projects'] * 5 +
        df['prog_lang_count'] * 4 +
        df['framework_count'] * 3 +
        df['tool_count'] * 2 +
        df['certification_count'] * 3 +
        df['exp_years'] * 5
    ).clip(0, 100)
    
    # Add readiness labels
    conditions = [
        df['portfolio_score_100'] >= 80,
        (df['portfolio_score_100'] >= 50) & (df['portfolio_score_100'] < 80),
        df['portfolio_score_100'] < 50
    ]
    choices = ['Job Ready', 'Almost Ready', 'Needs Improvement']
    df['readiness_label'] = np.select(conditions, choices)
    
    return df


if __name__ == "__main__":
    train_and_save_model()