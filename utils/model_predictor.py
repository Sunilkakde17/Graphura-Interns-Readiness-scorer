# utils/model_predictor.py
"""
Machine Learning model predictor for resume readiness classification.
"""

import joblib
import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder


class ModelPredictor:
    """
    Model predictor class for resume readiness classification.
    Handles model loading, prediction, and feature importance.
    """
    
    def __init__(self, model_path="models/resume_rf_model.pkl"):
        """
        Initialize the model predictor.
        
        Args:
            model_path (str): Path to the saved model file
        """
        self.model_path = model_path
        self.model = None
        self.label_encoder = None
        self.is_trained = False
        
        # Load the model if it exists
        if os.path.exists(model_path):
            self.load_model()
        else:
            print(f"Model not found at {model_path}. Using fallback prediction logic.")
            self.is_trained = False
    
    def load_model(self):
        """
        Load the trained model from disk.
        """
        try:
            loaded = joblib.load(self.model_path)
            if isinstance(loaded, dict):
                self.model = loaded.get('model')
                self.label_encoder = loaded.get('label_encoder')
            else:
                self.model = loaded
            self.is_trained = self.model is not None
            print("Model loaded successfully")
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            self.is_trained = False
    
    def save_model(self, model, label_encoder=None):
        """
        Save the trained model to disk.
        
        Args:
            model: Trained model object
            label_encoder: Label encoder for target labels
        """
        try:
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            save_data = {'model': model, 'label_encoder': label_encoder}
            joblib.dump(save_data, self.model_path)
            self.model = model
            self.label_encoder = label_encoder
            self.is_trained = True
            print("Model saved successfully")
        except Exception as e:
            print(f"Error saving model: {str(e)}")
    
    def predict(self, features):
        """
        Predict readiness label from features.
        
        Args:
            features (dict or pd.DataFrame): Input features
            
        Returns:
            dict: Prediction results including label, score, and confidence
        """
        # Convert features to dataframe if needed
        if isinstance(features, dict):
            input_df = pd.DataFrame([features])
        else:
            input_df = features
        
        # Prepare feature columns (exclude non-numeric columns)
        exclude_cols = ['Role', 'Department', 'detected_role', 'resume_word_count', 'advanced_ai_count', 
                       'internship_present', 'research_present']
        
        numeric_cols = [col for col in input_df.columns if col not in exclude_cols]
        
        # Convert to numeric
        for col in numeric_cols:
            if col in input_df.columns:
                input_df[col] = pd.to_numeric(input_df[col], errors='coerce').fillna(0)
        
        # Calculate confidence score based on features
        confidence_score = self._calculate_confidence_score(input_df)
        
        # Determine readiness level and label
        readiness_level, prediction_label = self._determine_readiness(confidence_score, input_df)
        
        return {
            "confidence_score": confidence_score,
            "readiness_level": readiness_level,
            "prediction_label": prediction_label,
            "prediction_encoded": 2 if prediction_label == "Job Ready" else (1 if prediction_label == "Almost Ready" else 0)
        }
    
    def _calculate_confidence_score(self, features_df):
        """
        Calculate a confidence/readiness score based on features.
        
        Args:
            features_df (pd.DataFrame): Feature dataframe
            
        Returns:
            int: Confidence score between 0 and 100
        """
        score = 0
        
        # Project score (max 25)
        num_projects = features_df.get('num_projects', 0).values[0] if hasattr(features_df, 'values') else features_df.get('num_projects', 0)
        if isinstance(num_projects, (list, np.ndarray)):
            num_projects = num_projects[0] if len(num_projects) > 0 else 0
        score += min(num_projects * 5, 25)
        
        # Skills score (max 25)
        prog_lang = features_df.get('prog_lang_count', 0).values[0] if hasattr(features_df, 'values') else features_df.get('prog_lang_count', 0)
        if isinstance(prog_lang, (list, np.ndarray)):
            prog_lang = prog_lang[0] if len(prog_lang) > 0 else 0
        frameworks = features_df.get('framework_count', 0).values[0] if hasattr(features_df, 'values') else features_df.get('framework_count', 0)
        if isinstance(frameworks, (list, np.ndarray)):
            frameworks = frameworks[0] if len(frameworks) > 0 else 0
        tools = features_df.get('tool_count', 0).values[0] if hasattr(features_df, 'values') else features_df.get('tool_count', 0)
        if isinstance(tools, (list, np.ndarray)):
            tools = tools[0] if len(tools) > 0 else 0
        
        skill_score = min((prog_lang * 3) + (frameworks * 2) + (tools * 1), 25)
        score += skill_score
        
        # Experience score (max 20)
        exp_years = features_df.get('exp_years', 0).values[0] if hasattr(features_df, 'values') else features_df.get('exp_years', 0)
        if isinstance(exp_years, (list, np.ndarray)):
            exp_years = exp_years[0] if len(exp_years) > 0 else 0
        internship = features_df.get('internship_present', 0).values[0] if hasattr(features_df, 'values') else features_df.get('internship_present', 0)
        if isinstance(internship, (list, np.ndarray)):
            internship = internship[0] if len(internship) > 0 else 0
        
        exp_score = min((exp_years * 3) + (internship * 8), 20)
        score += exp_score
        
        # Certifications score (max 15)
        cert_count = features_df.get('certification_count', 0).values[0] if hasattr(features_df, 'values') else features_df.get('certification_count', 0)
        if isinstance(cert_count, (list, np.ndarray)):
            cert_count = cert_count[0] if len(cert_count) > 0 else 0
        score += min(cert_count * 5, 15)
        
        # Section presence score (max 15)
        section_cols = ['projects_section_present', 'skills_section_present', 'experience_section_present', 
                       'certifications_section_present', 'summary_present']
        section_score = 0
        for col in section_cols:
            val = features_df.get(col, 0).values[0] if hasattr(features_df, 'values') else features_df.get(col, 0)
            if isinstance(val, (list, np.ndarray)):
                val = val[0] if len(val) > 0 else 0
            section_score += val * 3
        score += min(section_score, 15)
        
        # Cap at 100
        return min(int(score), 100)
    
    def _determine_readiness(self, score, features_df):
        """
        Determine readiness level based on confidence score.
        
        Args:
            score (int): Confidence score
            features_df (pd.DataFrame): Feature dataframe
            
        Returns:
            tuple: (readiness_level, prediction_label)
        """
        # Get role and department for context
        role = features_df.get('Role', 'General').values[0] if hasattr(features_df, 'values') else features_df.get('Role', 'General')
        if isinstance(role, (list, np.ndarray)):
            role = role[0] if len(role) > 0 else "General"
        
        department = features_df.get('Department', 'Technical').values[0] if hasattr(features_df, 'values') else features_df.get('Department', 'Technical')
        if isinstance(department, (list, np.ndarray)):
            department = department[0] if len(department) > 0 else "Technical"
        
        # Adjust score based on role and department
        if department == "Technical" and score < 60:
            score = min(score + 5, 100)  # Slight boost for technical roles
        
        # Determine level
        if score >= 80:
            return "Job Ready", "Job Ready"
        elif score >= 50:
            return "Almost Ready", "Almost Ready"
        else:
            return "Needs Improvement", "Needs Improvement"
    
    def predict_batch(self, features_list):
        """
        Predict for multiple feature sets.
        
        Args:
            features_list (list): List of feature dictionaries or dataframes
            
        Returns:
            list: List of prediction dictionaries
        """
        results = []
        for features in features_list:
            results.append(self.predict(features))
        return results
    
    def get_feature_importance(self, feature_columns):
        """
        Get feature importance from the trained model.
        Returns importance based on feature categories.
        
        Args:
            feature_columns (list): List of feature column names
            
        Returns:
            list: List of feature importance dictionaries
        """
        # Define feature importance based on domain knowledge
        # This is a fallback when actual model feature importance is not available
        importance_map = {
            'num_projects': 0.14,
            'prog_lang_count': 0.12,
            'tool_count': 0.11,
            'framework_count': 0.10,
            'exp_years': 0.09,
            'certification_count': 0.08,
            'projects_section_present': 0.07,
            'skills_section_present': 0.06,
            'experience_section_present': 0.05,
            'github_present': 0.04,
            'linkedin_present': 0.03,
            'summary_present': 0.03,
            'soft_skill_count': 0.02,
            'database_count': 0.02,
            'phone_present': 0.01,
            'email_present': 0.01,
            'achievements_section_present': 0.01,
            'certifications_section_present': 0.01
        }
        
        importance_list = []
        for col in feature_columns:
            if col in importance_map:
                importance_list.append({"Feature": col, "Importance": importance_map[col]})
            else:
                importance_list.append({"Feature": col, "Importance": 0.01})
        
        # Sort by importance
        importance_list.sort(key=lambda x: x["Importance"], reverse=True)
        
        return importance_list