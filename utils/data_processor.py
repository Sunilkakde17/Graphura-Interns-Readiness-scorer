# utils/data_processor.py - FIXED (remove the self-import)

"""
Data processing utilities for managing intern portfolio data.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import os


class DataProcessor:
    """
    Data processor class for handling intern portfolio data.
    """
    
    def __init__(self):
        """Initialize the data processor."""
        self.label_encoders = {}
        self.df_master = None
    
    def load_and_process_data(self, filepath="data/Graphura_Intern_Portfolio_ML_Dataset.xlsx"):
        """
        Load and process the Excel file properly.
        
        Args:
            filepath (str): Path to the Excel file
            
        Returns:
            pd.DataFrame: Processed dataframe with required columns
        """
        if not os.path.exists(filepath):
            print(f"File not found: {filepath}, using sample data")
            return self.create_sample_data()
        
        try:
            # Load with header=1 (second row contains column names)
            df = pd.read_excel(filepath, sheet_name="4. ML-Ready Features", header=1)
            print(f"Loaded {len(df)} records from ML-Ready Features sheet")
            
            # Convert numeric columns
            numeric_cols = ['portfolio_score_100', 'skills_score_10', 'tools_score_10', 
                           'projects_score_10', 'docs_score_10', 'certs_score_10', 
                           'exp_score_10', 'intern_score_10']
            
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            
            # Create readiness_label if not exists
            if 'readiness_label' not in df.columns:
                def get_readiness(score):
                    if score >= 80:
                        return 'Job Ready'
                    elif score >= 50:
                        return 'Almost Ready'
                    else:
                        return 'Needs Improvement'
                df['readiness_label'] = df['portfolio_score_100'].apply(get_readiness)
            
            # Fill NaN values
            df = df.fillna(0)
            
            return df
            
        except Exception as e:
            print(f"Error loading data: {e}")
            return self.create_sample_data()
    
    def create_sample_data(self):
        """Create sample data for testing when real data is not available."""
        np.random.seed(42)
        n_samples = 50
        
        roles = ['Backend Developer', 'Frontend Developer', 'Full Stack Developer', 
                'Data Science & Analytics', 'MERN Stack Developer', 'Sales', 
                'Digital Marketing', 'Content Creator']
        
        data = {
            'Intern_ID': [f'GRP{i:03d}' for i in range(1, n_samples + 1)],
            'Name': [f'Intern_{i}' for i in range(1, n_samples + 1)],
            'Role': np.random.choice(roles, n_samples),
            'Department': np.random.choice(['Technical', 'Non-Technical'], n_samples, p=[0.6, 0.4]),
            'num_projects': np.random.randint(1, 6, n_samples),
            'prog_lang_count': np.random.randint(1, 5, n_samples),
            'framework_count': np.random.randint(0, 4, n_samples),
            'database_count': np.random.randint(0, 3, n_samples),
            'tool_count': np.random.randint(1, 6, n_samples),
            'soft_skill_count': np.random.randint(1, 4, n_samples),
            'total_experiences': np.random.randint(0, 3, n_samples),
            'exp_years': np.random.choice([0, 0.5, 1, 1.5, 2, 3], n_samples),
            'certification_count': np.random.randint(0, 4, n_samples),
            'achievement_count': np.random.randint(0, 3, n_samples),
            'github_present': np.random.choice([0, 1], n_samples, p=[0.3, 0.7]),
            'linkedin_present': np.random.choice([0, 1], n_samples, p=[0.2, 0.8]),
            'projects_section_present': np.random.choice([0, 1], n_samples, p=[0.1, 0.9]),
        }
        
        df = pd.DataFrame(data)
        
        # Calculate portfolio score
        df['portfolio_score_100'] = (
            df['num_projects'] * 5 +
            df['prog_lang_count'] * 4 +
            df['framework_count'] * 3 +
            df['tool_count'] * 2 +
            df['certification_count'] * 3 +
            df['exp_years'] * 5 +
            df['github_present'] * 5 +
            df['linkedin_present'] * 3
        ).clip(0, 100)
        
        # Add readiness labels
        readiness_labels = []
        for score in df['portfolio_score_100']:
            if score >= 80:
                readiness_labels.append('Job Ready')
            elif score >= 50:
                readiness_labels.append('Almost Ready')
            else:
                readiness_labels.append('Needs Improvement')
        df['readiness_label'] = readiness_labels
        
        return df
    
    def get_feature_columns(self):
        """Get the list of feature columns."""
        return [
            'num_projects', 'prog_lang_count', 'framework_count', 'database_count', 
            'tool_count', 'soft_skill_count', 'total_experiences', 'exp_years', 
            'certification_count', 'achievement_count', 'github_present', 
            'linkedin_present', 'projects_section_present'
        ]
    
    def get_dashboard_stats(self, df):
        """Generate dashboard statistics from the data."""
        if df is None or len(df) == 0:
            return {
                'total_interns': 0,
                'job_ready_count': 0,
                'almost_ready_count': 0,
                'needs_improvement_count': 0,
                'average_score': 0,
                'job_ready_percentage': 0,
                'almost_ready_percentage': 0,
                'needs_improvement_percentage': 0
            }
        
        total = len(df)
        
        if 'readiness_label' in df.columns:
            job_ready_count = len(df[df['readiness_label'] == 'Job Ready'])
            almost_ready_count = len(df[df['readiness_label'] == 'Almost Ready'])
            needs_improvement_count = len(df[df['readiness_label'] == 'Needs Improvement'])
        else:
            job_ready_count = almost_ready_count = needs_improvement_count = 0
        
        if 'portfolio_score_100' in df.columns:
            avg_score = float(df['portfolio_score_100'].mean())
        else:
            avg_score = 0
        
        stats = {
            'total_interns': total,
            'job_ready_count': job_ready_count,
            'almost_ready_count': almost_ready_count,
            'needs_improvement_count': needs_improvement_count,
            'average_score': avg_score,
            'job_ready_percentage': round((job_ready_count / total) * 100, 1) if total > 0 else 0,
            'almost_ready_percentage': round((almost_ready_count / total) * 100, 1) if total > 0 else 0,
            'needs_improvement_percentage': round((needs_improvement_count / total) * 100, 1) if total > 0 else 0
        }
        
        return stats
    
    def get_department_stats(self, df):
        """Generate department-wise statistics."""
        dept_stats = []
        
        if df is not None and 'Department' in df.columns:
            for dept in df['Department'].unique():
                dept_df = df[df['Department'] == dept]
                stats = {
                    'department': dept,
                    'count': len(dept_df),
                    'average_score': float(dept_df['portfolio_score_100'].mean()) if 'portfolio_score_100' in dept_df.columns else 0,
                }
                dept_stats.append(stats)
        
        return dept_stats
    
    def get_leaderboard_data(self, df, limit=50):
        """Get leaderboard data sorted by portfolio score."""
        if df is None or len(df) == 0:
            return []
        
        leaderboard_df = df.copy()
        
        # Ensure required columns exist
        if 'portfolio_score_100' not in leaderboard_df.columns:
            leaderboard_df['portfolio_score_100'] = 50
        
        if 'readiness_label' not in leaderboard_df.columns:
            leaderboard_df['readiness_label'] = 'Almost Ready'
        
        # Select columns for display
        display_cols = ['Intern_ID', 'Name', 'Role', 'Department', 'portfolio_score_100', 'readiness_label']
        available_cols = [col for col in display_cols if col in leaderboard_df.columns]
        
        if not available_cols:
            available_cols = leaderboard_df.columns[:5].tolist()
        
        leaderboard_df = leaderboard_df[available_cols]
        
        # Convert score to numeric and sort
        if 'portfolio_score_100' in leaderboard_df.columns:
            leaderboard_df['portfolio_score_100'] = pd.to_numeric(leaderboard_df['portfolio_score_100'], errors='coerce').fillna(50)
            leaderboard_df = leaderboard_df.sort_values('portfolio_score_100', ascending=False)
        
        # Clean up NaN values
        leaderboard_df = leaderboard_df.fillna('-')
        
        # Convert to records
        records = leaderboard_df.head(limit).to_dict(orient='records')
        
        # Ensure each record has all required fields
        for record in records:
            if 'portfolio_score_100' not in record:
                record['portfolio_score_100'] = 50
            if 'readiness_label' not in record:
                record['readiness_label'] = 'Almost Ready'
        
        return records