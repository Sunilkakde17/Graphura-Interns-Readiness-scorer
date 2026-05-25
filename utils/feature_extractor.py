# utils/feature_extractor.py
"""
Feature extraction utilities for resume analysis.
Extracts relevant features from resume text for ML prediction.
"""

import re
import pandas as pd


# Skill keywords for detection
SKILLS_KEYWORDS = {
    'programming': ['python', 'java', 'c++', 'javascript', 'typescript', 'go', 'rust', 'php', 'ruby', 'swift', 'kotlin', 'c#', 'r', 'sql'],
    'frameworks': ['react', 'angular', 'vue', 'django', 'flask', 'spring', 'node', 'express', 'tensorflow', 'pytorch', 'keras', 'scikit-learn', 'pandas', 'numpy', 'bootstrap', 'tailwind', 'jquery'],
    'databases': ['mysql', 'postgresql', 'mongodb', 'sqlite', 'firebase', 'oracle', 'redis', 'cassandra', 'dynamodb'],
    'tools': ['git', 'github', 'docker', 'kubernetes', 'aws', 'azure', 'gcp', 'jenkins', 'jira', 'postman', 'figma', 'power bi', 'tableau', 'excel', 'linux', 'opencv']
}

# Section keywords
SECTION_KEYWORDS = {
    'projects': ['project', 'developed', 'built', 'created', 'implemented', 'designed', 'application', 'system', 'website', 'platform'],
    'skills': ['skill', 'technical skill', 'technology', 'programming language', 'framework', 'tool'],
    'experience': ['experience', 'work', 'internship', 'intern', 'employment', 'job', 'position', 'role'],
    'certifications': ['certification', 'certified', 'certificate', 'course', 'training', 'credential'],
    'education': ['education', 'degree', 'university', 'college', 'bachelor', 'master', 'b.tech', 'b.e', 'm.tech', 'bca', 'mca', 'b.com', 'm.com', 'b.sc', 'm.sc', 'b.b.a', 'm.b.a'],
    'achievements': ['achievement', 'award', 'recognition', 'honor', 'prize', 'winner', 'scholarship']
}

# Role detection keywords
ROLE_KEYWORDS = {
    "Backend Developer": ['backend', 'api', 'rest', 'database', 'server', 'node', 'django', 'flask', 'spring', 'sql', 'nosql'],
    "Frontend Developer": ['frontend', 'react', 'angular', 'vue', 'html', 'css', 'javascript', 'ui', 'user interface', 'responsive'],
    "Full Stack Developer": ['full stack', 'fullstack', 'frontend', 'backend', 'mern', 'mean', 'react', 'node', 'express', 'mongodb'],
    "Data Science & Analytics": ['data science', 'data analyst', 'machine learning', 'ml', 'ai', 'tensorflow', 'pytorch', 'pandas', 'numpy', 'data visualization', 'statistics'],
    "MERN Stack Developer": ['mern', 'mongodb', 'express', 'react', 'node', 'mean'],
    "Sales": ['sales', 'business development', 'client', 'customer', 'revenue', 'marketing', 'lead generation', 'negotiation'],
    "Digital Marketing": ['digital marketing', 'seo', 'social media', 'content', 'google analytics', 'campaign', 'email marketing'],
    "Content Creator": ['content', 'writing', 'blog', 'creative', 'copywriting', 'social media', 'video', 'editing'],
    "HR": ['human resource', 'recruitment', 'hiring', 'talent', 'onboarding', 'employee', 'hr'],
    "UI/UX Designer": ['ui', 'ux', 'design', 'figma', 'adobe xd', 'prototype', 'wireframe', 'user experience', 'user interface']
}


def extract_features_from_text(text, feature_columns=None):
    """
    Extract features from resume text for ML prediction.
    
    Args:
        text (str): Extracted resume text
        feature_columns (list, optional): List of expected feature columns
        
    Returns:
        dict: Dictionary of extracted features
    """
    features = {}
    
    if not text:
        return {col: 0 for col in feature_columns} if feature_columns else {}
    
    # ==================== BASIC SECTIONS DETECTION ====================
    
    # LinkedIn presence
    features['linkedin_present'] = int('linkedin' in text or 'linked.in' in text)
    
    # GitHub presence
    features['github_present'] = int('github' in text or 'git hub' in text)
    
    # Summary/About section
    features['summary_present'] = int('summary' in text or 'about' in text or 'profile' in text)
    
    # Projects section
    features['projects_section_present'] = int(any(keyword in text for keyword in SECTION_KEYWORDS['projects']))
    
    # Skills section
    features['skills_section_present'] = int(any(keyword in text for keyword in SECTION_KEYWORDS['skills']))
    
    # Experience section
    features['experience_section_present'] = int(any(keyword in text for keyword in SECTION_KEYWORDS['experience']))
    
    # Certifications section
    features['certifications_section_present'] = int(any(keyword in text for keyword in SECTION_KEYWORDS['certifications']))
    
    # Achievements section
    features['achievements_section_present'] = int(any(keyword in text for keyword in SECTION_KEYWORDS['achievements']))
    
    # Extracurricular section
    features['extracurricular_section_present'] = int('extracurricular' in text or 'volunteer' in text or 'co-curricular' in text)
    
    # ==================== PROJECT COUNT ====================
    
    project_keywords = SECTION_KEYWORDS['projects'] + ['built', 'created', 'made', 'developed']
    features['num_projects'] = min(sum(text.count(keyword) for keyword in project_keywords) // 2, 6)
    
    # ==================== SKILL COUNTS ====================
    
    # Extract skills section text for better accuracy
    skills_text = text
    if 'skills' in text:
        start = text.find('skills')
        skills_text = text[start:start + 2000]
    elif 'technical skills' in text:
        start = text.find('technical skills')
        skills_text = text[start:start + 2000]
    
    # Count programming languages
    features['prog_lang_count'] = sum(1 for skill in SKILLS_KEYWORDS['programming'] if skill in skills_text)
    
    # Count frameworks
    features['framework_count'] = sum(1 for skill in SKILLS_KEYWORDS['frameworks'] if skill in skills_text)
    
    # Count databases
    features['database_count'] = sum(1 for skill in SKILLS_KEYWORDS['databases'] if skill in skills_text)
    
    # Count tools
    features['tool_count'] = sum(1 for skill in SKILLS_KEYWORDS['tools'] if skill in skills_text)
    
    # Soft skills count
    soft_skills = ['communication', 'teamwork', 'leadership', 'problem solving', 'analytical', 'creative', 'time management', 'organization', 'presentation', 'research']
    features['soft_skill_count'] = sum(1 for skill in soft_skills if skill in text)
    
    # ==================== EXPERIENCE METRICS ====================
    
    # Count experience entries
    experience_indicators = ['experience', 'internship', 'intern at', 'worked as', 'worked at', 'employed at']
    features['total_experiences'] = sum(text.count(indicator) for indicator in experience_indicators) // 2
    
    # Extract years of experience
    year_pattern = r'(\d+\.?\d*)\s*years?'
    years_matches = re.findall(year_pattern, text)
    if years_matches:
        features['exp_years'] = max(float(y) for y in years_matches)
    else:
        features['exp_years'] = 0
    
    # ==================== CERTIFICATIONS ====================
    
    features['certification_count'] = sum(text.count(keyword) for keyword in SECTION_KEYWORDS['certifications'])
    
    # ==================== ACHIEVEMENTS ====================
    
    features['achievement_count'] = sum(text.count(keyword) for keyword in SECTION_KEYWORDS['achievements'])
    
    # ==================== ROLE DETECTION ====================
    
    detected_roles = []
    for role, keywords in ROLE_KEYWORDS.items():
        if any(keyword in text for keyword in keywords):
            detected_roles.append(role)
    
    features['Role'] = detected_roles[0] if detected_roles else "General"
    
    # ==================== DEPARTMENT DETECTION ====================
    
    if any(keyword in text for keyword in ['backend', 'frontend', 'full stack', 'mern', 'api', 'database', 'server']):
        features['Department'] = "Technical"
    elif any(keyword in text for keyword in ['data science', 'analytics', 'ml', 'ai', 'machine learning']):
        features['Department'] = "Technical"
    else:
        features['Department'] = "Non-Technical"
    
    # ==================== CONTACT FLAGS ====================
    
    features['phone_present'] = int(bool(re.search(r'\b\d{10}\b', text)))
    features['email_present'] = int(bool(re.search(r'[\w\.-]+@[\w\.-]+\.\w+', text)))
    
    # ==================== RESUME WORD COUNT ====================
    
    features['resume_word_count'] = len(text.split())
    
    # ==================== ADVANCED METRICS ====================
    
    # Advanced AI keywords
    advanced_ai_keywords = ['cnn', 'lstm', 'transformer', 'nlp', 'computer vision', 'deep learning', 'rag', 'llm', 'generative ai']
    features['advanced_ai_count'] = sum(1 for keyword in advanced_ai_keywords if keyword in text)
    
    # Internship detection
    features['internship_present'] = int('internship' in text or 'intern at' in text or 'interned' in text)
    
    # Research detection
    features['research_present'] = int('research' in text or 'published' in text or 'conference' in text)
    
    # ==================== DERIVED METRICS ====================
    
    # Role-relevant project count (simplified)
    features['role_relevant_project_count'] = min(features['num_projects'], 4)
    
    # Fill any missing feature columns with defaults
    if feature_columns:
        for col in feature_columns:
            if col not in features:
                if col in ['Role', 'Department']:
                    features[col] = "General"
                else:
                    features[col] = 0
    
    return features


def extract_features_from_dataframe(df, feature_columns):
    """
    Extract features from a dataframe for batch processing.
    
    Args:
        df (pd.DataFrame): Input dataframe
        feature_columns (list): List of feature columns to extract
        
    Returns:
        pd.DataFrame: DataFrame with extracted features
    """
    features_df = pd.DataFrame()
    
    for col in feature_columns:
        if col in df.columns:
            features_df[col] = df[col]
        else:
            features_df[col] = 0
    
    return features_df


def get_skill_summary(features):
    """
    Generate a summary of skills from extracted features.
    
    Args:
        features (dict): Extracted features dictionary
        
    Returns:
        dict: Skill summary with ratings
    """
    total_tech_skills = features.get('prog_lang_count', 0) + features.get('framework_count', 0) + features.get('database_count', 0)
    
    if total_tech_skills >= 8:
        tech_level = "Expert"
    elif total_tech_skills >= 5:
        tech_level = "Intermediate"
    elif total_tech_skills >= 2:
        tech_level = "Beginner"
    else:
        tech_level = "Limited"
    
    return {
        "technical_level": tech_level,
        "total_skills": total_tech_skills,
        "programming_languages": features.get('prog_lang_count', 0),
        "frameworks": features.get('framework_count', 0),
        "databases": features.get('database_count', 0),
        "tools": features.get('tool_count', 0),
        "soft_skills": features.get('soft_skill_count', 0)
    }