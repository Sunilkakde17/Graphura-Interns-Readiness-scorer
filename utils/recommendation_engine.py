# utils/recommendation_engine.py
"""
Recommendation engine for generating actionable feedback based on portfolio analysis.
"""


def generate_recommendations(prediction_result, features):
    """
    Generate personalized recommendations based on prediction results.
    
    Args:
        prediction_result (dict): Prediction results from model
        features (dict): Extracted features from resume
        
    Returns:
        dict: Dictionary containing strengths, weaknesses, and recommendations
    """
    strengths = []
    weaknesses = []
    recommendations = []
    
    # Analyze based on confidence score
    score = prediction_result.get('confidence_score', 50)
    label = prediction_result.get('prediction_label', 'Almost Ready')
    
    # ==================== STRENGTHS ANALYSIS ====================
    
    # Technical skills strength
    prog_lang = features.get('prog_lang_count', 0)
    frameworks = features.get('framework_count', 0)
    tools = features.get('tool_count', 0)
    databases = features.get('database_count', 0)
    
    if prog_lang >= 3:
        strengths.append(f"Strong programming skills with {prog_lang}+ languages")
    elif prog_lang >= 2:
        strengths.append("Good foundation in programming languages")
    
    if frameworks >= 3:
        strengths.append(f"Experience with {frameworks} different frameworks")
    elif frameworks >= 2:
        strengths.append("Knowledge of relevant frameworks")
    
    if tools >= 5:
        strengths.append("Excellent tool proficiency across development stack")
    elif tools >= 3:
        strengths.append("Good practical tool experience")
    
    # Projects strength
    num_projects = features.get('num_projects', 0)
    if num_projects >= 4:
        strengths.append(f"Strong project portfolio with {num_projects}+ projects")
    elif num_projects >= 2:
        strengths.append("Good number of projects demonstrating practical skills")
    
    # Experience strength
    exp_years = features.get('exp_years', 0)
    internship = features.get('internship_present', 0)
    
    if exp_years >= 2:
        strengths.append(f"{exp_years} years of relevant experience")
    elif internship:
        strengths.append("Internship experience available")
    
    # Certifications strength
    cert_count = features.get('certification_count', 0)
    if cert_count >= 3:
        strengths.append(f"{cert_count} professional certifications completed")
    elif cert_count >= 1:
        strengths.append("Industry certifications obtained")
    
    # GitHub/LinkedIn presence
    if features.get('github_present', 0):
        strengths.append("GitHub portfolio available for code review")
    if features.get('linkedin_present', 0):
        strengths.append("Professional LinkedIn profile present")
    
    # Section completeness
    sections_present = sum([
        features.get('projects_section_present', 0),
        features.get('skills_section_present', 0),
        features.get('experience_section_present', 0)
    ])
    if sections_present >= 3:
        strengths.append("Well-structured resume with complete sections")
    elif sections_present >= 2:
        strengths.append("Good resume structure with key sections")
    
    # ==================== WEAKNESSES ANALYSIS ====================
    
    # Technical skills gaps
    if prog_lang < 2:
        weaknesses.append("Limited programming language exposure")
    elif prog_lang == 0:
        weaknesses.append("No programming languages mentioned")
    
    if frameworks < 1:
        weaknesses.append("No frameworks listed in technical skills")
    
    if tools < 2:
        weaknesses.append("Limited tool/technology experience mentioned")
    
    # Project gaps
    if num_projects < 2:
        weaknesses.append("Few projects mentioned in portfolio")
        recommendations.append("Add at least 2-3 quality projects to showcase your skills")
    
    # Experience gaps
    if not internship and exp_years < 1:
        weaknesses.append("No internship or work experience mentioned")
        recommendations.append("Consider internships, freelance work, or open source contributions")
    
    # Certification gaps
    if cert_count < 2:
        weaknesses.append("Few professional certifications completed")
        recommendations.append("Complete industry-recognized certifications relevant to your role")
    
    # Section gaps
    if not features.get('projects_section_present', 0):
        weaknesses.append("Projects section missing from resume")
        recommendations.append("Add a dedicated projects section with detailed descriptions")
    
    if not features.get('skills_section_present', 0):
        weaknesses.append("Skills section missing")
        recommendations.append("Create a comprehensive skills section with categorized technical skills")
    
    if not features.get('github_present', 0):
        weaknesses.append("GitHub profile not linked")
        recommendations.append("Create a GitHub profile and link it to showcase your code")
    
    if not features.get('linkedin_present', 0):
        weaknesses.append("LinkedIn profile not linked")
        recommendations.append("Create/update LinkedIn profile for professional networking")
    
    # ==================== ROLE-SPECIFIC RECOMMENDATIONS ====================
    
    role = features.get('Role', 'General')
    department = features.get('Department', 'Technical')
    
    if department == "Technical":
        if prog_lang < 3:
            recommendations.append("Learn additional programming languages relevant to your target role")
        if frameworks < 2:
            recommendations.append("Master at least 2-3 popular frameworks in your domain")
        recommendations.append("Contribute to open source projects to build portfolio")
    else:
        recommendations.append("Build a portfolio of work samples showcasing your skills")
        recommendations.append("Get certified in relevant tools and platforms")
    
    # ==================== SCORE-BASED RECOMMENDATIONS ====================
    
    if label == "Job Ready":
        recommendations.append("Ready for job applications - focus on networking and interview preparation")
        recommendations.append("Consider targeting mid-level roles based on your skill level")
    elif label == "Almost Ready":
        recommendations.append("Complete remaining certifications and add more projects to become job ready")
        recommendations.append("Practice technical interviews and work on communication skills")
    else:
        recommendations.append("Focus on building foundational skills first")
        recommendations.append("Consider internships or entry-level positions to gain experience")
        recommendations.append("Complete beginner-level certifications and courses")
    
    # Ensure we have at least 3 strengths and 3 weaknesses
    if len(strengths) < 3:
        default_strengths = ["Good resume structure", "Basic technical skills present", "Projects mentioned in portfolio"]
        strengths.extend(default_strengths[:3 - len(strengths)])
    
    if len(weaknesses) < 3:
        default_weaknesses = ["Resume could be more detailed", "Consider adding more metrics", "Work on skill documentation"]
        weaknesses.extend(default_weaknesses[:3 - len(weaknesses)])
    
    if len(recommendations) < 4:
        default_recs = [
            "Update your resume with quantitative achievements",
            "Add links to deployed projects when possible",
            "Network with professionals in your target industry",
            "Practice coding challenges regularly"
        ]
        recommendations.extend(default_recs[:4 - len(recommendations)])
    
    return {
        "strengths": strengths[:6],
        "weaknesses": weaknesses[:6],
        "recommendations": recommendations[:8],
        "score": score,
        "readiness_level": label
    }


def generate_comparison_insights(results):
    """
    Generate insights comparing multiple resumes.
    
    Args:
        results (list): List of prediction results for comparison
        
    Returns:
        dict: Comparison insights
    """
    if len(results) < 2:
        return {"error": "Need at least 2 resumes for comparison"}
    
    # Sort by confidence score
    sorted_results = sorted(results, key=lambda x: x.get('confidence_score', 0), reverse=True)
    
    best_resume = sorted_results[0] if sorted_results else None
    worst_resume = sorted_results[-1] if sorted_results else None
    
    # Calculate averages
    avg_score = sum(r.get('confidence_score', 0) for r in results) / len(results)
    
    # Find common strengths and weaknesses across all
    all_strengths = []
    all_weaknesses = []
    
    for result in results:
        features = result.get('features', {})
        if features:
            if features.get('prog_lang_count', 0) >= 3:
                all_strengths.append("Multiple programming languages")
            if features.get('github_present', 0):
                all_strengths.append("GitHub presence")
            if features.get('projects_section_present', 0):
                all_strengths.append("Projects section")
            
            if features.get('prog_lang_count', 0) < 2:
                all_weaknesses.append("Limited programming languages")
            if not features.get('github_present', 0):
                all_weaknesses.append("No GitHub link")
    
    # Get unique strengths/weaknesses
    common_strengths = list(set(all_strengths))[:3]
    common_weaknesses = list(set(all_weaknesses))[:3]
    
    # Generate comparison text
    comparison_text = f"""
    === Resume Comparison Analysis ===
    
    Total Resumes Compared: {len(results)}
    Average Readiness Score: {avg_score:.1f}
    
    🏆 Best Resume: {best_resume.get('filename', 'Unknown') if best_resume else 'N/A'}
       Score: {best_resume.get('confidence_score', 0) if best_resume else 0}
       Level: {best_resume.get('readiness_level', 'N/A') if best_resume else 'N/A'}
    
    📉 Needs Most Improvement: {worst_resume.get('filename', 'Unknown') if worst_resume else 'N/A'}
       Score: {worst_resume.get('confidence_score', 0) if worst_resume else 0}
       Level: {worst_resume.get('readiness_level', 'N/A') if worst_resume else 'N/A'}
    
    📊 Score Distribution:
    """
    
    for i, result in enumerate(sorted_results, 1):
        comparison_text += f"\n   {i}. {result.get('filename', 'Unknown')[:30]} - {result.get('confidence_score', 0)} ({result.get('readiness_level', 'N/A')})"
    
    if common_strengths:
        comparison_text += f"\n\n✅ Common Strengths: {', '.join(common_strengths)}"
    
    if common_weaknesses:
        comparison_text += f"\n\n⚠️ Common Improvement Areas: {', '.join(common_weaknesses)}"
    
    return {
        "best_resume": best_resume,
        "worst_resume": worst_resume,
        "average_score": avg_score,
        "common_strengths": common_strengths,
        "common_weaknesses": common_weaknesses,
        "comparison_text": comparison_text,
        "rankings": sorted_results
    }