from flask import Flask, request, jsonify
from flask_cors import CORS
import fitz
import joblib
import numpy as np
import re
import os

# ==========================================================
# Flask App
# ==========================================================

app = Flask(__name__)
CORS(app)

# ==========================================================
# Load Trained Models
# ==========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

model = joblib.load(os.path.join(MODEL_DIR, "career_model.pkl"))
vectorizer = joblib.load(os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"))
feature_selector = joblib.load(os.path.join(MODEL_DIR, "feature_selector.pkl"))
label_encoder = joblib.load(os.path.join(MODEL_DIR, "label_encoder.pkl"))

print("✅ Models Loaded Successfully")

# ==========================================================
# Skills Dictionary
# ==========================================================

SKILLS = [
    # Languages
    "python", "java", "c", "c++", "sql", "r",
    # Web Technologies
    "html", "css", "javascript", "typescript",
    # Frontend
    "react", "angular", "vue",
    # Backend
    "nodejs", "express", "flask", "django", "fastapi",
    "spring", "spring boot", ".net", "junit",
    # Databases
    "mysql", "postgresql", "oracle", "oracle sql", "mongodb",
    # Cloud & DevOps
    "aws", "azure", "gcp", "docker", "kubernetes",
    # Version Control & Tools
    "git", "github", "jira", "postman", "vs code", "intellij idea",
    "jupyter notebook", "google colab", "mlflow", "google sheets",
    "google data studio",
    # Machine Learning
    "machine learning", "deep learning", "tensorflow", "keras",
    "scikit-learn", "xgboost", "opencv", "nlp",
    # Data Analysis
    "pandas", "numpy", "scipy", "matplotlib", "seaborn", "power bi",
    "tableau", "excel", "pivot tables", "vlookup", "power query", "dax",
    # Concepts
    "statistics", "probability", "feature engineering", "model evaluation",
    "supervised learning", "unsupervised learning", "a/b testing",
    "data cleaning", "descriptive statistics", "dashboarding",
    "kpi reporting", "oop", "object oriented programming",
    "data structures", "algorithms", "data structures and algorithms",
    "sdlc", "design patterns", "agile", "scrum",
    # Others
    "rest api", "streamlit"
]

# ==========================================================
# Job Role / Title Dictionary
# ==========================================================

ROLE_TITLES = [
    "data scientist intern", "software engineer intern", "data analyst intern",
    "machine learning engineer intern", "research intern",
    "data scientist", "data analyst", "data engineer", "ml engineer",
    "ai engineer", "software engineer", "software developer",
    "backend developer", "frontend developer", "full stack developer",
    "full stack engineer", "web developer", "mobile developer",
    "machine learning engineer", "business analyst", "business intelligence analyst",
    "quantitative analyst", "product manager", "project manager",
    "program manager", "devops engineer", "cloud engineer", "site reliability engineer",
    "qa engineer", "test engineer", "systems engineer",
    "system administrator", "database administrator", "research scientist",
    "research assistant", "teaching assistant", "consultant", "analyst",
    "intern"
]

# ==========================================================
# Extract Resume Text
# ==========================================================

def extract_text(pdf_file):
    text = ""
    pdf = fitz.open(stream=pdf_file.read(), filetype="pdf")
    for page in pdf:
        text += page.get_text()
    pdf.close()
    text = re.sub(r"\s+", " ", text)
    return text.lower().strip()

# ==========================================================
# Extract Skills
# ==========================================================

def extract_skills(text):
    skills = []
    for skill in SKILLS:
        if re.search(r"\b" + re.escape(skill) + r"\b", text):
            skills.append(skill.title())
    return sorted(list(set(skills)))

# ==========================================================
# Extract Roles / Job Titles
# ==========================================================

def extract_roles(text):
    roles = []
    for role in ROLE_TITLES:
        if re.search(r"\b" + re.escape(role) + r"\b", text):
            roles.append(role.title())
    # Drop shorter roles that are substrings of a longer matched role
    # (e.g. don't show "Analyst" separately if "Data Analyst" was found)
    roles = sorted(set(roles), key=len, reverse=True)
    filtered = []
    for r in roles:
        if not any(r.lower() in other.lower() and r != other for other in filtered):
            filtered.append(r)
    return sorted(filtered)

# ==========================================================
# Extract Education Entities (degrees + institutions)
# ==========================================================

def extract_education_entities(text):
    entities = set()

    degree_pattern = re.compile(
        r"\b(b\.?\s?tech|m\.?\s?tech|b\.?\s?e\.?|m\.?\s?s\.?|b\.?\s?s\.?|mca|bca|phd|"
        r"bachelor(?:'s)?(?:\s+of\s+[a-z]+(?:\s+[a-z]+)?)?|"
        r"master(?:'s)?(?:\s+of\s+[a-z]+(?:\s+[a-z]+)?)?)\b",
        re.IGNORECASE
    )
    for m in degree_pattern.finditer(text):
        cleaned = re.sub(r"\s+", " ", m.group(0)).strip()
        if len(cleaned) > 1:
            entities.add(cleaned.title())

    uni_pattern = re.compile(
        r"\b(?:university|institute(?:\s+of\s+technology)?|college)\s+of\s+[a-z]+(?:,\s*[a-z]+)?|"
        r"\b[a-z]+(?:\s+[a-z]+){0,3}\s+(?:university|institute of technology|college)\b",
        re.IGNORECASE
    )
    for m in uni_pattern.finditer(text):
        cleaned = re.sub(r"\s+", " ", m.group(0)).strip()
        if len(cleaned) > 4:
            entities.add(cleaned.title())

    field_pattern = re.compile(
        r"\b(computer science|data science|information technology|electronics|"
        r"electrical engineering|mechanical engineering|mathematics|statistics)\b",
        re.IGNORECASE
    )
    for m in field_pattern.finditer(text):
        entities.add(m.group(0).strip().title())

    return sorted(entities, key=len, reverse=True)

# ==========================================================
# Extract Summary
# ==========================================================

def extract_summary(text):
    sentences = re.split(r"[.!?]", text)
    summary = []
    for sentence in sentences:
        sentence = sentence.strip()
        if len(sentence.split()) > 8:
            summary.append(sentence)
        if len(summary) == 3:
            break
    if summary:
        return ". ".join(summary)
    return "Summary not available."

# ==========================================================
# Extract Experience
# ==========================================================

def extract_experience(text):
    pattern = re.compile(
        r"experience(.*?)(education|skills|projects|certifications|references|$)",
        re.IGNORECASE | re.DOTALL
    )
    match = pattern.search(text)
    if match:
        exp = match.group(1)
        exp = re.sub(r"\s+", " ", exp)
        return exp[:700]
    years = re.findall(r"\d+\+?\s+years?", text)
    if years:
        return ", ".join(years)
    return "Experience not found."

# ==========================================================
# Extract Education
# ==========================================================

def extract_education(text):
    pattern = re.compile(
        r"education(.*?)(experience|skills|projects|certifications|references|$)",
        re.IGNORECASE | re.DOTALL
    )
    match = pattern.search(text)
    if match:
        edu = match.group(1)
        edu = re.sub(r"\s+", " ", edu)
        return edu[:600]
    keywords = [
        "b.tech", "b.e", "bachelor", "master", "m.tech", "mca", "bca",
        "phd", "computer science"
    ]
    found = []
    for word in keywords:
        if word in text:
            found.append(word.upper())
    if found:
        return ", ".join(found)
    return "Education not found."

# ==========================================================
# Prediction Endpoint
# ==========================================================

@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "resume" not in request.files:
            return jsonify({"error": "Resume file not found."}), 400

        file = request.files["resume"]

        if file.filename == "":
            return jsonify({"error": "No file selected."}), 400

        text = extract_text(file)

        summary = extract_summary(text)
        experience = extract_experience(text)
        education = extract_education(text)
        skills = extract_skills(text)
        roles = extract_roles(text)
        education_entities = extract_education_entities(text)

        X = vectorizer.transform([text])
        X = feature_selector.transform(X)

        probabilities = model.predict_proba(X)[0]
        top3 = np.argsort(probabilities)[::-1][:3]

        recommendations = []
        confidence = []

        for idx in top3:
            role = label_encoder.inverse_transform([idx])[0]
            recommendations.append(role)
            confidence.append(round(float(probabilities[idx] * 100), 2))

        return jsonify({
            "summary": summary,
            "experience": experience,
            "education": education,
            "skills": skills,
            "roles": roles,
            "education_entities": education_entities,
            "recommendations": recommendations,
            "confidence": confidence
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ==========================================================
# Home Route
# ==========================================================

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "success",
        "message": "AI Career Recommendation API Running",
        "model": "Logistic Regression"
    })

# ==========================================================
# Health Check
# ==========================================================

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy"})

# ==========================================================
# Run Server
# ==========================================================

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)