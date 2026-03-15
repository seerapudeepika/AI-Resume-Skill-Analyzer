# AI-Based Resume Skill Analyzer
# Author: Deepika

import re
import PyPDF2
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -------------------------------
# Skill Database
# -------------------------------

skills_db = [
    "python","java","sql","machine learning","deep learning",
    "data analysis","numpy","pandas","tensorflow","statistics",
    "html","css","javascript","git","github",
    "flask","django","c++","mysql","mongodb",
    "power bi","excel","tableau","nlp","computer vision"
]

# -------------------------------
# Job Role Skills
# -------------------------------

job_roles = {
    "Data Scientist": ["python","sql","machine learning","statistics","data analysis"],
    "ML Engineer": ["python","machine learning","deep learning","tensorflow"],
    "Data Analyst": ["python","sql","data analysis","statistics","excel"],
    "Software Engineer": ["python","java","sql","git"],
    "Full Stack Developer": ["html","css","javascript","python","sql"],
    "AI Engineer": ["python","machine learning","deep learning","tensorflow","nlp"],
    "Backend Developer": ["python","sql","flask","django"],
    "Frontend Developer": ["html","css","javascript"]
}

# -------------------------------
# Learning Resources
# -------------------------------

learning_resources = {
    "python": "Python for Everybody – Coursera",
    "machine learning": "Andrew Ng Machine Learning Course – Coursera",
    "deep learning": "Deep Learning Specialization – Coursera",
    "tensorflow": "TensorFlow Official Tutorials",
    "data analysis": "Google Data Analytics Certificate",
    "statistics": "Khan Academy Statistics Course",
    "sql": "SQL for Data Science – Coursera",
    "excel": "Excel Skills for Business – Coursera",
    "html": "HTML & CSS Web Design – freeCodeCamp",
    "javascript": "JavaScript Algorithms – freeCodeCamp",
    "flask": "Flask Web Development – Udemy",
    "django": "Django for Beginners – Udemy"
}

# -------------------------------
# Extract Text from PDF
# -------------------------------

def extract_pdf_text(file):

    reader = PyPDF2.PdfReader(file)
    text = ""

    for page in reader.pages:
        if page.extract_text():
            text += page.extract_text()

    return text


# -------------------------------
# Clean Text
# -------------------------------

def clean_text(text):

    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)

    return text


# -------------------------------
# Skill Extraction
# -------------------------------

def extract_skills(text):

    found_skills = []

    for skill in skills_db:
        if skill in text:
            found_skills.append(skill)

    return found_skills


# -------------------------------
# Job Recommendation
# -------------------------------

def recommend_job(found_skills):

    best_role = None
    best_score = 0

    for role, required in job_roles.items():

        matched = len(set(found_skills) & set(required))
        score = (matched / len(required)) * 100

        if score > best_score:
            best_score = score
            best_role = role

    return best_role, best_score


# -------------------------------
# Skill Gap Detection
# -------------------------------

def skill_gap(role, skills):

    required = job_roles[role]

    missing = []

    for skill in required:
        if skill not in skills:
            missing.append(skill)

    return missing


# -------------------------------
# AI Similarity Score
# -------------------------------

def similarity_score(resume_text, role):

    job_text = " ".join(job_roles[role])

    documents = [resume_text, job_text]

    vectorizer = TfidfVectorizer(stop_words="english")

    vectors = vectorizer.fit_transform(documents)

    score = cosine_similarity(vectors[0], vectors[1])[0][0] * 100

    return score


# -------------------------------
# Streamlit Interface
# -------------------------------

st.title("🤖 AI Resume Skill Analyzer")

st.write("Upload your resume and get skill analysis, job recommendations, and learning suggestions.")

uploaded_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])

if uploaded_file:

    text = extract_pdf_text(uploaded_file)

    clean_resume = clean_text(text)

    skills = extract_skills(clean_resume)

    # -------------------------------
    # Detected Skills
    # -------------------------------

    st.subheader("🧠 Detected Skills")
    st.write(skills)

    # -------------------------------
    # Skills Chart
    # -------------------------------

    if skills:
        st.subheader("📊 Skills Chart")
        skill_data = {skill:1 for skill in skills}
        st.bar_chart(skill_data)

    # -------------------------------
    # Job Recommendation
    # -------------------------------

    role, match_score = recommend_job(skills)

    st.subheader("🎯 Recommended Job Role")
    st.write(role)

    st.subheader("📈 Rule Based Match Score")
    st.write(f"{match_score:.2f}%")

    # -------------------------------
    # AI Similarity Score
    # -------------------------------

    ai_score = similarity_score(clean_resume, role)

    st.subheader("🤖 AI Similarity Score")
    st.write(f"{ai_score:.2f}%")

    # -------------------------------
    # Missing Skills
    # -------------------------------

    missing = skill_gap(role, skills)

    st.subheader("⚠ Missing Skills")
    st.write(missing)

    # -------------------------------
    # Learning Recommendations
    # -------------------------------

    st.subheader("📚 Learning Recommendations")

    for skill in missing:
        if skill in learning_resources:
            st.write(f"{skill} → {learning_resources[skill]}")

    # -------------------------------
    # Resume Improvement Score
    # -------------------------------

    st.subheader("⭐ Resume Improvement Score")

    improvement = (len(skills) / len(skills_db)) * 100

    st.write(f"{improvement:.2f}%")

    if improvement < 50:
        st.error("Your resume needs improvement.")
    elif improvement < 75:
        st.warning("Your resume is average. Improve missing skills.")
    else:
        st.success("Your resume is strong!")
