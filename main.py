import re
import PyPDF2
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Resume Skill Analyzer", page_icon="⚡", layout="wide")

# Database of skills and jobs
skills_db = [
    "python", "java", "c++", "c#", "javascript", "typescript", "go", "rust", 
    "sql", "machine learning", "deep learning", "data analysis", "numpy", "pandas", "tensorflow", "pytorch", 
    "html", "css", "react", "angular", "vue", "tailwind", 
    "flask", "django", "node.js", "express", "mysql", "mongodb", "rest apis",
    "git", "github", "docker", "kubernetes", "aws", "azure", "linux",
    "power bi", "excel", "tableau", "figma", "ui/ux"
]

job_roles = {
    "Data Scientist": ["python", "sql", "machine learning", "statistics", "data analysis", "pandas", "numpy"],
    "ML Engineer": ["python", "machine learning", "deep learning", "tensorflow", "pytorch", "docker"],
    "Data Analyst": ["python", "sql", "data analysis", "excel", "tableau", "power bi"],
    "Software Engineer": ["python", "java", "c++", "sql", "git"],
    "Full Stack Developer": ["html", "css", "javascript", "react", "node.js", "python", "sql", "git"],
    "Frontend Developer": ["html", "css", "javascript", "react", "tailwind", "git"],
    "Backend Developer": ["python", "sql", "node.js", "express", "mongodb", "docker", "rest apis"],
    "Cloud & DevOps Engineer": ["aws", "docker", "kubernetes", "linux", "python", "git"]
}

learning_resources = {
    "python": "Python for Everybody – Coursera",
    "machine learning": "Andrew Ng Machine Learning Course – Coursera",
    "deep learning": "Deep Learning Specialization – Coursera",
    "tensorflow": "TensorFlow Official Tutorials",
    "sql": "SQL for Data Science – Coursera",
    "react": "React - The Complete Guide – Udemy",
    "node.js": "Node.js Developer Course – Udemy",
    "docker": "Docker Mastery – Udemy",
    "aws": "AWS Certified Solutions Architect"
}

def extract_pdf_text(file):
    reader = PyPDF2.PdfReader(file)
    text = ""
    for page in reader.pages:
        if page.extract_text():
            text += page.extract_text()
    return text

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text

def extract_skills(text):
    found = []
    for skill in skills_db:
        if skill in text:
            found.append(skill)
    return found

def recommend_job(found_skills):
    best_role = None
    best_score = 0
    for role, required in job_roles.items():
        matched = len(set(found_skills) & set(required))
        score = (matched / len(required)) * 100 if required else 0
        if score > best_score:
            best_score = score
            best_role = role
    return best_role, best_score

def similarity_score(resume_text, role):
    job_text = " ".join(job_roles[role])
    documents = [resume_text, job_text]
    vectorizer = TfidfVectorizer(stop_words="english")
    vectors = vectorizer.fit_transform(documents)
    score = cosine_similarity(vectors[0:1], vectors[1:2])[0][0] * 100
    return score

# Sidebar setup
st.sidebar.title("Menu")
app_mode = st.sidebar.radio("Select Mode:", ["Fresher / Job Seeker", "Recruiter / Company"])

uploaded_file = st.sidebar.file_uploader("Upload Resume (PDF)", type=["pdf"])

if app_mode == "Fresher / Job Seeker":
    st.title("🎓 Fresher Resume Readiness Checker")
    st.write("Upload your resume to check your job readiness, find missing skills, and get learning tips.")

    if uploaded_file:
        text = extract_pdf_text(uploaded_file)
        clean_resume = clean_text(text)
        skills = extract_skills(clean_resume)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Extracted Skills")
            st.write(skills if skills else "No skills found.")
        
        with col2:
            st.subheader("Readiness Score")
            score = (len(skills) / len(skills_db)) * 100
            st.metric("Profile Score", f"{score:.2f}%")
            if score < 30:
                st.error("Needs more keywords and skills.")
            elif score < 60:
                st.warning("Average profile. Try adding more tech skills.")
            else:
                st.success("Great job! Your profile looks strong.")

        if skills:
            role, match_score = recommend_job(skills)
            st.subheader(f"Best Matching Role: {role}")
            st.write(f"Match percentage: {match_score:.2f}%")

            required = job_roles[role]
            missing = [s for s in required if s not in skills]
            
            st.subheader("Missing Skills for this Role")
            if missing:
                st.write(missing)
                st.subheader("Learning Suggestions")
                for s in missing:
                    if s in learning_resources:
                        st.write(f"- **{s}**: {learning_resources[s]}")
            else:
                st.success("You have all the required skills for this role!")
    else:
        st.info("Please upload your PDF resume using the sidebar.")

else:
    st.title("🏢 Recruiter Candidate Screening")
    st.write("Check how well a candidate's resume matches a specific software job role.")

    target_role = st.selectbox("Select Job Role:", list(job_roles.keys()))

    if uploaded_file:
        text = extract_pdf_text(uploaded_file)
        clean_resume = clean_text(text)
        skills = extract_skills(clean_resume)
        
        required = job_roles[target_role]
        matched = list(set(skills) & set(required))
        missing = list(set(required) - set(skills))
        
        rule_score = (len(matched) / len(required)) * 100
        ai_score = similarity_score(clean_resume, target_role)

        st.subheader(f"Results for {target_role}")
        c1, c2, c3 = st.columns(3)
        c1.metric("Rule Match", f"{rule_score:.2f}%")
        c2.metric("AI Similarity", f"{ai_score:.2f}%")
        c3.metric("Total Skills Found", len(skills))

        st.write("**Matched Skills:**", matched if matched else "None")
        st.write("**Missing Skills:**", missing if missing else "None")

        avg = (rule_score + ai_score) / 2
        if avg >= 75:
            st.success("Recommendation: Strong fit, highly recommended.")
        elif avg >= 40:
            st.warning("Recommendation: Moderate fit, has some gaps.")
        else:
            st.error("Recommendation: Low fit for this role.")
    else:
        st.info("Please upload a candidate's PDF resume in the sidebar.")
