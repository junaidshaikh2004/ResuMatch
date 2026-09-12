"""
A curated bank of skill keywords, organized by job role.

Two things use this file:
  1. generate_dataset.py, which builds synthetic resumes/JDs out of these
     skill lists for each role.
  2. features.py, which matches these same keywords against real resume/JD
     text at inference time to compute skill overlap and missing skills.

This is intentionally a simple keyword list rather than a trained NER model
— it's transparent, fast, and easy to extend, which matters more here than
catching every possible synonym.
"""

ROLE_SKILLS = {
    "Data Scientist": [
        "python", "pandas", "numpy", "scikit-learn", "machine learning",
        "statistics", "sql", "data visualization", "jupyter", "a/b testing",
        "feature engineering", "regression", "classification",
    ],
    "Backend Developer": [
        "python", "django", "flask", "rest api", "sql", "postgresql",
        "docker", "git", "unit testing", "microservices", "redis", "celery",
    ],
    "Frontend Developer": [
        "javascript", "react", "html", "css", "typescript", "redux",
        "webpack", "responsive design", "rest api", "git", "accessibility",
    ],
    "DevOps Engineer": [
        "docker", "kubernetes", "aws", "ci/cd", "terraform", "linux",
        "jenkins", "monitoring", "ansible", "bash", "networking",
    ],
    "Machine Learning Engineer": [
        "python", "pytorch", "tensorflow", "machine learning", "docker",
        "mlops", "model deployment", "sql", "feature engineering", "aws",
        "deep learning",
    ],
    "Data Analyst": [
        "sql", "excel", "power bi", "tableau", "python", "data visualization",
        "statistics", "reporting", "pandas", "a/b testing",
    ],
    "QA Engineer": [
        "test automation", "selenium", "manual testing", "python",
        "test case design", "regression testing", "jira", "ci/cd",
        "api testing",
    ],
    "Mobile Developer": [
        "swift", "kotlin", "android", "ios", "react native", "flutter",
        "rest api", "git", "mobile ui design",
    ],
    "Cloud Engineer": [
        "aws", "azure", "gcp", "terraform", "docker", "kubernetes",
        "networking", "linux", "ci/cd", "cloud security",
    ],
    "Product Manager": [
        "product roadmap", "agile", "scrum", "stakeholder management",
        "user research", "data analysis", "jira", "product strategy",
        "communication",
    ],
    "UI/UX Designer": [
        "figma", "wireframing", "user research", "prototyping",
        "interaction design", "usability testing", "adobe xd",
        "design systems",
    ],
    "Cybersecurity Analyst": [
        "network security", "siem", "penetration testing", "incident response",
        "risk assessment", "firewalls", "python", "compliance", "linux",
    ],
}

# General/soft skills that show up across roles and are worth detecting too,
# even though they don't drive the role-specific "required skills" list.
GENERAL_SKILLS = [
    "communication", "leadership", "teamwork", "problem solving",
    "project management", "time management", "mentoring",
]

# Flattened, de-duplicated list used by features.py to scan arbitrary text.
ALL_SKILLS = sorted(set(
    skill
    for skills in ROLE_SKILLS.values()
    for skill in skills
) | set(GENERAL_SKILLS))
