# DecodeLabs AI Internship - Project 3: AI Recommendation Logic
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

print("=" * 60)
print(" DecodeLabs AI - Tech Stack & Career Recommender ")
print("=" * 60)

# Step 1: Load the Dataset
try:
    df = pd.read_csv('raw_skills.csv')
except FileNotFoundError:
    # Embedded fallback data if raw_skills.csv is missing
    data = {
        'Role': [
            'Data Scientist', 'DevOps Engineer', 'Backend Developer',
            'Frontend Developer', 'Mobile App Developer', 'AI Engineer',
            'Cybersecurity Analyst', 'Cloud Architect'
        ],
        'Skills': [
            'Python machine learning SQL data analysis statistics pandas numpy scikit-learn',
            'AWS Docker Kubernetes CI/CD Linux automation cloud Git Terraform',
            'Python Java SQL APIs PostgreSQL Django FastAPI backend REST',
            'JavaScript React HTML CSS TypeScript web design UI frontend Vue',
            'Flutter Dart Android iOS Swift React Native mobile APIs',
            'Python PyTorch TensorFlow deep learning neural networks NLP computer vision LLM',
            'Linux networking security penetration testing Python encryption ethical hacking',
            'AWS Azure cloud Docker Kubernetes microservices networking security'
        ]
    }
    df = pd.DataFrame(data)

# Step 2: Ingestion - Collect at least 3 skills from the user
print("\nEnter at least 3 skills or interests (separated by commas).")
print("Example: Python, Machine Learning, Data Analysis")
print("Example: AWS, Docker, Automation\n")

user_input = input("Enter your skills: ").strip()

# Cold start / empty input handling
if not user_input:
    user_input = "Python Machine Learning Cloud"
    print(f"\n[Notice] No input provided. Using default profile: '{user_input}'")

# Step 3: Vector Mapping & Scoring via TF-IDF + Cosine Similarity
# Combine job skill descriptions with user input to create a shared vocabulary space
corpus = df['Skills'].tolist() + [user_input]

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(corpus)

# The user vector is the last row in the matrix; job profiles are all preceding rows
job_vectors = tfidf_matrix[:-1]
user_vector = tfidf_matrix[-1:]

# Calculate Cosine Similarity between user profile and all job roles
similarity_scores = cosine_similarity(user_vector, job_vectors).flatten()

# Step 4: Sorting & Filtering (Top-3 Recommendations)
df['Match_Score'] = similarity_scores
top_recommendations = df.sort_values(by='Match_Score', ascending=False).head(3)

print("\n" + "=" * 60)
print(" Top 3 Recommended Career Paths ")
print("=" * 60)

for rank, (_, row) in enumerate(top_recommendations.iterrows(), start=1):
    score_pct = row['Match_Score'] * 100
    print(f"{rank}. {row['Role']} — Match Score: {score_pct:.2f}%")
    print(f"   Required Profile: {row['Skills']}")
    print("-" * 60)