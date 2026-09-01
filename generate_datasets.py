"""
Synthetic Benchmark Dataset Generator for Hiring System
Generates comprehensive candidate databases across diverse seniority levels, demographics, and technical competencies.
"""

import random
import csv
import os

NAMES = {
    ("Male", "Asian"): ["Aarav Sharma", "Wei Zhang", "Rohan Gupta", "Kenji Sato", "Jin Woo", "Aditya Verma", "Tatsuya Mori"],
    ("Female", "Asian"): ["Priya Patel", "Mei Ling", "Ananya Rao", "Yuki Tanaka", "Sun-hee Park", "Kavita Krishnan", "Lin Chen"],
    ("Male", "White"): ["David Miller", "James Wilson", "Alexander Smith", "Michael Brown", "Lucas Bennett", "Erik Lindqvist", "Thomas Clark"],
    ("Female", "White"): ["Sarah Johnson", "Emily Davis", "Olivia Taylor", "Emma Anderson", "Sophie Muller", "Elena Rostova", "Chloe Wright"],
    ("Male", "Black"): ["Marcus Washington", "Jamal Harris", "Malik Jackson", "Kofi Mensah", "Darius Robinson", "Kwame Asante", "Isaiah Brooks"],
    ("Female", "Black"): ["Keisha Washington", "Aaliyah Harris", "Ebony Jackson", "Zendaya Mensah", "Imani Robinson", "Nia Asante", "Amara Okafor"],
    ("Male", "Hispanic"): ["Carlos Gomez", "Mateo Rodriguez", "Alejandro Morales", "Diego Fernandez", "Gabriel Torres", "Javier Ramos", "Santiago Silva"],
    ("Female", "Hispanic"): ["Maria Gomez", "Sofia Rodriguez", "Valentina Morales", "Isabella Fernandez", "Camila Torres", "Lucia Ramos", "Elena Silva"],
    ("Male", "Middle Eastern"): ["Omar Al-Mansoor", "Tariq Siddiqui", "Bilal Zahra", "Youssef Hassan", "Farhan Qureshi", "Zayd Karim", "Mustafa Demir"],
    ("Female", "Middle Eastern"): ["Amina Al-Mansoor", "Zainab Siddiqui", "Fatima Zahra", "Layla Hassan", "Noor Qureshi", "Mariam Karim", "Yasmin Demir"]
}

ROLES = [
    {
        "role": "Senior ML Engineer",
        "skills": "Python; PyTorch; TensorFlow; Distributed Systems; MLOps; CUDA; SQL",
        "min_exp": 4, "max_exp": 10, "base_salary": 140
    },
    {
        "role": "Full Stack Software Engineer",
        "skills": "JavaScript; TypeScript; React; Node.js; PostgreSQL; GraphQL; Docker",
        "min_exp": 2, "max_exp": 8, "base_salary": 115
    },
    {
        "role": "Data Analyst / BI Specialist",
        "skills": "SQL; Python; Tableau; PowerBI; Pandas; Data Warehousing; Snowflake",
        "min_exp": 1, "max_exp": 6, "base_salary": 85
    },
    {
        "role": "Cloud DevOps & Platform Engineer",
        "skills": "Go; Kubernetes; Terraform; Docker; AWS; Prometheus; CI/CD pipelines",
        "min_exp": 3, "max_exp": 9, "base_salary": 130
    },
    {
        "role": "Junior Software Developer",
        "skills": "Python; Flask; Git; SQLite; HTML/CSS; JavaScript; OOP Basics",
        "min_exp": 0, "max_exp": 3, "base_salary": 70
    },
    {
        "role": "Principal AI Research Scientist",
        "skills": "Transformers; Multi-Modal AI; Reinforcement Learning; PyTorch; C++; Distributed GPU Training",
        "min_exp": 6, "max_exp": 14, "base_salary": 190
    },
    {
        "role": "Engineering Manager",
        "skills": "Agile Leadership; System Design; Team Mentorship; Cloud Architecture; Python; Java; Budgets",
        "min_exp": 7, "max_exp": 16, "base_salary": 175
    }
]

DEGREES = [
    "B.S. Computer Science", "M.S. Computer Science", "B.S. Software Engineering",
    "M.S. Software Engineering", "B.S. Data Science", "M.S. Artificial Intelligence",
    "Ph.D. Computer Science", "B.S. Electrical & Computer Engineering", "M.S. Statistics"
]

UNIVERSITIES = [
    ("Tier 1 - Top Global (Stanford / MIT / Berkeley / IIT / Cambridge)", 1.0),
    ("Tier 2 - State Flagship / Reputed Tech Institute", 0.9),
    ("Tier 3 - Regional Accredited University", 0.8)
]

def generate_candidates(n=100, output_path="data/hiring_master.csv", seed=42):
    random.seed(seed)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    genders = ["Male", "Female"]
    religions = ["Hindu", "Christian", "Muslim", "Jewish", "Catholic", "Buddhist", "Agnostic"]
    languages = ["Fluent", "Basic", "Native"]
    ethnicities = ["Asian", "White", "Black", "Hispanic", "Middle Eastern"]
    age_groups = ["18-24", "25-34", "35-44", "45-54"]
    
    records = []
    
    for i in range(1, n + 1):
        cand_id = f"C{i:03d}"
        gender = random.choice(genders)
        ethnicity = random.choice(ethnicities)
        religion = random.choice(religions)
        language = random.choice(languages)
        age_group = random.choice(age_groups)
        
        name_pool = NAMES.get((gender, ethnicity), [f"Candidate {i}"])
        name = random.choice(name_pool)
        
        role_info = random.choice(ROLES)
        role = role_info["role"]
        
        if age_group == "18-24":
            exp = random.randint(0, min(2, role_info["max_exp"]))
        elif age_group == "25-34":
            exp = random.randint(min(2, role_info["min_exp"]), min(8, role_info["max_exp"]))
        elif age_group == "35-44":
            exp = random.randint(max(4, role_info["min_exp"]), role_info["max_exp"] + 2)
        else:
            exp = random.randint(max(7, role_info["min_exp"]), role_info["max_exp"] + 5)
            
        skills = role_info["skills"]
        education = random.choice(DEGREES)
        uni_tier, uni_weight = random.choice(UNIVERSITIES)
        
        base_interview = random.randint(68, 96)
        interview_score = min(100, max(50, round(base_interview + (exp * 0.7) + (uni_weight * 3))))
        prior_salary = round(role_info["base_salary"] + (exp * 6.5) + random.uniform(-8, 12))
        github_contributions = random.randint(15, 850)
        certifications_count = random.randint(0, 4)
        
        records.append({
            "candidate_id": cand_id,
            "name": name,
            "gender": gender,
            "religion": religion,
            "language": language,
            "ethnicity": ethnicity,
            "age_group": age_group,
            "education": education,
            "university_tier": uni_tier.split(" - ")[0],
            "experience_years": exp,
            "technical_skills": skills,
            "interview_score": interview_score,
            "github_contributions": github_contributions,
            "certifications_count": certifications_count,
            "prior_salary_k": prior_salary,
            "expected_role": role
        })

    fieldnames = list(records[0].keys())
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
        
    print(f"Generated {len(records)} candidate records at '{output_path}'")
    return output_path

if __name__ == "__main__":
    generate_candidates(n=150, output_path="data/hiring_master.csv", seed=42)
    generate_candidates(n=60, output_path="data/hiring_tech.csv", seed=101)
    generate_candidates(n=40, output_path="data/hiring_leadership.csv", seed=202)
    generate_candidates(n=20, output_path="data/hiring_demo.csv", seed=303)
