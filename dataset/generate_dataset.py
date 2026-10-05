"""Generate a realistic disease-symptom dataset"""
import pandas as pd
import numpy as np
import random

random.seed(42)
np.random.seed(42)

diseases = {
    "Influenza": ["fever", "cough", "headache", "body_aches", "fatigue", "chills", "sore_throat"],
    "Common Cold": ["runny_nose", "sneezing", "sore_throat", "cough", "mild_fever", "nasal_congestion"],
    "COVID-19": ["fever", "cough", "fatigue", "loss_of_taste", "loss_of_smell", "shortness_of_breath", "headache"],
    "Pneumonia": ["fever", "cough", "shortness_of_breath", "chest_pain", "fatigue", "chills", "sweating"],
    "Bronchitis": ["cough", "chest_pain", "fatigue", "shortness_of_breath", "mild_fever", "mucus"],
    "Malaria": ["fever", "chills", "sweating", "headache", "nausea", "vomiting", "body_aches"],
    "Dengue": ["fever", "severe_headache", "rash", "joint_pain", "muscle_pain", "nausea", "vomiting"],
    "Typhoid": ["fever", "stomach_pain", "headache", "weakness", "constipation", "rash", "loss_of_appetite"],
    "Jaundice": ["yellowing_of_skin", "dark_urine", "fatigue", "nausea", "abdominal_pain", "loss_of_appetite"],
    "Diabetes": ["frequent_urination", "excessive_thirst", "fatigue", "blurred_vision", "slow_healing", "weight_loss"],
    "Hypertension": ["headache", "dizziness", "blurred_vision", "chest_pain", "shortness_of_breath", "nosebleeds"],
    "Gastroenteritis": ["nausea", "vomiting", "diarrhea", "stomach_cramps", "fever", "loss_of_appetite"],
    "Migraine": ["severe_headache", "nausea", "vomiting", "light_sensitivity", "sound_sensitivity", "blurred_vision"],
    "Asthma": ["shortness_of_breath", "wheezing", "chest_tightness", "cough", "difficulty_breathing"],
    "Appendicitis": ["abdominal_pain", "nausea", "vomiting", "fever", "loss_of_appetite", "bloating"],
    "UTI": ["frequent_urination", "burning_urination", "pelvic_pain", "cloudy_urine", "strong_odor_urine", "fever"],
    "Chickenpox": ["rash", "fever", "itching", "fatigue", "headache", "loss_of_appetite", "blister"],
    "Measles": ["fever", "rash", "cough", "runny_nose", "red_eyes", "light_sensitivity", "sore_throat"],
    "Tuberculosis": ["chronic_cough", "blood_in_cough", "fever", "night_sweats", "weight_loss", "fatigue", "chest_pain"],
    "Anemia": ["fatigue", "weakness", "pale_skin", "shortness_of_breath", "dizziness", "cold_hands", "headache"],
}

all_symptoms = sorted(set(s for symptoms in diseases.values() for s in symptoms))

rows = []
for disease, core_symptoms in diseases.items():
    for _ in range(120):
        row = {s: 0 for s in all_symptoms}
        selected = random.sample(core_symptoms, min(random.randint(3, 5), len(core_symptoms)))
        for s in selected:
            row[s] = 1
        noise = random.sample([s for s in all_symptoms if s not in selected], random.randint(0, 2))
        for s in noise:
            row[s] = 1
        row["disease"] = disease
        rows.append(row)

df = pd.DataFrame(rows)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
df.to_csv("./dataset/diseases.csv", index=False)
print(f"Dataset: {df.shape[0]} rows, {df.shape[1]} columns")
print(f"Diseases: {df['disease'].nunique()}")
print(f"Total symptoms: {len(all_symptoms)}")
