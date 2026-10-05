"""
MedPredict AI - Trusted Knowledge Base
Static, hand-written reference snippets per disease. This is the retrieval
source for the LLM explainer — the LLM is only allowed to phrase THIS content
in plain language, never to add outside medical claims.
"""

KNOWLEDGE = {
    "Influenza": "A common viral respiratory infection. Typically self-limiting in healthy adults; rest, fluids, and fever management are standard supportive care. Seek care if breathing difficulty or symptoms worsen after a few days.",
    "Common Cold": "A mild, self-limiting viral upper respiratory infection. Usually resolves within 7-10 days with supportive care.",
    "COVID-19": "A viral respiratory illness with a wide severity range. Testing is recommended for confirmation; isolation guidance varies by current local health authority policy.",
    "Pneumonia": "A lung infection that can range from mild to serious, especially in older adults or those with existing conditions. Medical evaluation, often including a chest X-ray, is generally advised.",
    "Bronchitis": "Inflammation of the airways, often following a viral infection. Usually resolves on its own; persistent or worsening cough warrants medical review.",
    "Malaria": "A mosquito-borne parasitic infection requiring prompt diagnosis (blood test) and treatment; can become severe if untreated.",
    "Dengue": "A mosquito-borne viral infection; most cases are mild, but warning signs (severe abdominal pain, persistent vomiting, bleeding) need urgent care.",
    "Typhoid": "A bacterial infection usually spread via contaminated food or water; requires antibiotic treatment guided by a clinician.",
    "Jaundice": "A symptom of yellowing skin/eyes reflecting an underlying liver or blood condition; needs clinical evaluation to identify the cause.",
    "Diabetes": "A chronic metabolic condition involving blood sugar regulation; diagnosis requires blood testing and ongoing medical management.",
    "Hypertension": "Elevated blood pressure; often symptomless, diagnosed via measurement over time, and managed with lifestyle and/or medication under medical guidance.",
    "Gastroenteritis": "Inflammation of the stomach/intestines, usually viral or bacterial; hydration is the main concern, especially in children and older adults.",
    "Migraine": "A neurological condition causing recurrent, often severe headaches; diagnosis and treatment planning benefit from a neurologist's input for recurrent cases.",
    "Asthma": "A chronic airway condition causing episodic breathing difficulty; managed with inhalers and trigger avoidance under medical supervision.",
    "Appendicitis": "Inflammation of the appendix; can escalate quickly and is considered a surgical emergency if suspected.",
    "UTI": "A bacterial infection of the urinary tract; typically treated with antibiotics after clinical or lab confirmation.",
    "Chickenpox": "A contagious viral infection causing an itchy rash; generally mild in children, can be more serious in adults.",
    "Measles": "A highly contagious viral infection preventable by vaccination; can cause serious complications, especially in unvaccinated individuals.",
    "Tuberculosis": "A bacterial infection primarily affecting the lungs; requires confirmatory testing and a prolonged antibiotic regimen under medical supervision.",
    "Anemia": "A condition of reduced red blood cells/hemoglobin causing fatigue and weakness; blood testing identifies the underlying cause and guides treatment.",
}

DISCLAIMER = (
    "This information is general and educational only. It is not a diagnosis "
    "and does not replace professional medical evaluation."
)


def retrieve(disease: str) -> str:
    return KNOWLEDGE.get(disease, "No reference information available for this condition.")