import pandas as pd

conditions = [
    "diabetes",
    "hypertension",
    "anemia",
    "cholesterol",
    "asthma",
    "arthritis",
    "migraine",
    "obesity",
    "depression",
    "allergy"
]

data = []

for condition in conditions:
    for i in range(1):
        data.append({
            "question": f"What is {condition}?",
            "answer": f"{condition.capitalize()} is a medical condition affecting human health."
        })

df = pd.DataFrame(data)
df.to_csv("data/dataset.csv", index=False)

print("Dataset generated with", len(df), "samples.")