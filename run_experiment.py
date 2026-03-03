import pandas as pd
import time
from models.rag_pipeline import RAGSystem
from evaluation.metrics import compute_binary_metrics, hallucination_rate
from evaluation.statistical import t_test

DATASET_PATH = "data/dataset.csv"

models = ["phi3:mini"]

results = {}

df = pd.read_csv(DATASET_PATH)
questions = df["question"].tolist()
answers = df["answer"].tolist()

for model_name in models:
    print("\nRunning model:", model_name)

    system = RAGSystem(model_name)
    system.build_vectorstore(DATASET_PATH)

    base_preds = []
    rag_preds = []
    latencies = []

    for i, q in enumerate(questions):
        print(f"Processing question {i+1}/{len(questions)}")

        start = time.time()
        rag_resp = system.rag_answer(q)
        latency = time.time() - start

        print("RAG done")

        base_resp = system.base_answer(q)
        print("BASE done")

        rag_preds.append(rag_resp)
        latencies.append(latency)
        base_preds.append(base_resp)

    y_true = [1]*len(answers)

    y_rag = []

    for expected, predicted in zip(answers, rag_preds):
        expected_words = set(expected.lower().split())
        predicted_words = set(predicted.lower().split())

        overlap = expected_words.intersection(predicted_words)

        if len(overlap) >= 2:   # allow partial match
            y_rag.append(1)
        else:
            y_rag.append(0)

    precision, recall, f1 = compute_binary_metrics(y_true, y_rag)
    halluc_rate = hallucination_rate(answers, rag_preds)

    results[model_name] = {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "hallucination_rate": halluc_rate,
        "avg_latency": sum(latencies)/len(latencies)
    }

print("\nFINAL RESULTS\n")
for model, metrics in results.items():
    print(model, metrics)