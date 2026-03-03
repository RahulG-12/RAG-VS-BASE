import time
import pandas as pd
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

def measure_latency(function, *args):
    start = time.time()
    result = function(*args)
    end = time.time()
    return result, round(end - start, 3)


def evaluate_models(questions, expected_answers, base_fn, rag_fn, vectorstore):
    results = []

    for q, expected in zip(questions, expected_answers):
        base_resp = base_fn(q)
        rag_resp = rag_fn(vectorstore, q)

        base_correct = expected.lower() in base_resp.lower()
        rag_correct = expected.lower() in rag_resp.lower()

        results.append({
            "question": q,
            "expected": expected,
            "base_correct": base_correct,
            "rag_correct": rag_correct
        })

    df = pd.DataFrame(results)

    base_acc = df["base_correct"].mean()
    rag_acc = df["rag_correct"].mean()

    return df, base_acc, rag_acc


def plot_accuracy(base_acc, rag_acc):
    models = ["Base", "RAG"]
    scores = [base_acc, rag_acc]

    plt.figure()
    plt.bar(models, scores)
    plt.title("Accuracy Comparison")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.show()