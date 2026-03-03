from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np


def compute_binary_metrics(y_true, y_pred):
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    return precision, recall, f1


def hallucination_rate(expected, predicted):
    hallucinations = 0

    for exp, pred in zip(expected, predicted):
        if exp.lower() not in pred.lower():
            hallucinations += 1

    return hallucinations / len(expected)