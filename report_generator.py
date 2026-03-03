def generate_report(results):
    report = "EXPERIMENTAL FINDINGS\n\n"

    for model, metrics in results.items():
        report += f"Model: {model}\n"
        for k, v in metrics.items():
            report += f"{k}: {v}\n"
        report += "\n"

    with open("experiment_report.txt", "w") as f:
        f.write(report)

    print("Report saved as experiment_report.txt")