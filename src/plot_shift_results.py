# Visualization of Gaussian noise distribution-shift experiment results

"""
Load gaussian_noise_results.csv
        ↓
Validate and sort experiment results
        ↓
Create accuracy-vs-severity plot
        ↓
Create accuracy-vs-confidence plot
        ↓
Save both plots as PNG files
"""

import csv
from pathlib import Path

import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"

CSV_PATH = RESULTS_DIR / "gaussian_noise_results.csv"

ACCURACY_PLOT_PATH = RESULTS_DIR / "gaussian_noise_accuracy.png"
ACCURACY_CONFIDENCE_PLOT_PATH = (
    RESULTS_DIR / "gaussian_noise_accuracy_confidence.png"
)


# required CSV columns
REQUIRED_COLUMNS = {
    "severity",
    "sigma",
    "accuracy",
    "mean_confidence",
}


def load_results(): # load and validate Gaussian noise experiment results from the CSV file

    if not CSV_PATH.exists():
        raise FileNotFoundError(
            f"Experiment results not found: {CSV_PATH}\n"
            "Run src/evaluate_shifts.py before creating the plots."
        )

    results = []

    with open(CSV_PATH, "r", newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)

        if reader.fieldnames is None:
            raise ValueError(
                "The results CSV does not contain a header."
            )

        missing_columns = REQUIRED_COLUMNS - set(reader.fieldnames)

        if missing_columns:
            raise ValueError(
                "The results CSV is missing required columns: "
                f"{sorted(missing_columns)}"
            )

        for row in reader:
            results.append({
                "severity": int(row["severity"]),
                "sigma": float(row["sigma"]),
                "accuracy": float(row["accuracy"]),
                "mean_confidence": float(row["mean_confidence"]),
            })

    if not results:
        raise ValueError(
            "The results CSV does not contain any experiment results."
        )

    results.sort(key=lambda result: result["severity"]) # sort results by severity so that the x-axis follows severity 0 -> 5

    return results


def validate_results(results):

    if len(results) != 6:
        raise ValueError(
            f"Expected exactly 6 experiment results, but found {len(results)}."
        )

    expected_severities = set(range(6))
    actual_severities = {
        result["severity"]
        for result in results
    }

    if actual_severities != expected_severities:
        raise ValueError(
            "Expected Gaussian noise severities 0-5, "
            f"but found {sorted(actual_severities)}."
        )


def create_x_axis_data(results):

    severities = [
        result["severity"]
        for result in results
    ]

    labels = [
        f"{result['severity']}\nσ={result['sigma']:.2f}"
        for result in results
    ]

    return severities, labels


# plot classification accuracy across Gaussian noise severity levels
def plot_accuracy(results):

    severities, severity_labels = create_x_axis_data(results)

    accuracies = [
        result["accuracy"]
        for result in results
    ]

    plt.figure(figsize=(8, 5))

    plt.plot(
        severities,
        accuracies,
        marker="o",
        linewidth=2
    )

    plt.title("Baseline CNN Accuracy under Gaussian Noise")
    plt.xlabel("Gaussian Noise Severity")
    plt.ylabel("Accuracy (%)")

    plt.xticks(
        severities,
        severity_labels
    )

    plt.ylim(0, 100) # use the full percentage range to avoid visually exaggerating changes

    plt.grid(
        True,
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        ACCURACY_PLOT_PATH,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    print(f"Accuracy plot saved to: {ACCURACY_PLOT_PATH}")


# compare classification accuracy and mean prediction confidence across Gaussian noise severity levels
def plot_accuracy_vs_confidence(results):

    severities, severity_labels = create_x_axis_data(results)

    accuracies = [
        result["accuracy"]
        for result in results
    ]

    mean_confidences = [
        result["mean_confidence"]
        for result in results
    ]

    plt.figure(figsize=(8, 5))

    plt.plot(
        severities,
        accuracies,
        marker="o",
        linewidth=2,
        label="Accuracy"
    )

    plt.plot(
        severities,
        mean_confidences,
        marker="o",
        linewidth=2,
        label="Mean Confidence"
    )

    plt.title("Accuracy vs. Confidence under Gaussian Noise")
    plt.xlabel("Gaussian Noise Severity")
    plt.ylabel("Percentage (%)")

    plt.xticks(
        severities,
        severity_labels
    )

    plt.ylim(0, 100)

    plt.grid(
        True,
        alpha=0.3
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        ACCURACY_CONFIDENCE_PLOT_PATH,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    print(
        "Accuracy-confidence plot saved to: "
        f"{ACCURACY_CONFIDENCE_PLOT_PATH}"
    )


def main():

    print("\nLoading Gaussian noise experiment results...\n")

    results = load_results()
    validate_results(results)
    print(f"Loaded {len(results)} severity levels successfully.\n")
    plot_accuracy(results)
    plot_accuracy_vs_confidence(results)
    print("\nGaussian noise plots created successfully!")


if __name__ == "__main__":
    main()