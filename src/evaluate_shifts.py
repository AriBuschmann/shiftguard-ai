# Evaluation of the baseline CNN under Gaussian noise distribution shift

"""
workflow:

Load trained baseline CNN
        ↓
Evaluate severity levels 0–5
        ↓
Load the same 10,000 CIFAR-10 test images
        ↓
ToTensor()
        ↓
GaussianNoise(severity)
        ↓
Normalize()
        ↓
CNN
        ↓
Calculate accuracy and prediction confidence
        ↓
Calculate confidence-accuracy gap
        ↓
Save results to CSV

The same trained model and the same CIFAR-10 test set are used for every severity level. Only the strength of the Gaussian noise changes.
"""

import csv
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import BaselineCNN
from shifts import GaussianNoise, GAUSSIAN_NOISE_SIGMAS


BATCH_SIZE = 64
RANDOM_SEED = 42 # ensures reproducibility of the random noise pattern across different severity levels


# project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODEL_PATH = PROJECT_ROOT / "models" / "baseline_cnn.pth"
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_PATH = RESULTS_DIR / "gaussian_noise_results.csv"


# CIFAR-10 normalization used during training
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


# choose GPU / CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")


# create the baseline CNN and load the trained model weights
def load_model():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained model not found: {MODEL_PATH}\n"
            "Run src/train.py before evaluating distribution shifts."
        )

    model = BaselineCNN().to(device)

    state_dict = torch.load(
        MODEL_PATH,
        map_location=device,
        weights_only=True
    )

    model.load_state_dict(state_dict)
    model.eval() # switch the model to evaluation mode

    return model


# create a CIFAR-10 test DataLoader for one Gaussian noise severity level
def create_test_loader(severity):

    transform = transforms.Compose([
        transforms.ToTensor(),

        GaussianNoise(severity=severity),

        transforms.Normalize(
            mean=CIFAR10_MEAN,
            std=CIFAR10_STD
        )
    ])

    test_dataset = datasets.CIFAR10(
        root=DATA_DIR,
        train=False,
        download=False,
        transform=transform
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    return test_loader


# evaluate the trained model on all 10,000 CIFAR-10 test images using one Gaussian noise severity level
def evaluate_severity(model, severity):

    sigma = GAUSSIAN_NOISE_SIGMAS[severity]

    torch.manual_seed(RANDOM_SEED) # reset the random seed before every severity level which makes the underlying random noise pattern reproducible and comparable across different noise strengths

    test_loader = create_test_loader(severity)

    correct_predictions = 0
    total_samples = 0

    total_confidence = 0.0

    correct_confidence = 0.0
    incorrect_confidence = 0.0

    correct_count = 0
    incorrect_count = 0

    with torch.no_grad():

        for images, labels in test_loader:

            # move batch to GPU / CPU
            images = images.to(device)
            labels = labels.to(device)

            # forward pass
            outputs = model(images)

            # convert logits into probabilities
            probabilities = torch.softmax(outputs, dim=1)

            # highest probability represents the predicted class and its corresponding confidence
            confidences, predicted_classes = probabilities.max(dim=1)

            # determine correct and incorrect predictions
            correct_mask = predicted_classes == labels
            incorrect_mask = ~correct_mask

            batch_size = labels.size(0)

            total_samples += batch_size
            correct_predictions += correct_mask.sum().item()

            # accumulate confidence across all predictions
            total_confidence += confidences.sum().item()

            # confidence for correct predictions
            correct_confidence += confidences[correct_mask].sum().item()
            correct_count += correct_mask.sum().item()

            # confidence for incorrect predictions
            incorrect_confidence += confidences[incorrect_mask].sum().item()
            incorrect_count += incorrect_mask.sum().item()


    accuracy = (
        correct_predictions / total_samples
    ) * 100

    mean_confidence = (
        total_confidence / total_samples
    ) * 100

    mean_correct_confidence = (
        correct_confidence / correct_count
    ) * 100 if correct_count > 0 else 0.0

    mean_incorrect_confidence = (
        incorrect_confidence / incorrect_count
    ) * 100 if incorrect_count > 0 else 0.0

    confidence_accuracy_gap = mean_confidence - accuracy # positive values indicate that mean confidence exceeds accuracy, suggesting overall model overconfidence


    return {
        "severity": severity,
        "sigma": sigma,
        "accuracy": accuracy,
        "mean_confidence": mean_confidence,
        "correct_confidence": mean_correct_confidence,
        "incorrect_confidence": mean_incorrect_confidence,
        "confidence_accuracy_gap": confidence_accuracy_gap,
    }


# save all Gaussian noise experiment results to a CSV file
def save_results(results):

    fieldnames = [
        "severity",
        "sigma",
        "accuracy",
        "mean_confidence",
        "correct_confidence",
        "incorrect_confidence",
        "confidence_accuracy_gap",
    ]

    with open(RESULTS_PATH, "w", newline="", encoding="utf-8") as csv_file:

        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames
        )

        writer.writeheader()

        for result in results:
            writer.writerow({
                "severity": result["severity"],
                "sigma": f"{result['sigma']:.2f}",
                "accuracy": f"{result['accuracy']:.4f}",
                "mean_confidence": f"{result['mean_confidence']:.4f}",
                "correct_confidence": f"{result['correct_confidence']:.4f}",
                "incorrect_confidence": f"{result['incorrect_confidence']:.4f}",
                "confidence_accuracy_gap": f"{result['confidence_accuracy_gap']:.4f}",
            })

    print(f"\nResults saved to: {RESULTS_PATH}")


# run the complete Gaussian noise distribution-shift experiment
def main():

    model = load_model()

    results = []

    print("\nEvaluating Gaussian noise distribution shift...\n")

    for severity in sorted(GAUSSIAN_NOISE_SIGMAS):

        sigma = GAUSSIAN_NOISE_SIGMAS[severity]

        print(
            f"Evaluating severity {severity} "
            f"(sigma = {sigma:.2f})..."
        )

        result = evaluate_severity(
            model=model,
            severity=severity
        )

        results.append(result)

        print(
            f"Accuracy: {result['accuracy']:.2f}% | "
            f"Mean Confidence: {result['mean_confidence']:.2f}% | "
            f"Correct Confidence: {result['correct_confidence']:.2f}% | "
            f"Incorrect Confidence: {result['incorrect_confidence']:.2f}% | "
            f"Confidence-Accuracy Gap: "
            f"{result['confidence_accuracy_gap']:.2f} pp\n"
        )

    save_results(results)

    print("\nGaussian noise evaluation completed successfully!")


if __name__ == "__main__":
    main()