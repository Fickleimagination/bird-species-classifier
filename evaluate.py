import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from src.data import get_splits, make_dataset

def save_evaluation(name, y_true, probabilities, class_names):
    """Write the metrics, classification report and confusion matrix for one model to outputs/."""
    y_pred = np.argmax(probabilities, axis=1)
    top5 = np.argsort(probabilities, axis=1)[:, -5:]
    top5_accuracy = float(np.mean([label in candidates for label, candidates in zip(y_true, top5)]))
    results = {
        "model": name,
        "test_images": int(len(y_true)),
        "top1_accuracy": float(accuracy_score(y_true, y_pred)),
        "top5_accuracy": top5_accuracy,
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    }
    Path("outputs").mkdir(exist_ok=True)
    with open("outputs/" + name + "_metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    report = classification_report(
        y_true, y_pred, labels=list(range(len(class_names))),
        target_names=class_names, zero_division=0
    )
    with open("outputs/" + name + "_classification_report.txt", "w") as f:
        f.write(report)

    # A full 200 x 200 confusion matrix is dense; save a figure for the 20 most frequent true classes.
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    support = np.bincount(y_true, minlength=len(class_names))
    top_classes = np.argsort(support)[-20:]
    plt.figure(figsize=(13, 10))
    sns.heatmap(cm[np.ix_(top_classes, top_classes)], cmap="Blues", xticklabels=[class_names[i] for i in top_classes],
                yticklabels=[class_names[i] for i in top_classes])
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.title("Confusion matrix: 20 classes with highest test support")
    plt.tight_layout()
    plt.savefig(f"outputs/{name}_confusion_matrix.png", dpi=160)
    plt.close()
    return results

def main():
    parser = argparse.ArgumentParser(description="Evaluate a saved bird classifier on the official test split.")
    parser.add_argument("--dataset", default="data/CUB_200_2011")
    parser.add_argument("--model", default="models/transfer.keras")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}. Train it first.")
    _, _, test_df, class_names = get_splits(args.dataset)
    test_ds = make_dataset(test_df, args.batch_size)
    model = tf.keras.models.load_model(model_path)
    probabilities = model.predict(test_ds, verbose=1)
    results = save_evaluation(model_path.stem, test_df["label"].to_numpy(), probabilities, class_names)
    results["model"] = str(model_path)
    with open("outputs/" + model_path.stem + "_metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))
    print("Detailed classification report: outputs/" + model_path.stem + "_classification_report.txt")

if __name__ == "__main__":
    main()
