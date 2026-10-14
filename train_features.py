import argparse
import json
from pathlib import Path
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from evaluate import save_evaluation
from src.data import get_splits
from src.features import BACKBONES, extract_features, get_device, load_backbone

def load_or_extract(model, frame, cache_path, device, batch_size):
    if cache_path.exists():
        features = np.load(cache_path)
        if len(features) == len(frame):
            print(f"Loaded cached features: {cache_path}")
            return features
    print(f"Extracting features for {len(frame)} images -> {cache_path}")
    features = extract_features(model, frame["path"], device, batch_size)
    np.save(cache_path, features)
    return features

def main():
    parser = argparse.ArgumentParser(
        description="Train a logistic-regression classifier on frozen DINOv2 features."
    )
    parser.add_argument("--dataset", default="data/CUB_200_2011")
    parser.add_argument("--backbone", choices=sorted(BACKBONES), default="dinov2_base")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--c-values", type=float, nargs="+", default=[0.01, 0.03, 0.1, 0.3],
                        help="Inverse regularisation strengths to compare on the validation split.")
    args = parser.parse_args()

    Path("models").mkdir(exist_ok=True)
    cache_dir = Path("outputs/features")
    cache_dir.mkdir(parents=True, exist_ok=True)

    train_df, val_df, test_df, class_names = get_splits(args.dataset, seed=args.seed)
    print(f"Images — train: {len(train_df)}, validation: {len(val_df)}, test: {len(test_df)}")
    device = get_device()
    print(f"Backbone: {BACKBONES[args.backbone]} on {device}")
    model = load_backbone(args.backbone, device)

    splits = {"train": train_df, "val": val_df, "test": test_df}
    features = {
        name: load_or_extract(model, frame, cache_dir / f"{args.backbone}_{name}.npy", device, args.batch_size)
        for name, frame in splits.items()
    }
    labels = {name: frame["label"].to_numpy() for name, frame in splits.items()}

    # Choose the regularisation strength on the validation split only.
    best_c, best_classifier, best_val = None, None, -1.0
    for c in args.c_values:
        classifier = make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=3000))
        classifier.fit(features["train"], labels["train"])
        train_acc = classifier.score(features["train"], labels["train"])
        val_acc = classifier.score(features["val"], labels["val"])
        print(f"C={c:<6} train accuracy {train_acc:.4f}  validation accuracy {val_acc:.4f}")
        if val_acc > best_val:
            best_c, best_classifier, best_val = c, classifier, val_acc
    print(f"Selected C={best_c} (validation accuracy {best_val:.4f})")

    joblib.dump(
        {"backbone": args.backbone, "classifier": best_classifier, "class_names": class_names},
        f"models/{args.backbone}.joblib"
    )
    with open("models/class_names.json", "w", encoding="utf-8") as f:
        json.dump(class_names, f, indent=2)

    results = save_evaluation(
        args.backbone, labels["test"], best_classifier.predict_proba(features["test"]), class_names
    )
    results["train_accuracy"] = float(best_classifier.score(features["train"], labels["train"]))
    results["validation_accuracy"] = float(best_val)
    results["C"] = best_c
    with open(f"outputs/{args.backbone}_metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))
    print(f"Saved classifier: models/{args.backbone}.joblib")

if __name__ == "__main__":
    main()
