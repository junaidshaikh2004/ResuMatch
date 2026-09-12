"""
Trains and evaluates both models on the synthetic dataset:
  - a Logistic Regression baseline (scikit-learn)
  - FitNet, a small hand-built PyTorch feedforward network

Both are trained on the exact same 6-number feature vector (see
matcher/ml/features.py), computed with the exact same code path used at
inference time in predict.py, so there's no train/inference skew.

Data is split three ways: train (70%), validation (15%), test (15%).
The validation set is used to pick each model's decision threshold
(recall on the "good fit" class matters more than raw accuracy here, so we
search for the threshold that maximizes recall while keeping precision
above a floor). The held-out test set is only ever used for the final,
reported metrics — never for threshold tuning — so those numbers aren't
inflated by fitting the threshold to the same data we evaluate on.

Run with: python manage.py train_models
"""

import json

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer

from django.conf import settings
from django.core.management.base import BaseCommand, CommandError

from matcher.ml.features import FEATURE_NAMES, build_feature_dict, feature_dict_to_vector
from matcher.ml.nn_model import FitNet
from matcher.ml.preprocessing import clean_text

PRECISION_FLOOR = 0.55  # don't accept a threshold whose precision drops below this
THRESHOLD_GRID = np.arange(0.05, 0.96, 0.05)


def build_features(df, vectorizer):
    """Runs build_feature_dict on every row of the dataframe, returns (X, y)."""
    rows = []
    for _, row in df.iterrows():
        features = build_feature_dict(row["resume_text"], row["jd_text"], vectorizer)
        rows.append(feature_dict_to_vector(features))
    X = np.vstack(rows)
    y = df["label"].to_numpy()
    return X, y


def precision_recall_f1(y_true, y_pred):
    return {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def find_best_threshold(y_true, y_scores):
    """
    Scans a grid of thresholds and returns the one that maximizes recall on
    the "good fit" class, as long as precision stays above PRECISION_FLOOR.
    Falls back to the best F1 threshold if none clears that floor.
    """
    candidates = []
    for t in THRESHOLD_GRID:
        y_pred = (y_scores >= t).astype(int)
        scores = precision_recall_f1(y_true, y_pred)
        scores["threshold"] = float(t)
        candidates.append(scores)

    acceptable = [c for c in candidates if c["precision"] >= PRECISION_FLOOR]
    pool = acceptable if acceptable else candidates
    best = max(pool, key=lambda c: (c["recall"], c["f1"]))
    return best["threshold"]


def evaluate_at_threshold(y_true, y_scores, threshold):
    y_pred = (y_scores >= threshold).astype(int)
    metrics = precision_recall_f1(y_true, y_pred)
    metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
    metrics["confusion_matrix"] = confusion_matrix(y_true, y_pred).tolist()
    metrics["threshold"] = float(threshold)
    return metrics


def train_pytorch_model(X_train, y_train, epochs=60, lr=0.001):
    """Manual PyTorch training loop for FitNet, using a class-imbalance-aware loss."""
    X_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_tensor = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)

    dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
    loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

    # Weight the positive class by the negative/positive ratio so the loss
    # doesn't just learn to always predict the majority class.
    num_pos = y_train.sum()
    num_neg = len(y_train) - num_pos
    pos_weight = torch.tensor([num_neg / max(num_pos, 1)], dtype=torch.float32)

    model = FitNet(input_size=X_train.shape[1])
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for batch_X, batch_y in loader:
            optimizer.zero_grad()
            logits = model(batch_X)
            loss = loss_fn(logits, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if (epoch + 1) % 10 == 0:
            print(f"  epoch {epoch + 1}/{epochs} - avg loss: {total_loss / len(loader):.4f}")

    model.eval()
    return model


def predict_proba_pytorch(model, X):
    with torch.no_grad():
        logits = model(torch.tensor(X, dtype=torch.float32))
        return torch.sigmoid(logits).numpy().reshape(-1)


class Command(BaseCommand):
    help = "Trains the Logistic Regression and PyTorch models on the resume/JD dataset."

    def handle(self, *args, **options):
        dataset_path = settings.DATASET_PATH
        if not dataset_path.exists():
            raise CommandError(
                f"No dataset found at {dataset_path}. Run 'python manage.py generate_dataset' first."
            )

        df = pd.read_csv(dataset_path)
        self.stdout.write(f"Loaded {len(df)} rows from {dataset_path}")

        # 70% train / 15% validation / 15% test, stratified so the class
        # balance is preserved in every split.
        train_df, temp_df = train_test_split(
            df, test_size=0.3, stratify=df["label"], random_state=42
        )
        val_df, test_df = train_test_split(
            temp_df, test_size=0.5, stratify=temp_df["label"], random_state=42
        )
        self.stdout.write(f"Split: {len(train_df)} train / {len(val_df)} val / {len(test_df)} test")

        # Fit the TF-IDF vectorizer on the training text only, to avoid
        # leaking test/val vocabulary into training.
        train_corpus = [
            clean_text(t) for t in pd.concat([train_df["resume_text"], train_df["jd_text"]])
        ]
        vectorizer = TfidfVectorizer(max_features=2000, stop_words="english")
        vectorizer.fit(train_corpus)

        self.stdout.write("Building feature vectors for train/val/test...")
        X_train, y_train = build_features(train_df, vectorizer)
        X_val, y_val = build_features(val_df, vectorizer)
        X_test, y_test = build_features(test_df, vectorizer)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)

        # --- Logistic Regression baseline ---
        self.stdout.write("Training Logistic Regression...")
        logistic_model = LogisticRegression(class_weight="balanced", max_iter=1000)
        logistic_model.fit(X_train_scaled, y_train)
        lr_val_scores = logistic_model.predict_proba(X_val_scaled)[:, 1]
        lr_test_scores = logistic_model.predict_proba(X_test_scaled)[:, 1]

        # --- PyTorch neural network ---
        self.stdout.write("Training PyTorch FitNet...")
        nn_model = train_pytorch_model(X_train_scaled, y_train)
        nn_val_scores = predict_proba_pytorch(nn_model, X_val_scaled)
        nn_test_scores = predict_proba_pytorch(nn_model, X_test_scaled)

        # Pick each model's decision threshold on the validation set...
        lr_threshold = find_best_threshold(y_val, lr_val_scores)
        nn_threshold = find_best_threshold(y_val, nn_val_scores)

        # ...then report final metrics on the untouched test set.
        lr_metrics = evaluate_at_threshold(y_test, lr_test_scores, lr_threshold)
        nn_metrics = evaluate_at_threshold(y_test, nn_test_scores, nn_threshold)

        metrics_by_model = {"logistic_regression": lr_metrics, "pytorch_nn": nn_metrics}

        # The model shown as "the" verdict is whichever has better recall on
        # the good-fit class (ties broken by F1) — reflecting that missing a
        # genuine good fit is worse here than a false positive.
        primary_model = max(
            metrics_by_model, key=lambda name: (metrics_by_model[name]["recall"], metrics_by_model[name]["f1"])
        )

        self.stdout.write(self.style.SUCCESS(f"Logistic Regression: {lr_metrics}"))
        self.stdout.write(self.style.SUCCESS(f"PyTorch FitNet: {nn_metrics}"))
        self.stdout.write(self.style.SUCCESS(f"Primary model (by recall): {primary_model}"))

        artifacts_dir = settings.ML_ARTIFACTS_DIR
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        joblib.dump(vectorizer, artifacts_dir / "tfidf_vectorizer.joblib")
        joblib.dump(scaler, artifacts_dir / "scaler.joblib")
        joblib.dump(logistic_model, artifacts_dir / "logistic_regression.joblib")
        torch.save(nn_model.state_dict(), artifacts_dir / "pytorch_nn.pt")

        metrics_payload = {
            "feature_names": FEATURE_NAMES,
            "primary_model": primary_model,
            "thresholds": {"logistic_regression": lr_threshold, "pytorch_nn": nn_threshold},
            "metrics": metrics_by_model,
            "dataset_size": {"train": len(train_df), "val": len(val_df), "test": len(test_df)},
        }
        with open(artifacts_dir / "metrics.json", "w") as f:
            json.dump(metrics_payload, f, indent=2)

        # --- EDA export for the dashboard ---
        good_fit_mask = df["label"] == 1
        eda_payload = {
            "class_balance": {
                "good_fit": int(good_fit_mask.sum()),
                "not_a_fit": int((~good_fit_mask).sum()),
            },
            "role_counts": df["jd_role"].value_counts().to_dict(),
        }
        # Feature distribution summary (mean/std per class), computed on the
        # full training feature matrix so it reflects what the models saw.
        full_X = np.vstack([X_train, X_val, X_test])
        full_y = np.concatenate([y_train, y_val, y_test])
        feature_distributions = {}
        for i, name in enumerate(FEATURE_NAMES):
            column = full_X[:, i]
            feature_distributions[name] = {
                "good_fit": {"mean": float(column[full_y == 1].mean()), "std": float(column[full_y == 1].std())},
                "not_a_fit": {"mean": float(column[full_y == 0].mean()), "std": float(column[full_y == 0].std())},
            }
        eda_payload["feature_distributions"] = feature_distributions

        with open(artifacts_dir / "eda.json", "w") as f:
            json.dump(eda_payload, f, indent=2)

        self.stdout.write(self.style.SUCCESS(f"Artifacts written to {artifacts_dir}"))
