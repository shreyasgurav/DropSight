"""
model.py
--------
Trains Random Forest and Logistic Regression classifiers to predict
the drop-off page (1–5). Saves the best model as model.pkl, and
generates feature importance + confusion matrix charts.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, accuracy_score,
                             confusion_matrix)
from sklearn.preprocessing import LabelEncoder

# --------------- paths ---------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLEAN_CSV = os.path.join(BASE_DIR, "data", "cleaned_data.csv")
MODEL_PATH = os.path.join(BASE_DIR, "output", "model.pkl")
CHARTS_DIR = os.path.join(BASE_DIR, "output", "charts")
os.makedirs(CHARTS_DIR, exist_ok=True)

# --------------- config ---------------
# Note: 'order' removed to avoid data leakage (order step correlates with page depth)
# This gives more realistic accuracy based on behavioral features
FEATURE_COLS = ["main_category", "colour", "location", "photo_type",
                "price", "price_range", "country", "month"]
TARGET = "page"


def prepare_data():
    """Load cleaned CSV, encode categorical features, split into train/test."""
    df = pd.read_csv(CLEAN_CSV)

    # Encode categorical string columns
    label_encoders = {}
    for col in ["main_category", "photo_type", "price_range"]:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le

    X = df[FEATURE_COLS]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    return X_train, X_test, y_train, y_test, label_encoders


def train_models(X_train, X_test, y_train, y_test):
    """Train Random Forest & Logistic Regression, return results dict."""
    results = {}

    # --- Random Forest ---
    rf = RandomForestClassifier(n_estimators=150, max_depth=20,
                                random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_preds = rf.predict(X_test)
    rf_acc = accuracy_score(y_test, rf_preds)
    rf_report = classification_report(y_test, rf_preds, output_dict=True)
    results["Random Forest"] = {
        "model": rf, "accuracy": rf_acc,
        "report": rf_report, "preds": rf_preds
    }
    print(f"Random Forest Accuracy: {rf_acc:.4f}")
    print(classification_report(y_test, rf_preds))

    # --- Logistic Regression ---
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train, y_train)
    lr_preds = lr.predict(X_test)
    lr_acc = accuracy_score(y_test, lr_preds)
    lr_report = classification_report(y_test, lr_preds, output_dict=True)
    results["Logistic Regression"] = {
        "model": lr, "accuracy": lr_acc,
        "report": lr_report, "preds": lr_preds
    }
    print(f"Logistic Regression Accuracy: {lr_acc:.4f}")
    print(classification_report(y_test, lr_preds))

    return results


def save_best_model(results):
    """Save the model with the higher accuracy."""
    best_name = max(results, key=lambda k: results[k]["accuracy"])
    best_model = results[best_name]["model"]
    joblib.dump(best_model, MODEL_PATH)
    print(f"\n✅ Best model: {best_name} → saved to {MODEL_PATH}")
    return best_name, best_model


def plot_feature_importance(model, feature_names):
    """Bar chart of feature importances (Random Forest)."""
    importances = model.feature_importances_
    idx = np.argsort(importances)[::-1]
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=importances[idx], y=np.array(feature_names)[idx],
                palette="viridis", ax=ax)
    ax.set_title("Feature Importance (Random Forest)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Importance")
    ax.set_ylabel("Feature")
    plt.tight_layout()
    fig.savefig(os.path.join(CHARTS_DIR, "feature_importance.png"), dpi=150)
    plt.close(fig)
    print("✅ feature_importance.png saved")


def plot_confusion_matrix(y_test, preds, labels=None):
    """Heatmap of the confusion matrix."""
    cm = confusion_matrix(y_test, preds, labels=labels)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_title("Confusion Matrix (Best Model)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Predicted Page")
    ax.set_ylabel("Actual Page")
    plt.tight_layout()
    fig.savefig(os.path.join(CHARTS_DIR, "confusion_matrix.png"), dpi=150)
    plt.close(fig)
    print("✅ confusion_matrix.png saved")


def save_metrics(results):
    """Save accuracy scores to a small CSV for the dashboard."""
    rows = []
    for name, data in results.items():
        rows.append({"model": name, "accuracy": data["accuracy"]})
    pd.DataFrame(rows).to_csv(os.path.join(BASE_DIR, "output", "model_metrics.csv"), index=False)
    print("✅ model_metrics.csv saved")


# --------------- main ---------------
if __name__ == "__main__":
    X_train, X_test, y_train, y_test, label_encoders = prepare_data()
    results = train_models(X_train, X_test, y_train, y_test)
    best_name, best_model = save_best_model(results)
    save_metrics(results)

    # Charts
    if best_name == "Random Forest":
        plot_feature_importance(best_model, FEATURE_COLS)
    else:
        # Still plot RF importance even if LR won
        rf_model = results["Random Forest"]["model"]
        plot_feature_importance(rf_model, FEATURE_COLS)

    page_labels = sorted(y_test.unique())
    plot_confusion_matrix(y_test, results[best_name]["preds"], labels=page_labels)

    print("\n🎉 Model training complete!")
