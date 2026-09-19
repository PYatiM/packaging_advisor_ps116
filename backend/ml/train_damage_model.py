"""
Damage-Risk ML Training Pipeline
=================================

End-to-end script that:
  1. Generates a synthetic dataset (10 000 samples)
  2. Trains XGBoost and RandomForest classifiers
  3. Evaluates Accuracy, F1, ROC-AUC on a held-out test set
  4. Computes feature-importance rankings
  5. Persists the best model + scaler via joblib
  6. Saves evaluation plots (ROC curves, confusion matrices, importance)

Run:
    python -m backend.ml.train_damage_model

Outputs land in  backend/ml/artifacts/
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # headless backend — no display needed
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
)
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import joblib

from backend.ml.dataset_generator import generate_dataset

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ARTIFACT_DIR = os.path.join("backend", "ml", "artifacts")
DATA_DIR = os.path.join("backend", "ml", "data")

FEATURE_COLS = [
    "max_fragility",
    "void_space_pct",
    "total_weight_kg",
    "item_count",
    "has_liquid",
    "orientation_sensitive",
    "distance_multiplier",
    "weight_density",
    "category_encoded",
]

LABEL_COL = "damaged"


def ensure_dirs():
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# 1.  Data preparation
# ---------------------------------------------------------------------------
def prepare_data(n_samples: int = 10_000, test_size: float = 0.20, seed: int = 42):
    df = generate_dataset(n_samples=n_samples, seed=seed)

    # Persist raw dataset
    csv_path = os.path.join(DATA_DIR, "damage_dataset.csv")
    df.to_csv(csv_path, index=False)
    print(f"[DATA]  {len(df)} samples → {csv_path}")
    print(f"[DATA]  Class balance — damaged: "
          f"{df[LABEL_COL].mean():.2%}, intact: {1 - df[LABEL_COL].mean():.2%}")

    X = df[FEATURE_COLS].values
    y = df[LABEL_COL].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y,
    )

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    return X_train_sc, X_test_sc, y_train, y_test, scaler, df


# ---------------------------------------------------------------------------
# 2.  Model definitions
# ---------------------------------------------------------------------------
def build_models(seed: int = 42):
    return {
        "XGBoost": XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.08,
            subsample=0.85,
            colsample_bytree=0.85,
            min_child_weight=3,
            reg_alpha=0.1,
            reg_lambda=1.0,
            scale_pos_weight=1.0,
            eval_metric="logloss",
            random_state=seed,
            use_label_encoder=False,
            verbosity=0,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=400,
            max_depth=12,
            min_samples_split=5,
            min_samples_leaf=3,
            max_features="sqrt",
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
        ),
    }


# ---------------------------------------------------------------------------
# 3.  Training + evaluation
# ---------------------------------------------------------------------------
def evaluate_model(name, model, X_train, X_test, y_train, y_test):
    print(f"\n{'=' * 60}")
    print(f"  Training: {name}")
    print(f"{'=' * 60}")

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    ap = average_precision_score(y_test, y_prob)

    # Cross-val on training set
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_auc = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc")

    metrics = {
        "model": name,
        "accuracy": round(acc, 4),
        "f1_score": round(f1, 4),
        "roc_auc": round(auc, 4),
        "avg_precision": round(ap, 4),
        "cv_auc_mean": round(cv_auc.mean(), 4),
        "cv_auc_std": round(cv_auc.std(), 4),
    }

    print(f"\n  Accuracy        : {acc:.4f}")
    print(f"  F1 Score        : {f1:.4f}")
    print(f"  ROC-AUC         : {auc:.4f}")
    print(f"  Avg Precision   : {ap:.4f}")
    print(f"  5-Fold CV AUC   : {cv_auc.mean():.4f} ± {cv_auc.std():.4f}")
    print(f"\n  Classification Report:\n{classification_report(y_test, y_pred, target_names=['Intact', 'Damaged'])}")

    return model, y_pred, y_prob, metrics


# ---------------------------------------------------------------------------
# 4.  Feature importance
# ---------------------------------------------------------------------------
def get_importance(name, model, feature_names):
    if hasattr(model, "feature_importances_"):
        imp = model.feature_importances_
    else:
        imp = np.zeros(len(feature_names))

    imp_df = pd.DataFrame({
        "feature": feature_names,
        "importance": imp,
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    print(f"\n  Feature Importance ({name}):")
    for _, row in imp_df.iterrows():
        bar = "█" * int(row["importance"] * 50)
        print(f"    {row['feature']:25s}  {row['importance']:.4f}  {bar}")

    return imp_df


# ---------------------------------------------------------------------------
# 5.  Plotting
# ---------------------------------------------------------------------------
def plot_results(results: dict, y_test):
    """Generate and save all evaluation plots."""
    sns.set_theme(style="whitegrid", font_scale=1.1)

    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle("Damage-Risk Classifier — Evaluation Dashboard",
                 fontsize=16, fontweight="bold", y=1.02)

    colors = {"XGBoost": "#2563eb", "RandomForest": "#16a34a"}

    # ---- ROC Curves --------------------------------------------------------
    ax = axes[0, 0]
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Random")
    for name, res in results.items():
        fpr, tpr, _ = roc_curve(y_test, res["y_prob"])
        ax.plot(fpr, tpr, color=colors[name], lw=2,
                label=f"{name} (AUC={res['metrics']['roc_auc']:.3f})")
    ax.set(xlabel="False Positive Rate", ylabel="True Positive Rate",
           title="ROC Curves")
    ax.legend(loc="lower right")

    # ---- Precision-Recall Curves -------------------------------------------
    ax = axes[0, 1]
    for name, res in results.items():
        prec, rec, _ = precision_recall_curve(y_test, res["y_prob"])
        ax.plot(rec, prec, color=colors[name], lw=2,
                label=f"{name} (AP={res['metrics']['avg_precision']:.3f})")
    ax.set(xlabel="Recall", ylabel="Precision",
           title="Precision-Recall Curves")
    ax.legend(loc="lower left")

    # ---- Confusion Matrices ------------------------------------------------
    for idx, (name, res) in enumerate(results.items()):
        ax = axes[0, 2] if idx == 0 else axes[1, 2]
        cm = confusion_matrix(y_test, res["y_pred"])
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=["Intact", "Damaged"],
                    yticklabels=["Intact", "Damaged"])
        ax.set(xlabel="Predicted", ylabel="Actual",
               title=f"Confusion Matrix — {name}")

    # ---- Feature Importance (side by side) ---------------------------------
    for idx, (name, res) in enumerate(results.items()):
        ax = axes[1, idx]
        imp = res["importance"]
        bars = ax.barh(imp["feature"], imp["importance"],
                       color=colors[name], alpha=0.85, edgecolor="white")
        ax.set(xlabel="Importance", title=f"Feature Importance — {name}")
        ax.invert_yaxis()
        for bar, val in zip(bars, imp["importance"]):
            ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                    f"{val:.3f}", va="center", fontsize=9)

    plt.tight_layout()
    plot_path = os.path.join(ARTIFACT_DIR, "evaluation_dashboard.png")
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[PLOT]  Dashboard saved → {plot_path}")
    return plot_path


# ---------------------------------------------------------------------------
# 6.  Model persistence
# ---------------------------------------------------------------------------
def save_best_model(results, scaler):
    """Save the model with the highest AUC."""
    best_name = max(results, key=lambda k: results[k]["metrics"]["roc_auc"])
    best_model = results[best_name]["model"]

    model_path = os.path.join(ARTIFACT_DIR, "best_damage_model.joblib")
    scaler_path = os.path.join(ARTIFACT_DIR, "feature_scaler.joblib")
    meta_path = os.path.join(ARTIFACT_DIR, "model_metadata.json")

    joblib.dump(best_model, model_path)
    joblib.dump(scaler, scaler_path)

    metadata = {
        "best_model": best_name,
        "feature_columns": FEATURE_COLS,
        "metrics": results[best_name]["metrics"],
        "dataset": "synthetic_rule_perturbed_with_noise",
        "n_train_samples": 8000,
        "n_test_samples": 2000,
    }
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n[SAVE]  Best model ({best_name}) → {model_path}")
    print(f"[SAVE]  Scaler → {scaler_path}")
    print(f"[SAVE]  Metadata → {meta_path}")
    return best_name


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ensure_dirs()

    # 1. Data
    X_train, X_test, y_train, y_test, scaler, df = prepare_data()

    # 2. Models
    models = build_models()

    # 3. Train + Evaluate
    results = {}
    for name, model in models.items():
        fitted, y_pred, y_prob, metrics = evaluate_model(
            name, model, X_train, X_test, y_train, y_test,
        )
        importance = get_importance(name, fitted, FEATURE_COLS)
        results[name] = {
            "model": fitted,
            "y_pred": y_pred,
            "y_prob": y_prob,
            "metrics": metrics,
            "importance": importance,
        }

    # 4. Plots
    plot_path = plot_results(results, y_test)

    # 5. Persist best model
    best_name = save_best_model(results, scaler)

    # 6. Summary JSON
    summary = {name: res["metrics"] for name, res in results.items()}
    summary_path = os.path.join(ARTIFACT_DIR, "evaluation_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[SAVE]  Summary → {summary_path}")

    print(f"\n{'=' * 60}")
    print(f"  ✓  Training complete — best model: {best_name}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
