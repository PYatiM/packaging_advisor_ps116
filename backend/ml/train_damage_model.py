"""Damage-Risk ML Training & Calibration Pipeline.

Generates 12,000 physics-grounded shipment samples, trains and calibrates
XGBoost, RandomForest, and HistGradientBoosting classifiers with isotonic
regression (CalibratedClassifierCV), evaluates ROC-AUC, Average Precision,
F1, Accuracy, and Brier Score, and persists the best model + scaler.

Run:
    python -m backend.ml.train_damage_model
"""

import json
import os
import warnings
import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from backend.ml.dataset_generator import FEATURE_COLS, generate_dataset

warnings.filterwarnings("ignore", category=FutureWarning)

ARTIFACT_DIR = os.path.join("backend", "ml", "artifacts")
DATA_DIR = os.path.join("backend", "ml", "data")
LABEL_COL = "damaged"


def ensure_dirs():
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)


def prepare_data(n_samples: int = 12_000, test_size: float = 0.20, seed: int = 42):
    df = generate_dataset(n_samples=n_samples, seed=seed)
    csv_path = os.path.join(DATA_DIR, "damage_dataset.csv")
    df.to_csv(csv_path, index=False)
    print(f"[DATA]  {len(df)} samples -> {csv_path}")
    print(
        f"[DATA]  Class balance -- damaged: {df[LABEL_COL].mean():.2%}, "
        f"intact: {1 - df[LABEL_COL].mean():.2%}"
    )

    X = df[FEATURE_COLS].values
    y = df[LABEL_COL].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)
    return X_train_sc, X_test_sc, y_train, y_test, scaler, df


def build_models(seed: int = 42):
    base_models = {
        "XGBoost": XGBClassifier(
            n_estimators=220,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.85,
            colsample_bytree=0.90,
            min_child_weight=5,
            reg_alpha=0.2,
            reg_lambda=1.5,
            eval_metric="logloss",
            n_jobs=1,
            random_state=seed,
            verbosity=0,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=350,
            max_depth=12,
            min_samples_split=5,
            min_samples_leaf=3,
            max_features="sqrt",
            random_state=seed,
            n_jobs=-1,
        ),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            max_iter=300,
            max_depth=6,
            learning_rate=0.06,
            min_samples_leaf=20,
            l2_regularization=0.5,
            random_state=seed,
        ),
    }
    return {
        name: CalibratedClassifierCV(estimator=est, method="isotonic", cv=3)
        for name, est in base_models.items()
    }


def evaluate_model(name, model, X_train, X_test, y_train, y_test):
    print(f"\n{'=' * 60}")
    print(f"  Training & Calibrating: {name}")
    print(f"{'=' * 60}")

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = np.clip(model.predict_proba(X_test)[:, 1], 1e-6, 1.0 - 1e-6)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    ap = average_precision_score(y_test, y_prob)
    brier = brier_score_loss(y_test, y_prob)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_auc = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc")

    metrics = {
        "model": name,
        "accuracy": round(float(acc), 4),
        "f1_score": round(float(f1), 4),
        "roc_auc": round(float(auc), 4),
        "avg_precision": round(float(ap), 4),
        "brier_score": round(float(brier), 4),
        "cv_auc_mean": round(float(cv_auc.mean()), 4),
        "cv_auc_std": round(float(cv_auc.std()), 4),
    }

    print(f"\n  Accuracy        : {acc:.4f}")
    print(f"  F1 Score        : {f1:.4f}")
    print(f"  ROC-AUC         : {auc:.4f}")
    print(f"  Avg Precision   : {ap:.4f}")
    print(f"  Brier Score     : {brier:.4f}")
    print(f"  5-Fold CV AUC   : {cv_auc.mean():.4f} +/- {cv_auc.std():.4f}")
    print(
        f"\n  Classification Report:\n"
        f"{classification_report(y_test, y_pred, target_names=['Intact', 'Damaged'])}"
    )
    return model, y_pred, y_prob, metrics


def get_importance(name, model, X_test, y_test, feature_names):
    """Extract normalized feature importance from calibrated ensembles."""
    fold_importances = []
    if hasattr(model, "calibrated_classifiers_"):
        for cc in model.calibrated_classifiers_:
            est = getattr(cc, "estimator", None)
            if est is not None and hasattr(est, "feature_importances_"):
                fold_importances.append(est.feature_importances_)

    if fold_importances:
        imp = np.mean(fold_importances, axis=0)
    else:
        perm = permutation_importance(
            model, X_test, y_test, n_repeats=5, random_state=42, scoring="roc_auc"
        )
        imp = np.maximum(0.0, perm.importances_mean)
        total = np.sum(imp)
        if total > 0:
            imp = imp / total

    imp_df = (
        pd.DataFrame({"feature": feature_names, "importance": imp})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

    print(f"\n  Feature Importance ({name}):")
    for _, row in imp_df.iterrows():
        bar = "#" * int(row["importance"] * 50)
        print(f"    {row['feature']:26s}  {row['importance']:.4f}  {bar}")

    return imp_df


def plot_results(results: dict, y_test):
    """Generate ROC, PR, calibration, confusion matrix, and feature importance dashboard."""
    sns.set_theme(style="whitegrid", font_scale=1.0)
    n_models = len(results)
    fig, axes = plt.subplots(2, max(3, n_models), figsize=(22, 12))
    fig.suptitle(
        "Calibrated Damage-Risk Classifiers -- Evaluation Dashboard",
        fontsize=15,
        fontweight="bold",
        y=1.01,
    )

    palette = {
        "XGBoost": "#2563eb",
        "RandomForest": "#16a34a",
        "HistGradientBoosting": "#d97706",
    }

    # ROC Curves
    ax = axes[0, 0]
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Random")
    for name, res in results.items():
        fpr, tpr, _ = roc_curve(y_test, res["y_prob"])
        ax.plot(
            fpr,
            tpr,
            color=palette.get(name, "#334155"),
            lw=2,
            label=f"{name} (AUC={res['metrics']['roc_auc']:.3f})",
        )
    ax.set(xlabel="False Positive Rate", ylabel="True Positive Rate", title="ROC Curves")
    ax.legend(loc="lower right")

    # Precision-Recall Curves
    ax = axes[0, 1]
    for name, res in results.items():
        prec, rec, _ = precision_recall_curve(y_test, res["y_prob"])
        ax.plot(
            rec,
            prec,
            color=palette.get(name, "#334155"),
            lw=2,
            label=f"{name} (AP={res['metrics']['avg_precision']:.3f})",
        )
    ax.set(xlabel="Recall", ylabel="Precision", title="Precision-Recall Curves")
    ax.legend(loc="lower left")

    # Best model confusion matrix
    best_name = max(results, key=lambda k: results[k]["metrics"]["roc_auc"])
    ax_cm = axes[0, 2]
    cm = confusion_matrix(y_test, results[best_name]["y_pred"])
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=ax_cm,
        xticklabels=["Intact", "Damaged"],
        yticklabels=["Intact", "Damaged"],
    )
    ax_cm.set(
        xlabel="Predicted",
        ylabel="Actual",
        title=f"Confusion Matrix -- Best ({best_name})",
    )

    # Feature importances for each model
    for idx, (name, res) in enumerate(results.items()):
        ax_imp = axes[1, idx]
        imp = res["importance"].head(10)
        bars = ax_imp.barh(
            imp["feature"],
            imp["importance"],
            color=palette.get(name, "#334155"),
            alpha=0.85,
        )
        ax_imp.set(xlabel="Importance", title=f"Top Features -- {name}")
        ax_imp.invert_yaxis()
        for bar, val in zip(bars, imp["importance"]):
            ax_imp.text(
                bar.get_width() + 0.004,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}",
                va="center",
                fontsize=8,
            )

    plt.tight_layout()
    plot_path = os.path.join(ARTIFACT_DIR, "evaluation_dashboard.png")
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[PLOT]  Dashboard saved -> {plot_path}")
    return plot_path


def save_best_model(results, scaler, n_train: int, n_test: int):
    best_name = max(
        results,
        key=lambda k: (
            results[k]["metrics"]["roc_auc"],
            -results[k]["metrics"]["brier_score"],
        ),
    )
    best_model = results[best_name]["model"]

    model_path = os.path.join(ARTIFACT_DIR, "best_damage_model.joblib")
    scaler_path = os.path.join(ARTIFACT_DIR, "feature_scaler.joblib")
    meta_path = os.path.join(ARTIFACT_DIR, "model_metadata.json")

    joblib.dump(best_model, model_path)
    joblib.dump(scaler, scaler_path)

    metadata = {
        "best_model": best_name,
        "calibrated": True,
        "calibration_method": "isotonic",
        "feature_columns": FEATURE_COLS,
        "metrics": results[best_name]["metrics"],
        "dataset": "physics_grounded_synthetic_with_logistic_noise",
        "n_train_samples": n_train,
        "n_test_samples": n_test,
    }
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n[SAVE]  Best model ({best_name}) -> {model_path}")
    print(f"[SAVE]  Scaler -> {scaler_path}")
    print(f"[SAVE]  Metadata -> {meta_path}")
    return best_name


def main():
    ensure_dirs()
    X_train, X_test, y_train, y_test, scaler, _ = prepare_data()
    models = build_models()

    results = {}
    for name, model in models.items():
        fitted, y_pred, y_prob, metrics = evaluate_model(
            name, model, X_train, X_test, y_train, y_test
        )
        importance = get_importance(name, fitted, X_test, y_test, FEATURE_COLS)
        results[name] = {
            "model": fitted,
            "y_pred": y_pred,
            "y_prob": y_prob,
            "metrics": metrics,
            "importance": importance,
        }

    plot_results(results, y_test)
    best_name = save_best_model(results, scaler, len(X_train), len(X_test))

    summary = {name: res["metrics"] for name, res in results.items()}
    summary_path = os.path.join(ARTIFACT_DIR, "evaluation_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[SAVE]  Summary -> {summary_path}")
    print(f"\n{'=' * 60}")
    print(f"  Training complete -- best calibrated model: {best_name}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
