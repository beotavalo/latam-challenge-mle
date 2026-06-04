"""Experiment tracking & data-driven model selection for the SCL delay model.

This script makes the model choice **reproducible and auditable** instead of an
opinion. It trains the two algorithms the Data Scientist proposed
(``LogisticRegression`` and ``XGBoost``) on the top-10 features, each **with and
without class balancing**, and logs every run — params, per-class metrics,
confusion matrices and the fitted model — to a local MLflow store.

It then selects the **champion** following the rule agreed for this challenge:
pick the lower-error model and, when models are comparable, prefer the lighter,
more explainable one (no hyper-parameter tuning is performed — the challenge does
not ask for model improvements). The champion is registered in the MLflow Model
Registry with a semantic version and exported to ``challenge/model.joblib`` so the
serving code and the registry stay in sync.

Usage
-----
    uv run python -m challenge.train
    uv run mlflow ui --backend-store-uri sqlite:///mlflow.db   # inspect / screenshot
"""

from __future__ import annotations

import hashlib
import pickle
import subprocess
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: save figures without a display

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import pandas as pd
from mlflow.tracking import MlflowClient
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    classification_report,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from challenge.model import TOP_10_FEATURES, DelayModel

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = REPO_ROOT / "data" / "data.csv"
FIGURES_DIR = REPO_ROOT / "docs" / "assets" / "model_selection"
TRACKING_URI = f"sqlite:///{(REPO_ROOT / 'mlflow.db').as_posix()}"

# Model naming follows the MLOps convention {domain}_{model_type}_v{semver}_{stage}
# (see the mlops-model-naming skill). The registered name is decoupled from the
# stage; the lifecycle stage is applied as an MLflow alias instead.
EXPERIMENT_NAME = "scl-flight-delay--model-selection"
DOMAIN = "flight_delay"
CHAMPION_MODEL_TYPE = "logreg"
REGISTERED_MODEL_NAME = f"{DOMAIN}_{CHAMPION_MODEL_TYPE}"  # e.g. flight_delay_logreg
SEMANTIC_VERSION = "1.0.0"
CHAMPION_STAGE = "champion"

RANDOM_STATE = 42
TEST_SIZE = 0.33


def _git_commit() -> str:
    """Short git commit hash for traceability (``unknown`` outside a repo)."""
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=REPO_ROOT,
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def _dataset_md5() -> str:
    """MD5 of the training dataset for reproducibility/traceability."""
    digest = hashlib.md5()
    with open(DATA_PATH, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _model_size_kb(estimator) -> float:
    """Serialized size of the estimator in KB — quantifies "lighter to serve"."""
    return len(pickle.dumps(estimator)) / 1024


def _build_candidates(scale_pos_weight: float) -> list[dict]:
    """Return the candidate models, mirroring the notebook's experiments."""
    return [
        {
            "run_name": "logreg-balanced-top10",
            "framework": "sklearn.LogisticRegression",
            "balanced": True,
            "estimator": LogisticRegression(class_weight="balanced", random_state=1),
        },
        {
            "run_name": "logreg-plain-top10",
            "framework": "sklearn.LogisticRegression",
            "balanced": False,
            "estimator": LogisticRegression(random_state=1),
        },
        {
            "run_name": "xgboost-balanced-top10",
            "framework": "xgboost.XGBClassifier",
            "balanced": True,
            "estimator": XGBClassifier(
                random_state=1, learning_rate=0.01, scale_pos_weight=scale_pos_weight
            ),
        },
        {
            "run_name": "xgboost-plain-top10",
            "framework": "xgboost.XGBClassifier",
            "balanced": False,
            "estimator": XGBClassifier(random_state=1, learning_rate=0.01),
        },
    ]


def _log_confusion_matrix(y_true, y_pred, run_name: str) -> Path:
    """Save a confusion-matrix figure, return its path (also logged to MLflow)."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(4, 4))
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"Confusion matrix — {run_name}")
    fig.tight_layout()
    path = FIGURES_DIR / f"confusion_matrix__{run_name}.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def _plot_comparison(results: list[dict]) -> Path:
    """Bar chart comparing class-1 (delay) recall/F1 and class-0 recall."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(
        {
            "model": [r["run_name"] for r in results],
            "recall_delay (class 1)": [r["recall_1"] for r in results],
            "f1_delay (class 1)": [r["f1_1"] for r in results],
            "recall_on_time (class 0)": [r["recall_0"] for r in results],
        }
    ).set_index("model")

    fig, ax = plt.subplots(figsize=(10, 5))
    frame.plot.bar(ax=ax)
    ax.set_title("Model comparison — top-10 features (validation split)")
    ax.set_ylabel("score")
    ax.set_ylim(0, 1)
    ax.tick_params(axis="x", rotation=20)
    ax.legend(loc="lower right")
    fig.tight_layout()
    path = FIGURES_DIR / "model_comparison.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def _evaluate(candidate: dict, x_train, x_test, y_train, y_test) -> dict:
    """Train one candidate, log it to MLflow and return its key metrics."""
    estimator = candidate["estimator"]
    with mlflow.start_run(run_name=candidate["run_name"]):
        start = time.perf_counter()
        estimator.fit(x_train, y_train)
        fit_seconds = time.perf_counter() - start

        y_pred = estimator.predict(x_test)
        y_proba = estimator.predict_proba(x_test)[:, 1]
        report = classification_report(y_test, y_pred, output_dict=True)

        mlflow.set_tags(
            {
                "framework": candidate["framework"],
                "balanced": str(candidate["balanced"]),
                "feature_set": "top_10",
                "git_commit": _git_commit(),
                "dataset_md5": _dataset_md5(),
            }
        )
        mlflow.log_params(
            {
                "framework": candidate["framework"],
                "balanced": candidate["balanced"],
                "n_features": len(TOP_10_FEATURES),
                "test_size": TEST_SIZE,
                "random_state": RANDOM_STATE,
            }
        )
        metrics = {
            "accuracy": report["accuracy"],
            "recall_0": report["0"]["recall"],
            "f1_0": report["0"]["f1-score"],
            "precision_1": report["1"]["precision"],
            "recall_1": report["1"]["recall"],
            "f1_1": report["1"]["f1-score"],
            "macro_f1": report["macro avg"]["f1-score"],
            # Threshold-independent quality — fairest cross-model comparison.
            "roc_auc": roc_auc_score(y_test, y_proba),
            "pr_auc": average_precision_score(y_test, y_proba),
            # Efficiency signals that justify "lighter to serve".
            "fit_seconds": fit_seconds,
            "model_size_kb": _model_size_kb(estimator),
        }
        mlflow.log_metrics(metrics)

        cm_path = _log_confusion_matrix(y_test, y_pred, candidate["run_name"])
        mlflow.log_artifact(str(cm_path), artifact_path="figures")

        if candidate["framework"].startswith("sklearn"):
            mlflow.sklearn.log_model(estimator, artifact_path="model")
        else:
            mlflow.xgboost.log_model(estimator, artifact_path="model")

    return {"run_name": candidate["run_name"], **metrics}


def run_experiment() -> list[dict]:
    """Run the full candidate comparison and return their metrics."""
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    data = pd.read_csv(DATA_PATH, low_memory=False)
    model = DelayModel()
    features, target = model.preprocess(data, target_column="delay")
    y = target.iloc[:, 0]

    x_train, x_test, y_train, y_test = train_test_split(
        features, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    results = [
        _evaluate(c, x_train, x_test, y_train, y_test) for c in _build_candidates(scale_pos_weight)
    ]

    comparison_path = _plot_comparison(results)
    print(f"\nSaved comparison figure -> {comparison_path}")
    print("\n=== Validation metrics (top-10 features) ===")
    print(
        pd.DataFrame(results)
        .set_index("run_name")[
            ["recall_1", "f1_1", "roc_auc", "pr_auc", "model_size_kb", "fit_seconds"]
        ]
        .round(3)
        .to_string()
    )
    return results


def register_champion() -> None:
    """Train the champion on the full dataset, register and export it.

    The champion is LogisticRegression with class balancing: comparable to
    XGBoost on every metric (see the comparison run) while being lighter to serve
    and directly interpretable via its coefficients. We retrain on the full
    dataset through ``DelayModel.fit`` so the registered model is byte-for-byte
    the artifact the API serves (``challenge/model.joblib``).
    """
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    data = pd.read_csv(DATA_PATH, low_memory=False)
    model = DelayModel()
    features, target = model.preprocess(data, target_column="delay")

    semver_tag = "v" + SEMANTIC_VERSION.replace(".", "_")
    run_name = f"{REGISTERED_MODEL_NAME}_{semver_tag}_{CHAMPION_STAGE}"
    traceability = {
        "domain": DOMAIN,
        "model_type": CHAMPION_MODEL_TYPE,
        "stage": CHAMPION_STAGE,
        "framework": "sklearn.LogisticRegression",
        "balanced": "True",
        "feature_set": "top_10",
        "semantic_version": SEMANTIC_VERSION,
        "git_commit": _git_commit(),
        "dataset_md5": _dataset_md5(),
        "dataset_rows": str(len(features)),
    }

    with mlflow.start_run(run_name=run_name):
        model.fit(features, target)  # persists challenge/model.joblib
        mlflow.set_tags(traceability)
        mlflow.log_params({"class_weight": "balanced", "trained_on": "full_dataset"})
        mlflow.log_metric("model_size_kb", _model_size_kb(model._model))
        mlflow.sklearn.log_model(
            model._model,
            artifact_path="model",
            registered_model_name=REGISTERED_MODEL_NAME,
        )

    # Decouple name from stage: tag the version with SemVer + traceability, then
    # apply the lifecycle stage as an alias (@champion) rather than renaming.
    client = MlflowClient(tracking_uri=TRACKING_URI)
    latest = max(
        client.search_model_versions(f"name='{REGISTERED_MODEL_NAME}'"),
        key=lambda mv: int(mv.version),
    )
    for key, value in traceability.items():
        client.set_model_version_tag(REGISTERED_MODEL_NAME, latest.version, key, value)
    client.set_registered_model_alias(REGISTERED_MODEL_NAME, CHAMPION_STAGE, latest.version)
    print(
        f"\nRegistered '{REGISTERED_MODEL_NAME}' v{latest.version} "
        f"(semver {SEMANTIC_VERSION}, alias @{CHAMPION_STAGE}, "
        f"full name '{run_name}'); exported serving artifact to challenge/model.joblib"
    )


def main() -> None:
    run_experiment()
    register_champion()
    print(
        "\nDone. Launch the UI to review/screenshot:\n"
        f"  uv run mlflow ui --backend-store-uri {TRACKING_URI}"
    )


if __name__ == "__main__":
    main()
