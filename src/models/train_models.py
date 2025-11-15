from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, StandardScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

DATA_DIR = Path("data")
TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"
ARTIFACT_DIR = Path("artifacts")
MODEL_DIR = ARTIFACT_DIR / "models"
REGISTRY_PATH = ARTIFACT_DIR / "model_registry.json"
TARGET_COLUMN = "satisfaction"
ID_COLUMN = "id"


@dataclass
class ModelReport:
    name: str
    type: str
    metrics: Dict[str, Dict[str, float]]
    cv_metrics: Dict[str, Dict[str, float]]
    confusion_matrices: Dict[str, List[List[int]]]
    feature_importances: List[Dict[str, float]]
    artifact_path: str


def ensure_artifact_dirs() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)


def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing dataset at {path}")
    df = pd.read_csv(path)
    df.columns = [col.strip() for col in df.columns]
    return df


def clean_frame(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    unnamed = [col for col in df.columns if col.lower().startswith("unnamed")]
    if unnamed:
        df = df.drop(columns=unnamed)
    df = df.dropna(subset=[TARGET_COLUMN])
    df[TARGET_COLUMN] = (
        df[TARGET_COLUMN]
        .str.strip()
        .str.lower()
        .map({"satisfied": 1, "neutral or dissatisfied": 0})
    )
    df = df.dropna(subset=[TARGET_COLUMN])

    categorical = [
        "Gender",
        "Customer Type",
        "Type of Travel",
        "Class",
    ]

    numeric_cols = [
        col
        for col in df.columns
        if col not in {TARGET_COLUMN, ID_COLUMN, *categorical}
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=numeric_cols, how="all")
    return df


def build_preprocessor(categorical: List[str], numerical: List[str]) -> ColumnTransformer:
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("cat", categorical_pipe, categorical),
            ("num", numeric_pipe, numerical),
        ],
        remainder="drop",
        sparse_threshold=0.0,
    )


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray) -> Dict[str, float]:
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    try:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_score))
    except ValueError:
        metrics["roc_auc"] = float("nan")
    return metrics


def get_predictions(pipeline: Pipeline, X: pd.DataFrame, name: str) -> Tuple[np.ndarray, np.ndarray]:
    if hasattr(pipeline, "predict_proba"):
        y_score = pipeline.predict_proba(X)[:, 1]
    elif hasattr(pipeline, "decision_function"):
        raw_scores = pipeline.decision_function(X)
        raw_scores = np.asarray(raw_scores)
        if raw_scores.ndim > 1:
            raw_scores = raw_scores[:, -1]
        span = raw_scores.max() - raw_scores.min()
        if span == 0:
            span = 1.0
        y_score = (raw_scores - raw_scores.min()) / span
    else:
        y_score = pipeline.predict(X)

    y_pred = pipeline.predict(X)
    if name in {"linear_regression", "polynomial_regression"}:
        y_score = np.clip(np.asarray(y_score), 0, 1)
        y_pred = (y_score >= 0.5).astype(int)
    return y_pred, y_score


def compute_cv_scores(
    name: str,
    factory,
    X: pd.DataFrame,
    y: pd.Series,
    splits: int = 5,
) -> Dict[str, Dict[str, float]]:
    skf = StratifiedKFold(n_splits=splits, shuffle=True, random_state=42)
    fold_metrics: List[Dict[str, float]] = []
    for train_idx, val_idx in skf.split(X, y):
        pipeline = factory()
        pipeline.fit(X.iloc[train_idx], y.iloc[train_idx])
        y_pred, y_score = get_predictions(pipeline, X.iloc[val_idx], name)
        fold_metrics.append(evaluate_predictions(y.iloc[val_idx], y_pred, y_score))

    aggregated_mean: Dict[str, float] = {}
    aggregated_std: Dict[str, float] = {}
    if fold_metrics:
        keys = fold_metrics[0].keys()
        for key in keys:
            values = [metrics[key] for metrics in fold_metrics]
            aggregated_mean[key] = float(np.mean(values))
            aggregated_std[key] = float(np.std(values))
    return {"mean": aggregated_mean, "std": aggregated_std}


def extract_feature_order(preprocessor: ColumnTransformer) -> List[str]:
    feature_names: List[str] = []
    for name, transformer, cols in preprocessor.transformers_:
        if name == "remainder":
            continue
        if hasattr(transformer, "get_feature_names_out"):
            try:
                names = transformer.get_feature_names_out(cols)
            except TypeError:
                names = transformer.get_feature_names_out()
            feature_names.extend(list(names))
        else:
            feature_names.extend(cols)
    return feature_names


def normalize_importances(items: Dict[str, float]) -> List[Dict[str, float]]:
    if not items:
        return []
    total = sum(abs(v) for v in items.values())
    if total == 0:
        total = 1.0
    normalized = sorted(
        (
            {
                "feature": feature,
                "importance": abs(value) / total,
                "weight": float(value),
                "direction": float(np.sign(value)) if value != 0 else 0.0,
            }
            for feature, value in items.items()
        ),
        key=lambda item: item["importance"],
        reverse=True,
    )
    return normalized


def train_models() -> Dict[str, Any]:
    ensure_artifact_dirs()
    train_df = clean_frame(load_dataset(TRAIN_PATH))
    test_df = clean_frame(load_dataset(TEST_PATH))

    categorical = ["Gender", "Customer Type", "Type of Travel", "Class"]
    numerical = [
        col
        for col in train_df.columns
        if col not in {TARGET_COLUMN, ID_COLUMN, *categorical}
    ]

    drop_cols = [col for col in [TARGET_COLUMN, ID_COLUMN] if col in train_df.columns]
    X_train = train_df.drop(columns=drop_cols)
    y_train = train_df[TARGET_COLUMN].astype(int)

    drop_cols_test = [col for col in [TARGET_COLUMN, ID_COLUMN] if col in test_df.columns]
    X_test = test_df.drop(columns=drop_cols_test)
    y_test = test_df[TARGET_COLUMN].astype(int)

    base_rate = float(y_train.mean())
    def make_preprocessor() -> ColumnTransformer:
        return build_preprocessor(categorical, numerical)

    model_factories = {
        "logistic_regression": lambda: Pipeline(
            steps=[
                ("preprocess", make_preprocessor()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=2000,
                        class_weight="balanced",
                        solver="lbfgs",
                    ),
                ),
            ]
        ),
        "linear_regression": lambda: Pipeline(
            steps=[
                ("preprocess", make_preprocessor()),
                ("model", LinearRegression()),
            ]
        ),
        "polynomial_regression": lambda: Pipeline(
            steps=[
                ("preprocess", make_preprocessor()),
                ("poly", PolynomialFeatures(degree=2, include_bias=False)),
                ("model", LinearRegression()),
            ]
        ),
        "knn": lambda: Pipeline(
            steps=[
                ("preprocess", make_preprocessor()),
                ("model", KNeighborsClassifier(n_neighbors=7)),
            ]
        ),
        "decision_tree": lambda: Pipeline(
            steps=[
                ("preprocess", make_preprocessor()),
                (
                    "model",
                    DecisionTreeClassifier(
                        max_depth=12,
                        min_samples_leaf=10,
                        class_weight="balanced",
                        random_state=42,
                    ),
                ),
            ]
        ),
        "svm": lambda: Pipeline(
            steps=[
                ("preprocess", make_preprocessor()),
                (
                    "model",
                    CalibratedClassifierCV(
                        estimator=LinearSVC(
                            class_weight="balanced",
                            random_state=42,
                            max_iter=4000,
                        ),
                        cv=3,
                    ),
                ),
            ]
        ),
    }

    reports: List[ModelReport] = []

    for name, factory in model_factories.items():
        pipeline = factory()
        print(f"Training {name}...")
        pipeline.fit(X_train, y_train)

        train_pred, train_score = get_predictions(pipeline, X_train, name)
        test_pred, test_score = get_predictions(pipeline, X_test, name)

        metrics = {
            "train": evaluate_predictions(y_train, train_pred, train_score),
            "test": evaluate_predictions(y_test, test_pred, test_score),
        }

        confusion_mats = {
            "train": confusion_matrix(y_train, train_pred).tolist(),
            "test": confusion_matrix(y_test, test_pred).tolist(),
        }

        cv_metrics = compute_cv_scores(name, factory, X_train, y_train)

        artifact_path = MODEL_DIR / f"{name}.joblib"
        joblib.dump(pipeline, artifact_path)

        feature_weights: Dict[str, float] = {}
        preprocess_step = pipeline.named_steps["preprocess"]

        if name == "polynomial_regression":
            base_features = preprocess_step.get_feature_names_out()
            poly_step = pipeline.named_steps["poly"]
            feature_names = poly_step.get_feature_names_out(base_features)
            coefs = pipeline.named_steps["model"].coef_.ravel()
            feature_weights = dict(zip(feature_names, coefs))
        else:
            feature_names = extract_feature_order(preprocess_step)
            model = pipeline.named_steps.get("model")
            if hasattr(model, "coef_"):
                coefs = model.coef_.ravel()
                feature_weights = dict(zip(feature_names, coefs))
            elif hasattr(model, "feature_importances_"):
                feature_weights = dict(
                    zip(feature_names, model.feature_importances_)
                )

        reports.append(
            ModelReport(
                name=name,
                type=type(pipeline.named_steps["model"]).__name__,
                metrics=metrics,
                cv_metrics=cv_metrics,
                confusion_matrices=confusion_mats,
                feature_importances=normalize_importances(feature_weights)[:20],
                artifact_path=str(artifact_path),
            )
        )

    best_model = max(
        reports,
        key=lambda item: item.metrics.get("test", {}).get("f1", 0.0),
    )
    global_feature_scores: Dict[str, float] = {}
    for report in reports:
        for record in report.feature_importances:
            feature = record["feature"]
            score = record["importance"]
            global_feature_scores[feature] = global_feature_scores.get(feature, 0.0) + score

    registry = {
        "base_rate": base_rate,
        "models": [asdict(report) for report in reports],
        "best_model": best_model.name,
        "global_feature_rank": normalize_importances(global_feature_scores)[:25],
    }

    with REGISTRY_PATH.open("w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2)

    print(f"Saved registry with best model: {best_model.name}")
    return registry


if __name__ == "__main__":
    train_models()
