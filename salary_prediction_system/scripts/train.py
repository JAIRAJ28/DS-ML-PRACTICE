from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.salary_prediction import METADATA_PATH, MODEL_PATH, PROCESSED_DATA_PATH


TARGET_COLUMN = "salary_in_usd"
INPUT_COLUMNS = [
    "work_year",
    "experience_level",
    "employment_type",
    "job_title",
    "employee_residence",
    "remote_ratio",
    "company_location",
    "company_size",
]
NUMERIC_COLUMNS = ["work_year", "remote_ratio"]
CATEGORICAL_COLUMNS = [
    "experience_level",
    "employment_type",
    "job_title",
    "employee_residence",
    "company_location",
    "company_size",
]
RARE_CATEGORY_COLUMNS = ["job_title", "employee_residence", "company_location"]


def build_rare_category_mapping(
    data: pd.DataFrame,
    columns: list[str],
    rare_threshold: int,
) -> dict[str, dict[str, object]]:
    mapping = {}

    for column in columns:
        counts = data[column].value_counts()
        rare_categories = counts[counts < rare_threshold].index.tolist()
        kept_categories = counts[counts >= rare_threshold].index.tolist()
        mapping[column] = {
            "rare_threshold": rare_threshold,
            "rare_categories": rare_categories,
            "kept_categories": kept_categories,
        }

    return mapping


def apply_rare_mapping(
    data: pd.DataFrame,
    mapping: dict[str, dict[str, object]],
) -> pd.DataFrame:
    mapped_data = data.copy()

    for column, column_mapping in mapping.items():
        kept_categories = column_mapping["kept_categories"]
        mapped_data[column] = mapped_data[column].where(
            mapped_data[column].isin(kept_categories),
            "Other",
        )

    return mapped_data


def build_pipeline(alpha: float) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", "passthrough", NUMERIC_COLUMNS),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                CATEGORICAL_COLUMNS,
            ),
        ]
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", Ridge(alpha=alpha)),
        ]
    )


def train(args: argparse.Namespace) -> dict[str, float]:
    data = pd.read_csv(args.data_path)
    data = data[data[TARGET_COLUMN] <= args.high_salary_limit].copy()

    rare_mapping = build_rare_category_mapping(
        data,
        RARE_CATEGORY_COLUMNS,
        args.rare_threshold,
    )
    model_data = apply_rare_mapping(data, rare_mapping)

    X = model_data[INPUT_COLUMNS]
    y = model_data[TARGET_COLUMN]
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
    )

    evaluation_model = build_pipeline(args.alpha)
    evaluation_model.fit(X_train, y_train)
    predictions = evaluation_model.predict(X_test)
    metrics = {
        "mae": float(mean_absolute_error(y_test, predictions)),
        "rmse": float(mean_squared_error(y_test, predictions) ** 0.5),
        "r2": float(r2_score(y_test, predictions)),
    }

    final_model = build_pipeline(args.alpha)
    final_model.fit(X, y)

    args.model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_model, args.model_path)

    metadata = {
        "model_name": args.model_path.stem,
        "model_type": "Ridge Regression",
        "ridge_alpha": args.alpha,
        "target_column": TARGET_COLUMN,
        "input_columns": INPUT_COLUMNS,
        "numeric_columns": NUMERIC_COLUMNS,
        "categorical_columns": CATEGORICAL_COLUMNS,
        "high_salary_limit": args.high_salary_limit,
        "rare_threshold": args.rare_threshold,
        "rare_category_columns": RARE_CATEGORY_COLUMNS,
        "rare_category_mapping": rare_mapping,
        "training_rows": int(len(X)),
        "test_size": args.test_size,
        "random_state": args.random_state,
        "metrics": metrics,
    }
    args.metadata_path.write_text(json.dumps(metadata, indent=4), encoding="utf-8")

    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the salary prediction model.")
    parser.add_argument("--data-path", type=Path, default=PROCESSED_DATA_PATH)
    parser.add_argument("--model-path", type=Path, default=MODEL_PATH)
    parser.add_argument("--metadata-path", type=Path, default=METADATA_PATH)
    parser.add_argument("--alpha", type=float, default=10.0)
    parser.add_argument("--high-salary-limit", type=int, default=300000)
    parser.add_argument("--rare-threshold", type=int, default=5)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    trained_metrics = train(parse_args())
    print(json.dumps(trained_metrics, indent=2))
