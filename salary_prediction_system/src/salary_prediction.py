from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
PROCESSED_DATA_PATH = DATA_DIR / "processed" / "cleaned_salary_data.csv"
MODELS_DIR = ROOT_DIR / "models"
MODEL_PATH = MODELS_DIR / "salary_prediction_ridge_pipeline.joblib"
METADATA_PATH = MODELS_DIR / "salary_prediction_metadata.json"


DEFAULT_PROFILE = {
    "work_year": 2022,
    "experience_level": "SE",
    "employment_type": "FT",
    "job_title": "Data Scientist",
    "employee_residence": "US",
    "remote_ratio": 100,
    "company_location": "US",
    "company_size": "M",
}


def load_metadata(metadata_path: Path = METADATA_PATH) -> dict[str, Any]:
    with metadata_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_model(model_path: Path = MODEL_PATH) -> Any:
    return joblib.load(model_path)


def load_artifacts(
    model_path: Path = MODEL_PATH,
    metadata_path: Path = METADATA_PATH,
) -> tuple[Any, dict[str, Any]]:
    return load_model(model_path), load_metadata(metadata_path)


def category_options(metadata: dict[str, Any], column: str) -> list[str]:
    if column in metadata["rare_category_mapping"]:
        return metadata["rare_category_mapping"][column]["kept_categories"]
    return []


def apply_rare_category_mapping(
    input_data: pd.DataFrame,
    metadata: dict[str, Any],
) -> pd.DataFrame:
    mapped_data = input_data.copy()

    for column in metadata["rare_category_columns"]:
        kept_categories = metadata["rare_category_mapping"][column]["kept_categories"]
        mapped_data[column] = mapped_data[column].where(
            mapped_data[column].isin(kept_categories),
            "Other",
        )

    return mapped_data


def prepare_input_row(profile: dict[str, Any], metadata: dict[str, Any]) -> pd.DataFrame:
    input_row = pd.DataFrame([{**DEFAULT_PROFILE, **profile}])
    return input_row[metadata["input_columns"]]


def predict_salary_usd(
    profile: dict[str, Any],
    model: Any | None = None,
    metadata: dict[str, Any] | None = None,
) -> float:
    if model is None or metadata is None:
        model, metadata = load_artifacts()

    input_row = prepare_input_row(profile, metadata)
    mapped_input = apply_rare_category_mapping(input_row, metadata)
    return max(float(model.predict(mapped_input)[0]), 0.0)
