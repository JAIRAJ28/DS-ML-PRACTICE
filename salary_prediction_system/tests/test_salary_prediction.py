from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.salary_prediction import (
    DEFAULT_PROFILE,
    apply_rare_category_mapping,
    load_artifacts,
    predict_salary_usd,
    prepare_input_row,
)


class SalaryPredictionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.model, cls.metadata = load_artifacts()

    def test_metadata_contains_expected_inputs(self) -> None:
        self.assertEqual(
            self.metadata["input_columns"],
            [
                "work_year",
                "experience_level",
                "employment_type",
                "job_title",
                "employee_residence",
                "remote_ratio",
                "company_location",
                "company_size",
            ],
        )

    def test_rare_categories_are_mapped_to_other(self) -> None:
        input_row = pd.DataFrame(
            [
                {
                    **DEFAULT_PROFILE,
                    "job_title": "Financial Data Analyst",
                    "employee_residence": "CH",
                    "company_location": "IE",
                }
            ]
        )

        mapped_row = apply_rare_category_mapping(input_row, self.metadata)

        self.assertEqual(mapped_row["job_title"].iloc[0], "Other")
        self.assertEqual(mapped_row["employee_residence"].iloc[0], "Other")
        self.assertEqual(mapped_row["company_location"].iloc[0], "Other")

    def test_saved_model_can_predict_default_profile(self) -> None:
        prediction = predict_salary_usd(
            DEFAULT_PROFILE,
            model=self.model,
            metadata=self.metadata,
        )

        self.assertGreater(prediction, 0)

    def test_prepare_input_row_orders_columns_for_model(self) -> None:
        input_row = prepare_input_row(DEFAULT_PROFILE, self.metadata)

        self.assertEqual(input_row.columns.tolist(), self.metadata["input_columns"])


if __name__ == "__main__":
    unittest.main()
