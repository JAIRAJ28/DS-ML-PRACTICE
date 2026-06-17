# AI Salary Prediction

This project is a regression workflow for predicting data-science salary in USD from role, experience, work mode, location, and company attributes.

> Note: the folder is currently named `house_price_prediction_Regression`, but the project content is salary prediction.

## Project Status

The modeling workflow is complete as a local learning project:

- Loads and explores the raw `ds_salaries.csv` dataset.
- Cleans duplicates and leakage columns.
- Trains a baseline linear regression model.
- Improves performance with high-salary outlier handling, rare-category grouping, and Ridge Regression.
- Saves a reusable model pipeline and metadata.
- Provides a Streamlit app for interactive predictions.
- Includes command-line training, command-line prediction, and smoke tests.

## Repository Layout

```text
.
├── app.py
├── data/
│   ├── raw/ds_salaries.csv
│   └── processed/cleaned_salary_data.csv
├── models/
│   ├── salary_prediction_ridge_pipeline.joblib
│   └── salary_prediction_metadata.json
├── notebooks/
│   ├── 01_data_loading.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_data_cleaning.ipynb
│   ├── 04_feature_engineering.ipynb
│   ├── 05_model_training.ipynb
│   ├── 06_model_evaluation.ipynb
│   ├── 07_model_explainability.ipynb
│   ├── 08_model_improvement.ipynb
│   ├── 09_save_model.ipynb
│   └── 10_prediction_demo.ipynb
├── scripts/
│   ├── predict.py
│   └── train.py
├── src/
│   └── salary_prediction.py
├── tests/
│   └── test_salary_prediction.py
└── requirements.txt
```

## Setup

Use Python 3.12.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run the App

```bash
streamlit run app.py
```

The app predicts salary in USD internally and displays the estimate in INR using the conversion value in `app.py`.

## Reproduce Training

The notebook workflow can be rerun in order from `01_data_loading.ipynb` through `10_prediction_demo.ipynb`.

For a script-based training run:

```bash
python scripts/train.py
```

This reads `data/processed/cleaned_salary_data.csv`, trains the Ridge pipeline, and writes:

- `models/salary_prediction_ridge_pipeline.joblib`
- `models/salary_prediction_metadata.json`

## Command-Line Prediction

Run with the default profile:

```bash
python scripts/predict.py
```

Run with a custom profile:

```bash
python scripts/predict.py --profile '{"job_title":"Data Scientist","experience_level":"SE","employee_residence":"US","company_location":"US"}'
```

Missing profile fields fall back to sensible defaults from `src/salary_prediction.py`.

## Model Summary

Final saved model:

- Model type: Ridge Regression
- Ridge alpha: `10.0`
- Target: `salary_in_usd`
- High salary cap: `$300,000`
- Rare category threshold: fewer than `5` rows
- Input features: `work_year`, `experience_level`, `employment_type`, `job_title`, `employee_residence`, `remote_ratio`, `company_location`, `company_size`

Notebook `08_model_improvement.ipynb` reports approximately:

- Test R2: `0.584`
- Test RMSE: `$37,239`

## Tests

```bash
python -m unittest discover -s tests
```

The tests check metadata shape, rare-category mapping, and a saved-model prediction smoke test.
