# Data

This project uses the `ds_salaries.csv` dataset for data-science salary prediction.

## Files

- `raw/ds_salaries.csv`: original dataset used by the first notebook.
- `processed/cleaned_salary_data.csv`: cleaned dataset after duplicate removal and leakage-column cleanup.

## Target

The regression target is `salary_in_usd`.

## Main Features

- `work_year`
- `experience_level`
- `employment_type`
- `job_title`
- `employee_residence`
- `remote_ratio`
- `company_location`
- `company_size`

## Preprocessing Summary

Notebook `03_data_cleaning.ipynb` removes duplicates and drops leakage columns such as the original salary amount and salary currency. Later notebooks cap high salary outliers above `$300,000` and group rare job/location categories as `Other` before training the final Ridge Regression model.
