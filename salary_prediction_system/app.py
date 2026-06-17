import pandas as pd
import streamlit as st

from src.salary_prediction import (
    apply_rare_category_mapping,
    category_options,
    load_artifacts,
)


EXPERIENCE_LEVELS = {
    "EN": "Entry",
    "MI": "Mid",
    "SE": "Senior",
    "EX": "Executive",
}

EMPLOYMENT_TYPES = {
    "FT": "Full time",
    "PT": "Part time",
    "CT": "Contract",
    "FL": "Freelance",
}

COMPANY_SIZES = {
    "S": "Small",
    "M": "Medium",
    "L": "Large",
}

REMOTE_OPTIONS = {
    "On-site": 0,
    "Hybrid": 50,
    "Remote": 100,
}

USD_TO_INR = 94.56


@st.cache_resource
def load_cached_artifacts():
    return load_artifacts()


def money(value):
    return f"${value:,.0f}"


def rupees(value):
    return f"₹{value:,.0f}"


st.set_page_config(
    page_title="AI Salary Prediction",
    page_icon="",
    layout="wide",
)

st.title("AI Salary Prediction")
st.caption(
    "Ridge Regression model with outlier handling and rare-category grouping. "
    "The model predicts USD internally; I convert the frontend display to INR."
)

model, metadata = load_cached_artifacts()

with st.sidebar:
    st.header("Model")
    st.metric("Model type", metadata["model_type"])
    st.metric("Ridge alpha", metadata["ridge_alpha"])
    st.metric(
        "Training salary cap",
        rupees(metadata["high_salary_limit"] * USD_TO_INR),
    )
    st.caption(f"Conversion used: 1 USD = ₹{USD_TO_INR:.2f}")
    st.caption("Predictions are estimates, not guaranteed compensation.")

left, right = st.columns([1.1, 0.9], gap="large")

with left:
    st.subheader("Profile")

    year_options = [2020, 2021, 2022]
    work_year = st.selectbox("Work year", year_options, index=len(year_options) - 1)

    experience_label = st.selectbox(
        "Experience level",
        list(EXPERIENCE_LEVELS.values()),
        index=2,
    )
    experience_level = {
        label: code for code, label in EXPERIENCE_LEVELS.items()
    }[experience_label]

    employment_label = st.selectbox(
        "Employment type",
        list(EMPLOYMENT_TYPES.values()),
        index=0,
    )
    employment_type = {
        label: code for code, label in EMPLOYMENT_TYPES.items()
    }[employment_label]

    job_titles = category_options(metadata, "job_title")
    job_title = st.selectbox("Job title", job_titles, index=job_titles.index("Data Scientist"))

    residence_options = category_options(metadata, "employee_residence")
    employee_residence = st.selectbox(
        "Employee residence",
        residence_options,
        index=residence_options.index("US"),
    )

    company_options = category_options(metadata, "company_location")
    company_location = st.selectbox(
        "Company location",
        company_options,
        index=company_options.index("US"),
    )

    remote_label = st.segmented_control(
        "Work mode",
        options=list(REMOTE_OPTIONS.keys()),
        default="Remote",
    )
    remote_ratio = REMOTE_OPTIONS[remote_label]

    company_label = st.selectbox(
        "Company size",
        list(COMPANY_SIZES.values()),
        index=1,
    )
    company_size = {
        label: code for code, label in COMPANY_SIZES.items()
    }[company_label]

input_row = pd.DataFrame(
    [
        {
            "work_year": work_year,
            "experience_level": experience_level,
            "employment_type": employment_type,
            "job_title": job_title,
            "employee_residence": employee_residence,
            "remote_ratio": remote_ratio,
            "company_location": company_location,
            "company_size": company_size,
        }
    ]
)

input_row = input_row[metadata["input_columns"]]
mapped_input = apply_rare_category_mapping(input_row, metadata)
raw_prediction = float(model.predict(mapped_input)[0])
display_prediction_usd = max(raw_prediction, 0)
display_prediction_inr = display_prediction_usd * USD_TO_INR

with right:
    st.subheader("Prediction")
    st.metric("Estimated salary", rupees(display_prediction_inr))
    st.caption(f"USD equivalent: {money(display_prediction_usd)}")

    if raw_prediction < 0:
        st.warning(
            "The raw model prediction was below 0, so I clipped it to 0 before converting to INR."
        )

    st.divider()
    st.write("Model input after rare-category handling")
    st.dataframe(mapped_input, use_container_width=True, hide_index=True)

    changed_columns = [
        column
        for column in metadata["rare_category_columns"]
        if input_row[column].iloc[0] != mapped_input[column].iloc[0]
    ]

    if changed_columns:
        st.info(
            "Some rare input values were mapped to `Other`: "
            + ", ".join(changed_columns)
        )
    else:
        st.success("No rare-category mapping was needed for this profile.")

st.divider()

st.subheader("How I Built This Model")
summary_cols = st.columns(4)
summary_cols[0].metric("Original rows", "565")
summary_cols[1].metric("Final rows", "555")
summary_cols[2].metric("Final Test R2", "0.584")
summary_cols[3].metric("Final Test RMSE", rupees(37239 * USD_TO_INR))

st.write(
    "I removed high salary outliers above $300,000, grouped rare job/location "
    "categories into `Other`, then trained a Ridge Regression model with alpha 10. "
    "The trained target is `salary_in_usd`, so INR values here are display conversions."
)
