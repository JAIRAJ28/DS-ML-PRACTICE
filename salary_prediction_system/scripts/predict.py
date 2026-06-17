from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.salary_prediction import DEFAULT_PROFILE, predict_salary_usd


def parse_profile(profile_json: str | None) -> dict[str, object]:
    if not profile_json:
        return DEFAULT_PROFILE

    profile = json.loads(profile_json)
    if not isinstance(profile, dict):
        raise ValueError("--profile must be a JSON object.")

    return {**DEFAULT_PROFILE, **profile}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict salary from a profile JSON.")
    parser.add_argument(
        "--profile",
        help="JSON object with model input fields. Missing fields use sensible defaults.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    salary_usd = predict_salary_usd(parse_profile(args.profile))
    print(json.dumps({"predicted_salary_usd": round(salary_usd, 2)}, indent=2))
