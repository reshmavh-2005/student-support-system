"""
Prediction Module for Early Academic Risk Prediction.

This module provides simple, beginner-friendly, reusable functions to load
the trained Logistic Regression pipeline, read the decision threshold,
validate input features, and generate academic risk predictions and
operational priority categories for student enrollments.
"""

from functools import lru_cache
from pathlib import Path
from typing import Union
import joblib
import pandas as pd

# Define project root and default artifact paths using pathlib
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH = PROJECT_ROOT / "artifacts" / "logistic_regression_pipeline.joblib"
DEFAULT_THRESHOLD_PATH = PROJECT_ROOT / "artifacts" / "model_threshold.txt"

# Exact 18 features expected by the trained pipeline in order
MODEL_FEATURES = [
    "code_module",
    "code_presentation",
    "gender",
    "region",
    "highest_education",
    "imd_band",
    "age_band",
    "disability",
    "num_of_prev_attempts",
    "studied_credits",
    "date_registration",
    "total_clicks",
    "active_days",
    "unique_sites",
    "pre_course_clicks",
    "early_assessments_submitted",
    "early_avg_score",
    "has_early_submission"
]


@lru_cache(maxsize=1)
def load_trained_model(model_path: Union[str, Path] = None):
    """
    Load and cache the trained Scikit-learn pipeline using Joblib.
    
    Uses lru_cache to avoid repeatedly reloading the model from disk.
    """
    path = Path(model_path) if model_path else DEFAULT_MODEL_PATH

    if not path.exists():
        raise FileNotFoundError(
            f"Model artifact not found at: {path}. "
            f"Please verify that the model file exists in the artifacts directory."
        )

    try:
        model = joblib.load(path)
        return model
    except Exception as e:
        raise RuntimeError(f"Error loading trained model from {path}: {e}") from e


@lru_cache(maxsize=1)
def load_threshold(threshold_path: Union[str, Path] = None) -> float:
    """
    Load and cache the classification decision threshold from text file.
    """
    path = Path(threshold_path) if threshold_path else DEFAULT_THRESHOLD_PATH

    if not path.exists():
        raise FileNotFoundError(
            f"Threshold file not found at: {path}. "
            f"Please verify that model_threshold.txt exists in the artifacts directory."
        )

    try:
        content = path.read_text(encoding="utf-8").strip()
        threshold = float(content)
        return threshold
    except Exception as e:
        raise RuntimeError(f"Error reading threshold value from {path}: {e}") from e


def assign_priority(probability: float) -> str:
    """
    Assign an operational support priority category based on predicted risk probability.
    
    Rules:
    - High Priority: probability >= 0.50
    - Watchlist:     0.35 <= probability < 0.50
    - Low Risk:      probability < 0.35
    
    Note: These operational categories are proposed support categories,
    not institutionally validated risk levels.
    """
    if probability >= 0.50:
        return "High Priority"
    elif probability >= 0.35:
        return "Watchlist"
    else:
        return "Low Risk"


def predict_student_risk(
    input_df: pd.DataFrame,
    model_path: Union[str, Path] = None,
    threshold_path: Union[str, Path] = None
) -> pd.DataFrame:
    """
    Generate academic risk probabilities, predictions, and operational priorities.
    
    Parameters:
    -----------
    input_df : pd.DataFrame
        DataFrame containing student records with required features.
    model_path : Path or str, optional
        Custom path to the pipeline file (defaults to artifacts/logistic_regression_pipeline.joblib).
    threshold_path : Path or str, optional
        Custom path to the threshold file (defaults to artifacts/model_threshold.txt).
        
    Returns:
    --------
    pd.DataFrame
        Copy of input DataFrame containing original student details (excluding target if present),
        plus new columns: 'risk_probability', 'prediction', and 'priority'.
    """
    # Verify input is a pandas DataFrame
    if not isinstance(input_df, pd.DataFrame):
        raise TypeError(
            f"Expected input_df to be a pandas DataFrame, got {type(input_df).__name__}."
        )

    if input_df.empty:
        raise ValueError("Input DataFrame is empty. Please provide at least one student record.")

    # 1. Validate that all 18 required features exist
    missing_features = [col for col in MODEL_FEATURES if col not in input_df.columns]
    if missing_features:
        raise ValueError(
            f"Input DataFrame is missing {len(missing_features)} required feature(s): {missing_features}. "
            f"The model requires all 18 features: {MODEL_FEATURES}"
        )

    # 2. Load the trained model and decision threshold (cached)
    model = load_trained_model(model_path)
    threshold = load_threshold(threshold_path)

    # 3. Select the exact 18 model features in the correct order
    # Note: id_student, target, and other extra columns are never passed to the model
    features_df = input_df[MODEL_FEATURES].copy()

    # 4. Generate risk probabilities using predict_proba for Class 1 (At Risk)
    # The pipeline handles missing value imputation and scaling internally
    probabilities = model.predict_proba(features_df)[:, 1]

    # 5. Generate binary predictions using the saved threshold (0.40)
    predictions = [
        "At Risk" if prob >= threshold else "Not At Risk"
        for prob in probabilities
    ]

    # 6. Assign operational priority categories
    priorities = [assign_priority(prob) for prob in probabilities]

    # 7. Build output DataFrame retaining available original student information
    output_df = input_df.copy()

    # Omit ground-truth target column if present (never overwrite or use target)
    if "target" in output_df.columns:
        output_df = output_df.drop(columns=["target"])

    # Attach prediction results
    output_df["risk_probability"] = probabilities
    output_df["prediction"] = predictions
    output_df["priority"] = priorities

    return output_df


# Convenient alias
predict = predict_student_risk
