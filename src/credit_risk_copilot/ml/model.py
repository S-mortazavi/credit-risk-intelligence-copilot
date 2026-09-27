from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from credit_risk_copilot.ml.preprocessing import build_preprocessor

def build_baseline_model() -> XGBClassifier:
    """
    Build the baseline XGBoost credit risk model.

    The model is intentionally kept relatively simple.
    Class weighting is not applied because the primary
    objective is probability estimation rather than
    threshold-based classification.

    Returns
    -------
    XGBClassifier
        Unfitted baseline model.
    """

    return XGBClassifier(
        n_estimators=500,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )


def build_pd_pipeline(
    numerical_features: list[str],
    categorical_features: list[str]) -> Pipeline:
    """
    Build the end-to-end PD prediction pipeline.

    The pipeline accepts raw applicant features and performs:

    1. Numerical and categorical preprocessing
    2. Probability of default estimation

    Parameters
    ----------
    numerical_features:
        Names of numerical input features.

    categorical_features:
        Names of categorical input features.

    Returns
    -------
    Pipeline
        Unfitted end-to-end PD pipeline.
    """

    preprocessor = build_preprocessor(
        numerical_features=numerical_features,
        categorical_features=categorical_features,
    )

    model = build_baseline_model()

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )


import numpy as np
import pandas as pd


def predict_pd(
    pipeline: Pipeline,
    applicant_data: pd.DataFrame,
) -> np.ndarray:
    """
    Predict raw probability of default for applicants.

    Parameters
    ----------
    pipeline:
        Fitted end-to-end PD pipeline.

    applicant_data:
        Raw applicant-level features.

    Returns
    -------
    np.ndarray
        Probability of default for each applicant.
    """

    return pipeline.predict_proba(
        applicant_data
    )[:, 1]