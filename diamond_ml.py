"""Machine-learning pipeline for the diamond price Streamlit prototype."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor


RANDOM_STATE = 42
CATEGORICAL_FEATURES = ["cut", "color", "clarity"]
NUMERICAL_FEATURES = ["carat", "depth", "table", "x", "y", "z"]
MODEL_FEATURES = [
    "carat",
    "cut",
    "color",
    "clarity",
    "depth",
    "table",
    "x",
    "y",
    "z",
]


MODEL_RATIONALE = pd.DataFrame(
    [
        {
            "Model": "Linear regression (baseline)",
            "Role": "Transparent baseline",
            "Why it is included": "Tests whether a simple additive relationship is sufficient.",
            "Configuration": "Default ordinary least squares",
        },
        {
            "Model": "Decision tree",
            "Role": "Single nonlinear learner",
            "Why it is included": "Captures thresholds and feature interactions without scaling.",
            "Configuration": "max_depth=12, min_samples_leaf=4",
        },
        {
            "Model": "Random forest",
            "Role": "Bagging ensemble",
            "Why it is included": "Reduces the variance of an individual tree.",
            "Configuration": "250 trees, max_features=sqrt",
        },
        {
            "Model": "Gradient boosting",
            "Role": "Sequential ensemble",
            "Why it is included": "Corrects earlier residuals and models nonlinear price structure.",
            "Configuration": "250 trees, learning_rate=0.05, Huber loss",
        },
        {
            "Model": "XGBoost",
            "Role": "Regularised boosted trees",
            "Why it is included": "Combines nonlinear learning, sampling and regularisation.",
            "Configuration": "Early stopping selects the final tree count",
        },
    ]
)


def create_preprocessor() -> ColumnTransformer:
    """Create a leakage-safe categorical and numerical transformer."""
    return ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
            ("numerical", "passthrough", NUMERICAL_FEATURES),
        ]
    )


def create_models(best_xgb_estimators: int) -> dict[str, object]:
    """Return the five project models with their documented configurations."""
    return {
        "Linear regression (baseline)": LinearRegression(),
        "Decision tree": DecisionTreeRegressor(
            max_depth=12,
            min_samples_leaf=4,
            random_state=RANDOM_STATE,
        ),
        "Random forest": RandomForestRegressor(
            n_estimators=250,
            max_features="sqrt",
            min_samples_leaf=1,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "Gradient boosting": GradientBoostingRegressor(
            n_estimators=250,
            learning_rate=0.05,
            max_depth=3,
            loss="huber",
            random_state=RANDOM_STATE,
        ),
        "XGBoost": XGBRegressor(
            n_estimators=best_xgb_estimators,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1,
            objective="reg:squarederror",
            eval_metric="rmse",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }


@st.cache_resource(show_spinner=False)
def train_model_bundle(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> dict[str, object]:
    """Select XGBoost rounds with validation data, then fit all final models."""
    X_es_train, X_val, y_es_train, y_val = train_test_split(
        X_train,
        y_train,
        test_size=0.10,
        random_state=RANDOM_STATE,
    )

    early_preprocessor = create_preprocessor()
    X_es_processed = early_preprocessor.fit_transform(X_es_train)
    X_val_processed = early_preprocessor.transform(X_val)

    early_model = XGBRegressor(
        n_estimators=3000,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1,
        objective="reg:squarederror",
        eval_metric="rmse",
        early_stopping_rounds=50,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    early_model.fit(
        X_es_processed,
        y_es_train,
        eval_set=[(X_val_processed, y_val)],
        verbose=False,
    )

    best_iteration = int(early_model.best_iteration)
    best_n_estimators = best_iteration + 1
    validation_rmse = early_model.evals_result()["validation_0"]["rmse"]

    trained_models: dict[str, Pipeline] = {}
    for model_name, model in create_models(best_n_estimators).items():
        pipeline = Pipeline(
            steps=[
                ("preprocessing", create_preprocessor()),
                ("model", model),
            ]
        )
        pipeline.fit(X_train, y_train)
        trained_models[model_name] = pipeline

    return {
        "models": trained_models,
        "best_iteration": best_iteration,
        "best_n_estimators": best_n_estimators,
        "validation_rmse": validation_rmse,
    }


@st.cache_data(show_spinner=False)
def evaluate_models(
    _trained_models: dict[str, Pipeline],
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Evaluate every fitted pipeline on the untouched holdout set."""
    rows: list[dict[str, float | str]] = []
    predictions: dict[str, np.ndarray] = {}

    for model_name, pipeline in _trained_models.items():
        y_pred = pipeline.predict(X_test)
        predictions[model_name] = y_pred
        rows.append(
            {
                "Model": model_name,
                "MAE": mean_absolute_error(y_test, y_pred),
                "RMSE": float(np.sqrt(mean_squared_error(y_test, y_pred))),
                "MAPE (%)": mean_absolute_percentage_error(y_test, y_pred) * 100,
                "R²": r2_score(y_test, y_pred),
            }
        )

    results = pd.DataFrame(rows).sort_values("RMSE").reset_index(drop=True)
    baseline_rmse = float(
        results.loc[results["Model"] == "Linear regression (baseline)", "RMSE"].iloc[0]
    )
    results.insert(0, "Rank", np.arange(1, len(results) + 1))
    results["RMSE gain vs baseline (%)"] = (
        (baseline_rmse - results["RMSE"]) / baseline_rmse * 100
    )
    return results, predictions


@st.cache_data(show_spinner=False)
def calculate_permutation_importance(
    model_name: str,
    _pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> pd.DataFrame:
    """Calculate model-agnostic importance on a reproducible holdout sample."""
    del model_name  # Included in the cache key; the pipeline itself is intentionally excluded.
    sample_size = min(1200, len(X_test))
    sample_index = X_test.sample(n=sample_size, random_state=RANDOM_STATE).index
    result = permutation_importance(
        _pipeline,
        X_test.loc[sample_index],
        y_test.loc[sample_index],
        scoring="neg_root_mean_squared_error",
        n_repeats=3,
        random_state=RANDOM_STATE,
        n_jobs=1,
    )
    return (
        pd.DataFrame(
            {
                "Feature": X_test.columns,
                "Importance": result.importances_mean,
            }
        )
        .sort_values("Importance", ascending=False)
        .reset_index(drop=True)
    )
