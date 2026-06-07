"""
model.py
========
Model training, evaluation and comparison for wind generation forecasting.
Compares: Baseline, Random Forest, XGBoost, LightGBM models.

Author: Hemanth Arepelly
"""

import pandas as pd
import numpy as np
import logging
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import xgboost as xgb
import lightgbm as lgb
import warnings
warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)


def train_test_split_timeseries(
    df: pd.DataFrame,
    target_col: str = "wind_generation_mw",
    test_ratio: float = 0.2
):
    """
    Time-series aware train/test split (no shuffling).

    Args:
        df: Feature DataFrame
        target_col: Target column name
        test_ratio: Fraction of data for testing

    Returns:
        X_train, X_test, y_train, y_test
    """
    split_idx = int(len(df) * (1 - test_ratio))

    feature_cols = [col for col in df.columns if col != target_col]

    X_train = df[feature_cols].iloc[:split_idx]
    X_test = df[feature_cols].iloc[split_idx:]
    y_train = df[target_col].iloc[:split_idx]
    y_test = df[target_col].iloc[split_idx:]

    logger.info(f"✅ Train: {len(X_train)} rows | Test: {len(X_test)} rows")
    return X_train, X_test, y_train, y_test


def compute_metrics(y_true: pd.Series, y_pred: np.ndarray, model_name: str) -> dict:
    """
    Compute comprehensive evaluation metrics.

    Args:
        y_true: Actual values
        y_pred: Predicted values
        model_name: Name of model for logging

    Returns:
        Dictionary of metrics
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 0.001))) * 100

    # Directional accuracy
    actual_diff = np.diff(y_true.values)
    pred_diff = np.diff(y_pred)
    directional_acc = np.mean(np.sign(actual_diff) == np.sign(pred_diff)) * 100

    metrics = {
        "model": model_name,
        "MAE": round(mae, 3),
        "RMSE": round(rmse, 3),
        "MAPE": round(mape, 2),
        "Directional_Accuracy": round(directional_acc, 2),
    }

    logger.info(f"📊 {model_name} → MAE: {mae:.2f} | RMSE: {rmse:.2f} | MAPE: {mape:.2f}% | Dir.Acc: {directional_acc:.2f}%")
    return metrics


def train_baseline(y_train: pd.Series, y_test: pd.Series) -> dict:
    """
    Persistence baseline: predict last known value.

    Args:
        y_train: Training target
        y_test: Test target

    Returns:
        Metrics dictionary
    """
    y_pred = y_test.shift(1).fillna(y_train.iloc[-1]).values
    return compute_metrics(y_test, y_pred, "Persistence Baseline")


def train_random_forest(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series
) -> tuple:
    """
    Train Random Forest model.

    Returns:
        (model, metrics, predictions)
    """
    logger.info("Training Random Forest...")
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=12,
        min_samples_leaf=5,
        n_jobs=-1,
        random_state=42
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    metrics = compute_metrics(y_test, y_pred, "Random Forest")
    return model, metrics, y_pred


def train_xgboost(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series
) -> tuple:
    """
    Train XGBoost model.

    Returns:
        (model, metrics, predictions)
    """
    logger.info("Training XGBoost...")
    model = xgb.XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        early_stopping_rounds=50,
        random_state=42,
        verbosity=0
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )
    y_pred = model.predict(X_test)
    metrics = compute_metrics(y_test, y_pred, "XGBoost")
    return model, metrics, y_pred


def train_lightgbm(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series
) -> tuple:
    """
    Train LightGBM model.

    Returns:
        (model, metrics, predictions)
    """
    logger.info("Training LightGBM...")
    model = lgb.LGBMRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        callbacks=[lgb.early_stopping(50, verbose=False)]
    )
    y_pred = model.predict(X_test)
    metrics = compute_metrics(y_test, y_pred, "LightGBM")
    return model, metrics, y_pred


def get_feature_importance(model, feature_cols: list, top_n: int = 15) -> pd.DataFrame:
    """
    Extract top N feature importances.

    Args:
        model: Trained model with feature_importances_
        feature_cols: List of feature names
        top_n: Number of top features to return

    Returns:
        DataFrame with feature importances
    """
    importance_df = pd.DataFrame({
        "feature": feature_cols,
        "importance": model.feature_importances_
    })
    return importance_df.sort_values("importance", ascending=False).head(top_n)


def train_all_models(df: pd.DataFrame) -> dict:
    """
    Master function — trains all models and returns results.

    Args:
        df: Feature-engineered DataFrame

    Returns:
        Dictionary with all models, metrics and predictions
    """
    logger.info("=" * 50)
    logger.info("Starting model training pipeline...")
    logger.info("=" * 50)

    X_train, X_test, y_train, y_test = train_test_split_timeseries(df)
    feature_cols = X_train.columns.tolist()

    results = {}

    # Baseline
    baseline_metrics = train_baseline(y_train, y_test)
    results["baseline"] = {"metrics": baseline_metrics}

    # Random Forest
    rf_model, rf_metrics, rf_pred = train_random_forest(X_train, X_test, y_train, y_test)
    results["random_forest"] = {
        "model": rf_model,
        "metrics": rf_metrics,
        "predictions": rf_pred,
        "feature_importance": get_feature_importance(rf_model, feature_cols)
    }

    # XGBoost
    xgb_model, xgb_metrics, xgb_pred = train_xgboost(X_train, X_test, y_train, y_test)
    results["xgboost"] = {
        "model": xgb_model,
        "metrics": xgb_metrics,
        "predictions": xgb_pred,
        "feature_importance": get_feature_importance(xgb_model, feature_cols)
    }

    # LightGBM
    lgb_model, lgb_metrics, lgb_pred = train_lightgbm(X_train, X_test, y_train, y_test)
    results["lightgbm"] = {
        "model": lgb_model,
        "metrics": lgb_metrics,
        "predictions": lgb_pred,
        "feature_importance": get_feature_importance(lgb_model, feature_cols)
    }

    # Best model
    model_names = ["random_forest", "xgboost", "lightgbm"]
    best = min(model_names, key=lambda m: results[m]["metrics"]["RMSE"])
    results["best_model_name"] = best
    results["X_test"] = X_test
    results["y_test"] = y_test
    results["feature_cols"] = feature_cols

    logger.info(f"🏆 Best model: {best} with RMSE={results[best]['metrics']['RMSE']}")
    return results
