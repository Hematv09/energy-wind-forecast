"""
features.py
===========
Feature engineering for wind generation forecasting.
Creates 40+ lag, rolling, calendar and weather interaction features.

Author: Hemanth Arepelly
"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


def create_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create calendar-based features.

    Args:
        df: DataFrame with datetime index

    Returns:
        DataFrame with calendar features added
    """
    df["hour"] = df.index.hour
    df["day_of_week"] = df.index.dayofweek
    df["day_of_month"] = df.index.day
    df["month"] = df.index.month
    df["quarter"] = df.index.quarter
    df["week_of_year"] = df.index.isocalendar().week.astype(int)
    df["is_weekend"] = (df.index.dayofweek >= 5).astype(int)
    df["is_night"] = ((df.index.hour >= 22) | (df.index.hour <= 5)).astype(int)

    # Cyclical encoding for hour and month
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

    logger.info("✅ Calendar features created: 14 features")
    return df


def create_lag_features(df: pd.DataFrame, target_col: str = "wind_generation_mw") -> pd.DataFrame:
    """
    Create lag features for time-series forecasting.

    Args:
        df: DataFrame with target column
        target_col: Name of target column

    Returns:
        DataFrame with lag features added
    """
    # Short-term lags (hours)
    for lag in [1, 2, 3, 6, 12, 24]:
        df[f"lag_{lag}h"] = df[target_col].shift(lag)

    # Day-based lags
    for lag in [48, 72, 168]:  # 2 days, 3 days, 1 week
        df[f"lag_{lag}h"] = df[target_col].shift(lag)

    # Same hour yesterday and last week
    df["lag_same_hour_yesterday"] = df[target_col].shift(24)
    df["lag_same_hour_lastweek"] = df[target_col].shift(168)

    logger.info("✅ Lag features created: 10 features")
    return df


def create_rolling_features(df: pd.DataFrame, target_col: str = "wind_generation_mw") -> pd.DataFrame:
    """
    Create rolling window statistical features.

    Args:
        df: DataFrame with target column
        target_col: Name of target column

    Returns:
        DataFrame with rolling features added
    """
    for window in [3, 6, 12, 24, 48, 168]:
        df[f"rolling_mean_{window}h"] = df[target_col].shift(1).rolling(window).mean()
        df[f"rolling_std_{window}h"] = df[target_col].shift(1).rolling(window).std()
        df[f"rolling_max_{window}h"] = df[target_col].shift(1).rolling(window).max()
        df[f"rolling_min_{window}h"] = df[target_col].shift(1).rolling(window).min()

    logger.info("✅ Rolling features created: 24 features")
    return df


def create_weather_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create weather interaction features.

    Args:
        df: DataFrame with weather columns

    Returns:
        DataFrame with weather interaction features
    """
    # Wind speed differences between heights
    if "wind_speed_100m" in df.columns and "wind_speed_10m" in df.columns:
        df["wind_shear"] = df["wind_speed_100m"] - df["wind_speed_10m"]
        df["wind_shear_ratio"] = df["wind_speed_100m"] / (df["wind_speed_10m"] + 0.001)

    # Wind power density
    if "wind_speed_100m" in df.columns:
        df["wind_power_density"] = 0.5 * 1.225 * df["wind_speed_100m"] ** 3

    # Wind direction components
    if "wind_direction_100m" in df.columns:
        df["wind_u"] = df["wind_speed_100m"] * np.sin(np.radians(df["wind_direction_100m"]))
        df["wind_v"] = df["wind_speed_100m"] * np.cos(np.radians(df["wind_direction_100m"]))

    # Temperature effect on air density
    if "temperature_2m" in df.columns:
        df["air_density_approx"] = 1.225 * (273.15 / (df["temperature_2m"] + 273.15))

    logger.info("✅ Weather features created: 6 features")
    return df


def create_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Master function — applies all feature engineering steps.

    Args:
        df: Raw DataFrame from data_loader

    Returns:
        Feature-rich DataFrame ready for model training
    """
    logger.info("Starting feature engineering pipeline...")

    df = create_calendar_features(df)
    df = create_lag_features(df)
    df = create_rolling_features(df)
    df = create_weather_features(df)

    # Drop rows with NaN from lag/rolling features
    initial_rows = len(df)
    df = df.dropna()
    dropped = initial_rows - len(df)

    total_features = len(df.columns) - 1  # exclude target
    logger.info(f"✅ Feature engineering complete:")
    logger.info(f"   Total features: {total_features}")
    logger.info(f"   Rows after dropna: {len(df)} (dropped {dropped})")

    return df


def get_feature_columns(df: pd.DataFrame, target_col: str = "wind_generation_mw") -> list:
    """
    Get list of feature columns excluding target.

    Args:
        df: Feature DataFrame
        target_col: Target column name

    Returns:
        List of feature column names
    """
    exclude = [target_col]
    return [col for col in df.columns if col not in exclude]
