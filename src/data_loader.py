"""
data_loader.py
==============
Fetches real wind generation and weather data from:
- Open-Meteo API (free, no API key needed)
- Simulated ENTSO-E style generation data

Author: Hemanth Arepelly
"""

import requests
import requests_cache
import pandas as pd
import numpy as np
from retry_requests import retry
import logging
from datetime import datetime, timedelta

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Setup cached session for API calls
cache_session = requests_cache.CachedSession(".cache", expire_after=3600)
retry_session = retry(cache_session, retries=3, backoff_factor=0.2)


def fetch_weather_data(
    latitude: float = 52.52,
    longitude: float = 13.41,
    days: int = 365
) -> pd.DataFrame:
    """
    Fetch historical hourly weather data from Open-Meteo API.

    Args:
        latitude: Location latitude (default: Berlin, Germany)
        longitude: Location longitude
        days: Number of historical days to fetch

    Returns:
        DataFrame with hourly weather variables
    """
    logger.info(f"Fetching weather data for lat={latitude}, lon={longitude}")

    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")

    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": [
            "wind_speed_10m",
            "wind_speed_100m",
            "wind_direction_10m",
            "wind_direction_100m",
            "wind_gusts_10m",
            "temperature_2m",
            "surface_pressure",
            "cloud_cover",
            "precipitation",
        ],
        "timezone": "Europe/Berlin",
    }

    try:
        response = retry_session.get(url, params=params)
        response.raise_for_status()
        data = response.json()

        df = pd.DataFrame(data["hourly"])
        df["timestamp"] = pd.to_datetime(df["time"])
        df = df.drop(columns=["time"])
        df = df.set_index("timestamp")

        # Drop rows with all NaN weather values
        df = df.dropna(how="all")

        logger.info(f"✅ Weather data fetched: {len(df)} rows")
        return df

    except Exception as e:
        logger.error(f"❌ Failed to fetch weather data: {e}")
        raise


def generate_wind_generation(weather_df: pd.DataFrame) -> pd.DataFrame:
    """
    Simulate realistic wind generation from weather data.
    Uses a power curve model: P = 0.5 * rho * A * Cp * v^3

    Args:
        weather_df: DataFrame with wind speed columns

    Returns:
        DataFrame with simulated wind generation in MW
    """
    logger.info("Generating wind power from weather data...")

    df = weather_df.copy()

    # Wind turbine parameters (realistic onshore wind farm)
    rho = 1.225        # Air density kg/m3
    A = 5026.0         # Rotor swept area m2 (80m diameter turbine)
    Cp = 0.35          # Power coefficient
    n_turbines = 50    # Number of turbines in farm
    rated_power = 2.0  # MW per turbine
    cut_in = 3.0       # Cut-in wind speed m/s
    cut_out = 25.0     # Cut-out wind speed m/s
    rated_speed = 12.0 # Rated wind speed m/s

    wind_speed = df["wind_speed_100m"].fillna(df["wind_speed_10m"])

    def power_curve(v):
        """Realistic wind turbine power curve."""
        if v < cut_in or v > cut_out:
            return 0.0
        elif v >= rated_speed:
            return rated_power
        else:
            return rated_power * ((v - cut_in) / (rated_speed - cut_in)) ** 3

    # Apply power curve
    df["wind_generation_mw"] = wind_speed.apply(power_curve) * n_turbines

    # Add realistic noise
    noise = np.random.normal(0, 2.0, len(df))
    df["wind_generation_mw"] = (df["wind_generation_mw"] + noise).clip(0, n_turbines * rated_power)

    logger.info(f"✅ Wind generation simulated: mean={df['wind_generation_mw'].mean():.1f} MW")
    return df


def load_full_dataset(
    latitude: float = 52.52,
    longitude: float = 13.41,
    days: int = 365
) -> pd.DataFrame:
    """
    Main function to load complete dataset.

    Args:
        latitude: Farm latitude
        longitude: Farm longitude
        days: History in days

    Returns:
        Complete DataFrame ready for feature engineering
    """
    weather_df = fetch_weather_data(latitude, longitude, days)
    full_df = generate_wind_generation(weather_df)

    # Basic quality checks
    assert len(full_df) > 0, "Empty dataset returned"
    assert "wind_generation_mw" in full_df.columns, "Missing target column"
    assert full_df.index.is_monotonic_increasing, "Index not sorted"

    logger.info(f"✅ Full dataset ready: {len(full_df)} rows, {len(full_df.columns)} columns")
    return full_df


if __name__ == "__main__":
    df = load_full_dataset()
    print(df.tail())
    print(df.describe())
