"""
app.py
======
Main Streamlit dashboard for Wind Generation Forecasting System.
Production-grade energy market forecasting application.

Author: Hemanth Arepelly
GitHub: https://github.com/hematv09/energy-wind-forecast
"""

import streamlit as st
import pandas as pd
import numpy as np
import logging
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data_loader import load_full_dataset
from src.features import create_all_features, get_feature_columns
from src.model import train_all_models
from src.visualizations import (
    plot_actual_vs_predicted,
    plot_metrics_comparison,
    plot_feature_importance,
    plot_error_distribution,
    plot_wind_generation_overview
)

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Wind Generation Forecast | Energy Markets",
    page_icon="🌬️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ───────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .metric-card {
        background: linear-gradient(135deg, #1e2130, #2a2f45);
        border-radius: 12px;
        padding: 20px;
        border-left: 4px solid #2E86AB;
        margin: 8px 0;
    }
    .best-model-badge {
        background: #1a472a;
        color: #90BE6D;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 13px;
        font-weight: bold;
    }
    .stMetric label { font-size: 13px !important; color: #aaa !important; }
    .stMetric value { font-size: 28px !important; font-weight: bold !important; }
    div[data-testid="stSidebar"] { background-color: #161b2e; }
</style>
""", unsafe_allow_html=True)


# ── Sidebar ──────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/wind-turbine.png", width=80)
    st.title("⚡ Wind Forecast")
    st.markdown("---")

    st.subheader("📍 Location Settings")
    location = st.selectbox(
        "Select Wind Farm Location",
        ["Berlin, Germany", "London, UK", "Amsterdam, Netherlands",
         "Copenhagen, Denmark", "Madrid, Spain"]
    )

    location_coords = {
        "Berlin, Germany": (52.52, 13.41),
        "London, UK": (51.50, -0.12),
        "Amsterdam, Netherlands": (52.37, 4.90),
        "Copenhagen, Denmark": (55.67, 12.57),
        "Madrid, Spain": (40.41, -3.70)
    }
    lat, lon = location_coords[location]

    st.subheader("📅 Data Settings")
    days = st.slider("Historical Data (days)", 180, 730, 365, 30)

    st.subheader("🤖 Model Settings")
    test_ratio = st.slider("Test Set Size", 0.10, 0.30, 0.20, 0.05)

    st.markdown("---")
    run_btn = st.button("🚀 Run Forecast", type="primary", use_container_width=True)

    st.markdown("---")
    st.markdown("""
    **About**
    - 📊 Real weather data via Open-Meteo API
    - 🔬 40+ engineered features
    - 🤖 XGBoost, LightGBM, Random Forest
    - 📈 Full backtesting metrics
    """)
    st.markdown("[GitHub](https://github.com/hematv09/energy-wind-forecast) | Built by Hemanth Arepelly")


# ── Header ───────────────────────────────────────────────────
st.title("🌬️ Wind Generation Forecasting System")
st.markdown("""
> **Production-grade forecasting platform** for energy markets.
> Fetches real weather data, engineers 40+ features, compares ML models,
> and delivers actionable forecasts with full evaluation metrics.
""")
st.markdown("---")


# ── Main Pipeline ────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def run_pipeline(lat, lon, days, test_ratio):
    """Cached pipeline: load → feature engineer → train → evaluate."""
    raw_df = load_full_dataset(latitude=lat, longitude=lon, days=days)
    feature_df = create_all_features(raw_df)
    results = train_all_models(feature_df)
    return raw_df, feature_df, results


if run_btn or "results" in st.session_state:

    if run_btn:
        with st.spinner("⚡ Fetching real weather data & training models..."):
            try:
                raw_df, feature_df, results = run_pipeline(lat, lon, days, test_ratio)
                st.session_state["results"] = results
                st.session_state["raw_df"] = raw_df
                st.session_state["feature_df"] = feature_df
                st.success("✅ Pipeline complete!")
            except Exception as e:
                st.error(f"❌ Error: {e}")
                st.stop()
    else:
        results = st.session_state["results"]
        raw_df = st.session_state["raw_df"]
        feature_df = st.session_state["feature_df"]

    best_model = results["best_model_name"]
    y_test = results["y_test"]

    # ── KPI Metrics Row ──────────────────────────────────────
    st.subheader("📊 Model Performance Summary")

    col1, col2, col3, col4, col5 = st.columns(5)

    best_metrics = results[best_model]["metrics"]

    with col1:
        st.metric("🏆 Best Model", best_model.replace("_", " ").title())
    with col2:
        st.metric("📉 MAE", f"{best_metrics['MAE']:.2f} MW")
    with col3:
        st.metric("📉 RMSE", f"{best_metrics['RMSE']:.2f} MW")
    with col4:
        st.metric("📊 MAPE", f"{best_metrics['MAPE']:.1f}%")
    with col5:
        st.metric("🎯 Directional Accuracy", f"{best_metrics['Directional_Accuracy']:.1f}%")

    st.markdown("---")

    # ── Data Overview Tab ────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Data Overview",
        "🔮 Forecast Results",
        "📊 Model Comparison",
        "🔬 Feature Analysis",
        "📋 Raw Data"
    ])

    with tab1:
        st.subheader(f"Wind Generation Overview — {location}")
        st.plotly_chart(
            plot_wind_generation_overview(raw_df),
            use_container_width=True
        )

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Records", f"{len(raw_df):,}")
        with col2:
            st.metric("Avg Generation", f"{raw_df['wind_generation_mw'].mean():.1f} MW")
        with col3:
            st.metric("Peak Generation", f"{raw_df['wind_generation_mw'].max():.1f} MW")
        with col4:
            st.metric("Engineered Features", f"{len(feature_df.columns) - 1}")

    with tab2:
        st.subheader("🔮 Actual vs Predicted Wind Generation")

        predictions = {
            name: results[name]["predictions"]
            for name in ["random_forest", "xgboost", "lightgbm"]
        }

        st.plotly_chart(
            plot_actual_vs_predicted(y_test, predictions, best_model),
            use_container_width=True
        )

        st.subheader("🔍 Error Analysis")
        selected_model = st.selectbox(
            "Select model for error analysis",
            ["random_forest", "xgboost", "lightgbm"]
        )
        st.plotly_chart(
            plot_error_distribution(
                y_test,
                results[selected_model]["predictions"],
                selected_model
            ),
            use_container_width=True
        )

    with tab3:
        st.subheader("📊 All Models Comparison")

        all_metrics = [
            results[m]["metrics"]
            for m in ["random_forest", "xgboost", "lightgbm"]
        ]
        all_metrics.append(results["baseline"]["metrics"])

        st.plotly_chart(
            plot_metrics_comparison(all_metrics),
            use_container_width=True
        )

        st.subheader("📋 Detailed Metrics Table")
        metrics_df = pd.DataFrame(all_metrics)
        st.dataframe(
            metrics_df.style.highlight_min(
                subset=["MAE", "RMSE", "MAPE"], color="#1a472a"
            ).highlight_max(
                subset=["Directional_Accuracy"], color="#1a472a"
            ),
            use_container_width=True
        )

    with tab4:
        st.subheader("🔬 Feature Importance Analysis")

        fi_model = st.selectbox(
            "Select model",
            ["random_forest", "xgboost", "lightgbm"],
            key="fi_model"
        )

        st.plotly_chart(
            plot_feature_importance(
                results[fi_model]["feature_importance"],
                fi_model
            ),
            use_container_width=True
        )

        st.info(f"""
        💡 **Feature Engineering Summary**
        - **Calendar features**: hour, day, month, cyclical encodings
        - **Lag features**: 1h, 2h, 3h, 6h, 12h, 24h, 48h, 168h lags
        - **Rolling features**: 3h to 168h rolling mean, std, max, min
        - **Weather features**: wind shear, power density, air density
        - **Total features**: {len(feature_df.columns) - 1}
        """)

    with tab5:
        st.subheader("📋 Raw Dataset")
        st.dataframe(raw_df.tail(200), use_container_width=True)

        csv = raw_df.to_csv().encode("utf-8")
        st.download_button(
            "⬇️ Download Dataset as CSV",
            csv,
            "wind_generation_data.csv",
            "text/csv"
        )

else:
    # ── Landing State ────────────────────────────────────────
    st.info("👈 Configure settings in the sidebar and click **Run Forecast** to start.")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        ### 🌬️ What this does
        - Fetches real hourly weather data
        - Simulates wind farm generation
        - Engineers 40+ ML features
        """)
    with col2:
        st.markdown("""
        ### 🤖 Models compared
        - Persistence Baseline
        - Random Forest
        - XGBoost
        - LightGBM
        """)
    with col3:
        st.markdown("""
        ### 📊 Metrics computed
        - MAE & RMSE
        - MAPE
        - Directional Accuracy
        - Feature Importance
        """)
