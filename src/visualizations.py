"""
visualizations.py
=================
All Plotly charts for the Streamlit dashboard.

Author: Hemanth Arepelly
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import logging

logger = logging.getLogger(__name__)


def plot_actual_vs_predicted(
    y_test: pd.Series,
    predictions: dict,
    best_model: str
) -> go.Figure:
    """
    Plot actual vs predicted wind generation for all models.

    Args:
        y_test: Actual test values
        predictions: Dict of model_name -> predictions array
        best_model: Name of best performing model

    Returns:
        Plotly figure
    """
    fig = go.Figure()

    # Actual
    fig.add_trace(go.Scatter(
        x=y_test.index,
        y=y_test.values,
        name="Actual",
        line=dict(color="#2E86AB", width=2),
        opacity=0.9
    ))

    colors = {
        "random_forest": "#E84855",
        "xgboost": "#F9C74F",
        "lightgbm": "#90BE6D"
    }

    for model_name, y_pred in predictions.items():
        is_best = model_name == best_model
        fig.add_trace(go.Scatter(
            x=y_test.index,
            y=y_pred,
            name=f"{model_name.replace('_', ' ').title()} {'⭐' if is_best else ''}",
            line=dict(
                color=colors.get(model_name, "#999"),
                width=2.5 if is_best else 1.5,
                dash="solid" if is_best else "dot"
            ),
            opacity=1.0 if is_best else 0.6
        ))

    fig.update_layout(
        title="Actual vs Predicted Wind Generation",
        xaxis_title="Timestamp",
        yaxis_title="Wind Generation (MW)",
        hovermode="x unified",
        template="plotly_dark",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        height=450
    )
    return fig


def plot_metrics_comparison(all_metrics: list) -> go.Figure:
    """
    Bar chart comparing all model metrics.

    Args:
        all_metrics: List of metric dicts from compute_metrics

    Returns:
        Plotly figure
    """
    df_metrics = pd.DataFrame(all_metrics)

    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=("MAE (lower=better)", "RMSE (lower=better)", "Directional Accuracy % (higher=better)")
    )

    colors = ["#E84855", "#F9C74F", "#90BE6D", "#577590"]

    for i, metric in enumerate(["MAE", "RMSE", "Directional_Accuracy"]):
        fig.add_trace(
            go.Bar(
                x=df_metrics["model"],
                y=df_metrics[metric],
                marker_color=colors[:len(df_metrics)],
                showlegend=False,
                text=df_metrics[metric].round(2),
                textposition="outside"
            ),
            row=1, col=i + 1
        )

    fig.update_layout(
        title="Model Performance Comparison",
        template="plotly_dark",
        height=400
    )
    return fig


def plot_feature_importance(importance_df: pd.DataFrame, model_name: str) -> go.Figure:
    """
    Horizontal bar chart of feature importances.

    Args:
        importance_df: DataFrame with feature and importance columns
        model_name: Name of model

    Returns:
        Plotly figure
    """
    fig = go.Figure(go.Bar(
        x=importance_df["importance"],
        y=importance_df["feature"],
        orientation="h",
        marker_color="#2E86AB",
        text=importance_df["importance"].round(4),
        textposition="outside"
    ))

    fig.update_layout(
        title=f"Top Feature Importances — {model_name.replace('_', ' ').title()}",
        xaxis_title="Importance Score",
        yaxis_title="Feature",
        template="plotly_dark",
        height=500,
        yaxis=dict(autorange="reversed")
    )
    return fig


def plot_error_distribution(y_test: pd.Series, y_pred: np.ndarray, model_name: str) -> go.Figure:
    """
    Histogram of prediction errors.

    Args:
        y_test: Actual values
        y_pred: Predicted values
        model_name: Model name for title

    Returns:
        Plotly figure
    """
    errors = y_test.values - y_pred

    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=("Error Distribution", "Actual vs Predicted Scatter"))

    # Histogram
    fig.add_trace(
        go.Histogram(
            x=errors,
            nbinsx=50,
            marker_color="#90BE6D",
            name="Errors",
            opacity=0.8
        ),
        row=1, col=1
    )

    # Scatter
    fig.add_trace(
        go.Scatter(
            x=y_test.values,
            y=y_pred,
            mode="markers",
            marker=dict(color="#F9C74F", size=3, opacity=0.5),
            name="Predictions"
        ),
        row=1, col=2
    )

    # Perfect prediction line
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    fig.add_trace(
        go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode="lines",
            line=dict(color="red", dash="dash"),
            name="Perfect Fit"
        ),
        row=1, col=2
    )

    fig.update_layout(
        title=f"Error Analysis — {model_name.replace('_', ' ').title()}",
        template="plotly_dark",
        height=400
    )
    return fig


def plot_wind_generation_overview(df: pd.DataFrame) -> go.Figure:
    """
    Time series overview of wind generation data.

    Args:
        df: Full dataset DataFrame

    Returns:
        Plotly figure
    """
    daily = df["wind_generation_mw"].resample("D").mean()

    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=("Daily Average Wind Generation (MW)", "Hourly Wind Generation — Last 7 Days"),
        row_heights=[0.6, 0.4]
    )

    fig.add_trace(
        go.Scatter(
            x=daily.index,
            y=daily.values,
            fill="tozeroy",
            line=dict(color="#2E86AB", width=1.5),
            fillcolor="rgba(46,134,171,0.2)",
            name="Daily Avg MW"
        ),
        row=1, col=1
    )

    last_7d = df["wind_generation_mw"].last("7D")
    fig.add_trace(
        go.Scatter(
            x=last_7d.index,
            y=last_7d.values,
            line=dict(color="#90BE6D", width=1.5),
            name="Hourly MW"
        ),
        row=2, col=1
    )

    fig.update_layout(
        template="plotly_dark",
        height=550,
        showlegend=True
    )
    return fig
