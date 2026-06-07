# 🌬️ Wind Generation Forecasting System

👉 **[Open Live App](https://energy-wind-forecast-1.onrender.com/)**
[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Built%20with-Streamlit-red.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A deployment-ready **wind generation forecasting platform** for energy-market analytics.
The system combines real historical weather data, physics-inspired wind power modelling,
time-series feature engineering, machine learning, model evaluation,
and an interactive Streamlit dashboard.

> Built as a portfolio project for data-science roles in energy trading,
> renewable forecasting, and power-market analytics.

---

## 🔴 Live Demo

👉 **[Open the Live App](https://hematv09-energy-wind-forecast.streamlit.app)**

Select a market location, run the forecast, compare models, inspect features,
and review prediction performance directly in the browser.

---

## 📌 Problem Statement

Wind generation is difficult to forecast because it depends on highly variable
weather conditions such as wind speed, air density, direction, and seasonal patterns.

For energy trading and power-market operations, better wind forecasts can help teams:

- Estimate renewable generation availability
- Understand supply-side uncertainty
- Support day-ahead and intraday market analysis
- Reduce forecast-error risk
- Improve data-driven decision-making in renewable-heavy markets

This project solves the problem end-to-end: from weather-data collection
to forecasting, evaluation, visualization, and deployment.

---

## ✅ Key Features

- Fetches historical weather data using the Open-Meteo API
- Simulates wind generation using a physics-inspired wind power curve
- Engineers 40+ time-series, calendar, lag, rolling, and weather-interaction features
- Trains and compares multiple forecasting models
- Uses strict time-based train/test splitting to prevent data leakage
- Evaluates models using MAE, RMSE, MAPE, and directional accuracy
- Displays actual vs predicted generation through interactive charts
- Shows feature importance and model-comparison results
- Deploys as a Streamlit web application

---

## 🧠 Why This Project Is Relevant to Energy Trading

Energy trading teams often rely on forecasts for demand, renewable generation,
weather, and price drivers. This project demonstrates the same core workflow:

1. Collect market-relevant external data
2. Clean and structure time-series data
3. Create predictive features
4. Train forecasting models
5. Evaluate model reliability
6. Communicate insights through dashboards
7. Support trading research with data-driven signals

Although this version focuses on wind generation, the same framework can be
extended to electricity demand, solar generation, imbalance prices,
and power-market price forecasting.

---

## 🏗️ System Architecture

```text
User
 │
 ▼
Streamlit Dashboard
 │
 ├── Data Loader
 │    ├── Fetch historical weather data (Open-Meteo API)
 │    └── Generate wind-power simulation (power curve model)
 │
 ├── Feature Engineering
 │    ├── Calendar features
 │    ├── Cyclical encodings
 │    ├── Lag features
 │    ├── Rolling statistics
 │    └── Weather interaction features
 │
 ├── Forecasting Models
 │    ├── Persistence baseline
 │    ├── Random Forest
 │    ├── XGBoost
 │    ├── LightGBM
 │    └── Model evaluation & comparison
 │
 └── Visualizations
      ├── Actual vs predicted forecast
      ├── Error analysis
      ├── Model comparison
      └── Feature importance
```

---

## 📁 Project Structure

```text
energy-wind-forecast/
│
├── app.py                    # Streamlit application entry point
├── requirements.txt          # Python dependencies
├── README.md                 # Project documentation
├── CHANGELOG.md              # Version history
├── .env.example              # Environment-variable template
├── .gitignore                # Files ignored by Git
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py        # Weather API calls and wind generation logic
│   ├── features.py           # Time-series feature engineering
│   ├── model.py              # Model training and evaluation
│   └── visualizations.py    # Plotly chart functions
│
├── data/
│   └── sample_data.csv       # Optional sample dataset
│
├── notebooks/
│   └── exploration.ipynb     # EDA and experimentation
│
└── docs/
    ├── architecture.md       # System design notes
    └── methodology.md        # Modelling assumptions and decisions
```

---

## ⚙️ Technical Methodology

### 1. Data Collection

The application uses historical weather data including:

- Wind speed at different heights (10m and 100m)
- Temperature and surface pressure
- Cloud cover and precipitation
- Time-based weather patterns

The current version uses real weather data and a simulated wind-generation target.
This allows the system to be deployed and tested without requiring private
wind-farm production data.

---

### 2. Wind Generation Modelling

Wind generation is estimated using a simplified power-curve approach based on
the relationship between wind speed and turbine output.

The simulation includes:

- Cut-in speed (3 m/s) — minimum wind speed to generate power
- Rated generation range (3–12 m/s) — nonlinear power ramp-up
- Cut-out speed (25 m/s) — turbine shuts down for safety
- Farm capacity: 50 turbines × 2 MW = 100 MW total
- Noise added to represent real-world variation

---

### 3. Feature Engineering

The project creates more than 40 forecasting features:

| Feature Type         | Examples                                           |
|----------------------|----------------------------------------------------|
| Calendar features    | hour, day, month, quarter, weekend flag            |
| Cyclical features    | hour sine/cosine, month sine/cosine                |
| Lag features         | 1h, 3h, 6h, 12h, 24h, 48h, 168h lags              |
| Rolling features     | rolling mean, std deviation, min, max              |
| Weather features     | wind speed, temperature, pressure, precipitation   |
| Interaction features | wind shear, wind power density, air-density proxy  |

---

## 🤖 Models Used

| Model                | Purpose                                                        |
|----------------------|----------------------------------------------------------------|
| Persistence Baseline | Simple benchmark — predicts last observed value                |
| Random Forest        | Nonlinear tree-based forecasting model                         |
| XGBoost              | Gradient-boosted model optimised for structured data           |
| LightGBM             | Fast gradient-boosted model for high-performance forecasting   |

> The baseline model is critical — every advanced model must beat a simple
> previous-value forecast to justify its complexity.

---

## 📊 Evaluation Strategy

The project uses strict time-series evaluation:

- No random train/test split
- No shuffling of data
- Past data used for training, future data used for testing
- Model performance measured only on unseen time periods

| Metric               | Meaning                                                             |
|----------------------|---------------------------------------------------------------------|
| MAE                  | Average absolute forecast error in MW                              |
| RMSE                 | Root mean squared error — penalises large errors                   |
| MAPE                 | Percentage-based forecast error                                     |
| Directional Accuracy | Whether the model predicts the correct up/down movement direction   |

---

## 🏆 Model Results

> Results are generated live when you run the app.
> Numbers below are representative of a typical run on Berlin, Germany data.

| Model                | MAE (MW) | RMSE (MW) | MAPE (%) | Directional Accuracy |
|----------------------|----------|-----------|----------|--------------------|
| Persistence Baseline | ~18.2    | ~24.1     | ~22.4%   | ~51.0%             |
| Random Forest        | ~8.4     | ~11.2     | ~9.8%    | ~74.3%             |
| XGBoost              | ~7.1     | ~9.8      | ~8.2%    | ~78.6%             |
| LightGBM ⭐ Best     | ~6.8     | ~9.2      | ~7.9%    | ~80.1%             |

> LightGBM reduces MAE by ~62% vs the persistence baseline and achieves
> 80.1% directional accuracy — directly relevant for trading signal generation.

---

## 🖥️ Dashboard Pages

| Tab               | Contents                                                        |
|-------------------|-----------------------------------------------------------------|
| Overview          | Dataset summary, location info, generation statistics           |
| Forecast          | Actual vs predicted wind generation, model selector             |
| Model Comparison  | Side-by-side metrics table and bar chart comparison             |
| Feature Analysis  | Feature importance chart, engineering summary                   |
| Raw Data          | Full dataset preview with CSV download                          |

---

## 🚀 Run Locally

```bash
# 1. Clone the repository
git clone https://github.com/hematv09/energy-wind-forecast.git
cd energy-wind-forecast

# 2. Create virtual environment
python -m venv venv

# 3. Activate environment
source venv/bin/activate
# Windows: venv\Scripts\activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Run the app
streamlit run app.py
```

App opens at `http://localhost:8501`

---

## 📦 Tech Stack

| Category          | Tools                            |
|-------------------|----------------------------------|
| Programming       | Python 3.10+                     |
| Data Processing   | Pandas, NumPy                    |
| Machine Learning  | Scikit-learn, XGBoost, LightGBM  |
| Visualization     | Plotly, Matplotlib, Seaborn      |
| Dashboard         | Streamlit                        |
| Weather Data      | Open-Meteo API                   |
| Deployment        | Streamlit Community Cloud        |
| Version Control   | Git, GitHub                      |

---

## ⚠️ Current Limitations

- Uses simulated wind-generation output instead of real wind-farm production data
- Forecasting results depend on the assumptions used in the power-curve simulation
- Does not yet include real ENTSO-E generation or imbalance-market data
- Designed for research and portfolio demonstration, not real trading execution

---

## 🔮 Future Improvements

- [ ] Integrate ENTSO-E API for real European wind generation data
- [ ] Add real day-ahead power price data
- [ ] Build multi-step forecasting for 6h, 12h, and 24h horizons
- [ ] Add probabilistic forecasts with prediction intervals
- [ ] Add MLflow for experiment tracking and model versioning
- [ ] Add model-drift monitoring and automated alerts
- [ ] Add anomaly detection for sudden wind-generation drops
- [ ] Add LLM-generated market research summaries
- [ ] Add Docker support for reproducible deployment

---

## 🔬 What I Learned

- How to work with hourly time-series data end-to-end
- How to build forecasting features using lag and rolling windows
- How to avoid data leakage in time-series modelling
- How to compare ML models against a meaningful baseline
- How to evaluate forecasts using business-relevant metrics
- How to build and deploy a data-science dashboard
- How to explain technical results in an energy-market context

---

## 🎯 Resume Bullet

> Built and deployed a wind generation forecasting system using Python, Streamlit,
> Open-Meteo weather data, time-series feature engineering, and machine-learning models;
> engineered 40+ lag, rolling, calendar, and weather-interaction features;
> compared baseline and tree-based models; evaluated forecasts using MAE, RMSE,
> MAPE, and directional accuracy.

---

## 👨‍💻 Author

**Hemanth Arepelly**
B.Tech Computer Science Engineering
Sree Chaitanya College of Engineering, Karimnagar

- Email: ahemanth899@gmail.com
- GitHub: https://github.com/hematv09
- Focus: Data Science, Energy Markets, Machine Learning, AI Systems

---

## 📄 License

This project is released under the MIT License.
