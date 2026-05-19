# Retail-Demand-Forecasting-and-Pricing-Optimization-System

Retail Demand Forecasting & Dynamic Pricing Optimization Project — Built entirely using Python (Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, XGBoost, LightGBM, SHAP)

I developed a complete AI-driven Retail Analytics and Pricing Intelligence System using the UCI Online Retail dataset to forecast product demand, analyze pricing behavior, and optimize revenue through machine learning and elasticity-based pricing strategies.

## Project Breakdown:

### Data Cleaning and Preprocessing

Cleaned and standardized large-scale retail transaction data
Handled missing values, removed invalid transactions, filtered negative quantities/prices, and normalized date formats
Aggregated transaction-level data into daily product-level demand datasets for forecasting and pricing analysis

### Advanced Feature Engineering

Created time-series forecasting features including lag variables, rolling averages, rolling standard deviations, demand momentum, and price momentum
Engineered seasonal and event-based features such as weekends, month-end effects, holidays, and promotional impact periods
Built competitor pricing, price-difference, price-ratio, and demand volatility features for pricing intelligence

### Demand Forecasting System

Developed multiple machine learning forecasting models including Random Forest, Gradient Boosting, XGBoost, and LightGBM
Implemented strict chronological train-validation-test splitting and TimeSeries cross-validation to prevent data leakage
Compared models using RMSE, MAE, R², sMAPE, WAPE, Bias, and forecasting stability metrics
Built ensemble forecasting to improve predictive accuracy and robustness

### Leakage Detection and Model Validation

Designed automatic leakage detection checks for lag features, rolling windows, and target leakage prevention
Performed feature redundancy analysis and correlation-based feature elimination
Conducted residual diagnostics including QQ plots, heteroscedasticity checks, and residual autocorrelation analysis

### Explainable AI and Feature Importance

Implemented SHAP explainability to interpret model predictions and feature contributions
Visualized feature importance stability across multiple time-series validation folds
Analyzed the influence of pricing, seasonality, and demand history on forecast performance

### Price Elasticity and Revenue Optimization

Calculated product-level price elasticity using arc elasticity methods on out-of-sample test data
Built a pricing simulation engine to estimate demand changes under multiple pricing scenarios
Designed a dynamic pricing optimization system to identify revenue-maximizing product prices using machine learning and elasticity blending

## Key Visualizations Created:

Demand distribution and skewness analysis plots
Feature correlation heatmaps and importance comparison charts
Actual vs. predicted demand scatter plots
Residual diagnostics and autocorrelation visualizations
SHAP explainability summary plots
Price elasticity distribution graphs
Demand-vs-price and revenue-vs-price optimization curves
Monthly revenue trend analysis and business insight dashboards
