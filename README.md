
# Customer Lifetime Value Prediction Using Machine Learning

**CODETECH Data Science Internship Project**

## 1. Introduction

Customer Lifetime Value (CLV) is an important business metric used
to understand the revenue a customer may generate over a period of
time. Predicting future customer value can help businesses prioritize
retention campaigns, personalize offers, and allocate marketing
budgets.

This project develops a machine learning pipeline that forecasts
customer revenue over the next 90 days using historical transaction
data.

## 2. Objectives

- Clean and validate retail transaction data.
- Perform customer-level feature engineering.
- Compare multiple regression algorithms.
- Evaluate predictions using MAE, RMSE and R-squared.
- Segment customers using RFM features and K-Means.
- Build an interactive Streamlit dashboard.
- Generate reports that support business analysis.

## 3. Technology Stack

- Python
- Pandas and NumPy
- Scikit-learn
- XGBoost
- Streamlit
- Plotly
- Joblib
- Jupyter Notebook

## 4. Repository Structure

```text
customer-lifetime-value-prediction/
├── app.py
├── train.py
├── requirements.txt
├── README.md
├── .gitignore
├── src/
├── data/
├── models/
├── reports/
├── notebooks/
└── docs/
```

## 5. Dataset

The project is designed for transaction-level retail data containing:

- CustomerID
- InvoiceDate
- InvoiceNo
- Quantity
- UnitPrice

One possible source is the UCI Online Retail dataset:

https://archive.ics.uci.edu/dataset/352/online+retail

Download the dataset and prepare a CSV named
`online_retail.csv` inside the `data/` directory.

Check the dataset's terms of use and preserve its original attribution.

## 6. Installation

Create a virtual environment:

```bash
python -m venv .venv
```

Activate the environment.

Windows:

```bash
.venv\Scripts\activate
```

macOS/Linux:

```bash
source .venv/bin/activate
```

Install the required packages:

```bash
pip install -r requirements.txt
```

## 7. Model Training

Run:

```bash
python train.py
```

The training script creates a future revenue target, compares
regression models, saves the selected model and generates reports.

## 8. Launch the Dashboard

```bash
streamlit run app.py
```

The dashboard provides customer revenue predictions, model
evaluation charts, holdout prediction plots and segmentation summaries.

## 9. Evaluation Metrics

**MAE:** Average absolute prediction error.

**RMSE:** Square root of the average squared prediction error.

**R-squared:** Measures the proportion of target variance explained
relative to a constant-prediction baseline.

Model metrics are generated from the actual dataset and should be
reported only after the training process has completed successfully.

## 10. Business Applications

- Customer retention planning
- Marketing campaign prioritization
- Customer spending analysis
- Customer segmentation
- Revenue forecasting

## 11. Limitations

The model predicts revenue for a 90-day period. This is a
short-horizon CLV proxy, not a complete lifetime profit estimate.

It does not automatically account for profit margins, acquisition
costs, future discounting, returns or customer survival over multiple
years.

The quality of predictions depends on data quality, time coverage,
customer behavior and validation methodology.

## 12. Future Enhancements

- Customer churn prediction
- Survival analysis and probabilistic CLV
- SHAP-based model explanations
- Time-based backtesting
- Hyperparameter optimization
- Cloud deployment and model monitoring

## 13. Author

Name: [Your Name]

Internship: CODETECH

Domain: Data Science

Submission Date: [Your Date]