
import json
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st


MODEL_PATH = Path("models/clv_model.joblib")
METRICS_PATH = Path("reports/model_comparison.csv")
PREDICTIONS_PATH = Path("reports/predictions.csv")
SEGMENTS_PATH = Path("reports/customer_segments.csv")
SUMMARY_PATH = Path("reports/segment_summary.csv")

st.set_page_config(
    page_title="Customer Value Intelligence",
    page_icon="📊",
    layout="wide",
)

st.title("Customer Lifetime Value Prediction")
st.caption(
    "CODETECH Data Science Project | Customer revenue analytics"
)

if not MODEL_PATH.exists():
    st.warning("Model not found. Run `python train.py` first.")
    st.stop()

model = joblib.load(MODEL_PATH)

metadata_path = Path("models/metadata.json")
metadata = (
    json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata_path.exists()
    else {}
)

horizon = metadata.get("prediction_horizon_days", 90)

st.sidebar.header("Customer profile")

recency = st.sidebar.number_input(
    "Days since last purchase",
    min_value=0,
    value=15,
)

tenure = st.sidebar.number_input(
    "Customer tenure (days)",
    min_value=0,
    value=180,
)

orders = st.sidebar.number_input(
    "Historical order count",
    min_value=1,
    value=8,
)

quantity = st.sidebar.number_input(
    "Historical quantity purchased",
    min_value=1,
    value=40,
)

average_order = st.sidebar.number_input(
    "Average order value",
    min_value=0.01,
    value=75.0,
)

historical_revenue = st.sidebar.number_input(
    "Historical total revenue",
    min_value=0.0,
    value=600.0,
)

profile = pd.DataFrame([{
    "RecencyDays": float(recency),
    "CustomerTenureDays": float(tenure),
    "TotalOrders": float(orders),
    "TotalQuantity": float(quantity),
    "AverageOrderValue": float(average_order),
    "PurchaseFrequency": orders / (tenure / 30 + 1),
    "AverageRevenuePerOrder": historical_revenue / orders,
    "HistoricalRevenue": float(historical_revenue),
}])

if st.sidebar.button("Predict customer value", type="primary"):
    estimate = max(float(model.predict(profile)[0]), 0.0)

    st.subheader("Customer revenue prediction")

    st.metric(
        f"Predicted revenue over {horizon} days",
        f"{estimate:,.2f}",
    )

    st.info(
        "This is a revenue forecast, not profit or guaranteed future "
        "spending. The displayed amount uses the source dataset's currency."
    )

st.divider()
st.subheader("Model performance")

if METRICS_PATH.exists():
    metrics = pd.read_csv(METRICS_PATH)

    if not metrics.empty:
        best = metrics.sort_values("MAE").iloc[0]

        col1, col2, col3 = st.columns(3)
        col1.metric("Selected model", str(best["Model"]))
        col2.metric("Validation MAE", f'{best["MAE"]:,.2f}')
        col3.metric("Validation RMSE", f'{best["RMSE"]:,.2f}')

        st.plotly_chart(
            px.bar(
                metrics,
                x="Model",
                y="MAE",
                color="Model",
                title="Model comparison — lower MAE is better",
            ),
            use_container_width=True,
        )

        st.plotly_chart(
            px.bar(
                metrics,
                x="Model",
                y="R2",
                color="Model",
                title="R² comparison",
            ),
            use_container_width=True,
        )
else:
    st.info("Run the training script to generate model metrics.")

if PREDICTIONS_PATH.exists():
    predictions = pd.read_csv(PREDICTIONS_PATH)

    if {
        "ActualRevenue", "PredictedRevenue"
    }.issubset(predictions.columns):
        st.subheader("Actual versus predicted revenue")

        st.plotly_chart(
            px.scatter(
                predictions,
                x="ActualRevenue",
                y="PredictedRevenue",
                opacity=0.65,
                title="Holdout predictions",
            ),
            use_container_width=True,
        )

if SEGMENTS_PATH.exists():
    segments = pd.read_csv(SEGMENTS_PATH)

    st.divider()
    st.subheader("Customer segmentation")

    if "Segment" in segments.columns:
        st.plotly_chart(
            px.histogram(
                segments,
                x="Segment",
                color="Segment",
                title="Number of customers per segment",
            ),
            use_container_width=True,
        )

if SUMMARY_PATH.exists():
    st.subheader("Customer segment summary")
    st.dataframe(
        pd.read_csv(SUMMARY_PATH),
        use_container_width=True,
        hide_index=True,
    )

st.divider()
st.caption(
    "Built with Python, Pandas, Scikit-learn, XGBoost and Streamlit."
)
