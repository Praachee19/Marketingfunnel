import streamlit as st
import pandas as pd
import numpy as np
from io import StringIO

# -------------------------------------------------
# APP CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="Fashion eCommerce Marketing Analytics",
    layout="wide"
)

st.title("Fashion eCommerce. Digital + Performance Marketing Analytics")

# -------------------------------------------------
# TEMPLATE GENERATION
# -------------------------------------------------
def campaign_template():
    return pd.DataFrame({
        "dt": ["2025-12-01", "2025-12-02"],
        "campaign": ["Winter_Sale", "New_Arrivals"],
        "impressions": [100000, 85000],
        "clicks": [3200, 2800],
        "spend": [4200.0, 3900.0],
        "orders": [260, 210],
        "revenue": [780000.0, 640000.0],
        "cost": [410000.0, 350000.0]
    })

# -------------------------------------------------
# VALIDATION
# -------------------------------------------------
REQUIRED_COLUMNS = [
    "dt", "campaign", "impressions", "clicks",
    "spend", "orders", "revenue", "cost"
]

def validate_campaign_file(df):
    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        st.error(f"Missing columns: {', '.join(missing)}")
        st.stop()

    if df.empty:
        st.error("Uploaded file has no rows.")
        st.stop()

    df["dt"] = pd.to_datetime(df["dt"], errors="coerce")
    if df["dt"].isna().any():
        st.error("Invalid dates found in 'dt' column.")
        st.stop()

    numeric_cols = [
        "impressions", "clicks", "spend",
        "orders", "revenue", "cost"
    ]
    for col in numeric_cols:
        if not pd.api.types.is_numeric_dtype(df[col]):
            st.error(f"Column '{col}' must be numeric.")
            st.stop()

    return df

# -------------------------------------------------
# DEMO DATA
# -------------------------------------------------
@st.cache_data
def load_demo_campaigns():
    dates = pd.date_range("2025-11-01", "2025-12-17")

    return pd.DataFrame({
        "dt": np.random.choice(dates, 60),
        "campaign": np.random.choice(["Winter_Sale", "New_Arrivals"], 60),
        "impressions": np.random.randint(50000, 150000, 60),
        "clicks": np.random.randint(2000, 5000, 60),
        "spend": np.random.randint(3000, 8000, 60),
        "orders": np.random.randint(150, 400, 60),
        "revenue": np.random.randint(400000, 1200000, 60),
        "cost": np.random.randint(200000, 700000, 60)
    })

# -------------------------------------------------
# KPI LOGIC
# -------------------------------------------------
def campaign_kpis(df):
    df = df.copy()

    df["CTR"] = df["clicks"] / df["impressions"]
    df["CVR"] = df["orders"] / df["clicks"]
    df["ROAS"] = df["revenue"] / df["spend"]
    df["GM_ROAS"] = (df["revenue"] - df["cost"]) / df["spend"]
    df["AOV"] = df["revenue"] / df["orders"]

    return df

# -------------------------------------------------
# DATA MODE
# -------------------------------------------------
st.sidebar.header("Data Options")

data_mode = st.sidebar.radio(
    "Choose data source",
    ["Demo data", "Upload campaign file"]
)

# -------------------------------------------------
# TEMPLATE DOWNLOAD
# -------------------------------------------------
st.sidebar.subheader("Campaign Template")

template_csv = campaign_template().to_csv(index=False)
st.sidebar.download_button(
    label="Download campaign CSV template",
    data=template_csv,
    file_name="campaign_upload_template.csv",
    mime="text/csv"
)

# -------------------------------------------------
# LOAD DATA
# -------------------------------------------------
if data_mode == "Demo data":
    campaign_df = load_demo_campaigns()
else:
    uploaded_file = st.file_uploader(
        "Upload completed campaign CSV",
        type=["csv"]
    )

    if uploaded_file is None:
        st.info("Upload a campaign CSV to continue.")
        st.stop()

    campaign_df = pd.read_csv(uploaded_file)
    campaign_df = validate_campaign_file(campaign_df)

# -------------------------------------------------
# RUN KPIS
# -------------------------------------------------
campaign_df = campaign_kpis(campaign_df)

# -------------------------------------------------
# DASHBOARD
# -------------------------------------------------
st.subheader("Campaign Performance")

st.dataframe(
    campaign_df.sort_values("ROAS", ascending=False),
    use_container_width=True
)

# -------------------------------------------------
# SUMMARY KPIs
# -------------------------------------------------
st.subheader("Overall Summary")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Spend", f"{campaign_df['spend'].sum():,.0f}")
col2.metric("Total Revenue", f"{campaign_df['revenue'].sum():,.0f}")
col3.metric("Overall ROAS", f"{campaign_df['revenue'].sum() / campaign_df['spend'].sum():.2f}")
col4.metric("Total Orders", f"{campaign_df['orders'].sum():,}")
