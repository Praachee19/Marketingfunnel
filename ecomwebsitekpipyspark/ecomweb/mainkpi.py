import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import timedelta
import streamlit as st
st.write("APP STARTED")

# --------------------------------------------------
# APP CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="Marketing Analytics Dashboard",
    layout="wide"
)

st.title("Marketing Analytics & Budget Allocation Dashboard")

# --------------------------------------------------
# SYNTHETIC DATA GENERATOR
# --------------------------------------------------
@st.cache_data
def generate_data():
    dates = pd.date_range("2025-01-01", periods=120)

    ads = pd.DataFrame({
        "date": np.random.choice(dates, 400),
        "channel": np.random.choice(["Meta", "Google", "Email"], 400, p=[0.45, 0.4, 0.15]),
        "campaign": np.random.choice(["Sale", "Always_On", "Launch"], 400),
        "impressions": np.random.randint(20000, 200000, 400),
        "clicks": np.random.randint(300, 6000, 400),
        "spend": np.random.randint(1000, 12000, 400)
    })

    orders = pd.DataFrame({
        "order_id": range(1, 900),
        "user_id": np.random.randint(1, 3500, 899),
        "order_date": np.random.choice(dates, 899),
        "revenue": np.random.randint(1500, 6500, 899),
        "channel": np.random.choice(["Meta", "Google", "Email"], 899)
    })

    return ads, orders

ads, orders = generate_data()

# --------------------------------------------------
# DATE FILTER
# --------------------------------------------------
st.sidebar.header("Filters")

start_date, end_date = st.sidebar.date_input(
    "Select date range",
    [ads["date"].min(), ads["date"].max()]
)

ads = ads[(ads["date"] >= pd.to_datetime(start_date)) & (ads["date"] <= pd.to_datetime(end_date))]
orders = orders[(orders["order_date"] >= pd.to_datetime(start_date)) & (orders["order_date"] <= pd.to_datetime(end_date))]

# --------------------------------------------------
# KPI CALCULATIONS
# --------------------------------------------------
ads_agg = ads.groupby("channel").agg(
    impressions=("impressions", "sum"),
    clicks=("clicks", "sum"),
    spend=("spend", "sum")
).reset_index()

sales_agg = orders.groupby("channel").agg(
    orders=("order_id", "count"),
    revenue=("revenue", "sum"),
    customers=("user_id", "nunique")
).reset_index()

kpi = ads_agg.merge(sales_agg, on="channel", how="left").fillna(0)

kpi["CTR"] = kpi["clicks"] / kpi["impressions"]
kpi["CVR"] = kpi["orders"] / kpi["clicks"]
kpi["CAC"] = kpi["spend"] / kpi["customers"]
kpi["ROAS"] = kpi["revenue"] / kpi["spend"]
kpi["AOV"] = kpi["revenue"] / kpi["orders"]
kpi["LTV"] = kpi["AOV"] * 3  # simple 3-purchase assumption
kpi["CLV"] = kpi["LTV"] - kpi["CAC"]

# --------------------------------------------------
# TOP KPI CARDS
# --------------------------------------------------
col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Spend", f"${kpi['spend'].sum():,.0f}")
col2.metric("Total Revenue", f"${kpi['revenue'].sum():,.0f}")
col3.metric("Overall ROAS", f"{kpi['revenue'].sum() / kpi['spend'].sum():.2f}")
col4.metric("Avg CAC", f"${kpi['CAC'].mean():.0f}")

# --------------------------------------------------
# PERFORMANCE MARKETING VIEW
# --------------------------------------------------
st.subheader("Performance Marketing. Channel Efficiency")

fig_roas = px.bar(
    kpi,
    x="channel",
    y="ROAS",
    color="channel",
    title="ROAS by Channel"
)

fig_cac = px.bar(
    kpi,
    x="channel",
    y="CAC",
    color="channel",
    title="CAC by Channel"
)

st.plotly_chart(fig_roas, use_container_width=True)
st.plotly_chart(fig_cac, use_container_width=True)

# --------------------------------------------------
# FUNNEL VIEW
# --------------------------------------------------
st.subheader("Marketing Funnel")

funnel = pd.DataFrame({
    "Stage": ["Impressions", "Clicks", "Orders"],
    "Count": [
        kpi["impressions"].sum(),
        kpi["clicks"].sum(),
        kpi["orders"].sum()
    ]
})

fig_funnel = px.funnel(
    funnel,
    x="Count",
    y="Stage",
    title="Marketing Funnel"
)

st.plotly_chart(fig_funnel, use_container_width=True)

# --------------------------------------------------
# BUDGET ALLOCATION INSIGHT
# --------------------------------------------------
st.subheader("Budget Allocation Recommendation")

kpi["budget_weight"] = kpi["ROAS"] / kpi["ROAS"].sum()

fig_budget = px.pie(
    kpi,
    names="channel",
    values="budget_weight",
    title="Recommended Budget Split Based on ROAS"
)

st.plotly_chart(fig_budget, use_container_width=True)

# --------------------------------------------------
# RETENTION & LTV VIEW
# --------------------------------------------------
st.subheader("Retention & Customer Lifetime Value")

orders_sorted = orders.sort_values("order_date")
first_purchase = orders_sorted.groupby("user_id")["order_date"].min().reset_index()
orders_sorted = orders_sorted.merge(first_purchase, on="user_id", suffixes=("", "_first"))

orders_sorted["days_since_first"] = (
    orders_sorted["order_date"] - orders_sorted["order_date_first"]
).dt.days

retention = orders_sorted[orders_sorted["days_since_first"] <= 90]

retention_rate = retention["user_id"].nunique() / orders["user_id"].nunique()

st.metric("90-Day Retention Rate", f"{retention_rate:.1%}")

# --------------------------------------------------
# DATA TABLE
# --------------------------------------------------
st.subheader("Channel KPI Table")
st.dataframe(kpi.round(2), use_container_width=True)

# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.caption(
    "Pandas execution layer with Spark compatible logic. "
    "Designed for scalable migration to Databricks or PySpark."
)
