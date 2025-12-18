# main.py
# Streamlit. Fashion eCommerce Marketing Analytics Dashboard (Pandas engine)
# Features: Synthetic demo mode, CSV upload mode, template downloads, schema validation,
# funnel + performance + growth KPIs, budget allocation view, attribution + incrementality (lightweight).

import io
import numpy as np
import pandas as pd
import streamlit as st

# Plotly is used for interactive charts
# Install if missing: pip install plotly
import plotly.express as px
import plotly.graph_objects as go


# -----------------------------
# Page
# -----------------------------
st.set_page_config(page_title="Marketing Analytics Dashboard", layout="wide")
st.title("Marketing Analytics Dashboard. Budget Allocation, Funnel, Performance, Growth")
st.caption("Runs on Pandas for reliability on Windows. Logic is Spark compatible.")

# -----------------------------
# Schemas (templates)
# -----------------------------
ADS_SCHEMA = {
    "dt": "date",
    "channel": "string",
    "campaign": "string",
    "impressions": "int",
    "clicks": "int",
    "spend": "float",
    # Optional columns for experiments / MMM style slicing
    "geo": "string (optional)",
    "holdout_group": "string (optional, Test or Control)"
}

ORDERS_SCHEMA = {
    "order_id": "string",
    "user_id": "string",
    "order_ts": "date",
    "revenue": "float",
    "cost": "float",
    "channel": "string",
    "campaign": "string"
}

WEB_SCHEMA = {
    "user_id": "string",
    "session_id": "string",
    "event_ts": "date",
    "event_name": "string (page_view, product_view, add_to_cart, begin_checkout)",
    "source": "string",
    "medium": "string",
    "campaign": "string"
}

REQUIRED_ADS_COLS = ["dt", "channel", "campaign", "impressions", "clicks", "spend"]
REQUIRED_ORDERS_COLS = ["order_id", "user_id", "order_ts", "revenue", "cost", "channel", "campaign"]
REQUIRED_WEB_COLS = ["user_id", "session_id", "event_ts", "event_name", "source", "medium", "campaign"]

FUNNEL_STAGES = ["page_view", "product_view", "add_to_cart", "begin_checkout"]


def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")


def template_ads() -> pd.DataFrame:
    return pd.DataFrame(
        [
            ["2025-12-01", "Meta", "Winter_Sale", 120000, 3200, 4200.0, "Delhi", "Test"],
            ["2025-12-01", "Google", "Winter_Sale", 90000, 2400, 3800.0, "Mumbai", "Control"],
        ],
        columns=["dt", "channel", "campaign", "impressions", "clicks", "spend", "geo", "holdout_group"],
    )


def template_orders() -> pd.DataFrame:
    return pd.DataFrame(
        [
            ["ORD-1001", "U-501", "2025-12-01", 3499.0, 1600.0, "Meta", "Winter_Sale"],
            ["ORD-1002", "U-777", "2025-12-02", 2799.0, 1300.0, "Google", "Winter_Sale"],
        ],
        columns=["order_id", "user_id", "order_ts", "revenue", "cost", "channel", "campaign"],
    )


def template_web() -> pd.DataFrame:
    return pd.DataFrame(
        [
            ["U-501", "S-9001", "2025-12-01", "page_view", "Meta", "paid", "Winter_Sale"],
            ["U-501", "S-9001", "2025-12-01", "product_view", "Meta", "paid", "Winter_Sale"],
            ["U-501", "S-9001", "2025-12-01", "add_to_cart", "Meta", "paid", "Winter_Sale"],
            ["U-501", "S-9001", "2025-12-01", "begin_checkout", "Meta", "paid", "Winter_Sale"],
        ],
        columns=["user_id", "session_id", "event_ts", "event_name", "source", "medium", "campaign"],
    )


# -----------------------------
# Validation helpers
# -----------------------------
def require_columns(df: pd.DataFrame, required_cols: list, name: str) -> None:
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        st.error(f"{name} is missing required columns: {missing}")
        st.stop()


def coerce_types(ads: pd.DataFrame, orders: pd.DataFrame, web: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ads = ads.copy()
    orders = orders.copy()
    web = web.copy()

    ads["dt"] = pd.to_datetime(ads["dt"], errors="coerce").dt.date
    orders["order_ts"] = pd.to_datetime(orders["order_ts"], errors="coerce").dt.date
    web["event_ts"] = pd.to_datetime(web["event_ts"], errors="coerce").dt.date

    numeric_ads = ["impressions", "clicks", "spend"]
    for c in numeric_ads:
        ads[c] = pd.to_numeric(ads[c], errors="coerce")

    for c in ["revenue", "cost"]:
        orders[c] = pd.to_numeric(orders[c], errors="coerce")

    # Basic null checks
    if ads["dt"].isna().any():
        st.error("ads_daily. Column dt has invalid dates. Fix your CSV.")
        st.stop()
    if orders["order_ts"].isna().any():
        st.error("orders. Column order_ts has invalid dates. Fix your CSV.")
        st.stop()
    if web["event_ts"].isna().any():
        st.error("web_events. Column event_ts has invalid dates. Fix your CSV.")
        st.stop()

    if ads[numeric_ads].isna().any().any():
        st.error("ads_daily. impressions, clicks, spend must be numeric. Fix your CSV.")
        st.stop()
    if orders[["revenue", "cost"]].isna().any().any():
        st.error("orders. revenue and cost must be numeric. Fix your CSV.")
        st.stop()

    # Funnel taxonomy check
    bad_events = sorted(set(web["event_name"].astype(str)) - set(FUNNEL_STAGES))
    if bad_events:
        st.warning(f"web_events. Unknown event_name values found: {bad_events}. Funnel charts will use only {FUNNEL_STAGES}.")

    return ads, orders, web


# -----------------------------
# Synthetic data
# -----------------------------
@st.cache_data
def generate_synthetic(seed: int = 7):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2025-11-01", "2025-12-17", freq="D")

    channels = ["Meta", "Google", "Email", "Organic"]
    campaigns = ["Winter_Sale", "New_Arrivals", "Always_On"]

    # Ads
    n_ads = 900
    ads = pd.DataFrame({
        "dt": rng.choice(dates, n_ads),
        "channel": rng.choice(channels, n_ads, p=[0.38, 0.34, 0.10, 0.18]),
        "campaign": rng.choice(campaigns, n_ads, p=[0.45, 0.25, 0.30]),
        "impressions": rng.integers(20000, 240000, n_ads),
        "clicks": rng.integers(250, 6500, n_ads),
        "spend": rng.uniform(700, 12000, n_ads).round(2),
        "geo": rng.choice(["Delhi", "Mumbai", "Bengaluru", "Pune"], n_ads),
        "holdout_group": rng.choice(["Test", "Control"], n_ads, p=[0.85, 0.15]),
    })
    ads["dt"] = ads["dt"].dt.date

    # Orders
    n_orders = 2600
    user_ids = [f"U-{i}" for i in rng.integers(1, 6500, n_orders)]
    orders = pd.DataFrame({
        "order_id": [f"ORD-{i}" for i in range(1, n_orders + 1)],
        "user_id": user_ids,
        "order_ts": rng.choice(dates, n_orders),
        "revenue": rng.integers(999, 6999, n_orders).astype(float),
        "cost": rng.integers(450, 3600, n_orders).astype(float),
        "channel": rng.choice(channels, n_orders, p=[0.33, 0.32, 0.15, 0.20]),
        "campaign": rng.choice(campaigns, n_orders, p=[0.42, 0.26, 0.32]),
    })
    orders["order_ts"] = orders["order_ts"].dt.date

    # Web events
    n_events = 14000
    web_users = [f"U-{i}" for i in rng.integers(1, 6500, n_events)]
    sessions = [f"S-{i}" for i in rng.integers(1, 9500, n_events)]

    web = pd.DataFrame({
        "user_id": web_users,
        "session_id": sessions,
        "event_ts": rng.choice(dates, n_events),
        "event_name": rng.choice(FUNNEL_STAGES, n_events, p=[0.40, 0.30, 0.20, 0.10]),
        "source": rng.choice(["Meta", "Google", "Email", "Direct"], n_events, p=[0.34, 0.33, 0.10, 0.23]),
        "medium": rng.choice(["paid", "organic", "email", "direct"], n_events, p=[0.55, 0.20, 0.10, 0.15]),
        "campaign": rng.choice(campaigns, n_events, p=[0.45, 0.25, 0.30]),
    })
    web["event_ts"] = web["event_ts"].dt.date

    return ads, orders, web


# -----------------------------
# Metrics
# -----------------------------
def build_daily_paid(ads: pd.DataFrame, orders: pd.DataFrame) -> pd.DataFrame:
    a = ads.groupby(["dt", "channel", "campaign"], as_index=False).agg(
        impressions=("impressions", "sum"),
        clicks=("clicks", "sum"),
        spend=("spend", "sum"),
    )

    o = orders.groupby(["order_ts", "channel", "campaign"], as_index=False).agg(
        orders=("order_id", "nunique"),
        buyers=("user_id", "nunique"),
        revenue=("revenue", "sum"),
        cost=("cost", "sum"),
    ).rename(columns={"order_ts": "dt"})

    df = a.merge(o, on=["dt", "channel", "campaign"], how="left").fillna(0)

    df["CTR"] = np.where(df["impressions"] > 0, df["clicks"] / df["impressions"], 0.0)
    df["CVR"] = np.where(df["clicks"] > 0, df["orders"] / df["clicks"], 0.0)
    df["ROAS"] = np.where(df["spend"] > 0, df["revenue"] / df["spend"], 0.0)
    df["GM_ROAS"] = np.where(df["spend"] > 0, (df["revenue"] - df["cost"]) / df["spend"], 0.0)
    df["AOV"] = np.where(df["orders"] > 0, df["revenue"] / df["orders"], 0.0)

    return df


def build_cac(ads: pd.DataFrame, orders: pd.DataFrame) -> pd.DataFrame:
    # New buyers defined by first purchase date in current filtered range
    od = orders.copy()
    od["order_ts"] = pd.to_datetime(od["order_ts"])
    first = od.groupby("user_id", as_index=False)["order_ts"].min().rename(columns={"order_ts": "first_order_ts"})
    first["dt"] = first["first_order_ts"].dt.date

    new_buyers = first.groupby("dt", as_index=False).agg(new_buyers=("user_id", "nunique"))

    spend = ads.groupby("dt", as_index=False).agg(spend=("spend", "sum"))
    df = spend.merge(new_buyers, on="dt", how="left").fillna(0)
    df["CAC"] = np.where(df["new_buyers"] > 0, df["spend"] / df["new_buyers"], np.nan)
    return df


def build_ltv_retention(orders: pd.DataFrame, horizon_days: int = 180) -> tuple[pd.DataFrame, pd.DataFrame]:
    od = orders.copy()
    od["order_ts"] = pd.to_datetime(od["order_ts"])
    od["order_dt"] = od["order_ts"].dt.date

    first = od.groupby("user_id", as_index=False)["order_ts"].min().rename(columns={"order_ts": "first_purchase_ts"})
    first["first_purchase_dt"] = first["first_purchase_ts"].dt.date
    first["cohort_month"] = first["first_purchase_ts"].dt.to_period("M").dt.to_timestamp().dt.date

    od = od.merge(first[["user_id", "first_purchase_ts", "cohort_month"]], on="user_id", how="inner")
    od["days_since_first"] = (od["order_ts"] - od["first_purchase_ts"]).dt.days

    # LTV within horizon
    ltv = od[(od["days_since_first"] >= 0) & (od["days_since_first"] <= horizon_days)]
    ltv_by_cohort = ltv.groupby("cohort_month", as_index=False).agg(
        customers=("user_id", "nunique"),
        revenue_h=("revenue", "sum")
    )
    ltv_by_cohort["LTV_horizon"] = np.where(ltv_by_cohort["customers"] > 0, ltv_by_cohort["revenue_h"] / ltv_by_cohort["customers"], 0.0)

    # Retention (monthly)
    od["order_month"] = od["order_ts"].dt.to_period("M").dt.to_timestamp().dt.date
    retention = od.groupby(["cohort_month", "order_month"], as_index=False).agg(active=("user_id", "nunique"))
    cohort_size = first.groupby("cohort_month", as_index=False).agg(cohort_size=("user_id", "nunique"))
    retention = retention.merge(cohort_size, on="cohort_month", how="inner")
    retention["retention_rate"] = retention["active"] / retention["cohort_size"]

    # Month index
    cm = pd.to_datetime(retention["cohort_month"])
    om = pd.to_datetime(retention["order_month"])
    retention["months_since_cohort"] = ((om.dt.year - cm.dt.year) * 12 + (om.dt.month - cm.dt.month)).astype(int)

    retention = retention.sort_values(["cohort_month", "months_since_cohort"])
    return ltv_by_cohort, retention


def build_funnel(web: pd.DataFrame) -> pd.DataFrame:
    w = web.copy()
    w = w[w["event_name"].isin(FUNNEL_STAGES)]
    f = (
        w.groupby(["campaign", "event_name"])["session_id"]
        .nunique()
        .reset_index()
        .pivot(index="campaign", columns="event_name", values="session_id")
        .fillna(0)
        .reset_index()
    )
    for c in FUNNEL_STAGES:
        if c not in f.columns:
            f[c] = 0
    f["pdp_rate"] = np.where(f["page_view"] > 0, f["product_view"] / f["page_view"], 0.0)
    f["atc_rate"] = np.where(f["product_view"] > 0, f["add_to_cart"] / f["product_view"], 0.0)
    f["checkout_rate"] = np.where(f["add_to_cart"] > 0, f["begin_checkout"] / f["add_to_cart"], 0.0)
    return f


def last_touch_attribution(web: pd.DataFrame, orders: pd.DataFrame) -> pd.DataFrame:
    # Last web event per user defines campaign credit (simple last-touch)
    w = web.copy()
    w["event_ts"] = pd.to_datetime(w["event_ts"])
    last = w.sort_values("event_ts").groupby("user_id", as_index=False).last()[["user_id", "campaign"]]
    att = orders.merge(last, on="user_id", how="left", suffixes=("", "_lt"))
    att["campaign_final"] = att["campaign_lt"].fillna(att["campaign"])
    out = att.groupby("campaign_final", as_index=False).agg(
        revenue=("revenue", "sum"),
        orders=("order_id", "nunique"),
        buyers=("user_id", "nunique")
    ).rename(columns={"campaign_final": "campaign"})
    return out.sort_values("revenue", ascending=False)


def linear_attribution(orders: pd.DataFrame) -> pd.DataFrame:
    # Lightweight fallback. Uses orders' campaign as credit. Works without event path data.
    out = orders.groupby("campaign", as_index=False).agg(
        revenue=("revenue", "sum"),
        orders=("order_id", "nunique"),
        buyers=("user_id", "nunique")
    ).sort_values("revenue", ascending=False)
    return out


def incrementality_holdout(ads: pd.DataFrame, orders: pd.DataFrame) -> pd.DataFrame:
    # If holdout_group exists in ads, compare ROAS between Test and Control at a high level.
    # If missing, simulate groups for demonstration.
    a = ads.copy()
    if "holdout_group" not in a.columns:
        a["holdout_group"] = np.where(np.random.rand(len(a)) < 0.85, "Test", "Control")

    spend = a.groupby("holdout_group", as_index=False).agg(spend=("spend", "sum"))

    # Assign orders to groups by channel share (proxy). For real use, upload a group label in orders.
    o = orders.copy()
    o["holdout_group"] = np.where(np.random.rand(len(o)) < 0.85, "Test", "Control")
    rev = o.groupby("holdout_group", as_index=False).agg(revenue=("revenue", "sum"))

    df = spend.merge(rev, on="holdout_group", how="outer").fillna(0)
    df["ROAS"] = np.where(df["spend"] > 0, df["revenue"] / df["spend"], 0.0)

    # Incremental lift
    test_rev = float(df.loc[df["holdout_group"] == "Test", "revenue"].sum())
    ctrl_rev = float(df.loc[df["holdout_group"] == "Control", "revenue"].sum())
    lift = (test_rev - ctrl_rev) / ctrl_rev if ctrl_rev > 0 else np.nan

    summary = pd.DataFrame(
        {
            "metric": ["Test revenue", "Control revenue", "Incremental lift"],
            "value": [test_rev, ctrl_rev, lift],
        }
    )
    return df.sort_values("holdout_group"), summary


# -----------------------------
# Sidebar. Mode + Downloads + Uploads
# -----------------------------
st.sidebar.header("Data Mode")

mode = st.sidebar.radio(
    "Choose data source",
    ["Use synthetic demo data", "Upload my CSV files"],
    index=0
)

st.sidebar.divider()
st.sidebar.subheader("Download CSV templates")

st.sidebar.download_button(
    "Download ads_daily template",
    data=df_to_csv_bytes(template_ads()),
    file_name="ads_daily_template.csv",
    mime="text/csv"
)
st.sidebar.download_button(
    "Download orders template",
    data=df_to_csv_bytes(template_orders()),
    file_name="orders_template.csv",
    mime="text/csv"
)
st.sidebar.download_button(
    "Download web_events template",
    data=df_to_csv_bytes(template_web()),
    file_name="web_events_template.csv",
    mime="text/csv"
)

st.sidebar.divider()
st.sidebar.subheader("Filters")

# Load data
if mode == "Use synthetic demo data":
    ads, orders, web = generate_synthetic(seed=11)
else:
    ads_file = st.sidebar.file_uploader("Upload ads_daily.csv", type=["csv"])
    orders_file = st.sidebar.file_uploader("Upload orders.csv", type=["csv"])
    web_file = st.sidebar.file_uploader("Upload web_events.csv", type=["csv"])

    if not (ads_file and orders_file and web_file):
        st.info("Upload all three files using the sidebar. Or switch to synthetic demo data.")
        st.stop()

    ads = pd.read_csv(ads_file)
    orders = pd.read_csv(orders_file)
    web = pd.read_csv(web_file)

    require_columns(ads, REQUIRED_ADS_COLS, "ads_daily.csv")
    require_columns(orders, REQUIRED_ORDERS_COLS, "orders.csv")
    require_columns(web, REQUIRED_WEB_COLS, "web_events.csv")

ads, orders, web = coerce_types(ads, orders, web)

# Date filters
min_dt = min(pd.to_datetime(ads["dt"]).min(), pd.to_datetime(orders["order_ts"]).min(), pd.to_datetime(web["event_ts"]).min()).date()
max_dt = max(pd.to_datetime(ads["dt"]).max(), pd.to_datetime(orders["order_ts"]).max(), pd.to_datetime(web["event_ts"]).max()).date()

date_range = st.sidebar.date_input("Date range", value=(min_dt, max_dt), min_value=min_dt, max_value=max_dt)
if isinstance(date_range, tuple) and len(date_range) == 2:
    start_dt, end_dt = date_range
else:
    start_dt, end_dt = min_dt, max_dt

# Optional channel / campaign filters
all_channels = sorted(set(ads["channel"].astype(str)) | set(orders["channel"].astype(str)))
sel_channels = st.sidebar.multiselect("Channels", options=all_channels, default=all_channels)

all_campaigns = sorted(set(ads["campaign"].astype(str)) | set(orders["campaign"].astype(str)) | set(web["campaign"].astype(str)))
sel_campaigns = st.sidebar.multiselect("Campaigns", options=all_campaigns, default=all_campaigns)

attr_model = st.sidebar.selectbox("Attribution model", ["Last touch (web based)", "Order campaign (fallback)"], index=0)

total_budget = st.sidebar.number_input("Scenario. Total budget to allocate", min_value=0.0, value=200000.0, step=5000.0)

# Apply filters
ads_f = ads[(pd.to_datetime(ads["dt"]).dt.date >= start_dt) & (pd.to_datetime(ads["dt"]).dt.date <= end_dt)]
orders_f = orders[(pd.to_datetime(orders["order_ts"]).dt.date >= start_dt) & (pd.to_datetime(orders["order_ts"]).dt.date <= end_dt)]
web_f = web[(pd.to_datetime(web["event_ts"]).dt.date >= start_dt) & (pd.to_datetime(web["event_ts"]).dt.date <= end_dt)]

ads_f = ads_f[ads_f["channel"].astype(str).isin(sel_channels) & ads_f["campaign"].astype(str).isin(sel_campaigns)]
orders_f = orders_f[orders_f["channel"].astype(str).isin(sel_channels) & orders_f["campaign"].astype(str).isin(sel_campaigns)]
web_f = web_f[web_f["campaign"].astype(str).isin(sel_campaigns)]

if ads_f.empty or orders_f.empty:
    st.warning("Your filters produced no data. Relax the date range, channel, or campaign filters.")
    st.stop()

# -----------------------------
# Build marts
# -----------------------------
paid_daily = build_daily_paid(ads_f, orders_f)
cac_daily = build_cac(ads_f, orders_f)
ltv_by_cohort, retention = build_ltv_retention(orders_f, horizon_days=180)
funnel = build_funnel(web_f)

if attr_model.startswith("Last"):
    attribution = last_touch_attribution(web_f, orders_f)
else:
    attribution = linear_attribution(orders_f)

holdout_detail, holdout_summary = incrementality_holdout(ads_f, orders_f)

# -----------------------------
# Top KPIs
# -----------------------------
total_spend = float(paid_daily["spend"].sum())
total_rev = float(paid_daily["revenue"].sum())
total_orders = int(paid_daily["orders"].sum())
total_buyers = int(paid_daily["buyers"].sum())
overall_roas = (total_rev / total_spend) if total_spend > 0 else 0.0
overall_ctr = (paid_daily["clicks"].sum() / paid_daily["impressions"].sum()) if paid_daily["impressions"].sum() > 0 else 0.0
overall_cvr = (paid_daily["orders"].sum() / paid_daily["clicks"].sum()) if paid_daily["clicks"].sum() > 0 else 0.0
avg_aov = (total_rev / total_orders) if total_orders > 0 else 0.0

k1, k2, k3, k4, k5, k6 = st.columns(6)
k1.metric("Spend", f"{total_spend:,.0f}")
k2.metric("Revenue", f"{total_rev:,.0f}")
k3.metric("ROAS", f"{overall_roas:.2f}")
k4.metric("CTR", f"{overall_ctr:.2%}")
k5.metric("CVR", f"{overall_cvr:.2%}")
k6.metric("AOV", f"{avg_aov:,.0f}")

st.divider()

# -----------------------------
# Tabs
# -----------------------------
tab1, tab2, tab3, tab4 = st.tabs(["Budget allocation", "Funnel", "Performance", "Growth. Retention and LTV"])

# -----------------------------
# Tab 1. Budget allocation
# -----------------------------
with tab1:
    st.subheader("Budget allocation. Decision support")

    # Channel performance summary
    ch = paid_daily.groupby("channel", as_index=False).agg(
        spend=("spend", "sum"),
        revenue=("revenue", "sum"),
        orders=("orders", "sum"),
        buyers=("buyers", "sum"),
        impressions=("impressions", "sum"),
        clicks=("clicks", "sum"),
    )
    ch["ROAS"] = np.where(ch["spend"] > 0, ch["revenue"] / ch["spend"], 0.0)
    ch["CAC"] = np.where(ch["buyers"] > 0, ch["spend"] / ch["buyers"], np.nan)
    ch["CTR"] = np.where(ch["impressions"] > 0, ch["clicks"] / ch["impressions"], 0.0)
    ch["CVR"] = np.where(ch["clicks"] > 0, ch["orders"] / ch["clicks"], 0.0)

    # Recommended split: ROAS-weighted with guardrails
    # Guardrails to prevent extreme allocation. You can tune these.
    min_share = 0.05
    max_share = 0.70

    if ch["ROAS"].sum() == 0:
        ch["rec_share"] = 1.0 / len(ch)
    else:
        ch["rec_share"] = ch["ROAS"] / ch["ROAS"].sum()

    ch["rec_share"] = ch["rec_share"].clip(lower=min_share, upper=max_share)
    ch["rec_share"] = ch["rec_share"] / ch["rec_share"].sum()
    ch["rec_budget"] = ch["rec_share"] * total_budget

    left, right = st.columns([1.1, 0.9])

    with left:
        fig = px.bar(
            ch.sort_values("ROAS", ascending=False),
            x="channel",
            y="ROAS",
            title="ROAS by channel",
            text="ROAS"
        )
        fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")
        fig.update_layout(yaxis_title="ROAS", xaxis_title="")
        st.plotly_chart(fig, use_container_width=True)

        fig2 = px.scatter(
            ch,
            x="CAC",
            y="ROAS",
            size="spend",
            color="channel",
            hover_data=["CTR", "CVR", "buyers", "orders"],
            title="Efficiency map. ROAS vs CAC (bubble size = spend)"
        )
        fig2.update_layout(xaxis_title="CAC", yaxis_title="ROAS")
        st.plotly_chart(fig2, use_container_width=True)

    with right:
        st.markdown("### Recommended budget split")
        fig3 = px.pie(ch, names="channel", values="rec_budget", title="Recommended allocation")
        st.plotly_chart(fig3, use_container_width=True)

        st.markdown("### Allocation table")
        st.dataframe(
            ch[["channel", "spend", "revenue", "ROAS", "CAC", "CTR", "CVR", "rec_share", "rec_budget"]]
            .sort_values("rec_budget", ascending=False)
            .round({"ROAS": 2, "CAC": 0, "CTR": 4, "CVR": 4, "rec_share": 4, "rec_budget": 0}),
            use_container_width=True
        )

# -----------------------------
# Tab 2. Funnel
# -----------------------------
with tab2:
    st.subheader("Marketing funnel. Sessions across stages")

    # Campaign selector for funnel detail
    funnel_campaigns = sorted(funnel["campaign"].astype(str).unique().tolist())
    sel_funnel_campaign = st.selectbox("Select funnel campaign", options=funnel_campaigns, index=0)

    frow = funnel[funnel["campaign"].astype(str) == str(sel_funnel_campaign)].iloc[0]
    funnel_counts = pd.DataFrame({
        "stage": ["Page view", "Product view", "Add to cart", "Begin checkout"],
        "count": [frow["page_view"], frow["product_view"], frow["add_to_cart"], frow["begin_checkout"]],
    })

    figf = px.funnel(funnel_counts, x="count", y="stage", title=f"Funnel. {sel_funnel_campaign}")
    st.plotly_chart(figf, use_container_width=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("PDP rate", f"{float(frow['pdp_rate']):.2%}")
    c2.metric("ATC rate", f"{float(frow['atc_rate']):.2%}")
    c3.metric("Checkout rate", f"{float(frow['checkout_rate']):.2%}")

    st.markdown("### Funnel table by campaign")
    st.dataframe(
        funnel[["campaign", "page_view", "product_view", "add_to_cart", "begin_checkout", "pdp_rate", "atc_rate", "checkout_rate"]]
        .sort_values("page_view", ascending=False)
        .round({"pdp_rate": 4, "atc_rate": 4, "checkout_rate": 4}),
        use_container_width=True
    )

# -----------------------------
# Tab 3. Performance marketing
# -----------------------------
with tab3:
    st.subheader("Performance marketing. Trend and campaign performance")

    # Trend chart
    daily = paid_daily.groupby("dt", as_index=False).agg(
        spend=("spend", "sum"),
        revenue=("revenue", "sum"),
        clicks=("clicks", "sum"),
        impressions=("impressions", "sum"),
        orders=("orders", "sum"),
    )
    daily["ROAS"] = np.where(daily["spend"] > 0, daily["revenue"] / daily["spend"], 0.0)
    daily["CTR"] = np.where(daily["impressions"] > 0, daily["clicks"] / daily["impressions"], 0.0)
    daily["CVR"] = np.where(daily["clicks"] > 0, daily["orders"] / daily["clicks"], 0.0)

    figt = go.Figure()
    figt.add_trace(go.Scatter(x=daily["dt"], y=daily["spend"], name="Spend"))
    figt.add_trace(go.Scatter(x=daily["dt"], y=daily["revenue"], name="Revenue"))
    figt.update_layout(title="Spend and Revenue over time", xaxis_title="", yaxis_title="")
    st.plotly_chart(figt, use_container_width=True)

    figro = px.line(daily, x="dt", y="ROAS", title="ROAS over time")
    st.plotly_chart(figro, use_container_width=True)

    # Campaign table and chart
    camp = paid_daily.groupby(["channel", "campaign"], as_index=False).agg(
        spend=("spend", "sum"),
        revenue=("revenue", "sum"),
        orders=("orders", "sum"),
        buyers=("buyers", "sum"),
        impressions=("impressions", "sum"),
        clicks=("clicks", "sum"),
    )
    camp["ROAS"] = np.where(camp["spend"] > 0, camp["revenue"] / camp["spend"], 0.0)
    camp["CTR"] = np.where(camp["impressions"] > 0, camp["clicks"] / camp["impressions"], 0.0)
    camp["CVR"] = np.where(camp["clicks"] > 0, camp["orders"] / camp["clicks"], 0.0)
    camp["CAC"] = np.where(camp["buyers"] > 0, camp["spend"] / camp["buyers"], np.nan)

    topn = st.slider("Top campaigns to display", min_value=5, max_value=30, value=10, step=1)
    camp_view = camp.sort_values("revenue", ascending=False).head(topn)

    figc = px.bar(camp_view, x="campaign", y="revenue", color="channel", title="Top campaigns by revenue")
    st.plotly_chart(figc, use_container_width=True)

    st.markdown("### Attribution view")
    st.dataframe(attribution.round({"revenue": 0}), use_container_width=True)

    st.markdown("### Incrementality (holdout)")
    l1, l2 = st.columns([0.55, 0.45])
    with l1:
        st.dataframe(holdout_detail.round({"spend": 0, "revenue": 0, "ROAS": 2}), use_container_width=True)
    with l2:
        st.dataframe(holdout_summary, use_container_width=True)

# -----------------------------
# Tab 4. Growth marketing. Retention and LTV
# -----------------------------
with tab4:
    st.subheader("Growth marketing. Retention and LTV / CLV")

    # LTV chart
    figl = px.bar(
        ltv_by_cohort.sort_values("cohort_month"),
        x="cohort_month",
        y="LTV_horizon",
        title="Cohort LTV (180 day revenue per customer)",
    )
    st.plotly_chart(figl, use_container_width=True)

    # Retention heatmap
    heat = retention.pivot(index="cohort_month", columns="months_since_cohort", values="retention_rate").fillna(0)
    heat = heat.sort_index()

    fig_h = px.imshow(
        heat.values,
        x=[str(c) for c in heat.columns],
        y=[str(i) for i in heat.index],
        aspect="auto",
        title="Retention heatmap. Cohort month vs months since cohort",
        labels={"x": "Months since cohort", "y": "Cohort month", "color": "Retention"}
    )
    st.plotly_chart(fig_h, use_container_width=True)

    # CLV / CAC summary (high level)
    avg_ltv = float(ltv_by_cohort["LTV_horizon"].mean()) if not ltv_by_cohort.empty else 0.0
    avg_cac = float(cac_daily["CAC"].dropna().mean()) if not cac_daily.empty and cac_daily["CAC"].notna().any() else np.nan
    clv = avg_ltv - avg_cac if pd.notna(avg_cac) else np.nan

    c1, c2, c3 = st.columns(3)
    c1.metric("Avg LTV (180d)", f"{avg_ltv:,.0f}")
    c2.metric("Avg CAC", f"{avg_cac:,.0f}" if pd.notna(avg_cac) else "NA")
    c3.metric("CLV (LTV - CAC)", f"{clv:,.0f}" if pd.notna(clv) else "NA")

    st.markdown("### Cohort LTV table")
    st.dataframe(ltv_by_cohort.round({"revenue_h": 0, "LTV_horizon": 0}), use_container_width=True)

st.divider()
st.caption("Tip. Use the template downloads to format your CSVs. Upload all three files to switch from synthetic to real data.")
