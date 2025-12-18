# main.py (complete)
# ------------------
# Adds a Windows-compatible stub for UnixStreamServer before importing pyspark,
# and raises a clear error if PySpark import fails. Rest of the original app unchanged.

import socketserver
if not hasattr(socketserver, "UnixStreamServer"):
    import socket
    class UnixStreamServer(socketserver.TCPServer):
        allow_reuse_address = True
    socketserver.UnixStreamServer = UnixStreamServer

import io
import os
from datetime import datetime, timedelta

import pandas as pd
import streamlit as st

try:
    from pyspark.sql import SparkSession
    from pyspark.sql import functions as F
    from pyspark.sql.window import Window
except Exception as e:
    raise RuntimeError(
        "PySpark import failed. Make sure you're running the Python interpreter that "
        "has PySpark installed and that a Java runtime is available (JAVA_HOME). "
        f"Original error: {e}"
    )

# =========================================
# Streamlit setup
# =========================================
st.set_page_config(page_title="Fashion eCommerce Marketing Analytics", layout="wide")
st.title("Fashion eCommerce. Digital Marketing + Performance Marketing Analytics")

# =========================================
# Spark session
# =========================================
@st.cache_resource
def get_spark():
    return (
        SparkSession.builder
        .appName("marketing-analytics-dashboard")
        .master("local[*]")
        .config("spark.sql.session.timeZone", "UTC")
        .getOrCreate()
    )

spark = get_spark()

# =========================================
# Required schemas
# =========================================
REQ = {
    "ads_daily": {
        "cols": ["dt", "channel", "campaign", "impressions", "clicks", "spend"],
        "types": {"dt": "date", "impressions": "int", "clicks": "int", "spend": "float"}
    },
    "web_events": {
        "cols": ["user_id", "session_id", "event_ts", "event_name", "device", "source", "medium", "campaign"],
        "types": {"event_ts": "datetime"}
    },
    "orders": {
        "cols": ["order_id", "user_id", "order_ts", "revenue", "cost", "channel", "campaign"],
        "types": {"order_ts": "datetime", "revenue": "float", "cost": "float"}
    },
    "customers": {
        "cols": ["user_id", "first_seen_ts"],
        "types": {"first_seen_ts": "datetime"}
    },
    "exposure_holdout": {
        "cols": ["user_id", "campaign", "group", "exposure_ts"],
        "types": {"exposure_ts": "datetime"}
    }
}

ALLOWED_EVENTS = {"page_view", "product_view", "add_to_cart", "begin_checkout"}

# =========================================
# Templates for download
# =========================================
def template_csv_bytes(name: str) -> bytes:
    df = pd.DataFrame(columns=REQ[name]["cols"])
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")

st.sidebar.header("1) Download templates")
for key in ["ads_daily", "web_events", "orders", "customers", "exposure_holdout"]:
    st.sidebar.download_button(
        label=f"Download {key}.csv template",
        data=template_csv_bytes(key),
        file_name=f"{key}.csv",
        mime="text/csv",
        use_container_width=True
    )

# =========================================
# Upload controls
# =========================================
st.sidebar.header("2) Upload your CSV files")
u_ads = st.sidebar.file_uploader("Upload ads_daily.csv", type=["csv"])
u_we  = st.sidebar.file_uploader("Upload web_events.csv", type=["csv"])
u_od  = st.sidebar.file_uploader("Upload orders.csv", type=["csv"])
u_cu  = st.sidebar.file_uploader("Upload customers.csv", type=["csv"])
u_exp = st.sidebar.file_uploader("Upload exposure_holdout.csv (optional)", type=["csv"])

START_DATE = st.sidebar.text_input("Start date (YYYY-MM-DD)", "2025-11-01")
END_DATE   = st.sidebar.text_input("End date (YYYY-MM-DD)", "2025-12-17")
LOOKBACK_LTV_DAYS = st.sidebar.number_input("LTV lookback days", min_value=30, max_value=365, value=180, step=10)

run_btn = st.sidebar.button("Run analytics", type="primary", use_container_width=True)

# =========================================
# Validation helpers
# =========================================
def validate_df(df: pd.DataFrame, name: str):
    errors = []
    req_cols = REQ[name]["cols"]
    missing = [c for c in req_cols if c not in df.columns]
    extra = [c for c in df.columns if c not in req_cols]

    if missing:
        errors.append(f"Missing columns: {missing}")
    if extra:
        errors.append(f"Unexpected columns: {extra}. Remove them or rename to match the template.")

    if errors:
        return False, errors

    tmap = REQ[name].get("types", {})
    for col, t in tmap.items():
        try:
            if t == "date":
                df[col] = pd.to_datetime(df[col], errors="raise").dt.date
            elif t == "datetime":
                df[col] = pd.to_datetime(df[col], errors="raise")
            elif t == "int":
                df[col] = pd.to_numeric(df[col], errors="raise").astype("int64")
            elif t == "float":
                df[col] = pd.to_numeric(df[col], errors="raise").astype("float64")
        except Exception as e:
            errors.append(f"Column '{col}' cannot be parsed as {t}. Error: {str(e)}")

    if name == "web_events":
        bad = df[~df["event_name"].isin(ALLOWED_EVENTS)]
        if len(bad) > 0:
            errors.append(f"web_events.event_name has invalid values. Allowed: {sorted(list(ALLOWED_EVENTS))}. Bad rows: {min(len(bad), 5)} shown in preview.")

    if name == "exposure_holdout":
        badg = df[~df["group"].astype(str).str.lower().isin(["test", "control"])]
        if len(badg) > 0:
            errors.append("exposure_holdout.group must be 'test' or 'control'.")

    ok = len(errors) == 0
    return ok, errors

def read_uploaded_csv(uploaded) -> pd.DataFrame:
    return pd.read_csv(uploaded)

def to_spark(df: pd.DataFrame):
    return spark.createDataFrame(df)

# =========================================
# Synthetic demo data
# =========================================
def build_synthetic(start_date: str, end_date: str):
    start = datetime.strptime(start_date, "%Y-%m-%d").date()
    end = datetime.strptime(end_date, "%Y-%m-%d").date()
    days = (end - start).days + 1
    if days < 1:
        days = 30
        start = datetime.strptime("2025-11-01", "%Y-%m-%d").date()
        end = start + timedelta(days=days - 1)

    # Ads
    ad_rows = []
    channels = ["Meta", "Google"]
    campaigns = ["Winter_Sale", "New_Arrivals"]
    for i in range(days):
        dt = start + timedelta(days=i)
        for ch in channels:
            for ca in campaigns:
                imp = int(20000 + (i * 300) + (5000 if ch == "Meta" else 3000))
                clk = int(max(200, imp * (0.018 if ch == "Meta" else 0.015)))
                spend = float(600 + (clk * (0.65 if ch == "Meta" else 0.75)))
                ad_rows.append([dt, ch, ca, imp, clk, spend])
    ads = pd.DataFrame(ad_rows, columns=REQ["ads_daily"]["cols"])

    # Web events
    import random
    we_rows = []
    users = [f"U{n}" for n in range(1, 4001)]
    for i in range(days * 1200):
        dt = start + timedelta(days=random.randint(0, days - 1))
        user = random.choice(users)
        session = f"S{random.randint(1, 20000)}"
        r = random.random()
        if r < 0.30:
            ev = "page_view"
        elif r < 0.55:
            ev = "product_view"
        elif r < 0.78:
            ev = "add_to_cart"
        else:
            ev = "begin_checkout"
        device = "mobile" if random.random() < 0.6 else "desktop"
        source = "Meta" if random.random() < 0.5 else "Google"
        medium = "paid"
        campaign = "Winter_Sale" if random.random() < 0.5 else "New_Arrivals"
        ts = datetime(dt.year, dt.month, dt.day, 12, 0, 0)
        we_rows.append([user, session, ts, ev, device, source, medium, campaign])
    web_events = pd.DataFrame(we_rows, columns=REQ["web_events"]["cols"])

    # Orders
    od_rows = []
    for oid in range(1, 1501):
        dt = start + timedelta(days=random.randint(0, days - 1))
        user = random.choice(users)
        rev = float(random.randint(1400, 4800))
        cost = float(max(400, rev * random.uniform(0.45, 0.65)))
        channel = "Meta" if random.random() < 0.5 else "Google"
        campaign = "Winter_Sale" if random.random() < 0.5 else "New_Arrivals"
        ts = datetime(dt.year, dt.month, dt.day, 14, 0, 0)
        od_rows.append([f"O{oid}", user, ts, rev, cost, channel, campaign])
    orders = pd.DataFrame(od_rows, columns=REQ["orders"]["cols"])

    # Customers
    cust = orders[["user_id", "order_ts"]].copy()
    cust["first_seen_ts"] = cust.groupby("user_id")["order_ts"].transform("min")
    customers = cust[["user_id", "first_seen_ts"]].drop_duplicates()

    # Optional holdout exposure, makes incrementality demo work
    exp_rows = []
    sample_users = users[:1200]
    for u in sample_users:
        group = "test" if random.random() < 0.7 else "control"
        camp = "Winter_Sale" if random.random() < 0.5 else "New_Arrivals"
        dt = start + timedelta(days=random.randint(0, min(days - 1, 14)))
        ts = datetime(dt.year, dt.month, dt.day, 10, 0, 0)
        exp_rows.append([u, camp, group, ts])
    exposure = pd.DataFrame(exp_rows, columns=REQ["exposure_holdout"]["cols"])

    return ads, web_events, orders, customers, exposure

# =========================================
# Attribution models
# =========================================
def attribution_models(we_s, od_s):
    touches = (we_s
        .select("user_id", "event_ts", "source", "medium", "campaign")
        .withColumnRenamed("source", "touch_channel")
        .withColumnRenamed("medium", "touch_medium")
        .withColumnRenamed("campaign", "touch_campaign")
    )

    base_orders = od_s.select("order_id", "user_id", "order_ts", "revenue")

    joined = (base_orders
        .join(touches, on="user_id", how="left")
        .filter(F.col("event_ts").isNotNull())
        .filter(F.col("event_ts") <= F.col("order_ts"))
        .withColumn("mins_before_order", (F.unix_timestamp("order_ts") - F.unix_timestamp("event_ts")) / 60.0)
    )

    w_last = Window.partitionBy("order_id").orderBy(F.col("event_ts").desc())
    last_click = (joined
        .withColumn("rn", F.row_number().over(w_last))
        .filter(F.col("rn") == 1)
        .groupBy("touch_channel", "touch_campaign")
        .agg(F.countDistinct("order_id").alias("orders"),
             F.sum("revenue").alias("revenue_last_click"))
    )

    w_first = Window.partitionBy("order_id").orderBy(F.col("event_ts").asc())
    first_click = (joined
        .withColumn("rn", F.row_number().over(w_first))
        .filter(F.col("rn") == 1)
        .groupBy("touch_channel", "touch_campaign")
        .agg(F.sum("revenue").alias("revenue_first_click"))
    )

    touch_counts = joined.groupBy("order_id").agg(F.count("*").alias("touches"))
    linear = (joined
        .join(touch_counts, on="order_id", how="inner")
        .withColumn("revenue_share", F.col("revenue") / F.col("touches"))
        .groupBy("touch_channel", "touch_campaign")
        .agg(F.sum("revenue_share").alias("revenue_linear"))
    )

    k = 0.00048
    decay = (joined
        .withColumn("w", F.exp(F.lit(-k) * F.col("mins_before_order")))
    )
    denom = decay.groupBy("order_id").agg(F.sum("w").alias("w_sum"))
    decay = (decay
        .join(denom, on="order_id", how="inner")
        .withColumn("revenue_share", F.col("revenue") * (F.col("w") / F.col("w_sum")))
        .groupBy("touch_channel", "touch_campaign")
        .agg(F.sum("revenue_share").alias("revenue_time_decay"))
    )

    out = (last_click
        .join(first_click, ["touch_channel", "touch_campaign"], "left")
        .join(linear, ["touch_channel", "touch_campaign"], "left")
        .join(decay, ["touch_channel", "touch_campaign"], "left")
        .fillna(0)
        .withColumnRenamed("touch_channel", "channel")
        .withColumnRenamed("touch_campaign", "campaign")
    )
    return out

# =========================================
# Incrementality uplift
# =========================================
def incrementality_uplift(exposure_s, od_s, lookback_days=30):
    exp = (exposure_s
        .withColumn("group", F.lower(F.col("group")))
        .select("user_id", "campaign", "group", "exposure_ts")
    )

    orders = od_s.select("user_id", "campaign", "order_ts", "revenue")

    joined = (exp
        .join(orders, on=["user_id", "campaign"], how="left")
        .withColumn("days_after", F.datediff(F.to_date("order_ts"), F.to_date("exposure_ts")))
        .filter((F.col("order_ts").isNull()) | ((F.col("days_after") >= 0) & (F.col("days_after") <= F.lit(lookback_days))))
        .withColumn("converted", F.when(F.col("order_ts").isNotNull(), F.lit(1)).otherwise(F.lit(0)))
        .withColumn("rev_in_window", F.when(F.col("order_ts").isNotNull(), F.col("revenue")).otherwise(F.lit(0.0)))
    )

    user_level = (joined
        .groupBy("user_id", "campaign", "group")
        .agg(F.max("converted").alias("converted"),
             F.sum("rev_in_window").alias("revenue"))
    )

    cohort = (user_level
        .groupBy("campaign", "group")
        .agg(F.countDistinct("user_id").alias("users"),
             F.sum("converted").alias("converters"),
             F.sum("revenue").alias("revenue"))
        .withColumn("cvr", F.when(F.col("users") > 0, F.col("converters") / F.col("users")))
        .withColumn("rpu", F.when(F.col("users") > 0, F.col("revenue") / F.col("users")))
    )

    test = cohort.filter(F.col("group") == "test").select(
        "campaign",
        F.col("users").alias("users_test"),
        F.col("cvr").alias("cvr_test"),
        F.col("rpu").alias("rpu_test")
    )
    control = cohort.filter(F.col("group") == "control").select(
        "campaign",
        F.col("users").alias("users_control"),
        F.col("cvr").alias("cvr_control"),
        F.col("rpu").alias("rpu_control")
    )

    uplift = (test.join(control, on="campaign", how="inner")
        .withColumn("uplift_cvr_abs", F.col("cvr_test") - F.col("cvr_control"))
        .withColumn("uplift_rpu_abs", F.col("rpu_test") - F.col("rpu_control"))
        .withColumn("incremental_converters", (F.col("uplift_cvr_abs") * F.col("users_test")))
        .withColumn("incremental_revenue", (F.col("uplift_rpu_abs") * F.col("users_test")))
    )
    return uplift

# =========================================
# Main run
# =========================================
if run_btn:
    st.subheader("Run status")

    using_synth = not (u_ads and u_we and u_od and u_cu)

    if using_synth:
        st.info("No complete upload set detected. Using synthetic demo data.")
        ads_pd, we_pd, od_pd, cu_pd, exp_pd = build_synthetic(START_DATE, END_DATE)
    else:
        st.success("Uploads detected. Validating schemas.")
        ads_pd = read_uploaded_csv(u_ads)
        we_pd  = read_uploaded_csv(u_we)
        od_pd  = read_uploaded_csv(u_od)
        cu_pd  = read_uploaded_csv(u_cu)
        exp_pd = read_uploaded_csv(u_exp) if u_exp else None

        ok_ads, err_ads = validate_df(ads_pd, "ads_daily")
        ok_we,  err_we  = validate_df(we_pd, "web_events")
        ok_od,  err_od  = validate_df(od_pd, "orders")
        ok_cu,  err_cu  = validate_df(cu_pd, "customers")

        all_ok = ok_ads and ok_we and ok_od and ok_cu
        if not all_ok:
            st.error("Schema validation failed. Fix these issues and re upload.")
            for e in err_ads: st.error(f"ads_daily. {e}")
            for e in err_we:  st.error(f"web_events. {e}")
            for e in err_od:  st.error(f"orders. {e}")
            for e in err_cu:  st.error(f"customers. {e}")
            st.stop()

        if exp_pd is not None:
            ok_exp, err_exp = validate_df(exp_pd, "exposure_holdout")
            if not ok_exp:
                st.error("exposure_holdout schema invalid. Incrementality will be skipped.")
                for e in err_exp:
                    st.error(f"exposure_holdout. {e}")
                exp_pd = None

        st.success("Validation passed.")

    ads_s = to_spark(ads_pd)
    we_s  = to_spark(we_pd)
    od_s  = to_spark(od_pd)
    cu_s  = to_spark(cu_pd)
    ads_s.createOrReplaceTempView("ads_daily")
    we_s.createOrReplaceTempView("web_events")
    od_s.createOrReplaceTempView("orders")
    cu_s.createOrReplaceTempView("customers")

    if exp_pd is not None:
        exp_s = to_spark(exp_pd)
        exp_s.createOrReplaceTempView("exposure_holdout")
    else:
        exp_s = None

    st.success("Data loaded into Spark.")

    start_dt = F.to_date(F.lit(START_DATE))
    end_dt   = F.to_date(F.lit(END_DATE))

    web_events = spark.table("web_events")
    orders     = spark.table("orders")
    ads_daily  = spark.table("ads_daily")

    we = (web_events
        .withColumn("dt", F.to_date("event_ts"))
        .filter((F.col("dt") >= start_dt) & (F.col("dt") <= end_dt))
    )

    od = (orders
        .withColumn("dt", F.to_date("order_ts"))
        .filter((F.col("dt") >= start_dt) & (F.col("dt") <= end_dt))
        .withColumn("margin", F.col("revenue") - F.col("cost"))
    )

    ad = (ads_daily
        .withColumn("dt", F.to_date("dt"))
        .filter((F.col("dt") >= start_dt) & (F.col("dt") <= end_dt))
    )

    funnel_map = F.create_map(
        [F.lit("page_view"), F.lit("session"),
         F.lit("product_view"), F.lit("view"),
         F.lit("add_to_cart"), F.lit("atc"),
         F.lit("begin_checkout"), F.lit("checkout")]
    )

    we_funnel = (we
        .withColumn("funnel_stage", funnel_map.getItem(F.col("event_name")))
        .filter(F.col("funnel_stage").isNotNull())
    )

    stage_daily = (we_funnel
        .groupBy("dt", "source", "medium", "campaign", "funnel_stage")
        .agg(F.countDistinct("session_id").alias("sessions"))
    )

    purchase_daily = (od
        .groupBy("dt", "channel", "campaign")
        .agg(F.countDistinct("order_id").alias("orders"),
             F.sum("revenue").alias("revenue"),
             F.sum("margin").alias("margin"),
             F.countDistinct("user_id").alias("buyers"))
    )

    paid_kpis = (ad
        .groupBy("dt", "channel", "campaign")
        .agg(F.sum("impressions").alias("impressions"),
             F.sum("clicks").alias("clicks"),
             F.sum("spend").alias("spend"))
        .withColumn("CTR", F.when(F.col("impressions") > 0, F.col("clicks") / F.col("impressions")))
        .withColumn("CPC", F.when(F.col("clicks") > 0, F.col("spend") / F.col("clicks")))
        .withColumn("CPM", F.when(F.col("impressions") > 0, (F.col("spend") * 1000) / F.col("impressions")))
    )

    paid_perf = (paid_kpis
        .join(purchase_daily, on=["dt", "channel", "campaign"], how="left")
        .fillna({"orders": 0, "revenue": 0.0, "margin": 0.0, "buyers": 0})
        .withColumn("CVR_click_to_order", F.when(F.col("clicks") > 0, F.col("orders") / F.col("clicks")))
        .withColumn("ROAS", F.when(F.col("spend") > 0, F.col("revenue") / F.col("spend")))
        .withColumn("GM_ROAS", F.when(F.col("spend") > 0, F.col("margin") / F.col("spend")))
        .withColumn("MER", F.when(F.col("spend") > 0, F.col("revenue") / F.col("spend")))
        .withColumn("AOV", F.when(F.col("orders") > 0, F.col("revenue") / F.col("orders")))
    )

    first_order = (spark.table("orders")
        .select("user_id", "order_ts")
        .withColumn("first_order_ts", F.min("order_ts").over(Window.partitionBy("user_id")))
        .select("user_id", "first_order_ts")
        .distinct()
        .withColumn("first_order_dt", F.to_date("first_order_ts"))
    )

    new_buyers_daily = (first_order
        .filter((F.col("first_order_dt") >= start_dt) & (F.col("first_order_dt") <= end_dt))
        .groupBy(F.col("first_order_dt").alias("dt"))
        .agg(F.countDistinct("user_id").alias("new_buyers"))
    )

    cac_daily = (ad
        .groupBy("dt")
        .agg(F.sum("spend").alias("spend"))
        .join(new_buyers_daily, on="dt", how="left")
        .fillna({"new_buyers": 0})
        .withColumn("CAC", F.when(F.col("new_buyers") > 0, F.col("spend") / F.col("new_buyers")))
    )

    od_all = orders.withColumn("order_dt", F.to_date("order_ts"))

    first_purchase = (od_all
        .groupBy("user_id")
        .agg(F.min("order_dt").alias("first_purchase_dt"))
        .withColumn("cohort_month", F.date_trunc("month", F.col("first_purchase_dt")).cast("date"))
    )

    od_ltv = (od_all
        .join(first_purchase, on="user_id", how="inner")
        .withColumn("days_since_first", F.datediff(F.col("order_dt"), F.col("first_purchase_dt")))
        .filter((F.col("days_since_first") >= 0) & (F.col("days_since_first") <= F.lit(int(LOOKBACK_LTV_DAYS))))
    )

    ltv_by_cohort = (od_ltv
        .groupBy("cohort_month")
        .agg(F.countDistinct("user_id").alias("customers"),
             F.sum("revenue").alias("revenue_lookback"))
        .withColumn("LTV_lookback", F.when(F.col("customers") > 0, F.col("revenue_lookback") / F.col("customers")))
    )

    od_monthly = (od_all
        .withColumn("order_month", F.date_trunc("month", F.col("order_dt")).cast("date"))
        .join(first_purchase.select("user_id", "cohort_month"), on="user_id", how="inner")
    )

    retention = (od_monthly
        .groupBy("cohort_month", "order_month")
        .agg(F.countDistinct("user_id").alias("active_customers"))
    )

    cohort_size = (first_purchase
        .groupBy("cohort_month")
        .agg(F.countDistinct("user_id").alias("cohort_size"))
    )

    retention_rates = (retention
        .join(cohort_size, on="cohort_month", how="inner")
        .withColumn("retention_rate", F.col("active_customers") / F.col("cohort_size"))
        .withColumn("months_since_cohort", F.months_between(F.col("order_month"), F.col("cohort_month")).cast("int"))
        .orderBy("cohort_month", "months_since_cohort")
    )

    attrib = attribution_models(we, od)

    if exp_s is not None:
        incr = incrementality_uplift(exp_s, od, lookback_days=30)
    else:
        incr = None

    paid_perf.write.mode("overwrite").saveAsTable("mart_paid_perf_daily")
    stage_daily.write.mode("overwrite").saveAsTable("mart_funnel_stage_daily")
    cac_daily.write.mode("overwrite").saveAsTable("mart_cac_daily")
    ltv_by_cohort.write.mode("overwrite").saveAsTable("mart_ltv_cohort")
    retention_rates.write.mode("overwrite").saveAsTable("mart_retention_cohort")
    attrib.write.mode("overwrite").saveAsTable("mart_attribution_summary")

    if incr is not None:
        incr.write.mode("overwrite").saveAsTable("mart_incrementality_uplift")

    st.success("Marts created. Showing previews below.")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### Paid performance daily")
        st.dataframe(paid_perf.orderBy(F.col("dt").desc()).limit(50).toPandas(), use_container_width=True)

    with c2:
        st.markdown("### Funnel stage daily")
        st.dataframe(stage_daily.orderBy(F.col("dt").desc()).limit(50).toPandas(), use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        st.markdown("### CAC daily")
        st.dataframe(cac_daily.orderBy(F.col("dt").desc()).limit(50).toPandas(), use_container_width=True)

    with c4:
        st.markdown("### Attribution summary")
        st.dataframe(attrib.orderBy(F.col("revenue_last_click").desc()).limit(50).toPandas(), use_container_width=True)

    st.markdown("### LTV by cohort")
    st.dataframe(ltv_by_cohort.orderBy(F.col("cohort_month").desc()).limit(50).toPandas(), use_container_width=True)

    st.markdown("### Retention rates")
    st.dataframe(retention_rates.limit(100).toPandas(), use_container_width=True)

    st.markdown("### Incrementality uplift")
    if incr is None:
        st.info("Upload exposure_holdout.csv to enable incrementality. Template is available in downloads.")
    else:
        st.dataframe(incr.orderBy(F.col("incremental_revenue").desc()).toPandas(), use_container_width=True)

else:
    st.info("Click Run analytics to load synthetic data, or upload files and run.")

