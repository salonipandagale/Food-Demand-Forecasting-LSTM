import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Food Demand Forecasting",
    page_icon="🍴",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "cleaned_food_delivery.csv"
MODEL_PATH = BASE_DIR / "models" / "final_lstm_v3.keras"


# ============================================================
# ZOMATO-INSPIRED UI
# ============================================================

st.markdown(
    """
    <style>

    /* =========================
       GLOBAL
       ========================= */

    .stApp {
        background: #ffffff;
    }

    [data-testid="stAppViewContainer"] {
        background: #ffffff;
    }

    [data-testid="stHeader"] {
        background: #ffffff;
    }

    .main .block-container {
        max-width: 1180px;
        padding-top: 2.2rem;
        padding-bottom: 3rem;
    }

    h1, h2, h3 {
        color: #1f1f1f !important;
        font-family: Arial, sans-serif !important;
    }

    p, label, span, div {
        font-family: Arial, sans-serif;
    }

    /* =========================
       SIDEBAR
       ========================= */

    section[data-testid="stSidebar"] {
        background: #111111 !important;
    }

    section[data-testid="stSidebar"] > div {
        background: #111111 !important;
    }

    section[data-testid="stSidebar"] * {
        color: #ffffff !important;
    }

    section[data-testid="stSidebar"] label {
        color: #f2f2f2 !important;
        font-weight: 700 !important;
    }

    section[data-testid="stSidebar"] hr {
        border-color: #333333 !important;
    }

    section[data-testid="stSidebar"] .stSelectbox > div > div {
        background: #ffffff !important;
        border-radius: 8px !important;
        border: 1px solid #dddddd !important;
    }

    section[data-testid="stSidebar"] [data-baseweb="select"] {
        background: #ffffff !important;
    }

    section[data-testid="stSidebar"] [data-baseweb="select"] * {
        color: #222222 !important;
    }

    section[data-testid="stSidebar"] [role="listbox"] * {
        color: #222222 !important;
    }

    section[data-testid="stSidebar"] .stMarkdown {
        color: #ffffff !important;
    }

    .sidebar-brand {
        text-align: center;
        padding: 0.8rem 0 1.2rem 0;
    }

    .sidebar-brand-icon {
        font-size: 2.7rem;
        margin-bottom: 0.3rem;
    }

    .sidebar-brand-name {
        color: #ffffff !important;
        font-size: 1.45rem;
        font-weight: 800;
        margin: 0;
    }

    .sidebar-brand-subtitle {
        color: #aaaaaa !important;
        font-size: 0.82rem;
        margin-top: 0.3rem;
    }

    .sidebar-note {
        color: #bdbdbd !important;
        font-size: 0.82rem;
        line-height: 1.55;
        margin-top: 1rem;
    }

    /* =========================
       HERO
       ========================= */

    .hero-box {
        background: linear-gradient(135deg, #e23744 0%, #cb202d 100%);
        border-radius: 18px;
        padding: 2.4rem 2.5rem;
        margin-bottom: 2rem;
        box-shadow: 0 12px 30px rgba(203, 32, 45, 0.18);
    }

    .hero-box h1 {
        color: #ffffff !important;
        font-size: 2.65rem !important;
        margin: 0 0 0.55rem 0 !important;
        font-weight: 800 !important;
    }

    .hero-box p {
        color: #ffffff !important;
        font-size: 1.05rem;
        line-height: 1.55;
        margin: 0;
        max-width: 760px;
    }

    .hero-badge-row {
        margin-top: 1.15rem;
    }

    .hero-badge {
        display: inline-block;
        color: #ffffff !important;
        background: rgba(255,255,255,0.16);
        border: 1px solid rgba(255,255,255,0.30);
        border-radius: 20px;
        padding: 0.35rem 0.75rem;
        margin-right: 0.45rem;
        margin-bottom: 0.35rem;
        font-size: 0.78rem;
        font-weight: 700;
    }

    /* =========================
       SECTION HEADINGS
       ========================= */

    .section-heading {
        font-size: 1.55rem;
        font-weight: 800;
        color: #1f1f1f !important;
        margin-top: 1.6rem;
        margin-bottom: 0.15rem;
    }

    .section-description {
        color: #777777 !important;
        font-size: 0.93rem;
        margin-bottom: 0.9rem;
    }

    /* =========================
       METRICS
       ========================= */

    div[data-testid="stMetric"] {
        background: #ffffff !important;
        border: 1px solid #e8e8e8 !important;
        border-radius: 14px !important;
        padding: 1rem 1.1rem !important;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05) !important;
    }

    div[data-testid="stMetricLabel"] {
        color: #6f6f6f !important;
    }

    div[data-testid="stMetricValue"] {
        color: #222222 !important;
        font-weight: 750 !important;
    }

    div[data-testid="stMetricDelta"] {
        color: #e23744 !important;
    }

    /* =========================
       FORECAST CARD
       ========================= */

    .forecast-box {
        background: #fff5f6;
        border: 2px solid #e23744;
        border-radius: 16px;
        padding: 1.5rem 1.7rem;
        min-height: 155px;
    }

    .forecast-label {
        color: #777777 !important;
        font-size: 0.78rem;
        font-weight: 800;
        letter-spacing: 0.6px;
    }

    .forecast-number {
        color: #e23744 !important;
        font-size: 3rem;
        line-height: 1.1;
        font-weight: 850;
        margin: 0.35rem 0;
    }

    .forecast-small {
        color: #666666 !important;
        font-size: 0.9rem;
    }

    /* =========================
       INFO CARDS
       ========================= */

    .info-box {
        background: #ffffff;
        border: 1px solid #e7e7e7;
        border-radius: 15px;
        padding: 1.25rem 1.35rem;
        min-height: 155px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.045);
    }

    .info-box h4 {
        color: #222222 !important;
        margin: 0 0 0.7rem 0;
        font-size: 1rem;
    }

    .info-box p {
        color: #666666 !important;
        margin: 0;
        font-size: 0.88rem;
        line-height: 1.6;
    }

    .accent {
        color: #e23744 !important;
        font-weight: 800;
    }

    /* =========================
       TABS
       ========================= */

    button[data-baseweb="tab"] {
        color: #555555 !important;
        font-weight: 700 !important;
    }

    button[data-baseweb="tab"][aria-selected="true"] {
        color: #e23744 !important;
    }

    /* =========================
       DATAFRAME
       ========================= */

    div[data-testid="stDataFrame"] {
        border: 1px solid #e8e8e8;
        border-radius: 12px;
        overflow: hidden;
    }

    /* =========================
       BUTTONS
       ========================= */

    .stButton > button {
        border: 1px solid #e23744 !important;
        color: #e23744 !important;
        background: #ffffff !important;
        border-radius: 8px !important;
        font-weight: 700 !important;
    }

    .stButton > button:hover {
        color: #ffffff !important;
        background: #e23744 !important;
        border-color: #e23744 !important;
    }

    /* =========================
       EXPANDER
       ========================= */

    div[data-testid="stExpander"] {
        border: 1px solid #e6e6e6 !important;
        border-radius: 12px !important;
        background: #ffffff !important;
    }

    /* =========================
       FOOTER
       ========================= */

    .footer {
        text-align: center;
        color: #999999 !important;
        font-size: 0.78rem;
        padding: 2.4rem 0 0.4rem 0;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# LOAD DATA AND MODEL
# ============================================================

if not DATA_PATH.exists():
    st.error(
        "Dataset file was not found.\n\n"
        f"Expected: {DATA_PATH}"
    )
    st.stop()

if not MODEL_PATH.exists():
    st.error(
        "Trained model file was not found.\n\n"
        f"Expected: {MODEL_PATH}"
    )
    st.stop()


@st.cache_data
def load_data(path):
    return pd.read_csv(path)


@st.cache_resource
def load_model(path):
    return tf.keras.models.load_model(path)


try:
    df = load_data(DATA_PATH)
except Exception as exc:
    st.error("The dataset could not be loaded.")
    st.code(str(exc))
    st.stop()

try:
    model = load_model(MODEL_PATH)
except Exception as exc:
    st.error("The trained LSTM model could not be loaded.")
    st.code(str(exc))
    st.stop()


# ============================================================
# VALIDATE REQUIRED COLUMNS
# ============================================================

required_columns = {
    "center_id",
    "meal_id",
    "week",
    "num_orders",
    "checkout_price",
    "base_price",
    "emailer_for_promotion",
    "homepage_featured",
}

missing_columns = sorted(required_columns.difference(df.columns))

if missing_columns:
    st.error(
        "The dataset is missing required columns: "
        + ", ".join(missing_columns)
    )
    st.stop()


# ============================================================
# MODEL PREPARATION
# ============================================================

LOOKBACK = 8

FEATURES = [
    "orders_log",
    "checkout_price",
    "base_price",
    "emailer_for_promotion",
    "homepage_featured",
]

df = df.copy()

df["orders_log"] = np.log1p(
    pd.to_numeric(df["num_orders"], errors="coerce")
)

# The scaler is fitted ONLY on the same training period used
# during V3 model development.
training_rows = df.loc[
    df["week"] <= 101,
    FEATURES
].dropna()

if training_rows.empty:
    st.error("No training-period rows were found for scaler preparation.")
    st.stop()

scaler = MinMaxScaler()
scaler.fit(training_rows)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        """
        <div class="sidebar-brand">
            <div class="sidebar-brand-icon">🍴</div>
            <div class="sidebar-brand-name">Demand Predictor</div>
            <div class="sidebar-brand-subtitle">
                Food Delivery Intelligence
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    st.markdown("### Forecast Controls")

    center_options = sorted(
        df["center_id"].dropna().unique().tolist()
    )

    selected_center = st.selectbox(
        "Fulfilment Center",
        center_options,
        format_func=lambda x: str(int(x)),
    )

    meal_options = sorted(
        df.loc[
            df["center_id"] == selected_center,
            "meal_id"
        ].dropna().unique().tolist()
    )

    selected_meal = st.selectbox(
        "Meal",
        meal_options,
        format_func=lambda x: str(int(x)),
    )

    st.divider()

    st.markdown(
        """
        <div class="sidebar-note">
            Select a fulfilment center and meal to explore
            historical demand and generate a next-week forecast
            using the trained multivariate LSTM model.
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# HERO
# ============================================================

st.markdown(
    """
    <div class="hero-box">
        <h1>Food Demand Forecasting</h1>
        <p>
            Predict next-week meal demand using a multivariate
            LSTM time-series forecasting model.
        </p>
        <div class="hero-badge-row">
            <span class="hero-badge">Multivariate LSTM</span>
            <span class="hero-badge">8-Week Lookback</span>
            <span class="hero-badge">5 Input Features</span>
            <span class="hero-badge">MAE 134.13</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SELECTED SERIES
# ============================================================

series = df.loc[
    (df["center_id"] == selected_center)
    & (df["meal_id"] == selected_meal)
].sort_values("week").copy()

if series.empty:
    st.warning("No historical data is available for this selection.")
    st.stop()


# ============================================================
# OVERVIEW
# ============================================================

st.markdown(
    '<div class="section-heading">Overview</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="section-description">'
    'Historical demand for the selected meal and fulfilment center.'
    '</div>',
    unsafe_allow_html=True,
)

avg_orders = float(series["num_orders"].mean())
historical_weeks = int(series["week"].nunique())
max_orders = float(series["num_orders"].max())

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.metric("Fulfilment Center", str(int(selected_center)))

with c2:
    st.metric("Meal", str(int(selected_meal)))

with c3:
    st.metric("Avg Weekly Orders", f"{avg_orders:,.0f}")

with c4:
    st.metric("Historical Weeks", str(historical_weeks))


# ============================================================
# DEMAND TREND
# ============================================================

st.markdown(
    '<div class="section-heading">Demand Trend</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="section-description">'
    'Weekly order history for the selected center-meal combination.'
    '</div>',
    unsafe_allow_html=True,
)

chart_data = (
    series[["week", "num_orders"]]
    .set_index("week")
    .rename(columns={"num_orders": "Orders"})
)

st.line_chart(
    chart_data,
    height=330,
)


# ============================================================
# FORECAST
# ============================================================

st.markdown(
    '<div class="section-heading">Next-Week Forecast</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="section-description">'
    'The model uses the latest 8 observations of demand, pricing '
    'and promotional signals.'
    '</div>',
    unsafe_allow_html=True,
)

if len(series) < LOOKBACK:

    st.warning(
        f"This series contains only {len(series)} observations. "
        f"The LSTM requires at least {LOOKBACK} observations."
    )

else:

    recent = series.tail(LOOKBACK).copy()

    # Keep exactly the five features used during V3 training.
    X = recent[FEATURES].copy()

    # Transform with a DataFrame so sklearn receives the same
    # feature names that were present during scaler.fit().
    X_scaled = scaler.transform(X)

    X_scaled = X_scaled.reshape(
        1,
        LOOKBACK,
        len(FEATURES),
    )

    try:
        prediction_scaled = float(
            model.predict(
                X_scaled,
                verbose=0,
            ).reshape(-1)[0]
        )

        # Inverse-transform the first column (orders_log).
        # Use a DataFrame with matching feature names to avoid
        # sklearn feature-name warnings.
        inverse_frame = pd.DataFrame(
            np.zeros((1, len(FEATURES))),
            columns=FEATURES,
        )

        inverse_frame.loc[0, "orders_log"] = prediction_scaled

        prediction_log = float(
            scaler.inverse_transform(inverse_frame)[0, 0]
        )

        prediction = float(
            np.expm1(prediction_log)
        )

        prediction = max(0.0, prediction)

    except Exception as exc:
        st.error("Forecast generation failed.")
        st.code(str(exc))
        st.stop()

    latest_orders = float(
        series.iloc[-1]["num_orders"]
    )

    if latest_orders > 0:
        demand_change = (
            (prediction - latest_orders)
            / latest_orders
            * 100.0
        )
    else:
        demand_change = 0.0

    direction = (
        "increase"
        if demand_change >= 0
        else "decrease"
    )

    forecast_col, insight_col = st.columns(
        [1.05, 1.0]
    )

    with forecast_col:
        st.markdown(
            f"""
            <div class="forecast-box">
                <div class="forecast-label">
                    PREDICTED NEXT-WEEK ORDERS
                </div>
                <div class="forecast-number">
                    {prediction:,.0f}
                </div>
                <div class="forecast-small">
                    Estimated weekly orders
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with insight_col:
        st.info(
            f"**Forecast insight**\n\n"
            f"The model predicts approximately "
            f"**{prediction:,.0f} orders** for the next week.\n\n"
            f"Latest observed demand: **{latest_orders:,.0f} orders** "
            f"({abs(demand_change):.1f}% {direction})."
        )


# ============================================================
# RECENT INPUT WINDOW
# ============================================================

st.markdown(
    '<div class="section-heading">Recent 8-Week Input</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="section-description">'
    'These are the latest observations supplied to the LSTM.'
    '</div>',
    unsafe_allow_html=True,
)

recent_display = series.tail(LOOKBACK)[
    [
        "week",
        "num_orders",
        "checkout_price",
        "base_price",
        "emailer_for_promotion",
        "homepage_featured",
    ]
].copy()

recent_display.columns = [
    "Week",
    "Orders",
    "Checkout Price",
    "Base Price",
    "Email Promotion",
    "Homepage Featured",
]

recent_display["Orders"] = (
    recent_display["Orders"]
    .round()
    .astype(int)
)

recent_display["Checkout Price"] = (
    recent_display["Checkout Price"]
    .round(2)
)

recent_display["Base Price"] = (
    recent_display["Base Price"]
    .round(2)
)

recent_display["Email Promotion"] = (
    recent_display["Email Promotion"]
    .map({1: "Yes", 0: "No"})
    .fillna("No")
)

recent_display["Homepage Featured"] = (
    recent_display["Homepage Featured"]
    .map({1: "Yes", 0: "No"})
    .fillna("No")
)

st.dataframe(
    recent_display,
    use_container_width=True,
    hide_index=True,
)


# ============================================================
# DEMAND DRIVERS
# ============================================================

st.markdown(
    '<div class="section-heading">Demand Drivers</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="section-description">'
    'Latest pricing and promotional signals used by the model.'
    '</div>',
    unsafe_allow_html=True,
)

latest_row = recent.iloc[-1]

latest_price = float(
    latest_row["checkout_price"]
)

latest_base_price = float(
    latest_row["base_price"]
)

email_status = (
    "Active"
    if int(latest_row["emailer_for_promotion"]) == 1
    else "Not Active"
)

homepage_status = (
    "Featured"
    if int(latest_row["homepage_featured"]) == 1
    else "Not Featured"
)

driver1, driver2, driver3 = st.columns(3)

with driver1:
    st.markdown(
        f"""
        <div class="info-box">
            <h4>Checkout Price</h4>
            <p>
                Latest checkout price:
                <span class="accent">₹{latest_price:,.2f}</span>
                <br><br>
                Base price:
                <span class="accent">₹{latest_base_price:,.2f}</span>
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with driver2:
    st.markdown(
        f"""
        <div class="info-box">
            <h4>Email Promotion</h4>
            <p>
                Current status:
                <span class="accent">{email_status}</span>
                <br><br>
                Email promotion is one of the five
                features supplied to the LSTM.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with driver3:
    st.markdown(
        f"""
        <div class="info-box">
            <h4>Homepage Feature</h4>
            <p>
                Current status:
                <span class="accent">{homepage_status}</span>
                <br><br>
                Homepage visibility is included as
                an input feature.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# MODEL PERFORMANCE
# ============================================================

st.markdown(
    '<div class="section-heading">Model Performance</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="section-description">'
    'Evaluation on the held-out forecasting period.'
    '</div>',
    unsafe_allow_html=True,
)

m1, m2, m3, m4 = st.columns(4)

with m1:
    st.metric("Model", "Multivariate LSTM")

with m2:
    st.metric("Test MAE", "134.13")

with m3:
    st.metric("Test RMSE", "335.70")

with m4:
    st.metric("Lookback", "8 Weeks")


# ============================================================
# MODEL EXPLANATION
# ============================================================

with st.expander("How does the forecasting model work?"):

    st.markdown(
        """
        ### 1. Input data

        The model receives five features from the previous
        **8 weekly observations**:

        - Historical order demand
        - Checkout price
        - Base price
        - Email promotion
        - Homepage featured

        ### 2. Target transformation

        Demand was transformed using `log1p` during training
        to reduce the effect of extreme order spikes.

        ### 3. LSTM architecture

        ```text
        8 Weeks × 5 Features
                 ↓
             LSTM (64)
                 ↓
            Dropout (0.2)
                 ↓
          Dense (32, ReLU)
                 ↓
             Output
                 ↓
        Predicted Weekly Orders
        ```

        ### 4. Model selection

        The multivariate LSTM achieved:

        - **MAE:** 134.13
        - **RMSE:** 335.70

        It was selected because it achieved the lowest MAE
        among the tested forecasting models.
        """
    )


# ============================================================
# PROJECT HIGHLIGHTS
# ============================================================

st.markdown(
    '<div class="section-heading">Project Highlights</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="section-description">'
    'How the forecasting system can support food-delivery operations.'
    '</div>',
    unsafe_allow_html=True,
)

h1, h2, h3 = st.columns(3)

with h1:
    st.markdown(
        """
        <div class="info-box">
            <h4>📈 Time-Series Modeling</h4>
            <p>
                Uses an 8-week historical window to learn
                temporal demand patterns and forecast future orders.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with h2:
    st.markdown(
        """
        <div class="info-box">
            <h4>🧠 Multivariate Forecasting</h4>
            <p>
                Combines demand, pricing and promotional
                signals to improve forecasting decisions.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with h3:
    st.markdown(
        """
        <div class="info-box">
            <h4>🍴 Business Application</h4>
            <p>
                Forecasts can support inventory planning,
                staffing and fulfilment operations.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        Food Delivery Demand Forecasting
        &nbsp; • &nbsp;
        Multivariate LSTM
        &nbsp; • &nbsp;
        Python + TensorFlow + Streamlit
    </div>
    """,
    unsafe_allow_html=True,
)
