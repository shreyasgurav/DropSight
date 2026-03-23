"""
app.py
------
Streamlit dashboard for the AI-Based User Journey & Drop-Off
Prediction System. Run with:  streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
from sklearn.preprocessing import LabelEncoder

# --------------- paths ---------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

import sys
sys.path.insert(0, os.path.join(BASE_DIR, "src"))
from recommendations import get_recommendation
CLEAN_CSV = os.path.join(BASE_DIR, "data", "cleaned_data.csv")
MODEL_PATH = os.path.join(BASE_DIR, "output", "model.pkl")
METRICS_CSV = os.path.join(BASE_DIR, "output", "model_metrics.csv")
CHARTS_DIR = os.path.join(BASE_DIR, "output", "charts")

# --------------- page config ---------------
st.set_page_config(
    page_title="E-Commerce Drop-Off Predictor",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------- custom CSS ---------------
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px; border-radius: 12px; text-align: center;
        color: white; box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    .metric-card h2 { margin: 0; font-size: 2rem; }
    .metric-card p  { margin: 5px 0 0 0; font-size: 0.95rem; opacity: 0.9; }
    .funnel-step {
        text-align: center; padding: 12px; border-radius: 8px;
        color: white; font-weight: bold; margin: 4px 0;
    }
    .funnel-active  { background-color: #28a745; }
    .funnel-dropped { background-color: #dc3545; }
    .funnel-future  { background-color: #6c757d; opacity: 0.4; }
    .insight-box {
        background-color: #f0f2f6; padding: 12px 16px;
        border-left: 4px solid #667eea; border-radius: 4px;
        margin-bottom: 18px; font-size: 0.92rem;
    }
</style>
""", unsafe_allow_html=True)


# --------------- data loading (cached) ---------------
@st.cache_data
def load_cleaned_data():
    return pd.read_csv(CLEAN_CSV)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_metrics():
    return pd.read_csv(METRICS_CSV)


# --------------- sidebar navigation ---------------
st.sidebar.title("🛒 Navigation")
page = st.sidebar.radio(
    "Go to",
    ["📊 Overview", "📈 EDA & Insights", "🔮 Drop-Off Predictor", "🏆 Model Performance"],
)

df = load_cleaned_data()


# =====================================================================
# PAGE 1 — Overview
# =====================================================================
if page == "📊 Overview":
    st.title("🛒 AI-Based User Journey & Drop-Off Prediction")
    st.markdown(
        "Analyze where users drop off while browsing a clothing e-commerce "
        "website and predict the drop-off page using Machine Learning."
    )

    # --- Metric cards ---
    total_sessions = df["session_id"].nunique()
    total_records = len(df)
    total_countries = df["country"].nunique()
    pages_tracked = df["page"].nunique()

    c1, c2, c3, c4 = st.columns(4)
    for col, label, value in [
        (c1, "Total Sessions", f"{total_sessions:,}"),
        (c2, "Total Records", f"{total_records:,}"),
        (c3, "Countries", str(total_countries)),
        (c4, "Pages Tracked", str(pages_tracked)),
    ]:
        col.markdown(
            f'<div class="metric-card"><h2>{value}</h2><p>{label}</p></div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.subheader("📋 Cleaned Dataset (first 100 rows)")
    st.dataframe(df.head(100), use_container_width=True, height=420)


# =====================================================================
# PAGE 2 — EDA & Insights
# =====================================================================
elif page == "📈 EDA & Insights":
    st.title("📈 Exploratory Data Analysis")

    charts_info = [
        ("dropoff_per_page.png",
         "Drop-Off Count per Page",
         "Most users drop off on the first page — homepage improvements "
         "and engaging hero content can reduce early exits."),
        ("category_distribution.png",
         "Category Distribution",
         "Shows which product categories attract the most browsing. "
         "Under-represented categories may need better visibility."),
        ("sessions_by_country.png",
         "Top 10 Countries by Sessions",
         "Identifies the primary geographic markets. Marketing budget "
         "and localisation efforts should prioritise these countries."),
        ("sessions_per_month.png",
         "Sessions per Month",
         "Reveals seasonal traffic patterns. Spike months are ideal "
         "for promotions; low months need re-engagement campaigns."),
        ("correlation_heatmap.png",
         "Correlation Matrix",
         "Highlights linear relationships between numeric features. "
         "Strong correlations guide feature selection for the ML model."),
        ("dropoff_by_category.png",
         "Drop-Off Rate by Category",
         "Categories with higher drop-off rates may suffer from poor "
         "product imagery, pricing, or navigation UX."),
        ("avg_order_steps_by_country.png",
         "Avg Order Steps Before Drop-Off (Top 10 Countries)",
         "Countries where users take more steps before dropping off "
         "indicate higher intent — retargeting can be very effective."),
        ("dropoff_by_price.png",
         "Drop-Off Rate by Price Range",
         "Compares Low vs High price ranges. A higher drop-off for "
         "expensive items suggests adding trust signals or alternatives."),
    ]

    for fname, title, insight in charts_info:
        path = os.path.join(CHARTS_DIR, fname)
        if os.path.exists(path):
            st.subheader(title)
            st.image(path, use_container_width=True)
            st.markdown(f'<div class="insight-box">💡 <b>Insight:</b> {insight}</div>',
                        unsafe_allow_html=True)
        else:
            st.warning(f"Chart not found: {fname}. Run eda.py first.")


# =====================================================================
# PAGE 3 — Drop-Off Predictor
# =====================================================================
elif page == "🔮 Drop-Off Predictor":
    st.title("🔮 Predict User Drop-Off Page")
    st.markdown("Fill in the user's browsing context and click **Predict**.")

    model = load_model()

    # --- Encode helpers (must match model.py encoding) ---
    cat_map = {"Blouses": 0, "Sale": 1, "Skirts": 2, "Trousers": 3}
    photo_map = {"En-face": 0, "Profile": 1}
    price_map = {"High": 0, "Low": 1}

    countries = sorted(df["country"].unique().tolist())

    # --- Input form ---
    with st.form("predict_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            main_cat = st.selectbox("Main Category", ["Trousers", "Skirts", "Blouses", "Sale"])
            country = st.selectbox("Country", countries)
            price_range = st.selectbox("Price Range", ["Low", "High"])
        with col2:
            colour = st.slider("Colour Code", 1, 14, 1)
            location = st.slider("Location Code", 1, 6, 1)
            order_step = st.slider("Order Step", 1, 20, 1)
        with col3:
            month = st.slider("Month", 4, 8, 4)
            photo_type = st.selectbox("Photo Type", ["En-face", "Profile"])

        submitted = st.form_submit_button("🔮 Predict Drop-Off Page")

    if submitted:
        # Build feature vector (same order as FEATURE_COLS in model.py)
        features = np.array([[
            cat_map[main_cat],    # main_category
            colour,               # colour
            location,             # location
            photo_map[photo_type],# photo_type
            df[df["main_category"] == main_cat]["price"].median(),  # price proxy
            price_map[price_range],  # price_range
            country,              # country
            order_step,           # order
            month,                # month
        ]])

        predicted_page = model.predict(features)[0]
        probas = model.predict_proba(features)[0]
        confidence = probas.max() * 100

        # --- Results ---
        st.markdown("---")
        r1, r2 = st.columns(2)
        with r1:
            st.metric("Predicted Drop-Off Page", f"Page {predicted_page}")
            st.metric("Confidence", f"{confidence:.1f}%")
        with r2:
            rec = get_recommendation(predicted_page, price_range)
            st.info(f"**📌 Recommendation:** {rec['full_recommendation']}")

        # --- Probability distribution ---
        st.subheader("Prediction Probability Distribution")
        prob_df = pd.DataFrame({
            "Page": [f"Page {c}" for c in model.classes_],
            "Probability": probas
        })
        st.bar_chart(prob_df.set_index("Page"))

        # --- Funnel visualisation ---
        st.subheader("User Journey Funnel")
        funnel_labels = {
            1: "Homepage",
            2: "Category Page",
            3: "Product Page",
            4: "Cart / Details",
            5: "Checkout",
        }
        cols = st.columns(5)
        for i, c in enumerate(cols, start=1):
            label = funnel_labels[i]
            if i < predicted_page:
                css = "funnel-active"
                icon = "✅"
            elif i == predicted_page:
                css = "funnel-dropped"
                icon = "🚫"
            else:
                css = "funnel-future"
                icon = "⬜"
            c.markdown(
                f'<div class="funnel-step {css}">{icon} Page {i}<br>{label}</div>',
                unsafe_allow_html=True,
            )


# =====================================================================
# PAGE 4 — Model Performance
# =====================================================================
elif page == "🏆 Model Performance":
    st.title("🏆 Model Performance Comparison")

    # --- Accuracy cards ---
    metrics_df = load_metrics()
    cols = st.columns(len(metrics_df))
    for idx, row in metrics_df.iterrows():
        with cols[idx]:
            st.markdown(
                f'<div class="metric-card"><h2>{row["accuracy"]:.2%}</h2>'
                f'<p>{row["model"]}</p></div>',
                unsafe_allow_html=True,
            )

    st.markdown("---")

    # --- Charts ---
    c1, c2 = st.columns(2)
    cm_path = os.path.join(CHARTS_DIR, "confusion_matrix.png")
    fi_path = os.path.join(CHARTS_DIR, "feature_importance.png")

    with c1:
        st.subheader("Confusion Matrix")
        if os.path.exists(cm_path):
            st.image(cm_path, use_container_width=True)
        else:
            st.warning("Run model.py first to generate this chart.")

    with c2:
        st.subheader("Feature Importance")
        if os.path.exists(fi_path):
            st.image(fi_path, use_container_width=True)
        else:
            st.warning("Run model.py first to generate this chart.")

    # --- Classification report table ---
    st.markdown("---")
    st.subheader("Classification Report (Best Model)")
    model = load_model()
    # Regenerate report from cleaned data
    from sklearn.metrics import classification_report
    from sklearn.model_selection import train_test_split

    @st.cache_data
    def get_report_df():
        data = pd.read_csv(CLEAN_CSV)
        for col in ["main_category", "photo_type", "price_range"]:
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col].astype(str))
        feature_cols = ["main_category", "colour", "location", "photo_type",
                        "price", "price_range", "country", "order", "month"]
        X = data[feature_cols]
        y = data["page"]
        _, X_test, _, y_test = train_test_split(X, y, test_size=0.2,
                                                 random_state=42, stratify=y)
        preds = model.predict(X_test)
        report = classification_report(y_test, preds, output_dict=True)
        return pd.DataFrame(report).transpose()

    report_df = get_report_df()
    st.dataframe(report_df.style.format("{:.2f}"), use_container_width=True)
