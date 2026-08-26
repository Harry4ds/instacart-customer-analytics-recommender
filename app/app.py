
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# =========================
# Page config
# =========================

st.set_page_config(
    page_title="Instacart Customer Analytics & Recommender",
    page_icon="🛒",
    layout="wide",
)

# Optional: small CSS touch for nicer feel
st.markdown(
    """
    <style>
    .main {
        background-color: #f8fafc;
    }
    .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================
# Paths & data loading
# =========================

DATA_DIR = Path(__file__).parent

RFM_PATH = DATA_DIR / "rfm_user_details.csv"
CLV_PATH = DATA_DIR / "clv_segment_summary.csv"
ALS_ARTIFACTS_PATH = DATA_DIR / "als_artifacts.pkl"


@st.cache_data
def load_rfm_users():
    df = pd.read_csv(RFM_PATH)
    if df["user_id"].dtype not in ("int32", "int64"):
        df["user_id"] = df["user_id"].astype(int)
    return df


@st.cache_data
def load_clv_summary():
    return pd.read_csv(CLV_PATH)


@st.cache_resource
def load_als_artifacts():
    with open(ALS_ARTIFACTS_PATH, "rb") as f:
        art = pickle.load(f)
    # normalize types a bit
    art["user_to_idx"] = {int(k): int(v) for k, v in art["user_to_idx"].items()}
    art["idx_to_product"] = {int(k): int(v) for k, v in art["idx_to_product"].items()}
    return art


rfm_users = load_rfm_users()
clv_seg = load_clv_summary()
als_artifacts = load_als_artifacts()

als_model = als_artifacts["als_model"]
user_to_idx = als_artifacts["user_to_idx"]
idx_to_product = als_artifacts["idx_to_product"]
user_item_matrix = als_artifacts["user_item_matrix"]
product_lookup = als_artifacts["product_lookup"]

# Detect CLV column if present
CLV_COL = None
for c in ["clv_value", "clv", "clv_hist_items", "avg_clv_value_user"]:
    if c in rfm_users.columns:
        CLV_COL = c
        break


# =========================
# Helper: get ALS recs
# =========================

def get_cf_recommendations(user_id: int, n: int = 10) -> pd.DataFrame:
    """Return ALS top-N recommendations for a single user."""
    u_idx = user_to_idx.get(int(user_id))
    if u_idx is None:
        return pd.DataFrame()

    # implicit expects a CSR row for this user
    user_items_row = user_item_matrix[u_idx]

    recs = als_model.recommend(
        userid=u_idx,
        user_items=user_items_row,
        N=n,
        filter_already_liked_items=True,
    )

    if recs is None or (isinstance(recs, (list, tuple)) and len(recs) == 0):
        return pd.DataFrame()

    if isinstance(recs, tuple) and len(recs) == 2:
        item_indices = list(recs[0])
        scores = list(recs[1])
    else:
        pairs = list(recs)
        if not pairs:
            return pd.DataFrame()
        item_indices = [p[0] for p in pairs]
        scores = [p[1] for p in pairs]

    # Some implicit versions return internal indices, some return actual product_ids.
    # If the key is not in idx_to_product, just treat it as a product_id directly.
    product_ids = [
        idx_to_product.get(int(i), int(i))   # fallback: use i itself as product_id
        for i in item_indices
    ]

    product_names = [
        product_lookup.get(pid, f"Product {pid}")
        for pid in product_ids
    ]

    df = pd.DataFrame(
        {
            "product_id": product_ids,
            "product_name": product_names,
            "cf_score": scores,
        }
    )
    df["cf_score"] = df["cf_score"].astype(float)
    df = df.sort_values("cf_score", ascending=False).reset_index(drop=True)
    return df


# =========================
# Sidebar controls
# =========================

st.sidebar.title("🧭 Instacart Explorer")

segment_col = "segment" if "segment" in rfm_users.columns else None
if segment_col:
    segments = ["All"] + sorted(rfm_users[segment_col].dropna().unique().tolist())
    selected_segment = st.sidebar.selectbox("Filter by segment", segments, index=0)
else:
    selected_segment = "All"

if selected_segment != "All" and segment_col:
    filtered_users = rfm_users[rfm_users[segment_col] == selected_segment]
else:
    filtered_users = rfm_users

user_ids = sorted(filtered_users["user_id"].unique().tolist())
selected_user = st.sidebar.selectbox("Select a customer ID", user_ids)

st.sidebar.markdown("---")
st.sidebar.caption("RFM segmentation + CLV proxy + ALS recommendations")


# =========================
# Main layout
# =========================

st.title("🛒 Instacart Customer Analytics & Recommendation Dashboard")

st.write(
    """
Explore customer segments, estimated customer value,
and personalised product recommendations.
"""
)

st.caption(
    "CLV values are proxy estimates based on historical item counts "
    "and an assumed average item price of $3 because actual product "
    "prices are not available in the dataset."
)

# --------- High-level KPIs row ---------
total_customers = len(rfm_users)
n_segments = rfm_users[segment_col].nunique() if segment_col else 0
avg_clv_overall = float(rfm_users[CLV_COL].mean()) if CLV_COL else None

kpi1, kpi2, kpi3 = st.columns(3)
kpi1.metric("Total customers", f"{total_customers:,}")
kpi2.metric("RFM segments", n_segments if n_segments else "N/A")
if avg_clv_overall is not None:
    kpi3.metric("Avg CLV proxy ($)", f"{avg_clv_overall:,.2f}")
else:
    kpi3.metric("Avg CLV proxy ($)", "N/A")

st.markdown("---")

# --------- Customer profile (always visible) ---------
st.subheader("👤 Customer Profile")

u_row = rfm_users[rfm_users["user_id"] == selected_user].iloc[0]

col1, col2, col3, col4 = st.columns(4)
if segment_col:
    col1.metric("Segment", str(u_row[segment_col]))
else:
    col1.metric("Segment", "N/A")

col2.metric("Recency (days)", int(u_row["recency"]) if "recency" in u_row else "N/A")
col3.metric("Frequency", int(u_row["frequency"]) if "frequency" in u_row else "N/A")
col4.metric("Monetary (items)", int(u_row["monetary"]) if "monetary" in u_row else "N/A")

if CLV_COL:
    st.metric("Estimated CLV proxy ($)", round(float(u_row[CLV_COL]), 2))

st.markdown("---")

# =========================
# Buttons for actions
# =========================

st.subheader("⚙️ Actions")

col_a, col_b = st.columns(2)
with col_a:
    show_recs = st.button("🎯 Get Recommendations")
with col_b:
    show_clv_chart = st.button("📊 Show CLV Proxy by Segment")

# --------- Recommendations (only when button clicked) ---------
st.subheader("🎯 ALS Product Recommendations")

if show_recs:
    recs_df = get_cf_recommendations(int(selected_user), n=10)

    if recs_df.empty:
        st.warning("No recommendations available for this user (possibly cold-start).")
    else:
        # Style by cf_score for a bit of colour
        styled = recs_df.style.background_gradient(subset=["cf_score"], cmap="Blues")
        st.dataframe(styled, use_container_width=True)
        st.caption(
            "Recommendations are generated by the ALS collaborative filtering "
            "model using the user–item interaction matrix."
        )
else:
    st.info("Click **🎯 Get Recommendations** to generate product suggestions for this customer.")

st.markdown("---")

# --------- CLV by segment (only when button clicked) ---------
st.subheader("📊 Average CLV Proxy by Segment")

if show_clv_chart:
    if "segment" in clv_seg.columns and "avg_clv_value" in clv_seg.columns:
        clv_chart_df = clv_seg.set_index("segment")["avg_clv_value"]
        st.bar_chart(clv_chart_df)
    else:
        st.info(
            "Could not find columns 'segment' and 'avg_clv_value' in "
            "clv_segment_summary.csv, so the CLV chart is not displayed."
        )
else:
    st.info("Click **📊 Show CLV by Segment** to display the CLV summary chart.")

with st.expander("📄 Show CLV segment summary table"):
    st.dataframe(clv_seg, use_container_width=True)
