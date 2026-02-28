import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from folium.plugins import HeatMap
import streamlit.components.v1 as components
import os

st.set_page_config(
    page_title="Olist Premium Dashboard",
    page_icon="📊",
    layout="wide"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(BASE_DIR, "main_data.csv")

# ==============================
# LOAD DATA
# ==============================
@st.cache_data
def load_data(path):
    df = pd.read_csv(path)
    df["order_purchase_timestamp"] = pd.to_datetime(df["order_purchase_timestamp"])
    return df

df = load_data(file_path)

st.title("analisis bisnis e-commerce")

# ===============================
# SIDEBAR FILTER
# ===============================
with st.sidebar:
    st.header("📌 Filter Panel")

    min_date = df["order_purchase_timestamp"].min()
    max_date = df["order_purchase_timestamp"].max()

    start_date, end_date = st.date_input(
        "Date Range",
        [min_date, max_date]
    )

    selected_category = st.multiselect(
        "Product Category",
        df["product_category_name_english"].unique(),
        default=df["product_category_name_english"].unique()
    )

filtered_df = df[
    (df["order_purchase_timestamp"] >= pd.to_datetime(start_date)) &
    (df["order_purchase_timestamp"] <= pd.to_datetime(end_date)) &
    (df["product_category_name_english"].isin(selected_category))
]

# ===============================
# TABS
# ===============================
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Overview",
    "📦 Product Analysis",
    "👥 Customer & RFM",
    "🗺️ Geospatial"
])

# ===============================
# TAB 1 — OVERVIEW
# ===============================
with tab1:
    st.subheader("rangkuman dagangan gw")

    col1, col2, col3, col4 = st.columns(4)

    total_revenue = filtered_df["untung"].sum()
    total_orders = filtered_df["order_id"].nunique()
    total_customers = filtered_df["customer_id"].nunique()
    avg_order_value = total_revenue / total_orders if total_orders > 0 else 0

    col1.metric("Total Revenue", f"R$ {total_revenue:,.0f}")
    col2.metric("Total Orders", total_orders)
    col3.metric("Total Customers", total_customers)
    col4.metric("Avg Order Value", f"R$ {avg_order_value:,.0f}")

    st.divider()

    st.subheader("untung harian")

    daily_revenue = (
        filtered_df.groupby(filtered_df["order_purchase_timestamp"].dt.date)["untung"]
        .sum()
    )

    fig, ax = plt.subplots(figsize=(10,4))
    ax.plot(daily_revenue.index, daily_revenue.values)
    ax.set_ylabel("untung")
    ax.set_xlabel("")
    st.pyplot(fig)

    st.info("tren untung per periode")

# ===============================
# TAB 2 — PRODUCT
# ===============================
with tab2:
    st.subheader("kategori dari untungnye")

    top_category = (
        filtered_df.groupby("product_category_name_english")["untung"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
    )

    fig, ax = plt.subplots(figsize=(10,5))
    top_category.plot(kind="bar", ax=ax)
    ax.set_ylabel("untung")
    ax.set_xlabel("")
    st.pyplot(fig)

    st.success("kategori yang nguntungin")

# ===============================
# TAB 3 — CUSTOMER & RFM
# ===============================
with tab3:
    st.subheader("RFM")

    rfm = filtered_df.groupby("customer_id").agg({
        "order_purchase_timestamp": "max",
        "order_id": "nunique",
        "untung": "sum"
    }).reset_index()

    rfm.columns = ["customer_id", "last_purchase", "frequency", "monetary"]

    recent_date = filtered_df["order_purchase_timestamp"].max()
    rfm["recency"] = (recent_date - rfm["last_purchase"]).dt.days

    # Manual Binning (Clustering without ML)
    rfm["R_score"] = pd.qcut(rfm["recency"], 3, labels=[3,2,1])
    rfm["F_score"] = pd.qcut(rfm["frequency"].rank(method="first"), 3, labels=[1,2,3])
    rfm["M_score"] = pd.qcut(rfm["monetary"], 3, labels=[1,2,3])

    col1, col2, col3 = st.columns(3)
    col1.metric("Avg Recency", round(rfm["recency"].mean(),1))
    col2.metric("Avg Frequency", round(rfm["frequency"].mean(),2))
    col3.metric("Avg Monetary", f"R$ {rfm['monetary'].mean():,.0f}")

    st.divider()

    st.subheader("pelanggan borong")

    top_customers = rfm.sort_values(by="monetary", ascending=False).head(5)

    fig, ax = plt.subplots(figsize=(8,4))
    sns.barplot(data=top_customers, x="customer_id", y="monetary", ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    st.pyplot(fig)

    st.warning("pelanggan yang kaga pelit")

# ===============================
# TAB 4 — GEOSPATIAL
# ===============================
from streamlit_folium import st_folium
from folium.plugins import HeatMap

with tab4:
    st.subheader("🗺️ distribusi pembeli Heatmap")

    required_cols = ["geolocation_lat", "geolocation_lng"]

    if not all(col in filtered_df.columns for col in required_cols):
        st.error("Kolom geolocation_lat & geolocation_lng tidak ditemukan di dataset.")
    else:
        data_serlok = filtered_df[required_cols].dropna()

        if data_serlok.empty:
            st.warning("Tidak ada data lokasi setelah filter diterapkan.")
        else:
            m = folium.Map(
                location=[-14.2350, -51.9253],
                zoom_start=4,
                tiles="CartoDB positron"
            )

            HeatMap(
                data=data_serlok.values.tolist(),  # jangan geo_data
                radius=8,
                blur=15,
                min_opacity=0.4
            ).add_to(m)

            st_folium(m, width=900, height=600)
            st.caption("serlokan pelanggan")
            
st.caption("©Dashboard gw nih bjir")
