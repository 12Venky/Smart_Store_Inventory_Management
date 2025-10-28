import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import os

st.set_page_config(page_title="Smart Inventory Dashboard", layout="wide")

st.markdown("""
<style>
.stApp {
    background-color: #1E1E1E; 
    color: #F0F2F6;
}
.stSidebar { 
    background-color: #2D2D2D;
    color: #F0F2F6;
}
h1, h2, h3, h4, h5, h6 {
    color: #F0F2F6;
}
.stButton>button, .css-1dp5fjs {
    background-color: #6A0DAD;
    color: white;
}
[data-testid="stMetricValue"] {
    color: #BB86FC;
}
.stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {
    border-bottom: 3px solid #BB86FC;
}
/* Custom styling for dataframes */
.stDataFrame {
    color: #F0F2F6;
    background-color: #2D2D2D;
}
/* Ensure charts use dark mode colors */
.stBarChart, .stLineChart {
    color: #BB86FC;
}
</style>
""", unsafe_allow_html=True)

try:
    df = pd.read_csv("cleaned_retail_data.csv")
    df.columns = df.columns.str.strip()
except FileNotFoundError:
    st.error("⚠ Could not find 'cleaned_retail_data.csv'. Please ensure the file is accessible.")
    st.stop()

df = df.rename(columns={
    "Date": "date",
    "Units Sold": "forecast_best"
})

required_cols = ["Product ID", "date", "forecast_best", "Inventory Level"]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    st.error(f"Missing configured columns: {missing}")
    st.stop()

st.title("📦 Milestone 4: Smart Inventory Dashboard & Reporting")

# Sidebar parameters for optimization
lead = st.sidebar.slider("Lead Time", 1, 30, 7)
oc = st.sidebar.slider("Ordering Cost", 10, 200, 50)
hc = st.sidebar.slider("Holding Cost", 1, 20, 2)
z = {"90%":1.28, "95%":1.65, "99%":2.33}[st.sidebar.selectbox("Service Level", ["90%", "95%", "99%"], index=1)]

# --- Inventory Planning Calculation ---
plan = []
for p in df["Product ID"].unique():
    d = df[df["Product ID"] == p]
    dem = d["forecast_best"].sum()
    avg = dem / len(d["date"].unique())
    std = d["forecast_best"].std()
    
    if dem > 0 and hc > 0:
        eoq = np.sqrt((2 * dem * oc) / hc)
    else:
        eoq = 0

    ss = z * std * np.sqrt(lead) if not pd.isna(std) else 0 
    rop = (avg * lead) + ss
    
    plan.append({
        "Product": p,
        "AvgDailySales": avg,
        "TotalDemand": dem,
        "EOQ": eoq,
        "SafetyStock": ss,
        "ReorderPoint": rop
    })
inv = pd.DataFrame(plan)

# ABC Analysis Feature: Classify based on TotalDemand (proxy for value/volume)
inv['TotalDemandValue'] = inv['TotalDemand']
inv = inv.sort_values(by="TotalDemandValue", ascending=False).reset_index(drop=True)
inv["Cumulative%"] = inv["TotalDemandValue"].cumsum() / inv["TotalDemandValue"].sum() * 100
inv["ABC_Category"] = inv["Cumulative%"].apply(
    lambda x: "A (High Value)" if x <= 20 else "B (Medium Value)" if x <= 50 else "C (Low Value)"
)

# --- Merge with Current Stock ---
latest_inv = df.sort_values("date").drop_duplicates(subset=["Product ID"], keep="last")
latest_inv = latest_inv[["Product ID", "Inventory Level"]].rename(columns={"Inventory Level": "CurrentStock"})
alert_df = pd.merge(inv, latest_inv, left_on="Product", right_on="Product ID", how="left").drop(columns=["Product ID"])

alert_df["CurrentStock"] = alert_df["CurrentStock"].fillna(0).astype(float)
alert_df["ReorderPoint"] = alert_df["ReorderPoint"].astype(float)
alert_df["Action"] = np.where(alert_df["CurrentStock"] < alert_df["ReorderPoint"], "Reorder 🚨", "OK ✅")
    
# --- Top Metrics ---
col_m1, col_m2, col_m3 = st.columns(3)
col_m1.metric("Total Products", f"{len(inv):,}")
col_m2.metric("Total Demand", f"{inv['TotalDemand'].sum():,.0f} Units")
col_m3.metric("Urgent Reorders", f"{len(alert_df[alert_df['Action'].str.contains('Reorder')]):,}")
st.markdown("---")


# --- Tabs ---
tab1, tab2, tab3, tab4 = st.tabs(["Historical Demand", "Inventory Plan", "Stock Alerts", "Reports"])

# Tab 1: Historical Demand
with tab1:
    prod = st.selectbox("Select Product for Trend", df["Product ID"].unique())
    sub = df[df["Product ID"] == prod]
    
    sub_agg = sub.groupby("date")["forecast_best"].sum().reset_index()

    plt.style.use('dark_background')
    plt.figure(figsize=(10, 4))
    plt.plot(sub_agg["date"], sub_agg["forecast_best"], label="Units Sold", color="#BB86FC", linewidth=2)
    plt.xticks(sub_agg["date"][::max(1, len(sub_agg) // 10)], rotation=45)
    plt.title(f"Historical Demand for {prod}", color='#F0F2F6')
    plt.xlabel("Date", color='#F0F2F6')
    plt.ylabel("Units Sold", color='#F0F2F6')
    plt.legend()
    plt.tick_params(colors='#F0F2F6')
    plt.tight_layout()
    st.pyplot(plt.gcf())
    plt.close()

# Tab 2: Inventory Plan
with tab2:
    st.subheader("Optimal Inventory Parameters & ABC Classification")
    inv_display = inv.copy()
    
    for col in ["AvgDailySales", "TotalDemand", "EOQ", "SafetyStock", "ReorderPoint"]:
        inv_display[col] = inv_display[col].round(0).astype(int)
        
    display_cols = ["Product", "ABC_Category", "TotalDemand", "EOQ", "ReorderPoint", "SafetyStock"]
    st.dataframe(inv_display[display_cols].set_index("Product"), use_container_width=True)

# Tab 3: Stock Alerts
with tab3:
    st.subheader("Current Stock vs. Reorder Point")
    
    display_alert = alert_df.copy()
    display_alert["ReorderPoint"] = display_alert["ReorderPoint"].round(0).astype(int)
    display_alert["CurrentStock"] = display_alert["CurrentStock"].round(0).astype(int)

    st.dataframe(display_alert[["Product", "ABC_Category", "CurrentStock", "ReorderPoint", "Action"]], use_container_width=True)
    
    # Bar chart focused only on products needing reorder for clarity
    reorder_needed = display_alert[display_alert['Action'].str.contains('Reorder')]
    
    if not reorder_needed.empty:
        st.markdown("#### Products Needing Immediate Reorder")
        st.bar_chart(reorder_needed.set_index("Product")[["CurrentStock", "ReorderPoint"]], color=["#4CAF50", "#FF4500"])
    else:
        st.info("No products currently below the Reorder Point. Inventory is healthy! ✅")

# Tab 4: Reports
with tab4:
    st.subheader("Download Full Inventory Report")
    st.download_button(
        "📥 Download Full Inventory & Alert Report (CSV)", 
        alert_df.to_csv(index=False).encode('utf-8'), 
        "full_inventory_alert_report.csv",
        mime="text/csv"
    )

    st.subheader("ABC Classification Summary")
    abc_summary = inv["ABC_Category"].value_counts().reset_index()
    abc_summary.columns = ["Category", "Product Count"]
    
    abc_colors = {"A (High Value)": "#FF4500", "B (Medium Value)": "#FFB300", "C (Low Value)": "#4CAF50"}
    
    plt.style.use('dark_background')
    plt.figure(figsize=(7, 4))
    plt.bar(
        abc_summary["Category"], 
        abc_summary["Product Count"], 
        color=[abc_colors.get(cat, '#555555') for cat in abc_summary["Category"]]
    )
    plt.title("Product Count by ABC Category", color='#F0F2F6')
    plt.xlabel("Category", color='#F0F2F6')
    plt.ylabel("Number of Products", color='#F0F2F6')
    plt.tick_params(colors='#F0F2F6')
    plt.tight_layout()
    st.pyplot(plt.gcf())
    plt.close()

upl = st.sidebar.file_uploader("Upload New Sales Data", type="csv")
if upl:
    new = pd.read_csv(upl)
    new.columns = new.columns.str.strip()
    st.sidebar.success("File uploaded ✅")
    st.sidebar.info("Re-run forecasting.py manually to refresh predictions.")
