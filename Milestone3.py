import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Inventory Optimization Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
.stApp {
    background-color: #1E1E1E; 
    color: #F0F2F6;
}
.css-1d391kg { 
    background-color: #2D2D2D;
}
h1, h2, h3 {
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
</style>
""", unsafe_allow_html=True)

try:
    df = pd.read_csv("cleaned_retail_data.csv")
except FileNotFoundError:
    st.error("Error: 'cleaned_retail_data.csv' not found. Please ensure the file is in the same directory.")
    st.stop()
    
required_cols = ['Product ID', 'Demand Forecast']
if not all(col in df.columns for col in required_cols):
    st.error(f"Error: Data must contain columns: {', '.join(required_cols)}")
    st.stop()

st.title("💜 Supply Chain Sentinel: Inventory Optimization")

with st.sidebar:
    st.header("⚙️ Optimization Parameters")
    
    products = df['Product ID'].unique()
    selected_product = st.selectbox(
        "Focus Product", 
        products
    )
    
    st.markdown("---")
    
    lead_time = st.slider("Lead Time (days)", 1, 30, 7)
    ordering_cost = st.slider("Ordering Cost ($)", 25, 300, 75)
    holding_cost = st.slider("Holding Cost ($/unit)", 0.5, 25.0, 3.5, step=0.5)
    
    st.markdown("---")
    
    service_levels = {"90% (z=1.28)": 1.28, "95% (z=1.65)": 1.65, "99% (z=2.33)": 2.33}
    z_key = st.selectbox(
        "Desired Service Level", 
        list(service_levels.keys()), 
        index=1
    )
    z = service_levels[z_key]

inventory_plan = []
for product in products:
    prod_df = df[df['Product ID'] == product]
    
    demand_per_period = prod_df['Demand Forecast'].sum()
    avg_daily_demand = prod_df['Demand Forecast'].mean() / 30
    std_dev_demand = prod_df['Demand Forecast'].std()
    
    try:
        eoq = np.sqrt((2 * demand_per_period * ordering_cost) / holding_cost)
    except ZeroDivisionError:
        eoq = 0
    
    ss = z * std_dev_demand * np.sqrt(lead_time)
    rop = (avg_daily_demand * lead_time) + ss
    
    inventory_plan.append({
        "Product": product,
        "AvgDailyDemand": round(avg_daily_demand, 2),
        "AnnualDemand": round(demand_per_period, 2),
        "EOQ (Units)": round(eoq, 2),
        "SafetyStock (Units)": round(ss, 2),
        "ReorderPoint (Units)": round(rop, 2)
    })

inv_df = pd.DataFrame(inventory_plan)

inv_df["Value"] = inv_df["AnnualDemand"] * holding_cost
inv_df = inv_df.sort_values(by="Value", ascending=False)
inv_df["Cumulative%"] = inv_df["Value"].cumsum() / inv_df["Value"].sum() * 100
inv_df["ABC_Category"] = inv_df["Cumulative%"].apply(lambda x: "A (Critical)" if x <= 20 else "B (Important)" if x <= 50 else "C (Routine)")

selected_row = inv_df[inv_df["Product"] == selected_product].iloc[0]

st.subheader(f"✨ Optimal Inventory Plan for: **{selected_product}**")
st.markdown(f"**ABC Classification:** <span style='color:#BB86FC; font-weight:bold;'>{selected_row['ABC_Category']}</span>", unsafe_allow_html=True)
st.markdown("---")

col_metrics, col_chart = st.columns([1, 2])

with col_metrics:
    st.metric(
        "Reorder Point (ROP)", 
        f"{selected_row['ReorderPoint (Units)']:.0f} units"
    )
    st.metric(
        "Economic Order Quantity (EOQ)", 
        f"{selected_row['EOQ (Units)']:.0f} units"
    )
    st.metric(
        "Safety Stock (SS)", 
        f"{selected_row['SafetyStock (Units)']:.0f} units"
    )

with col_chart:
    st.markdown("##### Visualizing Inventory Level Simulation")
    time_points = np.arange(0, 50)
    inventory_level = np.concatenate([
        np.linspace(selected_row['ReorderPoint (Units)'] + selected_row['EOQ (Units)'], selected_row['ReorderPoint (Units)'], 20),
        np.linspace(selected_row['ReorderPoint (Units)'], selected_row['SafetyStock (Units)'] + 10, 10),
        np.linspace(selected_row['SafetyStock (Units)'] + 10, selected_row['ReorderPoint (Units)'] + selected_row['EOQ (Units)'], 20)
    ])

    inventory_level = np.clip(inventory_level, selected_row['SafetyStock (Units)'] - 5, None)

    plt.style.use('dark_background')
    plt.figure(figsize=(9, 4.5))
    plt.plot(time_points, inventory_level[:len(time_points)], label="Inventory Level", color="#BB86FC", linewidth=2)
    plt.axhline(y=selected_row["ReorderPoint (Units)"], color="#FFB300", linestyle="-.", label="Reorder Point", linewidth=1.5)
    plt.axhline(y=selected_row["SafetyStock (Units)"], color="#FF4500", linestyle="--", label="Safety Stock", linewidth=1.5)
    
    plt.title("Inventory Trajectory Simulation", color='#F0F2F6')
    plt.xlabel("Time (Days/Periods)", color='#F0F2F6')
    plt.ylabel("Inventory Units", color='#F0F2F6')
    plt.legend()
    st.pyplot(plt.gcf())

tab1, tab2 = st.tabs(["📈 Demand Trends", "📋 Full ABC Analysis"])

with tab1:
    st.subheader("Weekly Demand Forecast Trend")
    prod_trend = df[df["Product ID"] == selected_product].groupby("week")["Demand Forecast"].mean().reset_index()
    
    st.line_chart(
        prod_trend, 
        x="week", 
        y="Demand Forecast",
        color="#BB86FC"
    )
    st.caption("This chart shows the average forecasted demand per week for the selected product.")

with tab2:
    st.subheader("Inventory Distribution by Classification")
    
    abc_summary = inv_df["ABC_Category"].value_counts().reset_index()
    abc_summary.columns = ["Category", "Product Count"]
    
    abc_colors = {"A (Critical)": "#FF4500", "B (Important)": "#FFB300", "C (Routine)": "#4CAF50"}
    
    plt.style.use('dark_background')
    plt.figure(figsize=(7, 4))
    plt.bar(
        abc_summary["Category"], 
        abc_summary["Product Count"], 
        color=[abc_colors[cat] for cat in abc_summary["Category"]]
    )
    plt.title("Product Count by ABC Category", color='#F0F2F6')
    plt.xlabel("Category", color='#F0F2F6')
    plt.ylabel("Number of Products", color='#F0F2F6')
    st.pyplot(plt.gcf())

    st.markdown("---")
    
    st.subheader("Detailed Inventory Plan Table")
    display_cols = ["Product", "ABC_Category", "AnnualDemand", "EOQ (Units)", "ReorderPoint (Units)", "SafetyStock (Units)"]
    st.dataframe(inv_df[display_cols].set_index("Product"), use_container_width=True)

st.markdown("---")
st.download_button(
    "⬇️ Download Full Inventory Plan (CSV)", 
    inv_df.to_csv(index=False).encode('utf-8'), 
    "inventory_optimization_plan_full.csv",
    mime="text/csv"
)

st.markdown(
    "<div style='text-align: center; color: #555555; margin-top: 20px;'>Data courtesy of 'cleaned_retail_data.csv' simulation.</div>", 
    unsafe_allow_html=True
)