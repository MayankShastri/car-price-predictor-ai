import streamlit as st
import pandas as pd
import pickle
import plotly.express as px
import plotly.graph_objects as go
from datetime import date

# 1. Setup & Data Loading
st.set_page_config(page_title="Car Analytics Pro", layout="wide")

@st.cache_resource
def load_model():
    with open('models/car_model.pkl', 'rb') as f:
        return pickle.load(f)

@st.cache_data
def load_data():
    df = pd.read_csv('data/Car details v3.csv')
    df['Price (Lakhs)'] = df['selling_price'] / 100000
    return df

model_data = load_model()
raw_df = load_data()

st.title("🚗 Group 1: AI Car Value Analytics")
st.markdown("---")

left_col, right_col = st.columns([1, 1.5])

with left_col:
    st.subheader("🛠️ Car Specifications")
    kms = st.number_input("Kilometers Driven", value=30000)
    mil = st.number_input("Mileage (km/l)", value=18.0)
    eng = st.number_input("Engine (CC)", value=1200)
    pwr = st.number_input("Max Power (bhp)", value=80.0)
    yr = st.slider("Manufacturing Year", 2010, 2026, 2018)
    fuel = st.selectbox("Fuel Type", ["Diesel", "Petrol", "CNG", "LPG"])
    trans = st.selectbox("Transmission", ["Manual", "Automatic"])
    
    predict_btn = st.button("🚀 Analyze Market Value", use_container_width=True)

if predict_btn:
    # --- PREDICTION LOGIC ---
    feats = model_data["features"]
    input_df = pd.DataFrame(0, index=[0], columns=feats)
    input_df['km_driven'] = kms
    input_df['mileage'] = mil
    input_df['engine'] = eng
    input_df['max_power'] = pwr
    input_df['Car_Age'] = date.today().year - yr
    if f"fuel_{fuel}" in feats: input_df[f"fuel_{fuel}"] = 1
    if f"transmission_{trans}" in feats: input_df[f"transmission_{trans}"] = 1

    res = model_data["model"].predict(input_df)[0]

    with right_col:
        st.subheader("📊 AI Market Analysis")
        st.metric(label="Estimated Resale Value", value=f"₹{res:.2f} Lakhs")
        
        # --- ENHANCED VISUALIZATION ---
        chart_data = raw_df[(raw_df['year'] >= yr - 4) & (raw_df['year'] <= yr + 4)]
        
        if not chart_data.empty:
            # 1. Create Base Scatter (Background)
            fig = px.scatter(
                chart_data, 
                x="km_driven", 
                y="Price (Lakhs)",
                color="fuel",
                title=f"Market Comparison (Cars from {yr-4} to {yr+4})",
                template="plotly_dark",
                hover_data=['name'],
                opacity=0.3 # Fade background dots so the star pops
            )
            
            # 2. Add a "White Halo" Star (Behind the gold star)
            fig.add_trace(go.Scatter(
                x=[kms], y=[res],
                mode='markers',
                marker=dict(size=28, color='white', symbol='star'),
                showlegend=False,
                hoverinfo='skip'
            ))
            
            # 3. Add the "Gold Star" (Final Layer on top)
            fig.add_trace(go.Scatter(
                x=[kms], y=[res],
                mode='markers',
                marker=dict(
                    size=22, 
                    color='gold', 
                    symbol='star',
                    line=dict(width=1, color='black') # Subtle black outline
                ),
                name="YOUR PREDICTION"
            ))

            # 4. Final Layout Adjustments
            fig.update_layout(
                legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
                xaxis_title="Kilometers Driven",
                yaxis_title="Price (Lakhs)"
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No data found to build a comparison chart.")