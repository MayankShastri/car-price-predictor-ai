import streamlit as st
import pandas as pd
import pickle
import plotly.express as px
import plotly.graph_objects as go
from datetime import date

st.set_page_config(page_title="Car Price AI", layout="wide")
st.title("🚗 Group 1: Professional Car Analytics")

# 1. Load Everything
@st.cache_resource
def get_assets():
    with open('models/car_model.pkl', 'rb') as f:
        m_data = pickle.load(f)
    raw = pd.read_csv('data/Car details v3.csv')
    raw['Price_Lakhs'] = raw['selling_price'] / 100000
    return m_data, raw

try:
    assets, raw_df = get_assets()
    model = assets["model"]
    feats = assets["features"]
except Exception as e:
    st.error(f"⚠️ MODEL ERROR: Run train_model.py first! ({e})")
    st.stop()

# 2. UI Layout
left, right = st.columns([1, 1.5])

with left:
    st.subheader("🛠️ Specifications")
    kms = st.number_input("Kms Driven", value=30000)
    mil = st.number_input("Mileage (km/l)", value=18.0)
    eng = st.number_input("Engine (CC)", value=1200)
    pwr = st.number_input("Power (bhp)", value=80.0)
    yr = st.slider("Year", 2010, 2026, 2018)
    fuel = st.selectbox("Fuel", ["Diesel", "Petrol", "CNG", "LPG"])
    trans = st.selectbox("Transmission", ["Manual", "Automatic"])
    
    if st.button("🚀 Analyze Market", use_container_width=True):
        # 3. Shape-Matching Logic
        input_df = pd.DataFrame(0, index=[0], columns=feats)
        input_df['km_driven'] = kms
        input_df['mileage'] = mil
        input_df['engine'] = eng
        input_df['max_power'] = pwr
        input_df['Car_Age'] = 2026 - yr
        
        # Match categories
        for col in [f"fuel_{fuel}", f"transmission_{trans}"]:
            if col in feats: 
                input_df[col] = 1

        # Calculate prediction
        prediction = model.predict(input_df)[0]

        with right: # FIXED: Changed from right_col to right
            st.subheader("📊 AI Market Analysis")
            st.metric("Estimated Resale Value", f"₹{prediction:.2f} Lakhs") # FIXED: Changed res to prediction
            
            # 4. Filter for context (Cars within +/- 4 years)
            chart_data = raw_df[(raw_df['year'] >= yr-4) & (raw_df['year'] <= yr+4)]
            
            if not chart_data.empty:
                fig = go.Figure()

                # TRACE 1: The Background Market (Layer 1)
                fig.add_trace(go.Scatter(
                    x=chart_data['km_driven'], 
                    y=chart_data['Price_Lakhs'],
                    mode='markers',
                    name='Market Context',
                    marker=dict(
                        color='rgba(180, 180, 180, 0.2)', # Very light gray & transparent
                        size=6
                    ),
                    zorder=1 # Forces this to the bottom layer
                ))

                # TRACE 2: THE STAR (Layer 10 - Absolute Front)
                fig.add_trace(go.Scatter(
                    x=[kms], 
                    y=[prediction], # FIXED: Changed res to prediction
                    mode='markers',
                    name='YOUR PREDICTION',
                    marker=dict(
                        color='#FFD700',       # Gold
                        size=30,               # Massive size
                        symbol='star', 
                        line=dict(width=3, color='white') # High-contrast white halo
                    ),
                    zorder=10 # Forces this trace to sit on top
                ))

                fig.update_layout(
                    template="plotly_dark",
                    xaxis_title="Kilometers Driven",
                    yaxis_title="Price (Lakhs)",
                    xaxis=dict(showgrid=False, zeroline=False),
                    yaxis=dict(showgrid=False, zeroline=False),
                    legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
                )

                # Force SVG rendering mode to ensure zorder is respected
                st.plotly_chart(fig, use_container_width=True, theme=None)
            else:
                st.warning("Not enough data to generate a comparison chart for this year range.")