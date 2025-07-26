import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pickle
from datetime import datetime

# Set page configuration
st.set_page_config(
    page_title="IoT Plant Growth Monitoring",
    page_icon="🌱",
    layout="wide"
)

# Load the trained model
@st.cache_resource
def load_model():
    with open('plant_growth_model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

# Load and cache the data
@st.cache_data
def load_data():
    df = pd.read_csv('Advanced_IoT_Dataset.csv')
    return df

# Main function
def main():
    # Add title and description
    st.title("🌱 IoT Plant Growth Monitoring System")
    st.markdown("""
    This dashboard provides real-time monitoring and predictions for plant growth based on IoT sensor data.
    """)

    # Load data and model
    try:
        df = load_data()
        model = load_model()
    except Exception as e:
        st.error(f"Error loading data or model: {str(e)}")
        return

    # Sidebar
    st.sidebar.header("Dashboard Controls")
    
    # Date filter
    date_range = st.sidebar.date_input(
        "Select Date Range",
        value=[df['Timestamp'].min(), df['Timestamp'].max()]
    )

    # Main dashboard layout
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Environmental Conditions Over Time")
        
        # Temperature trend
        fig_temp = px.line(df, x='Timestamp', y='Temperature',
                          title='Temperature Variation')
        st.plotly_chart(fig_temp, use_container_width=True)

        # Humidity trend
        fig_humidity = px.line(df, x='Timestamp', y='Humidity',
                             title='Humidity Variation')
        st.plotly_chart(fig_humidity, use_container_width=True)

    with col2:
        st.subheader("Plant Growth Metrics")
        
        # Growth rate visualization
        fig_growth = px.scatter(df, x='Temperature', y='Growth_Rate',
                              color='Humidity', title='Growth Rate vs Temperature',
                              size='Light_Intensity')
        st.plotly_chart(fig_growth, use_container_width=True)

        # Light intensity distribution
        fig_light = px.histogram(df, x='Light_Intensity',
                               title='Light Intensity Distribution')
        st.plotly_chart(fig_light, use_container_width=True)

    # Predictions section
    st.header("Growth Prediction")
    
    # Input form for predictions
    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            temperature = st.number_input("Temperature (°C)", 
                                        min_value=0.0, 
                                        max_value=50.0,
                                        value=25.0)
        
        with col2:
            humidity = st.number_input("Humidity (%)", 
                                     min_value=0.0,
                                     max_value=100.0,
                                     value=60.0)
        
        with col3:
            light_intensity = st.number_input("Light Intensity (lux)",
                                            min_value=0.0,
                                            max_value=100000.0,
                                            value=5000.0)
        
        submitted = st.form_submit_button("Predict Growth Rate")
        
        if submitted:
            # Make prediction
            input_data = np.array([[temperature, humidity, light_intensity]])
            prediction = model.predict(input_data)[0]
            
            st.success(f"Predicted Growth Rate: {prediction:.2f} cm/day")

    # Historical statistics
    st.header("Historical Statistics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Avg Temperature", 
                 f"{df['Temperature'].mean():.1f}°C",
                 f"{df['Temperature'].std():.1f}°C")
    
    with col2:
        st.metric("Avg Humidity",
                 f"{df['Humidity'].mean():.1f}%",
                 f"{df['Humidity'].std():.1f}%")
    
    with col3:
        st.metric("Avg Light Intensity",
                 f"{df['Light_Intensity'].mean():.0f} lux",
                 f"{df['Light_Intensity'].std():.0f} lux")
    
    with col4:
        st.metric("Avg Growth Rate",
                 f"{df['Growth_Rate'].mean():.2f} cm/day",
                 f"{df['Growth_Rate'].std():.2f} cm/day")

if __name__ == "__main__":
    main()
