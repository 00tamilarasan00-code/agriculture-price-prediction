import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Import custom modules
from sample_data import generate_sample_data, get_latest_prices
from data_processor import DataProcessor
from ml_models import CommodityPricePredictor

# Page configuration
st.set_page_config(
    page_title="Agriculture Commodity Price Prediction",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #2E8B57;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #2E8B57;
    }
    .prediction-box {
        background-color: #e8f5e8;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border: 2px solid #2E8B57;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'models_trained' not in st.session_state:
    st.session_state.models_trained = False

# Main title
st.markdown('<h1 class="main-header">🌾 Agriculture Commodity Price Prediction</h1>', unsafe_allow_html=True)

# Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.selectbox(
    "Choose a page",
    ["🏠 Home", "📊 Data Analysis", "🤖 ML Models", "🔮 Price Prediction", "📈 Market Trends", "📋 Protocol"]
)

# Load data
@st.cache_data
def load_data():
    return generate_sample_data()

# Initialize components
data_processor = DataProcessor()
predictor = CommodityPricePredictor()

# Home Page
if page == "🏠 Home":
    st.markdown("## Welcome to the Agriculture Commodity Price Prediction System")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        ### 🎯 Purpose
        This application helps farmers, traders, and policymakers make informed decisions by predicting commodity prices using advanced machine learning algorithms.
        """)
    
    with col2:
        st.markdown("""
        ### 🔧 Features
        - Real-time price predictions
        - Multiple ML models comparison
        - Historical data analysis
        - Market trend visualization
        - Research protocol documentation
        """)
    
    with col3:
        st.markdown("""
        ### 🌾 Commodities
        - Rice
        - Wheat
        - Corn
        - Soybeans
        - Cotton
        - Sugar
        """)
    
    # Load and display sample data
    if st.button("Load Sample Data", type="primary"):
        with st.spinner("Loading data..."):
            df = load_data()
            st.session_state.data_loaded = True
            st.success(f"✅ Loaded {len(df)} records for {df['Commodity'].nunique()} commodities")
    
    if st.session_state.data_loaded:
        df = load_data()
        
        # Display latest prices
        st.markdown("### 📊 Latest Commodity Prices")
        latest_prices = get_latest_prices()
        
        cols = st.columns(3)
        for i, (_, row) in enumerate(latest_prices.iterrows()):
            with cols[i % 3]:
                st.markdown(f"""
                <div class="metric-card">
                    <h4>{row['Commodity']}</h4>
                    <h2>${row['Price']:.2f}/ton</h2>
                    <p>Last updated: {row['Date'].strftime('%Y-%m-%d')}</p>
                </div>
                """, unsafe_allow_html=True)

# Data Analysis Page
elif page == "📊 Data Analysis":
    st.markdown("## 📊 Data Analysis and Visualization")
    
    if not st.session_state.data_loaded:
        st.warning("Please load data from the Home page first.")
        if st.button("Load Data Now"):
            df = load_data()
            st.session_state.data_loaded = True
            st.rerun()
    else:
        df = load_data()
        
        # Data overview
        st.markdown("### Data Overview")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Records", len(df))
        with col2:
            st.metric("Commodities", df['Commodity'].nunique())
        with col3:
            st.metric("Date Range", f"{(df['Date'].max() - df['Date'].min()).days} days")
        with col4:
            st.metric("Avg Price", f"${df['Price'].mean():.2f}")
        
        # Price trends
        st.markdown("### 📈 Price Trends Over Time")
        
        selected_commodities = st.multiselect(
            "Select commodities to display:",
            options=df['Commodity'].unique(),
            default=df['Commodity'].unique()[:3]
        )
        
        if selected_commodities:
            filtered_df = df[df['Commodity'].isin(selected_commodities)]
            
            fig = px.line(
                filtered_df, 
                x='Date', 
                y='Price', 
                color='Commodity',
                title="Commodity Price Trends",
                labels={'Price': 'Price (USD/ton)', 'Date': 'Date'}
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
        
        # Correlation analysis
        st.markdown("### 🔗 Correlation Analysis")
        
        numeric_cols = ['Price', 'Rainfall', 'Temperature', 'Market_Demand', 'Supply_Index']
        correlation_matrix = df[numeric_cols].corr()
        
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, ax=ax)
        st.pyplot(fig)
        
        # Seasonal analysis
        st.markdown("### 🌿 Seasonal Price Analysis")
        
        seasonal_data = df.groupby(['Commodity', 'Season'])['Price'].mean().reset_index()
        
        fig = px.bar(
            seasonal_data,
            x='Season',
            y='Price',
            color='Commodity',
            title="Average Prices by Season",
            labels={'Price': 'Average Price (USD/ton)'}
        )
        st.plotly_chart(fig, use_container_width=True)

# ML Models Page
elif page == "🤖 ML Models":
    st.markdown("## 🤖 Machine Learning Models")
    
    if not st.session_state.data_loaded:
        st.warning("Please load data from the Home page first.")
    else:
        df = load_data()
        
        # Train models
        if st.button("Train ML Models", type="primary"):
            with st.spinner("Training models... This may take a few minutes."):
                # Preprocess data
                processed_df = data_processor.preprocess_data(df)
                X, y, feature_names = data_processor.prepare_features(processed_df)
                
                # Train models
                performance = predictor.train_models(X, y)
                st.session_state.models_trained = True
                st.session_state.model_performance = performance
                st.session_state.feature_names = feature_names
                
                st.success("✅ Models trained successfully!")
        
        if st.session_state.models_trained:
            st.markdown("### 📊 Model Performance Comparison")
            
            performance_df = pd.DataFrame(st.session_state.model_performance).T
            
            # Display metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("#### Mean Absolute Error (MAE)")
                mae_fig = px.bar(
                    x=performance_df.index,
                    y=performance_df['MAE'],
                    title="MAE Comparison"
                )
                st.plotly_chart(mae_fig, use_container_width=True)
            
            with col2:
                st.markdown("#### Root Mean Square Error (RMSE)")
                rmse_fig = px.bar(
                    x=performance_df.index,
                    y=performance_df['RMSE'],
                    title="RMSE Comparison"
                )
                st.plotly_chart(rmse_fig, use_container_width=True)
            
            with col3:
                st.markdown("#### R-squared Score")
                r2_fig = px.bar(
                    x=performance_df.index,
                    y=performance_df['R2'],
                    title="R² Comparison"
                )
                st.plotly_chart(r2_fig, use_container_width=True)
            
            # Feature importance
            st.markdown("### 🎯 Feature Importance (Random Forest)")
            
            importance = predictor.get_feature_importance()
            if importance is not None:
                importance_df = pd.DataFrame({
                    'Feature': st.session_state.feature_names,
                    'Importance': importance
                }).sort_values('Importance', ascending=False)
                
                fig = px.bar(
                    importance_df.head(10),
                    x='Importance',
                    y='Feature',
                    orientation='h',
                    title="Top 10 Most Important Features"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Performance table
            st.markdown("### 📋 Detailed Performance Metrics")
            st.dataframe(performance_df.round(4))

# Price Prediction Page
elif page == "🔮 Price Prediction":
    st.markdown("## 🔮 Commodity Price Prediction")
    
    if not st.session_state.models_trained:
        st.warning("Please train the ML models first from the ML Models page.")
    else:
        df = load_data()
        
        st.markdown("### Input Parameters for Prediction")
        
        col1, col2 = st.columns(2)
        
        with col1:
            commodity = st.selectbox(
                "Select Commodity:",
                options=df['Commodity'].unique()
            )
            
            rainfall = st.slider(
                "Rainfall (mm):",
                min_value=0.0,
                max_value=300.0,
                value=100.0,
                step=5.0
            )
            
            temperature = st.slider(
                "Temperature (°C):",
                min_value=0.0,
                max_value=50.0,
                value=25.0,
                step=1.0
            )
        
        with col2:
            market_demand = st.slider(
                "Market Demand Index:",
                min_value=0.5,
                max_value=1.5,
                value=1.0,
                step=0.1
            )
            
            supply_index = st.slider(
                "Supply Index:",
                min_value=0.5,
                max_value=1.5,
                value=1.0,
                step=0.1
            )
            
            season = st.selectbox(
                "Season:",
                options=['Spring', 'Summer', 'Autumn', 'Winter']
            )
        
        # Model selection
        model_type = st.selectbox(
            "Select Prediction Model:",
            options=['Random Forest', 'Linear Regression']
        )
        
        if st.button("Predict Price", type="primary"):
            with st.spinner("Making prediction..."):
                # Create input data
                input_df = data_processor.create_prediction_input(
                    commodity, rainfall, temperature, market_demand, supply_index, season
                )
                
                # Process input
                processed_input = data_processor.preprocess_data(input_df)
                X_input, _, _ = data_processor.prepare_features(processed_input)
                
                # Make prediction
                predicted_price = predictor.predict(X_input, model_type=model_type)[0]
                
                # Display prediction
                st.markdown(f"""
                <div class="prediction-box">
                    <h2>🎯 Prediction Result</h2>
                    <h1 style="color: #2E8B57;">${predicted_price:.2f} per ton</h1>
                    <p><strong>Commodity:</strong> {commodity}</p>
                    <p><strong>Model Used:</strong> {model_type}</p>
                    <p><strong>Prediction Date:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Show confidence interval (simplified)
                confidence_range = predicted_price * 0.1  # ±10% confidence interval
                st.info(f"📊 Confidence Interval: ${predicted_price - confidence_range:.2f} - ${predicted_price + confidence_range:.2f}")
                
                # Historical comparison
                historical_data = df[df['Commodity'] == commodity]
                avg_price = historical_data['Price'].mean()
                price_change = ((predicted_price - avg_price) / avg_price) * 100
                
                if price_change > 0:
                    st.success(f"📈 Predicted price is {price_change:.1f}% higher than historical average (${avg_price:.2f})")
                else:
                    st.warning(f"📉 Predicted price is {abs(price_change):.1f}% lower than historical average (${avg_price:.2f})")

# Market Trends Page
elif page == "📈 Market Trends":
    st.markdown("## 📈 Market Trends and Insights")
    
    if not st.session_state.data_loaded:
        st.warning("Please load data from the Home page first.")
    else:
        df = load_data()
        
        # Market overview
        st.markdown("### 🌍 Market Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            highest_price = df.loc[df['Price'].idxmax()]
            st.metric(
                "Highest Price",
                f"${highest_price['Price']:.2f}",
                f"{highest_price['Commodity']}"
            )
        
        with col2:
            lowest_price = df.loc[df['Price'].idxmin()]
            st.metric(
                "Lowest Price",
                f"${lowest_price['Price']:.2f}",
                f"{lowest_price['Commodity']}"
            )
        
        with col3:
            most_volatile = df.groupby('Commodity')['Price'].std().idxmax()
            volatility = df.groupby('Commodity')['Price'].std().max()
            st.metric(
                "Most Volatile",
                most_volatile,
                f"σ = {volatility:.2f}"
            )
        
        with col4:
            avg_growth = ((df.groupby('Commodity')['Price'].last() / df.groupby('Commodity')['Price'].first() - 1) * 100).mean()
            st.metric(
                "Avg Growth",
                f"{avg_growth:.1f}%",
                "2-year period"
            )
        
        # Price distribution
        st.markdown("### 📊 Price Distribution by Commodity")
        
        fig = px.box(
            df,
            x='Commodity',
            y='Price',
            title="Price Distribution Across Commodities"
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
        
        # Weather impact analysis
        st.markdown("### 🌦️ Weather Impact on Prices")
        
        selected_commodity = st.selectbox(
            "Select commodity for weather analysis:",
            options=df['Commodity'].unique()
        )
        
        commodity_data = df[df['Commodity'] == selected_commodity]
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig1 = px.scatter(
                commodity_data,
                x='Rainfall',
                y='Price',
                title=f"Rainfall vs Price - {selected_commodity}",
                trendline="ols"
            )
            st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            fig2 = px.scatter(
                commodity_data,
                x='Temperature',
                y='Price',
                title=f"Temperature vs Price - {selected_commodity}",
                trendline="ols"
            )
            st.plotly_chart(fig2, use_container_width=True)
        
        # Market sentiment
        st.markdown("### 📊 Market Sentiment Analysis")
        
        sentiment_data = df.groupby(['Date', 'Commodity']).agg({
            'Market_Demand': 'mean',
            'Supply_Index': 'mean',
            'Price': 'mean'
        }).reset_index()
        
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Market Demand Over Time', 'Supply Index Over Time'),
            vertical_spacing=0.1
        )
        
        for commodity in df['Commodity'].unique()[:3]:  # Show top 3 commodities
            commodity_sentiment = sentiment_data[sentiment_data['Commodity'] == commodity]
            
            fig.add_trace(
                go.Scatter(
                    x=commodity_sentiment['Date'],
                    y=commodity_sentiment['Market_Demand'],
                    name=f'{commodity} Demand',
                    mode='lines'
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=commodity_sentiment['Date'],
                    y=commodity_sentiment['Supply_Index'],
                    name=f'{commodity} Supply',
                    mode='lines'
                ),
                row=2, col=1
            )
        
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)

# Protocol Page
elif page == "📋 Protocol":
    st.markdown("## 📋 Research Protocol and Methodology")
    
    # Read and display protocol
    try:
        with open('protocol.md', 'r') as f:
            protocol_content = f.read()
        st.markdown(protocol_content)
    except FileNotFoundError:
        st.error("Protocol file not found. Please ensure protocol.md exists in the project directory.")
    
    # Download button for protocol
    if st.button("📥 Download Protocol Document"):
        try:
            with open('protocol.md', 'r') as f:
                protocol_content = f.read()
            st.download_button(
                label="Download Protocol.md",
                data=protocol_content,
                file_name="agriculture_commodity_prediction_protocol.md",
                mime="text/markdown"
            )
        except FileNotFoundError:
            st.error("Protocol file not found.")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    <p>🌾 Agriculture Commodity Price Prediction System | Built with Streamlit & Machine Learning</p>
    <p>For research and educational purposes | © 2024</p>
</div>
""", unsafe_allow_html=True)
