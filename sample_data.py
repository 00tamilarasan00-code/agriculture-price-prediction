import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

def generate_sample_data():
    """Generate sample commodity price data for demonstration"""
    
    # Define commodities
    commodities = ['Rice', 'Wheat', 'Corn', 'Soybeans', 'Cotton', 'Sugar']
    
    # Generate date range (2 years of data)
    start_date = datetime.now() - timedelta(days=730)
    dates = [start_date + timedelta(days=x) for x in range(730)]
    
    data = []
    
    for commodity in commodities:
        # Base prices (USD per ton)
        base_prices = {
            'Rice': 400,
            'Wheat': 250,
            'Corn': 200,
            'Soybeans': 450,
            'Cotton': 1500,
            'Sugar': 350
        }
        
        base_price = base_prices[commodity]
        
        for i, date in enumerate(dates):
            # Add seasonal variation
            seasonal_factor = 1 + 0.1 * np.sin(2 * np.pi * i / 365)
            
            # Add trend
            trend_factor = 1 + (i / 730) * 0.2  # 20% increase over 2 years
            
            # Add random variation
            random_factor = 1 + random.uniform(-0.15, 0.15)
            
            # Calculate price
            price = base_price * seasonal_factor * trend_factor * random_factor
            
            # Add market factors
            rainfall = random.uniform(50, 200)  # mm
            temperature = random.uniform(15, 35)  # Celsius
            market_demand = random.uniform(0.8, 1.2)
            supply_index = random.uniform(0.7, 1.3)
            
            data.append({
                'Date': date,
                'Commodity': commodity,
                'Price': round(price, 2),
                'Rainfall': round(rainfall, 1),
                'Temperature': round(temperature, 1),
                'Market_Demand': round(market_demand, 2),
                'Supply_Index': round(supply_index, 2),
                'Season': 'Summer' if date.month in [6,7,8] else 'Winter' if date.month in [12,1,2] else 'Spring' if date.month in [3,4,5] else 'Autumn'
            })
    
    return pd.DataFrame(data)

def get_latest_prices():
    """Get latest prices for each commodity"""
    df = generate_sample_data()
    latest_data = df.groupby('Commodity').tail(1).reset_index(drop=True)
    return latest_data

if __name__ == "__main__":
    df = generate_sample_data()
    print(df.head())
    print(f"Generated {len(df)} records for {df['Commodity'].nunique()} commodities")