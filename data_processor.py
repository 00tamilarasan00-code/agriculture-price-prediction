import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

class DataProcessor:
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.feature_columns = []
        self.imputer = SimpleImputer(strategy='mean')
        
    def clean_data(self, df):
        """Comprehensive data cleaning and preprocessing"""
        
        try:
            st.write("🧹 **Starting Data Cleaning Process...**")
            
            # Create a copy to avoid modifying original data
            cleaned_df = df.copy()
            initial_rows = len(cleaned_df)
            
            # Display initial data info
            st.write(f"📊 **Initial Dataset Info:**")
            st.write(f"- Total rows: {initial_rows}")
            st.write(f"- Total columns: {len(cleaned_df.columns)}")
            
            # Check for missing values
            missing_summary = cleaned_df.isnull().sum()
            missing_percent = (missing_summary / len(cleaned_df)) * 100
            
            if missing_summary.sum() > 0:
                st.write("⚠️ **Missing Values Detected:**")
                missing_df = pd.DataFrame({
                    'Column': missing_summary.index,
                    'Missing Count': missing_summary.values,
                    'Missing Percentage': missing_percent.values
                })
                missing_df = missing_df[missing_df['Missing Count'] > 0]
                st.dataframe(missing_df)
            else:
                st.write("✅ **No missing values found**")
            
            # Handle missing values for different column types
            for column in cleaned_df.columns:
                missing_count = cleaned_df[column].isnull().sum()
                if missing_count > 0:
                    if cleaned_df[column].dtype in ['object', 'string']:
                        # For categorical columns, fill with mode or 'Unknown'
                        mode_value = cleaned_df[column].mode()
                        if len(mode_value) > 0:
                            cleaned_df[column] = cleaned_df[column].fillna(mode_value[0])
                            st.write(f"🔧 Filled {missing_count} missing values in '{column}' with mode: '{mode_value[0]}'")
                        else:
                            cleaned_df[column] = cleaned_df[column].fillna('Unknown')
                            st.write(f"🔧 Filled {missing_count} missing values in '{column}' with 'Unknown'")
                    else:
                        # For numerical columns, fill with median
                        median_value = cleaned_df[column].median()
                        cleaned_df[column] = cleaned_df[column].fillna(median_value)
                        st.write(f"🔧 Filled {missing_count} missing values in '{column}' with median: {median_value:.2f}")
            
            # Remove duplicate rows
            duplicates = cleaned_df.duplicated().sum()
            if duplicates > 0:
                cleaned_df = cleaned_df.drop_duplicates()
                st.write(f"🗑️ Removed {duplicates} duplicate rows")
            else:
                st.write("✅ No duplicate rows found")
            
            # Handle outliers in price columns
            price_columns = [col for col in cleaned_df.columns if 'price' in col.lower()]
            for price_col in price_columns:
                if cleaned_df[price_col].dtype in ['int64', 'float64']:
                    # Remove extreme outliers using IQR method
                    Q1 = cleaned_df[price_col].quantile(0.25)
                    Q3 = cleaned_df[price_col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 3 * IQR  # Using 3*IQR for extreme outliers
                    upper_bound = Q3 + 3 * IQR
                    
                    outliers_count = len(cleaned_df[(cleaned_df[price_col] < lower_bound) | (cleaned_df[price_col] > upper_bound)])
                    if outliers_count > 0:
                        # Cap outliers instead of removing them
                        cleaned_df[price_col] = cleaned_df[price_col].clip(lower=lower_bound, upper=upper_bound)
                        st.write(f"📊 Capped {outliers_count} outliers in '{price_col}' (Range: {lower_bound:.2f} - {upper_bound:.2f})")
            
            # Standardize text columns
            text_columns = cleaned_df.select_dtypes(include=['object']).columns
            for col in text_columns:
                if col in cleaned_df.columns:
                    # Remove extra spaces and standardize case
                    cleaned_df[col] = cleaned_df[col].astype(str).str.strip().str.title()
                    st.write(f"📝 Standardized text format in '{col}'")
            
            # Validate and clean date columns
            date_columns = [col for col in cleaned_df.columns if 'date' in col.lower()]
            for date_col in date_columns:
                try:
                    # Try to convert to datetime
                    cleaned_df[date_col] = pd.to_datetime(cleaned_df[date_col], errors='coerce')
                    invalid_dates = cleaned_df[date_col].isnull().sum()
                    if invalid_dates > 0:
                        st.write(f"⚠️ Found {invalid_dates} invalid dates in '{date_col}', removing those rows")
                        cleaned_df = cleaned_df.dropna(subset=[date_col])
                except Exception as e:
                    st.write(f"❌ Error processing date column '{date_col}': {str(e)}")
            
            # Remove rows with negative prices
            for price_col in price_columns:
                if price_col in cleaned_df.columns:
                    negative_prices = (cleaned_df[price_col] < 0).sum()
                    if negative_prices > 0:
                        cleaned_df = cleaned_df[cleaned_df[price_col] >= 0]
                        st.write(f"🗑️ Removed {negative_prices} rows with negative prices in '{price_col}'")
            
            final_rows = len(cleaned_df)
            rows_removed = initial_rows - final_rows
            
            st.write("✅ **Data Cleaning Completed!**")
            st.write(f"📊 **Final Dataset Info:**")
            st.write(f"- Final rows: {final_rows}")
            st.write(f"- Rows removed: {rows_removed} ({(rows_removed/initial_rows)*100:.1f}%)")
            st.write(f"- Data quality improved: {((final_rows/initial_rows)*100):.1f}% data retained")
            
            return cleaned_df
            
        except Exception as e:
            st.error(f"Error in data cleaning: {e}")
            return df
    
    def preprocess_data(self, df):
        """Preprocess the raw data with comprehensive cleaning"""
        
        try:
            import streamlit as st
            
            # First, clean the data
            processed_df = self.clean_data(df)
            
            st.write("🔄 **Starting Data Preprocessing...**")
            
            # Ensure Date column is datetime
            if 'Date' in processed_df.columns:
                processed_df['Date'] = pd.to_datetime(processed_df['Date'])
                
                # Extract date features
                processed_df['Year'] = processed_df['Date'].dt.year
                processed_df['Month'] = processed_df['Date'].dt.month
                processed_df['Day'] = processed_df['Date'].dt.day
                processed_df['DayOfYear'] = processed_df['Date'].dt.dayofyear
                processed_df['WeekOfYear'] = processed_df['Date'].dt.isocalendar().week
                processed_df['Quarter'] = processed_df['Date'].dt.quarter
                processed_df['DayOfWeek'] = processed_df['Date'].dt.dayofweek
                
                st.write("📅 Extracted date features: Year, Month, Day, DayOfYear, WeekOfYear, Quarter, DayOfWeek")
            
            # Handle categorical variables
            categorical_columns = ['Commodity', 'Season', 'State', 'District', 'Market', 'Variety', 'Grade']
            encoded_columns = []
            
            for col in categorical_columns:
                if col in processed_df.columns:
                    if col not in self.label_encoders:
                        self.label_encoders[col] = LabelEncoder()
                        processed_df[f'{col}_encoded'] = self.label_encoders[col].fit_transform(processed_df[col].astype(str))
                        encoded_columns.append(f'{col}_encoded')
                    else:
                        # Handle unseen categories
                        try:
                            processed_df[f'{col}_encoded'] = self.label_encoders[col].transform(processed_df[col].astype(str))
                            encoded_columns.append(f'{col}_encoded')
                        except ValueError:
                            # If new category, assign a default value
                            processed_df[f'{col}_encoded'] = 0
                            encoded_columns.append(f'{col}_encoded')
            
            if encoded_columns:
                st.write(f"🔤 Encoded categorical variables: {', '.join(encoded_columns)}")
            
            # Fill missing values for optional columns with intelligent defaults
            optional_columns = {
                'Rainfall': 100.0,  # Default rainfall
                'Temperature': 25.0,   # Default temperature
                'Market_Demand': 1.0,    # Default market demand
                'Supply_Index': 1.0,    # Default supply index
                'Humidity': 60.0,       # Default humidity
                'Wind_Speed': 5.0       # Default wind speed
            }
            
            filled_columns = []
            for col, default_value in optional_columns.items():
                if col in processed_df.columns:
                    missing_count = processed_df[col].isnull().sum()
                    if missing_count > 0:
                        processed_df[col] = processed_df[col].fillna(default_value)
                        filled_columns.append(f"{col} ({missing_count} values)")
                else:
                    # Create default values if column doesn't exist
                    processed_df[col] = default_value
                    filled_columns.append(f"{col} (new column)")
            
            if filled_columns:
                st.write(f"📊 Filled/created optional columns: {', '.join(filled_columns)}")
            
            # Create default Season if not exists
            if 'Season' not in processed_df.columns and 'Month' in processed_df.columns:
                processed_df['Season'] = processed_df['Month'].map({
                    12: 'Winter', 1: 'Winter', 2: 'Winter',
                    3: 'Spring', 4: 'Spring', 5: 'Spring',
                    6: 'Summer', 7: 'Summer', 8: 'Summer',
                    9: 'Autumn', 10: 'Autumn', 11: 'Autumn'
                })
                # Encode the created Season column
                if 'Season' not in self.label_encoders:
                    self.label_encoders['Season'] = LabelEncoder()
                    processed_df['Season_encoded'] = self.label_encoders['Season'].fit_transform(processed_df['Season'])
                
                st.write("🌿 Created Season column based on month")
            
            # Normalize numerical columns
            numerical_columns = processed_df.select_dtypes(include=[np.number]).columns
            skewed_columns = []
            
            for col in numerical_columns:
                if col not in ['Year', 'Month', 'Day'] and processed_df[col].std() > 0:
                    # Check for skewness
                    skewness = processed_df[col].skew()
                    if abs(skewness) > 1:  # Highly skewed
                        # Apply log transformation for positive skewed data
                        if processed_df[col].min() > 0:
                            processed_df[f'{col}_log'] = np.log1p(processed_df[col])
                            skewed_columns.append(f"{col}_log")
            
            if skewed_columns:
                st.write(f"📈 Applied log transformation to skewed columns: {', '.join(skewed_columns)}")
            
            # Create interaction features for important combinations
            if 'Temperature' in processed_df.columns and 'Rainfall' in processed_df.columns:
                processed_df['Temp_Rainfall_Interaction'] = processed_df['Temperature'] * processed_df['Rainfall']
                st.write("🔗 Created Temperature-Rainfall interaction feature")
            
            if 'Market_Demand' in processed_df.columns and 'Supply_Index' in processed_df.columns:
                processed_df['Demand_Supply_Ratio'] = processed_df['Market_Demand'] / processed_df['Supply_Index']
                st.write("⚖️ Created Demand-Supply ratio feature")
            
            st.write("✅ **Data Preprocessing Completed Successfully!**")
            
            return processed_df
            
        except Exception as e:
            print(f"Error in preprocessing: {e}")
            return df
    
    def prepare_features(self, df):
        """Prepare features for machine learning with advanced feature engineering"""
        
        try:
            import streamlit as st
            
            st.write("🤖 **Preparing Features for Machine Learning...**")
            
            # Select feature columns
            base_feature_cols = [
                'Rainfall', 'Temperature', 'Market_Demand', 'Supply_Index',
                'Year', 'Month', 'Day', 'DayOfYear', 'WeekOfYear', 'Quarter', 'DayOfWeek',
                'Commodity_encoded', 'Season_encoded'
            ]
            
            # Add optional encoded columns
            optional_encoded_cols = [
                'State_encoded', 'District_encoded', 'Market_encoded', 
                'Variety_encoded', 'Grade_encoded'
            ]
            
            # Add interaction and transformation features
            interaction_cols = [
                'Temp_Rainfall_Interaction', 'Demand_Supply_Ratio',
                'Rainfall_log', 'Temperature_log', 'Price_log'
            ]
            
            # Combine all potential feature columns
            all_potential_cols = base_feature_cols + optional_encoded_cols + interaction_cols
            
            # Filter existing columns
            available_cols = [col for col in all_potential_cols if col in df.columns]
            
            if len(available_cols) == 0:
                raise ValueError("No valid feature columns found")
            
            st.write(f"📊 Selected {len(available_cols)} features: {', '.join(available_cols)}")
            
            # Create feature matrix
            X = df[available_cols].copy()
            
            # Handle any remaining missing values
            missing_features = X.isnull().sum()
            if missing_features.sum() > 0:
                st.write("🔧 Handling remaining missing values in features...")
                # Use median for numerical, mode for categorical
                for col in X.columns:
                    if X[col].isnull().sum() > 0:
                        if X[col].dtype in ['int64', 'float64']:
                            X[col] = X[col].fillna(X[col].median())
                        else:
                            X[col] = X[col].fillna(X[col].mode()[0] if len(X[col].mode()) > 0 else 0)
            
            # Create advanced lag features for price (if enough data)
            if 'Price' in df.columns and len(df) > 10:
                st.write("📈 Creating lag and moving average features...")
                
                # Sort by commodity and date for proper lag calculation
                sort_cols = ['Commodity_encoded']
                if 'Date' in df.columns:
                    sort_cols.append('Date')
                
                df_sorted = df.sort_values(sort_cols)
                
                # Add lag features
                lag_features = []
                for lag in [1, 3, 7, 14]:
                    if len(df_sorted) > lag:
                        lag_col = f'Price_lag_{lag}'
                        df_sorted[lag_col] = df_sorted.groupby('Commodity_encoded')['Price'].shift(lag)
                        if len(df_sorted) == len(X):  # Ensure same length
                            X[lag_col] = df_sorted[lag_col].fillna(df_sorted['Price'].median())
                            lag_features.append(lag_col)
                
                # Add moving averages
                ma_features = []
                for window in [3, 7, 14, 30]:
                    if len(df_sorted) > window:
                        ma_col = f'Price_ma_{window}'
                        df_sorted[ma_col] = df_sorted.groupby('Commodity_encoded')['Price'].rolling(window=window).mean().reset_index(0, drop=True)
                        if len(df_sorted) == len(X):  # Ensure same length
                            X[ma_col] = df_sorted[ma_col].fillna(df_sorted['Price'].median())
                            ma_features.append(ma_col)
                
                # Add volatility features
                volatility_features = []
                for window in [7, 14, 30]:
                    if len(df_sorted) > window:
                        vol_col = f'Price_volatility_{window}'
                        df_sorted[vol_col] = df_sorted.groupby('Commodity_encoded')['Price'].rolling(window=window).std().reset_index(0, drop=True)
                        if len(df_sorted) == len(X):  # Ensure same length
                            X[vol_col] = df_sorted[vol_col].fillna(0)
                            volatility_features.append(vol_col)
                
                # Add price change features
                change_features = []
                for period in [1, 7, 30]:
                    if len(df_sorted) > period:
                        change_col = f'Price_change_{period}'
                        df_sorted[change_col] = df_sorted.groupby('Commodity_encoded')['Price'].pct_change(periods=period)
                        if len(df_sorted) == len(X):  # Ensure same length
                            X[change_col] = df_sorted[change_col].fillna(0)
                            change_features.append(change_col)
                
                all_new_features = lag_features + ma_features + volatility_features + change_features
                if all_new_features:
                    st.write(f"📊 Created {len(all_new_features)} time-series features: {', '.join(all_new_features)}")
            
            # Fill any remaining NaN values
            X = X.fillna(X.median())
            
            # Remove infinite values
            X = X.replace([np.inf, -np.inf], np.nan)
            X = X.fillna(X.median())
            
            # Target variable
            y = df['Price'].values if 'Price' in df.columns else None
            
            # Store feature names
            self.feature_columns = X.columns.tolist()
            
            st.write(f"✅ **Feature preparation completed!**")
            st.write(f"📊 Final feature matrix: {X.shape[0]} rows × {X.shape[1]} features")
            
            return X, y, self.feature_columns
            
        except Exception as e:
            print(f"Error preparing features: {e}")
            # Return basic features if advanced feature engineering fails
            basic_cols = ['Rainfall', 'Temperature', 'Market_Demand', 'Supply_Index']
            available_basic = [col for col in basic_cols if col in df.columns]
            if available_basic:
                X = df[available_basic].fillna(df[available_basic].median())
                y = df['Price'].values if 'Price' in df.columns else None
                return X, y, available_basic
            else:
                return pd.DataFrame(), None, []
    
    def create_prediction_input(self, commodity, rainfall, temperature, market_demand, supply_index, season):
        """Create input data for prediction"""
        
        try:
            # Create input dataframe
            input_data = {
                'Commodity': [commodity],
                'Rainfall': [rainfall],
                'Temperature': [temperature],
                'Market_Demand': [market_demand],
                'Supply_Index': [supply_index],
                'Season': [season],
                'Date': [datetime.now()],
                'Price': [0]  # Placeholder
            }
            
            return pd.DataFrame(input_data)
        except Exception as e:
            print(f"Error creating prediction input: {e}")
            return pd.DataFrame()
    
    def validate_prediction_input(self, df, required_features):
        """Validate that prediction input has all required features"""
        
        try:
            processed_df = self.preprocess_data(df)
            X, _, _ = self.prepare_features(processed_df)
            
            # Ensure all required features are present
            missing_features = [f for f in required_features if f not in X.columns]
            
            if missing_features:
                # Add missing features with default values
                for feature in missing_features:
                    if 'lag' in feature.lower():
                        X[feature] = 0  # Default lag value
                    elif 'ma' in feature.lower():
                        X[feature] = 0  # Default moving average
                    elif 'volatility' in feature.lower():
                        X[feature] = 0  # Default volatility
                    else:
                        X[feature] = 0  # Default value
            
            # Reorder columns to match training features
            X = X.reindex(columns=required_features, fill_value=0)
            
            return X
        except Exception as e:
            print(f"Error validating prediction input: {e}")
            return pd.DataFrame()
    
    def get_data_quality_report(self, df):
        """Generate a comprehensive data quality report"""
        
        try:
            import streamlit as st
            
            st.write("📋 **Data Quality Report**")
            
            # Basic statistics
            total_rows = len(df)
            total_cols = len(df.columns)
            
            # Missing values analysis
            missing_summary = df.isnull().sum()
            missing_percent = (missing_summary / total_rows) * 100
            
            # Data types analysis
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            categorical_cols = df.select_dtypes(include=['object']).columns
            
            # Duplicates
            duplicate_rows = df.duplicated().sum()
            
            # Create quality report
            quality_report = {
                'Total Rows': total_rows,
                'Total Columns': total_cols,
                'Numeric Columns': len(numeric_cols),
                'Categorical Columns': len(categorical_cols),
                'Duplicate Rows': duplicate_rows,
                'Columns with Missing Values': (missing_summary > 0).sum(),
                'Total Missing Values': missing_summary.sum(),
                'Missing Value Percentage': f"{(missing_summary.sum() / (total_rows * total_cols)) * 100:.2f}%"
            }
            
            # Display report
            for key, value in quality_report.items():
                st.write(f"- **{key}**: {value}")
            
            return quality_report
            
        except Exception as e:
            print(f"Error generating data quality report: {e}")
            return {}