import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

class CommodityPricePredictor:
    def __init__(self):
        self.linear_model = LinearRegression()
        self.rf_model = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=1)
        self.models_trained = False
        self.training_features = []
        
    def train_models(self, X, y):
        """Train all ML models"""
        
        try:
            if X.empty or y is None or len(X) == 0:
                raise ValueError("Invalid training data")
            
            # Store training features for later validation
            self.training_features = X.columns.tolist()
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Train Linear Regression
            self.linear_model.fit(X_train, y_train)
            
            # Train Random Forest
            self.rf_model.fit(X_train, y_train)
            
            # Evaluate models
            models_performance = {}
            
            # Linear Regression performance
            lr_pred = self.linear_model.predict(X_test)
            models_performance['Linear Regression'] = {
                'MAE': mean_absolute_error(y_test, lr_pred),
                'RMSE': np.sqrt(mean_squared_error(y_test, lr_pred)),
                'R2': r2_score(y_test, lr_pred)
            }
            
            # Random Forest performance
            rf_pred = self.rf_model.predict(X_test)
            models_performance['Random Forest'] = {
                'MAE': mean_absolute_error(y_test, rf_pred),
                'RMSE': np.sqrt(mean_squared_error(y_test, rf_pred)),
                'R2': r2_score(y_test, rf_pred)
            }
            
            self.models_trained = True
            return models_performance
        except Exception as e:
            print(f"Error training models: {e}")
            return {}
    
    def predict(self, X, model_type='Random Forest'):
        """Make predictions using specified model"""
        
        if not self.models_trained:
            raise ValueError("Models not trained yet!")
        
        try:
            # Ensure input has same features as training data
            if hasattr(X, 'columns'):
                missing_features = [f for f in self.training_features if f not in X.columns]
                if missing_features:
                    for feature in missing_features:
                        X[feature] = 0  # Add missing features with default value
                
                # Reorder columns to match training features
                X = X[self.training_features]
            
            if model_type == 'Linear Regression':
                return self.linear_model.predict(X)
            elif model_type == 'Random Forest':
                return self.rf_model.predict(X)
            else:
                return self.rf_model.predict(X)  # Default to Random Forest
        except Exception as e:
            print(f"Error making prediction: {e}")
            return np.array([0])
    
    def get_feature_importance(self):
        """Get feature importance from Random Forest model"""
        
        if not self.models_trained:
            return None
            
        try:
            return self.rf_model.feature_importances_
        except Exception as e:
            print(f"Error getting feature importance: {e}")
            return None
    
    def predict_price_trend(self, X, days_ahead=30):
        """Predict price trend for next few days"""
        
        try:
            if not self.models_trained:
                return []
            
            predictions = []
            current_X = X.copy()
            
            for day in range(days_ahead):
                # Make prediction
                pred = self.predict(current_X, model_type='Random Forest')[0]
                predictions.append(pred)
                
                # Update features for next prediction (simplified approach)
                if hasattr(current_X, 'columns') and len(current_X.columns) > 0:
                    # Update lag features if they exist
                    lag_columns = [col for col in current_X.columns if 'lag' in col.lower()]
                    for i, lag_col in enumerate(sorted(lag_columns)):
                        if i == 0:  # First lag (lag_1)
                            current_X.iloc[0, current_X.columns.get_loc(lag_col)] = pred
                        elif i < len(predictions):  # Other lags
                            lag_num = int(lag_col.split('_')[-1])
                            if len(predictions) >= lag_num:
                                current_X.iloc[0, current_X.columns.get_loc(lag_col)] = predictions[-lag_num]
                    
                    # Update moving average features if they exist
                    ma_columns = [col for col in current_X.columns if 'ma' in col.lower()]
                    for ma_col in ma_columns:
                        window = int(ma_col.split('_')[-1])
                        if len(predictions) >= window:
                            ma_value = np.mean(predictions[-window:])
                            current_X.iloc[0, current_X.columns.get_loc(ma_col)] = ma_value
                    
                    # Update volatility if it exists
                    vol_columns = [col for col in current_X.columns if 'volatility' in col.lower()]
                    for vol_col in vol_columns:
                        if len(predictions) >= 7:  # Need at least 7 days for volatility
                            vol_value = np.std(predictions[-7:])
                            current_X.iloc[0, current_X.columns.get_loc(vol_col)] = vol_value
                    
                    # Add some random variation to prevent identical predictions
                    noise_factor = 0.02  # 2% noise
                    for col in current_X.columns:
                        if col not in ['Commodity_encoded', 'Season_encoded', 'Year', 'Month', 'Day']:
                            current_val = current_X.iloc[0, current_X.columns.get_loc(col)]
                            noise = np.random.normal(0, abs(current_val) * noise_factor)
                            current_X.iloc[0, current_X.columns.get_loc(col)] = current_val + noise
            
            return predictions
        except Exception as e:
            print(f"Error predicting trend: {e}")
            return []
    
    def get_model_info(self):
        """Get information about trained models"""
        
        if not self.models_trained:
            return {"status": "Models not trained"}
        
        return {
            "status": "Models trained successfully",
            "linear_regression": {
                "type": "Linear Regression",
                "features": len(self.training_features),
                "coefficients": len(self.linear_model.coef_) if hasattr(self.linear_model, 'coef_') else 0
            },
            "random_forest": {
                "type": "Random Forest",
                "n_estimators": self.rf_model.n_estimators,
                "features": len(self.training_features),
                "max_depth": self.rf_model.max_depth
            },
            "training_features": self.training_features
        }