"""
Multi-Tier Modeling Pipeline: LightGBM Multi-Horizon Forecaster & Isolation Forest Anomaly Engine
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.ensemble import IsolationForest
from sklearn.metrics import mean_absolute_percentage_error
from .features import wape

class FPAModelPipeline:
    def __init__(self, n_estimators=300, learning_rate=0.03, contamination=0.02, random_state=42):
        self.random_state = random_state
        self.feature_cols = [
            'entity_id', 'cost_center', 'gl_account', 'budget_allocated', 'actual_spend',
            'headcount', 'sofr_rate', 'cpi_index', 'treasury_spread_10_2', 'variance_pct',
            'spend_per_head', 'quarter_end_compression_index', 'actual_lag_1m', 'actual_lag_2m',
            'actual_lag_3m', 'rolling_mean_3m', 'rolling_var_3m'
        ]
        self.target_col = 'target_next_spend'
        
        self.forecaster = lgb.LGBMRegressor(
            objective='huber',
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            num_leaves=31,
            random_state=random_state,
            verbosity=-1
        )
        
        self.anomaly_detector = IsolationForest(
            contamination=contamination,
            random_state=random_state
        )
        
        self.is_fitted = False

    def fit(self, df_train: pd.DataFrame):
        X_train = df_train[self.feature_cols]
        y_train = df_train[self.target_col]
        
        # Fit LightGBM Regressor
        self.forecaster.fit(X_train, y_train)
        
        # Fit Anomaly Detector
        self.anomaly_detector.fit(df_train[['actual_spend', 'variance_pct', 'spend_per_head']])
        self.is_fitted = True
        return self

    def evaluate(self, df_test: pd.DataFrame) -> dict:
        if not self.is_fitted:
            raise ValueError("Model pipeline must be fitted prior to evaluation.")
            
        X_test = df_test[self.feature_cols]
        y_test = df_test[self.target_col]
        
        # Baseline (Last month run-rate)
        baseline_preds = X_test['actual_spend']
        base_wape = wape(y_test, baseline_preds)
        base_mape = mean_absolute_percentage_error(y_test, baseline_preds)
        
        # LightGBM Predictions
        ml_preds = self.forecaster.predict(X_test)
        ml_wape = wape(y_test, ml_preds)
        ml_mape = mean_absolute_percentage_error(y_test, ml_preds)
        
        # Anomaly scoring
        anomaly_scores = self.anomaly_detector.predict(df_test[['actual_spend', 'variance_pct', 'spend_per_head']])
        n_anomalies = int(np.sum(anomaly_scores == -1))
        
        return {
            "baseline_wape": base_wape,
            "baseline_mape": base_mape,
            "model_wape": ml_wape,
            "model_mape": ml_mape,
            "wape_reduction_pct": ((base_wape - ml_wape) / base_wape) * 100,
            "anomalies_detected": n_anomalies
        }

    def predict(self, df_input: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model pipeline must be fitted prior to inference.")
        return self.forecaster.predict(df_input[self.feature_cols])
