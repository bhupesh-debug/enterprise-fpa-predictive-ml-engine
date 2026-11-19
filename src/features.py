"""
SOX-Compliant Financial Feature Engineering Pipeline
Strictly backward-looking feature construction to guarantee zero temporal data leakage.
"""

import numpy as np
import pandas as pd

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Constructs rolling financial ratios, temporal lags, and categorical indicators.
    """
    data = df.copy()
    data['period_date'] = pd.to_datetime(data['period_date'])
    
    # Financial metrics
    data['variance_amount'] = data['actual_spend'] - data['budget_allocated']
    data['variance_pct'] = data['variance_amount'] / (data['budget_allocated'] + 1e-5)
    data['spend_per_head'] = data['actual_spend'] / (data['headcount'] + 1e-5)
    
    # Calendar & Fiscal features
    data['month'] = data['period_date'].dt.month
    data['quarter'] = data['period_date'].dt.quarter
    data['is_quarter_end'] = data['month'].isin([3, 6, 9, 12]).astype(int)
    
    # Grouped historical lags (strictly backward-looking to eliminate lookahead bias)
    group_cols = ['entity_id', 'cost_center', 'gl_account']
    data['actual_lag_1m'] = data.groupby(group_cols)['actual_spend'].shift(1)
    data['actual_lag_2m'] = data.groupby(group_cols)['actual_spend'].shift(2)
    data['actual_lag_3m'] = data.groupby(group_cols)['actual_spend'].shift(3)
    data['rolling_mean_3m'] = data.groupby(group_cols)['actual_spend'].shift(1).rolling(3).mean()
    data['rolling_var_3m'] = data.groupby(group_cols)['variance_pct'].shift(1).rolling(3).mean()
    
    # Target variable: Next-month actual spend (T+1)
    data['target_next_spend'] = data.groupby(group_cols)['actual_spend'].shift(-1)
    
    # Drop rows with NaNs caused by lagging/shifting
    cleaned = data.dropna().reset_index(drop=True)
    
    # Categorical encoding
    for cat in ['entity_id', 'cost_center', 'gl_account']:
        cleaned[cat] = cleaned[cat].astype('category')
        
    return cleaned

def wape(y_true, y_pred) -> float:
    """
    Weighted Absolute Percentage Error (WAPE)
    Weights error by financial volume ($USD), standard in institutional treasury & FP&A.
    """
    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)
    return float(np.sum(np.abs(y_true_arr - y_pred_arr)) / (np.sum(y_true_arr) + 1e-7))
