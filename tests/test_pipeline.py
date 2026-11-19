import pytest
import pandas as pd
import numpy as np
from data.synthetic_generator import generate_fpa_data
from src.features import build_features, wape
from src.models import FPAModelPipeline

def test_data_generation_dimensions():
    df = generate_fpa_data(n_months=12, random_state=42)
    assert len(df) > 0
    assert "actual_spend" in df.columns
    assert "budget_allocated" in df.columns
    assert "sofr_rate" in df.columns

def test_zero_lookahead_leakage():
    raw_df = generate_fpa_data(n_months=24, random_state=42)
    feat_df = build_features(raw_df)
    
    # Check that lag_1m for any period equals the previous month's actual_spend
    group_cols = ['entity_id', 'cost_center', 'gl_account']
    sample_group = feat_df.groupby(group_cols).get_group(('US_Banking', 'IT_Infra', '6200_CloudCompute')).sort_values('period_date')
    
    actuals = sample_group['actual_spend'].values
    lags = sample_group['actual_lag_1m'].values
    
    # The lag at index i should equal the actual at index i-1 in the sorted time series
    for i in range(1, len(lags)):
        assert lags[i] == actuals[i-1]

def test_wape_calculation():
    y_true = np.array([100.0, 200.0, 300.0])
    y_pred = np.array([110.0, 190.0, 300.0]) # Errors: 10 + 10 + 0 = 20; Total = 600
    assert abs(wape(y_true, y_pred) - (20.0 / 600.0)) < 1e-6

def test_model_pipeline_execution():
    raw_df = generate_fpa_data(n_months=24, random_state=42)
    feat_df = build_features(raw_df)
    
    cutoff_date = feat_df['period_date'].max() - pd.DateOffset(months=4)
    train_df = feat_df[feat_df['period_date'] <= cutoff_date]
    test_df = feat_df[feat_df['period_date'] > cutoff_date]
    
    pipeline = FPAModelPipeline(n_estimators=50)
    pipeline.fit(train_df)
    metrics = pipeline.evaluate(test_df)
    
    assert "model_wape" in metrics
    assert metrics["model_wape"] < 0.15 # Must meet baseline performance
