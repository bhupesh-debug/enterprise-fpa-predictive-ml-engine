"""
Automated Pipeline Execution & Verification Script
Runs end-to-end synthetic data generation, temporal train-test splitting, model benchmarking, and audit logging.
"""

from data.synthetic_generator import generate_fpa_data
from src.features import build_features
from src.models import FPAModelPipeline
import pandas as pd

def main():
    print("=" * 70)
    print("🚀 RUNNING ENTERPRISE FP&A PREDICTIVE ML ENGINE")
    print("=" * 70)
    
    # 1. Ingest Data
    raw_df = generate_fpa_data(n_months=48, random_state=42)
    print(f"✓ Synthetic Subledger Records Ingested: {len(raw_df):,} lines")
    
    # 2. Build Features
    feat_df = build_features(raw_df)
    print(f"✓ Feature Engineering Completed: {feat_df.shape[1]} features extracted")
    
    # 3. Out-Of-Time Temporal Split
    cutoff_date = feat_df['period_date'].max() - pd.DateOffset(months=8)
    train_df = feat_df[feat_df['period_date'] <= cutoff_date].copy()
    test_df = feat_df[feat_df['period_date'] > cutoff_date].copy()
    print(f"✓ Out-of-Time Temporal Split: Train={len(train_df)} rows | Holdout Test={len(test_df)} rows")
    
    # 4. Fit & Audit Pipeline
    pipeline = FPAModelPipeline()
    pipeline.fit(train_df)
    metrics = pipeline.evaluate(test_df)
    
    print("\n" + "-" * 40)
    print("📊 BENCHMARKING & AUDIT RESULTS")
    print("-" * 40)
    print(f"• Baseline Static Run-Rate WAPE : {metrics['baseline_wape']*100:.2f}% | MAPE: {metrics['baseline_mape']*100:.2f}%")
    print(f"• LightGBM Multi-Horizon WAPE   : {metrics['model_wape']*100:.2f}% | MAPE: {metrics['model_mape']*100:.2f}%")
    print(f"• WAPE Error Reduction          : {metrics['wape_reduction_pct']:.1f}% improvement")
    print(f"• Subledger Anomalies Flagged   : {metrics['anomalies_detected']} transactions")
    print("=" * 70)

if __name__ == "__main__":
    main()
