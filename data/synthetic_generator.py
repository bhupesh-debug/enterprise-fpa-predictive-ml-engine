"""
Synthetic Multi-Entity Financial Subledger Data Generator
Simulates SAP / NetSuite general ledger records across enterprise business units,
incorporating macroeconomic drivers (SOFR, CPI) and real-world variance anomalies.
"""

import numpy as np
import pandas as pd
from datetime import datetime

def generate_fpa_data(n_months: int = 48, random_state: int = 42) -> pd.DataFrame:
    """
    Generates a realistic multi-entity monthly FP&A dataset with macro indicators.
    """
    np.random.seed(random_state)
    dates = pd.date_range(start="2021-01-01", periods=n_months, freq="MS")
    entities = ["US_Banking", "Enterprise_Ops", "India_Sub"]
    cost_centers = ["IT_Infra", "Sales_GTM", "Operations", "Treasury"]
    gl_accounts = ["6100_VendorConsult", "6200_CloudCompute", "6300_Travel", "6400_Headcount"]
    
    rows = []
    for d in dates:
        # Macro exogenous factors (SOFR rate hike regime and CPI trend)
        sofr = 0.5 + (4.8 / (1 + np.exp(-(d.year - 2022) * 2))) + np.random.normal(0, 0.1)
        cpi = 260 + (d.year - 2021) * 12 + d.month * 0.8 + np.random.normal(0, 0.5)
        
        for e in entities:
            for cc in cost_centers:
                for gl in gl_accounts:
                    base_spend = {
                        "6100_VendorConsult": 150000,
                        "6200_CloudCompute": 350000,
                        "6300_Travel": 80000,
                        "6400_Headcount": 750000
                    }[gl] * (1.5 if e == "US_Banking" else 0.8)
                    
                    headcount = np.random.randint(40, 300)
                    # Seasonality & Quarter-End compression
                    seasonal_factor = 1.15 if d.month in [3, 6, 9, 12] else 0.98
                    macro_shock = 1.0 + (sofr * 0.015 if gl in ["6100_VendorConsult", "6200_CloudCompute"] else 0.0)
                    
                    budget = base_spend * seasonal_factor * (1 + np.random.normal(0, 0.03))
                    actual = budget * macro_shock * (1 + np.random.normal(0.01, 0.05))
                    
                    # Inject controlled leakage anomalies in 2% of records (simulating unapproved spend)
                    if np.random.rand() < 0.02:
                        actual *= np.random.uniform(1.3, 1.8)
                        
                    rows.append({
                        "period_date": d,
                        "entity_id": e,
                        "cost_center": cc,
                        "gl_account": gl,
                        "budget_allocated": round(budget, 2),
                        "actual_spend": round(actual, 2),
                        "headcount": headcount,
                        "sofr_rate": round(sofr, 2),
                        "cpi_index": round(cpi, 2)
                    })
                    
    df = pd.DataFrame(rows)
    df.sort_values(by=["entity_id", "cost_center", "gl_account", "period_date"], inplace=True)
    return df.reset_index(drop=True)

if __name__ == "__main__":
    df = generate_fpa_data(48)
    print(f"Generated {len(df)} financial ledger records across {df['period_date'].nunique()} periods.")
    print(df.head())
