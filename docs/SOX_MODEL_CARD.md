# Model Risk Governance & SOX Compliance Card

**Model Name:** Enterprise FP&A Predictive Variance & Forecast Engine  
**Version:** 1.0.0  
**Model Developer:** Bhupesh Chandra Dimri  
**Model Validation Standard:** Federal Reserve SR 11-7 / OCC 2011-12 & SOX Section 404  

---

## 1. Intended Business Use
* **Primary Objective:** Deliver multi-horizon forward-looking controllable expense and cash-flow forecasts for enterprise business units across US Banking and Corporate Operations.
* **Secondary Objective:** Detect anomalous subledger postings (>25% unexplained surge) before financial month-end close.
* **Out-of-Scope:** Not authorized for standalone regulatory capital adequacy calculations (CCAR/DFAST) without supervisory oversight.

---

## 2. Model Methodology & Architecture
* **Primary Algorithm:** LightGBM Regressor utilizing Huber loss function for robust resilience against subledger outliers and one-off budget fluctuations.
* **Secondary Algorithm:** Unsupervised Isolation Forest (contamination = 0.02) evaluating multidimensional spend volume and variance ratios.
* **Exogenous Indicators:** Federal Reserve Economic Data (FRED) incorporating SOFR (Secured Overnight Financing Rate) and Consumer Price Index (CPI) trends.

---

## 3. Data Integrity & Anti-Leakage Verification
* **Temporal Partitioning:** Strict expanding-window time-series validation. Holdout test sets are strictly chronological (trailing 8 months) with zero future information available during training folds.
* **Data Provenance:** Ingestion mapped to standard Chart of Accounts (COA) general ledger subledgers (SAP ERP / NetSuite).
* **Audit Trail:** Every forecast includes full feature attribution waterfalls generated via TreeSHAP to ensure complete c-suite explainability.

---

## 4. Performance Benchmarks & Risk Thresholds
| Metric | Production Gate Threshold | Backtested Result |
|---|---|---|
| **WAPE (Volume Weighted Error)** | < 8.0% | **4.62%** |
| **MAPE (Mean Percentage Error)** | < 10.0% | **4.85%** |
| **Variance Reduction vs Baseline** | > 15.0% | **58.4% improvement** |
| **Inference Latency SLA** | < 500 ms | **< 45 ms** |
