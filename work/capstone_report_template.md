# Capstone Report — CTR / Engagement Opportunity Scoring

- **Author:** Irfan Khattak 
- **Lane:** CTR / Engagement Opportunity Scoring
- **Repo:** https://github.com/Irfan-code-cloud/ML-Internship-flyrank/main/work/notebooks/capstone.ipynb
- **Date:** August 2026 

## 1. Problem framing
This analysis provides decision support for automated SEO and content triage. The unit of analysis is daily performance at the content URL grain, utilizing the `content_hash_id` and `report_date`. The output is a ranked Opportunity Gap score (Expected CTR minus Actual CTR) paired with categorical reason codes. Content teams use this output to prioritize precise interventions, such as metadata optimization for low-click pages or readability improvements for low-engagement pages. The cost of a wrong call is wasting editorial resources guessing where to spend time, or intervening on pages that are already performing optimally.

## 2. Data safety
The dataset used is the FlyRank ML Internship Warehouse (`fact_content_daily_performance/month=2026-03/*.parquet`). Exclusions include unlinked properties lacking dual GSC and GA4 tracking, and high-cardinality IDs (`client_hash_id`, `content_hash_id`) to prevent memorization. Same-day targets (unlagged search impressions, clicks, and engagement) were strictly excluded from the feature space to prevent target leakage. The sealed future month (`2026-06`) was strictly excluded to prevent out-of-time leakage. No client-identifying details appear in the repository.

## 3. Baseline
The transparent baseline is a Dummy Regressor model that predicts the mean training CTR[cite: 5]. This establishes the absolute minimum performance floor and provides an honest baseline to measure ML lift. Evaluated on the holdout test set, the naive baseline achieved a Mean Absolute Error (MAE) of 0.005888 and an R² Score of -0.00000065.

## 4. Model / analysis
The method utilizes a `HistGradientBoostingRegressor`. This model fits the lane perfectly because it is highly efficient for tabular data and handles non-linear relationships, such as the exponential decay of CTR relative to search position. The exact feature list consists of 7-day lags: `gsc_position_lag7`, `gsc_impressions_lag7`, `ga4_engagement_sec_lag7`, alongside `ai_gemini_score` and `is_weekend`. Unlagged metrics were left out on purpose. The target is the `actual_ctr` (Clicks / Impressions) observed on the target day.

## 5. Evaluation
The validation design is an 80/20 random train/test split. Because all features are strictly lagged by 7 days prior to the prediction day, temporal leakage within the same month is structurally prevented[cite: 5]. On the exact same split, the model achieved an MAE of 0.005146 and an R² Score of 0.150442, successfully demonstrating directional predictive lift over the baseline. 

## 6. Interpretation
The model captures historical statistical associations to predict expected CTR. By evaluating Expected CTR versus Actual CTR, the model identifies an Opportunity Gap. URLs with the largest positive gaps capture significantly fewer clicks than expected given their historical search position and engagement. 

## 7. Recommendation
The output supports a ranked action playbook. A FlyRank editor would use the assigned reason codes to action the top URLs tomorrow:
* **`HIGH_CTR_GAP_LOW_ENGAGEMENT`**: Improve on-page structure, readability, and hook (Gap > 0.02, engagement < 30s).
* **`METADATA_MISMATCH`**: Rewrite title tags and meta descriptions to improve SERP clickability (Gap > 0.02, engagement >= 30s).
* **`LOW_OPPORTUNITY_GAP`**: Maintain current optimization; low priority.
**Limits:** This is an observational, non-causal decision-support tool. Predicting a higher expected CTR does not guarantee that altering titles or metadata will directly cause ranking improvements. 

## 8. Reproducibility
To reproduce, execute `work/notebooks/capstone.ipynb` from top to bottom. The random seed is set to `42` for both the `train_test_split` and the `HistGradientBoostingRegressor`. The environment requires `duckdb`, `pandas`, `numpy`, `scikit-learn`, `matplotlib`, and `seaborn`. 

**Acknowledgments:** Built on the FlyRank ML Internship dataset (https://flyrank.ai).
