# Part B: Business Case Analysis
## Scenario: Promotion Effectiveness at a Fashion Retail Chain

---

## B1. Problem Formulation

### B1(a) — ML Problem Formulation

**Target Variable:** Number of items sold (sales volume) per store per month, broken down by promotion type.

**Candidate Input Features:**
- Store attributes: location type (urban/semi-urban/rural), store size, monthly footfall
- Competitive context: local competition density
- Customer demographics: age distribution, income bracket, loyalty membership rate
- Promotion type: one of five categories (Flat Discount, BOGO, Free Gift with Purchase, Category-Specific Offer, Loyalty Points Bonus)
- Calendar features: month, season, weekends count, festival flags
- Historical performance: past sales volume per promotion type at that store

**ML Problem Type:** This is a **multiclass classification** problem (recommend the best promotion out of 5) or alternatively a **regression** problem (predict items sold for each promotion, then pick the max).

The regression framing is preferable — it not only recommends the best promotion but also tells you *by how much* it outperforms others, giving the marketing team richer information.

---

### B1(b) — Why Items Sold > Revenue as Target Variable

Revenue is influenced by factors outside the model's control — price changes, discounts applied, product mix shifts, and margin differences across categories. A promotion might boost revenue by pushing high-price items even if unit volume drops, or conversely, a BOGO might inflate volume while cutting revenue per unit.

Items sold (sales volume) is a cleaner, more direct signal of how a promotion drives customer behaviour, unconfounded by pricing noise.

**Broader principle:** Target variables should measure what the business *actually wants to optimise*, and should be resistant to confounders. In real-world ML, there is often a temptation to use a readily available proxy (like revenue) instead of a more meaningful but harder-to-isolate signal. Poor target variable choice leads to a model that optimises the metric you measure, not the outcome you want.

---

### B1(c) — Alternative to a Single Global Model

Rather than one global model, use a **store-cluster or hierarchical modelling strategy**:

1. **Cluster stores** by location type, size, footfall, and demographics (e.g., k-means or expert segmentation into urban / semi-urban / rural tiers).
2. **Train one model per cluster** — stores within a cluster share promotion-response patterns, providing enough data while capturing local differences.
3. Optionally, use **mixed-effects (hierarchical) models** where store-level random effects allow individual store behaviour to deviate from the cluster baseline.

**Justification:** A single global model treats all stores identically, which is invalid when a Loyalty Points Bonus resonates with urban loyalty-card holders but does nothing in a rural store where footfall is transactional. Cluster models capture the group structure, avoiding both underfitting (one global model) and overfitting (50 separate models with insufficient data each).

---

## B2. Data and EDA Strategy

### B2(a) — Joining Tables and Dataset Grain

**Join strategy:**
- Start with `transactions` (grain: one row per transaction).
- Aggregate to **store × month** level: sum items sold, count transactions, compute average basket size.
- Left-join `store_attributes` on `store_id` — adds size, location type, footfall, demographics (static per store).
- Left-join `promotion_details` on `store_id` + `month` — adds which promotion was active and its parameters.
- Left-join `calendar` on `month` — adds weekend count, festival flags, and seasonal indicators.

**Final modelling grain:** One row = one store × one month × one promotion.

**Aggregations before modelling:**
- Total items sold per store per month (target)
- Transaction count, average basket size (features)
- Proportion of weekends and festival days in that month
- Rolling 3-month average sales per store (trend feature)

---

### B2(b) — EDA Analyses

| Analysis | What to look for | Modelling impact |
|---|---|---|
| **Promotion × items sold box plot** | Which promotions have higher median sales; variance by promotion type | Confirms target has signal; may inform class weighting |
| **Sales by store-type heatmap** | Interaction between location type and promotion type | Motivates location-type as a key feature; may reveal need for separate models |
| **Monthly seasonality line chart** | Peaks around festivals, year-end; troughs in off-season | Add month/season features; consider lag features |
| **Correlation matrix of numeric features** | Multicollinearity between footfall, store size, revenue | Feature selection; avoid including highly correlated redundant features |
| **Promotion frequency per store** | Whether each store has been exposed to all 5 promotions | Data sparsity issue — stores with few observations of a promotion type are unreliable for that class |

---

### B2(c) — Class Imbalance (80% No-Promotion Transactions)

**Problem:** A model trained on this data will be heavily biased toward predicting "no promotion" as the best option, since that label dominates. It will have poor precision/recall for actual promotion types.

**Steps to address it:**
1. **Resample:** Oversample promotion-active months (SMOTE for tabular data) or undersample no-promotion months to balance classes during training.
2. **Class weights:** Assign higher loss weight to promotion classes in the model's objective function.
3. **Separate modelling:** Train the model only on promotion-active rows to learn *which* promotion works best, treating no-promotion as a baseline benchmark separately.
4. **Evaluation:** Use F1-score or PR-AUC rather than accuracy, since accuracy will be misleadingly high due to class imbalance.

---

## B3. Model Evaluation and Deployment

### B3(a) — Train-Test Split and Evaluation Metrics

**Why random split is inappropriate:** This is time-series data. A random split would allow future data to leak into training (e.g., training on December and testing on October), artificially inflating performance. In production, the model always predicts forward in time.

**Correct approach — temporal split:**
- Train on months 1–30 (2.5 years), test on months 31–36 (last 6 months).
- Optionally use **walk-forward validation**: retrain on expanding windows and test on the next month each time, giving a robust estimate across different time periods.

**Evaluation metrics:**

| Metric | Interpretation |
|---|---|
| **Top-1 Accuracy** | How often the model's recommended promotion is actually the best-performing one. Primary business metric. |
| **Macro F1-Score** | Balanced performance across all 5 promotion types — penalises the model for ignoring minority promotions. |
| **Mean Absolute Error (if regression)** | How far off predicted items sold is from actual, in units — directly interpretable for the business. |
| **Regret / Opportunity Cost** | Difference in items sold between the recommended promotion and the true best promotion — quantifies the cost of wrong recommendations in business terms. |

---

### B3(b) — Feature Importance to Explain Seasonal Recommendation Differences

Using a tree-based model (e.g., XGBoost or Random Forest), extract **feature importance scores** or use **SHAP values** for interpretability.

**For Store 12 — December vs March:**
- Pull SHAP values for each prediction. In December, high SHAP contributions likely come from `festival_flag=1`, `month=12`, `high_footfall` — conditions where Loyalty Points Bonus works well because customers are buying more frequently and want to accumulate points for the new year.
- In March, SHAP values likely show `low_footfall`, `off-season=1` driving the recommendation — Flat Discount works better when footfall is lower and price sensitivity increases.

**Communicating to marketing team:**
> "In December, Store 12 sees festival-driven high footfall with loyal repeat customers — the model has learned from historical data that Loyalty Points Bonus boosts volume the most in these conditions. In March, footfall drops and customers are more price-sensitive, so a straightforward Flat Discount drives the highest unit sales."

Provide a visual SHAP waterfall chart for each recommendation to make this concrete and non-technical.

---

### B3(c) — End-to-End Deployment Process

**Saving the model:**
- Serialise trained model using `joblib` or `pickle`; store in a versioned model registry (e.g., MLflow, S3 bucket with version tags).
- Save the preprocessing pipeline (scalers, encoders) alongside the model artefact.

**Monthly inference pipeline (automated, no retraining):**
1. At the start of each month, a scheduled job (Airflow/cron) pulls the latest store transaction data from the data warehouse.
2. Aggregates to store × month grain and applies the same feature engineering pipeline used during training.
3. Loads the saved model and runs inference for all 50 stores.
4. Outputs a recommendation table: `store_id | recommended_promotion | predicted_items_sold`.
5. Results are pushed to a dashboard or emailed to the marketing team before month start.

**Monitoring for model degradation:**
- **Performance monitoring:** Each month, after actual sales data arrives, compute MAE / Top-1 accuracy of last month's recommendations. Alert if rolling 3-month accuracy drops >10% below training baseline.
- **Data drift detection:** Monitor input feature distributions (e.g., footfall, demographics) using PSI (Population Stability Index). If distributions shift significantly, the model's assumptions may be invalid.
- **Trigger for retraining:** If performance drops below a defined threshold OR data drift is detected in >3 key features, trigger a retrain job using the updated full history.
- **Retraining cadence:** Even without degradation signals, schedule a full retrain every 6 months to incorporate new data.
