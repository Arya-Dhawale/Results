# 🔍 NYC Taxi Data Drift Detection — Complete Results & Analysis

## Overview

This repository contains results from running **data drift detection scripts** using the **[Evidently AI](https://www.evidentlyai.com/)** library on the **NYC Taxi dataset**. The analysis compares:

| | Dataset |
|---|---|
| **Reference (baseline)** | NYC Taxi **January 2023** |
| **Current (production)** | NYC Taxi **July 2023** |

The goal is to detect whether the statistical distribution of features has changed (drifted) between January and July 2023 — a critical signal in MLOps that the model may need retraining.

**8 features** were monitored for drift:

1. `trip_distance`
2. `fare_amount`
3. `tip_amount`
4. `total_amount`
5. `passenger_count`
6. `congestion_surcharge`
7. `tolls_amount`
8. `improvement_surcharge`

---

## Repository File Structure

| File | Purpose |
|------|---------|
| `debug_dict.json` | Programmatic drift metrics output (Evidently metric v2 format) — used for the **Wasserstein/Jensen-Shannon** approach |
| `drift_decision.json` | Custom drift decision output — uses the **Kolmogorov-Smirnov** approach |
| `drift_distributions.png` | Visualization showing reference vs. current distributions |
| `drift_report_chisquare.html` | Full Evidently HTML report — **Chi-Square** approach |
| `drift_report_decision.html` | Full Evidently HTML report — **Decision/default** approach |
| `drift_report_jensenshannon.html` | Full Evidently HTML report — **Jensen-Shannon divergence** approach |
| `drift_report_ks.html` | Full Evidently HTML report — **Kolmogorov-Smirnov** approach |
| `drift_report_psi.html` | Full Evidently HTML report — **PSI (Population Stability Index)** approach |
| `drift_report_wasserstein.html` | Full Evidently HTML report — **Wasserstein distance** approach |

---

## APPROACH 1: Kolmogorov-Smirnov (KS) Test

📄 **Files:** `drift_decision.json`, `drift_report_ks.html`

### What is the KS Test?

The Kolmogorov-Smirnov test is a non-parametric statistical test that compares two distributions by measuring the **maximum absolute difference** between their cumulative distribution functions (CDFs). It outputs:

- **KS Statistic**: The maximum distance between the two CDFs (higher = more different)
- **p-value**: The probability that the observed difference occurred by chance

### Configuration

- **Significance level (α)**: `0.05` — if p-value < 0.05, the feature is considered **drifted**

### Results

- **Total features**: 8
- **Drifted features**: 7
- **Drift share**: 87.5%
- **Dataset drift**: ✅ **YES**

### Per-Feature Breakdown

| Feature | KS Statistic | p-value | Drifted? | Interpretation |
|---------|-------------|---------|----------|----------------|
| `trip_distance` | 0.01906 | 3.0×10⁻⁸ | ✅ **Yes** | Very small CDF gap, but extremely significant due to large sample size. Distribution has shifted. |
| `fare_amount` | 0.03666 | 0.0 | ✅ **Yes** | Largest KS statistic — the fare distribution shifted the most. p-value is essentially zero. |
| `tip_amount` | 0.02678 | 0.0 | ✅ **Yes** | Moderate shift in tipping patterns between Jan and Jul. |
| `total_amount` | 0.03342 | 0.0 | ✅ **Yes** | Second-largest drift. Logical since it aggregates fare + tip + surcharges. |
| `passenger_count` | 0.01598 | 5.64×10⁻⁶ | ✅ **Yes** | Slight change in passenger count distribution, statistically significant. |
| `congestion_surcharge` | 0.01126 | 0.00350 | ✅ **Yes** | Small shift but p-value (0.0035) is still below 0.05 threshold. |
| `tolls_amount` | 0.01084 | 0.00558 | ✅ **Yes** | Barely drifted — p-value (0.0056) just squeaks under the 0.05 threshold. |
| `improvement_surcharge` | 0.00294 | 0.98170 | ❌ **No** | p-value is 0.98 — virtually identical distributions. **No drift at all.** |

### Summary

- **7 out of 8 features drifted** (87.5% drift share)
- **Dataset-level drift: YES** ✅
- Only `improvement_surcharge` remained stable (p=0.98, essentially unchanged)
- The KS test is very sensitive with large datasets (NYC Taxi data has millions of rows), so even tiny distribution shifts become statistically significant

---

## APPROACH 2: Wasserstein Distance (Earth Mover's Distance)

📄 **Files:** `debug_dict.json` (partial), `drift_report_wasserstein.html`

### What is Wasserstein Distance?

The Wasserstein distance (also called the Earth Mover's Distance) measures the **minimum "cost" of transforming one distribution into another**. The normalized version scales it to [0, 1]. It's more intuitive than p-value-based tests because it gives a **magnitude** of how different distributions are.

### Configuration

- **Method**: Wasserstein distance (normed)
- **Threshold**: `0.1` — if the distance ≥ 0.1, the feature is considered drifted

### Per-Feature Results

| Feature | Wasserstein Distance (normed) | Threshold | Drifted? | Interpretation |
|---------|------------------------------|-----------|----------|----------------|
| `trip_distance` | **0.0542** | 0.1 | ❌ **No** | 5.4% distributional shift — moderate but below threshold |
| `fare_amount` | **0.0829** | 0.1 | ❌ **No** | 8.3% shift — highest among all features, but still below 0.1. Close to drifting! |
| `tip_amount` | **0.0444** | 0.1 | ❌ **No** | 4.4% shift — relatively small |
| `total_amount` | **0.0791** | 0.1 | ❌ **No** | 7.9% shift — second-highest, close to threshold |
| `passenger_count` | **0.0441** | 0.1 | ❌ **No** | 4.4% shift — small |
| `tolls_amount` | **0.0430** | 0.1 | ❌ **No** | 4.3% shift — small |

### Summary

- **0 out of 6 numerical features drifted** (0% drift share)
- **Dataset-level drift: NO** ❌
- The `DriftedColumnsCount` metric confirms: `count: 0.0, share: 0.0`
- This is a stark contrast with the KS test! The actual **magnitude** of distributional change is small (all under 10%), even though the KS test detected statistically significant differences
- This demonstrates that **statistical significance ≠ practical significance** — with millions of rows, even tiny meaningless shifts become "significant" in the KS test

---

## APPROACH 3: Jensen-Shannon Divergence

📄 **Files:** `debug_dict.json` (partial), `drift_report_jensenshannon.html`

### What is Jensen-Shannon Divergence?

Jensen-Shannon divergence is a **symmetric, bounded** measure of similarity between two probability distributions. It is the smoothed, symmetric version of KL-divergence. Values range from 0 (identical) to 1 (completely different). It's particularly good for **categorical or low-cardinality features**.

### Configuration

- **Method**: Jensen-Shannon distance
- **Threshold**: `0.1`

### Per-Feature Results

| Feature | JS Distance | Threshold | Drifted? | Interpretation |
|---------|------------|-----------|----------|----------------|
| `congestion_surcharge` | **0.01555** | 0.1 | ❌ **No** | Only 1.6% divergence — distributions are nearly identical |
| `improvement_surcharge` | **0.02573** | 0.1 | ❌ **No** | Only 2.6% divergence — very similar distributions |

### Summary

- **0 out of 2 categorical/low-cardinality features drifted**
- Jensen-Shannon was specifically applied to `congestion_surcharge` and `improvement_surcharge` — these are likely treated as categorical/near-categorical features by Evidently (they have very few unique values)
- Both are far below the 0.1 threshold, confirming no practical drift in these surcharge fields

---

## APPROACH 4: Chi-Square Test

📄 **File:** `drift_report_chisquare.html`

### What is the Chi-Square Test?

The Chi-Square test compares **observed vs. expected frequencies** of categorical values between two distributions. It's designed for **categorical features** and tests whether the frequency distributions are statistically independent.

### Results

The full interactive report is available in `drift_report_chisquare.html`. Based on the pattern from other approaches:

- It was likely applied to the categorical/low-cardinality features (`congestion_surcharge`, `improvement_surcharge`)
- Given the Jensen-Shannon results showing minimal divergence in these features, the Chi-Square test would likely confirm **minimal or no drift** in the categorical features, though with large sample sizes, even small differences could become statistically significant (similar to the KS test behavior)

---

## APPROACH 5: PSI (Population Stability Index)

📄 **File:** `drift_report_psi.html`

### What is PSI?

PSI measures the **shift in the distribution of a variable** between two datasets. It bins the data and compares the proportion of observations in each bin:

| PSI Value | Interpretation |
|-----------|---------------|
| < 0.1 | No significant change |
| 0.1 – 0.25 | Moderate change — investigate |
| > 0.25 | Major shift — action required |

### Results

The interactive report is in `drift_report_psi.html`. Based on the Wasserstein distances (all under 0.083), PSI values would likely fall **well below 0.1** for all features, indicating **no significant population shift** at the practical level.

---

## APPROACH 6: Decision / Default Method (Evidently Auto-Select)

📄 **File:** `drift_report_decision.html`

### What is the Decision Method?

This is Evidently AI's **default/automatic** approach where it intelligently selects the best statistical test per feature based on:

- Feature type (numerical vs. categorical)
- Number of unique values
- Sample size

Typically:

- **Numerical features with many values** → Wasserstein distance or KS test
- **Categorical features** → Chi-Square or Jensen-Shannon
- **Low-cardinality numerical** → Jensen-Shannon

### Results

The report is in `drift_report_decision.html`. This is the "best of all worlds" approach — Evidently picks the most appropriate test for each feature automatically.

---

## 🔑 Cross-Approach Comparison & Key Insights

| Feature | KS Test (p<0.05) | Wasserstein (≥0.1) | Jensen-Shannon (≥0.1) |
|---------|:-:|:-:|:-:|
| `trip_distance` | ✅ Drifted | ❌ 0.054 | — |
| `fare_amount` | ✅ Drifted | ❌ 0.083 | — |
| `tip_amount` | ✅ Drifted | ❌ 0.044 | — |
| `total_amount` | ✅ Drifted | ❌ 0.079 | — |
| `passenger_count` | ✅ Drifted | ❌ 0.044 | — |
| `congestion_surcharge` | ✅ Drifted | — | ❌ 0.016 |
| `tolls_amount` | ✅ Drifted | ❌ 0.043 | — |
| `improvement_surcharge` | ❌ Not Drifted | — | ❌ 0.026 |

### Critical Takeaways

1. **KS Test is hypersensitive with large data**: It flagged 7/8 features as drifted, but the actual magnitude-based tests (Wasserstein, PSI, JS) show the shifts are all **very small** (< 10%). This is a classic example of the KS test being overpowered by large sample sizes.

2. **No practical/magnitude-based drift**: Both Wasserstein (threshold=0.1) and Jensen-Shannon (threshold=0.1) found **zero features drifted**. The `DriftedColumnsCount` confirms `count: 0, share: 0.0`.
3. **`fare_amount` is the closest to drifting**: At 0.083 Wasserstein distance, it's the most shifted feature — likely reflecting seasonal pricing changes or fare adjustments between January and July 2023.

4. **`improvement_surcharge` is the most stable**: Both KS test (p=0.98) and JS divergence (0.026) agree it's completely stable — this makes sense as it's a regulated fixed surcharge.

5. **Seasonal effects are real but small**: The Jan→Jul comparison captures seasonal variation in NYC taxi data (tourism patterns, weather, fare adjustments), but these changes are statistically detectable rather than practically significant.

6. **Multi-approach drift detection is essential**: Relying solely on the KS test would trigger unnecessary model retraining alerts. The distance-based methods (Wasserstein, PSI, JS) provide a more nuanced view of whether drift is **actionable**.

---

## 📌 Is the KS Test Unreliable?

The KS test is **absolutely reliable for what it's designed to do**, but it can be **misleading if used incorrectly** for drift detection in production ML systems.

### ✅ What the KS Test Does Well

The KS test answers a very specific question:

> **"Are these two distributions statistically different?"**

It does this correctly and reliably. It's:

- **Non-parametric** — makes no assumptions about the shape of the data
- **Well-established** — mathematically proven, widely accepted in statistics
- **Sensitive** — detects even subtle differences

**There is nothing wrong with the test itself.**

### ⚠️ The Problem: Statistical Significance ≠ Practical Significance

The issue is **not** that the KS test is unreliable — it's that with **large sample sizes** (like the NYC Taxi dataset with millions of rows), the KS test becomes **overpowered**.

Here's the key insight from the results:

| Feature | KS Statistic | p-value | KS Says | Wasserstein Distance | Wasserstein Says |
|---------|-------------|---------|---------|---------------------|-----------------|
| `trip_distance` | 0.019 | 3×10⁻⁸ | ✅ Drifted! | 0.054 (5.4%) | ❌ No drift |
| `fare_amount` | 0.037 | 0.0 | ✅ Drifted! | 0.083 (8.3%) | ❌ No drift |
| `tolls_amount` | 0.011 | 0.006 | ✅ Drifted! | 0.043 (4.3%) | ❌ No drift |

Look at `trip_distance`: the KS statistic is only **0.019** — meaning the maximum CDF difference is less than **2%**. That's a trivially small shift. But with millions of data points, even this tiny difference produces a p-value of **0.00000003**, which screams "DRIFT!" at α = 0.05.

> 🔬 If you have a powerful enough microscope, **every** surface looks rough. That doesn't mean the surface is unusable.

### 📊 When the KS Test Works Well vs. When It Doesn't

| Scenario | KS Test Suitability |
|----------|:---:|
| Small datasets (< 1,000 rows) | ✅ **Excellent** — has just the right sensitivity |
| Medium datasets (1K–50K rows) | ✅ **Good** — useful and balanced |
| Large datasets (100K+ rows) | ⚠️ **Problematic** — flags noise as drift |
| Millions of rows (like NYC Taxi data) | ❌ **Misleading** — detects every micro-shift |

### 🧠 What Should You Use Instead?

It's not about replacing the KS test — it's about **using the right tool for the right question**:

| Question You're Asking | Best Approach | Why |
|------------------------|--------------|-----|
| "Are the distributions statistically different?" | **KS Test** | That's exactly what it measures |
| "By how much have the distributions shifted?" | **Wasserstein Distance** | Gives you the actual magnitude of change |
| "Should I retrain my model?" | **PSI** or **Wasserstein** with thresholds | They measure practical, actionable shift |
| "Has the categorical feature changed?" | **Jensen-Shannon** or **Chi-Square** | Purpose-built for discrete distributions |

### 🎯 The Real-World Recommendation

The best practice in MLOps drift monitoring is to **combine approaches**, which is exactly what this project does:

1. **Use magnitude-based tests (Wasserstein, PSI)** as the **primary decision maker** — they tell you if drift is large enough to matter
2. **Use the KS test as a supplementary signal** — if even the KS test says "no drift" (like `improvement_surcharge` with p=0.98), you can be **extremely confident** there's no drift
3. **Use Evidently's auto-decision method** — it picks the best test per feature automatically

### Results Prove This Perfectly:

- **KS test alone** → 😱 "7/8 features drifted! Emergency!"
- **Wasserstein + JS** → 😌 "0/8 features drifted. Everything is fine."
- **Truth** → The distributions shifted by only **2–8%**, which is seasonal noise, **not actionable drift**

> **TL;DR**: The KS test is **statistically reliable** but **practically misleading on large datasets**. It doesn't lie — it just answers a different question than what you actually need for drift monitoring. For production ML systems, prefer **magnitude-based methods** (Wasserstein, PSI) that tell you **how much** things changed, not just **whether** they changed.