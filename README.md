# Mobile Game Live-Ops A/B Test Evaluator

## Use case: A comprehensive, generic A/B Test Evaluation tool that suits a wide range of live-ops AB tests.
  - evaluate experimental features
  - track multi-metric performance
  - check baseline integrity and
  - detect cross-contamination across concurrent A/B tests
  - 
## Target audience: Game data analysts, product managers, and key stakeholders 

# 1. Quick Start & Data Schema Requirements
Before uploading your transaction or activity logs, select your primary scenario and structure your dataset as a long-form player-day level CSV file.

## Scenario 1: Active Player Base (Existing Users / Calendar Trends)
Use this mode for calendar-based live events, sales, or UI overhauls targeted at existing active users.
Required Columns: Player ID, Variant Group, Login Date, plus one or more numeric metrics (e.g., Spend, Ads Watched, Sessions).
Optional Columns: Additional Variant columns for concurrent tests (e.g., Test B Group, Test C Group).

## Scenario 2: New Cohorts (Onboarding Users / Lifecycle LTV Curves)
Use this mode for onboarding changes, FTUE (First-Time User Experience) tweaks, or retention/LTV optimizations tracking players from their install date.
Required Columns: Player ID, Variant Group, Login Date, Cohort Date, and numeric metric columns.

# 2. Single-Metric Statistical Engine
When evaluating a specific metric between a baseline Control group and one or more Test Variants, the evaluator executes three distinct statistical frameworks simultaneously:

## Methods Overview
1. Two-Sample Parametric Tests (Welch's T-Test / Chi-Square):
Continuous Metrics (e.g., ARPU): Uses Welch’s t-test, which does not assume equal variances between groups.
Binary Metrics (e.g., Payer Conversion %): Uses Chi-Square test of independence / large-N Z-test.
Use Case: Best for quick significance checks when data sample sizes are large ($N > 1,000$).

2. Frequentist Bootstrapping Engine:
Resamples your dataset 1,000 times with replacement to estimate empirical confidence intervals (2.5th to 97.5th percentiles).
Use Case: Highly recommended for mobile gaming metrics (like spend), which are heavily right-skewed and violate standard normal distribution assumptions.

3. Bayesian Conjugacy Models:
Continuous Metrics: Evaluates Normal-Normal conjugacy to compute $P(\text{Variant} > \text{Control})$.
Binary Metrics: Evaluates Beta-Binomial conjugacy using uninformative Beta(1,1) priors.
Use Case: Provides clear executive answers like "There is a 98.4% probability that Variant B is superior to Control."

## Multi-Metric Summary Matrix
The Multi-Metric Summary Matrix tab summarizes performance across multiple numeric metrics simultaneously (e.g., D1 Retention, Ad Impressions, ARPU, IAP Spend).

## Analyst Workflow:
In the sidebar, select all relevant numeric metrics under "Select Metrics for Summary Matrix".
Review the output table to quickly identify guardrail metric drops (e.g., spend increased, but retention decreased).

# 4. Cross-Contamination & Multi-Test Interaction Matrix
When running 2 to 4 concurrent A/B tests on the same user cohort, feature interference can invalidate test results. The evaluator automatically detects the metric type and routes the data to the appropriate interaction model:

## How to Use the Cross-Contamination Tab
Check "Enable Factorial Cross-Contamination Test" in the sidebar.
Select 2 to 4 variant columns (e.g., Test_UI, Test_Pricing, Test_Difficulty).
Open the "🔀 Multi-Test Cross-Contamination Matrix" tab.

## Model Routing Logic
Continuous Metrics (e.g., Spend, Session Time):
Fits an Ordinary Least Squares (OLS) N-Way ANOVA model:

Decision: If any interaction term has probability < 0.05%, a cross-contamination error is flagged.

Binary Metrics (e.g., Conversion 0/1, Retention Day-1 0/1):
Fits a Binomial Logistic Regression (Logit GLM) model:

Evaluates Odds Ratios and alerts if combined feature exposure distorts conversion odds beyond individual main effects.

# 5. Best Practices Checklist for Analysts
- Winsorize Outliers First: Mobile game spend data usually contains "whales." Exercise judgement and use the Outlier Management Slider (e.g., 99th percentile capping) to keep extreme users from skewing the results.
- Run A/A Verification: In calendar-based tests, inspect the "🧪 Pre-Event Verification" tab. If $p < 0.05$ during the pre-event period, your baseline groups were imbalanced prior to launch.
- Monitor SRM (Sample Ratio Mismatch): Ensure user allocation ratios match expectations (e.g., 50/50 split). Significant Chi-Square mismatches indicate bucket assignment bugs.


