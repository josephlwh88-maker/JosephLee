import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols, logit
import matplotlib.pyplot as plt
from itertools import combinations

st.set_page_config(page_title="Mobile Game Live-Ops A/B Evaluator", layout="wide")

st.title("🎯 Mobile Game Live-Ops A/B Test Evaluator")
st.write("Select your analysis scenario and verify your CSV format against the schema template below.")

# Scenario Selector
scenario = st.radio(
    "Select your A/B Test Scenario:",
    ("Active Player Base (Existing Users, Calendar Trends)", 
     "New Cohorts (Onboarding Users, Lifecycle LTV Curves)"),
    horizontal=True
)

# Required CSV Data Schema Guide
with st.expander("📋 Required CSV Data Schema Guide", expanded=True):
    st.write("Ensure your uploaded CSV contains a long-form (player-day) format matching this layout:")
    
    if scenario == "Active Player Base (Existing Users, Calendar Trends)":
        schema_data = {
            "Expected Column Concept": ["Player ID", "Test A Group", "Test B Group", "Test C Group (Optional)", "Test D Group (Optional)", "Activity Date", "Primary Metric", "Secondary Metric"],
            "Sample Cell Value": ["player_101", "Control_A", "Variant_B1", "Control_C", "Variant_D2", "2026-06-01", "4.99", "1"],
            "Format Rule": ["Unique Text/Integer", "Text flag", "Text flag", "Text flag", "Text flag", "Date string (YYYY-MM-DD)", "Numeric Float/Integer", "Binary (0 or 1) or Integer"]
        }
    else:
        schema_data = {
            "Expected Column Concept": ["Player ID", "Test A Group", "Test B Group", "Test C Group (Optional)", "Test D Group (Optional)", "Activity Date", "Signup Date", "Primary Metric", "Secondary Metric"],
            "Sample Cell Value": ["player_901", "Variant_A1", "Control_B", "Variant_C1", "Control_D", "2026-06-16", "2026-06-15", "1.99", "0"],
            "Format Rule": ["Unique Text/Integer", "Text flag", "Text flag", "Text flag", "Text flag", "Date string (YYYY-MM-DD)", "Date string (YYYY-MM-DD)", "Numeric Float/Integer", "Binary (0 or 1) or Integer"]
        }
        
    schema_df = pd.DataFrame(schema_data)
    st.table(schema_df)

# File Upload Component
uploaded_file = st.file_uploader("Upload your transaction/activity CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("File uploaded successfully!")
    
    with st.expander("👀 Preview Uploaded Data"):
        st.dataframe(df.head(10))
        
    # Sidebar Configuration & Column Mapping
    st.sidebar.header("⚙️ Column Mapping")
    col_player = st.sidebar.selectbox("Player ID Column", df.columns)
    col_variant = st.sidebar.selectbox("Primary Test A Variant Column", df.columns)
    
    # --- MULTI-TEST CONCURRENT SELECTION SYSTEM (UP TO 4 TESTS) ---
    enable_factorial = st.sidebar.checkbox("Enable Factorial Cross-Contamination Test")
    factorial_cols = []
    if enable_factorial:
        available_test_cols = [c for c in df.columns if c not in [col_player]]
        factorial_cols = st.sidebar.multiselect(
            "Select Concurrent Test Columns (2 to 4 Tests)",
            options=available_test_cols,
            default=[col_variant] if col_variant in available_test_cols else []
        )
        if len(factorial_cols) < 2:
            st.sidebar.warning("Select at least 2 test columns for cross-contamination analysis.")
        elif len(factorial_cols) > 4:
            st.sidebar.error("Maximum 4 concurrent test columns allowed.")
            factorial_cols = factorial_cols[:4]

    col_date = st.sidebar.selectbox("Date / Event Date Column", df.columns)
    
    # Primary Group Selection System
    st.sidebar.header("👥 Primary Test A Group Assignment")
    unique_groups = sorted(df[col_variant].dropna().unique().tolist())
    
    if len(unique_groups) >= 2:
        ctrl_grp = st.sidebar.selectbox("Select Control Group (Baseline)", unique_groups, index=0)
        remaining_groups = [g for g in unique_groups if g != ctrl_grp]
        
        var_grps = st.sidebar.multiselect(
            "Select Test/Variant Groups (Max 3)", 
            remaining_groups, 
            default=[remaining_groups[0]] if remaining_groups else []
        )
        
        if len(var_grps) > 3:
            st.sidebar.error("Please select a maximum of 3 test variant groups.")
            var_grps = var_grps[:3]
    else:
        st.sidebar.error("Error: Less than 2 unique groups found in Primary Test Variant column.")
        ctrl_grp, var_grps = None, []
    
    # Target Metric Toggle System
    st.sidebar.header("📊 Metric Configuration")
    col_rev = st.sidebar.selectbox("Select Primary Evaluation Metric", df.columns, index=min(3, len(df.columns)-1))
    col_sec = st.sidebar.selectbox("Select Secondary Benchmark Metric", df.columns, index=min(4, len(df.columns)-1))
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    default_selected = list(set([c for c in [col_rev, col_sec] if c in numeric_cols]))
    matrix_metrics = st.sidebar.multiselect("Select Metrics for Summary Matrix", options=numeric_cols, default=default_selected)
    
    cohort_mode = "Cumulative"
    if scenario == "New Cohorts (Onboarding Users, Lifecycle LTV Curves)":
        col_signup = st.sidebar.selectbox("Signup / Install Date Column", df.columns)
        st.sidebar.header("📈 Cohort Accumulation Mode")
        cohort_mode = st.sidebar.radio(
            "Metric Calculation Mode:",
            ("Overall Lifetime (D0 to D14 Total)",
             "Cumulative (Running Total up to Day-N)")
        )
    
    st.sidebar.header("🐳 Outlier Management")
    cap_percentile = st.sidebar.slider("Winsorize (Cap) Metrics at Percentile:", 95.0, 100.0, 99.0, 0.01)

    # Clean datetime formats
    df[col_date] = pd.to_datetime(df[col_date])
    if scenario == "New Cohorts (Onboarding Users, Lifecycle LTV Curves)":
        df[col_signup] = pd.to_datetime(df[col_signup])

    # Method 1: Two-Sample Parametric Tests Engine
    def parametric_ttest(ctrl_data, var_data):
        ctrl_array = ctrl_data.to_numpy()
        var_array = var_data.to_numpy()
        if len(ctrl_array) <= 1 or len(var_array) <= 1:
            return 0.0, 0.0, 1.0
        t_stat, p_val = stats.ttest_ind(var_array, ctrl_array, equal_var=False)
        mean_diff = np.mean(var_array) - np.mean(ctrl_array)
        se_diff = np.sqrt((np.var(var_array, ddof=1) / len(var_array)) + (np.var(ctrl_array, ddof=1) / len(ctrl_array)))
        
        v_ctrl = np.var(ctrl_array, ddof=1) / len(ctrl_array)
        v_var = np.var(var_array, ddof=1) / len(var_array)
        df_welch = ((v_ctrl + v_var) ** 2) / ((v_ctrl ** 2 / (len(ctrl_array) - 1)) + (v_var ** 2 / (len(var_array) - 1)))
        
        t_crit = stats.t.ppf(0.975, df=df_welch)
        low = mean_diff - (t_crit * se_diff)
        high = mean_diff + (t_crit * se_diff)
        return low, high, p_val

    # Method 2: Frequentist Bootstrap Engine
    def bootstrap_ci(ctrl_data, var_data, n_resamples=1000):
        diffs = []
        ctrl_array = ctrl_data.to_numpy()
        var_array = var_data.to_numpy()
        if len(ctrl_array) == 0 or len(var_array) == 0:
            return 0.0, 0.0, 1.0
        for _ in range(n_resamples):
            boot_ctrl = np.random.choice(ctrl_array, size=len(ctrl_array), replace=True)
            boot_var = np.random.choice(var_array, size=len(var_array), replace=True)
            diffs.append(np.mean(boot_var) - np.mean(boot_ctrl))
        low = np.percentile(diffs, 2.5)
        high = np.percentile(diffs, 97.5)
        p_val = np.mean(np.array(diffs) <= 0) if np.mean(var_array) > np.mean(ctrl_array) else np.mean(np.array(diffs) >= 0)
        return low, high, min(p_val * 2, 1.0)

    # Method 3: Bayesian Continuous Normal-Normal Test
    def bayesian_continuous_test(ctrl_data, var_data, n_samples=30000):
        n_ctrl, mean_ctrl, std_ctrl = len(ctrl_data), ctrl_data.mean(), ctrl_data.std(ddof=1)
        n_var, mean_var, std_var = len(var_data), var_data.mean(), var_data.std(ddof=1)
        if n_ctrl == 0 or n_var == 0 or np.isnan(std_ctrl) or np.isnan(std_var) or std_ctrl == 0 or std_var == 0:
            return 0.5
        se_ctrl = std_ctrl / np.sqrt(n_ctrl)
        se_var = std_var / np.sqrt(n_var)
        return np.mean(np.random.normal(mean_var, se_var, n_samples) > np.random.normal(mean_ctrl, se_ctrl, n_samples))

    # Helper to generate Automated Callout Recommendation
    def display_recommendation_callout(ctrl_data, var_data, var_name, metric_name, min_sample_threshold=1000):
        n_ctrl = len(ctrl_data)
        n_var = len(var_data)
        
        if n_ctrl < min_sample_threshold or n_var < min_sample_threshold:
            st.warning(
                f"⚠️ **INSUFFICIENT DATA / TEST INCONCLUSIVE ({var_name} vs Baseline)**\n\n"
                f"* **Sample Size Status:** Current sample ($N_{{ctrl}} = {n_ctrl:,}$, $N_{{var}} = {n_var:,}$) "
                f"is below the recommended minimum threshold of {min_sample_threshold:,} users per variant.\n"
                f"* **Action:** **Keep test running.** Calling the test now risks high Type II error (false negatives)."
            )
            return

        mean_ctrl, mean_var = ctrl_data.mean(), var_data.mean()
        lift = ((mean_var - mean_ctrl) / mean_ctrl * 100) if mean_ctrl != 0 else 0
        _, _, t_p = parametric_ttest(ctrl_data, var_data)
        prob_better = bayesian_continuous_test(ctrl_data, var_data)
        
        if t_p < 0.05 and lift > 0 and prob_better >= 0.95:
            st.success(
                f"🚀 **RECOMMENDATION: SHIP VARIANT `{var_name}`**\n\n"
                f"* **Statistically Significant Positive Impact:** '{metric_name}' showed a **{lift:+.2f}% Lift** ($p = {t_p:.4f}$, $P(\\text{{Var}} > \\text{{Ctrl}}) = {prob_better*100:.1f}\\%$).\n"
                f"* **Sample Integrity:** Adequate sample size ($N_{{ctrl}} = {n_ctrl:,}$, $N_{{var}} = {n_var:,}$)."
            )
        elif t_p < 0.05 and lift < 0 and prob_better <= 0.05:
            st.error(
                f"🛑 **RECOMMENDATION: DO NOT SHIP VARIANT `{var_name}`**\n\n"
                f"* **Statistically Significant Negative Impact:** '{metric_name}' suffered a **{lift:+.2f}% Drop** ($p = {t_p:.4f}$, $P(\\text{{Var}} > \\text{{Ctrl}}) = {prob_better*100:.1f}\\%$).\n"
                f"* **Sample Integrity:** Adequate sample size ($N_{{ctrl}} = {n_ctrl:,}$, $N_{{var}} = {n_var:,}$)."
            )
        else:
            st.info(
                f"⚖️ **RECOMMENDATION: NO STATISTICALLY SIGNIFICANT DIFFERENCE (`{var_name}`)**\n\n"
                f"* **Inconclusive Impact:** Observed lift of **{lift:+.2f}%** on '{metric_name}' is not statistically significant ($p = {t_p:.4f}$).\n"
                f"* **Action:** Feature has neutral impact. Decide roll-out based on secondary guardrails or operational costs."
            )

    # Helper to evaluate conversion/proportion tests safely
    def display_proportion_results(ctrl_conv, total_ctrl, var_conv, total_var, ctrl_grp, var_grp):
        if total_ctrl == 0 or total_var == 0:
            st.warning(f"⚠️ Insufficient sample data to compute proportion test for {var_grp} vs {ctrl_grp}.")
            return

        conv_rate_ctrl = (ctrl_conv / total_ctrl) * 100
        conv_rate_var = (var_conv / total_var) * 100
        lift_conv = ((conv_rate_var - conv_rate_ctrl) / conv_rate_ctrl) * 100 if conv_rate_ctrl != 0 else 0
        
        table = [[ctrl_conv, total_ctrl - ctrl_conv], [var_conv, total_var - var_conv]]
        ctrl_no = total_ctrl - ctrl_conv
        var_no = total_var - var_conv
        
        if ctrl_conv == 0 or var_conv == 0 or ctrl_no == 0 or var_no == 0:
            _, t_prop_p = stats.fisher_exact(table)
            test_label = "Fisher's Exact Test (Zero-Freq Fallback)"
        else:
            try:
                _, t_prop_p, _, _ = stats.chi2_contingency(table)
                test_label = "Chi-Square Test"
            except ValueError:
                _, t_prop_p = stats.fisher_exact(table)
                test_label = "Fisher's Exact Test"

        ctrl_bools = np.zeros(total_ctrl); ctrl_bools[:ctrl_conv] = 1
        var_bools = np.zeros(total_var); var_bools[:var_conv] = 1
        boot_low, boot_high, boot_p = bootstrap_ci(pd.Series(ctrl_bools), pd.Series(var_bools))
        
        prior_alpha, prior_beta = 1, 1
        ctrl_post = np.random.beta(prior_alpha + ctrl_conv, prior_beta + (total_ctrl - ctrl_conv), 30000)
        var_post = np.random.beta(prior_alpha + var_conv, prior_beta + (total_var - var_conv), 30000)
        prob_better = np.mean(var_post > ctrl_post)

        st.metric(label=f"Conversion Rate ({var_grp} vs {ctrl_grp})", value=f"{conv_rate_var:.2f}% vs {conv_rate_ctrl:.2f}%", delta=f"{lift_conv:+.2f}% Lift")
        
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("🌐 **1) Proportion Test**")
            if t_prop_p < 0.05: st.success(f"Significant (p={t_prop_p:.4f})")
            else: st.error(f"NS (p={t_prop_p:.4f})")
            st.caption(test_label)
        with c2:
            st.markdown("🥾 **2) Bootstrapping**")
            if boot_p < 0.05: st.success(f"Significant (p={boot_p:.4f})")
            else: st.error(f"NS (p={boot_p:.4f})")
        with c3:
            st.markdown("🔮 **3) Bayesian**")
            if prob_better >= 0.95: st.success(f"P(Var>Ctrl): {prob_better*100:.1f}% 🔥")
            elif prob_better <= 0.05: st.error(f"P(Var>Ctrl): {prob_better*100:.1f}% ❌")
            else: st.warning(f"P(Var>Ctrl): {prob_better*100:.1f}%")

    # Helper to process continuous evaluations
    def display_continuous_results(ctrl_data, var_data, ctrl_grp, var_grp):
        mean_ctrl, mean_var = ctrl_data.mean(), var_data.mean()
        lift = ((mean_var - mean_ctrl) / mean_ctrl) * 100 if mean_ctrl != 0 else 0
        
        t_low, t_high, t_p = parametric_ttest(ctrl_data, var_data)
        b_low, b_high, b_p = bootstrap_ci(ctrl_data, var_data)
        prob_better = bayesian_continuous_test(ctrl_data, var_data)

        st.metric(label=f"Per-User Avg ({var_grp} vs {ctrl_grp})", value=f"{mean_var:.3f} vs {mean_ctrl:.3f}", delta=f"{lift:+.2f}% Lift")
        
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("🌐 **1) Two-Sample T-Test**")
            if t_p < 0.05: st.success(f"Significant (p={t_p:.4f})")
            else: st.error(f"NS (p={t_p:.4f})")
        with c2:
            st.markdown("🥾 **2) Bootstrapping**")
            if b_p < 0.05: st.success(f"Significant (p={b_p:.4f})")
            else: st.error(f"NS (p={b_p:.4f})")
        with c3:
            st.markdown("🔮 **3) Bayesian**")
            if prob_better >= 0.95: st.success(f"P(Var>Ctrl): {prob_better*100:.1f}% 🔥")
            elif prob_better <= 0.05: st.error(f"P(Var>Ctrl): {prob_better*100:.1f}% ❌")
            else: st.warning(f"P(Var>Ctrl): {prob_better*100:.1f}%")

    # Helper function to generate Multi-Metric Summary Matrix
    def generate_summary_matrix(data_df, group_col, player_col, ctrl_group, var_groups, selected_metrics):
        if not selected_metrics or not var_groups: return pd.DataFrame()
        matrix_rows = []
        user_agg = data_df.groupby([player_col, group_col])[selected_metrics].sum().reset_index()
        ctrl_users = user_agg[user_agg[group_col] == ctrl_group]
        
        for metric in selected_metrics:
            c_series = ctrl_users[metric]
            ctrl_mean = c_series.mean()
            for v in var_groups:
                v_series = user_agg[user_agg[group_col] == v][metric]
                if v_series.empty or c_series.empty: continue
                var_mean = v_series.mean()
                lift = ((var_mean - ctrl_mean) / ctrl_mean * 100) if ctrl_mean != 0 else 0
                _, _, t_p = parametric_ttest(c_series, v_series)
                
                matrix_rows.append({
                    "Metric": metric,
                    "Variant": v,
                    "Baseline": ctrl_group,
                    "Baseline Mean": round(ctrl_mean, 4),
                    "Variant Mean": round(var_mean, 4),
                    "Lift (%)": f"{lift:+.2f}%",
                    "p-Value": f"{t_p:.4f}",
                    "Significant": "✅ Yes" if t_p < 0.05 else "❌ No"
                })
        return pd.DataFrame(matrix_rows)

    # Multi-Test Factorial Matrix Engine
    def render_multi_factorial_matrix(data_df, player_col, test_cols, metric_col):
        st.subheader(f"🧪 Multi-Factorial Interaction Matrix for '{metric_col}' ({len(test_cols)} Concurrent Tests)")
        
        raw_unique = set(data_df[metric_col].dropna().unique())
        is_binary = raw_unique.issubset({0, 1, 0.0, 1.0})
        
        if is_binary:
            p_agg = data_df.groupby([player_col] + test_cols)[metric_col].max().reset_index()
        else:
            p_agg = data_df.groupby([player_col] + test_cols)[metric_col].sum().reset_index()
        
        clean_cols = [f"test_{i}" for i in range(len(test_cols))]
        col_rename_map = dict(zip(test_cols, clean_cols))
        p_agg_clean = p_agg.rename(columns=col_rename_map)
        
        if is_binary:
            matrix_summary = p_agg.groupby(test_cols).agg(
                User_Count=(player_col, 'count'),
                Conversions=(metric_col, 'sum'),
                Conversion_Rate=(metric_col, 'mean')
            ).reset_index()
            matrix_summary['Conversion_Rate'] = (matrix_summary['Conversion_Rate'] * 100).round(2).astype(str) + "%"
        else:
            matrix_summary = p_agg.groupby(test_cols).agg(
                User_Count=(player_col, 'count'),
                Mean_Metric=(metric_col, 'mean'),
                Std_Dev=(metric_col, 'std')
            ).reset_index()
        
        st.markdown(f"#### 📌 Multi-Test Bucket Combination Breakdown: {metric_col}")
        st.dataframe(matrix_summary, use_container_width=True)
        
        main_effects = [f"C({c})" for c in clean_cols]
        interaction_effects = [f"C({c1}):C({c2})" for c1, c2 in combinations(clean_cols, 2)]
        formula = f"{metric_col} ~ " + " + ".join(main_effects + interaction_effects)

        if is_binary:
            st.markdown(f"#### 🔬 Logistic Regression Interaction Assessment (Binary Metric): {metric_col}")
            st.caption("Modeled via Binomial GLM / Logistic Regression to evaluate Odds Ratio interaction effects.")
            try:
                cell_counts = p_agg_clean.groupby(clean_cols)[metric_col].agg(['count', 'sum'])
                has_separation = (cell_counts['sum'] == 0).any() or (cell_counts['sum'] == cell_counts['count']).any()
                
                if has_separation:
                    st.warning(
                        f"⚡ **Complete Separation Detected:** One or more cell combinations contain 0% or 100% conversion for `{metric_col}`. "
                        f"Applying **L2 Regularized Penalized Regression** to penalize infinite log-odds estimates."
                    )
                    logit_model = logit(formula, data=p_agg_clean)
                    model = logit_model.fit_regularized(method='l1', alpha=0.1, disp=False)
                    
                    summary_df = pd.DataFrame({
                        "Coefficient": model.params,
                        "Odds Ratio": np.exp(model.params)
                    })
                else:
                    model = sm.GLM.from_formula(formula, data=p_agg_clean, family=sm.families.Binomial()).fit()
                    
                    summary_df = pd.DataFrame({
                        "Coefficient": model.params,
                        "Odds Ratio": np.exp(model.params),
                        "Std Error": model.bse,
                        "z-Statistic": model.tvalues,
                        "p-Value": model.pvalues
                    })
                
                new_idx = []
                for idx in summary_df.index:
                    lbl = str(idx)
                    for orig, clean in col_rename_map.items():
                        lbl = lbl.replace(f"C({clean})", f"C({orig})")
                    new_idx.append(lbl)
                summary_df.index = new_idx
                
                st.dataframe(summary_df, use_container_width=True)
                
                if "p-Value" in summary_df.columns:
                    interaction_rows = [i for i in summary_df.index if ":" in i]
                    flagged = [i for i in interaction_rows if summary_df.loc[i, "p-Value"] < 0.05]
                    
                    if flagged:
                        st.error(f"⚠️ **CROSS-CONTAMINATION DETECTED FOR {metric_col.upper()}!** Significant logistic interaction terms found in: **{', '.join(flagged)}**.")
                    else:
                        st.success(f"✅ **NO CROSS-CONTAMINATION DETECTED FOR {metric_col.upper()}.** All interaction term p-values are $\ge 0.05$.")
            except np.linalg.LinAlgError:
                st.error(
                    f"❌ **Singular Matrix Error for '{metric_col}'**: The selected test columns are redundant or perfectly collinear. "
                    f"Check if one test group completely depends on another or if a variant bucket has no assigned users."
                )
            except Exception as e:
                st.warning(f"Could not calculate Logistic Regression model for {metric_col}: {e}")
        else:
            st.markdown(f"#### 🔬 N-Way ANOVA Interaction Assessment (Continuous Metric): {metric_col}")
            st.caption("Modeled via Ordinary Least Squares (OLS) N-Way ANOVA to evaluate mean variance interaction terms.")
            try:
                model = ols(formula, data=p_agg_clean).fit()
                anova_table = sm.stats.anova_lm(model, typ=2)
                
                display_anova = anova_table.copy()
                new_idx = []
                for idx in display_anova.index:
                    lbl = str(idx)
                    for orig, clean in col_rename_map.items():
                        lbl = lbl.replace(f"C({clean})", f"C({orig})")
                    new_idx.append(lbl)
                display_anova.index = new_idx
                
                st.dataframe(display_anova, use_container_width=True)
                
                interaction_rows = [i for i in display_anova.index if ":" in i]
                flagged = [i for i in interaction_rows if display_anova.loc[i, "PR(>F)"] < 0.05]
                
                if flagged:
                    st.error(f"⚠️ **CROSS-CONTAMINATION DETECTED FOR {metric_col.upper()}!** Significant ANOVA interaction terms found in: **{', '.join(flagged)}**.")
                else:
                    st.success(f"✅ **NO CROSS-CONTAMINATION DETECTED FOR {metric_col.upper()}.** All pairwise interaction term p-values are $\ge 0.05$.")
            except Exception as e:
                st.warning(f"Could not calculate multi-way ANOVA model for {metric_col}: {e}")

    # Guardrails: Active analysis pipeline requires valid groups
    all_active_groups = [ctrl_grp] + var_grps if ctrl_grp and var_grps else []
    
    if len(all_active_groups) >= 2:
        # =========================================================================
        # WORKFLOW 1: ACTIVE PLAYER BASE
        # =========================================================================
        if scenario == "Active Player Base (Existing Users, Calendar Trends)":
            st.header("⏳ Calendar Timeline Selection")
            unique_dates = sorted(df[col_date].unique())
            col_t1, col_t2 = st.columns(2)
            with col_t1: pre_event_range = st.date_input("Pre-Event Baseline Range", [unique_dates[0], unique_dates[min(6, len(unique_dates)-1)]])
            with col_t2: event_range = st.date_input("Live-Event Range", [unique_dates[min(7, len(unique_dates)-1)], unique_dates[min(20, len(unique_dates)-1)]])
                
            if len(pre_event_range) == 2 and len(event_range) == 2:
                df_pre = df[df[col_variant].isin(all_active_groups) & (df[col_date] >= pd.Timestamp(pre_event_range[0])) & (df[col_date] <= pd.Timestamp(pre_event_range[1]))]
                df_event = df[df[col_variant].isin(all_active_groups) & (df[col_date] >= pd.Timestamp(event_range[0])) & (df[col_date] <= pd.Timestamp(event_range[1]))]
                
                tab_list = ["🚀 Live-Event Impact Analysis", "📊 Multi-Metric Summary Matrix", "🧪 Pre-Event Verification (A/A Check)"]
                if enable_factorial and len(factorial_cols) >= 2:
                    tab_list.append("🔀 Multi-Test Cross-Contamination Matrix")
                    
                tabs = st.tabs(tab_list)
                
                # Tab 1: Live Event Impact
                with tabs[0]:
                    player_event = df_event.groupby([col_player, col_variant]).agg(
                        primary_raw=(col_rev, 'sum'), 
                        sec_raw=(col_sec, 'sum')
                    ).reset_index()
                    
                    if cap_percentile < 100.0:
                        p_cap = np.percentile(player_event['primary_raw'], cap_percentile) if not player_event.empty else 0
                        s_cap = np.percentile(player_event['sec_raw'], cap_percentile) if not player_event.empty else 0
                        player_event['primary_capped'] = np.where(player_event['primary_raw'] > p_cap, p_cap, player_event['primary_raw'])
                        player_event['sec_capped'] = np.where(player_event['sec_raw'] > s_cap, s_cap, player_event['sec_raw'])
                    else:
                        player_event['primary_capped'] = player_event['primary_raw']
                        player_event['sec_capped'] = player_event['sec_raw']

                    player_event['is_conv_p'] = np.where(player_event['primary_raw'] > 0, 1, 0)
                    player_event['is_conv_s'] = np.where(player_event['sec_raw'] > 0, 1, 0)
                    
                    ctrl_df = player_event[player_event[col_variant] == ctrl_grp]
                    for v in var_grps:
                        if v in player_event[col_variant].values:
                            var_df = player_event[player_event[col_variant] == v]
                            st.markdown(f"### 📦 Evaluation Summary: `{v}` vs. `{ctrl_grp}` (Baseline)")
                            
                            display_recommendation_callout(ctrl_df['primary_capped'], var_df['primary_capped'], v, col_rev)
                            
                            exp_p = st.expander(f"🔑 Primary Metric Row: {col_rev}", expanded=True)
                            with exp_p:
                                display_continuous_results(ctrl_df['primary_capped'], var_df['primary_capped'], ctrl_grp, v)
                                st.markdown("---")
                                display_proportion_results(int(ctrl_df['is_conv_p'].sum()), len(ctrl_df), int(var_df['is_conv_p'].sum()), len(var_df), ctrl_grp, v)
                            
                            exp_s = st.expander(f"🛡️ Guardrail Secondary Metric Row: {col_sec}", expanded=False)
                            with exp_s:
                                display_continuous_results(ctrl_df['sec_capped'], var_df['sec_capped'], ctrl_grp, v)
                                st.markdown("---")
                                display_proportion_results(int(ctrl_df['is_conv_s'].sum()), len(ctrl_df), int(var_df['is_conv_s'].sum()), len(var_df), ctrl_grp, v)

                # Tab 2: Summary Matrix
                with tabs[1]:
                    if matrix_metrics:
                        summary_matrix_df = generate_summary_matrix(df_event, col_variant, col_player, ctrl_grp, var_grps, matrix_metrics)
                        st.dataframe(summary_matrix_df, use_container_width=True)
                    else:
                        st.info("Select metrics in sidebar for matrix calculation.")

                # Tab 3: Pre-event Verification
                with tabs[2]:
                    player_pre = df_pre.groupby([col_player, col_variant])[col_rev].sum().reset_index()
                    if not player_pre.empty and ctrl_grp in player_pre[col_variant].values:
                        c_pre = player_pre[player_pre[col_variant] == ctrl_grp][col_rev]
                        for v in var_grps:
                            if v in player_pre[col_variant].values:
                                v_pre = player_pre[player_pre[col_variant] == v][col_rev]
                                _, _, p_val = parametric_ttest(c_pre, v_pre)
                                if p_val < 0.05: st.error(f"⚠️ Imbalance in {v}! (p={p_val:.3f})")
                                else: st.success(f"✅ Baseline balanced with {v} (p={p_val:.3f})")

                # Tab 4: Multi-Test Factorial Matrix
                if enable_factorial and len(factorial_cols) >= 2:
                    with tabs[3]:
                        render_multi_factorial_matrix(df_event, col_player, factorial_cols, col_rev)

        # =========================================================================
        # WORKFLOW 2: NEW USER COHORTS (WITH RUNNING TOTAL UP TO DAY-N TOGGLE)
        # =========================================================================
        else:
            # Calculate player age in days relative to install date
            df['player_age'] = (df[col_date] - df[col_signup]).dt.days
            
            # Filter cohort for valid days (0 to 14)
            valid_cohort_df = df[(df['player_age'] >= 0) & (df['player_age'] <= 14) & df[col_variant].isin(all_active_groups)]
            
            if cohort_mode == "Cumulative (Running Total up to Day-N)":
                target_day_n = st.slider("Select Running Total Target Day (Day-N):", 0, 14, 7)
                cohort_df = valid_cohort_df[valid_cohort_df['player_age'] <= target_day_n]
                eval_title_suffix = f"Running Total up to Day-{target_day_n}"
            else:
                cohort_df = valid_cohort_df
                eval_title_suffix = "Overall Lifetime Cohort (D0 - D14)"
            
            tab_list = ["📊 Lifecycle Milestone Results", "📊 Multi-Metric Summary Matrix"]
            if enable_factorial and len(factorial_cols) >= 2:
                tab_list.append("🔀 Multi-Test Cross-Contamination Matrix")
                
            tabs = st.tabs(tab_list)
            
            with tabs[0]:
                raw_vals = set(cohort_df[col_rev].dropna().unique())
                is_binary = raw_vals.issubset({0, 1, 0.0, 1.0})
                
                # Aggregate at player level
                if is_binary:
                    snap = cohort_df.groupby([col_player, col_variant])[col_rev].max().reset_index()
                else:
                    snap = cohort_df.groupby([col_player, col_variant])[col_rev].sum().reset_index()
                    
                ctrl_df = snap[snap[col_variant] == ctrl_grp]
                
                for v in var_grps:
                    if v in snap[col_variant].values:
                        var_df = snap[snap[col_variant] == v]
                        st.markdown(f"### 📦 Cohort Results ({eval_title_suffix}): `{v}` vs. `{ctrl_grp}`")
                        
                        display_recommendation_callout(ctrl_df[col_rev], var_df[col_rev], v, col_rev)
                        
                        if is_binary:
                            display_proportion_results(
                                int(ctrl_df[col_rev].sum()), len(ctrl_df),
                                int(var_df[col_rev].sum()), len(var_df),
                                ctrl_grp, v
                            )
                        else:
                            display_continuous_results(ctrl_df[col_rev], var_df[col_rev], ctrl_grp, v)

            with tabs[1]:
                if matrix_metrics:
                    summary_matrix_df = generate_summary_matrix(cohort_df, col_variant, col_player, ctrl_grp, var_grps, matrix_metrics)
                    st.dataframe(summary_matrix_df, use_container_width=True)

            if enable_factorial and len(factorial_cols) >= 2:
                with tabs[2]:
                    render_multi_factorial_matrix(cohort_df, col_player, factorial_cols, col_rev)
    else:
        st.info("💡 Select a baseline Control and at least 1 Test Variant Group in the sidebar.")
else:
    st.info("💡 Waiting for a CSV file upload to run mapping diagnostics.")
