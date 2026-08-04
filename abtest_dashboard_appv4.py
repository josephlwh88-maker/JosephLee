import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

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
            "Expected Column Concept": ["Player ID", "Variant Group", "Activity Date", "Primary Metric (e.g. Spend)", "Secondary Metric (e.g. Ads)"],
            "Sample Cell Value": ["player_101", "Control", "2026-06-01", "4.99", "12"],
            "Format Rule": ["Unique Text or Integer", "Text flag (Control, Variant)", "Date string (YYYY-MM-DD)", "Numeric Float or Integer", "Numeric Float or Integer"]
        }
    else:
        schema_data = {
            "Expected Column Concept": ["Player ID", "Variant Group", "Activity Date", "Signup/Install Date", "Primary Metric (e.g. Spend)", "Secondary Metric (e.g. Ads)"],
            "Sample Cell Value": ["player_901", "Variant_B", "2026-06-16", "2026-06-15", "1.99", "5"],
            "Format Rule": ["Unique Text or Integer", "Text flag (Control, Variant)", "Date string (YYYY-MM-DD)", "Date string (YYYY-MM-DD)", "Numeric Float or Integer", "Numeric Float or Integer"]
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
    col_variant = st.sidebar.selectbox("Variant / Group Column", df.columns)
    col_date = st.sidebar.selectbox("Date / Event Date Column", df.columns)
    
    # --- MULTI-GROUP SELECTION SYSTEM ---
    st.sidebar.header("👥 Group Assignment (2-4 Groups)")
    unique_groups = sorted(df[col_variant].dropna().unique().tolist())
    
    if len(unique_groups) >= 2:
        ctrl_grp = st.sidebar.selectbox("Select Control Group (Baseline)", unique_groups, index=0)
        remaining_groups = [g for g in unique_groups if g != ctrl_grp]
        
        # Allow selecting 1 to 3 test variants
        var_grps = st.sidebar.multiselect(
            "Select Test/Variant Groups (Max 3)", 
            remaining_groups, 
            default=[remaining_groups[0]] if remaining_groups else []
        )
        
        if len(var_grps) > 3:
            st.sidebar.error("Please select a maximum of 3 test variant groups.")
            var_grps = var_grps[:3]
    else:
        st.sidebar.error("Error: Less than 2 unique groups found in the Variant column.")
        ctrl_grp, var_grps = None, []
    
    # Target Metric Toggle System
    st.sidebar.header("📊 Metric Configuration")
    col_rev = st.sidebar.selectbox(
        "Select Primary Evaluation Metric", 
        df.columns,
        index=min(3, len(df.columns)-1)
    )
    col_sec = st.sidebar.selectbox(
        "Select Secondary Benchmark Metric", 
        df.columns,
        index=min(4, len(df.columns)-1)
    )
    
    # Multi-Metric Selection for Summary Matrix
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    default_selected = list(set([c for c in [col_rev, col_sec] if c in numeric_cols]))
    matrix_metrics = st.sidebar.multiselect(
        "Select Metrics for Summary Matrix",
        options=numeric_cols,
        default=default_selected
    )
    
    cohort_mode = "Cumulative"
    if scenario == "New Cohorts (Onboarding Users, Lifecycle LTV Curves)":
        col_signup = st.sidebar.selectbox("Signup / Install Date Column", df.columns)
        st.sidebar.header("📈 Cohort Tracking Mode")
        cohort_mode = st.sidebar.radio(
            "Metric Accumulation Mode:",
            ("Non-Cumulative (Classic Retention, Exact Day-N Performance)",
             "Cumulative (LTV Curves, Running Total up to Day-N)")
        )
    
    st.sidebar.header("🐳 Outlier Management")
    cap_percentile = st.sidebar.slider("Winsorize (Cap) Metrics at Percentile:", 90.0, 100.0, 99.0, 0.5)

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

    # Bayesian Beta-Binomial Conversion Test
    def bayesian_conversion_test(ctrl_conv, ctrl_total, var_conv, var_total, n_samples=30000):
        prior_alpha, prior_beta = 1, 1
        ctrl_posterior = np.random.beta(prior_alpha + ctrl_conv, prior_beta + (ctrl_total - ctrl_conv), n_samples)
        var_posterior = np.random.beta(prior_alpha + var_conv, prior_beta + (var_total - var_conv), n_samples)
        return np.mean(var_posterior > ctrl_posterior)

    # Helper to evaluate conversion/proportion tests using 3 methods
    def display_proportion_results(ctrl_conv, total_ctrl, var_conv, total_var, ctrl_grp, var_grp):
        conv_rate_ctrl = (ctrl_conv / total_ctrl) * 100 if total_ctrl > 0 else 0
        conv_rate_var = (var_conv / total_var) * 100 if total_var > 0 else 0
        lift_conv = ((conv_rate_var - conv_rate_ctrl) / conv_rate_ctrl) * 100 if conv_rate_ctrl != 0 else 0
        
        chi2, t_prop_p, _, _ = stats.chi2_contingency([[ctrl_conv, total_ctrl - ctrl_conv], [var_conv, total_var - var_conv]])
        
        ctrl_bools = np.zeros(total_ctrl); ctrl_bools[:ctrl_conv] = 1
        var_bools = np.zeros(total_var); var_bools[:var_conv] = 1
        boot_low, boot_high, boot_p = bootstrap_ci(pd.Series(ctrl_bools), pd.Series(var_bools))
        
        prob_better = bayesian_conversion_test(ctrl_conv, total_ctrl, var_conv, total_var)
        
        p_ctrl, p_var = ctrl_conv / total_ctrl if total_ctrl > 0 else 0, var_conv / total_var if total_var > 0 else 0
        sd_ctrl, sd_var = np.sqrt(p_ctrl * (1 - p_ctrl)), np.sqrt(p_var * (1 - p_var))

        st.metric(label=f"Conversion Rate ({var_grp} vs {ctrl_grp})", value=f"{conv_rate_var:.2f}% vs {conv_rate_ctrl:.2f}%", delta=f"{lift_conv:+.2f}% Lift")
        st.markdown(f"**Metadata:** $N_{{ctrl}}$={total_ctrl:,}, $N_{{var}}$={total_var:,} | $SD_{{ctrl}}$={sd_ctrl:.4f}, $SD_{{var}}$={sd_var:.4f}")
        
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("🌐 **1) Two-Sample Test**")
            if t_prop_p < 0.05: st.success(f"Significant (p={t_prop_p:.4f})")
            else: st.error(f"NS (p={t_prop_p:.4f})")
            st.caption("Chi-Square / Large-N Z-test alternative")
        with c2:
            st.markdown("🥾 **2) Bootstrapping**")
            if boot_p < 0.05: st.success(f"Significant (p={boot_p:.4f})")
            else: st.error(f"NS (p={boot_p:.4f})")
            st.caption(f"95% CI: [{boot_low*100:+.2f}%, {boot_high*100:+.2f}%]")
        with c3:
            st.markdown("🔮 **3) Bayesian**")
            if prob_better >= 0.95: st.success(f"P(Var>Ctrl): {prob_better*100:.1f}% 🔥")
            elif prob_better <= 0.05: st.error(f"P(Var>Ctrl): {prob_better*100:.1f}% ❌")
            else: st.warning(f"P(Var>Ctrl): {prob_better*100:.1f}%")
            st.caption("Beta-Binomial Conjugacy Model")

    # Helper to process continuous evaluations using 3 distinct calculation frameworks
    def display_continuous_results(ctrl_data, var_data, ctrl_grp, var_grp):
        mean_ctrl, mean_var = ctrl_data.mean(), var_data.mean()
        lift = ((mean_var - mean_ctrl) / mean_ctrl) * 100 if mean_ctrl != 0 else 0
        
        t_low, t_high, t_p = parametric_ttest(ctrl_data, var_data)
        b_low, b_high, b_p = bootstrap_ci(ctrl_data, var_data)
        prob_better = bayesian_continuous_test(ctrl_data, var_data)
        
        n_ctrl, n_var = len(ctrl_data), len(var_data)
        sd_ctrl = ctrl_data.std(ddof=1) if n_ctrl > 1 else 0.0
        sd_var = var_data.std(ddof=1) if n_var > 1 else 0.0

        st.metric(label=f"Per-User Avg ({var_grp} vs {ctrl_grp})", value=f"{mean_var:.3f} vs {mean_ctrl:.3f}", delta=f"{lift:+.2f}% Lift")
        st.markdown(f"**Metadata:** $N_{{ctrl}}$={n_ctrl:,}, $N_{{var}}$={n_var:,} | $SD_{{ctrl}}$={sd_ctrl:.4f}, $SD_{{var}}$={sd_var:.4f}")
        
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("🌐 **1) Two-Sample T-Test**")
            if t_p < 0.05: st.success(f"Significant (p={t_p:.4f})")
            else: st.error(f"NS (p={t_p:.4f})")
            st.caption(f"Welch's 95% CI: [{t_low:.3f}, {t_high:.3f}]")
        with c2:
            st.markdown("🥾 **2) Bootstrapping**")
            if b_p < 0.05: st.success(f"Significant (p={b_p:.4f})")
            else: st.error(f"NS (p={b_p:.4f})")
            st.caption(f"Resample 95% CI: [{b_low:.3f}, {b_high:.3f}]")
        with c3:
            st.markdown("🔮 **3) Bayesian**")
            if prob_better >= 0.95: st.success(f"P(Var>Ctrl): {prob_better*100:.1f}% 🔥")
            elif prob_better <= 0.05: st.error(f"P(Var>Ctrl): {prob_better*100:.1f}% ❌")
            else: st.warning(f"P(Var>Ctrl): {prob_better*100:.1f}%")
            st.caption("Normal-Normal Conjugacy Model")

    # Helper function to generate the Multi-Metric Summary Matrix
    def generate_summary_matrix(data_df, group_col, player_col, ctrl_group, var_groups, selected_metrics):
        if not selected_metrics or not var_groups:
            return pd.DataFrame()
        
        matrix_rows = []
        user_agg = data_df.groupby([player_col, group_col])[selected_metrics].sum().reset_index()
        ctrl_users = user_agg[user_agg[group_col] == ctrl_group]
        
        for metric in selected_metrics:
            c_series = ctrl_users[metric]
            ctrl_mean = c_series.mean()
            
            for v in var_groups:
                v_series = user_agg[user_agg[group_col] == v][metric]
                if v_series.empty or c_series.empty:
                    continue
                
                var_mean = v_series.mean()
                lift = ((var_mean - ctrl_mean) / ctrl_mean * 100) if ctrl_mean != 0 else 0
                _, _, t_p = parametric_ttest(c_series, v_series)
                p_val_str = f"{t_p:.4f}"
                sig_flag = "✅ Yes" if t_p < 0.05 else "❌ No"
                
                matrix_rows.append({
                    "Metric": metric,
                    "Variant": v,
                    "Baseline Group": ctrl_group,
                    "Baseline Mean": round(ctrl_mean, 4),
                    "Variant Mean": round(var_mean, 4),
                    "Lift (%)": f"{lift:+.2f}%",
                    "p-Value": p_val_str,
                    "Statistically Significant": sig_flag
                })
                
        return pd.DataFrame(matrix_rows)

    # Guardrails: Active analysis pipeline requires valid groups select matching checkboxes
    all_active_groups = [ctrl_grp] + var_grps if ctrl_grp and var_grps else []
    
    if len(all_active_groups) >= 2:
        # =========================================================================
        # WORKFLOW 1: ACTIVE PLAYER BASE
        # =========================================================================
        if scenario == "Active Player Base (Existing Users, Calendar Trends)":
            st.header("⏳ Calendar Timeline Selection")
            unique_dates = sorted(df[col_date].unique())
            col_t1, col_t2 = st.columns(2)
            with col_t1: pre_event_range = st.date_input("Pre-Event Baseline Range (A/A Check)", [unique_dates[0], unique_dates[min(6, len(unique_dates)-1)]])
            with col_t2: event_range = st.date_input("Live-Event Range (2 Weeks)", [unique_dates[min(7, len(unique_dates)-1)], unique_dates[min(20, len(unique_dates)-1)]])
                
            if len(pre_event_range) == 2 and len(event_range) == 2:
                df_pre = df[df[col_variant].isin(all_active_groups) & (df[col_date] >= pd.Timestamp(pre_event_range[0])) & (df[col_date] <= pd.Timestamp(pre_event_range[1]))]
                df_event = df[df[col_variant].isin(all_active_groups) & (df[col_date] >= pd.Timestamp(event_range[0])) & (df[col_date] <= pd.Timestamp(event_range[1]))]
                
                def aggregate_player_calendar(data_slice):
                    if data_slice.empty: return pd.DataFrame()
                    player_df = data_slice.groupby([col_player, col_variant]).agg(primary_raw=(col_rev, 'sum'), sec_raw=(col_sec, 'sum')).reset_index()
                    if cap_percentile < 100.0:
                        p_cap = np.percentile(player_df['primary_raw'], cap_percentile) if not player_df.empty else 0
                        s_cap = np.percentile(player_df['sec_raw'], cap_percentile) if not player_df.empty else 0
                        player_df['primary_capped'] = np.where(player_df['primary_raw'] > p_cap, p_cap, player_df['primary_raw'])
                        player_df['sec_capped'] = np.where(player_df['sec_raw'] > s_cap, s_cap, player_df['sec_raw'])
                    else:
                        player_df['primary_capped'] = player_df['primary_raw']
                        player_df['sec_capped'] = player_df['sec_raw']
                    player_df['is_conv_p'] = np.where(player_df['primary_raw'] > 0, 1, 0)
                    player_df['is_conv_s'] = np.where(player_df['sec_raw'] > 0, 1, 0)
                    player_df['arpdau_capped'] = player_df['primary_capped']
                    return player_df

                player_pre = aggregate_player_calendar(df_pre)
                player_event = aggregate_player_calendar(df_event)
                
                # --- MULTI-METRIC SUMMARY MATRIX SECTION ---
                st.header("📊 Multi-Metric Summary Matrix")
                if matrix_metrics:
                    summary_matrix_df = generate_summary_matrix(
                        df_event, col_variant, col_player, ctrl_grp, var_grps, matrix_metrics
                    )
                    if not summary_matrix_df.empty:
                        st.dataframe(summary_matrix_df, use_container_width=True)
                    else:
                        st.info("No data available to construct the summary matrix.")
                else:
                    st.info("Please select metrics in the sidebar under 'Select Metrics for Summary Matrix'.")
                
                st.header("📊 Multi-Variant Parallel Evaluation Results")
                tab1, tab2 = st.tabs(["🚀 Live-Event Impact Analysis", "🧪 Pre-Event Verification (A/A Check)"])
                
                with tab2:
                    st.subheader("Pre-Event Baseline Check")
                    if not player_pre.empty and ctrl_grp in player_pre[col_variant].values:
                        c_pre = player_pre[player_pre[col_variant] == ctrl_grp]['primary_capped']
                        for v in var_grps:
                            if v in player_pre[col_variant].values:
                                v_pre = player_pre[player_pre[col_variant] == v]['primary_capped']
                                _, _, p_val = parametric_ttest(c_pre, v_pre)
                                if p_val < 0.05: st.error(f"⚠️ Pre-event baseline imbalance detected between Control and {v}! (p={p_val:.3f})")
                                else: st.success(f"✅ Baseline balanced between Control and {v} (p={p_val:.3f})")
                    else: st.warning("Insufficient baseline data rows available.")

                with tab1:
                    if not player_event.empty and ctrl_grp in player_event[col_variant].values:
                        ctrl_df = player_event[player_event[col_variant] == ctrl_grp]
                        
                        for v in var_grps:
                            if v in player_event[col_variant].values:
                                var_df = player_event[player_event[col_variant] == v]
                                st.markdown(f"### 📦 Evaluation Summary: `{v}` vs. `{ctrl_grp}` (Baseline)")
                                
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

                                exp_a = st.expander("💎 Combined Performance Tracker: ARPDAU", expanded=False)
                                with exp_a:
                                    display_continuous_results(ctrl_df['arpdau_capped'], var_df['arpdau_capped'], ctrl_grp, v)
                    else:
                        st.error("The chosen control group is missing from the active live window metrics log.")

                # Chart Engine
                st.header("📈 Multi-Group Daily Performance Trends")
                chart_tab1, chart_tab2 = st.tabs([f"Primary ({col_rev})", f"Secondary ({col_sec})"])
                daily_summary = df[df[col_variant].isin(all_active_groups)].groupby([col_date, col_variant]).agg(
                    p_spend=(col_rev, 'sum'), s_spend=(col_sec, 'sum'), dau=(col_player, 'nunique')
                ).reset_index()
                daily_summary['p_arpdau'] = daily_summary['p_spend'] / daily_summary['dau']
                daily_summary['s_arpdau'] = daily_summary['s_spend'] / daily_summary['dau']
                
                with chart_tab1:
                    fig, ax = plt.subplots(figsize=(12, 4))
                    for grp in all_active_groups:
                        g_data = daily_summary[daily_summary[col_variant] == grp]
                        if not g_data.empty: ax.plot(g_data[col_date], g_data['p_arpdau'], label=f"Group: {grp}", marker='o')
                    ax.axvspan(pd.Timestamp(pre_event_range[0]), pd.Timestamp(pre_event_range[1]), color='gray', alpha=0.1, label='Pre-Event')
                    ax.axvspan(pd.Timestamp(event_range[0]), pd.Timestamp(event_range[1]), color='green', alpha=0.07, label='Live-Event')
                    ax.set_ylabel("Daily Average"); ax.legend(); st.pyplot(fig)
                with chart_tab2:
                    fig, ax = plt.subplots(figsize=(12, 4))
                    for grp in all_active_groups:
                        g_data = daily_summary[daily_summary[col_variant] == grp]
                        if not g_data.empty: ax.plot(g_data[col_date], g_data['s_arpdau'], label=f"Group: {grp}", marker='o')
                    ax.axvspan(pd.Timestamp(pre_event_range[0]), pd.Timestamp(pre_event_range[1]), color='gray', alpha=0.1, label='Pre-Event')
                    ax.axvspan(pd.Timestamp(event_range[0]), pd.Timestamp(event_range[1]), color='green', alpha=0.07, label='Live-Event')
                    ax.set_ylabel("Daily Average"); ax.legend(); st.pyplot(fig)

        # =========================================================================
        # WORKFLOW 2: NEW USER COHORTS
        # =========================================================================
        else:
            st.header("⏳ Milestone Evaluation Target")
            target_age = st.slider("Evaluate Significance At Player Age Milestone (Day-N Lifecycle):", 0, 14, 7)
            
            df['player_age'] = (df[col_date] - df[col_signup]).dt.days
            cohort_df = df[(df['player_age'] >= 0) & (df['player_age'] <= 14) & df[col_variant].isin(all_active_groups)]
            all_players = cohort_df[[col_player, col_variant]].drop_duplicates()
            
            # --- OUTLIER MANAGEMENT AT INDIVIDUAL LEVEL ---
            player_totals = cohort_df.groupby([col_player, col_variant]).agg(
                total_p=(col_rev, 'sum'), 
                total_s=(col_sec, 'sum')
            ).reset_index()

            if cap_percentile < 100.0:
                cap_p = np.percentile(player_totals['total_p'], cap_percentile) if not player_totals.empty else 0
                cap_s = np.percentile(player_totals['total_s'], cap_percentile) if not player_totals.empty else 0
                
                cohort_df = cohort_df.merge(player_totals[[col_player, 'total_p', 'total_s']], on=col_player)
                
                cohort_df['p_capped'] = np.where(
                    cohort_df['total_p'] > cap_p, 
                    cohort_df[col_rev] * (cap_p / np.maximum(cohort_df['total_p'], 1e-6)), 
                    cohort_df[col_rev]
                )
                cohort_df['s_capped'] = np.where(
                    cohort_df['total_s'] > cap_s, 
                    cohort_df[col_sec] * (cap_s / np.maximum(cohort_df['total_s'], 1e-6)), 
                    cohort_df[col_sec]
                )
            else:
                cohort_df['p_capped'] = cohort_df[col_rev]
                cohort_df['s_capped'] = cohort_df[col_sec]

            # --- CUMULATIVE & NON-CUMULATIVE CALCULATION ---
            if "Cumulative" in cohort_mode:
                grouped_val = cohort_df.groupby([col_player, col_variant, 'player_age']).agg(
                    primary_val=('p_capped', 'sum'), 
                    sec_val=('s_capped', 'sum')
                )
                player_age_spend = grouped_val.groupby(level=[0, 1]).cumsum().reset_index()
                
                full_grid = pd.MultiIndex.from_product(
                    [all_players[col_player].unique(), range(15)], 
                    names=[col_player, 'player_age']
                ).to_frame().reset_index(drop=True)
                full_grid = full_grid.merge(all_players, on=col_player, how='left')
                
                merged_chart = pd.merge(full_grid, player_age_spend, on=[col_player, col_variant, 'player_age'], how='left').fillna(0)
                merged_chart['primary_val'] = merged_chart.groupby(col_player)['primary_val'].ffill().fillna(0)
                merged_chart['sec_val'] = merged_chart.groupby(col_player)['sec_val'].ffill().fillna(0)

                raw_snap = merged_chart[merged_chart['player_age'] == target_age]
                snapshot_df = raw_snap.rename(columns={'primary_val': 'primary_capped', 'sec_val': 'sec_capped'})
                
                ltv_summary = merged_chart.groupby(['player_age', col_variant]).agg(
                    chart_p=('primary_val', 'mean'), 
                    chart_s=('sec_val', 'mean')
                ).reset_index()
                
                y_label_p, y_label_s = "Cumulative LTV", "Cumulative LTV"
            else:
                full_grid = pd.MultiIndex.from_product(
                    [all_players[col_player].unique(), range(15)], 
                    names=[col_player, 'player_age']
                ).to_frame().reset_index(drop=True)
                full_grid = full_grid.merge(all_players, on=col_player, how='left')

                day_act = cohort_df.groupby([col_player, col_variant, 'player_age']).agg(
                    primary_val=('p_capped', 'sum'), 
                    sec_val=('s_capped', 'sum')
                ).reset_index()

                merged_chart = pd.merge(full_grid, day_act, on=[col_player, col_variant, 'player_age'], how='left').fillna(0)
                snapshot_df = merged_chart[merged_chart['player_age'] == target_age].rename(
                    columns={'primary_val': 'primary_capped', 'sec_val': 'sec_capped'}
                )

                ltv_summary = merged_chart.groupby(['player_age', col_variant]).agg(
                    chart_p=('primary_val', 'mean'), 
                    chart_s=('sec_val', 'mean')
                ).reset_index()
                
                y_label_p, y_label_s = "Per-User Daily Metric", "Per-User Daily Metric"

            # --- MULTI-METRIC SUMMARY MATRIX SECTION ---
            st.header("📊 Multi-Metric Summary Matrix")
            if matrix_metrics:
                summary_matrix_df = generate_summary_matrix(
                    cohort_df[cohort_df['player_age'] <= target_age], 
                    col_variant, col_player, ctrl_grp, var_grps, matrix_metrics
                )
                if not summary_matrix_df.empty:
                    st.dataframe(summary_matrix_df, use_container_width=True)
                else:
                    st.info("No data available to construct the summary matrix.")
            else:
                st.info("Please select metrics in the sidebar under 'Select Metrics for Summary Matrix'.")

            st.header("📊 Lifecycle Milestone Evaluation Results")
            if ctrl_grp not in all_players[col_variant].values:
                st.error("Baseline tracking lacks valid entries inside this timeline slice.")
            else:
                snapshot_df['is_conv_p'] = np.where(snapshot_df['primary_capped'] > 0, 1, 0)
                snapshot_df['is_conv_s'] = np.where(snapshot_df['sec_capped'] > 0, 1, 0)
                
                ctrl_df = snapshot_df[snapshot_df[col_variant] == ctrl_grp]
                
                for v in var_grps:
                    if v in snapshot_df[col_variant].values:
                        var_df = snapshot_df[snapshot_df[col_variant] == v]
                        st.markdown(f"### 📦 Milestone Summary: `{v}` vs. `{ctrl_grp}` (Baseline)")
                        
                        exp_p = st.expander(f"🔑 Day-{target_age} Primary Metric: {col_rev}", expanded=True)
                        with exp_p:
                            display_continuous_results(ctrl_df['primary_capped'], var_df['primary_capped'], ctrl_grp, v)
                            st.markdown("---")
                            display_proportion_results(int(ctrl_df['is_conv_p'].sum()), len(ctrl_df), int(var_df['is_conv_p'].sum()), len(var_df), ctrl_grp, v)
                        
                        exp_s = st.expander(f"🛡️ Day-{target_age} Secondary Metric: {col_sec}", expanded=False)
                        with exp_s:
                            display_continuous_results(ctrl_df['sec_capped'], var_df['sec_capped'], ctrl_grp, v)
                            st.markdown("---")
                            display_proportion_results(int(ctrl_df['is_conv_s'].sum()), len(ctrl_df), int(var_df['is_conv_s'].sum()), len(var_df), ctrl_grp, v)

                st.header("📈 New User Cohort Lifecycle Curves")
                chart_tab1, chart_tab2 = st.tabs([f"Primary Cumulative ({col_rev})", f"Secondary Cumulative ({col_sec})"])
                
                with chart_tab1:
                    fig, ax = plt.subplots(figsize=(12, 4))
                    for grp in all_active_groups:
                        g_data = ltv_summary[ltv_summary[col_variant] == grp]
                        if not g_data.empty: ax.plot(g_data['player_age'], g_data['chart_p'], label=f"Group Avg: {grp}", marker='s')
                    ax.axvline(x=target_age, color='red', linestyle='--', label=f'Target Milestone (Day {target_age})')
                    ax.set_ylabel(y_label_p); ax.set_xlabel("Days Since Install"); ax.set_xticks(range(0, 15)); ax.grid(True, linestyle=":", alpha=0.6); ax.legend(); st.pyplot(fig)
                with chart_tab2:
                    fig, ax = plt.subplots(figsize=(12, 4))
                    for grp in all_active_groups:
                        g_data = ltv_summary[ltv_summary[col_variant] == grp]
                        if not g_data.empty: ax.plot(g_data['player_age'], g_data['chart_s'], label=f"Group Avg: {grp}", marker='s')
                    ax.axvline(x=target_age, color='red', linestyle='--', label=f'Target Milestone (Day {target_age})')
                    ax.set_ylabel(y_label_s); ax.set_xlabel("Days Since Install"); ax.set_xticks(range(0, 15)); ax.grid(True, linestyle=":", alpha=0.6); ax.legend(); st.pyplot(fig)
    else:
        st.info("💡 Select a baseline Control and at least 1 Test Variant Group in the sidebar to execute multi-variant matrix processing.")
else:
    st.info("💡 Waiting for a CSV file upload to run mapping diagnostics.")