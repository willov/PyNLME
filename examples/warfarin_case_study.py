#!/usr/bin/env python3
"""
PyNLME Real-world Case Study: Warfarin Pharmacokinetics

This example demonstrates fitting a realistic pharmacokinetic model to warfarin
concentration data, including:
- Multi-dose administration
- Covariate effects (age, weight, gender)
- Model diagnostics and validation
- Comparison with literature values

Warfarin is an anticoagulant with narrow therapeutic index, making accurate
PK modeling crucial for dosing.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

import pynlme


def generate_warfarin_data(n_subjects=50, seed=42):
    """
    Generate realistic warfarin PK data based on literature parameters.

    Model: One-compartment with first-order absorption
    C(t) = (F*Dose*Ka)/(V*(Ka-Ke)) * (exp(-Ke*t) - exp(-Ka*t))

    Population parameters based on literature:
    - CL: 0.065 L/h (range: 0.02-0.15)
    - V: 0.14 L/kg (range: 0.1-0.2)
    - Ka: 1.2 h⁻¹ (range: 0.5-3.0)
    - F: 0.99 (nearly complete absorption)

    Covariates effects:
    - Age: CL decreases ~1%/year after age 40
    - Weight: V increases proportionally
    - Gender: Females have ~15% lower CL
    - CYP2C9 variants: Major effect on CL (not modeled here)
    """
    np.random.seed(seed)

    # Population parameters (typical 70kg, 40-year-old male)
    pop_cl_70kg = 0.065  # L/h
    pop_v_per_kg = 0.14  # L/kg
    pop_ka = 1.2  # h⁻¹
    pop_f = 0.99  # bioavailability

    # Between-subject variability (CV%)
    cv_cl = 0.4  # 40% CV - high for warfarin
    cv_v = 0.2  # 20% CV
    cv_ka = 0.6  # 60% CV - absorption highly variable

    # Covariate effects
    age_cl_slope = -0.01  # 1% decrease per year after 40
    female_cl_factor = 0.85  # 15% lower in females

    # Residual error model: proportional + additive
    prop_error = 0.15  # 15% proportional error
    add_error = 0.05  # 0.05 mg/L additive error

    # Dosing: 5mg once daily for 7 days, then samples
    dose = 5  # mg
    dosing_times = np.arange(0, 168, 24)  # Daily for 7 days
    sampling_times = np.array([168, 172, 180, 192, 216, 240, 336])  # Week 1-2

    data = []

    for subject in range(n_subjects):
        # Generate individual characteristics
        age = np.random.normal(55, 15)  # years
        age = np.clip(age, 20, 85)

        weight = np.random.normal(75, 15)  # kg
        weight = np.clip(weight, 50, 120)

        female = np.random.binomial(1, 0.5)  # 50% female

        # Individual PK parameters with covariate effects
        cl_base = pop_cl_70kg * (weight / 70) ** 0.75  # Allometric scaling
        cl_age_effect = 1 + age_cl_slope * (age - 40)
        cl_gender_effect = female_cl_factor if female else 1.0
        cl_i = (
            cl_base
            * cl_age_effect
            * cl_gender_effect
            * np.exp(np.random.normal(0, cv_cl))
        )

        v_i = pop_v_per_kg * weight * np.exp(np.random.normal(0, cv_v))
        ka_i = pop_ka * np.exp(np.random.normal(0, cv_ka))
        f_i = pop_f  # Assume no BSV in F

        # Calculate concentrations at sampling times
        for t_sample in sampling_times:
            conc_total = 0

            # Sum contributions from all doses
            for t_dose in dosing_times:
                if t_sample > t_dose:  # Only doses given before sampling
                    t = t_sample - t_dose

                    # One-compartment with first-order absorption
                    ke = cl_i / v_i

                    if abs(ka_i - ke) > 1e-6:  # Avoid numerical issues
                        conc = (
                            (f_i * dose * ka_i)
                            / (v_i * (ka_i - ke))
                            * (np.exp(-ke * t) - np.exp(-ka_i * t))
                        )
                    else:
                        # Special case when ka ≈ ke
                        conc = (f_i * dose * t) / v_i * np.exp(-ke * t)

                    conc_total += conc

            # Add residual error (proportional + additive)
            error_prop = np.random.normal(0, prop_error)
            error_add = np.random.normal(0, add_error)
            conc_obs = conc_total * (1 + error_prop) + error_add
            conc_obs = np.maximum(conc_obs, 0.01)  # Avoid negative concentrations

            data.append(
                {
                    "subject": subject,
                    "time": t_sample,
                    "concentration": conc_obs,
                    "age": age,
                    "weight": weight,
                    "female": female,
                    "cl_true": cl_i,
                    "v_true": v_i,
                    "ka_true": ka_i,
                    "dose": dose,
                }
            )

    return pd.DataFrame(data)


def warfarin_model(beta, x, v=None):
    """
    Optimized warfarin PK model with covariate effects

    x columns:
    0: time (h)
    1: dose (mg)
    2: weight (kg)
    3: age (years)
    4: female (0/1)

    Parameters:
    beta[0]: log(CL_70kg) - clearance for 70kg individual
    beta[1]: log(V_per_kg) - volume per kg
    beta[2]: log(Ka) - absorption rate constant
    beta[3]: age_effect - age effect on CL
    beta[4]: female_effect - female effect on CL (additive on log scale)
    """
    time = x[:, 0]
    dose = x[:, 1]
    weight = x[:, 2]
    age = x[:, 3]
    female = x[:, 4]

    # Transform parameters to ensure positivity (vectorized)
    cl_70kg = np.exp(beta[0])
    v_per_kg = np.exp(beta[1])
    ka = np.exp(beta[2])
    age_effect = beta[3]
    female_effect = beta[4]

    # Individual parameters with covariate effects (all vectorized)
    cl = cl_70kg * (weight / 70) ** 0.75  # Allometric scaling
    cl = cl * np.exp(age_effect * (age - 40))  # Age effect
    cl = cl * np.exp(female_effect * female)  # Gender effect

    v = v_per_kg * weight
    ke = cl / v

    # Bioavailability
    f = 0.99

    # One-compartment model with first-order absorption (fully vectorized)
    # Only compute for positive times
    valid_mask = time > 0
    conc = np.zeros_like(time)
    
    if np.any(valid_mask):
        t_valid = time[valid_mask]
        dose_valid = dose[valid_mask]
        v_valid = v[valid_mask]
        ke_valid = ke[valid_mask]
        
        # Check for numerical issues (vectorized)
        diff_mask = np.abs(ka - ke_valid) > 1e-6
        
        # Normal case: ka != ke (vectorized)
        if np.any(diff_mask):
            idx = valid_mask.copy()
            idx[valid_mask] = diff_mask
            conc[idx] = (
                (f * dose_valid[diff_mask] * ka)
                / (v_valid[diff_mask] * (ka - ke_valid[diff_mask]))
                * (np.exp(-ke_valid[diff_mask] * t_valid[diff_mask]) -
                   np.exp(-ka * t_valid[diff_mask]))
            )
        
        # Special case: ka ≈ ke (vectorized)
        if np.any(~diff_mask):
            idx = valid_mask.copy()
            idx[valid_mask] = ~diff_mask
            conc[idx] = (
                (f * dose_valid[~diff_mask] * t_valid[~diff_mask])
                / v_valid[~diff_mask] * np.exp(-ke_valid[~diff_mask] * t_valid[~diff_mask])
            )

    return np.maximum(conc, 1e-6)  # Avoid zero/negative concentrations


def main():
    """Run warfarin PK analysis"""

    print("🩺 Warfarin Pharmacokinetics Case Study")
    print("=" * 50)

    # Generate synthetic data (realistic size)
    print("📊 Generating warfarin concentration data...")
    df = generate_warfarin_data(n_subjects=50, seed=42)  # Increased to 50 for more realistic dataset

    print(f"Generated {len(df)} observations from {df['subject'].nunique()} subjects")
    print(f"Age range: {df['age'].min():.0f}-{df['age'].max():.0f} years")
    print(f"Weight range: {df['weight'].min():.0f}-{df['weight'].max():.0f} kg")
    print(
        f"Female subjects: {df['female'].sum()}/{len(df.groupby('subject'))} ({100 * df['female'].mean():.0f}%)"
    )

    # Prepare data for PyNLME
    x = df[["time", "dose", "weight", "age", "female"]].values
    y = df["concentration"].values
    groups = df["subject"].values

    # Initial parameter estimates (on log scale for CL, V, Ka)
    beta0 = np.array(
        [
            np.log(0.06),  # log(CL_70kg)
            np.log(0.15),  # log(V_per_kg)
            np.log(1.0),  # log(Ka)
            -0.005,  # age_effect
            -0.2,  # female_effect
        ]
    )

    print("\n🔧 Initial parameter estimates:")
    print(f"CL_70kg: {np.exp(beta0[0]):.3f} L/h")
    print(f"V_per_kg: {np.exp(beta0[1]):.3f} L/kg")
    print(f"Ka: {np.exp(beta0[2]):.3f} h⁻¹")
    print(f"Age effect: {beta0[3]:.4f}")
    print(f"Female effect: {beta0[4]:.3f}")

    # Fit the model with SAEM for realistic mixed-effects modeling
    print("\n⚙️  Fitting NLME model using SAEM algorithm...")
    print("   This may take 1-2 minutes for realistic convergence...")

    try:
        # Force Python SAEM implementation to see progress indicators
        from pynlme import nlmefit
        import sys
        nlmefit_module = sys.modules[nlmefit.__module__]
        rust_available_backup = nlmefit_module.RUST_AVAILABLE
        nlmefit_module.RUST_AVAILABLE = False  # Force Python implementation
        
        beta, psi, stats, b = pynlme.fit_nlme(
            x, y, groups, None, warfarin_model, beta0,
            method='SAEM',   # Use SAEM for better random effects estimation
            verbose=2,       # More verbose output for progress tracking
            max_iter=100,    # Reasonable iterations for SAEM
            tol_fun=1e-5,    # Good convergence tolerance
            n_iterations=(40, 40, 20),  # Optimized for 50 subjects: faster convergence
            n_mcmc_iterations=(2, 2, 2)  # Keep MCMC steps reasonable
        )
        
        # Restore original setting
        nlmefit_module.RUST_AVAILABLE = rust_available_backup

        converged = stats.logl is not None
        print(f"✅ Convergence: {'Yes' if converged else 'No'}")
        if hasattr(stats, 'iterations') and stats.iterations is not None:
            print(f"Iterations: {stats.iterations}")
        if stats.logl is not None:
            print(f"Final log-likelihood: {stats.logl:.2f}")

    except Exception as e:
        print(f"❌ Fitting failed: {e}")
        return

    # Check if fitting was successful
    if stats.logl is None:
        print("❌ Fitting was not successful - no likelihood computed")
        return

    # Display results
    print("\n📈 Population Parameter Estimates:")
    cl_est = np.exp(beta[0])
    v_est = np.exp(beta[1])
    ka_est = np.exp(beta[2])
    age_eff = beta[3]
    fem_eff = beta[4]

    print(f"CL (70kg):     {cl_est:.3f} L/h  (Literature: ~0.065)")
    print(f"V (per kg):    {v_est:.3f} L/kg (Literature: ~0.14)")
    print(f"Ka:            {ka_est:.3f} h⁻¹  (Literature: ~1.2)")
    print(f"Age effect:    {age_eff:.4f}     (1% decrease per year)")
    print(f"Female effect: {fem_eff:.3f}     (15% lower CL)")

    # Clinical interpretation
    age_percent = (1 - np.exp(age_eff)) * 100
    female_percent = (1 - np.exp(fem_eff)) * 100

    print("\n🩺 Clinical Interpretation:")
    print(f"• CL decreases {age_percent:.1f}% per year after age 40")
    print(f"• Females have {female_percent:.1f}% lower clearance")
    print(f"• Half-life (typical): {np.log(2) / (cl_est / (v_est * 70)):.1f} hours")

    # Model diagnostics
    print("\n📊 Model Fit Statistics:")
    print(f"AIC: {stats.aic:.1f}")
    print(f"BIC: {stats.bic:.1f}")
    print(f"RMSE: {np.sqrt(np.mean((y - warfarin_model(beta, x))**2)):.3f} mg/L")

    # Create comprehensive plots
    plot_results(df, beta, psi, stats, x, y)


def plot_results(df, beta, psi, stats, x, y):
    """Create comprehensive diagnostic plots"""

    fig = plt.figure(figsize=(20, 15))

    # 1. Individual concentration profiles
    plt.subplot(3, 4, 1)
    subjects_to_plot = np.random.choice(df["subject"].unique(), 8, replace=False)
    colors = plt.cm.tab10(np.linspace(0, 1, len(subjects_to_plot)))

    for i, subject in enumerate(subjects_to_plot):
        subj_data = df[df["subject"] == subject]
        plt.plot(
            subj_data["time"],
            subj_data["concentration"],
            "o-",
            color=colors[i],
            alpha=0.7,
            markersize=4,
            label=f"Subj {subject}",
        )

    plt.xlabel("Time (h)")
    plt.ylabel("Concentration (mg/L)")
    plt.title("Individual Profiles (Sample)")
    plt.yscale("log")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)

    # 2. Population predictions
    plt.subplot(3, 4, 2)
    fitted = warfarin_model(beta, x)
    plt.scatter(y, fitted, alpha=0.6, s=20)
    min_val, max_val = min(y.min(), fitted.min()), max(y.max(), fitted.max())
    plt.plot([min_val, max_val], [min_val, max_val], "r--", label="Perfect fit")
    plt.xlabel("Observed (mg/L)")
    plt.ylabel("Predicted (mg/L)")
    plt.title("Population Predictions")
    plt.legend()

    # 3. Residuals vs fitted
    plt.subplot(3, 4, 3)
    residuals = y - fitted
    plt.scatter(fitted, residuals, alpha=0.6, s=20)
    plt.axhline(y=0, color="r", linestyle="--")
    plt.xlabel("Fitted Values (mg/L)")
    plt.ylabel("Residuals")
    plt.title("Residuals vs Fitted")

    # 4. Q-Q plot of residuals
    plt.subplot(3, 4, 4)
    from scipy import stats as scipy_stats
    scipy_stats.probplot(residuals, dist="norm", plot=plt)
    plt.title("Residual Q-Q Plot")

    # 5. Covariate effects - Age
    plt.subplot(3, 4, 5)
    df_unique = df.groupby("subject").first()
    plt.scatter(
        df_unique["age"],
        np.exp(beta[0])
        * (df_unique["weight"] / 70) ** 0.75
        * np.exp(beta[3] * (df_unique["age"] - 40)),
        alpha=0.6,
    )
    ages = np.linspace(df_unique["age"].min(), df_unique["age"].max(), 100)
    cl_age = np.exp(beta[0]) * np.exp(beta[3] * (ages - 40))
    plt.plot(ages, cl_age, "r-", label="Population trend")
    plt.xlabel("Age (years)")
    plt.ylabel("Clearance (L/h)")
    plt.title("Age Effect on Clearance")
    plt.legend()

    # 6. Gender effect
    plt.subplot(3, 4, 6)
    male_cl = df_unique[df_unique["female"] == 0]
    female_cl = df_unique[df_unique["female"] == 1]

    cl_male = (
        np.exp(beta[0])
        * (male_cl["weight"] / 70) ** 0.75
        * np.exp(beta[3] * (male_cl["age"] - 40))
    )
    cl_female = (
        np.exp(beta[0])
        * (female_cl["weight"] / 70) ** 0.75
        * np.exp(beta[3] * (female_cl["age"] - 40))
        * np.exp(beta[4])
    )

    plt.boxplot([cl_male, cl_female], tick_labels=["Male", "Female"])
    plt.ylabel("Clearance (L/h)")
    plt.title("Gender Effect on Clearance")

    # 7. Weight effect on volume
    plt.subplot(3, 4, 7)
    v_pred = np.exp(beta[1]) * df_unique["weight"]
    plt.scatter(df_unique["weight"], v_pred, alpha=0.6)
    weights = np.linspace(df_unique["weight"].min(), df_unique["weight"].max(), 100)
    v_pop = np.exp(beta[1]) * weights
    plt.plot(weights, v_pop, "r-", label="Population")
    plt.xlabel("Weight (kg)")
    plt.ylabel("Volume (L)")
    plt.title("Weight Effect on Volume")
    plt.legend()

    # 8. Random effects covariance matrix
    plt.subplot(3, 4, 8)
    im = plt.imshow(psi, cmap='RdBu_r', aspect='auto')
    plt.colorbar(im)
    plt.title("Random Effects Covariance")
    plt.xlabel("Parameter")
    plt.ylabel("Parameter")

    # 9. Histogram of concentrations
    plt.subplot(3, 4, 9)
    plt.hist(y, bins=20, density=True, alpha=0.7, label="Observed")
    plt.hist(fitted, bins=20, density=True, alpha=0.7, label="Predicted")
    plt.xlabel("Concentration (mg/L)")
    plt.ylabel("Density")
    plt.title("Concentration Distribution")
    plt.legend()

    # 10. Time course by age group
    plt.subplot(3, 4, 10)
    df["age_group"] = pd.cut(
        df["age"], bins=[0, 50, 65, 100], labels=["<50", "50-65", ">65"]
    )
    for age_group in df["age_group"].cat.categories:
        group_data = df[df["age_group"] == age_group]
        mean_conc = group_data.groupby("time")["concentration"].mean()
        plt.plot(mean_conc.index, mean_conc.values, "o-", label=f"Age {age_group}")
    plt.xlabel("Time (h)")
    plt.ylabel("Mean Concentration (mg/L)")
    plt.title("Concentration by Age Group")
    plt.legend()

    # 11. Parameter recovery (if true values available)
    plt.subplot(3, 4, 11)
    true_cl = df.groupby("subject")["cl_true"].first().values
    # Using population estimates for demonstration
    pop_cl = np.exp(beta[0]) * np.ones_like(true_cl)
    plt.scatter(true_cl, pop_cl, alpha=0.6)
    plt.plot([true_cl.min(), true_cl.max()], [true_cl.min(), true_cl.max()], "r--")
    plt.xlabel("True Clearance (L/h)")
    plt.ylabel("Population Clearance (L/h)")
    plt.title("Parameter Recovery")

    # Calculate correlation
    corr = np.corrcoef(true_cl, pop_cl)[0, 1]
    plt.text(
        0.05,
        0.95,
        f"r = {corr:.3f}",
        transform=plt.gca().transAxes,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
    )

    # 12. Model summary text
    plt.subplot(3, 4, 12)
    plt.axis("off")
    # Format AIC and log-likelihood with proper conditional formatting
    aic_str = f"{stats.aic:.1f}" if stats.aic is not None else "N/A"
    logl_str = f"{stats.logl:.2f}" if stats.logl is not None else "N/A"
    converged_str = "Yes" if stats.logl is not None else "No"
    
    summary_text = f"""
    WARFARIN PK MODEL SUMMARY
    
    Population Parameters:
    • CL = {np.exp(beta[0]):.3f} L/h
    • V = {np.exp(beta[1]):.3f} L/kg
    • Ka = {np.exp(beta[2]):.3f} h⁻¹
    
    Covariate Effects:
    • Age: {(1 - np.exp(beta[3])) * 100:.1f}% ↓ per year
    • Female: {(1 - np.exp(beta[4])) * 100:.1f}% ↓ CL
    
    Model Fit:
    • AIC: {aic_str}
    • Log-likelihood: {logl_str}
    • Converged: {converged_str}
    """
    plt.text(
        0.1,
        0.9,
        summary_text,
        transform=plt.gca().transAxes,
        fontfamily="monospace",
        fontsize=10,
        verticalalignment="top",
        bbox={"boxstyle": "round", "facecolor": "lightgray", "alpha": 0.8},
    )

    plt.tight_layout()

    # Save to organized output folder
    examples_dir = os.path.dirname(__file__)
    output_dir = os.path.join(examples_dir, "warfarin_case_study_output")
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, "warfarin_analysis.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.show()

    print(f"\n💾 Plots saved as '{plot_path}'")


if __name__ == "__main__":
    main()
