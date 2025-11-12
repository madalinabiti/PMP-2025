import numpy as np
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt

# a.
data = np.array([56,60,58,55,57,59,61,56,58,60])
x_bar = data.mean()
print(f"Sample mean = {x_bar:.2f}, Sample std = {data.std(ddof=1):.2f}")

with pm.Model() as weak_model:
    mu = pm.Normal("mu", mu=x_bar, sigma=10)
    sigma = pm.HalfNormal("sigma", sigma=10)
    y = pm.Normal("y", mu=mu, sigma=sigma, observed=data)
    
    trace_weak = pm.sample(2000, tune=2000, target_accept=0.9, random_seed=42)
    summary_weak = az.summary(trace_weak, var_names=["mu", "sigma"], hdi_prob=0.95)

print("\nPosterior summaries (Weak Prior):")
print(summary_weak)

# b - Summarize posterior
summary_weak = az.summary(trace_weak, var_names=["mu", "sigma"], hdi_prob=0.95)

print("\nPosterior summaries (Weak Prior):")
print(summary_weak)

az.plot_posterior(trace_weak, var_names=["mu", "sigma"], hdi_prob=0.95)

# c

x_std = data.std(ddof=1)

print(f"Frequentist mean (x̄) = {x_bar:.2f}")
print(f"Frequentist std (s)  = {x_std:.2f}")

mu_mean_bayes = summary_weak.loc["mu", "mean"]
mu_hdi_low = summary_weak.loc["mu", "hdi_2.5%"]
mu_hdi_high = summary_weak.loc["mu", "hdi_97.5%"]

sigma_mean_bayes = summary_weak.loc["sigma", "mean"]
sigma_hdi_low = summary_weak.loc["sigma", "hdi_2.5%"]
sigma_hdi_high = summary_weak.loc["sigma", "hdi_97.5%"]

print("\nBayesian posterior (weak prior):")
print(f"μ_mean = {mu_mean_bayes:.2f}, 95% HDI = [{mu_hdi_low:.2f}, {mu_hdi_high:.2f}]")
print(f"σ_mean = {sigma_mean_bayes:.2f}, 95% HDI = [{sigma_hdi_low:.2f}, {sigma_hdi_high:.2f}]")

# Simple comparison
print("\nComparison:")
print(f"Frequentist μ = {x_bar:.2f} vs Bayesian μ ≈ {mu_mean_bayes:.2f}")
print(f"Frequentist σ = {x_std:.2f} vs Bayesian σ ≈ {sigma_mean_bayes:.2f}")

