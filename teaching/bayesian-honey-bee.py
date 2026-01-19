# HoneyBee

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pymc as pm
import arviz as az
from sklearn.preprocessing import StandardScaler

# Assuming df is your DataFrame with the data
# Scale the data for numerical stability
scaler_x = StandardScaler()
scaler_y = StandardScaler()
x = scaler_x.fit_transform(df['Precipitation Percent of Average (%)'].values.reshape(-1, 1)).flatten()
y = scaler_y.fit_transform(df['Pesticide Use (lbs)'].values.reshape(-1, 1)).flatten()

# Scatterplot (using original data for visualization)
plt.figure(figsize=(10, 6))
sns.scatterplot(x=df['Precipitation Percent of Average (%)'], y=df['Pesticide Use (lbs)'], 
                hue=df['Precipitation Percent of Average (%)'], s=150, 
                edgecolor='black', linewidth=1.5)

# Bayesian linear regression with MCMC (on scaled data)
with pm.Model() as model:
    # Priors (adjusted for scaled data)
    intercept = pm.Normal('intercept', mu=0, sigma=10)
    slope = pm.Normal('slope', mu=0, sigma=10)
    # Optional: Add quadratic term if nonlinear relationship is suspected
    slope_quad = pm.Normal('slope_quad', mu=0, sigma=10)
    sigma = pm.HalfNormal('sigma', sigma=10)

    # Linear + quadratic model
    mu = intercept + slope * x + slope_quad * x**2

    # Likelihood (use StudentT for robustness to outliers)
    likelihood = pm.StudentT('likelihood', nu=5, mu=mu, sigma=sigma, observed=y)

    # MCMC sampling with increased tuning for better convergence
    trace = pm.sample(3000, tune=2000, return_inferencedata=True, target_accept=0.95)

# Posterior predictive regression line (on original scale)
x_orig = df['Precipitation Percent of Average (%)'].values
x_scaled = scaler_x.transform(x_orig.reshape(-1, 1)).flatten()
x_range = np.linspace(x_scaled.min(), x_scaled.max(), 100)
x_range_orig = scaler_x.inverse_transform(x_range.reshape(-1, 1)).flatten()

# Compute mean regression line
posterior = trace.posterior
intercept_mean = posterior['intercept'].mean().item()
slope_mean = posterior['slope'].mean().item()
slope_quad_mean = posterior['slope_quad'].mean().item()
y_scaled = intercept_mean + slope_mean * x_range + slope_quad_mean * x_range**2
y_orig = scaler_y.inverse_transform(y_scaled.reshape(-1, 1)).flatten()

# Plot mean regression line
plt.plot(x_range_orig, y_orig, color='lightblue', linestyle='--', linewidth=2.5, 
         label=f'ML Fit: intercept={intercept_mean:.2f}, slope={slope_mean:.2f}, quad={slope_quad_mean:.2f}')

# Plot 95% credible interval for uncertainty
y_samples = []
for i in np.random.randint(0, posterior['intercept'].size, 100):
    intercept_s = posterior['intercept'].values.flatten()[i]
    slope_s = posterior['slope'].values.flatten()[i]
    slope_quad_s = posterior['slope_quad'].values.flatten()[i]
    y_s = intercept_s + slope_s * x_range + slope_quad_s * x_range**2
    y_samples.append(scaler_y.inverse_transform(y_s.reshape(-1, 1)).flatten())
y_samples = np.array(y_samples)
y_lower = np.percentile(y_samples, 2.5, axis=0)
y_upper = np.percentile(y_samples, 97.5, axis=0)
plt.fill_between(x_range_orig, y_lower, y_upper, color='gray', alpha=0.2, label='95% Credible Interval')

# Customize plot
plt.title('Pesticide Use vs Precipitation Percent of Average (California, 2017-2022)')
plt.ylabel('Pesticide Use (lbs)')
plt.legend(title='ML Regression')
plt.tight_layout()
plt.show()

# Check MCMC diagnostics
print(az.summary(trace, hdi_prob=0.95))
az.plot_trace(trace)
plt.show()

# Posterior predictive check
with model:
    ppc = pm.sample_posterior_predictive(trace, extend_inferencedata=True)
y_ppc = ppc.posterior_predictive['likelihood'].values
y_ppc_orig = scaler_y.inverse_transform(y_ppc.reshape(-1, y.shape[0])).mean(axis=0)
plt.figure(figsize=(10, 6))
plt.scatter(x_orig, df['Pesticide Use (lbs)'], label='Data', s=100, edgecolor='black')
plt.scatter(x_orig, y_ppc_orig, color='red', alpha=0.5, label='Posterior Predictive Mean', s=50)
plt.xlabel('Precipitation Percent of Average (%)')
plt.ylabel('Pesticide Use (lbs)')
plt.title('Posterior Predictive Check')
plt.legend()
plt.show()
