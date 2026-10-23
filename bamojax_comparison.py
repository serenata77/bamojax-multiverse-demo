"""
Minimal demonstration of Bayesian model comparison using JAX.
Aligns with the core philosophy of bamojax: functional, jit-compiled, and vectorized.
"""

from functools import partial
import jax
import jax.numpy as jnp
from jax.scipy.stats import norm, t
import pytest

# ---------------------------------------------------------------------------
# 1. Define Model Hypotheses (Log-Joint Probabilities)
# ---------------------------------------------------------------------------

def log_joint_gaussian(params, x, y):
    beta, log_sigma = params
    sigma = jnp.exp(log_sigma)
    
    # Priors: beta ~ N(0, 5), log_sigma ~ N(0, 1)
    log_prior = norm.logpdf(beta, loc=0.0, scale=5.0) + norm.logpdf(log_sigma, loc=0.0, scale=1.0)
    
    # Likelihood
    mu = beta * x
    log_lik = jnp.sum(norm.logpdf(y, loc=mu, scale=sigma))
    
    return log_prior + log_lik


def log_joint_robust(params, x, y, df=3.0):
    beta, log_sigma = params
    sigma = jnp.exp(log_sigma)
    
    # Priors
    log_prior = norm.logpdf(beta, loc=0.0, scale=5.0) + norm.logpdf(log_sigma, loc=0.0, scale=1.0)
    
    # Likelihood (Student-t for outlier resilience)
    mu = beta * x
    log_lik = jnp.sum(t.logpdf((y - mu) / sigma, df=df) - log_sigma)
    
    return log_prior + log_lik


# ---------------------------------------------------------------------------
# 2. Vectorized Grid Evaluation across Parameter Space
# ---------------------------------------------------------------------------

@partial(jax.jit, static_argnums=(0,))
def evaluate_parameter_grid(model_fn, grid_params, x, y):
    vmapped_eval = jax.vmap(lambda p: model_fn(p, x, y))
    return vmapped_eval(grid_params)


# ---------------------------------------------------------------------------
# 3. Model Comparison Execution
# ---------------------------------------------------------------------------

def run_model_comparison():
    key = jax.random.PRNGKey(42)
    
    # Synthetic data with an outlier at index 5
    key, subkey = jax.random.split(key)
    x = jnp.linspace(-2.0, 2.0, 20)
    y = 1.5 * x + jax.random.normal(subkey, shape=x.shape) * 0.5
    y = y.at[5].set(8.0)
    
    # Evaluation grid
    betas = jnp.linspace(0.0, 3.0, 50)
    log_sigmas = jnp.linspace(-1.0, 1.0, 50)
    B, S = jnp.meshgrid(betas, log_sigmas)
    grid_params = jnp.column_stack([B.ravel(), S.ravel()])
    
    # Evaluate across models
    log_joints_gauss = evaluate_parameter_grid(log_joint_gaussian, grid_params, x, y)
    log_joints_robust = evaluate_parameter_grid(log_joint_robust, grid_params, x, y)
    
    # Approximate marginal log-evidence using logsumexp
    evidence_gauss = float(jax.nn.logsumexp(log_joints_gauss))
    evidence_robust = float(jax.nn.logsumexp(log_joints_robust))
    log_bf = evidence_robust - evidence_gauss
    
    print(f"Log-evidence (Gaussian Likelihood): {evidence_gauss:.2f}")
    print(f"Log-evidence (Student-t Likelihood):  {evidence_robust:.2f}")
    print(f"Log Bayes Factor (Robust vs Gaussian): {log_bf:.2f}")
    
    return log_bf


# ---------------------------------------------------------------------------
# 4. Automated Tests
# ---------------------------------------------------------------------------

def test_evidence_prefers_robust_on_outliers():
    log_bf = run_model_comparison()
    assert log_bf > 0.0, "Expected robust model to outperform Gaussian on contaminated data."

def test_deterministic_jit_compilation():
    x = jnp.array([1.0, 2.0])
    y = jnp.array([1.5, 3.0])
    params = jnp.array([[1.5, 0.0], [1.5, 0.0]])
    res = evaluate_parameter_grid(log_joint_gaussian, params, x, y)
    assert jnp.allclose(res[0], res[1])


if __name__ == "__main__":
    run_model_comparison()