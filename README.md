# Vectorized Bayesian Model Comparison in JAX

A minimal demonstration of candidate hypothesis evaluation in JAX, aligning with the functional programming principles of `bamojax` (stateless execution, `@partial(jax.jit, static_argnums=...)`, and `jax.vmap`).

## Features
- Compares marginal log-evidence across a parameter grid for competing observation likelihoods (Gaussian vs. Student-t) on contaminated data.
- Approximates the log Bayes factor using parallelized log-sum-exp reductions over vectorized model evaluations.
- Includes automated regression assertions using \`pytest\`.

## Requirements & Usage
\`\`\`bash
pip install jax pytest
python bamojax_comparison.py
pytest bamojax_comparison.py
\`\`\`

