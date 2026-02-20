"""
Active learning loop using BoTorch's SingleTaskGP.

This module runs the full active-learning experiment and exposes the results
as module-level variables that the Dash app (active_learning_dash_app.py)
imports.  It is NOT meant to be run as a standalone script.

Overview of the algorithm
-------------------------
Active learning iteratively decides *where* to evaluate an expensive function
so as to learn it with as few evaluations as possible.  The recipe:

    1. Start with a small set of random observations.
    2. Fit a Gaussian Process (GP) surrogate model to the data.
    3. Use an acquisition function to choose the next evaluation point.
       Here we use **uncertainty sampling** — we pick the point where the
       GP's posterior standard deviation is highest, i.e. where the model
       is least sure about the function value.
    4. Evaluate the true function at that point and add the result.
    5. Repeat from step 2.

Key BoTorch concepts used
-------------------------
* **SingleTaskGP** — a standard GP model for single-output regression.
* **Normalize / Standardize** — input and outcome transforms that let the
  GP work in a well-conditioned internal space while we stay in the
  original input domain.
* **train_Yvar** — when set to a small constant (1e-6) the GP treats
  observations as essentially noiseless, which is correct for a
  deterministic objective.  Omit it when your observations are noisy and
  you want the GP to estimate the noise variance from data.
* **PosteriorStandardDeviation** — an analytic acquisition function that
  returns the GP's posterior std dev at a point.  Maximising it gives the
  uncertainty-sampling policy.
* **optimize_acqf** — BoTorch's utility to maximise an acquisition function
  over a bounded domain using multi-start L-BFGS.
"""

import numpy as np
import torch
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
from botorch.test_functions import Branin
from botorch.acquisition.analytic import PosteriorStandardDeviation
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood


# ── Experiment settings ───────────────────────────────────────────────────
# All tensors use float64 on CPU for numerical precision.
tkwargs = {"dtype": torch.double, "device": "cpu"}

N_INITIAL = 5        # number of initial random observations
N_ITERATIONS = 15    # number of active-learning steps after the initial set
GRID_RES = 80        # resolution of the dense grid used for visualisation
SEED = 42            # random seed for reproducibility

torch.manual_seed(SEED)


# ── Ground-truth objective ────────────────────────────────────────────────
# The Branin function is a standard 2-D test function for optimisation
# benchmarks.  Its domain is x1 in [-5, 10], x2 in [0, 15].
branin = Branin(negate=False).to(**tkwargs)

# `bounds` is a (2, d) tensor: bounds[0] = lower, bounds[1] = upper.
bounds = branin.bounds.to(**tkwargs)


def evaluate(X: torch.Tensor) -> torch.Tensor:
    """Evaluate the Branin function.  Returns shape (n, 1)."""
    return branin(X).unsqueeze(-1)


# ── Dense evaluation grid (for plotting) ──────────────────────────────────
# We build a regular grid that spans the full Branin domain so we can
# visualise the true function and the GP predictions as heatmaps / surfaces.
x1 = torch.linspace(bounds[0, 0].item(), bounds[1, 0].item(), GRID_RES, **tkwargs)
x2 = torch.linspace(bounds[0, 1].item(), bounds[1, 1].item(), GRID_RES, **tkwargs)
X1, X2 = torch.meshgrid(x1, x2, indexing="ij")
grid = torch.stack([X1.reshape(-1), X2.reshape(-1)], dim=-1)  # (GRID_RES^2, 2)

# Evaluate the true function on the grid (used for the "ground truth" plot).
true_Y: np.ndarray = branin(grid).reshape(GRID_RES, GRID_RES).detach().numpy()

# Numpy versions of the grid axes, used by the Dash app for Plotly traces.
x1_np: np.ndarray = x1.numpy()
x2_np: np.ndarray = x2.numpy()
X1_np: np.ndarray = X1.numpy()
X2_np: np.ndarray = X2.numpy()


# ── Active-learning loop ─────────────────────────────────────────────────
# We store a "snapshot" of the GP state after every iteration so that the
# Dash app can display any step without re-fitting.

print("Running active-learning loop …")

# Draw initial points uniformly at random inside the domain.
train_X = bounds[0] + (bounds[1] - bounds[0]) * torch.rand(N_INITIAL, 2, **tkwargs)
train_Y = evaluate(train_X)

# Because the Branin function is deterministic (no observation noise), we
# fix the observation noise variance to a tiny constant for numerical
# stability.  If your real objective is noisy, simply omit `train_Yvar`
# and the GP will estimate the noise from data.
train_Yvar = torch.full_like(train_Y, 1e-6)

# Each snapshot stores numpy arrays so the Dash app can use them directly.
snapshots: list[dict] = []

for i in range(N_ITERATIONS + 1):
    # ── Step 2: fit the GP ────────────────────────────────────────────
    # * Normalize(d=2, bounds=bounds) maps inputs to [0, 1]^2 internally.
    # * Standardize(m=1) zero-means and unit-variances the outputs.
    # Both transforms are applied transparently — we always pass data and
    # query points in the *original* input/output space.
    gp = SingleTaskGP(
        train_X,
        train_Y,
        train_Yvar=train_Yvar,
        input_transform=Normalize(d=2, bounds=bounds),
        outcome_transform=Standardize(m=1),
    ).to(**tkwargs)

    # ExactMarginalLogLikelihood is the objective for GP hyperparameter
    # optimisation (kernel lengthscales, outputscale, etc.).
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)

    # fit_gpytorch_mll runs L-BFGS to find the MAP hyperparameters.
    fit_gpytorch_mll(mll)

    # ── Evaluate the GP on the dense grid for plotting ────────────────
    with torch.no_grad():
        posterior = gp.posterior(grid)
        # posterior.mean  -> (GRID_RES^2, 1) predictive mean
        # posterior.variance -> (GRID_RES^2, 1) predictive variance
        mean = posterior.mean.squeeze(-1).numpy().reshape(GRID_RES, GRID_RES)
        std = posterior.variance.squeeze(-1).sqrt().numpy().reshape(GRID_RES, GRID_RES)

    snapshots.append(
        {
            "mean": mean,           # GP posterior mean on the grid
            "std": std,             # GP posterior std dev on the grid
            "train_X": train_X.numpy().copy(),  # all training inputs so far
            "train_Y": train_Y.numpy().copy(),  # all training outputs so far
        }
    )
    print(f"  iteration {i}/{N_ITERATIONS}")

    # After the final snapshot we stop — no more acquisition needed.
    if i == N_ITERATIONS:
        break

    # ── Step 3: choose the next evaluation point ──────────────────────
    # PosteriorStandardDeviation returns sigma(x).  Maximising it picks
    # the point of greatest model uncertainty ("uncertainty sampling").
    acqf = PosteriorStandardDeviation(model=gp)

    # optimize_acqf performs multi-start optimisation over the domain.
    # `q=1` means we select one new point per iteration (sequential AL).
    candidate, _ = optimize_acqf(
        acq_function=acqf,
        bounds=bounds,       # original-space bounds (transforms handle the rest)
        q=1,
        num_restarts=8,      # number of L-BFGS restarts
        raw_samples=128,     # random candidates to seed the restarts
    )

    # ── Step 4: evaluate and augment the dataset ──────────────────────
    new_X = candidate
    new_Y = evaluate(new_X)
    train_X = torch.cat([train_X, new_X])
    train_Y = torch.cat([train_Y, new_Y])
    train_Yvar = torch.full_like(train_Y, 1e-6)

print(f"Active learning complete — {len(snapshots)} snapshots stored.\n")


# ── Derived quantities used by the Dash app ───────────────────────────────
# Fixed colour ranges so that all iterations use the same scale.
mean_zmin: float = min(float(true_Y.min()), min(s["mean"].min() for s in snapshots))
mean_zmax: float = max(float(true_Y.max()), max(s["mean"].max() for s in snapshots))
std_zmax: float = max(float(s["std"].max()) for s in snapshots)
zaxis_range: list[float] = [mean_zmin - 20, mean_zmax + 20]
