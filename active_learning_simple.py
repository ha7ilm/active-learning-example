"""
Simple Active Learning with BoTorch and SingleTaskGP.

We aim to learn an unknown function (Branin) using as few evaluations as
possible.  At each step we:
  1. Fit a SingleTaskGP to the observations collected so far.
  2. Pick the next point to evaluate by maximizing an acquisition function
     that measures model uncertainty (here, the posterior standard deviation,
     which corresponds to an "uncertainty sampling" strategy).
  3. Evaluate the true function at that point and add the result to our data.

After the loop we plot the model's predictions vs. the true function.
"""

import torch
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
from botorch.test_functions import Branin
from botorch.acquisition.analytic import PosteriorStandardDeviation
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood

# ── Settings ──────────────────────────────────────────────────────────────
tkwargs = {"dtype": torch.double, "device": "cpu"}
N_INITIAL = 5       # initial random observations
N_ITERATIONS = 15   # active-learning steps
SEED = 42

torch.manual_seed(SEED)

# ── Ground-truth function (Branin, 2-D) ──────────────────────────────────
branin = Branin(negate=False).to(**tkwargs)
bounds = branin.bounds.to(**tkwargs)           # shape (2, 2)

def evaluate(X):
    """Evaluate Branin (returns shape (n, 1))."""
    return branin(X).unsqueeze(-1)

# ── Initial data (random Latin-hypercube-ish points) ─────────────────────
train_X = bounds[0] + (bounds[1] - bounds[0]) * torch.rand(N_INITIAL, 2, **tkwargs)
train_Y = evaluate(train_X)

# Fixed near-zero observation noise: the objective is deterministic.
# Omit train_Yvar to let the GP estimate the noise variance from data
# (useful when observations are noisy).
TRAIN_YVAR = torch.full_like(train_Y, 1e-6)

print(f"{'Iter':>4}  {'New x1':>8}  {'New x2':>8}  {'f(x)':>10}  {'Model RMSE':>10}")
print("-" * 52)

# ── Active-learning loop ─────────────────────────────────────────────────
for i in range(N_ITERATIONS):
    # Fit GP – Normalize and Standardize let BoTorch handle input/output
    # scaling automatically; we work in the original input space everywhere.
    gp = SingleTaskGP(
        train_X,
        train_Y,
        train_Yvar=TRAIN_YVAR,
        input_transform=Normalize(d=2, bounds=bounds),
        outcome_transform=Standardize(m=1),
    ).to(**tkwargs)
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    fit_gpytorch_mll(mll)

    # Acquisition: maximize posterior std dev (uncertainty sampling)
    acqf = PosteriorStandardDeviation(model=gp)
    candidate, acq_value = optimize_acqf(
        acq_function=acqf,
        bounds=bounds,
        q=1,
        num_restarts=8,
        raw_samples=128,
    )

    new_X = candidate
    new_Y = evaluate(new_X)

    # Compute model RMSE on a test grid
    with torch.no_grad():
        test_X = bounds[0] + (bounds[1] - bounds[0]) * torch.rand(500, 2, **tkwargs)
        pred = gp.posterior(test_X).mean
        rmse = (pred - evaluate(test_X)).pow(2).mean().sqrt().item()

    print(
        f"{i + 1:4d}  {new_X[0, 0]:8.3f}  {new_X[0, 1]:8.3f}  "
        f"{new_Y.item():10.3f}  {rmse:10.3f}"
    )

    # Augment training data
    train_X = torch.cat([train_X, new_X])
    train_Y = torch.cat([train_Y, new_Y])
    TRAIN_YVAR = torch.full_like(train_Y, 1e-6)

print(f"\nTotal evaluations: {train_X.shape[0]}  (initial {N_INITIAL} + {N_ITERATIONS} AL steps)")
print("Done.")
