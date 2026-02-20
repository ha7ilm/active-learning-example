"""
Dash app to visualize active learning with BoTorch / SingleTaskGP.

Precomputes the GP posterior (mean + std-dev) on a dense grid at every AL
iteration, then lets you scrub through iterations with a slider.

Layout
------
 Row 1:  True Branin  |  GP Mean  |  GP Std Dev
 Row 2:  Slider to pick iteration
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

import dash
from dash import dcc, html, Input, Output
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── AL settings ───────────────────────────────────────────────────────────
tkwargs = {"dtype": torch.double, "device": "cpu"}
N_INITIAL = 5
N_ITERATIONS = 15
GRID_RES = 80
SEED = 42

torch.manual_seed(SEED)

# ── Ground truth ──────────────────────────────────────────────────────────
branin = Branin(negate=False).to(**tkwargs)
bounds = branin.bounds.to(**tkwargs)

x1 = torch.linspace(bounds[0, 0].item(), bounds[1, 0].item(), GRID_RES, **tkwargs)
x2 = torch.linspace(bounds[0, 1].item(), bounds[1, 1].item(), GRID_RES, **tkwargs)
X1, X2 = torch.meshgrid(x1, x2, indexing="ij")
grid = torch.stack([X1.reshape(-1), X2.reshape(-1)], dim=-1)  # (GRID_RES^2, 2)

true_Y = branin(grid).reshape(GRID_RES, GRID_RES).detach().numpy()
x1_np, x2_np = x1.numpy(), x2.numpy()


def evaluate(X):
    return branin(X).unsqueeze(-1)


# ── Run AL loop and cache snapshots ──────────────────────────────────────
print("Running active-learning loop …")

train_X = bounds[0] + (bounds[1] - bounds[0]) * torch.rand(N_INITIAL, 2, **tkwargs)
train_Y = evaluate(train_X)

# Fixed near-zero observation noise: the objective is deterministic.
# Omit train_Yvar to let the GP estimate the noise variance from data
# (useful when observations are noisy).
train_Yvar = torch.full_like(train_Y, 1e-6)

snapshots = []  # one dict per iteration (including iteration 0 = before any AL)

for i in range(N_ITERATIONS + 1):
    # Normalize and Standardize let BoTorch handle input/output scaling
    # automatically; we work in the original input space everywhere.
    gp = SingleTaskGP(
        train_X,
        train_Y,
        train_Yvar=train_Yvar,
        input_transform=Normalize(d=2, bounds=bounds),
        outcome_transform=Standardize(m=1),
    ).to(**tkwargs)
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    fit_gpytorch_mll(mll)

    with torch.no_grad():
        posterior = gp.posterior(grid)
        mean = posterior.mean.squeeze(-1).numpy().reshape(GRID_RES, GRID_RES)
        std = posterior.variance.squeeze(-1).sqrt().numpy().reshape(GRID_RES, GRID_RES)

    snapshots.append(
        {
            "mean": mean,
            "std": std,
            "train_X": train_X.numpy().copy(),
            "train_Y": train_Y.numpy().copy(),
        }
    )
    print(f"  iteration {i}/{N_ITERATIONS}")

    if i == N_ITERATIONS:
        break

    # Pick next point
    acqf = PosteriorStandardDeviation(model=gp)
    candidate, _ = optimize_acqf(
        acq_function=acqf,
        bounds=bounds,
        q=1,
        num_restarts=8,
        raw_samples=128,
    )
    new_X = candidate
    new_Y = evaluate(new_X)
    train_X = torch.cat([train_X, new_X])
    train_Y = torch.cat([train_Y, new_Y])
    train_Yvar = torch.full_like(train_Y, 1e-6)

print("Done. Starting Dash server …\n")

# ── Shared colour ranges (fixed across iterations for easier comparison) ─
mean_zmin = min(s["mean"].min() for s in snapshots)
mean_zmax = max(s["mean"].max() for s in snapshots)
std_zmax = max(s["std"].max() for s in snapshots)

# ── Dash app ──────────────────────────────────────────────────────────────
app = dash.Dash(__name__)

app.layout = html.Div(
    style={"fontFamily": "system-ui, sans-serif", "maxWidth": "1400px", "margin": "0 auto",
           "padding": "20px"},
    children=[
        html.H2("Active Learning with SingleTaskGP"),
        html.Div(
            style={"display": "flex", "alignItems": "center", "gap": "16px",
                   "marginBottom": "8px"},
            children=[
                html.Span("Iteration:", style={"fontWeight": "bold"}),
                html.Span(id="iter-label"),
                html.Span(id="n-points-label", style={"color": "#666"}),
            ],
        ),
        dcc.Slider(
            id="iter-slider",
            min=0,
            max=N_ITERATIONS,
            step=1,
            value=0,
            marks={i: str(i) for i in range(N_ITERATIONS + 1)},
        ),
        dcc.Graph(id="main-graph", style={"height": "560px"}),
    ],
)


@app.callback(
    Output("main-graph", "figure"),
    Output("iter-label", "children"),
    Output("n-points-label", "children"),
    Input("iter-slider", "value"),
)
def update(iteration):
    snap = snapshots[iteration]
    n_pts = snap["train_X"].shape[0]

    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=["True Branin", "GP Posterior Mean", "GP Posterior Std Dev"],
        horizontal_spacing=0.06,
    )

    common_contour = dict(x=x1_np, y=x2_np, line_width=0)
    scatter_kw = dict(
        mode="markers",
        marker=dict(size=6, color="white", line=dict(width=1, color="black")),
        showlegend=False,
    )

    # -- True function --
    fig.add_trace(
        go.Contour(z=true_Y.T, zmin=mean_zmin, zmax=mean_zmax,
                    colorscale="Viridis", showscale=False, **common_contour),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(x=snap["train_X"][:, 0], y=snap["train_X"][:, 1], **scatter_kw),
        row=1, col=1,
    )

    # -- GP mean --
    fig.add_trace(
        go.Contour(z=snap["mean"].T, zmin=mean_zmin, zmax=mean_zmax,
                    colorscale="Viridis", colorbar=dict(x=0.66, len=0.9),
                    **common_contour),
        row=1, col=2,
    )
    # highlight most-recently added point
    if iteration > 0:
        fig.add_trace(
            go.Scatter(
                x=snap["train_X"][:-1, 0], y=snap["train_X"][:-1, 1], **scatter_kw,
            ),
            row=1, col=2,
        )
        fig.add_trace(
            go.Scatter(
                x=[snap["train_X"][-1, 0]],
                y=[snap["train_X"][-1, 1]],
                mode="markers",
                marker=dict(size=10, color="red", symbol="x",
                            line=dict(width=2, color="darkred")),
                showlegend=False,
            ),
            row=1, col=2,
        )
    else:
        fig.add_trace(
            go.Scatter(x=snap["train_X"][:, 0], y=snap["train_X"][:, 1], **scatter_kw),
            row=1, col=2,
        )

    # -- GP std dev --
    fig.add_trace(
        go.Contour(z=snap["std"].T, zmin=0, zmax=std_zmax,
                    colorscale="Inferno", colorbar=dict(x=1.0, len=0.9),
                    **common_contour),
        row=1, col=3,
    )
    fig.add_trace(
        go.Scatter(x=snap["train_X"][:, 0], y=snap["train_X"][:, 1], **scatter_kw),
        row=1, col=3,
    )
    # next query shown as red X on the std-dev panel too
    if iteration > 0:
        fig.add_trace(
            go.Scatter(
                x=[snap["train_X"][-1, 0]],
                y=[snap["train_X"][-1, 1]],
                mode="markers",
                marker=dict(size=10, color="red", symbol="x",
                            line=dict(width=2, color="darkred")),
                showlegend=False,
            ),
            row=1, col=3,
        )

    fig.update_layout(
        margin=dict(t=40, b=20, l=40, r=40),
        template="plotly_white",
    )
    for c in range(1, 4):
        fig.update_xaxes(title_text="x₁", row=1, col=c)
        fig.update_yaxes(title_text="x₂", row=1, col=c)

    return (
        fig,
        f"{iteration}",
        f"({n_pts} training points)",
    )


if __name__ == "__main__":
    app.run(debug=False, port=8050)
