"""
Dash app to visualize the active learning experiment.

Imports pre-computed snapshots from active_learning_algorithm.py and lets
the user explore the GP state at every iteration via a slider.

Two view modes:
  - 2D (default): three contour plots — True Branin | GP Mean | GP Std Dev
  - 3D (checkbox): two side-by-side 3D surfaces — True Branin | GP + bounds

Run:
    python active_learning_dash_app.py
"""

import dash
from dash import dcc, html, Input, Output
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# All experiment data comes from the algorithm module.
from active_learning_algorithm import (
    N_ITERATIONS,
    snapshots,
    true_Y,
    x1_np,
    x2_np,
    X1_np,
    X2_np,
    mean_zmin,
    mean_zmax,
    std_zmax,
    zaxis_range,
)


# ── Figure builders ───────────────────────────────────────────────────────

def build_2d_figure(snap, iteration):
    """Three-panel contour plot: true function, GP mean, GP std dev."""
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=["True Branin", "GP Posterior Mean", "GP Posterior Std Dev"],
        horizontal_spacing=0.06,
    )

    common_contour = dict(x=x1_np, y=x2_np, line_width=0)
    scatter_kw = dict(
        mode="markers",
        marker=dict(size=6, color="white", line=dict(width=1, color="black")),
        showlegend=False,
    )

    # True function
    fig.add_trace(
        go.Contour(z=true_Y.T, zmin=mean_zmin, zmax=mean_zmax,
                    colorscale="Viridis", showscale=False, **common_contour),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(x=snap["train_X"][:, 0], y=snap["train_X"][:, 1], **scatter_kw),
        row=1, col=1,
    )

    # GP mean
    fig.add_trace(
        go.Contour(z=snap["mean"].T, zmin=mean_zmin, zmax=mean_zmax,
                    colorscale="Viridis", colorbar=dict(x=0.66, len=0.9),
                    **common_contour),
        row=1, col=2,
    )
    if iteration > 0:
        fig.add_trace(
            go.Scatter(x=snap["train_X"][:-1, 0], y=snap["train_X"][:-1, 1],
                       **scatter_kw),
            row=1, col=2,
        )
        fig.add_trace(
            go.Scatter(x=[snap["train_X"][-1, 0]], y=[snap["train_X"][-1, 1]],
                       mode="markers",
                       marker=dict(size=10, color="red", symbol="x",
                                   line=dict(width=2, color="darkred")),
                       showlegend=False),
            row=1, col=2,
        )
    else:
        fig.add_trace(
            go.Scatter(x=snap["train_X"][:, 0], y=snap["train_X"][:, 1], **scatter_kw),
            row=1, col=2,
        )

    # GP std dev
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
    if iteration > 0:
        fig.add_trace(
            go.Scatter(x=[snap["train_X"][-1, 0]], y=[snap["train_X"][-1, 1]],
                       mode="markers",
                       marker=dict(size=10, color="red", symbol="x",
                                   line=dict(width=2, color="darkred")),
                       showlegend=False),
            row=1, col=3,
        )

    fig.update_layout(
        margin=dict(t=40, b=20, l=40, r=40),
        template="plotly_white",
        height=560,
    )
    for c in range(1, 4):
        fig.update_xaxes(title_text="x\u2081", row=1, col=c)
        fig.update_yaxes(title_text="x\u2082", row=1, col=c)

    return fig


def _add_training_points_3d(fig, snap, iteration, col):
    """Add observation markers to a 3D subplot at the given column."""
    tx, ty = snap["train_X"], snap["train_Y"]
    show = (col == 1)  # only show legend entries once
    if iteration > 0:
        fig.add_trace(go.Scatter3d(
            x=tx[:-1, 0], y=tx[:-1, 1], z=ty[:-1, 0],
            mode="markers",
            marker=dict(size=4, color="white", line=dict(width=1, color="black")),
            name="Observations", showlegend=show, legendgroup="obs",
        ), row=1, col=col)
        fig.add_trace(go.Scatter3d(
            x=[tx[-1, 0]], y=[tx[-1, 1]], z=[ty[-1, 0]],
            mode="markers",
            marker=dict(size=6, color="red", symbol="x",
                        line=dict(width=1, color="darkred")),
            name="Latest query", showlegend=show, legendgroup="latest",
        ), row=1, col=col)
    else:
        fig.add_trace(go.Scatter3d(
            x=tx[:, 0], y=tx[:, 1], z=ty[:, 0],
            mode="markers",
            marker=dict(size=4, color="white", line=dict(width=1, color="black")),
            name="Observations", showlegend=show, legendgroup="obs",
        ), row=1, col=col)


def build_3d_figure(snap, iteration):
    """Two side-by-side 3D scenes: true Branin and GP posterior."""
    mean = snap["mean"]
    std = snap["std"]

    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "scene"}, {"type": "scene"}]],
        subplot_titles=["True Branin", "GP Posterior (mean \u00b1 2\u03c3)"],
        horizontal_spacing=0.02,
    )

    # Left: true Branin surface
    fig.add_trace(go.Surface(
        x=X1_np, y=X2_np, z=true_Y,
        colorscale="Viridis", cmin=mean_zmin, cmax=mean_zmax,
        opacity=0.9, name="True Branin", showlegend=True, showscale=False,
    ), row=1, col=1)
    _add_training_points_3d(fig, snap, iteration, col=1)

    # Right: GP mean
    fig.add_trace(go.Surface(
        x=X1_np, y=X2_np, z=mean,
        colorscale="Viridis", cmin=mean_zmin, cmax=mean_zmax,
        opacity=0.9, name="GP mean", showlegend=True,
        colorbar=dict(title="f(x)", x=1.02),
    ), row=1, col=2)

    # Right: +2 std confidence surface
    fig.add_trace(go.Surface(
        x=X1_np, y=X2_np, z=mean + 2 * std,
        colorscale=[[0, "rgba(100,100,255,0.25)"], [1, "rgba(100,100,255,0.25)"]],
        showscale=False, opacity=0.5,
        name="+2\u03c3", showlegend=True,
    ), row=1, col=2)

    # Right: -2 std confidence surface
    fig.add_trace(go.Surface(
        x=X1_np, y=X2_np, z=mean - 2 * std,
        colorscale=[[0, "rgba(100,100,255,0.25)"], [1, "rgba(100,100,255,0.25)"]],
        showscale=False, opacity=0.5,
        name="\u22122\u03c3", showlegend=True,
    ), row=1, col=2)

    _add_training_points_3d(fig, snap, iteration, col=2)

    scene_opts = dict(
        xaxis_title="x\u2081",
        yaxis_title="x\u2082",
        zaxis_title="f(x)",
        zaxis=dict(range=zaxis_range),
    )
    fig.update_layout(
        scene=scene_opts,
        scene2=scene_opts,
        margin=dict(t=30, b=10, l=10, r=10),
        template="plotly_white",
        height=640,
        legend=dict(x=0.01, y=0.99),
    )

    return fig


# ── Dash app ──────────────────────────────────────────────────────────────
app = dash.Dash(__name__)

app.layout = html.Div(
    style={"fontFamily": "system-ui, sans-serif", "maxWidth": "1400px",
           "margin": "0 auto", "padding": "20px"},
    children=[
        # Title row with 3D checkbox
        html.Div(
            style={"display": "flex", "alignItems": "center", "gap": "16px"},
            children=[
                html.H2("Active Learning with BoTorch's SingleTaskGP",
                         style={"margin": "0"}),
                dcc.Checklist(
                    id="view-3d",
                    options=[{"label": " in 3D view", "value": "3d"}],
                    value=[],
                    style={"fontSize": "16px"},
                ),
            ],
        ),
        html.Div(style={"height": "12px"}),
        # Iteration controls
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
            min=0, max=N_ITERATIONS, step=1, value=0,
            marks={i: str(i) for i in range(N_ITERATIONS + 1)},
        ),
        dcc.Graph(id="main-graph"),
        # Educational description
        html.Div(
            style={"marginTop": "24px", "padding": "16px 20px",
                   "backgroundColor": "#f7f7f8", "borderRadius": "8px",
                   "lineHeight": "1.6", "color": "#333", "fontSize": "14px"},
            children=[
                html.H3("What is active learning?",
                         style={"marginTop": "0", "fontSize": "16px"}),
                html.P([
                    "Active learning is a strategy where the model ",
                    html.B("chooses which data to collect next"),
                    " rather than relying on a pre-determined dataset. "
                    "The idea is simple: some observations are more informative "
                    "than others, so we should spend our evaluation budget where "
                    "it matters most."
                ]),
                html.H3("How it works here",
                         style={"fontSize": "16px"}),
                html.P([
                    "We fit a Gaussian Process (SingleTaskGP) to the data we have "
                    "so far. The GP gives us not just a prediction (the ",
                    html.B("posterior mean"),
                    ") but also a measure of uncertainty (the ",
                    html.B("posterior standard deviation"),
                    ") at every point in the input space."
                ]),
                html.P([
                    "At each iteration the acquisition function picks the point "
                    "where the model is ",
                    html.B("most uncertain"),
                    " \u2014 this is called ",
                    html.I("uncertainty sampling"),
                    ". We evaluate the true function there and add the result to "
                    "our training set."
                ]),
                html.H3("What to look for",
                         style={"fontSize": "16px"}),
                html.Ul([
                    html.Li([
                        html.B("GP Mean panel: "),
                        "should progressively look more like the True Branin panel "
                        "as iterations increase."
                    ]),
                    html.Li([
                        html.B("GP Std Dev panel: "),
                        "starts with large uncertainty everywhere and shrinks as "
                        "new observations fill in the space. Notice that new queries "
                        "(red X) always land in the brightest (most uncertain) regions."
                    ]),
                    html.Li([
                        html.B("3D view: "),
                        "the translucent confidence band (\u00b12\u03c3) should "
                        "tighten around the GP mean surface as data accumulates, "
                        "especially near observed points."
                    ]),
                ]),
            ],
        ),
    ],
)


@app.callback(
    Output("main-graph", "figure"),
    Output("iter-label", "children"),
    Output("n-points-label", "children"),
    Input("iter-slider", "value"),
    Input("view-3d", "value"),
)
def update(iteration, view_3d):
    snap = snapshots[iteration]
    n_pts = snap["train_X"].shape[0]
    is_3d = "3d" in (view_3d or [])

    fig = build_3d_figure(snap, iteration) if is_3d else build_2d_figure(snap, iteration)

    return fig, f"{iteration}", f"({n_pts} training points)"


if __name__ == "__main__":
    app.run(debug=False, port=8050)
