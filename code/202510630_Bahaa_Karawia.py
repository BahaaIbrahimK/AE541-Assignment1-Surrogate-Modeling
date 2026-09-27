"""
AE 541 - Digital Twin in Modern Engineering
Assignment 1: Surrogate model development for two-variable test functions

Author : Bahaa Karawia (KFUPM ID 202510630)
Part A : Rosenbrock function (common)
Part B : Trid function, 2 variables (assigned)

The workflow follows the course reference notebook
(himmelblau_surrogate_tutorial.ipynb); the LHS, cubic RBF and GP set-up are
adapted from it with acknowledgment.

Run:   python 202510630_Bahaa_Karawia.py
Needs: numpy, scipy, matplotlib, scikit-learn, pandas (versions in requirements.txt).
Repository: https://github.com/BahaaIbrahimK/AE541-Assignment1-Surrogate-Modeling
Every figure and table in the report is written to ./outputs next to this file.
"""

import json
import platform
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")                     # write files only, no GUI window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import sklearn
from matplotlib.patches import Rectangle
from matplotlib.colors import LogNorm
from matplotlib.ticker import LogLocator, MaxNLocator, ScalarFormatter
from scipy.optimize import minimize
from scipy.spatial.distance import cdist, pdist
from scipy.stats import spearmanr
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_predict, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


# =============================== CONFIGURATION ===============================
SEED = 7                      # master seed; every random draw derives from it
SEED_DOE_STUDY = SEED + 1     # repeated designs for the random-vs-LHS comparison
SEED_STARTS = SEED + 2        # start points of the multi-start optimisation
SEED_SUB_BOX = SEED + 3       # design inside the reduced (extrapolation) box
SEED_STRIDE = 1000            # learning-curve draw r at size N uses SEED + 1000 r + N
N_SAMPLES = 60                # main DOE budget = training + held-out test
TEST_FRACTION = 0.25          # 45 training / 15 test points
CV_FOLDS = 5                  # K-fold CV, applied to the training set only
POLY_DEGREES = range(1, 9)    # degree sweep for the polynomial surrogate
PARSIMONY_TOL = 0.05          # degrees within 5 % of the best CV error count as tied
ROUNDOFF_REL = 1e-9           # CV RMSE below this x std(y_train) counts as exact
MAXIMIN_CANDIDATES = 100      # LHS candidates screened for the maximin design
DOE_REALISATIONS = 500        # repeated designs used to compare the DOE criteria
GRID_N = 120                  # dense truth grid, GRID_N x GRID_N
GP_RESTARTS = 8               # restarts of the marginal-likelihood optimiser
GP_SIGNAL_MIN = 1e-3          # bounds on sigma_f^2 (outputs are normalised)
GP_SIGNAL_MAX = 1e4
GP_SIGNAL_TRIALS = [1e4, 1e6, 1e8]  # upper bounds tried when sigma_f^2 hits its limit
GP_LENGTH_BOUNDS = (1e-2, 1e3)      # length scales, in standardised input units
GP_NOISE_BOUNDS = (1e-12, 1e-2)     # sigma_n^2; tiny because the data are noiseless
GP_INIT = {"signal": 1.0, "length": [1.0, 1.0], "noise": 1e-8}   # optimiser start
GP_ALPHA = 1e-10              # diagonal jitter for numerical stability (sklearn default);
                              # together with sigma_n^2 it sets a floor under sigma_hat
GP_BOUND_TOL = 1e-2           # |log theta - log bound| below this counts as "at bound"
CORE_FRAC = 0.75              # central fraction of the box used for "core" errors
LEARNING_N = [20, 30, 45, 65, 90]   # training sizes for the learning curve
LEARNING_REPEATS = 5          # independent LHS draws per training size
N_STARTS = 20                 # multi-start local searches on each surrogate
NM_XATOL, NM_FATOL = 1e-9, 1e-12    # Nelder-Mead stopping tolerances
NM_MAXITER = 5000
CLUSTER_FRAC = 0.02           # minima closer than this x box width are merged
BOUNDARY_TOL = 1e-6           # an optimum this close to a bound is flagged
SCATTER_POINTS = 3000         # grid points shown in the sigma-vs-error scatter
UNDERSHOOT_TOL = 1e-3         # surrogate values below min(f) by more than this x range
                              # of f are drawn in red on the field maps
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"


def rosenbrock(x1, x2):
    return 100.0 * (x2 - x1**2) ** 2 + (1.0 - x1) ** 2


def trid(x1, x2):
    return (x1 - 1.0) ** 2 + (x2 - 1.0) ** 2 - x1 * x2


FUNCTIONS = {
    "rosenbrock": {
        "name": "Rosenbrock",
        "func": rosenbrock,
        "bounds": np.array([[-2.0, 2.0], [-2.0, 2.0]]),
        "minima": np.array([[1.0, 1.0]]),
        "f_min": 0.0,
        "sub_box": np.array([[-1.0, 1.0], [-1.0, 1.0]]),
        # expanded form: 100x1^4 - 200x1^2x2 + 100x2^2 + x1^2 - 2x1 + 1
        "exact_coefs": {"1": 1.0, "x1": -2.0, "x1^2": 1.0, "x2^2": 100.0,
                        "x1^2 x2": -200.0, "x1^4": 100.0},
        "line_levels": [0.5, 3, 15, 60, 250, 1000, 2500],
        "view": (28, -118),
    },
    "trid": {
        "name": "Trid",
        "func": trid,
        "bounds": np.array([[-4.0, 4.0], [-4.0, 4.0]]),
        "minima": np.array([[2.0, 2.0]]),
        "f_min": -2.0,
        "sub_box": np.array([[-2.0, 2.0], [-2.0, 2.0]]),
        # expanded form: x1^2 + x2^2 - x1x2 - 2x1 - 2x2 + 2
        "exact_coefs": {"1": 2.0, "x1": -2.0, "x2": -2.0, "x1^2": 1.0,
                        "x1 x2": -1.0, "x2^2": 1.0},
        "line_levels": [-1.5, 0, 3, 8, 15, 25, 38],
        "view": (28, -130),
    },
}

MODEL_NAMES = ["Polynomial", "Cubic RBF", "Gaussian process"]
MODEL_COLORS = {"Polynomial": "#2a78d6", "Cubic RBF": "#eb6834",
                "Gaussian process": "#1baf7a"}
MODEL_MARKERS = {"Polynomial": "o", "Cubic RBF": "s", "Gaussian process": "D"}
# =============================================================================


def set_plot_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["STIXGeneral", "Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 9,
        "axes.labelsize": 9.5,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 8,
        "axes.linewidth": 0.8,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "lines.linewidth": 1.5,
        "lines.markersize": 5,
        "legend.framealpha": 0.92,
        "legend.edgecolor": "0.75",
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03,
    })


def save_figure(fig, folder, name):
    folder.mkdir(parents=True, exist_ok=True)
    fig.savefig(folder / f"{name}.pdf")
    fig.savefig(folder / f"{name}.png")
    plt.close(fig)


def panel_letter(ax, letter, three_d=False):
    # letters sit just outside the top-left corner so they never cover data
    kw = dict(fontsize=10, fontweight="bold")
    if three_d:
        ax.text2D(0.0, 0.97, f"({letter})", transform=ax.transAxes, **kw)
    else:
        ax.text(-0.02, 1.02, f"({letter})", transform=ax.transAxes,
                va="bottom", ha="right", **kw)


def field_levels(Y):
    """Contour levels for f and f_hat maps, plus a colour map that flags undershoot.

    The lower limit sits just below min(f), so round-off never triggers the
    flag, while a surrogate that predicts clearly impossible values shows up red.
    """
    lo, hi = Y.min(), Y.max()
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_under("#e34948")
    return np.linspace(lo - UNDERSHOOT_TOL * (hi - lo), hi, 61), cmap


def latex_sci(v, digits=2):
    mantissa, exponent = f"{v:.{digits}e}".split("e")
    return rf"{mantissa}\times10^{{{int(exponent)}}}"


def sci_colorbar(fig, mappable, ax, label):
    cb = fig.colorbar(mappable, ax=ax, pad=0.02, fraction=0.05)
    lo, hi = mappable.norm.vmin, mappable.norm.vmax
    # keep ticks inside the data range, otherwise extend="both" colour bars
    # get a tick drawn on the arrow that collides with its neighbour
    cb.set_ticks([t for t in MaxNLocator(6).tick_values(lo, hi) if lo <= t <= hi])
    fmt = ScalarFormatter(useMathText=True)
    fmt.set_powerlimits((-3, 4))
    cb.formatter = fmt
    cb.update_ticks()
    cb.set_label(label)
    return cb


# ----------------------------- design of experiments -----------------------------

def latin_hypercube(n, d, rng):
    """Classic LHS in the unit cube: one point per bin on every axis."""
    bins = np.column_stack([rng.permutation(n) for _ in range(d)])
    return (bins + rng.random((n, d))) / n


def uniform_random(n, d, rng):
    return rng.random((n, d))


def maximin_lhs(n, d, rng, n_candidates):
    """Best of several LHS draws by the maximin criterion.

    Generating a design costs no function evaluations, so screening candidates
    improves the spread for free while keeping the LHS stratification.
    """
    best, best_score = None, -np.inf
    for _ in range(n_candidates):
        U = latin_hypercube(n, d, rng)
        score = pdist(U).min()
        if score > best_score:
            best, best_score = U, score
    return best


def to_box(U, bounds):
    return bounds[:, 0] + U * (bounds[:, 1] - bounds[:, 0])


def min_distance(U):
    return pdist(U).min()


def corner_gaps(X_train, bounds):
    """Distance from each corner of the box to its nearest training point."""
    corners = np.array([[bounds[0, i], bounds[1, j]] for i in (0, 1) for j in (0, 1)])
    return {f"({c[0]:g}, {c[1]:g})": float(np.min(np.linalg.norm(X_train - c, axis=1)))
            for c in corners}


def compare_designs(rng):
    """Distribution of the maximin criterion over many realisations, in [0,1]^2."""
    scores = {"Uniform random": [], "LHS": []}
    for _ in range(DOE_REALISATIONS):
        scores["Uniform random"].append(min_distance(uniform_random(N_SAMPLES, 2, rng)))
        scores["LHS"].append(min_distance(latin_hypercube(N_SAMPLES, 2, rng)))
    return {k: np.array(v) for k, v in scores.items()}


# ----------------------------------- surrogates -----------------------------------

class CubicRBF(RegressorMixin, BaseEstimator):
    """Cubic RBF interpolant with a linear polynomial tail.

    f_hat(x) = sum_i w_i ||x - x_i||^3 + c0 + c1 x1 + c2 x2, with sum w_i = 0 and
    sum w_i x_i = 0 so that the weights carry no linear trend of their own.
    """

    def fit(self, X, y):
        X, y = np.asarray(X, float), np.asarray(y, float)
        n, d = X.shape
        Phi = cdist(X, X) ** 3
        P = np.hstack([np.ones((n, 1)), X])
        A = np.block([[Phi, P], [P.T, np.zeros((d + 1, d + 1))]])
        sol = np.linalg.solve(A, np.concatenate([y, np.zeros(d + 1)]))
        self.centres_, self.weights_, self.tail_ = X, sol[:n], sol[n:]
        self.condition_number_ = np.linalg.cond(A)
        return self

    def predict(self, X):
        X = np.asarray(X, float)
        P = np.hstack([np.ones((len(X), 1)), X])
        return cdist(X, self.centres_) ** 3 @ self.weights_ + P @ self.tail_


def make_polynomial(degree):
    # scaling sits inside the pipeline, so every fit (including each CV fold)
    # learns mu and sigma from its own training rows only - no leakage
    return make_pipeline(StandardScaler(),
                         PolynomialFeatures(degree, include_bias=False),
                         LinearRegression())


def make_rbf():
    return make_pipeline(StandardScaler(), CubicRBF())


def make_gp(seed=SEED, signal_max=GP_SIGNAL_MAX):
    kernel = (ConstantKernel(GP_INIT["signal"], (GP_SIGNAL_MIN, signal_max))
              * RBF(length_scale=GP_INIT["length"], length_scale_bounds=GP_LENGTH_BOUNDS)
              + WhiteKernel(GP_INIT["noise"], GP_NOISE_BOUNDS))
    return make_pipeline(StandardScaler(),
                         GaussianProcessRegressor(kernel=kernel, normalize_y=True, alpha=GP_ALPHA,
                                                  n_restarts_optimizer=GP_RESTARTS,
                                                  random_state=seed))


def make_model(name, degree):
    if name == "Polynomial":
        return make_polynomial(degree)
    if name == "Cubic RBF":
        return make_rbf()
    return make_gp()


def n_terms(p):
    return (p + 1) * (p + 2) // 2


def gp_bound_sensitivity(X_train, y_train):
    """Refit the GP with wider sigma_f^2 bounds, judged by training CV only.

    A wider bound still contains the old optimum, so a lower likelihood after
    widening means the optimiser failed, not that the model got worse; the
    abnormal terminations are counted to make that visible.
    """
    rows = []
    for smax in GP_SIGNAL_TRIALS:
        m = make_gp(signal_max=smax)
        n_abnormal = fit_gp_counting_failures(m, X_train, y_train)
        hp = gp_hyperparameters(m)
        rows.append({"signal_max": smax,
                     "signal_variance": hp["signal_variance_normalised"],
                     "ell1_phys": hp["length_scale_phys"][0],
                     "ell2_phys": hp["length_scale_phys"][1],
                     "noise_variance": hp["noise_variance_normalised"],
                     "log_marginal_likelihood": hp["log_marginal_likelihood"],
                     "CV_RMSE": cv_rmse(make_gp(signal_max=smax), X_train, y_train),
                     "abnormal_terminations": n_abnormal,
                     "optimizer_runs": GP_RESTARTS + 1,
                     "at_bounds": ";".join(hp["parameters_at_bounds"])})
    return pd.DataFrame(rows)


def gp_hyperparameters(gp_pipeline):
    """Fitted GP kernel parameters, with length scales in physical units too."""
    gp = gp_pipeline[-1]
    scaler = gp_pipeline[0]
    k = gp.kernel_
    ell_std = np.atleast_1d(k.k1.k2.length_scale)
    at_bound = []
    for hp, theta in zip(k.hyperparameters, k.theta):
        lo, hi = np.log(hp.bounds[0])
        if np.isclose(theta, lo, atol=GP_BOUND_TOL) or np.isclose(theta, hi, atol=GP_BOUND_TOL):
            at_bound.append(hp.name)
    return {
        "signal_variance_normalised": float(k.k1.k1.constant_value),
        "length_scale_std": ell_std.tolist(),
        "length_scale_phys": (ell_std * scaler.scale_).tolist(),
        "noise_variance_normalised": float(k.k2.noise_level),
        "log_marginal_likelihood": float(gp.log_marginal_likelihood_value_),
        "parameters_at_bounds": at_bound,
    }


# ----------------------------------- validation -----------------------------------

def metrics(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    y_range = y_true.max() - y_true.min()
    return {"RMSE": rmse,
            "MAE": float(mean_absolute_error(y_true, y_pred)),
            "R2": float(r2_score(y_true, y_pred)),
            "MaxAE": float(np.max(np.abs(y_true - y_pred))),
            "NRMSE": rmse / y_range if y_range > 0 else float("nan")}


def cv_rmse(model, X, y):
    """Pooled out-of-fold RMSE; cross_val_predict refits the whole pipeline per fold."""
    folds = KFold(CV_FOLDS, shuffle=True, random_state=SEED)
    with warnings.catch_warnings():
        # GP bound warnings are checked explicitly on the final model instead
        warnings.simplefilter("ignore", ConvergenceWarning)
        pred = cross_val_predict(model, X, y, cv=folds)
    return float(np.sqrt(mean_squared_error(y, pred)))


def design_matrix_condition(X_train, degree):
    Z = StandardScaler().fit_transform(X_train)
    return float(np.linalg.cond(PolynomialFeatures(degree).fit_transform(Z)))


def degree_sweep(X_train, y_train):
    """Training-only quantities for each degree; nothing here sees the test set."""
    rows = []
    fold_size = int(len(X_train) * (CV_FOLDS - 1) / CV_FOLDS)
    for p in POLY_DEGREES:
        m = make_polynomial(p).fit(X_train, y_train)
        rows.append({
            "degree": p,
            "n_terms": n_terms(p),
            "points_per_fold": fold_size,
            "cond_design": design_matrix_condition(X_train, p),
            "train_RMSE": metrics(y_train, m.predict(X_train))["RMSE"],
            "CV_RMSE": cv_rmse(make_polynomial(p), X_train, y_train),
        })
    return pd.DataFrame(rows)


def select_degree(sweep, y_train):
    """Smallest degree whose CV error ties with the best one, or is at round-off.

    With noiseless polynomial data every degree at or above the true one gives
    round-off error, and the argmin among those is decided by noise. Preferring
    the simplest adequate degree is the parsimony rule.
    """
    threshold = max((1 + PARSIMONY_TOL) * sweep["CV_RMSE"].min(),
                    ROUNDOFF_REL * np.std(y_train))
    return int(sweep.loc[sweep["CV_RMSE"] <= threshold, "degree"].min())


def add_sweep_diagnostics(sweep, X_train, y_train, X_test, y_test, X_grid, y_grid):
    """Test and dense-grid error per degree, added only after the degree is locked."""
    test, grid = [], []
    for p in sweep["degree"]:
        m = make_polynomial(p).fit(X_train, y_train)
        test.append(metrics(y_test, m.predict(X_test))["RMSE"])
        grid.append(metrics(y_grid, m.predict(X_grid))["RMSE"])
    return sweep.assign(test_RMSE=test, grid_RMSE=grid)


def polynomial_readback(model, degree, X_grid):
    """Coefficients of the fitted surrogate in the raw monomials of x.

    An affine input scaling keeps the polynomial degree, so f_hat is an exact
    degree-p polynomial in x and projecting it on raw monomials recovers them.
    """
    pf = PolynomialFeatures(degree)
    Phi = pf.fit_transform(X_grid)
    coef, *_ = np.linalg.lstsq(Phi, model.predict(X_grid), rcond=None)
    return dict(zip(pf.get_feature_names_out(["x1", "x2"]), coef))


def quadratic_stationary_point(c):
    """Solve grad f = 0 for f = c0 + a x1 + b x2 + A x1^2 + B x1 x2 + C x2^2."""
    H = np.array([[2 * c["x1^2"], c["x1 x2"]], [c["x1 x2"], 2 * c["x2^2"]]])
    g0 = np.array([c["x1"], c["x2"]])
    return np.linalg.solve(H, -g0), H


# ---------------------------------- optimisation ----------------------------------

def multistart_minimise(objective, bounds, starts, cluster_tol):
    """Bounded Nelder-Mead from every start, then merge nearby end points.

    Nelder-Mead is derivative free, which avoids the finite-difference trap on
    GP predictions noted in tutorial Sec. 15.2.
    """
    ends, n_eval = [], 0
    for x0 in starts:
        res = minimize(lambda x: float(objective(x[None, :])[0]), x0,
                       method="Nelder-Mead", bounds=list(map(tuple, bounds)),
                       options=dict(xatol=NM_XATOL, fatol=NM_FATOL, maxiter=NM_MAXITER))
        ends.append((float(res.fun), res.x))
        n_eval += res.nfev
    ends.sort(key=lambda t: t[0])
    clusters = []
    for fval, x in ends:
        for c in clusters:
            if np.linalg.norm(x - c["x"]) <= cluster_tol:
                c["hits"] += 1
                break
        else:
            clusters.append({"f_hat": fval, "x": x, "hits": 1})
    return clusters, n_eval


def on_boundary(x, bounds, tol=BOUNDARY_TOL):
    return bool(np.any(np.abs(x - bounds[:, 0]) < tol) or np.any(np.abs(x - bounds[:, 1]) < tol))


# ------------------------------------- figures -------------------------------------

def plot_truth(spec, G, out):
    X1, X2, Y = G["X1"], G["X2"], G["Y"]
    fig = plt.figure(figsize=(7.2, 3.0))
    ax = fig.add_subplot(1, 2, 1)
    cs = ax.contourf(X1, X2, Y, levels=60, cmap="viridis")
    cs.set_rasterized(True)
    cl = ax.contour(X1, X2, Y, levels=spec["line_levels"], colors="white",
                    linewidths=0.6, alpha=0.8)
    ax.clabel(cl, fontsize=7, fmt="%g")
    if spec["name"] == "Rosenbrock":
        x = np.linspace(-np.sqrt(2), np.sqrt(2), 200)
        ax.plot(x, x**2, "--", color="#eb6834", lw=1.2, label=r"valley $x_2=x_1^2$")
    else:
        t = np.linspace(-6, 6, 50)
        ax.plot(2 + t, 2 + t, "--", color="#eb6834", lw=1.1, label="Hessian eigenvectors")
        ax.plot(2 + t, 2 - t, "--", color="#eb6834", lw=1.1)
        ax.set(xlim=spec["bounds"][0], ylim=spec["bounds"][1])
    ax.plot(*spec["minima"].T, "*", ms=13, mfc="#e34948", mec="white", mew=0.8,
            ls="none", label=f"global minimum, $f={spec['f_min']:g}$")
    ax.set(xlabel="$x_1$", ylabel="$x_2$", aspect="equal")
    ax.legend(loc="lower right", fontsize=7.5)
    sci_colorbar(fig, cs, ax, "$f(x_1,x_2)$")
    panel_letter(ax, "a")

    ax3 = fig.add_subplot(1, 2, 2, projection="3d")
    ax3.plot_surface(X1, X2, Y, cmap="viridis", linewidth=0, antialiased=True,
                     rstride=2, cstride=2, rasterized=True)
    fx, fy = spec["minima"][0]
    ax3.scatter([fx], [fy], [spec["f_min"]], marker="*", s=90, c="#e34948",
                edgecolors="white", depthshade=False, zorder=10)
    ax3.set_xlabel("$x_1$", labelpad=-2)
    ax3.set_ylabel("$x_2$", labelpad=-2)
    ax3.set_zlabel("$f$", labelpad=0)
    ax3.tick_params(pad=-1)
    ax3.view_init(*spec["view"])
    ax3.set_box_aspect(None, zoom=0.85)
    panel_letter(ax3, "b", three_d=True)
    fig.subplots_adjust(wspace=0.3)
    save_figure(fig, out, "fig01_truth")


def plot_doe(spec, designs, scores, chosen_score, out):
    fig, axes = plt.subplots(2, 2, figsize=(6.4, 6.0))
    b = spec["bounds"]
    pad = 0.07 * (b[0, 1] - b[0, 0])
    for ax, (label, U), letter in zip(axes.flat[:3], designs.items(), "abc"):
        X = to_box(U, b)
        ax.plot(X[:, 0], X[:, 1], "o", ms=4.2, mfc="#2a78d6", mec="black", mew=0.4, ls="none")
        # rug ticks along the edges show the one-dimensional projections
        ax.plot(X[:, 0], np.full(len(X), b[1, 0] - 0.55 * pad), "|", color="#e34948", ms=6)
        ax.plot(np.full(len(X), b[0, 0] - 0.55 * pad), X[:, 1], "_", color="#e34948", ms=6)
        ax.set(xlim=(b[0, 0] - pad, b[0, 1] + 0.02 * pad),
               ylim=(b[1, 0] - pad, b[1, 1] + 0.02 * pad),
               xlabel="$x_1$", ylabel="$x_2$", aspect="equal")
        panel_letter(ax, letter)
    ax = axes[1, 1]
    bins = np.linspace(0, max(s.max() for s in scores.values()) * 1.05, 40)
    for (label, s), color in zip(scores.items(), ["#2a78d6", "#eb6834"]):
        ax.hist(s, bins=bins, histtype="stepfilled", alpha=0.25, color=color)
        ax.hist(s, bins=bins, histtype="step", lw=1.3, color=color,
                label=f"{label} (median {np.median(s):.4f})")
    ax.axvline(chosen_score, color="#4a3aa7", ls="--", lw=1.3, label="maximin LHS used")
    ax.set(xlabel=r"minimum pairwise distance $\phi_{\min}$ in $[0,1]^2$",
           ylabel=f"count ({DOE_REALISATIONS} designs)")
    ax.set_ylim(0, ax.get_ylim()[1] * 1.4)     # headroom so the legend clears the bars
    ax.set_xlim(0, 1.08 * max(chosen_score, bins[-1]))
    ax.legend(fontsize=7.5, loc="upper left")
    ax.set_box_aspect(1)
    panel_letter(ax, "d")
    fig.tight_layout()
    save_figure(fig, out, "fig02_doe")


def plot_split(spec, G, X_train, X_test, out):
    fig, ax = plt.subplots(figsize=(3.9, 3.4))
    ax.contour(G["X1"], G["X2"], G["Y"], levels=spec["line_levels"], colors="0.55", linewidths=0.6)
    ax.plot(X_train[:, 0], X_train[:, 1], "o", ms=5, mfc="#2a78d6", mec="black", mew=0.5,
            ls="none", label=f"training ($n={len(X_train)}$)")
    ax.plot(X_test[:, 0], X_test[:, 1], "s", ms=5.5, mfc="#eb6834", mec="black", mew=0.5,
            ls="none", label=f"test ($n={len(X_test)}$)")
    ax.plot(*spec["minima"].T, "*", ms=13, mfc="#e34948", mec="white", mew=0.8, ls="none",
            label="global minimum")
    ax.set(xlabel="$x_1$", ylabel="$x_2$", aspect="equal",
           xlim=spec["bounds"][0], ylim=spec["bounds"][1])
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=3, fontsize=7.5,
              handletextpad=0.3, columnspacing=0.8)
    save_figure(fig, out, "fig03_split")


def plot_degree_sweep(sweep, chosen, out):
    fig, (ax, axc) = plt.subplots(1, 2, figsize=(7.0, 2.8), gridspec_kw={"width_ratios": [1.6, 1]})
    p = sweep["degree"]
    under = sweep.loc[sweep["n_terms"] > sweep["points_per_fold"], "degree"]
    for a in (ax, axc):
        if len(under):
            a.axvspan(under.min() - 0.5, p.max() + 0.5, color="0.9", zorder=0)
        a.axvline(chosen, color="0.35", ls=":", lw=1.2, zorder=1)
        a.grid(alpha=0.3, lw=0.5)
        a.set_xticks(list(p))
    floor = 1e-16
    series = [("train_RMSE", "train", "#2a78d6", "o", "-"),
              ("CV_RMSE", f"{CV_FOLDS}-fold CV (selection)", "#eb6834", "D", "-"),
              ("test_RMSE", "test (diagnostic)", "#1baf7a", "s", "--"),
              ("grid_RMSE", "dense grid (diagnostic)", "#4a3aa7", "^", "--")]
    for col, label, color, mk, ls in series:
        ax.semilogy(p, np.maximum(sweep[col], floor), ls=ls, marker=mk, color=color,
                    ms=4.5, label=label)
    ax.set(xlabel="polynomial degree $p$", ylabel="RMSE", xlim=(0.5, p.max() + 0.5))
    ax.legend(fontsize=7.5, loc="lower left")
    ax.text(chosen + 0.1, 0.97, f"selected $p={chosen}$", transform=ax.get_xaxis_transform(),
            va="top", fontsize=7.5, color="0.25")
    if len(under):
        ax.text(0.5 * (under.min() + p.max()), 1.01, "terms > points\nper CV fold",
                transform=ax.get_xaxis_transform(), fontsize=7, color="0.35",
                ha="center", va="bottom")
    panel_letter(ax, "a")
    axc.semilogy(p, sweep["cond_design"], "o-", color="0.15", ms=4.5)
    axc.set(xlabel="polynomial degree $p$", ylabel=r"cond$(\mathbf{\Psi})$", xlim=(0.5, p.max() + 0.5))
    panel_letter(axc, "b")
    fig.tight_layout()
    save_figure(fig, out, "fig04_degree_sweep")


def plot_parity(y_test, preds, labels, out):
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.55))
    lo = min(y_test.min(), min(v.min() for v in preds.values()))
    hi = max(y_test.max(), max(v.max() for v in preds.values()))
    pad = 0.05 * (hi - lo)
    lim = (lo - pad, hi + pad)
    for ax, name, letter in zip(axes, MODEL_NAMES, "abc"):
        ax.plot(lim, lim, "-", color="0.5", lw=1.0, zorder=1)
        ax.plot(y_test, preds[name], MODEL_MARKERS[name], ms=5, mfc=MODEL_COLORS[name],
                mec="black", mew=0.5, ls="none", zorder=3)
        m = metrics(y_test, preds[name])
        ax.text(0.97, 0.04, f"{labels[name]}\nRMSE $={latex_sci(m['RMSE'])}$\n$R^2={m['R2']:.6f}$",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=7.5,
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.7", alpha=0.9))
        ax.set(xlim=lim, ylim=lim, aspect="equal", xlabel="true $f$ (test set)")
        ax.grid(alpha=0.3, lw=0.5)
        panel_letter(ax, letter)
    axes[0].set_ylabel(r"predicted $\hat f$")
    fig.tight_layout()
    save_figure(fig, out, "fig05_parity")


def plot_side_by_side(spec, G, Y_hat, X_train, label, out, name):
    X1, X2, Y = G["X1"], G["X2"], G["Y"]
    E = Y - Y_hat
    levels, cmap = field_levels(Y)
    emax = np.abs(E).max()
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.35))
    panels = [(Y, "True $f$", "$f$"), (Y_hat, f"{label} $\\hat f$", "$\\hat f$")]
    for ax, (Z, title, cb_label) in zip(axes[:2], panels):
        # both panels use the same levels, so a colour means the same value;
        # extend="both" shows (rather than hides) surrogate values outside them
        cs = ax.contourf(X1, X2, Z, levels=levels, cmap=cmap, extend="both")
        cs.set_rasterized(True)
        ax.contour(X1, X2, Z, levels=spec["line_levels"], colors="white", linewidths=0.5, alpha=0.75)
        ax.set_title(title, fontsize=8.5, pad=3)
        sci_colorbar(fig, cs, ax, cb_label)
    axes[0].plot(*spec["minima"].T, "*", ms=10, mfc="#e34948", mec="white", mew=0.7, ls="none")
    axes[1].plot(X_train[:, 0], X_train[:, 1], "o", ms=2.8, mfc="white", mec="black", mew=0.4, ls="none")
    cs = axes[2].contourf(X1, X2, E, levels=np.linspace(-emax, emax, 41), cmap="RdBu_r",
                          vmin=-emax, vmax=emax)
    cs.set_rasterized(True)
    axes[2].plot(X_train[:, 0], X_train[:, 1], "o", ms=3, mfc="#eda100", mec="black", mew=0.45,
                 ls="none")
    axes[2].set_title("Error $f-\\hat f$", fontsize=8.5, pad=3)
    sci_colorbar(fig, cs, axes[2], "$f-\\hat f$")
    for ax in axes:
        ax.set(xlabel="$x_1$", ylabel="$x_2$", aspect="equal")
    fig.tight_layout(w_pad=0.6)
    save_figure(fig, out, f"fig06_side_by_side_{name}")


def plot_surfaces(spec, G, Y_hat, X_train, y_train, label, out, name):
    X1, X2, Y = G["X1"], G["X2"], G["Y"]
    vmin, vmax = Y.min(), Y.max()
    zlim = (min(vmin, Y_hat.min()), max(vmax, Y_hat.max()))
    fig = plt.figure(figsize=(7.0, 3.0))
    for i, (Z, title) in enumerate([(Y, "True $f$"), (Y_hat, f"{label} $\\hat f$")]):
        ax = fig.add_subplot(1, 2, i + 1, projection="3d")
        surf = ax.plot_surface(X1, X2, Z, cmap="viridis", vmin=vmin, vmax=vmax, linewidth=0,
                               rstride=2, cstride=2, antialiased=True, rasterized=True)
        if i == 1:
            ax.scatter(X_train[:, 0], X_train[:, 1], y_train, c="#e34948", s=7,
                       depthshade=False, label="training data")
            ax.legend(loc="upper right", fontsize=7.5)
        ax.set(xlabel="$x_1$", ylabel="$x_2$", zlabel="$f$" if i == 0 else r"$\hat f$",
               zlim=zlim, xlim=spec["bounds"][0], ylim=spec["bounds"][1])
        ax.set_title(title, fontsize=8.5, pad=0)
        ax.view_init(*spec["view"])
        ax.set_box_aspect(None, zoom=0.86)
    fig.colorbar(surf, ax=fig.axes, shrink=0.7, pad=0.04, label="$f$, $\\hat f$ (shared scale)")
    save_figure(fig, out, f"fig07_surfaces_{name}")


def plot_uncertainty(G, sigma, abs_err, X_train, out):
    fig = plt.figure(figsize=(7.4, 2.45))
    axes = [fig.add_subplot(1, 3, i + 1) for i in range(3)]
    # both fields span several decades and peak in one corner, so a linear
    # scale would show a single colour; the log scale keeps the interior visible
    for ax, Z, label, letter in [(axes[0], sigma, r"GP std. dev. $\hat\sigma$", "a"),
                                 (axes[1], abs_err, r"true $|f-\hat f|$", "b")]:
        Zc = np.maximum(Z, Z.max() * 1e-5)
        levels = np.logspace(np.log10(Zc.min()), np.log10(Zc.max()), 41)
        cs = ax.contourf(G["X1"], G["X2"], Zc, levels=levels, cmap="cividis", norm=LogNorm())
        cs.set_rasterized(True)
        ax.plot(X_train[:, 0], X_train[:, 1], "o", ms=2.8, mfc="white", mec="black", mew=0.4, ls="none")
        ax.set(xlabel="$x_1$", ylabel="$x_2$", aspect="equal")
        cb = fig.colorbar(cs, ax=ax, pad=0.02, fraction=0.05, label=label)
        cb.locator = LogLocator()
        cb.update_ticks()
        panel_letter(ax, letter)
    ax = axes[2]
    s, e = sigma.ravel(), abs_err.ravel()
    keep = np.random.default_rng(SEED).choice(len(s), SCATTER_POINTS, replace=False)
    ax.loglog(s[keep], np.maximum(e[keep], 1e-16), ".", ms=2, color="#1baf7a", alpha=0.5)
    ref = np.logspace(np.log10(s.min()), np.log10(s.max()), 10)
    ax.loglog(ref, 2 * ref, "--", color="0.2", lw=1.0, label=r"$|e|=2\hat\sigma$")
    ax.set(xlabel=r"$\hat\sigma$", ylabel=r"$|f-\hat f|$")
    ax.legend(fontsize=7.5, loc="upper left")
    ax.set_box_aspect(1)
    ax.grid(alpha=0.3, lw=0.5, which="major")
    panel_letter(ax, "c")
    fig.tight_layout(w_pad=0.5)
    save_figure(fig, out, "fig08_gp_uncertainty")


def plot_learning_curve(curve, y_grid_std, out):
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9), sharey=True)
    for ax, col, ylabel, letter in [(axes[0], "grid_RMSE", "RMSE on the dense grid", "a"),
                                    (axes[1], "core_RMSE", "", "b")]:
        for name in MODEL_NAMES:
            d = curve[curve["model"] == name].groupby("N")[col]
            med, lo, hi = d.median(), d.min(), d.max()
            n = med.index.values
            # band = spread over the repeated LHS draws, line = median
            ax.fill_between(n, np.maximum(lo, 1e-15), np.maximum(hi, 1e-15),
                            color=MODEL_COLORS[name], alpha=0.15, lw=0)
            ax.loglog(n, np.maximum(med, 1e-15), marker=MODEL_MARKERS[name],
                      color=MODEL_COLORS[name], ms=4.5, label=name)
        ax.axhline(y_grid_std, color="0.4", ls=":", lw=1.2, label="mean predictor (std of $f$)")
        ax.set(xlabel="training points $N$", ylabel=ylabel)
        ax.set_xticks(LEARNING_N)
        ax.set_xticklabels([str(n) for n in LEARNING_N])
        ax.xaxis.set_minor_locator(plt.NullLocator())
        ax.grid(alpha=0.3, lw=0.5)
        panel_letter(ax, letter)
    axes[1].set_ylabel(f"RMSE in the central box ({int(CORE_FRAC * 100)}% of each side)")
    axes[1].yaxis.set_tick_params(labelleft=True)
    handles, names = axes[0].get_legend_handles_labels()
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.legend(handles, names, loc="lower center", ncol=4, fontsize=7.5)
    save_figure(fig, out, "fig09_learning_curve")


def plot_optimisation(spec, G, surfaces, found, starts, out):
    levels, cmap = field_levels(G["Y"])
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.55))
    for ax, name, letter in zip(axes, MODEL_NAMES, "abc"):
        cs = ax.contourf(G["X1"], G["X2"], surfaces[name], levels=levels, cmap=cmap,
                         extend="both")
        cs.set_rasterized(True)
        ax.contour(G["X1"], G["X2"], surfaces[name], levels=spec["line_levels"], colors="white",
                   linewidths=0.45, alpha=0.7)
        ax.plot(starts[:, 0], starts[:, 1], ".", ms=3, color="white", alpha=0.9)
        xs = np.array([c["x"] for c in found[name]])
        ax.plot(*spec["minima"].T, "*", ms=13, mfc="#e34948", mec="white", mew=0.8, ls="none")
        ax.plot(xs[:, 0], xs[:, 1], "X", ms=8, mfc="#eda100", mec="black", mew=0.7, ls="none")
        ax.set(xlabel="$x_1$", ylabel="$x_2$" if letter == "a" else "", aspect="equal")
        ax.text(0.03, 0.04, name, transform=ax.transAxes, fontsize=7.5, color="white")
        panel_letter(ax, letter)
    handles = [plt.Line2D([], [], marker=".", color="white", mec="0.3", ls="none", ms=6),
               plt.Line2D([], [], marker="*", mfc="#e34948", mec="white", ls="none", ms=11),
               plt.Line2D([], [], marker="X", mfc="#eda100", mec="black", ls="none", ms=8)]
    fig.legend(handles, ["start points", "known global minimum", "surrogate minima"],
               loc="lower center", ncol=3, fontsize=7.5, bbox_to_anchor=(0.45, -0.07))
    cb = fig.colorbar(cs, ax=axes, pad=0.015, fraction=0.03, label=r"$\hat f$")
    cb.set_ticks([t for t in MaxNLocator(6).tick_values(levels[0], levels[-1])
                  if levels[0] <= t <= levels[-1]])
    save_figure(fig, out, "fig10_optimisation")


def plot_extrapolation(spec, G, Y_hat, X_sub, label, out, name):
    X1, X2, Y = G["X1"], G["X2"], G["Y"]
    levels, cmap = field_levels(Y)
    E = Y - Y_hat
    emax = np.abs(E).max()
    sb = spec["sub_box"]
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.35))
    for ax, Z, title, cb_label in [(axes[0], Y, "True $f$", "$f$"),
                                   (axes[1], Y_hat, f"{label} $\\hat f$ (sub-box training)",
                                    "$\\hat f$")]:
        # outside the sub-box the surrogate may leave the true range; extend="both"
        # shows that instead of rescaling the colours to hide it
        cs = ax.contourf(X1, X2, Z, levels=levels, cmap=cmap, extend="both")
        cs.set_rasterized(True)
        ax.set_title(title, fontsize=8.5, pad=3)
        sci_colorbar(fig, cs, ax, cb_label)
    axes[1].plot(X_sub[:, 0], X_sub[:, 1], "o", ms=2.6, mfc="white", mec="black", mew=0.4, ls="none")
    cs = axes[2].contourf(X1, X2, E, levels=np.linspace(-emax, emax, 41), cmap="RdBu_r")
    cs.set_rasterized(True)
    axes[2].set_title("Error $f-\\hat f$", fontsize=8.5, pad=3)
    sci_colorbar(fig, cs, axes[2], "$f-\\hat f$")
    for ax in axes:
        ax.add_patch(Rectangle((sb[0, 0], sb[1, 0]), sb[0, 1] - sb[0, 0], sb[1, 1] - sb[1, 0],
                               fill=False, ec="#e34948", lw=1.6))
        ax.set(xlabel="$x_1$", ylabel="$x_2$", aspect="equal")
    fig.tight_layout(w_pad=0.6)
    save_figure(fig, out, f"fig11_extrapolation_{name}")


# ------------------------------------ the study ------------------------------------

def fit_gp_counting_failures(model, X, y):
    """Fit a GP pipeline and count optimiser runs that ended abnormally.

    The warnings are recorded rather than silenced, so the report can state
    how often the marginal-likelihood search failed to converge cleanly.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(X, y)
    messages = [str(w.message) for w in caught if issubclass(w.category, ConvergenceWarning)]
    return sum("ABNORMAL" in msg for msg in messages)


def run_study(key, spec):
    print(f"\n{spec['name']} function")
    f = spec["func"]
    b = spec["bounds"]
    fig_dir = OUTPUT_DIR / "figures" / key
    tab_dir = OUTPUT_DIR / "tables" / key
    tab_dir.mkdir(parents=True, exist_ok=True)
    summary = {"function": spec["name"], "bounds": b.tolist()}

    def f_vec(X):
        return f(X[:, 0], X[:, 1])

    # step 1: define the truth and check it at the known minimum before fitting
    x1 = np.linspace(*b[0], GRID_N)
    x2 = np.linspace(*b[1], GRID_N)
    X1, X2 = np.meshgrid(x1, x2)
    G = {"X1": X1, "X2": X2, "Y": f(X1, X2)}
    X_grid = np.column_stack([X1.ravel(), X2.ravel()])
    y_grid = G["Y"].ravel()
    summary["truth"] = {"f_at_known_min": float(f(*spec["minima"][0])),
                        "grid_min": float(y_grid.min()), "grid_max": float(y_grid.max()),
                        "grid_mean": float(y_grid.mean()), "grid_std": float(y_grid.std())}
    plot_truth(spec, G, fig_dir)

    # step 2: designs compared in the unit square with the same budget
    rng = np.random.default_rng(SEED)
    designs = {"Uniform random": uniform_random(N_SAMPLES, 2, rng),
               "LHS": latin_hypercube(N_SAMPLES, 2, rng),
               "Maximin LHS": maximin_lhs(N_SAMPLES, 2, rng, MAXIMIN_CANDIDATES)}
    scores = compare_designs(np.random.default_rng(SEED_DOE_STUDY))
    summary["doe"] = {k: float(min_distance(U)) for k, U in designs.items()}
    summary["doe_realisations"] = {k: {"median": float(np.median(s)), "mean": float(s.mean()),
                                       "p05": float(np.percentile(s, 5)), "max": float(s.max())}
                                   for k, s in scores.items()}
    plot_doe(spec, designs, scores, summary["doe"]["Maximin LHS"], fig_dir)

    X_doe = to_box(designs["Maximin LHS"], b)
    y_doe = f_vec(X_doe)                 # the N expensive evaluations

    # step 3: the test rows are set aside here and only used after all choices are fixed
    X_train, X_test, y_train, y_test = train_test_split(
        X_doe, y_doe, test_size=TEST_FRACTION, random_state=SEED)
    split = pd.concat([pd.DataFrame({"set": "train", "x1": X_train[:, 0], "x2": X_train[:, 1], "f": y_train}),
                       pd.DataFrame({"set": "test", "x1": X_test[:, 0], "x2": X_test[:, 1], "f": y_test})])
    split["dist_to_x_min"] = np.min(np.linalg.norm(
        split[["x1", "x2"]].to_numpy()[:, None, :] - spec["minima"][None, :, :], axis=2), axis=1)
    split.to_csv(tab_dir / "doe_split.csv", index=False)
    summary["split"] = {"n_train": len(X_train), "n_test": len(X_test),
                        "y_test_range": float(y_test.max() - y_test.min()),
                        "corner_gaps": corner_gaps(X_train, b)}
    plot_split(spec, G, X_train, X_test, fig_dir)

    # steps 4-5: polynomial degree chosen by training-only CV, then locked;
    # the test and grid curves are attached afterwards purely as diagnostics
    sweep = degree_sweep(X_train, y_train)
    degree = select_degree(sweep, y_train)
    sweep = add_sweep_diagnostics(sweep, X_train, y_train, X_test, y_test, X_grid, y_grid)
    sweep.to_csv(tab_dir / "degree_sweep.csv", index=False)
    summary["selected_degree"] = degree
    plot_degree_sweep(sweep, degree, fig_dir)

    models = {name: make_model(name, degree) for name in MODEL_NAMES}
    models["Polynomial"].fit(X_train, y_train)
    models["Cubic RBF"].fit(X_train, y_train)
    n_abnormal = fit_gp_counting_failures(models["Gaussian process"], X_train, y_train)
    labels = {"Polynomial": f"Polynomial ($p={degree}$)", "Cubic RBF": "Cubic RBF",
              "Gaussian process": "GP"}

    readback = polynomial_readback(models["Polynomial"], degree, X_grid)
    exact = spec["exact_coefs"]
    coef_rows = [{"term": t, "fitted": c, "exact": exact.get(t, 0.0),
                  "abs_diff": abs(c - exact.get(t, 0.0))} for t, c in readback.items()]
    pd.DataFrame(coef_rows).to_csv(tab_dir / "polynomial_coefficients.csv", index=False)
    summary["coef_max_abs_diff"] = float(max(r["abs_diff"] for r in coef_rows))
    summary["rbf_condition_number"] = float(models["Cubic RBF"][-1].condition_number_)
    summary["gp"] = gp_hyperparameters(models["Gaussian process"])
    summary["gp"]["optimizer_runs"] = GP_RESTARTS + 1
    summary["gp"]["abnormal_terminations"] = n_abnormal
    if summary["gp"]["parameters_at_bounds"]:
        gp_bound_sensitivity(X_train, y_train).to_csv(tab_dir / "gp_bound_sensitivity.csv",
                                                      index=False)

    if key == "trid":
        # hand derivation: grad f = 0  ->  [2 -1; -1 2] x = [2; 2]
        H_true = np.array([[2.0, -1.0], [-1.0, 2.0]])
        x_hand = np.linalg.solve(H_true, np.array([2.0, 2.0]))
        x_fit, H_fit = quadratic_stationary_point(readback)
        summary["hand_check"] = {"x_hand": x_hand.tolist(), "f_hand": float(f(*x_hand)),
                                 "hessian_eigenvalues": np.linalg.eigvalsh(H_true).tolist(),
                                 "x_from_fitted_quadratic": x_fit.tolist(),
                                 "fitted_hessian": H_fit.tolist()}

    # step 6: held-out metrics against the mean-of-training baseline
    preds = {name: m.predict(X_test) for name, m in models.items()}
    rows = [{"model": "Baseline mean(y_train)",
             **metrics(y_test, np.full_like(y_test, y_train.mean()))}]
    rows += [{"model": labels[n].replace("$", ""), **metrics(y_test, preds[n])} for n in MODEL_NAMES]
    test_table = pd.DataFrame(rows)
    test_table.to_csv(tab_dir / "test_metrics.csv", index=False)
    plot_parity(y_test, preds, labels, fig_dir)

    # step 7: CV on the training partition, with a dense-grid diagnostic next to it
    grid_pred = {n: models[n].predict(X_grid) for n in ("Polynomial", "Cubic RBF")}
    grid_pred["Gaussian process"], gp_std = models["Gaussian process"].predict(X_grid, return_std=True)
    cv_rows = [{"model": labels[n].replace("$", ""),
                "CV_RMSE": cv_rmse(make_model(n, degree), X_train, y_train),
                "test_RMSE": metrics(y_test, preds[n])["RMSE"],
                **{f"grid_{k}": v for k, v in metrics(y_grid, grid_pred[n]).items()}}
               for n in MODEL_NAMES]
    cv_table = pd.DataFrame(cv_rows)
    cv_table.to_csv(tab_dir / "cv_and_grid.csv", index=False)
    summary["best_by_cv"] = MODEL_NAMES[int(np.argmin(cv_table["CV_RMSE"]))]

    # step 8: side-by-side maps for every model. The CV-best polynomial is the
    # required figure, but its error map is pure round-off here, so the RBF and
    # GP maps are the ones that show where a model without the true form fails
    surfaces = {n: grid_pred[n].reshape(X1.shape) for n in MODEL_NAMES}
    tags = {"Polynomial": "poly", "Cubic RBF": "rbf", "Gaussian process": "gp"}
    for n in MODEL_NAMES:
        plot_side_by_side(spec, G, surfaces[n], X_train, labels[n], fig_dir, tags[n])
        plot_surfaces(spec, G, surfaces[n], X_train, y_train, labels[n], fig_dir, tags[n])

    # the core region drops the outer band, to separate edge effects from the
    # interior; it is used for the uncertainty floor and the learning curve
    centre, half = b.mean(axis=1), 0.5 * CORE_FRAC * (b[:, 1] - b[:, 0])
    core = np.all(np.abs(X_grid - centre) <= half, axis=1)

    # step 9: GP standard deviation against the error it cannot see
    abs_err = np.abs(G["Y"] - surfaces["Gaussian process"])
    sigma = gp_std.reshape(X1.shape)
    outside = abs_err > 2 * sigma
    # jitter plus noise put a floor under sigma_hat; where sigma_hat sits on it,
    # it carries no information about where the error is
    gp = models["Gaussian process"][-1]
    sigma_floor = np.sqrt(gp.alpha + gp.kernel_.k2.noise_level) * y_train.std()
    summary["gp_uncertainty"] = {
        "pearson": float(np.corrcoef(sigma.ravel(), abs_err.ravel())[0, 1]),
        "spearman": float(spearmanr(sigma.ravel(), abs_err.ravel())[0]),
        "coverage_2sigma": float(np.mean(~outside)),
        "sigma_max": float(sigma.max()), "abs_err_max": float(abs_err.max()),
        # where the 2-sigma band fails matters more than how often
        "abs_err_max_outside_band": float(abs_err[outside].max()) if outside.any() else 0.0,
        "centroid_outside_band": ([float(X1[outside].mean()), float(X2[outside].mean())]
                                  if outside.any() else None),
        "sigma_floor": float(sigma_floor),
        "sigma_at_train_max": float(models["Gaussian process"].predict(
            X_train, return_std=True)[1].max()),
        "sigma_median_core": float(np.median(gp_std[core]))}
    plot_uncertainty(G, sigma, abs_err, X_train, fig_dir)

    # step 10: learning curve on fixed grid, several LHS draws per size
    lc_rows = []
    for n in LEARNING_N:
        p = degree
        while n_terms(p) > n and p > 1:      # keep the least-squares problem determined
            p -= 1
        for r in range(LEARNING_REPEATS):
            U = latin_hypercube(n, 2, np.random.default_rng(SEED + SEED_STRIDE * r + n))
            Xn = to_box(U, b)
            yn = f_vec(Xn)
            for name in MODEL_NAMES:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", ConvergenceWarning)
                    m = make_model(name, p).fit(Xn, yn)
                yg = m.predict(X_grid)
                lc_rows.append({"N": n, "repeat": r, "model": name, "degree_used": p,
                                "grid_RMSE": metrics(y_grid, yg)["RMSE"],
                                "core_RMSE": metrics(y_grid[core], yg[core])["RMSE"],
                                "rbf_condition_number": (m[-1].condition_number_
                                                         if name == "Cubic RBF" else np.nan)})
    curve = pd.DataFrame(lc_rows)
    curve.to_csv(tab_dir / "learning_curve.csv", index=False)
    lc_summary = (curve.groupby(["model", "N"])[["grid_RMSE", "core_RMSE", "rbf_condition_number"]]
                  .agg(["median", "min", "max"]))
    lc_summary.columns = ["_".join(c) for c in lc_summary.columns]
    lc_summary.reset_index().to_csv(tab_dir / "learning_curve_summary.csv", index=False)
    summary["core_area_fraction"] = CORE_FRAC ** 2
    plot_learning_curve(curve, y_grid.std(), fig_dir)

    # step 11: optimise every surrogate, then pay one true evaluation per candidate
    cluster_tol = CLUSTER_FRAC * (b[0, 1] - b[0, 0])
    starts = to_box(latin_hypercube(N_STARTS, 2, np.random.default_rng(SEED_STARTS)), b)
    found, opt_rows, n_verify = {}, [], 0
    for name in MODEL_NAMES:
        clusters, _ = multistart_minimise(models[name].predict, b, starts, cluster_tol)
        found[name] = clusters
        for rank, c in enumerate(clusters, start=1):
            f_true = float(f(*c["x"]))
            n_verify += 1
            opt_rows.append({
                "model": name, "rank": rank, "hits": c["hits"],
                "x1": c["x"][0], "x2": c["x"][1],
                "f_hat": c["f_hat"], "f_true": f_true,
                "abs_pred_error": abs(c["f_hat"] - f_true),
                "gap_to_f_min": f_true - spec["f_min"],
                "dist_to_x_min": float(np.min(np.linalg.norm(spec["minima"] - c["x"], axis=1))),
                "on_boundary": on_boundary(c["x"], b)})
    true_clusters, true_nfev = multistart_minimise(f_vec, b, starts, cluster_tol)
    for rank, c in enumerate(true_clusters, start=1):
        opt_rows.append({
            "model": "True function (check)", "rank": rank, "hits": c["hits"],
            "x1": c["x"][0], "x2": c["x"][1], "f_hat": np.nan, "f_true": c["f_hat"],
            "abs_pred_error": np.nan, "gap_to_f_min": c["f_hat"] - spec["f_min"],
            "dist_to_x_min": float(np.min(np.linalg.norm(spec["minima"] - c["x"], axis=1))),
            "on_boundary": on_boundary(c["x"], b)})
    opt_table = pd.DataFrame(opt_rows)
    opt_table.to_csv(tab_dir / "optimisation.csv", index=False)
    summary["optimisation"] = {"n_starts": N_STARTS, "cluster_tol": cluster_tol,
                               "true_function_nfev": int(true_nfev)}
    plot_optimisation(spec, G, surfaces, found, starts, fig_dir)

    # step 12: train on a reduced box, then look inside and outside it separately
    sb = spec["sub_box"]
    X_sub = to_box(latin_hypercube(len(X_train), 2, np.random.default_rng(SEED_SUB_BOX)), sb)
    y_sub = f_vec(X_sub)
    inside = np.all((X_grid >= sb[:, 0]) & (X_grid <= sb[:, 1]), axis=1)
    ex_rows = []
    for name in MODEL_NAMES:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            m = make_model(name, degree).fit(X_sub, y_sub)
        yg = m.predict(X_grid)
        m_in, m_out = metrics(y_grid[inside], yg[inside]), metrics(y_grid[~inside], yg[~inside])
        ex_rows.append({"model": name, "RMSE_inside": m_in["RMSE"], "RMSE_outside": m_out["RMSE"],
                        "MaxAE_inside": m_in["MaxAE"], "MaxAE_outside": m_out["MaxAE"],
                        "RMSE_ratio_out_in": m_out["RMSE"] / m_in["RMSE"]})
        if name != "Polynomial":
            plot_extrapolation(spec, G, yg.reshape(X1.shape), X_sub, labels[name], fig_dir, tags[name])
    pd.DataFrame(ex_rows).to_csv(tab_dir / "extrapolation.csv", index=False)

    # budget kept apart from the extra evaluations used only for teaching checks
    summary["evaluations"] = {
        "main_doe": N_SAMPLES,
        "dense_grid": GRID_N**2,
        "optimum_verification": n_verify,
        "true_function_multistart": int(true_nfev),
        "learning_curve": int(sum(LEARNING_N) * LEARNING_REPEATS),
        "reduced_box": len(X_sub)}

    # a short console summary; the full tables are in the CSV files
    print(f"  selected polynomial degree  : {degree} (training-only {CV_FOLDS}-fold CV)")
    for _, row in test_table.iterrows():
        print(f"  test {row['model']:<24s}: RMSE = {row['RMSE']:.3e}, NRMSE = {row['NRMSE']:.3e}")
    for name in MODEL_NAMES:
        best = found[name][0]
        print(f"  {name:<18s} optimum at ({best['x'][0]:.4f}, {best['x'][1]:.4f}), "
              f"true f = {f(*best['x']):.6g}")
    return summary


def environment_info():
    return {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
            "scikit-learn": sklearn.__version__, "matplotlib": matplotlib.__version__,
            "pandas": pd.__version__, "platform": platform.platform(), "seed": SEED}


def main():
    set_plot_style()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    env = environment_info()
    print("Environment:", env)
    results = {"environment": env}
    for key, spec in FUNCTIONS.items():
        results[key] = run_study(key, spec)
    with open(OUTPUT_DIR / "summary.json", "w") as fh:
        json.dump(results, fh, indent=2, default=float)
    print(f"\nAll figures and tables written to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
