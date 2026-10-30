"""
experiments.py -- every study in the ThermoGyro report, run on the REAL
Cranfield MPU-6050 chamber data. Writes tables to results/ and figures to
figures/. Run:  python src/experiments.py   (takes a few minutes)

Hero sensor for single-sensor plots: IMU 3, z-axis gyro (the yaw / heading
axis of a flat-mounted board -- the axis a bicycle or line-follower uses).
Every table also reports all 3 chips x 3 axes so nothing is cherry-picked.
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso

from models import (MLP, HybridModel, PolyRegression, SinglePointCal,
                    design_matrix, gradient_descent, r2, rmse)
from plot_style import (AQUA, BLUE, GREEN, INK2, MAGENTA, MODEL_COLORS, MUTED,
                        ORANGE, RED, VIOLET, YELLOW, plt, save)
from preprocess import bin_and_feature

ROOT = Path(__file__).resolve().parents[1]
PROC, RES, FIG = ROOT / "data" / "processed", ROOT / "results", ROOT / "figures"
RES.mkdir(exist_ok=True); FIG.mkdir(exist_ok=True)

B = pd.read_csv(PROC / "binned_10s.csv")
AXES = ["gx_dps", "gy_dps", "gz_dps"]
AXNAME = {"gx_dps": "x", "gy_dps": "y", "gz_dps": "z"}
HERO = (3, "gz_dps")
MLP_KW = dict(hidden=(16, 16), act="tanh", lr=3e-3, epochs=3000, l2=1e-4, seed=0)
SUMMARY = {}
np.random.seed(0)


def get(exp, imu):
    return B[(B.experiment == exp) & (B.imu == imu)].sort_values("t_s").reset_index(drop=True)


def val_split(df, every=7, block=18):
    """Carve a validation set out of TRAINING data: every 7th 3-minute block."""
    blk = (np.arange(len(df)) // block) % every
    return df[blk != 0], df[blk == 0]


# ------------------------------------------------------------ model factory
def fit(kind, tr, ax, val=None, degree=None, lam=0.0):
    T, y, d = tr.T_imu_C.values, tr[ax].values, tr.dTdt_Cpm.values
    if kind == "Single-point cal":
        m = SinglePointCal().fit(T, y); return lambda T, d: m.predict(T)
    if kind in ("Linear", "Quadratic", "Poly deg 9", "Poly"):
        deg = {"Linear": 1, "Quadratic": 2, "Poly deg 9": 9}.get(kind, degree)
        m = PolyRegression(deg, lam).fit(T, y); return lambda T, d: m.predict(T)
    if kind == "Quadratic + dT/dt":
        m = PolyRegression(2, use_rate=True).fit(T, y, d); return lambda T, d: m.predict(T, d)
    if kind == "MLP":
        if val is None:
            tr, val = val_split(tr)
        m = MLP(2, **MLP_KW).fit(np.c_[tr.T_imu_C, tr.dTdt_Cpm], tr[ax].values,
                                 np.c_[val.T_imu_C, val.dTdt_Cpm], val[ax].values)
        return lambda T, d: m.predict(np.c_[T, d])
    if kind == "ThermoGyro hybrid":
        if val is None:
            tr, val = val_split(tr)
        m = HybridModel(2, mlp_kwargs=MLP_KW).fit(
            tr.T_imu_C.values, tr[ax].values, tr.dTdt_Cpm.values,
            val.T_imu_C.values, val[ax].values, val.dTdt_Cpm.values)
        return lambda T, d: m.predict(T, d)
    raise ValueError(kind)


def evaluate(pred, te, ax):
    p = pred(te.T_imu_C.values, te.dTdt_Cpm.values)
    return rmse(te[ax].values, p), p


def anchored(pred, te, ax, secs=60):
    """Start-up anchoring: at power-on the robot sits still for `secs` seconds,
    measures its bias b0, and the model supplies only the CHANGE from there."""
    st = te[te.t_s <= te.t_s.min() + secs]
    b0 = st[ax].mean()
    p0 = pred(st.T_imu_C.values, st.dTdt_Cpm.values).mean()
    return lambda T, d: b0 + pred(T, d) - p0


# =============================================================== FIGURE 1-2: EDA
def fig_eda():
    df = pd.read_csv(PROC / "imu_long.csv")
    fig, axs = plt.subplots(2, 2, figsize=(11, 6.2))
    titles = {"thermal_sweep": "Thermal sweep (-18 to 72 °C)",
              "cycles": "20-degree cycles", "cycles_dwells": "20-degree cycles with dwells",
              "humidity": "Humidity run (~36 °C, RH 89% → 20%)"}
    for a, e in zip(axs.flat, titles):
        d = df[(df.experiment == e) & (df.imu == 1)]
        a.plot(d.t_s / 3600, d.T_chamber_C, color=MUTED, lw=1.2, label="Chamber air")
        a.plot(d.t_s / 3600, d.T_imu_C, color=BLUE, lw=1.2, label="IMU 1 die temperature")
        if e == "humidity":
            a2 = a.twinx(); a2.plot(d.t_s / 3600, d.RH_pct, color=MAGENTA, lw=1.2)
            a2.set_ylabel("Relative humidity (%)", color=MAGENTA); a2.grid(False)
            a2.spines["right"].set_visible(True)
            a.set_ylim(25, 40)
        a.set_title(titles[e]); a.set_xlabel("Time (h)"); a.set_ylabel("Temperature (°C)")
    axs[0, 0].legend(loc="upper center")
    fig.suptitle("Fig. 1  The four real chamber experiments (Cranfield MPU-6050 data)", x=0.01, ha="left",
                 fontsize=12, fontweight="bold")
    save(fig, FIG / "fig01_experiments.png")

    s = B[B.experiment == "thermal_sweep"]
    fig, axs = plt.subplots(3, 3, figsize=(11.5, 8.5), sharex=True)
    for i, imu in enumerate((1, 2, 3)):
        g = s[s.imu == imu]
        cool, heat = g[g.dTdt_Cpm < 0], g[g.dTdt_Cpm >= 0]
        for j, ax in enumerate(AXES):
            a = axs[i, j]
            a.scatter(cool.T_imu_C, cool[ax], s=4, color=BLUE, label="Cooling", alpha=.8)
            a.scatter(heat.T_imu_C, heat[ax], s=4, color=ORANGE, label="Heating", alpha=.8)
            rng = g[ax].max() - g[ax].min()
            a.set_title(f"IMU {imu} · {AXNAME[ax]}-gyro   (drift range {rng:.2f} °/s)", fontsize=9.5)
            if j == 0: a.set_ylabel("Bias (°/s)")
            if i == 2: a.set_xlabel("Die temperature (°C)")
    axs[0, 0].legend(markerscale=3)
    fig.suptitle("Fig. 2  Measured gyro bias vs temperature, thermal sweep, 10-s averages",
                 x=0.01, ha="left", fontsize=12, fontweight="bold")
    save(fig, FIG / "fig02_bias_vs_temperature.png")

    # noise floor at the two dwells (1 Hz raw), used in the report
    raw = df[df.experiment == "thermal_sweep"]
    rows = []
    for imu in (1, 2, 3):
        r = raw[(raw.imu == imu)]
        for lo, hi, lab in [(3500, 5000, "cold dwell (-18 °C)"), (9600, 10700, "hot dwell (72 °C)")]:
            w = r[(r.t_s > lo) & (r.t_s < hi)]
            for ax in AXES:
                rows.append({"imu": imu, "axis": AXNAME[ax], "dwell": lab,
                             "noise_std_1Hz_dps": w[ax].std(),
                             "noise_std_10s_dps": w[ax].groupby((w.t_s // 10)).mean().std()})
    nf = pd.DataFrame(rows); nf.to_csv(RES / "noise_floor.csv", index=False)
    drift = s.groupby("imu")[AXES].agg(lambda v: v.max() - v.min())
    drift.to_csv(RES / "drift_range_dps.csv")
    SUMMARY["noise_floor_10s_median"] = float(nf.noise_std_10s_dps.median())
    SUMMARY["noise_floor_1Hz_median"] = float(nf.noise_std_1Hz_dps.median())
    SUMMARY["drift_range"] = drift.round(3).to_dict()


# ====================================================== S1: inside calibrated range
S1_MODELS = ["Single-point cal", "Linear", "Quadratic", "Quadratic + dT/dt", "MLP", "ThermoGyro hybrid"]


def s1_interpolation(k=5):
    rows = []
    for imu in (1, 2, 3):
        g = get("thermal_sweep", imu)
        blk = (np.arange(len(g)) // 18) % k
        for ax in AXES:
            for m in S1_MODELS:
                te_r, tr_r = [], []
                for f in range(k):
                    tr, te = g[blk != f], g[blk == f]
                    pred = fit(m, tr, ax)
                    te_r.append(evaluate(pred, te, ax)[0]); tr_r.append(evaluate(pred, tr, ax)[0])
                rows.append({"imu": imu, "axis": AXNAME[ax], "model": m,
                             "train_rmse": np.mean(tr_r), "test_rmse": np.mean(te_r)})
            print("S1", imu, ax, "done")
    df = pd.DataFrame(rows); df.to_csv(RES / "s1_interpolation.csv", index=False)
    return df


# ========================================== S2: calibrated cool (<=30 C), ridden hot
def s2_extrapolation():
    rows, sel = [], []
    for imu in (1, 2, 3):
        g = get("thermal_sweep", imu)
        tr, te = g[g.T_imu_C <= 30], g[g.T_imu_C > 30]
        itr, ival = tr[tr.T_imu_C <= 20], tr[tr.T_imu_C > 20]     # validation mimics deployment
        for ax in AXES:
            # choose polynomial degree using ONLY the training side (Module 3)
            v = {d: evaluate(fit("Poly", itr, ax, degree=d), ival, ax)[0] for d in range(1, 10)}
            dsel = min(v, key=v.get)
            lv = {l: evaluate(fit("Poly", itr, ax, degree=9, lam=l), ival, ax)[0]
                  for l in [0, 1e-3, 1e-2, 0.1, 1, 10, 100, 1e3]}
            lsel = min(lv, key=lv.get)
            sel.append({"imu": imu, "axis": AXNAME[ax], "selected_degree": dsel,
                        "selected_lambda_deg9": lsel})
            models = {m: fit(m, tr, ax) for m in
                      ["Single-point cal", "Linear", "Quadratic", "Poly deg 9", "MLP"]}
            models[f"Degree chosen by validation (d={dsel})"] = fit("Poly", tr, ax, degree=dsel)
            models[f"Ridge deg 9 (lambda={lsel:g})"] = fit("Poly", tr, ax, degree=9, lam=lsel)
            models["ThermoGyro hybrid"] = fit("ThermoGyro hybrid", tr, ax)
            for m, pred in models.items():
                rows.append({"imu": imu, "axis": AXNAME[ax], "model": m,
                             "train_rmse": evaluate(pred, tr, ax)[0],
                             "test_rmse": evaluate(pred, te, ax)[0]})
            # fragility: refit the quadratic 7 times, each time dropping 1/7 of the
            # 3-minute calibration blocks, and record the spread of hot-zone RMSE
            blk = (np.arange(len(tr)) // 18) % 7
            fr = [evaluate(fit("Quadratic", tr[blk != k], ax), te, ax)[0] for k in range(7)]
            sel[-1].update({"quad_drop_block_rmse_min": min(fr), "quad_drop_block_rmse_max": max(fr)})
    df = pd.DataFrame(rows); df.to_csv(RES / "s2_extrapolation.csv", index=False)
    pd.DataFrame(sel).to_csv(RES / "s2_selected_hyperparams.csv", index=False)
    return df


def fig_extrapolation():
    imu, ax = HERO
    g = get("thermal_sweep", imu); tr, te = g[g.T_imu_C <= 30], g[g.T_imu_C > 30]
    Tg = np.linspace(-18, 72, 400)
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.6))
    a = axs[0]
    a.scatter(tr.T_imu_C, tr[ax], s=4, color=MUTED, label="Calibration data (≤30 °C)")
    a.scatter(te.T_imu_C, te[ax], s=4, color=INK2, alpha=.35, label="Hot data, never seen (>30 °C)")
    for m in ["Single-point cal", "Linear", "Quadratic", "Poly deg 9"]:
        p = fit(m, tr, ax)
        a.plot(Tg, p(Tg, np.zeros_like(Tg)), color=MODEL_COLORS[m], label=m)
    a.axvspan(30, 72, color=YELLOW, alpha=.08)
    lo, hi = g[ax].min() - 1, g[ax].max() + 1.5
    a.set_ylim(lo, hi)
    a.text(51, lo + .15 * (hi - lo), "Extrapolation zone\n(Trichy-summer temperatures)", ha="center",
           color=INK2, fontsize=8.5)
    a.set_xlabel("Die temperature (°C)"); a.set_ylabel("Gyro z bias (°/s)")
    a.set_title(f"IMU {imu} z-gyro: models calibrated in the cold, used in the heat")
    a.legend(loc="upper left", fontsize=7.8)
    # MLP vs hybrid on the actual hot sequence
    a = axs[1]
    a.plot(te.T_imu_C, te[ax], ".", ms=3, color=MUTED, label="Measured bias")
    for m in ["MLP", "ThermoGyro hybrid"]:
        p = fit(m, tr, ax)
        a.plot(te.T_imu_C, p(te.T_imu_C.values, te.dTdt_Cpm.values), ".", ms=3,
               color=MODEL_COLORS[m], label=m)
    a.set_xlabel("Die temperature (°C)"); a.set_ylabel("Gyro z bias (°/s)")
    a.set_title("Neural net alone flat-lines outside its data;\nthe range-guarded hybrid falls back to physics")
    a.legend(markerscale=3)
    fig.suptitle("Fig. 8  Cool-weather calibration, hot-weather riding (real data)", x=0.01, ha="left",
                 fontsize=12, fontweight="bold")
    save(fig, FIG / "fig08_extrapolation.png")


# ===================================================== degree sweep (Module 3)
def fig_degree_sweep():
    imu, ax = HERO
    g = get("thermal_sweep", imu)
    blk = (np.arange(len(g)) // 18) % 5
    tr2, te2 = g[g.T_imu_C <= 30], g[g.T_imu_C > 30]
    rows = []
    for d in range(1, 11):
        cv_tr, cv_te = [], []
        for f in range(5):
            tr, te = g[blk != f], g[blk == f]
            p = fit("Poly", tr, ax, degree=d)
            cv_tr.append(evaluate(p, tr, ax)[0]); cv_te.append(evaluate(p, te, ax)[0])
        p = fit("Poly", tr2, ax, degree=d)
        m = PolyRegression(d).fit(tr2.T_imu_C.values, tr2[ax].values)
        rows.append({"degree": d, "interp_train": np.mean(cv_tr), "interp_test": np.mean(cv_te),
                     "extrap_train": evaluate(p, tr2, ax)[0], "extrap_test": evaluate(p, te2, ax)[0],
                     "cond_XtX": m.cond})
    df = pd.DataFrame(rows); df.to_csv(RES / "degree_sweep_hero.csv", index=False)
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))
    a = axs[0]
    a.plot(df.degree, df.interp_train, "o-", color=MUTED, ms=5, label="Train")
    a.plot(df.degree, df.interp_test, "o-", color=BLUE, ms=5, label="Test (held-out blocks)")
    a.set_title("Inside the calibrated range (5-fold blocked CV)")
    a.set_xlabel("Polynomial degree"); a.set_ylabel("RMSE (°/s)"); a.legend()
    a = axs[1]
    a.semilogy(df.degree, df.extrap_train, "o-", color=MUTED, ms=5, label="Train (≤30 °C)")
    a.semilogy(df.degree, df.extrap_test, "o-", color=RED, ms=5, label="Test (>30 °C)")
    a.set_title("Outside it: high degree explodes (log scale)")
    a.set_xlabel("Polynomial degree"); a.set_ylabel("RMSE (°/s)"); a.legend()
    fig.suptitle(f"Fig. 3  Bias–variance in action, IMU {imu} z-gyro", x=0.01, ha="left",
                 fontsize=12, fontweight="bold")
    save(fig, FIG / "fig03_degree_sweep.png")
    return df


# ======================================================= ridge + lasso (Module 3)
def fig_regularisation():
    imu, ax = HERO
    g = get("thermal_sweep", imu)
    tr, te = g[g.T_imu_C <= 30], g[g.T_imu_C > 30]
    lams = np.logspace(-4, 4, 33)
    rows = []
    for l in lams:
        p = fit("Poly", tr, ax, degree=9, lam=l)
        m = PolyRegression(9, l).fit(tr.T_imu_C.values, tr[ax].values)
        rows.append({"lambda": l, "train": evaluate(p, tr, ax)[0], "test": evaluate(p, te, ax)[0],
                     "coef_norm": float(np.linalg.norm(m.W[1:])), **{f"w{k}": m.W[k] for k in range(10)}})
    df = pd.DataFrame(rows); df.to_csv(RES / "ridge_sweep_hero.csv", index=False)
    # lasso on the same standardised polynomial features (which powers survive?)
    Tz = (tr.T_imu_C.values - tr.T_imu_C.mean()) / tr.T_imu_C.std()
    X = design_matrix(Tz, degree=9)[:, 1:]
    lrows = []
    for a_ in [1e-4, 1e-3, 1e-2, 3e-2, 0.1]:
        las = Lasso(alpha=a_, max_iter=200000).fit(X, tr[ax].values)
        lrows.append({"alpha": a_, **{f"T^{k+1}": las.coef_[k] for k in range(9)},
                      "nonzero": int((np.abs(las.coef_) > 1e-8).sum())})
    pd.DataFrame(lrows).to_csv(RES / "lasso_terms_hero.csv", index=False)

    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))
    a = axs[0]
    a.loglog(df["lambda"], df.train, color=MUTED, label="Train (≤30 °C)")
    a.loglog(df["lambda"], df.test, color=RED, label="Test (>30 °C)")
    best = df.loc[df.test.idxmin()]
    a.set_xlabel("Ridge penalty λ"); a.set_ylabel("RMSE (°/s)"); a.legend()
    a.set_title("Degree-9 polynomial rescued by ridge")
    a = axs[1]
    a.semilogx(df["lambda"], df.coef_norm, color=VIOLET)
    a.set_xlabel("Ridge penalty λ"); a.set_ylabel("‖θ₁..θ₉‖ (standardised)")
    a.set_title("Shrinkage: coefficients pulled toward zero")
    fig.suptitle(f"Fig. 4  Regularisation, IMU {imu} z-gyro", x=0.01, ha="left",
                 fontsize=12, fontweight="bold")
    save(fig, FIG / "fig04_ridge.png")
    SUMMARY["ridge_hero"] = {"lambda0_test": float(df.test.iloc[0]),
                             "best_lambda_on_test_curve": float(best["lambda"]),
                             "best_test": float(best.test)}
    return df


# ===================================================== gradient descent (Module 3/4)
def fig_gradient_descent():
    imu, ax = HERO
    g = get("thermal_sweep", imu)
    y = g[ax].values; T = g.T_imu_C.values; N = len(y)
    out = {}
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))
    for a, (lab, x) in zip(axs, [("Unscaled T (°C)", T), ("Standardised T", (T - T.mean()) / T.std())]):
        X = np.c_[np.ones(N), x]
        H = X.T @ X / N
        ev = np.linalg.eigvalsh(H)
        amax = 2 / ev.max()
        W_ne = np.linalg.solve(X.T @ X, X.T @ y)
        J_ne = float(np.sum((X @ W_ne - y) ** 2) / (2 * N))
        res = {"eig_min": float(ev.min()), "eig_max": float(ev.max()), "cond": float(ev.max() / ev.min()),
               "alpha_max": float(amax), "J_normal_eq": J_ne, "runs": {}}
        for frac, col in [(0.5, BLUE), (0.95, AQUA), (1.05, RED)]:
            al = frac * amax
            W, h = gradient_descent(X, y, al, 3000)
            it = next((i for i, j in enumerate(h) if j - J_ne < 1e-6 * max(J_ne, 1e-12) + 1e-9), None)
            res["runs"][f"{frac:.2f}*alpha_max"] = {"alpha": al, "final_J": float(h[-1]),
                                                     "iters_to_converge": it, "W": W.tolist()}
            a.semilogy(np.maximum(h - J_ne, 1e-12), color=col, label=f"α = {frac:g}·α_max ({al:.3g})")
        a.set_title(f"{lab}\nκ = {res['cond']:.0f}, α_max = 2/λ_max = {amax:.3g}")
        a.set_xlabel("Iteration"); a.set_ylabel("J − J* (log)"); a.legend(fontsize=7.8)
        a.set_ylim(1e-12, 1e6)
        out[lab] = res
        out[lab]["W_normal_eq"] = W_ne.tolist()
    fig.suptitle(f"Fig. 5  Gradient descent on the linear model, IMU {imu} z-gyro (real data)",
                 x=0.01, ha="left", fontsize=12, fontweight="bold")
    save(fig, FIG / "fig05_gradient_descent.png")
    json.dump(out, open(RES / "gradient_descent.json", "w"), indent=2)
    return out


# ============================================================ MLP design study
def mlp_design():
    imu, ax = 3, "gx_dps"          # the axis with the clearest hysteresis gain
    g = get("thermal_sweep", imu)
    blk = (np.arange(len(g)) // 18) % 5
    configs = [("tanh", (4,)), ("tanh", (8, 8)), ("tanh", (16, 16)), ("tanh", (32, 32)),
               ("relu", (16, 16)), ("relu", (32, 32))]
    rows = []
    for act, hid in configs:
        r_te, r_tr, dead, ep = [], [], [], []
        for f in range(5):
            tr, te = g[blk != f], g[blk == f]
            itr, iv = val_split(tr)
            m = MLP(2, hidden=hid, act=act, lr=3e-3, epochs=3000, l2=1e-4, seed=0).fit(
                np.c_[itr.T_imu_C, itr.dTdt_Cpm], itr[ax].values, np.c_[iv.T_imu_C, iv.dTdt_Cpm], iv[ax].values)
            r_te.append(rmse(te[ax], m.predict(np.c_[te.T_imu_C, te.dTdt_Cpm])))
            r_tr.append(rmse(tr[ax], m.predict(np.c_[tr.T_imu_C, tr.dTdt_Cpm])))
            dead.append(m.dead_units(np.c_[tr.T_imu_C, tr.dTdt_Cpm])); ep.append(m.epochs_run)
        n_par = sum(a * b + a for a, b in zip([*hid, 1], [2, *hid]))
        rows.append({"activation": act, "hidden": str(hid), "params": n_par, "train_rmse": np.mean(r_tr),
                     "test_rmse": np.mean(r_te), "dead_relu_units": np.mean(dead), "epochs_run": np.mean(ep)})
        print("MLP design", act, hid, rows[-1]["test_rmse"])
    # Dying-ReLU demonstration (Module 4 Sec 4.9.4): biases initialised at -10
    tr, te = g[blk != 0], g[blk == 0]
    X = np.c_[tr.T_imu_C, tr.dTdt_Cpm]
    m = MLP(2, hidden=(16, 16), act="relu", lr=3e-3, epochs=1000, seed=0)
    m.b = [np.full_like(b, -10.0) for b in m.b[:-1]] + [m.b[-1]]
    m.fit(X, tr[ax].values)
    rows.append({"activation": "relu, biases init -10", "hidden": "(16, 16)", "params": 337,
                 "train_rmse": rmse(tr[ax], m.predict(X)),
                 "test_rmse": rmse(te[ax], m.predict(np.c_[te.T_imu_C, te.dTdt_Cpm])),
                 "dead_relu_units": m.dead_units(X), "epochs_run": m.epochs_run})
    df = pd.DataFrame(rows); df.to_csv(RES / "mlp_design.csv", index=False)

    # training curve for the chosen config on fold 0
    itr, iv = val_split(tr)
    m = MLP(2, **MLP_KW).fit(np.c_[itr.T_imu_C, itr.dTdt_Cpm], itr[ax].values,
                             np.c_[iv.T_imu_C, iv.dTdt_Cpm], iv[ax].values)
    fig, a = plt.subplots(figsize=(6.5, 4))
    ep = np.arange(len(m.loss_hist)) * 10
    a.semilogy(ep, m.loss_hist, color=MUTED, label="Training RMSE")
    a.semilogy(ep[:len(m.val_hist)], m.val_hist, color=ORANGE, label="Validation RMSE")
    bi = int(np.argmin(m.val_hist)) * 10
    a.axvline(bi, color=INK2, lw=1, ls="--"); a.text(bi, max(m.val_hist) * .7, " early-stop\n best epoch", fontsize=8, color=INK2)
    a.set_xlabel("Epoch (full-batch Adam)"); a.set_ylabel("RMSE (°/s)"); a.legend()
    a.set_title(f"Fig. 6  MLP [T, dT/dt] → bias, IMU {imu} x-gyro, fold 1", loc="left")
    save(fig, FIG / "fig06_mlp_training.png")
    return df


# =================================================== learning curve (Module 3)
def learning_curve():
    imu, ax = HERO
    g = get("thermal_sweep", imu)
    blk = (np.arange(len(g)) // 18) % 5
    rows = []
    rng = np.random.default_rng(0)
    for frac in [0.1, 0.2, 0.35, 0.5, 0.75, 1.0]:
        for m in ["Quadratic", "ThermoGyro hybrid"]:
            te_r, tr_r = [], []
            for f in range(5):
                tr, te = g[blk != f], g[blk == f]
                bid = np.unique(np.arange(len(tr)) // 18)
                keep = rng.choice(bid, max(3, int(round(frac * len(bid)))), replace=False)
                tr = tr[np.isin(np.arange(len(tr)) // 18, keep)]
                p = fit(m, tr, ax)
                te_r.append(evaluate(p, te, ax)[0]); tr_r.append(evaluate(p, tr, ax)[0])
            rows.append({"fraction": frac, "n_train": len(tr), "model": m,
                         "train_rmse": np.mean(tr_r), "test_rmse": np.mean(te_r)})
    df = pd.DataFrame(rows); df.to_csv(RES / "learning_curve_hero.csv", index=False)
    fig, a = plt.subplots(figsize=(6.5, 4))
    for m in ["Quadratic", "ThermoGyro hybrid"]:
        d = df[df.model == m]
        a.plot(d.n_train, d.test_rmse, "o-", ms=5, color=MODEL_COLORS[m], label=f"{m} – test")
        a.plot(d.n_train, d.train_rmse, "o--", ms=4, color=MODEL_COLORS[m], alpha=.55, label=f"{m} – train")
    a.set_xlabel("Training samples (10-s averages)"); a.set_ylabel("RMSE (°/s)"); a.legend(fontsize=7.8)
    a.set_title(f"Fig. 7  Learning curve, IMU {imu} z-gyro", loc="left")
    save(fig, FIG / "fig07_learning_curve.png")
    return df


# ====================================== S3: a different day (cross-experiment)
S3_MODELS = ["Linear", "Quadratic", "Quadratic + dT/dt", "MLP", "ThermoGyro hybrid"]


def s3_cross_day():
    rows = []
    for imu in (1, 2, 3):
        src = get("thermal_sweep", imu)
        for exp in ["cycles_dwells", "cycles"]:
            te = get(exp, imu)
            for ax in AXES:
                st = te[te.t_s <= te.t_s.min() + 60]
                b0 = st[ax].mean()
                rows.append({"imu": imu, "axis": AXNAME[ax], "test_run": exp,
                             "model": "Start-up calibration only", "rmse": rmse(te[ax], np.full(len(te), b0)),
                             "rmse_no_anchor": np.nan})
                for m in S3_MODELS:
                    p = fit(m, src, ax)
                    rows.append({"imu": imu, "axis": AXNAME[ax], "test_run": exp, "model": m,
                                 "rmse": evaluate(anchored(p, te, ax), te, ax)[0],
                                 "rmse_no_anchor": evaluate(p, te, ax)[0]})
        print("S3", imu, "done")
    df = pd.DataFrame(rows); df.to_csv(RES / "s3_cross_day.csv", index=False)
    return df


def day_offsets():
    """Turn-on bias repeatability: mean difference between a run and the
    thermal-sweep cubic fit evaluated at the same temperatures."""
    rows = []
    for imu in (1, 2, 3):
        src = get("thermal_sweep", imu)
        for exp in ["cycles_dwells", "cycles"]:
            te = get(exp, imu)
            for ax in AXES:
                m = PolyRegression(3).fit(src.T_imu_C.values, src[ax].values)
                rows.append({"imu": imu, "axis": AXNAME[ax], "run": exp,
                             "offset_vs_sweep_dps": float(np.mean(te[ax].values - m.predict(te.T_imu_C.values)))})
    df = pd.DataFrame(rows); df.to_csv(RES / "day_offsets.csv", index=False)
    return df


# ====================================================== S4: unseen chip
def s4_unseen_chip():
    rows = []
    for test_imu in (1, 2, 3):
        others = [i for i in (1, 2, 3) if i != test_imu]
        te = get("thermal_sweep", test_imu)
        for ax in AXES:
            # pooled SHAPE: each training chip centred on its own 25 degC bias
            pooled = []
            for i in others:
                g = get("thermal_sweep", i).copy()
                b25 = g.loc[(g.T_imu_C - 25).abs() <= 2, ax].mean()
                g[ax] = g[ax] - b25
                pooled.append(g)
            pooled = pd.concat(pooled)
            shape = PolyRegression(2).fit(pooled.T_imu_C.values, pooled[ax].values)
            raw = PolyRegression(2).fit(pd.concat([get("thermal_sweep", i) for i in others]).T_imu_C.values,
                                        pd.concat([get("thermal_sweep", i) for i in others])[ax].values)
            near = (te.T_imu_C - 25).abs() <= 2
            b25_new = te.loc[near, ax].mean()
            y = te[ax].values; Tt = te.T_imu_C.values
            rows += [
                {"test_chip": test_imu, "axis": AXNAME[ax], "method": "Other chips' model, used as-is",
                 "rmse": rmse(y, raw.predict(Tt))},
                {"test_chip": test_imu, "axis": AXNAME[ax], "method": "Single-point cal of new chip (25 °C)",
                 "rmse": rmse(y, np.full_like(y, b25_new))},
                {"test_chip": test_imu, "axis": AXNAME[ax], "method": "New chip's 25 °C point + borrowed curve shape",
                 "rmse": rmse(y, b25_new + shape.predict(Tt) - shape.predict(np.array([25.0]))[0])},
                {"test_chip": test_imu, "axis": AXNAME[ax], "method": "Chip's own quadratic (full calibration)",
                 "rmse": rmse(y, PolyRegression(2).fit(Tt, y).predict(Tt))},
            ]
    df = pd.DataFrame(rows); df.to_csv(RES / "s4_unseen_chip.csv", index=False)
    return df


# ================================================= S5: humidity (spurious?)
def s5_humidity():
    rows = []
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))
    for imu in range(1, 6):
        g = get("humidity", imu)
        blk = (np.arange(len(g)) // 18) % 5
        for ax in AXES:
            e = {"T only (quadratic)": [], "T + RH": [], "T + RH + time": []}
            for f in range(5):
                tr, te = g[blk != f], g[blk == f]
                def feats(d, use_rh, use_t):
                    cols = [np.ones(len(d)), d.T_imu_C - 36, (d.T_imu_C - 36) ** 2]
                    if use_rh: cols.append(d.RH_pct / 100)
                    if use_t: cols.append(d.t_s / 3600)
                    return np.column_stack(cols)
                for lab, u_rh, u_t in [("T only (quadratic)", 0, 0), ("T + RH", 1, 0), ("T + RH + time", 1, 1)]:
                    X = feats(tr, u_rh, u_t)
                    W = np.linalg.lstsq(X, tr[ax].values, rcond=None)[0]
                    e[lab].append(rmse(te[ax], feats(te, u_rh, u_t) @ W))
            Xf = np.column_stack([np.ones(len(g)), g.T_imu_C - 36, (g.T_imu_C - 36) ** 2, g.RH_pct / 100])
            Wf = np.linalg.lstsq(Xf, g[ax].values, rcond=None)[0]
            res = g[ax].values - Xf[:, :3] @ np.linalg.lstsq(Xf[:, :3], g[ax].values, rcond=None)[0]
            rows.append({"imu": imu, "axis": AXNAME[ax], **{k: np.mean(v) for k, v in e.items()},
                         "RH_coef_dps_per_100pct": Wf[3],
                         "corr_RH_time": float(np.corrcoef(g.RH_pct, g.t_s)[0, 1]),
                         "corr_RH_T": float(np.corrcoef(g.RH_pct, g.T_imu_C)[0, 1]),
                         "corr_residual_RH": float(np.corrcoef(res, g.RH_pct)[0, 1]),
                         "T_range_C": float(g.T_imu_C.max() - g.T_imu_C.min())})
            if (imu, ax) == (3, "gz_dps"):
                axs[0].scatter(g.RH_pct, res, s=4, c=g.t_s / 3600, cmap="Blues")
                axs[0].set_xlabel("Relative humidity (%)"); axs[0].set_ylabel("Residual after T-model (°/s)")
                axs[0].set_title("IMU 3 z: residual vs humidity (shade = time)")
                axs[1].plot(g.t_s / 3600, g.RH_pct, color=MAGENTA, label="RH (%)")
                axs[1].plot(g.t_s / 3600, (g.T_imu_C - 35) * 40, color=BLUE, label="(T − 35 °C) × 40")
                axs[1].set_xlabel("Time (h)"); axs[1].legend()
                axs[1].set_title("RH falls steadily with time → confounded with any slow drift")
    df = pd.DataFrame(rows); df.to_csv(RES / "s5_humidity.csv", index=False)
    fig.suptitle("Fig. 10  Does humidity matter? (5 chips at ~36 °C, RH 89% → 20%)", x=0.01, ha="left",
                 fontsize=12, fontweight="bold")
    save(fig, FIG / "fig10_humidity.png")
    return df


# ============================================== S6: data quality (real glitches)
def s6_data_quality():
    raw = pd.read_csv(PROC / "imu_long.csv")
    r = raw[(raw.experiment == "cycles") & (raw.imu == 2)].sort_values("t_s").reset_index(drop=True)
    dirty = bin_and_feature(r)
    clean = get("cycles", 2)
    te = get("cycles_dwells", 2)
    rows = []
    for ax in AXES:
        for lab, d in [("with glitches", dirty), ("glitches removed", clean)]:
            m = PolyRegression(2).fit(d.T_imu_C.values, d[ax].values)
            p = lambda T, dd, m=m: m.predict(T)
            rows.append({"axis": AXNAME[ax], "training_data": lab,
                         "theta0": m.W[0], "theta1": m.W[1], "theta2": m.W[2],
                         "slope_at_mean_dps_per_C": m.W[1] / m.Ts,
                         "rmse_on_other_day_anchored": evaluate(anchored(p, te, ax), te, ax)[0]})
    df = pd.DataFrame(rows); df.to_csv(RES / "s6_data_quality.csv", index=False)
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))
    axs[0].plot(r.t_s / 3600, r.T_imu_C, color=RED, lw=1, label="IMU 2 reported temperature (raw)")
    axs[0].plot(r.t_s / 3600, r.T_chamber_C, color=MUTED, lw=1, label="Chamber air")
    axs[0].set_xlabel("Time (h)"); axs[0].set_ylabel("°C"); axs[0].legend()
    axs[0].set_title("Real logging glitch: one reading of 90.7 °C")
    axs[1].scatter(dirty.T_imu_C, dirty.gy_dps, s=5, color=RED, label="10-s bins incl. glitches")
    axs[1].scatter(clean.T_imu_C, clean.gy_dps, s=5, color=BLUE, label="After glitch removal")
    axs[1].set_xlabel("Die temperature (°C)"); axs[1].set_ylabel("y-gyro bias (°/s)"); axs[1].legend()
    axs[1].set_title("Glitched bins become outliers that pull the fit")
    fig.suptitle("Fig. 11  Data quality: a few bad samples out of 10,887", x=0.01, ha="left",
                 fontsize=12, fontweight="bold")
    save(fig, FIG / "fig11_data_quality.png")
    return df


# ======================================= uncertainty band +/- 2 RMSE (Module 5)
def uncertainty():
    rows = []
    for imu in (1, 2, 3):
        g = get("thermal_sweep", imu)
        tr, te = g[g.T_imu_C <= 30], g[g.T_imu_C > 30]
        itr, iv = val_split(tr)
        for ax in AXES:
            for m in ["Quadratic", "ThermoGyro hybrid"]:
                # band width from VALIDATION residuals (never from the test set)
                p_i = fit(m, itr, ax)
                s = evaluate(p_i, iv, ax)[0]
                p = fit(m, tr, ax)
                pr = p(te.T_imu_C.values, te.dTdt_Cpm.values)
                inside = np.abs(te[ax].values - pr) <= 2 * s
                near = te.T_imu_C <= 40
                rows.append({"imu": imu, "axis": AXNAME[ax], "model": m, "band_halfwidth_dps": 2 * s,
                             "coverage_validation_like_range": None,
                             "coverage_30_40C": float(inside[near.values].mean()),
                             "coverage_40_72C": float(inside[~near.values].mean())})
        # in-range coverage via blocked CV
        blk = (np.arange(len(g)) // 18) % 5
        for ax in AXES:
            for m in ["Quadratic", "ThermoGyro hybrid"]:
                cov = []
                for f in range(5):
                    tr_, te_ = g[blk != f], g[blk == f]
                    itr_, iv_ = val_split(tr_)
                    s = evaluate(fit(m, itr_, ax), iv_, ax)[0]
                    pr = fit(m, tr_, ax)(te_.T_imu_C.values, te_.dTdt_Cpm.values)
                    cov.append(np.mean(np.abs(te_[ax].values - pr) <= 2 * s))
                for row in rows:
                    if row["imu"] == imu and row["axis"] == AXNAME[ax] and row["model"] == m:
                        row["coverage_validation_like_range"] = float(np.mean(cov))
    df = pd.DataFrame(rows); df.to_csv(RES / "uncertainty_coverage.csv", index=False)

    imu, ax = HERO
    g = get("thermal_sweep", imu); tr = g[g.T_imu_C <= 30]
    itr, iv = val_split(tr)
    s = evaluate(fit("ThermoGyro hybrid", itr, ax), iv, ax)[0]
    p = fit("ThermoGyro hybrid", tr, ax)
    pr = p(g.T_imu_C.values, g.dTdt_Cpm.values)
    fig, a = plt.subplots(figsize=(11, 4))
    t = g.t_s / 3600
    a.fill_between(t, pr - 2 * s, pr + 2 * s, color=AQUA, alpha=.18, label=f"±2·RMSE band (±{2*s:.3f} °/s)")
    a.plot(t, g[ax], color=INK2, lw=1, label="Measured bias")
    a.plot(t, pr, color=AQUA, lw=1.6, label="ThermoGyro hybrid prediction")
    hot = g.T_imu_C > 30
    a.fill_between(t, a.get_ylim()[0] if False else g[ax].min() - .3, g[ax].max() + .3, where=hot,
                   color=YELLOW, alpha=.08, label="T > 30 °C (outside calibration)")
    a.set_xlabel("Time in thermal sweep (h)"); a.set_ylabel("z-gyro bias (°/s)"); a.legend(fontsize=7.8, ncol=2)
    a.set_title(f"Fig. 12  Prediction with uncertainty band, IMU {imu} z-gyro (calibrated ≤30 °C)", loc="left")
    save(fig, FIG / "fig12_uncertainty.png")
    return df


# ================================== heading drift on RAW 1 Hz data (the payoff)
def heading_drift():
    raw = pd.read_csv(PROC / "imu_long.csv")
    imu, ax = HERO
    out = {}
    scenarios = {
        "cool": dict(exp="cycles_dwells", t0=0, dur=1800,
                     title="Cool morning ride, a different day", train="full"),
        "hot": dict(exp="thermal_sweep", t0=7600, dur=1800,
                    title="Hot afternoon, calibrated only ≤30 °C", train="cool"),
    }
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.4))
    for a, (key, sc) in zip(axs, scenarios.items()):
        r = raw[(raw.experiment == sc["exp"]) & (raw.imu == imu)].sort_values("t_s")
        r = r[(r.t_s >= sc["t0"]) & (r.t_s < sc["t0"] + sc["dur"])].reset_index(drop=True)
        Ts = r.T_imu_C.rolling(10, center=True, min_periods=1).mean().values
        d = np.gradient(Ts, r.t_s.values) * 60
        d = pd.Series(d).rolling(30, center=True, min_periods=1).mean().values
        src = get("thermal_sweep", imu)
        if sc["train"] == "cool":
            src = src[src.T_imu_C <= 30]
        rdf = pd.DataFrame({"t_s": r.t_s, "T_imu_C": Ts, "dTdt_Cpm": d, ax: r[ax]})
        b0 = rdf[rdf.t_s <= rdf.t_s.min() + 60][ax].mean()
        comps = {"No compensation": np.zeros(len(r)),
                 "Start-up calibration only": np.full(len(r), b0)}
        for m in ["Quadratic", "MLP", "ThermoGyro hybrid"]:
            p = anchored(fit(m, src, ax), rdf, ax)
            comps[m] = p(Ts, d)
        tt = (r.t_s - r.t_s.iloc[0]).values
        dt = np.diff(tt, prepend=tt[0])
        colors = {"No compensation": RED, "Start-up calibration only": MUTED, "Quadratic": BLUE,
                  "MLP": ORANGE, "ThermoGyro hybrid": AQUA}
        res = {}
        for lab, c in comps.items():
            head = np.cumsum((r[ax].values - c) * dt)
            res[lab] = {"heading_error_10min_deg": float(head[tt <= 600][-1]),
                        "heading_error_30min_deg": float(head[-1]),
                        "max_abs_deg": float(np.abs(head).max())}
            if lab != "No compensation":
                a.plot(tt / 60, head, color=colors[lab], label=f"{lab}: {head[-1]:+.1f}° after 30 min")
        res["T_start"] = float(Ts[0]); res["T_end"] = float(Ts[-1])
        a.axhline(0, color=INK2, lw=.8)
        a.set_title(f"{sc['title']}\n(die {Ts[0]:.0f} °C → {Ts.min() if key=='cool' else Ts.max():.0f} °C, 30 min)"); a.set_xlabel("Minutes since start"); a.set_ylabel("Heading error (°)")
        a.legend(fontsize=7.8)
        out[key] = res
    fig.suptitle("Fig. 13  What the rider sees: heading error from integrating the real, still gyro (IMU 3 z)",
                 x=0.01, ha="left", fontsize=12, fontweight="bold")
    save(fig, FIG / "fig13_heading_drift.png")
    json.dump(out, open(RES / "heading_drift.json", "w"), indent=2)
    return out


# ============================================================== summary figure
def fig_summary(s1, s2, s3):
    fig, axs = plt.subplots(1, 3, figsize=(13, 4.3), sharey=False)
    order = ["Single-point cal", "Linear", "Quadratic", "MLP", "ThermoGyro hybrid"]
    for a, (df, title, col) in zip(axs, [(s1, "Inside calibrated range\n(blocked CV, thermal sweep)", "test_rmse"),
                                          (s2, "Calibrated ≤30 °C → used >30 °C", "test_rmse")]):
        d = df[df.axis == "z"].pivot_table(index="model", columns="imu", values=col).reindex(order)
        x = np.arange(len(order)); w = .26
        for k, imu in enumerate((1, 2, 3)):
            a.bar(x + (k - 1) * w, d[imu], w * .92, color=[MODEL_COLORS[m] for m in order],
                  alpha=[.45, .7, 1][k], edgecolor="none")
        a.set_xticks(x); a.set_xticklabels([o.replace(" ", "\n", 1) for o in order], fontsize=7.8)
        a.set_ylabel("z-gyro RMSE (°/s)"); a.set_title(title)
    d3 = s3[(s3.axis == "z") & (s3.test_run == "cycles_dwells")]
    order3 = ["Start-up calibration only", "Linear", "Quadratic", "MLP", "ThermoGyro hybrid"]
    cmap = {"Start-up calibration only": MUTED, **MODEL_COLORS}
    p = d3.pivot_table(index="model", columns="imu", values="rmse").reindex(order3)
    x = np.arange(len(order3)); w = .26
    for k, imu in enumerate((1, 2, 3)):
        axs[2].bar(x + (k - 1) * w, p[imu], w * .92, color=[cmap[m] for m in order3], alpha=[.45, .7, 1][k])
    axs[2].set_xticks(x); axs[2].set_xticklabels([o.replace(" ", "\n", 1) for o in order3], fontsize=7.8)
    axs[2].set_ylabel("z-gyro RMSE (°/s)"); axs[2].set_title("A different day\n(start-up anchored)")
    fig.suptitle("Fig. 9  Model comparison, z-gyro of all three chips (bars left→right: IMU 1, 2, 3)",
                 x=0.01, ha="left", fontsize=12, fontweight="bold")
    save(fig, FIG / "fig09_summary.png")


if __name__ == "__main__":
    t0 = time.time()
    fig_eda();                 print("eda", round(time.time() - t0))
    s1 = s1_interpolation();   print("s1", round(time.time() - t0))
    s2 = s2_extrapolation();   print("s2", round(time.time() - t0))
    fig_extrapolation();       print("fig9", round(time.time() - t0))
    fig_degree_sweep();        print("deg", round(time.time() - t0))
    fig_regularisation();      print("ridge", round(time.time() - t0))
    SUMMARY["gd"] = fig_gradient_descent()
    mlp_design();              print("mlp", round(time.time() - t0))
    learning_curve();          print("lc", round(time.time() - t0))
    s3 = s3_cross_day();       print("s3", round(time.time() - t0))
    day_offsets()
    s4_unseen_chip(); s5_humidity(); s6_data_quality()
    uncertainty();             print("unc", round(time.time() - t0))
    SUMMARY["heading"] = heading_drift()
    fig_summary(s1, s2, s3)
    json.dump(SUMMARY, open(RES / "summary.json", "w"), indent=2, default=float)
    print("ALL DONE in", round(time.time() - t0), "s")
