"""
export_app.py -- train the deployable ThermoGyro models on the real data and
export them (plus real data excerpts) as JSON for the web app in docs/.

For every chip (1-3) and axis (x, y, z) we export two model sets:
  "full" : calibrated on the whole thermal sweep (-18 to 72 degC)
  "cool" : calibrated only on <=30 degC (the 'winter calibration' scenario)
Each set contains Linear, Quadratic, MLP and the ThermoGyro hybrid. The JS in
docs/app.js re-implements predict() exactly; test vectors are exported so the
JS output can be checked against Python (tests/check_js_parity.mjs).
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from models import MLP, HybridModel, PolyRegression
from experiments import MLP_KW, get, val_split, AXES, AXNAME

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "data"
OUT.mkdir(parents=True, exist_ok=True)


def poly_json(m):
    return {"degree": m.degree, "W": m.W.tolist(), "Tm": m.Tm, "Ts": m.Ts}


def train_set(df, ax):
    T, y, d = df.T_imu_C.values, df[ax].values, df.dTdt_Cpm.values
    tr, va = val_split(df)
    lin = PolyRegression(1).fit(T, y)
    quad = PolyRegression(2).fit(T, y)
    mlp = MLP(2, **MLP_KW).fit(np.c_[tr.T_imu_C, tr.dTdt_Cpm], tr[ax].values,
                               np.c_[va.T_imu_C, va.dTdt_Cpm], va[ax].values)
    hyb = HybridModel(2, mlp_kwargs=MLP_KW).fit(tr.T_imu_C.values, tr[ax].values, tr.dTdt_Cpm.values,
                                                va.T_imu_C.values, va[ax].values, va.dTdt_Cpm.values)
    # test vectors for JS parity
    Tq = np.linspace(-25, 80, 23); dq = np.linspace(-4, 2, 23)
    tv = {"T": Tq.tolist(), "d": dq.tolist(),
          "linear": lin.predict(Tq).tolist(), "quadratic": quad.predict(Tq).tolist(),
          "mlp": mlp.predict(np.c_[Tq, dq]).tolist(), "hybrid": hyb.predict(Tq, dq).tolist()}
    return {"linear": poly_json(lin), "quadratic": poly_json(quad), "mlp": mlp.export(),
            "hybrid": {"backbone": poly_json(hyb.backbone), "mlp": hyb.mlp.export(),
                       "rlo": hyb.rlo, "rhi": hyb.rhi, "Tlo": hyb.Tlo, "Thi": hyb.Thi,
                       "margin": hyb.margin},
            "T_range": [float(T.min()), float(T.max())]}, tv


def main():
    models, tests, data = {}, {}, {}
    raw = pd.read_csv(ROOT / "data" / "processed" / "imu_long.csv")
    for imu in (1, 2, 3):
        sweep = get("thermal_sweep", imu)
        for ax in AXES:
            key = f"{imu}{AXNAME[ax]}"
            full, tv = train_set(sweep, ax)
            cool, _ = train_set(sweep[sweep.T_imu_C <= 30], ax)
            models[key] = {"full": full, "cool": cool}
            tests[key] = tv
            print("exported", key)
        # binned data for curve plots (every 2nd bin keeps the file small)
        for exp in ["thermal_sweep", "cycles_dwells"]:
            g = get(exp, imu).iloc[::2]
            data[f"{exp}_{imu}"] = {c: np.round(g[c].values, 4).tolist()
                                    for c in ["t_s", "T_imu_C", "dTdt_Cpm", *AXES]}
        # raw 1 Hz segments for the heading playback
        for name, exp, t0 in [("cool", "cycles_dwells", 0), ("hot", "thermal_sweep", 7600)]:
            r = raw[(raw.experiment == exp) & (raw.imu == imu)].sort_values("t_s")
            r = r[(r.t_s >= t0) & (r.t_s < t0 + 1800)]
            data[f"ride_{name}_{imu}"] = {c: np.round(r[c].values, 5).tolist()
                                          for c in ["t_s", "T_imu_C", *AXES]}
    json.dump(models, open(OUT / "models.json", "w"))
    json.dump(data, open(OUT / "realdata.json", "w"))
    json.dump(tests, open(ROOT / "tests" / "js_test_vectors.json", "w"))
    results = {}
    res_dir = ROOT / "results"
    for f in ["s1_interpolation", "s2_extrapolation", "s3_cross_day"]:
        df = pd.read_csv(res_dir / f"{f}.csv").round(4)
        results[f] = df.astype(object).where(df.notna(), None).to_dict("records")
    results["heading"] = json.load(open(res_dir / "heading_drift.json"))
    json.dump(results, open(OUT / "results.json", "w"), allow_nan=False)
    for p in OUT.iterdir():
        print(p.name, round(p.stat().st_size / 1024), "KB")


if __name__ == "__main__":
    main()
