"""
preprocess.py -- clean the raw 1 Hz data and build model-ready features.

Steps (all documented in the report, Section "Data"):
 1. Glitch removal: a sample is flagged if any of
      |T_imu - rolling median(T_imu, 31 s)| > 2 degC, or
      |gyro - rolling median(gyro, 31 s)| > 8 * rolling MAD (robust z-score)
    These are single-sample communication / logging glitches (e.g. one IMU-2
    temperature reading of 90.7 degC in the 20-degree-cycles run while the
    chamber was at ~18 degC).
 2. 10-second averaging (binning): reduces white noise and the 1-LSB
    quantisation (1/131 deg/s) so the models see the bias trend, not noise.
 3. Features: T (degC) and dT/dt (degC/min, from a centred difference of the
    binned temperature). dT/dt tells the model whether the chip is heating
    or cooling, which matters because the real data shows hysteresis.
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
GYRO = ["gx_dps", "gy_dps", "gz_dps"]
BIN_S = 10.0


def flag_glitches(g: pd.DataFrame, win: int = 31) -> pd.Series:
    bad = pd.Series(False, index=g.index)
    Tm = g["T_imu_C"].rolling(win, center=True, min_periods=5).median()
    bad |= (g["T_imu_C"] - Tm).abs() > 2.0
    for c in GYRO:
        med = g[c].rolling(win, center=True, min_periods=5).median()
        mad = (g[c] - med).abs().rolling(win, center=True, min_periods=5).median()
        mad = mad.clip(lower=1.0 / 131.0)          # never below 1 LSB
        bad |= (g[c] - med).abs() > 8 * 1.4826 * mad
    return bad


def bin_and_feature(g: pd.DataFrame) -> pd.DataFrame:
    g = g.copy()
    g["bin"] = (g["t_s"] // BIN_S).astype(int)
    agg = {c: "mean" for c in ["t_s", "T_imu_C", "T_chamber_C", "RH_pct"] + GYRO}
    b = g.groupby("bin").agg(agg)
    b["n"] = g.groupby("bin").size()
    b = b[b["n"] >= 5].reset_index(drop=True)
    # dT/dt in degC per minute, centred difference
    T = b["T_imu_C"].to_numpy()
    t = b["t_s"].to_numpy()
    dT = np.gradient(T, t) * 60.0
    b["dTdt_Cpm"] = pd.Series(dT).rolling(3, center=True, min_periods=1).mean().to_numpy()
    return b


def build():
    df = pd.read_csv(PROC / "imu_long.csv")
    out, report = [], []
    for (exp, imu), g in df.groupby(["experiment", "imu"]):
        g = g.sort_values("t_s").reset_index(drop=True)
        bad = flag_glitches(g)
        report.append({"experiment": exp, "imu": imu, "samples": len(g),
                       "glitches_removed": int(bad.sum())})
        b = bin_and_feature(g[~bad])
        b.insert(0, "imu", imu)
        b.insert(0, "experiment", exp)
        out.append(b)
    binned = pd.concat(out, ignore_index=True)
    binned.to_csv(PROC / "binned_10s.csv", index=False)
    rep = pd.DataFrame(report)
    rep.to_csv(PROC / "glitch_report.csv", index=False)
    print(rep.to_string(index=False))
    print("binned rows:", len(binned))
    return binned


if __name__ == "__main__":
    build()
