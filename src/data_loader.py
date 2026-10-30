"""
data_loader.py -- turn the four Cranfield Excel files into one tidy table.

Source data (real, measured):
  Maton, Economou, Galvao Wall, Khan, Cooper (2024),
  "Low-cost IMU Sensor Temperature Humidity Zero Bias Data",
  Cranfield University, DOI 10.17862/cranfield.rd.24291112, licence CC BY 4.0.

Each Excel sheet has two header rows ("IMU 1:" ... and the channel names),
then one row per sample (~1 Hz). The sensors were held STILL inside an
environmental chamber, so whatever the gyroscope reads is pure error
(bias + noise) -- that is exactly the quantity we want to model.

Output: data/processed/imu_long.csv with columns
  experiment, imu, t_s, T_imu_C, T_chamber_C, RH_pct,
  gx_dps, gy_dps, gz_dps, ax_ms2, ay_ms2, az_ms2
(gyro converted from rad/s to deg/s, which is easier to read).
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"

EXPERIMENTS = {
    "thermal_sweep": "thermal_sweep.xlsx",
    "cycles": "20_deg_cycles.xlsx",
    "cycles_dwells": "20_deg_cycles_dwells.xlsx",
    "humidity": "humidity_data.xlsx",
}
CHANNELS = ["ax_ms2", "ay_ms2", "az_ms2", "gx_dps", "gy_dps", "gz_dps", "T_imu_C"]
RAD2DEG = 180.0 / np.pi


def load_experiment(name: str) -> pd.DataFrame:
    raw = pd.read_excel(RAW / EXPERIMENTS[name], header=None)
    imu_hdr = raw.iloc[0].tolist()
    ch_hdr = [str(c) for c in raw.iloc[1].tolist()]
    body = raw.iloc[2:].apply(pd.to_numeric, errors="coerce").reset_index(drop=True)

    # find the time / reference columns by header text
    def col(contains):
        for i, h in enumerate(ch_hdr):
            if contains.lower() in h.lower():
                return i
        return None

    t_col = col("t (s)") if col("t (s)") is not None else col("Time (s)")
    T_true_col = col("T_av_true")
    rh_col = col("humidity")

    imu_starts = [i for i, h in enumerate(imu_hdr) if isinstance(h, str) and h.startswith("IMU")]
    frames = []
    for k, start in enumerate(imu_starts, 1):
        block = body.iloc[:, start:start + 7].copy()
        block.columns = CHANNELS
        block["gx_dps"] *= RAD2DEG
        block["gy_dps"] *= RAD2DEG
        block["gz_dps"] *= RAD2DEG
        block.insert(0, "imu", k)
        block.insert(0, "experiment", name)
        block["t_s"] = body.iloc[:, t_col].values
        block["T_chamber_C"] = body.iloc[:, T_true_col].values if T_true_col is not None else np.nan
        block["RH_pct"] = body.iloc[:, rh_col].values if rh_col is not None else np.nan
        frames.append(block)
    df = pd.concat(frames, ignore_index=True)
    return df.dropna(subset=["t_s", "T_imu_C", "gz_dps"])


def load_all() -> pd.DataFrame:
    df = pd.concat([load_experiment(n) for n in EXPERIMENTS], ignore_index=True)
    cols = ["experiment", "imu", "t_s", "T_imu_C", "T_chamber_C", "RH_pct",
            "gx_dps", "gy_dps", "gz_dps", "ax_ms2", "ay_ms2", "az_ms2"]
    return df[cols]


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    df = load_all()
    df.to_csv(OUT / "imu_long.csv", index=False)
    print(df.groupby(["experiment", "imu"]).agg(
        n=("t_s", "size"), T_min=("T_imu_C", "min"), T_max=("T_imu_C", "max"),
        gz_mean=("gz_dps", "mean")).round(3).to_string())
    print("saved", OUT / "imu_long.csv", df.shape)
