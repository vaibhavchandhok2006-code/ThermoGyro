# Data

`raw/` contains the four Excel files of the public dataset, unmodified:

> D. Maton, J. Economou, D. Galvão Wall, I. Khan, R. Cooper (2024).
> *Low-cost IMU Sensor Temperature Humidity Zero Bias Data.* Cranfield University.
> DOI: [10.17862/cranfield.rd.24291112](https://doi.org/10.17862/cranfield.rd.24291112).
> Licence: CC BY 4.0. Redistributed here with attribution, unchanged.

| File | What it is | Chips | Samples (~1 Hz) | Temperature |
|---|---|---|---|---|
| `thermal_sweep.xlsx` | cool from 68 °C to −18 °C, dwell, heat to 72 °C, dwell | 3 | 10,777 | −18.6 … 72.1 °C (chamber) |
| `20_deg_cycles.xlsx` | repeated ramps between ~2 and ~25 °C | 3 | 10,887 | 1.7 … 24.5 °C (chamber) |
| `20_deg_cycles_dwells.xlsx` | ramps with dwells at each end | 3 | 17,640 | 1.4 … 24.7 °C (chamber) |
| `humidity_data.xlsx` | ~36 °C constant, relative humidity 89 % → 20 % | 5 | 14,298 | 35.3–37.6 °C (die), RH 20–89 % |

Each IMU block has a_x, a_y, a_z (m/s²), g_x, g_y, g_z (rad/s) and T_imu (°C, the
chip's own die-temperature sensor). The sweep and cycle files also log the
chamber reference temperature (T_1..3_true, T_av_true). All sensors were held
still, so every gyro reading is bias + noise.

`processed/` is created by `src/data_loader.py` and `src/preprocess.py` (git-ignored).
