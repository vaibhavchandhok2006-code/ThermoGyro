#!/usr/bin/env bash
# Reproduce every number, figure and the web-app model files from the raw data.
set -e
cd "$(dirname "$0")"
python src/data_loader.py        # 4 Excel files -> data/processed/imu_long.csv
python src/preprocess.py         # glitch removal, 10-s averaging, dT/dt
python -m pytest -q tests/       # backprop gradient check, normal equation, GD
python src/experiments.py        # all studies -> results/ and figures/
(cd src && python export_app.py) # trained models -> docs/data/*.json
python tools/build_pages.py      # web/thermogyro.html -> docs/index.html
node tests/check_js_parity.mjs   # JS inference == Python inference
echo "Done. Open the app with:  cd docs && python -m http.server  ->  http://localhost:8000"
