"""
build_pages.py -- turn web/thermogyro.html (an HTML fragment, the source of
the web app) into docs/index.html, a complete page for GitHub Pages.

Differences from the fragment:
  * adds <!doctype html>, <html>, <head> with charset/viewport, <body>
  * loads Chart.js from docs/vendor/ (bundled copy of chart.js 4.4.1, MIT)
    instead of the cdnjs URL, so the page also works offline.
Run:  python tools/build_pages.py
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CDN = '<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>'

frag = (ROOT / "web" / "thermogyro.html").read_text(encoding="utf-8")
assert CDN in frag, "Chart.js tag not found in fragment"
body_start = frag.index('<div class="wrap">')
head_part = frag[:body_start].replace(CDN, '<script src="vendor/chart.umd.min.js"></script>')
page = ("<!doctype html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n"
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1, viewport-fit=cover\">\n"
        "<meta name=\"description\" content=\"ThermoGyro: ML temperature-drift compensation for the MPU-6050 gyro, trained on real chamber data.\">\n"
        + head_part + "<style>html{color-scheme:light dark} body{margin:0} [hidden]{display:none!important}</style>\n"
        "</head>\n<body>\n" + frag[body_start:] + "\n</body>\n</html>\n")
(ROOT / "docs" / "index.html").write_text(page, encoding="utf-8")
print("wrote docs/index.html", len(page), "bytes")
