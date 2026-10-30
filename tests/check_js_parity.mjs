// Verifies that the web app's JavaScript inference (docs/index.html,
// <script id="tg-models">) reproduces the Python models exactly.
// Run: node tests/check_js_parity.mjs
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import path from "node:path";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const html = readFileSync(path.join(root, "docs/index.html"), "utf8");
const src = html.match(/<script id="tg-models">([\s\S]*?)<\/script>/)[1];
const module = { exports: {} };
new Function("module", src)(module);
const TG = module.exports;

const models = JSON.parse(readFileSync(path.join(root, "docs/data/models.json"), "utf8"));
const tv = JSON.parse(readFileSync(path.join(root, "tests/js_test_vectors.json"), "utf8"));
let worst = 0, n = 0;
for (const key of Object.keys(tv)) {
  const m = models[key].full, v = tv[key];
  v.T.forEach((T, i) => {
    const d = v.d[i];
    const got = { linear: TG.poly(m.linear, T), quadratic: TG.poly(m.quadratic, T),
                  mlp: TG.mlp(m.mlp, [T, d]), hybrid: TG.hybrid(m.hybrid, T, d) };
    for (const k of Object.keys(got)) { worst = Math.max(worst, Math.abs(got[k] - v[k][i])); n++; }
  });
}
console.log(`${n} predictions compared, max |JS - Python| = ${worst.toExponential(2)} deg/s`);
if (worst > 1e-9) { console.error("PARITY FAILED"); process.exit(1); }
console.log("PARITY OK");
