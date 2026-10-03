"""Check that the browser engine (lenia.js, run under node) reproduces the Python engine
for each catalogued specimen: same pattern, same rule, STEPS steps.
Reports max |board difference| at checkpoints, and compares behaviour (mean mass,
distance travelled) because for chaotic creatures float rounding alone eventually
makes the two runs diverge cell by cell.
usage: python3 verify_js.py specimens/*.json"""
import json, os, subprocess, sys, tempfile
import numpy as np
from search import N, World, place, decode, com

STEPS = 3000
CK = [100, 500, 1000, 2000, 3000]
JS = r"""
const {Lenia, decodePattern} = require(process.argv[2]);
const fs = require('fs');
const s = JSON.parse(fs.readFileSync(process.argv[3]));
const L = new Lenia(128, s.params); L.place(decodePattern(s.pattern), s.pattern.h, s.pattern.w);
const CK = %s, out = {mass: [], boards: {}};
for (let t = 1; t <= %d; t++) { L.step(); out.mass.push(L.mass()); if (CK.includes(t)) out.boards[t] = Array.from(L.A); }
fs.writeFileSync(process.argv[4], JSON.stringify(out));
""" % (json.dumps(CK), STEPS)




def main(paths):
    here = os.path.dirname(os.path.abspath(__file__))
    tmp = tempfile.mkdtemp()
    jsf = os.path.join(tmp, "v.js"); open(jsf, "w").write(JS)
    rows = []
    for pth in paths:
        s = json.load(open(pth))
        out = os.path.join(tmp, "o.json")
        subprocess.run(["node", jsf, os.path.join(here, "lenia.js"), pth, out], check=True)
        js = json.load(open(out))
        w = World(N, [s["params"]]); w.A[0] = place(decode(s["pattern"]))
        pm, diffs, cdist = [], {}, {}
        for t in range(1, STEPS + 1):
            w.step(); pm.append(w.A[0].sum())
            if t in CK:
                J = np.array(js["boards"][str(t)]).reshape(N, N)
                diffs[t] = float(np.abs(w.A[0] - J).max())
                d = (np.array(com(w.A[0])) - np.array(com(J)) + N / 2) % N - N / 2
                cdist[t] = float(np.hypot(*d))
        pm = np.array(pm); jm = np.array(js["mass"])
        name = os.path.basename(pth)[:-5]
        rows.append(dict(name=name, diffs=diffs, py_mass=float(pm[1000:].mean()), js_mass=float(jm[1000:].mean()), com_dist=cdist))
        print(f"{name:14s} max|py-js| " + "  ".join(f"@{t}:{d:.1e}" for t, d in diffs.items()) +
              f"   centre apart @3000: {cdist[3000]:.2f} cells   mean mass (steps 1000-3000) py {pm[1000:].mean():.2f} js {jm[1000:].mean():.2f}")
    return rows


if __name__ == "__main__":
    rows = main(sys.argv[1:])
    json.dump(rows, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "verify_js_results.json"), "w"), indent=1)
