"""Deeper measurements for catalogue candidates.
  - 10,000-step clean run from the saved pattern: persistence re-checked; descriptors
    measured over steps 2000-10000; body count tracked (division).
  - mirror symmetry of the body about its best axis.
  - encounters with a copy of itself (two or three set-ups, each run once for 4000 steps).
usage: python3 characterize.py survivors.jsonl out.jsonl cid [cid ...]
"""
import json, math, sys
import numpy as np
from multiprocessing import Pool
from scipy import ndimage
from search import (N, World, place, decode, encode, centred, crop, bodies, run_batch, persistent,
                    describe, com, YY, XX)

LONG2 = 10000


def mirror_score(A):
    """Best correlation between the centred body and its reflection over 180 axis angles (2 deg steps)."""
    B = centred(A)
    c = np.array(com(B))
    B = np.roll(B, (int(N / 2 - round(c[0])), int(N / 2 - round(c[1]))), (0, 1))
    B = B[N // 4:3 * N // 4, N // 4:3 * N // 4]
    best, ang = -1, 0
    for a in range(0, 180, 2):
        R = ndimage.rotate(B, a, reshape=False, order=1)
        F = R[:, ::-1]
        num = (R * F).sum(); den = math.sqrt((R * R).sum() * (F * F).sum())
        if den and num / den > best: best, ang = num / den, a
    return float(best)


def rot_score(A, k):
    """Correlation of the body with itself rotated by 360/k degrees."""
    B = centred(A)[N // 4:3 * N // 4, N // 4:3 * N // 4]
    R = ndimage.rotate(B, 360 / k, reshape=False, order=1)
    return float((B * R).sum() / math.sqrt((B * B).sum() * (R * R).sum()))


def classify(x, d):
    nb, M, area, masses = x
    if M < 0.2 * d["mass"]: return "both vanished"
    if area > 0.2 * N * N: return "spread over the world"
    if nb == 2 and all(abs(m / d["mass"] - 1) < 0.25 for m in masses): return "two of its kind remain"
    if nb == 1 and abs(M / d["mass"] - 1) < 0.25: return "one of its kind remains"
    if nb == 1: return f"one body of {M / d['mass']:.1f}x the usual mass"
    return f"{nb} bodies"


def encounter(f, d, setups):
    p = f["params"]; P = decode(f["pattern"])
    out = []
    boards = []
    for kind, off in setups:
        A = np.zeros((N, N))
        if d["moving"]:
            h = math.radians(d["heading"])
            u = np.array([-math.sin(h), math.cos(h)])  # (dy, dx) of travel
            nrm = np.array([u[1], -u[0]])
            c1 = np.array([N / 2, N / 2]) - u * N / 4
            c2 = c1 + u * N / 2 + nrm * off
            A += place(P, int(round(c1[0])), int(round(c1[1])))
            A = np.maximum(A, place(np.rot90(P, 2), int(round(c2[0])) % N, int(round(c2[1])) % N))
        else:
            gap = d["span"] + off
            A += place(P, N // 2, int(N / 2 - gap / 2))
            A = np.maximum(A, place(np.rot90(P, int(off) % 4), N // 2, int(N / 2 + gap / 2) % N))
        boards.append(A)
    w = World(N, [p] * len(boards)); w.A = np.array(boards)
    traces = [[] for _ in boards]
    for t in range(4000):
        w.step()
        if t % 10 == 9:
            for i, a in enumerate(w.A):
                nb, _, big = bodies(a)
                traces[i].append((nb, float(a.sum()), int((a > 0.1).sum()), sorted([m for _, m in big], reverse=True)))
    for (kind, off), tr in zip(setups, traces):
        cls = [classify(x, d) for x in tr]
        tc = next((i for i, c in enumerate(cls) if c != "two of its kind remain"), None)
        settle = 50
        if d["moving"]:
            settle = int(min(50, max(3, 0.4 * N / (d["speed"] / f["params"]["T"]) / 10)))
        first = cls[min(tc + settle, len(cls) - 1)] if tc is not None else "never met (two of its kind remain)"
        out.append(dict(setup=kind, offset=float(off), first_meeting=first,
                        contact_step=(tc + 1) * 10 if tc is not None else None, settle_steps=settle * 10,
                        after_4000=cls[-1], bodies_trace=[x[0] for x in tr][::5],
                        mass_trace=[round(x[1] / d["mass"], 3) for x in tr][::5]))
    return out


def sensitivity(f, settle=1000, steps=1500, eps=1e-10):
    """Twin run: does a 1e-10 difference grow? Returns growth factor and e-folding time (steps)."""
    p = f["params"]
    w = World(N, [p]); w.A[0] = place(decode(f["pattern"]))
    for _ in range(settle): w.step()
    rng = np.random.default_rng(0)
    B = w.A[0].copy(); B2 = np.clip(B + eps * rng.standard_normal(B.shape) * (B > 0), 0, 1)
    w2 = World(N, [p, p]); w2.A = np.array([B, B2])
    d = []
    for t in range(steps):
        w2.step()
        if t % 10 == 9: d.append(float(np.abs(w2.A[0] - w2.A[1]).max()))
    d = np.array(d); d0 = float(np.abs(B - B2).max())
    t = np.arange(1, len(d) + 1) * 10
    m = (d > 1e-9) & (d < 1e-3)
    efold = None
    if m.sum() >= 5:
        k = np.polyfit(t[m], np.log(d[m]), 1)[0]
        if k > 0: efold = float(1 / k)
    return dict(start=d0, end=float(d[-1]), growth=float(d[-1] / max(d0, 1e-300)), efold_steps=efold,
                trace=[float(x) for x in d[::5]])


def graded(A):
    """Share of the body's cells (A > 0.02) holding an intermediate value (0.02 < A < 0.98).
    Smooth Lenia bodies are mostly graded; a body near 0 here is a saturated, binary texture."""
    body = A > 0.02
    return float(((A < 0.98) & body).sum() / max(body.sum(), 1))


def run(f):
    p = f["params"]
    recs, fate = run_batch([p], [decode(f["pattern"])], LONG2, 2000)
    r = recs[0]
    res = dict(cid=f["cid"])
    if r is None:
        res.update(ok=False, why=fate[0]); return res
    ok, why = persistent(r, 2000, LONG2)
    nb = [c[1] for c in r["checks"]]
    d = describe(r, p, 2000, LONG2)
    res.update(ok=bool(ok and bodies(r["final"])[0] == 1), why=why, desc=d, max_bodies=int(max(nb)),
               frac_one_body=float(np.mean(np.array(nb) == 1)),
               mass_range=[float(r["mass"][2000:].min()), float(r["mass"][2000:].max())],
               mirror=mirror_score(r["final"]), graded=graded(r["final"]))
    if d["order"] >= 2:
        res["rot_corr"] = rot_score(r["final"], d["order"])
    # a mass trace thumbnail (every 10 steps over the last 2000) for the plate
    res["mass_trace"] = [round(float(x), 3) for x in r["mass"][-2000::10]]
    if d["moving"]:
        setups = [("head-on", 0.0), ("glancing", 0.35 * d["span"]), ("passing", 0.8 * d["span"])]
    else:
        setups = [("side by side, close", 0.25 * p["R"]), ("side by side, near", 0.75 * p["R"])]
    if res["ok"]:
        res["encounters"] = encounter(f, d, setups)
        res["sensitivity"] = sensitivity(f)
    return res


if __name__ == "__main__":
    S = {str(json.loads(l)["cid"]): json.loads(l) for l in open(sys.argv[1])}
    specs = [S[c] for c in sys.argv[3:]]
    with Pool(4) as pool:
        out = pool.map(run, specs)
    with open(sys.argv[2], "w") as fo:
        for o in out: fo.write(json.dumps(o) + "\n")
    for o in out:
        print(o["cid"], o["ok"], o.get("why"), o.get("mirror"), [(e["first_meeting"], e["contact_step"], e["after_4000"]) for e in o.get("encounters", [])])
