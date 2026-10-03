"""Open-ended novelty search for persistent Lenia creatures (single channel).

Pipeline per candidate (params + seed pattern), all on a 128x128 torus:
  1. screen   : up to SCREEN steps; killed early if it dies or spreads.
  2. long run : LONG steps total; persistence judged on steps [WARM, LONG).
  3. if the board ends with several bodies, the largest is cut out and re-run
     alone through 1-2 (it may have been born by division).
  4. robustness: the final body is re-seeded twice from perturbed copies
     (10% multiplicative noise; once rotated by a random multiple of 90 deg,
     once by a random arbitrary angle) and must persist again with similar
     mass and mobility.
  5. survivors are described by measured behaviour and offered to a novelty
     archive; new candidates are either fresh random draws or mutations of
     archive members chosen in proportion to their novelty.
Everything that persists is logged to survivors.jsonl, archive or not.
"""
import base64, json, math, os, sys, time
import numpy as np
from multiprocessing import Pool
from scipy import ndimage
from lenia import World

N = 128
SCREEN = 600
WARM = 1000
LONG = 5000
ROBUST = 3000
CHECK = 50
HATCH = 1000
ANG = 2 * np.pi * np.arange(N) / N
YY, XX = np.mgrid[0:N, 0:N]


# ------------------------------------------------------------------ sampling
def rand_params(rng):
    r = rng.random()
    kn, gn = (1, 1) if r < 0.5 else (2, 2) if r < 0.75 else (int(rng.integers(1, 3)), int(rng.integers(1, 3)))
    B = int(rng.choice([1, 2, 3, 4], p=[0.35, 0.3, 0.22, 0.13]))
    b = list(np.round(rng.random(B) * 12) / 12)
    b[int(rng.integers(B))] = 1.0
    m = float(rng.uniform(0.08, 0.45))
    s = float(m * math.exp(rng.uniform(math.log(0.04), math.log(0.4))))
    return dict(R=int(rng.integers(8, 17)), T=int(rng.choice([5, 10, 10, 20])), b=[float(x) for x in b],
                m=round(m, 4), s=round(s, 5), kn=kn, gn=gn)


def rand_seed(rng, R):
    r0 = R * rng.uniform(0.6, 1.6)
    a = rng.uniform(0.5, 1.0)
    th = rng.uniform(0, np.pi)
    h = int(2 * r0 + 4)
    y, x = np.mgrid[0:h, 0:h] - h / 2
    u = (x * np.cos(th) + y * np.sin(th)) / r0
    v = (-x * np.sin(th) + y * np.cos(th)) / (r0 * a)
    mask = (u ** 2 + v ** 2) < 1
    P = rng.random((h, h)) * rng.uniform(0.4, 1.0) * mask
    if rng.random() < 0.5:
        P = ndimage.gaussian_filter(P, rng.uniform(0.5, 2.0))
        P = P / max(P.max(), 1e-9) * rng.uniform(0.5, 1.0)
    return P


def soup_seed(rng):
    side = int(rng.integers(32, 97))
    P = rng.random((side, side)) * rng.uniform(0.3, 1.0)
    if rng.random() < 0.3:
        P = ndimage.gaussian_filter(P, rng.uniform(0.5, 1.5)); P = P / max(P.max(), 1e-9) * rng.uniform(0.5, 1.0)
    return P


def mutate(parent, rng):
    p = dict(parent["params"]); p["b"] = list(p["b"])
    k = rng.random()
    big = rng.random() < 0.15
    p["m"] = round(float(p["m"] * math.exp(rng.normal(0, 0.15 if big else 0.04))), 4)
    p["s"] = round(float(p["s"] * math.exp(rng.normal(0, 0.25 if big else 0.08))), 5)
    if k < 0.3:
        i = int(rng.integers(len(p["b"])))
        p["b"][i] = float(np.clip(round((p["b"][i] + rng.normal(0, 0.15)) * 12) / 12, 0, 1))
        if max(p["b"]) == 0: p["b"][i] = 1.0
        mx = max(p["b"]); p["b"] = [round(x / mx, 4) for x in p["b"]]
    elif k < 0.4:
        p["R"] = int(np.clip(p["R"] + rng.choice([-1, 1]), 8, 16))
    elif k < 0.45:
        p["T"] = int(rng.choice([5, 10, 20]))
    elif k < 0.5 and len(p["b"]) < 4:
        p["b"].append(round(float(rng.integers(0, 13)) / 12, 4))
    seed = decode(parent["pattern"]) if rng.random() < 0.75 else rand_seed(rng, p["R"])
    seed = np.clip(seed * (1 + 0.1 * rng.standard_normal(seed.shape)), 0, 1)
    return p, seed


# ------------------------------------------------------------------ patterns
def encode(P):
    q = np.round(np.clip(P, 0, 1) * 255).astype(np.uint8)
    return dict(h=int(q.shape[0]), w=int(q.shape[1]), b64=base64.b64encode(q.tobytes()).decode())


def decode(e):
    return np.frombuffer(base64.b64decode(e["b64"]), np.uint8).reshape(e["h"], e["w"]) / 255.0


def place(P, cy=N // 2, cx=N // 2):
    A = np.zeros((N, N))
    h, w = P.shape
    h, w = min(h, N), min(w, N)
    ys = (np.arange(h) + cy - h // 2) % N
    xs = (np.arange(w) + cx - w // 2) % N
    A[np.ix_(ys, xs)] = P[:h, :w]
    return A


def com(A):
    """Centre of mass on the torus (circular mean)."""
    cy = np.angle((A.sum(1) * np.exp(1j * ANG)).sum()) % (2 * np.pi) * N / (2 * np.pi)
    cx = np.angle((A.sum(0) * np.exp(1j * ANG)).sum()) % (2 * np.pi) * N / (2 * np.pi)
    return cy, cx


def centred(A):
    cy, cx = com(A)
    return np.roll(A, (int(round(N / 2 - cy)), int(round(N / 2 - cx))), (0, 1))


def bodies(A, thr=0.1):
    """Connected components (8-neighbour, wrapping) with >= 10% of total mass."""
    lab, n = ndimage.label(A > thr, structure=np.ones((3, 3)))
    if n == 0: return 0, lab, []
    # merge labels across the torus seams
    parent = list(range(n + 1))
    def f(i):
        while parent[i] != i: parent[i] = parent[parent[i]]; i = parent[i]
        return i
    for a, b in [(lab[0], lab[-1]), (lab[:, 0], lab[:, -1])]:
        for i, j in zip(a, b):
            if i and j: parent[f(i)] = f(j)
    roots = np.array([f(i) for i in range(n + 1)])
    lab = roots[lab]
    ids = np.unique(lab[lab > 0])
    masses = ndimage.sum(A, lab, ids)
    tot = masses.sum()
    big = [(int(i), float(mm)) for i, mm in zip(ids, masses) if mm >= 0.1 * tot]
    return len(big), lab, big


def crop(A, margin=3):
    """Cut the creature out of a centred board."""
    ys, xs = np.nonzero(A > 0.005)
    if len(ys) == 0: return np.zeros((1, 1))
    y0, y1 = max(ys.min() - margin, 0), min(ys.max() + margin + 1, N)
    x0, x1 = max(xs.min() - margin, 0), min(xs.max() + margin + 1, N)
    return A[y0:y1, x0:x1].copy()


# ------------------------------------------------------------------ simulate
def run_batch(params, seeds, steps, warm, kill_early=True, grace=0):
    """Run boards; return per-board records (None if dead/exploded).
    Records per-step mass and torus-unwrapped COM; every 10 steps, angular
    Fourier moments of the body about its COM; every CHECK steps, body count
    and radius of gyration."""
    n = len(params)
    w = World(N, params)
    for i, s in enumerate(seeds): w.A[i] = place(s)
    alive = np.ones(n, bool)
    fate = [None] * n
    mass = np.zeros((n, steps)); pos = np.zeros((n, steps, 2))
    last = np.array([com(w.A[i]) for i in range(n)])
    acc = np.zeros((n, 2))
    fk = [[] for _ in range(n)]; checks = [[] for _ in range(n)]
    idx = np.arange(n)  # board slot -> candidate index
    for t in range(steps):
        A = w.step()
        M = A.sum((1, 2))
        for j, i in enumerate(idx):
            c = np.array(com(A[j]))
            d = (c - last[i] + N / 2) % N - N / 2
            acc[i] += d; last[i] = c
            mass[i, t] = M[j]; pos[i, t] = acc[i]
            if t % 10 == 9 and t >= warm:
                dy = (YY - c[0] + N / 2) % N - N / 2
                dx = (XX - c[1] + N / 2) % N - N / 2
                th = np.arctan2(dy, dx)
                wgt = A[j] * ((dy ** 2 + dx ** 2) > 0.25)
                f0 = max(wgt.sum(), 1e-9)
                fk[i].append([(wgt * np.exp(-1j * k * th)).sum() / f0 for k in range(1, 9)])
        if t % CHECK == CHECK - 1:
            keep = []
            for j, i in enumerate(idx):
                a = A[j]
                area = (a > 0.1).sum()
                if M[j] < 0.02 * params[i]["R"] ** 2:
                    fate[i] = "died"; continue
                if area < 4:
                    fate[i] = "haze"; continue
                if area > 0.2 * N * N and t >= grace:
                    fate[i] = "spread"; continue
                nb, _, _ = bodies(a)
                ac = centred(a)
                rg = math.sqrt((ac * ((YY - N / 2) ** 2 + (XX - N / 2) ** 2)).sum() / ac.sum())
                checks[i].append((t, nb, rg, int(area)))
                if t < grace:
                    keep.append(j); continue
                if kill_early and t >= 300 and rg > N / 4.5 and nb > 3:
                    fate[i] = "scattered"; continue
                keep.append(j)
            if len(keep) < len(idx):
                keep = np.array(keep, int)
                w.A = w.A[keep]; w.KF = w.KF[keep]; w.m = w.m[keep]; w.s = w.s[keep]; w.dt = w.dt[keep]
                w.gn = [w.gn[j] for j in keep]; idx = idx[keep]
                if len(idx) == 0: break
    out = [None] * n
    for j, i in enumerate(idx):
        out[i] = dict(final=w.A[j].copy(), mass=mass[i], pos=pos[i], fk=np.array(fk[i]), checks=checks[i])
    return out, fate


def persistent(rec, warm, steps):
    """Return (ok, reason)."""
    ms = rec["mass"][warm:steps]
    mu = ms.mean()
    if ms.min() < 0.4 * mu or ms.max() > 2.5 * mu: return False, "mass swings"
    q = len(ms) // 4
    if abs(ms[-q:].mean() - ms[:q].mean()) / mu > 0.10: return False, "mass drifting"
    ch = [c for c in rec["checks"] if c[0] >= warm]
    nb = np.array([c[1] for c in ch]); rg = np.array([c[2] for c in ch])
    if (nb == 1).mean() < 0.9 or nb.max() > 2: return False, "not one body"
    if rg.max() > N / 5: return False, "too large"
    if rg[-len(rg) // 4:].mean() > 1.25 * rg[:len(rg) // 4].mean(): return False, "spreading"
    return True, "ok"


def describe(rec, p, warm, steps):
    T = p["T"]; R = p["R"]
    ms = rec["mass"][warm:steps]; pos = rec["pos"][warm:steps]
    n = len(ms)
    # motion: chunk velocities over 20-step blocks
    ch = 20
    P = pos[::ch]
    V = np.diff(P, axis=0) * T / ch  # cells per unit time
    sp = np.hypot(V[:, 0], V[:, 1])
    net = np.hypot(*(pos[-1] - pos[0])) * T / n
    speed = float(np.median(sp))
    moving = speed > 0.02 * R  # > 0.02 R per unit time
    straight = float(net / max(sp.mean(), 1e-9))
    turn = 0.0
    if moving:
        hd = np.unwrap(np.arctan2(-V[:, 0], V[:, 1]))  # y down -> math angle
        turn = float((hd[-1] - hd[0]) / (n / T))  # rad per unit time, + = anticlockwise on screen
    # angular moments
    F = rec["fk"]; amp = np.abs(F).mean(0)
    order = int(np.argmax(amp) + 1) if amp.max() > 0.04 else 0
    rot = 0.0
    if order and np.abs(F[:, order - 1]).min() > 0.02:
        ph = np.unwrap(np.angle(F[:, order - 1]))
        rot = float(-(ph[-1] - ph[0]) / order / (len(ph) * 10 / T))  # body rad per unit time, + = CLOCKWISE on screen (y points down; checked on a synthetic blob)
    # oscillation of mass
    x = ms - ms.mean()
    amp_rel = float(x.std() / ms.mean())
    period, regular = 0.0, 0.0
    if amp_rel > 1e-3:
        ac = np.correlate(x, x, "full")[n - 1:] / (x.var() * np.arange(n, 0, -1))
        lim = n // 3
        pk = [l for l in range(2, lim) if ac[l] > ac[l - 1] and ac[l] >= ac[l + 1] and ac[l] > 0.3]
        if pk:
            l = max(pk[:5], key=lambda l: ac[l])
            # prefer the first peak that is nearly as high as the best (fundamental, not a harmonic)
            l = next(q for q in pk if ac[q] > 0.85 * ac[l])
            period, regular = float(l), float(ac[l])
    rg = np.array([c[2] for c in rec["checks"] if c[0] >= warm])
    area = np.array([c[3] for c in rec["checks"] if c[0] >= warm])
    A = centred(rec["final"])
    span = 0.0
    ys, xs = np.nonzero(A > 0.1)
    if moving and len(ys):
        hdg = math.atan2(-(pos[-1] - pos[-ch * 5])[0], (pos[-1] - pos[-ch * 5])[1])
        proj = (xs - N / 2) * math.cos(hdg) - (ys - N / 2) * math.sin(hdg)
        span = float(proj.max() - proj.min() + 1)
    elif len(ys):
        span = float(max(np.ptp(ys), np.ptp(xs)) + 1)
    return dict(mass=float(ms.mean()), mass_per_R2=float(ms.mean() / R ** 2), speed=speed, speed_R=speed / R,
                net_speed=float(net), straight=straight, moving=bool(moving), turn=turn, rot=rot,
                order=order, sym=[float(a) for a in amp], osc_amp=amp_rel, period_steps=period, regular=regular,
                rg=float(rg.mean()), rg_R=float(rg.mean() / R), area=float(area.mean()), span=span,
                heading=float(math.degrees(math.atan2(-(pos[-1] - pos[0])[0], (pos[-1] - pos[0])[1]))))


def vec(d):
    """Behaviour descriptor for novelty, each axis scaled to roughly unit spread."""
    return np.array([
        d["speed_R"] / 0.15,
        min(abs(d["turn"]), 0.5) / 0.1 if d["moving"] else 0,
        min(abs(d["rot"]), 0.5) / 0.1,
        math.log1p(d["period_steps"]) / 1.5 if d["regular"] > 0.5 else 0,
        min(d["osc_amp"], 0.3) / 0.05,
        math.log(d["mass_per_R2"]) / 0.5,
        d["rg_R"] / 0.3,
        (d["order"] if d["order"] <= 6 else 7) * 0.7,
        (1 - min(d["straight"], 1)) * 2 if d["moving"] else 0,
    ])


def body_of(A):
    """Largest body cut out and centred."""
    nb, lab, big = bodies(A)
    if nb > 1:
        i = max(big, key=lambda x: x[1])[0]
        A = A * (lab == i)
    return crop(centred(A))


def find_edges(cands, rng, iters=7, steps=400):
    """For each rule (m to be found, s = ratio*m), bisect m on [0.05, 0.6] using one fixed soup:
    a soup that dies means m is too high, one that spreads (or hazes) means m too low.
    Returns, per rule, the bracket [lo, hi] and the m values at which the soup did neither."""
    lo = np.full(len(cands), 0.05); hi = np.full(len(cands), 0.6)
    soups = [soup_seed(rng) for _ in cands]
    alive_ms = [[] for _ in cands]
    for _ in range(iters):
        mid = (lo + hi) / 2
        ps = []
        for c, m in zip(cands, mid):
            p = dict(c["params"]); p["m"] = round(float(m), 4); p["s"] = round(float(m * c["ratio"]), 5); ps.append(p)
        recs, fate = run_batch(ps, soups, steps, steps, kill_early=False, grace=steps // 2)
        for i, (r, f) in enumerate(zip(recs, fate)):
            if f == "died": hi[i] = mid[i]
            elif f in ("spread", "haze"): lo[i] = mid[i]
            else:
                alive_ms[i].append(float(mid[i]))
                # something is still there: nudge upward (bodies sit just below the death line)
                lo[i] = mid[i]
    return lo, hi, alive_ms


def evaluate(task):
    """Worker: a list of (cid, params, seed) -> list of result dicts."""
    seed_rng = np.random.default_rng(task["seed"])
    stats = dict(n=len(task["cands"]), soups=0, soup_bodies=0, died=0, haze=0, spread=0, scattered=0,
                 unstable=0, multibody=0, fragile=0, lived=0)
    cands = [c for c in task["cands"] if c.get("kind") not in ("random", "edge")]
    soups = [c for c in task["cands"] if c.get("kind") == "random"]
    edges = [c for c in task["cands"] if c.get("kind") == "edge"]
    if edges:
        lo, hi, alive_ms = find_edges(edges, seed_rng)
        stats["edge_rules"] = len(edges)
        for c, l, h, am in zip(edges, lo, hi, alive_ms):
            ms = list(seed_rng.uniform(l, h, 2)) + (am[-1:] if am else [float(h)])
            for k, m in enumerate(ms):
                p = dict(c["params"]); p["m"] = round(float(m), 4); p["s"] = round(float(m * c["ratio"]), 5)
                soups.append(dict(cid=f"{c['cid']}e{k}", params=p, pattern=encode(soup_seed(seed_rng)), kind="random",
                                  origin="edge", edge_bracket=[float(l), float(h)]))
    if soups:
        # hatch: let each soup condense for HATCH steps, then cut out up to 3 separate bodies
        stats["soups"] = sum(1 for c in soups if c.get("origin") != "edge")
        stats["edge_soups"] = sum(1 for c in soups if c.get("origin") == "edge")
        hr, hf = run_batch([c["params"] for c in soups], [decode(c["pattern"]) for c in soups],
                           HATCH, HATCH, kill_early=False, grace=HATCH // 2)
        for c, r, f in zip(soups, hr, hf):
            if r is None:
                stats[f] += 1; continue
            nb, lab, big = bodies(r["final"])
            if nb == 0:
                stats["haze"] += 1; continue
            for k, (bid, _) in enumerate(sorted(big, key=lambda x: -x[1])[:3]):
                body = crop(centred(r["final"] * (lab == bid)))
                org = c.get("origin", "soup")
                cands.append(dict(cid=f"{c['cid']}.{k}", params=c["params"], pattern=encode(body),
                                  parent=None, kind="hatched", soup=c["pattern"], origin=org))
                stats[org + "_bodies"] = stats.get(org + "_bodies", 0) + 1
    if not cands:
        return dict(stats=stats, found=[])
    params = [c["params"] for c in cands]
    seeds = [decode(c["pattern"]) for c in cands]
    results = []
    recs, fate = run_batch(params, seeds, LONG, WARM)
    second = []
    for i, r in enumerate(recs):
        if r is None:
            stats[fate[i]] += 1; continue
        nb, _, _ = bodies(r["final"])
        if nb > 1:
            second.append((i, body_of(r["final"])))
            continue
        ok, why = persistent(r, WARM, LONG)
        if not ok:
            stats["unstable"] += 1; continue
        results.append((i, r, False))
    if second:
        recs2, fate2 = run_batch([params[i] for i, _ in second], [b for _, b in second], LONG, WARM)
        for (i, _), r in zip(second, recs2):
            if r is None or bodies(r["final"])[0] > 1 or not persistent(r, WARM, LONG)[0]:
                stats["multibody"] += 1; continue
            results.append((i, r, True))
    out = []
    if results:
        # robustness: two perturbed re-seeds per survivor
        rp, rs, who = [], [], []
        for i, r, _ in results:
            body = crop(centred(r["final"]))
            for trial in range(2):
                b = np.clip(body * (1 + 0.1 * seed_rng.standard_normal(body.shape)), 0, 1)
                if trial == 0:
                    b = np.rot90(b, int(seed_rng.integers(1, 4)))
                else:
                    b = np.clip(ndimage.rotate(b, seed_rng.uniform(0, 360), order=1, reshape=True), 0, 1)
                rp.append(params[i]); rs.append(b); who.append(i)
        rr, _ = run_batch(rp, rs, ROBUST, 1000)
        for i, r, born in results:
            d = describe(r, params[i], WARM, LONG)
            trials = [x for x, k in zip(rr, who) if k == i]
            okk = True
            for x in trials:
                if x is None or bodies(x["final"])[0] > 1 or not persistent(x, 1000, ROBUST)[0]:
                    okk = False; break
                dx = describe(x, params[i], 1000, ROBUST)
                if abs(dx["mass"] / d["mass"] - 1) > 0.25 or dx["moving"] != d["moving"] or \
                        (d["moving"] and abs(dx["speed"] / d["speed"] - 1) > 0.3):
                    okk = False; break
            if not okk:
                stats["fragile"] += 1; continue
            stats["lived"] += 1
            nbs = [c[1] for c in r["checks"]]
            org = cands[i].get("origin", "mutant" if cands[i].get("parent") else "soup")
            stats["lived_" + org] = stats.get("lived_" + org, 0) + 1
            out.append(dict(cid=cands[i]["cid"], parent=cands[i].get("parent"), origin=org,
                            edge_bracket=cands[i].get("edge_bracket"), params=params[i],
                            seed_pattern=cands[i]["pattern"], soup=cands[i].get("soup"), pattern=encode(crop(centred(r["final"]))),
                            desc=d, born_by_division=born, max_bodies_seen=int(max(nbs))))
    return dict(stats=stats, found=out)


# ------------------------------------------------------------------ master
def main(hours, outdir, workers=4, batch=24):
    os.makedirs(outdir, exist_ok=True)
    rng = np.random.default_rng(int(time.time()))
    surv_f = open(os.path.join(outdir, "survivors.jsonl"), "a")
    log_f = open(os.path.join(outdir, "search_log.jsonl"), "a")
    archive = []  # list of survivor dicts
    avec = []
    totals = dict(n=0, soups=0, soup_bodies=0, haze=0, died=0, spread=0, scattered=0, unstable=0, multibody=0, fragile=0, lived=0,
                  random=0, mutant=0, lived_random=0, lived_mutant=0, archived=0)
    thresh = 1.0
    cid = [0]
    # resume: rebuild archive and counters from a previous run in the same directory
    sp = os.path.join(outdir, "survivors.jsonl"); lp = os.path.join(outdir, "search_log.jsonl")
    if os.path.exists(lp) and os.path.getsize(lp):
        last = [json.loads(l) for l in open(lp) if '"archive"' in l]
        if last:
            prev = last[-1]; thresh = prev["thresh"]
            totals.update({k: v for k, v in prev.items() if k not in ("t", "archive", "thresh")})
    if os.path.exists(sp):
        for l in open(sp):
            f = json.loads(l)
            try: cid[0] = max(cid[0], int(str(f["cid"]).split(".")[0].split("e")[0]))
            except ValueError: pass
            if f.get("archived"):
                archive.append(f); avec.append(vec(f["desc"])); f["_nov"] = 1.0
    cid[0] += 100000 if cid[0] else 0
    t_end = time.time() + hours * 3600

    def novelty(v, k=5):
        if not avec: return 1e9
        d = np.sort(np.linalg.norm(np.array(avec) - v, axis=1))
        return float(d[:k].mean())

    def make_task():
        cands = []
        for _ in range(batch):
            cid[0] += 1
            r = rng.random()
            if archive and r < 0.5:
                nov = np.array([a["_nov"] for a in archive]) + 1e-3
                par = archive[int(rng.choice(len(archive), p=nov / nov.sum()))]
                p, s = mutate(par, rng)
                cands.append(dict(cid=cid[0], params=p, pattern=encode(s), parent=par["cid"], kind="mutant"))
                totals["mutant"] += 1
            elif r < (0.7 if archive else 0.6):
                p = rand_params(rng)
                cands.append(dict(cid=cid[0], params=p, ratio=p["s"] / p["m"], kind="edge"))
                totals["edge"] = totals.get("edge", 0) + 1
            else:
                p = rand_params(rng)
                cands.append(dict(cid=cid[0], params=p, pattern=encode(soup_seed(rng)), kind="random"))
                totals["random"] += 1
        return dict(cands=cands, seed=int(rng.integers(1 << 30)))

    with Pool(workers) as pool:
        pending = [pool.apply_async(evaluate, (make_task(),)) for _ in range(workers * 2)]
        kinds = {}
        while pending:
            done = [p for p in pending if p.ready()]
            if not done:
                time.sleep(0.5); continue
            for p in done:
                pending.remove(p)
                try:
                    res = p.get()
                except Exception as e:
                    log_f.write(json.dumps(dict(error=repr(e))) + "\n"); totals["errors"] = totals.get("errors", 0) + 1
                    if time.time() < t_end: pending.append(pool.apply_async(evaluate, (make_task(),)))
                    continue
                for k, v in res["stats"].items(): totals[k] = totals.get(k, 0) + v
                for f in res["found"]:
                    v = vec(f["desc"]); nov = novelty(v)
                    f["novelty_at_birth"] = nov if nov < 1e8 else None
                    f["archived"] = bool(nov > thresh)
                    surv_f.write(json.dumps(f) + "\n"); surv_f.flush()
                    if f["archived"]:
                        f["_nov"] = min(nov, 5.0); archive.append(f); avec.append(v); totals["archived"] += 1
                # refresh novelty of archive members (sparseness within archive)
                if len(avec) > 1:
                    M = np.array(avec)
                    D = np.linalg.norm(M[:, None] - M[None], axis=2)
                    D.sort(1)
                    for a, row in zip(archive, D):
                        a["_nov"] = float(row[1:6].mean())
                if time.time() < t_end:
                    pending.append(pool.apply_async(evaluate, (make_task(),)))
            # adapt threshold: aim for a steady trickle into the archive
            log_f.write(json.dumps(dict(t=time.time(), archive=len(archive), thresh=thresh, **totals)) + "\n"); log_f.flush()
            recent = totals["archived"]
            if totals["lived"] > 20:
                rate = totals["archived"] / totals["lived"]
                thresh *= 1.05 if rate > 0.3 else 0.97 if rate < 0.1 else 1.0
    json.dump(dict(totals=totals, archive=[a["cid"] for a in archive]), open(os.path.join(outdir, "summary.json"), "w"))
    print(totals)


if __name__ == "__main__":
    main(float(sys.argv[1]), sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 4)
