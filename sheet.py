"""Contact sheets so I can look at what the search found.
usage: python3 sheet.py survivors.jsonl out.png [cid ...]   (default: archived survivors)
Each row: one specimen, frames at steps 0, 100, 300, 600, 1000, 2000 (board recentred on its
centre of mass), plus a trail of its centre of mass over 2000 steps."""
import json, sys
import numpy as np
from PIL import Image, ImageDraw
from search import N, World, place, decode, centred, com

FRAMES = [0, 100, 300, 600, 1000, 2000]
INK = np.array([40, 30, 20]); PAPER = np.array([245, 238, 222])


def tone(A):
    a = np.clip(A, 0, 1)[..., None] ** 0.8
    return (PAPER * (1 - a) + INK * a).astype(np.uint8)


def rows(specs, out, scale=1):
    W = N * (len(FRAMES) + 1); H = (N + 14) * len(specs)
    img = Image.new("RGB", (W, H), tuple(PAPER)); dr = ImageDraw.Draw(img)
    for r, f in enumerate(specs):
        p = f["params"]; w = World(N, [p]); w.A[0] = place(decode(f["pattern"]))
        y0 = r * (N + 14)
        d = f["desc"]
        dr.text((2, y0), f"{f['cid']} R{p['R']} T{p['T']} k{p['kn']}g{p['gn']} b{[round(x,2) for x in p['b']]} m{p['m']} s{p['s']} | "
                f"v/R {d['speed_R']:.3f} turn {d['turn']:.3f} rot {d['rot']:.3f} ord {d['order']} osc {d['osc_amp']:.3f} per {d['period_steps']:.0f}",
                fill=(120, 30, 20))
        trail = []; last = np.array(com(w.A[0])); acc = np.zeros(2)
        for t in range(FRAMES[-1] + 1):
            if t:
                w.step(); c = np.array(com(w.A[0])); acc += (c - last + N / 2) % N - N / 2; last = c
            trail.append(acc.copy())
            if t in FRAMES:
                k = FRAMES.index(t)
                img.paste(Image.fromarray(tone(centred(w.A[0])[N // 4:3 * N // 4, N // 4:3 * N // 4])).resize((N, N), Image.NEAREST), (k * N, y0 + 14))
        tr = np.array(trail); span = max(np.abs(tr).max(), 1)
        x0 = len(FRAMES) * N
        pts = [(x0 + N / 2 + q[1] / span * N * 0.45, y0 + 14 + N / 2 + q[0] / span * N * 0.45) for q in tr[::5]]
        dr.line(pts, fill=(120, 30, 20)); dr.text((x0 + 2, y0 + 16), f"trail, {span:.0f} cells", fill=(80, 80, 80))
    img.save(out)


if __name__ == "__main__":
    S = [json.loads(l) for l in open(sys.argv[1])]
    ids = sys.argv[3:]
    specs = [f for f in S if str(f["cid"]) in ids] if ids else [f for f in S if f.get("archived")]
    rows(specs, sys.argv[2])
    print(len(specs), "rows")
