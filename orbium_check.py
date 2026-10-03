"""Reproduce Orbium unicaudatus from Chan's animals.json and check it glides.
Also cross-checks my FFT update against a literal transcription of the
reference calc_once (centred kernel + fftn + fftshift)."""
import json, sys, numpy as np
from lenia import World, rle2arr, place, KERNEL_CORE, GROWTH

animals = json.load(open(sys.argv[1]))
orb = next(a for a in animals if a["name"] == "Orbium unicaudatus")
P = dict(orb["params"]); P["b"] = [1.0]
cells = rle2arr(orb["cells"])
print("params", orb["params"], "pattern", cells.shape, "mass", cells.sum().round(3))
N = 128
w = World(N, [P]); w.A[0] = place(N, cells)

# literal reference transcription
def ref_kernel(N, R):
    I, J = np.mgrid[0:N, 0:N]
    D = np.sqrt((I - N//2)**2 + (J - N//2)**2) / R
    with np.errstate(all="ignore"):
        K = (D < 1) * np.nan_to_num(KERNEL_CORE[1](np.minimum(D % 1, 1)))
    return np.fft.fftn(K / K.sum())
KF = ref_kernel(N, 13); A = w.A[0].copy()
for t in range(200):
    U = np.fft.fftshift(np.real(np.fft.ifftn(KF * np.fft.fftn(A))))
    A = np.clip(A + 0.1 * GROWTH[1](U, 0.15, 0.015), 0, 1)
B = World(N, [P]); B.A[0] = place(N, cells)
for t in range(200): B.step()
print("max |mine - reference transcription| after 200 steps:", np.abs(B.A[0] - A).max())

# glide test
ang = 2*np.pi*np.arange(N)/N
def com(a):
    m = a.sum()
    cy = np.angle((a.sum(1) * np.exp(1j*ang)).sum()) % (2*np.pi) * N/(2*np.pi)
    cx = np.angle((a.sum(0) * np.exp(1j*ang)).sum()) % (2*np.pi) * N/(2*np.pi)
    return m, cy, cx
traj = []
for t in range(5001):
    if t: w.step()
    traj.append(com(w.A[0]))
traj = np.array(traj)
d = np.diff(traj[:, 1:], axis=0); d = (d + N/2) % N - N/2
pos = np.cumsum(d, axis=0)
for t0, t1 in [(0,1000),(1000,2000),(2000,3000),(3000,4000),(4000,5000)]:
    disp = pos[t1-1] - (pos[t0-1] if t0 else 0)
    ms = traj[t0:t1, 0]
    print(f"steps {t0:4d}-{t1:4d}: mass {ms.mean():.2f} (min {ms.min():.2f}, max {ms.max():.2f})  "
          f"moved {np.hypot(*disp):.2f} cells, heading {np.degrees(np.arctan2(-disp[0], disp[1])):.1f} deg")
v = np.hypot(*(pos[-1] - pos[999])) / 4000
print(f"speed {v:.4f} cells/step = {v*10:.4f} cells per unit time = {v*10/13:.5f} R per unit time")
