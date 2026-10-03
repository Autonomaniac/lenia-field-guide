"""Lenia core, following Bert Chan's reference LeniaND.py (github.com/Chakazul/Lenia).

World: 2-D torus of values in [0,1].
Step:  U = K * A (FFT convolution, K normalised to sum 1)
       A <- clip(A + dt * G(U), 0, 1),   dt = 1/T
Kernel: concentric rings. For B rings with peak weights b[i], a cell at
       normalised distance r (= dist/R, r<1) gets  b[floor(B r)] * core(B r mod 1).
Cores and growth functions are indexed exactly as in the reference:
       kn=1 polynomial (4r(1-r))^4      kn=2 exponential exp(4 - 1/(r(1-r)))
       gn=1 polynomial 2*max(0,1-(u-m)^2/(9s^2))^4 - 1
       gn=2 gaussian   2*exp(-(u-m)^2/(2s^2)) - 1
"""
import numpy as np

KERNEL_CORE = {
    1: lambda r: (4 * r * (1 - r)) ** 4,
    2: lambda r: np.exp(4 - 1 / np.maximum(r * (1 - r), 1e-9)),
}
GROWTH = {
    1: lambda u, m, s: np.maximum(0, 1 - (u - m) ** 2 / (9 * s ** 2)) ** 4 * 2 - 1,
    2: lambda u, m, s: np.exp(-(u - m) ** 2 / (2 * s ** 2)) * 2 - 1,
}


def kernel_fft(N, R, b, kn):
    """Normalised kernel on an N x N torus, returned as rfft2 (origin at [0,0])."""
    x = np.fft.fftfreq(N, 1 / N)  # 0,1,..,N/2-1,-N/2..-1 : distances from origin
    X, Y = np.meshgrid(x, x)
    D = np.sqrt(X ** 2 + Y ** 2) / R
    b = np.asarray(b, dtype=float)
    B = len(b)
    Br = B * D
    idx = np.minimum(np.floor(Br).astype(int), B - 1)
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        core = KERNEL_CORE[kn](np.minimum(Br % 1, 1))
    K = (D < 1) * np.nan_to_num(core) * b[idx]
    K = K / K.sum()
    return np.fft.rfft2(K)


class World:
    """One or a batch of boards sharing a grid size. Params may differ per board."""

    def __init__(self, N, params_list):
        self.N = N
        self.params = params_list
        self.KF = np.stack([kernel_fft(N, p["R"], p["b"], p.get("kn", 1)) for p in params_list])
        self.m = np.array([p["m"] for p in params_list])[:, None, None]
        self.s = np.array([p["s"] for p in params_list])[:, None, None]
        self.dt = np.array([1.0 / p["T"] for p in params_list])[:, None, None]
        self.gn = [p.get("gn", 1) for p in params_list]
        self.A = np.zeros((len(params_list), N, N))

    def step(self):
        U = np.fft.irfft2(np.fft.rfft2(self.A) * self.KF, s=(self.N, self.N))
        if all(g == self.gn[0] for g in self.gn):
            G = GROWTH[self.gn[0]](U, self.m, self.s)
        else:
            G = np.empty_like(U)
            for i, g in enumerate(self.gn):
                G[i] = GROWTH[g](U[i], self.m[i], self.s[i])
        self.A = np.clip(self.A + self.dt * G, 0, 1)
        return self.A


# ---------- Chan's RLE (2-D only) -------------------------------------------
def _ch2val(c):
    if c in ".b":
        return 0
    if c == "o":
        return 255
    if len(c) == 1:
        return ord(c) - ord("A") + 1
    return (ord(c[0]) - ord("p")) * 24 + (ord(c[1]) - ord("A") + 25)


def rle2arr(st):
    rows, row = [], []
    last, count = "", ""
    st = st.rstrip("!") + "$"
    for ch in st:
        if ch.isdigit():
            count += ch
        elif ch in "pqrstuvwxy@":
            last = ch
        else:
            if last + ch == "$":
                rows.append(row)
                rows.extend([[]] * (int(count) - 1 if count else 0))
                row = []
            else:
                row.extend([_ch2val(last + ch) / 255] * (int(count) if count else 1))
            last, count = "", ""
    w = max(len(r) for r in rows)
    return np.array([r + [0] * (w - len(r)) for r in rows])


def place(N, pattern, cy=None, cx=None):
    A = np.zeros((N, N))
    h, w = pattern.shape
    cy = N // 2 if cy is None else cy
    cx = N // 2 if cx is None else cx
    ys = (np.arange(h) + cy - h // 2) % N
    xs = (np.arange(w) + cx - w // 2) % N
    A[np.ix_(ys, xs)] = pattern
    return A
