// Lenia in plain JS: same kernel, growth and update as garden/lenia.py.
// 2-D FFT convolution on an N x N torus (N a power of two), float64 throughout.
(function (root) {
  "use strict";
  const CORE = {
    1: (r) => Math.pow(4 * r * (1 - r), 4),
    2: (r) => { const q = r * (1 - r); return q > 1e-9 ? Math.exp(4 - 1 / q) : 0; },
  };
  const GROW = {
    1: (u, m, s) => { const v = Math.max(0, 1 - (u - m) * (u - m) / (9 * s * s)); return 2 * v * v * v * v - 1; },
    2: (u, m, s) => 2 * Math.exp(-(u - m) * (u - m) / (2 * s * s)) - 1,
  };

  function makeFFT(n) {
    const lev = Math.log2(n) | 0;
    const rev = new Uint32Array(n);
    for (let i = 0; i < n; i++) { let x = i, y = 0; for (let k = 0; k < lev; k++) { y = (y << 1) | (x & 1); x >>= 1; } rev[i] = y; }
    const cos = new Float64Array(n / 2), sin = new Float64Array(n / 2);
    for (let i = 0; i < n / 2; i++) { cos[i] = Math.cos(2 * Math.PI * i / n); sin[i] = Math.sin(2 * Math.PI * i / n); }
    // in-place complex FFT of re/im at offset off with stride st; inv=true -> conjugate twiddles (unscaled)
    return function (re, im, off, st, inv) {
      for (let i = 0; i < n; i++) {
        const j = rev[i];
        if (j > i) {
          const a = off + i * st, b = off + j * st;
          let t = re[a]; re[a] = re[b]; re[b] = t; t = im[a]; im[a] = im[b]; im[b] = t;
        }
      }
      const sg = inv ? 1 : -1;
      for (let size = 2; size <= n; size <<= 1) {
        const half = size >> 1, step = n / size;
        for (let i = 0; i < n; i += size) {
          for (let j = 0, k = 0; j < half; j++, k += step) {
            const a = off + (i + j) * st, b = a + half * st;
            const wr = cos[k], wi = sg * sin[k];
            const tr = re[b] * wr - im[b] * wi, ti = re[b] * wi + im[b] * wr;
            re[b] = re[a] - tr; im[b] = im[a] - ti; re[a] += tr; im[a] += ti;
          }
        }
      }
    };
  }

  function fft2(re, im, N, fft, inv) {
    for (let r = 0; r < N; r++) fft(re, im, r * N, 1, inv);
    for (let c = 0; c < N; c++) fft(re, im, c, N, inv);
  }

  class Lenia {
    constructor(N, p) {
      this.N = N; this.p = p;
      this.fft = makeFFT(N);
      this.A = new Float64Array(N * N);
      this.re = new Float64Array(N * N); this.im = new Float64Array(N * N);
      this.dt = 1 / p.T;
      this.g = GROW[p.gn || 1];
      // kernel, origin at [0,0]
      const kr = new Float64Array(N * N), ki = new Float64Array(N * N);
      const core = CORE[p.kn || 1], b = p.b, B = b.length;
      let sum = 0;
      for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) {
        const dy = y < N / 2 ? y : y - N, dx = x < N / 2 ? x : x - N;
        const D = Math.sqrt(dx * dx + dy * dy) / p.R;
        if (D >= 1) continue;
        const Br = B * D, i = Math.min(Math.floor(Br), B - 1);
        const v = core(Math.min(Br % 1, 1)) * b[i];
        kr[y * N + x] = v; sum += v;
      }
      for (let i = 0; i < N * N; i++) kr[i] /= sum * N * N; // fold the inverse-FFT 1/N^2 in here
      fft2(kr, ki, N, this.fft, false);
      this.kr = kr; this.ki = ki;
    }
    place(pat, h, w, cy, cx) {
      const N = this.N;
      cy = cy === undefined ? N >> 1 : cy; cx = cx === undefined ? N >> 1 : cx;
      for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
        const Y = (((y + cy - (h >> 1)) % N) + N) % N, X = (((x + cx - (w >> 1)) % N) + N) % N;
        this.A[Y * N + X] = pat[y * w + x];
      }
    }
    step() {
      const N = this.N, NN = N * N, re = this.re, im = this.im, A = this.A;
      re.set(A); im.fill(0);
      fft2(re, im, N, this.fft, false);
      const kr = this.kr, ki = this.ki;
      for (let i = 0; i < NN; i++) {
        const a = re[i], b = im[i];
        re[i] = a * kr[i] - b * ki[i]; im[i] = a * ki[i] + b * kr[i];
      }
      fft2(re, im, N, this.fft, true);
      const m = this.p.m, s = this.p.s, dt = this.dt, g = this.g;
      for (let i = 0; i < NN; i++) {
        let v = A[i] + dt * g(re[i], m, s);
        A[i] = v < 0 ? 0 : v > 1 ? 1 : v;
      }
    }
    mass() { let s = 0; for (let i = 0; i < this.A.length; i++) s += this.A[i]; return s; }
  }

  function decodePattern(e) {
    const bin = typeof atob === "function" ? atob(e.b64) : Buffer.from(e.b64, "base64").toString("binary");
    const out = new Float64Array(e.h * e.w);
    for (let i = 0; i < out.length; i++) out[i] = bin.charCodeAt(i) / 255;
    return out;
  }

  root.Lenia = Lenia; root.decodePattern = decodePattern;
  if (typeof module !== "undefined") module.exports = { Lenia, decodePattern };
})(typeof window !== "undefined" ? window : globalThis);
