// Field-guide runtime: every plate runs its specimen live from the saved pattern.
(function () {
  "use strict";
  const N = 128;
  const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  const hex = (c) => {
    c = c.replace("#", "");
    if (c.length === 3) c = c.split("").map((x) => x + x).join("");
    const n = parseInt(c, 16);
    return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
  };
  let PAL = {};
  function readPalette() {
    PAL = { paper: hex(css("--paper")), ink: hex(css("--ink")), soft: hex(css("--ink-soft")), rule: hex(css("--rule")),
      accent: hex(css("--accent")), washes: [] };
    for (let i = 1; i <= 8; i++) PAL.washes.push(hex(css("--wash-" + i)));
    plates.forEach((p) => { p.lut = makeLUT(PAL.washes[p.washIdx]); p.dirty = true; drawFigures(p); });
  }
  function makeLUT(w) {
    // density -> colour: paper, through the species' wash, deepening to ink at the core
    const lut = new Uint8ClampedArray(256 * 3);
    for (let i = 0; i < 256; i++) {
      const a = i / 255, t = Math.min(1, Math.pow(a, 0.75) * 1.3), k = a > 0.55 ? (a - 0.55) / 0.45 * 0.7 : 0;
      for (let c = 0; c < 3; c++) {
        const v = PAL.paper[c] * (1 - t) + w[c] * t;
        lut[i * 3 + c] = v * (1 - k) + PAL.ink[c] * k;
      }
    }
    return lut;
  }

  const plates = [];
  const ang = new Float64Array(N);
  for (let i = 0; i < N; i++) ang[i] = 2 * Math.PI * i / N;
  const cosA = ang.map(Math.cos), sinA = ang.map(Math.sin);
  function com(A) {
    let ys = 0, yc = 0, xs = 0, xc = 0;
    for (let y = 0; y < N; y++) {
      let row = 0;
      for (let x = 0; x < N; x++) { const v = A[y * N + x]; row += v; xs += v * sinA[x]; xc += v * cosA[x]; }
      ys += row * sinA[y]; yc += row * cosA[y];
    }
    const f = (s, c) => ((Math.atan2(s, c) % (2 * Math.PI)) + 2 * Math.PI) % (2 * Math.PI) * N / (2 * Math.PI);
    return [f(ys, yc), f(xs, xc)];
  }

  function setup(el) {
    const spec = SPECIMENS.find((s) => s.id === el.dataset.id);
    if (!spec) return;
    const cv = el.querySelector("canvas.world");
    const p = { el, spec, cv, ctx: cv.getContext("2d"), img: null, running: true, visible: false, track: false,
      speed: 1, washIdx: (spec.wash || 1) - 1, step: 0, dirty: true, lut: null, W: N };
    p.scale = el.querySelector(".scale i");
    p.sizeScale = () => { if (p.scale) p.scale.style.width = (cv.clientWidth * spec.params.R / p.W).toFixed(1) + "px"; };
    p.setView = (lens) => {
      p.track = lens; p.W = lens ? N / 2 : N;
      cv.width = p.W; cv.height = p.W; p.img = p.ctx.createImageData(p.W, p.W);
      p.sizeScale();
      const b = el.querySelector('[data-act="track"]');
      if (b) b.textContent = lens ? "Show the whole world" : "Under the lens";
      p.dirty = true;
    };
    p.setView(!spec.moving);
    p.pattern = decodePattern(spec.pattern);
    p.reset = () => { p.sim = new Lenia(N, spec.params); p.sim.place(p.pattern, spec.pattern.h, spec.pattern.w); p.step = 0; p.dirty = true; };
    p.reset();
    p.clock = el.querySelector(".clock");
    el.querySelectorAll("[data-act]").forEach((b) => b.addEventListener("click", () => {
      const act = b.dataset.act;
      if (act === "play") { p.running = !p.running; b.setAttribute("aria-pressed", String(!p.running)); b.textContent = p.running ? "Pause" : "Resume"; }
      if (act === "reset") p.reset();
      if (act === "track") p.setView(!p.track);
      if (act === "fast") { p.speed = p.speed === 1 ? 4 : 1; b.setAttribute("aria-pressed", String(p.speed > 1)); }
    }));
    plates.push(p);
  }

  function render(p) {
    const A = p.sim.A, d = p.img.data, lut = p.lut, W = p.W;
    let oy = 0, ox = 0;
    if (p.track) { const c = com(A); oy = Math.round(c[0] - W / 2); ox = Math.round(c[1] - W / 2); }
    for (let y = 0; y < W; y++) {
      const sy = ((y + oy) % N + N) % N;
      for (let x = 0; x < W; x++) {
        const sx = ((x + ox) % N + N) % N;
        const v = (A[sy * N + sx] * 255) | 0, o = (y * W + x) * 4;
        d[o] = lut[v * 3]; d[o + 1] = lut[v * 3 + 1]; d[o + 2] = lut[v * 3 + 2]; d[o + 3] = 255;
      }
    }
    p.ctx.putImageData(p.img, 0, 0);
    if (p.clock) {
      const T = p.spec.params.T;
      p.clock.textContent = "step " + p.step.toLocaleString("en-GB") + " · t = " + Math.round(p.step / T).toLocaleString("en-GB");
    }
    p.dirty = false;
  }

  // ---- small line figures: kernel profile, growth curve, mass record
  function line(cv, xs, ys, opts) {
    const r = cv.getBoundingClientRect(), dpr = window.devicePixelRatio || 1;
    cv.width = Math.max(1, Math.round(r.width * dpr)); cv.height = Math.max(1, Math.round(r.height * dpr));
    const c = cv.getContext("2d"), W = cv.width, H = cv.height, pad = 4 * dpr;
    c.clearRect(0, 0, W, H);
    const x0 = Math.min(...xs), x1 = Math.max(...xs);
    const y0 = opts.ymin !== undefined ? opts.ymin : Math.min(...ys), y1 = opts.ymax !== undefined ? opts.ymax : Math.max(...ys);
    const X = (x) => pad + (x - x0) / (x1 - x0 || 1) * (W - 2 * pad);
    const Y = (y) => H - pad - (y - y0) / (y1 - y0 || 1) * (H - 2 * pad);
    const rgb = (a) => "rgb(" + a.join(",") + ")";
    c.lineWidth = dpr; c.strokeStyle = rgb(PAL.rule);
    if (opts.zero !== undefined) { c.beginPath(); c.moveTo(pad, Y(opts.zero)); c.lineTo(W - pad, Y(opts.zero)); c.stroke(); }
    if (opts.mark !== undefined) { c.setLineDash([2 * dpr, 3 * dpr]); c.beginPath(); c.moveTo(X(opts.mark), pad); c.lineTo(X(opts.mark), H - pad); c.stroke(); c.setLineDash([]); }
    c.strokeStyle = rgb(PAL.ink); c.lineWidth = 1.3 * dpr; c.beginPath();
    xs.forEach((x, i) => (i ? c.lineTo(X(x), Y(ys[i])) : c.moveTo(X(x), Y(ys[i]))));
    c.stroke();
  }
  function drawFigures(p) {
    const s = p.spec, P = s.params, el = p.el;
    const k = el.querySelector("canvas.kernel"), g = el.querySelector("canvas.growth"), m = el.querySelector("canvas.mass");
    if (k) {
      const xs = [], ys = [], B = P.b.length;
      for (let i = 0; i <= 200; i++) {
        const r = i / 200, Br = B * r, j = Math.min(Math.floor(Br), B - 1), q = Math.min(Br % 1, 1);
        const core = P.kn === 2 ? (q * (1 - q) > 1e-9 ? Math.exp(4 - 1 / (q * (1 - q))) : 0) : Math.pow(4 * q * (1 - q), 4);
        xs.push(r); ys.push(r < 1 ? core * P.b[j] : 0);
      }
      line(k, xs, ys, { ymin: 0, ymax: 1 });
    }
    if (g) {
      const xs = [], ys = [], hi = Math.min(1, P.m + 6 * P.s + 0.05);
      for (let i = 0; i <= 240; i++) {
        const u = hi * i / 240;
        const v = P.gn === 2 ? 2 * Math.exp(-(u - P.m) * (u - P.m) / (2 * P.s * P.s)) - 1
          : 2 * Math.pow(Math.max(0, 1 - (u - P.m) * (u - P.m) / (9 * P.s * P.s)), 4) - 1;
        xs.push(u); ys.push(v);
      }
      line(g, xs, ys, { ymin: -1.05, ymax: 1.05, zero: 0, mark: P.m });
    }
    if (m && s.mass_trace) {
      const ys = s.mass_trace, xs = ys.map((_, i) => i);
      const mn = Math.min(...ys), mx = Math.max(...ys), mid = (mn + mx) / 2, half = Math.max((mx - mn) / 2, mid * 0.02);
      line(m, xs, ys, { ymin: mid - half * 1.15, ymax: mid + half * 1.15 });
    }
  }

  // ---- one animation loop for the whole book, on a frame-time budget
  let msPerStep = 3, rr = 0;
  function frame() {
    const live = plates.filter((p) => p.visible && p.running);
    const budget = 11;
    let allowed = Math.max(1, Math.floor(budget / msPerStep));
    const t0 = performance.now();
    let n = 0;
    for (let i = 0; i < live.length && allowed > 0; i++) {
      const p = live[(rr + i) % live.length];
      for (let s = 0; s < p.speed && allowed > 0; s++) { p.sim.step(); p.step++; n++; allowed--; }
      p.dirty = true;
    }
    rr++;
    if (n) msPerStep = msPerStep * 0.9 + 0.1 * ((performance.now() - t0) / n);
    plates.forEach((p) => { if (p.visible && p.dirty) render(p); });
    requestAnimationFrame(frame);
  }

  function boot() {
    document.querySelectorAll("[data-id]").forEach(setup);
    readPalette();
    const io = new IntersectionObserver((es) => es.forEach((e) => {
      const p = plates.find((q) => q.el === e.target); if (p) { p.visible = e.isIntersecting; p.dirty = true; }
    }), { rootMargin: "100px" });
    plates.forEach((p) => { io.observe(p.el); render(p); p.sizeScale(); });
    const reduce = window.matchMedia("(prefers-reduced-motion: reduce)");
    if (reduce.matches) plates.forEach((p) => {
      p.running = false; const b = p.el.querySelector('[data-act="play"]');
      if (b) { b.textContent = "Resume"; b.setAttribute("aria-pressed", "true"); }
    });
    window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", readPalette);
    new MutationObserver(readPalette).observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
    let rt; window.addEventListener("resize", () => { clearTimeout(rt); rt = setTimeout(() => plates.forEach((p) => { drawFigures(p); p.sizeScale(); }), 150); });
    requestAnimationFrame(frame);
  }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", boot); else boot();
})();
