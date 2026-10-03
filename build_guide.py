"""Assemble field-guide.html from the template, the prose (guide_content.html),
the catalogue (catalogue.json: names + field notes I wrote) and the measured data
(specimens/*.json). Measurements on each plate are formatted from the data here,
never typed by hand."""
import glob, html, json, math, os, re

HERE = os.path.dirname(os.path.abspath(__file__))
ROMAN = ["", "I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X", "XI", "XII", "XIII", "XIV", "XV",
         "XVI", "XVII", "XVIII", "XIX", "XX", "XXI", "XXII", "XXIII", "XXIV"]


def fmt(x, d=2):
    return f"{x:,.{d}f}"


def frac(b):
    from fractions import Fraction
    out = []
    for x in b:
        f = Fraction(x).limit_denominator(12)
        out.append(str(f) if abs(float(f) - x) < 1e-3 else f"{x:.2f}")
    return "[" + ", ".join(out) + "]"


def measures(s):
    d = s["desc"]; p = s["params"]; T = p["T"]
    rows = []
    rows.append(("Mass", f"{fmt(d['mass'], 1)} (range {fmt(s['mass_range'][0], 1)}–{fmt(s['mass_range'][1], 1)})"))
    rows.append(("Body", f"{fmt(d['span'], 0)} cells across ({fmt(d['span'] / p['R'], 2)} R)"))
    if d["moving"]:
        v = d["speed"] / T  # cells per step
        bl = d["span"] / v if v else 0
        rows.append(("Speed", f"{fmt(v, 3)} cells/step; one body length every {fmt(bl, 0)} steps"))
        turn = math.degrees(d["turn"]) / T * 100
        if abs(turn) < 0.05:
            rows.append(("Course", "straight (heading changes < 0.05° per 100 steps)"))
        else:
            rows.append(("Course", f"turns {fmt(abs(turn), 2)}° per 100 steps, {'anticlockwise' if turn > 0 else 'clockwise'}"))
    else:
        rows.append(("Speed", "stationary (drift below 0.02 R per unit time)"))
    rot = math.degrees(d["rot"]) / T * 100
    if abs(rot) > 0.05 and not d["moving"]:
        per = 36000 / abs(rot)
        rows.append(("Spin", f"{fmt(abs(rot), 2)}° per 100 steps, {'clockwise' if rot > 0 else 'anticlockwise'} (a full turn in {fmt(per, 0)} steps)"))
    if d["osc_amp"] >= 0.005 and d["regular"] >= 0.5:
        rows.append(("Pulse", f"mass swings ±{fmt(d['osc_amp'] * 100, 1)}% with period {fmt(d['period_steps'], 0)} steps"))
    elif d["osc_amp"] >= 0.005:
        rows.append(("Pulse", f"mass varies ±{fmt(d['osc_amp'] * 100, 1)}%, without a clean period"))
    else:
        rows.append(("Pulse", f"steady (mass varies ±{fmt(d['osc_amp'] * 100, 2)}%)"))
    g = s.get("graded")
    if g is not None:
        rows.append(("Flesh", f"{fmt(g * 100, 0)}% of body cells graded; the rest pinned at 0 or 1"))
    se = s.get("sensitivity")
    if se:
        if se["end"] > 1e-3 and se["efold_steps"]:
            rows.append(("Temper", f"chaotic: a nudge of {se['start']:.0e} grows e-fold every {fmt(se['efold_steps'], 0)} steps"))
        elif se["end"] > 10 * se["start"]:
            rows.append(("Temper", f"steady, with slow drift: a nudge of {se['start']:.0e} grew to {se['end']:.0e} in 1,500 steps"))
        else:
            rows.append(("Temper", f"orderly: a nudge of {se['start']:.0e} was {se['end']:.0e} after 1,500 steps"))
    rows.append(("Mirror", f"{fmt(s['mirror'], 2)} correlation with its own reflection"))
    rows.append(("Proven", "10,000 steps alone; two perturbed re-seedings of 3,000 steps"))
    return rows


def plate(i, s):
    p = s["params"]
    ks = "polynomial" if p["kn"] == 1 else "exponential"
    gs = "polynomial" if p["gn"] == 1 else "gaussian"
    rule = (f"R = {p['R']} cells · T = {p['T']} (dt = {1 / p['T']:g}) · rings b = {frac(p['b'])} · "
            f"μ = {p['m']} · σ = {p['s']} · {ks} kernel, {gs} growth")
    ms = "".join(f"<dt>{html.escape(a)}</dt><dd>{html.escape(b)}</dd>" for a, b in measures(s))
    meet = ""
    if s.get("meetings_html"):
        meet = f'<div class="meetings"><h4>Meeting its own kind</h4>{s["meetings_html"]}</div>'
    scale = 100 * p["R"] / 128
    fig = "Frontispiece" if i == 0 else f"Plate {ROMAN[i]}"
    return f'''
<article class="plate" id="{s['id']}" data-id="{s['id']}" aria-labelledby="{s['id']}-name">
  <header class="plate-head">
    <div><div class="plate-no">{fig}</div><h3 id="{s['id']}-name">{html.escape(s['name'])}</h3></div>
    <div class="latin">{html.escape(s['latin'])}</div>
  </header>
  <div class="plate-body">
    <div class="specimen">
      <div class="lens"><canvas class="world" aria-label="Live simulation of {html.escape(s['name'])}"></canvas></div>
      <div class="lens-foot"><div class="scale"><i></i><span>R = {p['R']} cells</span></div><div class="clock"></div></div>
      <div class="controls">
        <button type="button" data-act="play" aria-pressed="false">Pause</button>
        <button type="button" data-act="reset">Restart</button>
        <button type="button" data-act="track">Under the lens</button>
        <button type="button" data-act="fast" aria-pressed="false">Four steps a frame</button>
      </div>
      <div class="figs">
        <figure class="fig" style="margin:0"><canvas class="kernel"></canvas><figcaption>Fig. a. Kernel by distance, 0 to R</figcaption></figure>
        <figure class="fig" style="margin:0"><canvas class="growth"></canvas><figcaption>Fig. b. Growth against potential; dotted at μ</figcaption></figure>
      </div>
      <figure class="fig" style="margin:0"><canvas class="mass"></canvas><figcaption>Fig. c. Mass over the last 2,000 of 10,000 steps (scale exaggerated)</figcaption></figure>
    </div>
    <div class="notes">
      <div class="field">{s['notes_html']}</div>
      <dl class="measures">{ms}</dl>
      {meet}
      <div class="rule-line">Rule: {html.escape(rule)}</div>
    </div>
  </div>
</article>'''


def build():
    tpl = open(os.path.join(HERE, "guide_template.html")).read()
    content = open(os.path.join(HERE, "guide_content.html")).read()
    cat = json.load(open(os.path.join(HERE, "catalogue.json")))
    specs = []
    for i, c in enumerate(cat):
        data = json.load(open(os.path.join(HERE, "specimens", c["file"])))
        s = dict(data); s.update(c); s["id"] = c["id"]
        specs.append(s)
    plates_html = "\n".join(plate(i, s) for i, s in enumerate(specs[1:], 1))
    content = content.replace("<!--FRONTISPIECE-->", plate(0, specs[0])).replace("<!--PLATES-->", plates_html)
    for key in ("preface", "closing", "colophon"):
        fp = os.path.join(HERE, f"text_{key}.html")
        content = content.replace(f"<!--{key.upper()}-->", open(fp).read() if os.path.exists(fp) else "")
    toc = "".join(f'<li><a href="#{s["id"]}">{html.escape(s["name"])}</a></li>' for s in specs[1:])
    content = content.replace("<!--TOC-->", toc)
    js_data = [dict(id=s["id"], params=s["params"], pattern=s["pattern"], wash=s.get("wash", 1),
                    mass_trace=s.get("mass_trace"), moving=s["desc"]["moving"]) for s in specs]
    out = (tpl.replace("/*CONTENT*/", content)
              .replace("/*LENIA_JS*/", open(os.path.join(HERE, "lenia.js")).read())
              .replace("/*SPECIMENS*/", json.dumps(js_data, separators=(",", ":")))
              .replace("/*GUIDE_JS*/", open(os.path.join(HERE, "guide.js")).read()))
    open(os.path.join(HERE, "field-guide.html"), "w").write(out)
    print("wrote field-guide.html", len(out) // 1024, "KB,", len(specs), "specimens")


if __name__ == "__main__":
    build()
