# The Garden — a field guide to Lenia

**Open the guide:** https://autonomaniac.github.io/lenia-field-guide/ — every specimen runs live in the browser.

Seventeen self-organising creatures (plus Bert Chan's Orbium as a reference specimen) from an overnight novelty
search of single-channel [Lenia](https://chakazul.github.io/lenia.html): 41,712 rule sets tried, 9,290 survivors
after persistence and robustness gates, descended from 129 independent origins. Each plate's field notes are
written only from measurements.

**What this is:** an AI-made hobby project, not research. It was designed, run and written by Claude
(Anthropic) in Claude Code cloud sessions, on spare cloud credits given by Mark Halligan, who asked for
"something creative and different". The search method is well established (e.g. Reinke et al., ICLR 2020;
Faldor & Cully, *Leniabreeder*, 2024); the field-guide form is the point.

## Contents
- `field-guide.html` / `index.html` — the guide (self-contained; fonts from Google Fonts).
- `lenia.py`, `lenia.js` — the engine, following Chan's reference `LeniaND.py`. `orbium_check.py` reproduces Orbium
  (mass 73.65, 0.623 cells/step); `verify_js.py` + `verify_js_results.json` check the JS engine against Python
  (agree to ~1e-13 for orderly specimens; chaotic ones agree in mean mass within 0.3%).
- `search.py` — the novelty search; `search_run/` — its full log, survivors and characterisations.
- `characterize.py`, `build_guide.py`, `catalogue.json`, `specimens/` — measurement and build.
- `BRIEF.md` — the brief the search session was given. `NOTES.md` — working notes, including what went wrong.

Specimens are plain JSON (rule parameters + pattern) and can be loaded into other Lenia implementations.
Corrections are welcome.
