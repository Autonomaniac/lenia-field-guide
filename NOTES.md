# The Garden -- working notes

**Branch:** `garden` (the designated branch for this session; pushed here, no pull request).

## Environment
- Cloud container, 4 cores, numpy 2.4 + scipy 1.17 (scipy installed for connected-component labelling), node 22.
- Worked from BRIEF.md alone.

## Implementation (garden/lenia.py)
Following Chan's reference `Python/LeniaND.py` from github.com/Chakazul/Lenia (fetched raw this session),
not my memory of it. One thing worth noting: in the reference, `kn=1`/`gn=1` index the **polynomial**
("quad4") kernel core and growth function, not the exponential/gaussian ones most tutorials use.
Orbium's catalogue entry has kn=1, gn=1, so the canonical Orbium is the polynomial one.

## Orbium reproduction check -- PASSED
`python3 orbium_check.py animals.json` (Chan's `Python/animals.json`, entry "Orbium unicaudatus":
R=13, T=10, b=[1], m=0.15, s=0.015, kn=1, gn=1; 20x20 RLE pattern decoded with a port of his `rle2arr`).
- My rfft2 update vs a literal transcription of the reference `calc_once` (centred kernel, fftn, fftshift):
  max abs difference 5.5e-13 after 200 steps on a 128x128 torus.
- 5000 steps on 128x128: mass 73.65 in every 1000-step window (range 73.07-76.86 including the first
  settling steps, 73.21-74.03 afterwards). Displacement per 1000 steps 623.2-623.6 cells, heading
  -68.2 deg in every window -- constant velocity, no turning.
- Speed 0.623 cells/step = 6.23 cells per unit time = 0.48 R per unit time. I was suspicious of how fast
  that is, so I checked the raw bounding box: it advances ~12 rows and ~4.5 columns every 20 steps. Real.

## The search -- design, and what went wrong on the way (garden/search.py)
World: 128x128 torus. Candidate = rule (R 8-16, T in {5,10,20}, 1-4 rings with weights in twelfths,
kernel core and growth function each polynomial or exponential/gaussian, m 0.08-0.45, s/m 0.04-0.4
log-uniform) + a seed.

Gates, in order: screen (killed at once if it dies, hazes or spreads over >20% of the grid); long run to
5,000 steps judged on steps 1,000-5,000 (mass never <0.4x or >2.5x its mean, mean drift <10% first to
last quarter, one body in >=90% of checks, radius of gyration < N/5 and not growing >25%); if the board
ends with several bodies the largest is cut out and run again alone; robustness: two re-seedings from
the final body with 10% multiplicative noise, one rotated by a multiple of 90 deg, one by an arbitrary
angle (interpolated), each 3,000 steps, must pass the same gates with mass within 25% and the same
mobility (speed within 30%). Survivors are described by behaviour and offered to a novelty archive
(k=5 nearest-neighbour distance, adaptive threshold).

Things that went wrong, in order:
1. First trial: random rules with random noise *blobs* as seeds. 0 of ~600 random candidates produced
   a glider; every random survivor was a stationary disc. I tested the obvious suspect: under Orbium's
   own rule, 48 of my blob seeds -- and then 120 more in five other blob styles -- all died. So the
   seeding was the problem, not (only) the rarity of gliders.
2. Under Orbium's rule, large noise *soups* (40-128 cells square) did hatch Orbium, 4 times in 72.
   Random candidates became soups: run 1,000 steps, cut out up to three separate bodies, test each.
   Check: 96 soups under Orbium's rule hatched Orbium 5 times and all 5 passed every gate.
3. Soups still hatch rarely (about 0.5%), so I added a second generator: for a random kernel and s/m
   ratio, bisect m on a fixed soup (dies -> m too high, spreads/hazes -> m too low) and hatch soups
   inside the final bracket. This presumes life sits near the death/spread boundary, so I kept plain
   random soups running alongside as the baseline to test that presumption against.
4. A pkill pattern matched its own shell and killed my command mid-restart. The search had already
   stopped cleanly; the run resumed from its survivor file (archive rebuilt, counters carried over).
5. JS/Python check showed chaotic specimens diverge cell-by-cell after ~500-1000 steps (expected:
   rounding differences grow exponentially in those). See "JS verification" below.

## Search statistics (final)
Run: 7.3 hours wall clock, 4 cores, 128x128 torus. Raw output in `search_run/` (survivors.jsonl.gz holds every
survivor with params, pattern, descriptors and parent; search_log.jsonl.gz the running counters).
- Candidates: 41,712 = 13,013 random soups + 7,967 edge-located rules (3 soups each, 23,901 soups) + 20,732 mutants.
- Soup -> separate body: random 77/13,013 = 0.59%; edge 359/23,901 = 1.50%.
- Fates (counted per body; a soup can yield up to 3): died 28,799; spread 16,616; haze 1,399; failed robustness
  1,127; failed long-run gates 333; would not stay one body 147; passed 9,290.
- Of the 9,290: 129 independent origins (37 random-soup, 92 edge-soup), 9,161 mutant descendants. Two
  still-disc ancestors (885.0, 154.0) account for 72%.
- Novelty archive: 970. Shortlist: 48 by greedy max-min over the descriptor vector + 8 I picked from contact
  sheets = 56. Re-run 10,000 steps alone: 53 held, 3 failed (112053 mass drift, 133675.0 broke into bodies,
  125317e0.0 died) -- a false-positive rate of ~1 in 19 for the search's own gates on this sample.
- Plates: 17 from the 53, plus Orbium as frontispiece.

## Measurements added during characterisation (characterize.py)
- mirror symmetry (best reflection correlation), encounters with a copy (2-3 set-ups, each once, 4,000 steps;
  first-meeting outcome judged a speed-scaled settle time after contact, separately from the 4,000-step state,
  because on a 128 torus movers meet again and again), twin-run sensitivity (1e-10 nudge, 1,500 steps),
  and "graded" (share of body cells strictly between 0.02 and 0.98).
- I first tried two spectral measures to catch lattice artefacts (high-frequency power, then power near the
  period-2 frequencies). Neither separated the maze-textured bodies from sharp-edged discs. Looking at raw cell
  values showed the mazes are saturated binary textures with ~4-cell period; "graded" catches saturation, but
  it also flags ordinary solid discs, so the maze verdict on Plate XVII rests on looking, not on one number.
- Spin direction: my `rot` sign convention was backwards (positive = clockwise on screen, since y points down).
  Caught with a synthetic rotating blob before any field note used it; builder fixed.
- The "order" descriptor is the dominant angular Fourier mode, not a count of arms. Every arm/petal count in the
  guide was checked by eye on 4x renders (star 7, cog 8, primrose 5, trefoil 3, rosette 6, hex-bellows 6).
- Spin rates were confirmed by eye frame-by-frame; for the rosette an independent image-registration measure
  agreed (66.9 vs 66.7 deg/100 steps). Registration failed on the trefoil (noisy) -- I relied on the mode
  measure plus visual confirmation there.

## JS verification (verify_js.py -> verify_js_results.json)
All 18 specimens, 3,000 steps, node vs numpy: 11 orderly species agree to < 1e-11 throughout; the Pearl drifts
to 3e-8; the six chaotic species (Flicker, Rambler, Looper, Pip, Hexagonal Bellows, Maze) diverge cell by cell
between steps 500 and 3,000 with mean mass agreeing within 0.3%. Every specimen that diverged is one the twin-run
test independently classed chaotic or drifting, and no other.

## Other things that went wrong
- My wait for the search to exit used `pgrep -f "python3 search.py"`, which matched the wait loop's own command
  line, so it never returned; the search had in fact finished ~1 hour earlier. Mark's other chat noticed and
  asked me to check. Same self-match mistake as the earlier pkill. Lesson: match on a pid, not a pattern.
- A `git add garden` swept in __pycache__; removed and .gitignore added.
- In the field notes I first overstated several things (Orbium "within 40 steps" of contact, "as fast as the
  Comet", Pearl's ring change timing, "smallest change I saw turn order into chaos", "no exception" for soft
  movers -- the Pip is one). All were caught re-reading notes against the data and corrected before the build.
