# Strike-Fit Viability Brief

**Goal:** Test whether Eichberger's rim model, calibrated *only* from in-situ click data (lap clicks, deflector-strike click, deflector ID, spin direction), predicts held-out deflector strikes as well as the existing departure-angle-calibrated pipeline (C++ `stator` implementation of Eichberger 2004). Deliverable is a Python reference implementation, a synthetic-data correctness harness, and a comparison harness against the baseline. Priorities: correctness and comparability. Runtime is secondary. `numpy`/`scipy`/`matplotlib` are allowed; do not mirror Numerical Recipes routine-for-routine.

Paper references (Eichberger 2004, "Roulette Physics") are cited by equation number throughout.

---

## 1. Hard data constraints

The **candidate model** may ingest, per spin:

1. Lap click times at one fixed reference deflector (successive passes of the ball).
2. Strike click time `t_def` (same clock, same `t=0`).
3. Struck deflector ID (an integer — deflectors are *counted*, never angularly measured).
4. Spin direction.

Nothing else. Specifically **forbidden** in the candidate code path:

- Departure angle or departure time annotations (any form).
- Any measured wheel geometry: R, δ, ε, surveyed deflector angles, track radius. Deflector positions are counted multiples of `2π/N_d`. The exit threshold `Ω̄f²` is a fixed convention constant (§4.3), not data and not a fitted parameter.

Legacy departure annotations **do** appear in the dataset. They are for the **baseline model and diagnostics only**. The loader must expose two views, and the candidate fit must assert it never touches the legacy fields.

Optional rotor clicks may appear in the data; they are for pocket-equivalent *reporting* only, never fitting.

---

## 2. Premises (design rationale, encoded as P1–P9)

- **P1** — Only the observables in §1 exist at play time; calibration must run on the same observables.
- **P2** — The rim model (paper eqs. 1, 24, 37, 39) is taken as valid and is not modified.
- **P3** — Lap clicks occur at complete revolutions of a fixed reference point, so tilt effects cancel per revolution (paper's proof that `t(k·2π)` is tilt-independent, §4.4 of the paper). Therefore `(a, b, Ω₀ᵢ)` are identifiable from lap clicks alone, without knowing `η, φ`.
- **P4** — Departure state `(θf, tf)` is **latent**: computed inside the fit, never measured, never compared to any observation.
- **P5** — `Ω̄f²` is unidentifiable from strike data: `∂tf/∂Ω̄f² ≈ −1/[2a·Ωf(Ωf²−b²)]` and the leading term of `∂θf/∂Ω̄f²` are the same for every spin (every ball exits at threshold speed regardless of launch energy), so a threshold change is a common shift in strike angle and strike time — exactly the columns of `φ_eff` and `C₀`. Consequence: **freeze it by convention** (7.62) and let `φ_eff, C₀` absorb the choice. Never let the optimizer touch it.
- **P6** — The descent (rim → deflector) is modeled as constant offsets: angle folded into a per-direction effective phase `φ_eff±`, time as one constant `C₀`. First-harmonic descent terms are added only if residual diagnostics demand them (out of scope for v1, see §11).
- **P7** — `η` (tilt strength) is identified by the *contrast* of the deflector-strike histogram (depth/width of the depopulated arc) plus the dependence of strike position on launch energy `Ω₀`. It is fully fitted.
- **P8** — Hover/graze events (ball departs, oscillates near the rim ≲1 revolution, then descends) appear as a late-time mixture: lobes in the time residual at `≈ +k·T_rev`, `T_rev ≈ 2π/√(Ω̄f²) ≈ 2.3 s`. They are quarantined by the robust (L1) loss and *diagnosed* post-fit, not detected at ingest. The Stage-A gate (§4) catches only *early-arrival* contamination; it is structurally blind to hover, by design.
- **P9** — All fitted parameters are **effective**, not physical. `φ_eff` blends low-point location + descent arc + convention offset + deflector edge-assignment offset; `η` is a contrast relative to the convention. This is harmless for prediction because play time runs the identical pipeline with the identical convention.

---

## 3. Symbols and conventions

| Symbol | Meaning |
|---|---|
| `θ` | Travel-frame angle: increases in the ball's direction of travel, `θ=0` at the reference deflector. Radians. |
| `t` | Time, `t=0` at the first lap click of the spin. |
| `Ω` | Ball angular speed (travel frame, always positive). |
| `a, b` | Deceleration parameters (`a=α`, `b²=β/α`). Shared across all spins and both directions. |
| `Ω₀ᵢ` | Ball speed of spin `i` at its first click. Per spin, from Stage A. |
| `c₀, c₁` | Integration constants (defined where used; Stage A and Stage B use different `c₁`). |
| `η` | Tilt-strength parameter (paper's `η = 2γ/(4a²+1)`). Shared across directions (one physical tilt). |
| `φ⁺, φ⁻` | Per-direction effective phase (free parameter; see P9). |
| `C₀` | Descent-time constant, shared. |
| `Ω̄f²` | Exit-threshold convention, **fixed at 7.62** (rad/s)². |
| `κ` | Exit-equation constant `κ = 1 + (4a²+1)/2` (≈1.5006 at a=0.0225). |
| `N_d` | Number of deflectors, default 8. Spacing `Δ = 2π/N_d`. |
| `wrap(x)` | Wrap to `(−π, π]`. |

**Direction/deflector convention.** Deflectors are numbered physically `0..N_d−1` **clockwise** from the reference deflector (ID 0 = reference). Travel-frame angular position of deflector `d`:

- clockwise spin: `ψ_d = d·Δ`
- counter-clockwise spin: `ψ_d = ((N_d − d) mod N_d)·Δ`

Implement this in one helper and use it everywhere. Sign errors here silently ruin the fit.

---

## 4. Stage A — deceleration fit `(a, b, {Ω₀ᵢ})` + final-lap gate

**Model** (paper eqs. 8–11, 35; level-wheel backbone — justified by P3). Predicted time of the k-th lap click of spin `i` (k = 0..nᵢ, `t̂ᵢ₀ = 0`):

```
c₀ᵢ = −arcoth(Ω₀ᵢ / b)                      # c₀ᵢ < 0
t̂ᵢₖ = (1/(a·b)) · [ c₀ᵢ − asinh( sinh(c₀ᵢ) · exp(a·2πk) ) ]
```

**Objective:** least squares over residuals `tᵢₖ − t̂ᵢₖ`, k = 1..nᵢ (k=0 defines the origin, not a residual). L2 is correct here (Gaussian click noise).

**Fit structure** (nested; robust and simple):

- Inner: for fixed `(a,b)`, each spin's SSE is a smooth 1-D function of `Ω₀ᵢ`; minimize per spin with Brent over `Ω₀ᵢ ∈ (1.05·b, 12)`.
- Outer: minimize total SSE over `(a, b)` with Nelder–Mead. Initialize `a=0.02, b=1.8`.
- Alternative (agent's choice): `scipy.optimize.least_squares` over the full `(a, b, {Ω₀ᵢ})` with a sparse Jacobian pattern. Either is acceptable; validate on synthetic data.

**Final-lap gate** (two passes). The last lap interval of a spin may be contaminated: the ball can depart mid-lap and cut inward, arriving *early* at the reference.

1. Pass 1: fit with every spin's final interval excluded.
2. Per spin: extrapolate `t̂ᵢ,ₙ` for the final click; `Δᵢ = tᵢ,ₙ − t̂ᵢ,ₙ`.
3. Accept the final interval iff `|Δᵢ| ≤ tol`, with `tol = max(3·σ̂_click, 0.05 s)`.
   - `Δᵢ < −tol` → flag `DESCENT_CONTAMINATED`, drop the interval.
   - `Δᵢ > +tol` → flag `LATE` (graze/reconnect or bad click), drop the interval. Keep the flag distinct — LATE spins feed the graze bookkeeping.
4. Pass 2: refit including accepted final intervals.
5. `σ̂_click` = 1.4826·MAD of accepted final-interval residuals. Report it; it seeds `s_t` in Stage B.

Spins need ≥3 clicks after gating to stay in Stage A; fewer → drop spin entirely (flag).

**Bypass hook (important for a clean comparison):** Stage A must be skippable by loading `(a, b, {Ω₀ᵢ})` from a JSON file exported from the existing C++ fit. This isolates Stage B as the only difference between candidate and baseline.

---

## 5. Stage B — strike fit `(η, φ⁺, φ⁻, C₀)`

Fixed inputs: `a, b, {Ω₀ᵢ}` from Stage A; `Ω̄f² = 7.62`.

### 5.1 Forward model, per spin, given `(η, φ)` for its direction

```
c₁ = Ω₀² − b² − η·(cos φ − 2a·sin φ)          # guard: c₁ > 0, else flag & skip
g(θ)  =  c₁·e^(−2aθ) + η·[κ·cos(θ+φ) − 2a·sin(θ+φ)] + b² − Ω̄f²      # paper eq. (37)
g'(θ) = −2a·c₁·e^(−2aθ) − η·[κ·sin(θ+φ) + 2a·cos(θ+φ)]
```

**Root:** smallest positive root of `g`. March from `θ=0` in steps `Δθ = π/8` until the first sign change, then `scipy.optimize.brentq` inside the bracket (tol ~1e−10). `g(0) > 0` always holds physically (ball is supercritical at the first click) — assert it. Cap the march at 30 revolutions; no crossing → flag `NO_EXIT`, skip spin. A dip that grazes zero *without* crossing produces no sign change and is correctly stepped over — those are the physically marginal exits; record the minimum of `g` seen before the accepted root as a per-spin `marginal_exit` diagnostic.

**Fall time** (paper eq. 39, with `c₀` recomputed from the *tilted* `c₁` — footnote 2 of the paper; this intentionally differs from Stage A's level-backbone `c₀`):

```
c₀' = −asinh( b / sqrt(c₁) )
tf  = (1/(a·b))·[ c₀' − asinh( sinh(c₀')·exp(a·θf) ) ]
      − (η/2) · ( sin(θf+φ) + 2a·cos(θf+φ) ) / ( c₁·e^(−2a·θf) + b² )^(3/2)
```

**Predictions:**

- Crossing angle `θ̂ = θf mod 2π`.
- Deflector: forward-edge assignment — the ball sweeps forward into the next deflector: `d̂_travel = ceil(θ̂/Δ) mod N_d`, then convert travel-frame index back to physical ID via the §3 helper. This introduces a constant ≈`+Δ/2` offset relative to the crossing angle; **do not correct for it** — it is absorbed by `φ_eff` (P9).
- Strike time `t̂ = tf + C₀`.

### 5.2 Residuals and objective

For each spin with a recorded strike:

```
r_θ = wrap( θf − ψ(d_obs) )        # ψ from the §3 helper, travel frame
r_t = t_def − tf − C₀
S(η, φ⁺, φ⁻) = Σ |r_θ|/s_θ + Σ |r_t|/s_t
```

L1 loss throughout (robust-estimation heritage of the paper's eq. 43; quarantines hover per P8). Given `(η, φ⁺, φ⁻)`, the optimal `C₀` is the **median** of `{t_def − tf}` over all strike spins, both directions — free inner step, recompute exactly at every objective evaluation.

**Scales:** initialize `s_θ = Δ/√12` (= 0.227 rad for N_d=8; quantization noise) and `s_t` from annotation mode (§6): 0.03 s mechanical, 0.15 s reactive. After the first converged fit, reset both to `1.4826·MAD` of their residuals and refit once.

### 5.3 Optimizer

Grid, then polish. The forward model depends on `(direction, η, φ)` only, so cache aggressively:

```
for dir in {+, −}:
    for η in 0 : 0.05 : 0.60:
        for φ in 0° : 15° : 345°:
            cache[dir, η, φ] = arrays (θf_i, tf_i) for all strike-spins of dir

for each (η, φ⁺, φ⁻):                      # η SHARED between directions
    pool = concat( t_def − tf  from cache[+,η,φ⁺] and cache[−,η,φ⁻] )
    C₀ = median(pool)
    S  = assemble from cached arrays        # cheap
```

Take the best cell; polish with golden-section coordinate descent (order: `φ⁺`, `φ⁻`, `η`; two sweeps; tolerances 0.5° and 0.005), exact inner median each evaluation. Constraint `η ≥ 0` — this kills the mirror minimum the paper notes in its footnote 4. Do **not** hand the raw surface to a derivative-based optimizer from a cold start: the wrapped angle residual makes it non-smooth; grid-then-polish is the design.

Report: fitted parameter table and 1-D objective profiles `S` vs each coordinate through the optimum (basin sanity check).

---

## 6. Deflection-time channel: expected bias and handling

Two annotation modes exist; the dataset header must declare which one it is, and a dataset must be **uniform** in mode.

- **Mechanical (scrub-to-contact):** annotator scrolls video to the first ball–deflector contact frame. Bias ≈ 0; error ≈ frame quantization + 1–2 frames of contact ambiguity → σ ≈ 0.02–0.03 s at 60 fps. This is the recommended mode for this offline viability test.
- **Reactive (live click):** human reaction latency. Expected bias **+0.15 to +0.25 s (late)**, scatter ≈ 0.05 s. Anticipatory clicking (predicting the strike) produces early clicks and must be forbidden in protocol.

Rules the implementation and analysis must respect:

1. Mean bias is *harmless to the fit* — it lands in `C₀` — but only if the latency structure is uniform. Therefore: **no selective re-annotation** of "badly missed" spins (asymmetric truncation of the error distribution biases `C₀`). Re-do all spins or none.
2. `C₀` is only transferable to play time if play-time clicks share the calibration latency structure. If calibration is mechanical and play is reactive, the personal reaction latency must be measured separately (bench test) and added as a known play-time offset. The report should state the calibrated `C₀` and its mode explicitly.
3. Hover/graze events contaminate this channel from the physics side: right-tail lobes at `≈ +k·T_rev ≈ +2.3k s`. The L1 loss discounts them; §8 diagnostics quantify them. Do not pre-filter them at ingest.

---

## 7. Synthetic data generator (correctness harness — build and validate this first)

Truth defaults: `a=0.0225`, `b=√(10/3)≈1.8257`, `Ω̄f²_true=7.62`, `η_true ∈ {0, 0.1, 0.2, 0.4}`, true low-point phases `φ⁺=1.0`, `φ⁻=2.2` rad, true descent offsets `A₀⁺=0.5`, `A₀⁻=0.7` rad (kept separate from φ so the test proves *prediction* equivalence, not parameter equality — see P9), `C₀=0.45 s`, directions 50/50, `N=1000` spins default, `N_d=8`.

Per spin: draw `Ω₀ ~ U(4.5, 7.5)` rad/s. Generate lap clicks at `t(k·2π)` (Stage-A closed form with `c₁ = Ω₀²−b²−η(cos φ−2a sin φ)` backbone is unnecessary — use the level formula; per P3 the tilt correction at complete revolutions is zero) for `k = 0..floor(θf/2π)`, add iid `N(0, 0.03²)` to every absolute click, then re-zero on the (noisy) first click — exactly what real data does. Solve `θf` with the *true* parameters and `Ω̄f²_true`. Strike angle `θs = θf + A₀_dir`; observed deflector = forward-edge assignment of `θs mod 2π`; strike time `t_s = tf + C₀ + bias(mode) + N(0, σ_mode²)`.

Inject, with flags carried as ground truth:

- Hover: probability 0.08 → `t_s += T_rev·(1 + N(0, 0.15²))`, `θs += 2π` (wrapped deflector nearly unchanged — this *is* the gating hypothesis).
- Missing strike: probability 0.05 → strike = null.
- Early final click (exercises the Stage-A gate): probability 0.05 → append a spurious lap click at extrapolated-next-lap − `U(0.1, 0.4)` s.

**Required assertions (pytest), at N=1000, η=0.2, mechanical mode:**

1. Recovery: `|η̂ − 0.2| < 0.03`; `|Ĉ₀ − (C₀+bias)| < 0.02 s`; held-out deflector hit rate within noise of the oracle (predictions from true parameters).
2. **Convention invariance:** refit everything with `Ω̄f²` frozen at **6.5** instead of 7.62. On held-out spins, median `|Δt̂| < 0.02 s`, median `|wrap(Δθ̂)| < 2°`, hit-rate delta < 1 %. This is the load-bearing test of P5.
3. Null sanity: at `η_true = 0`, fitted `η̂ ≈ 0` and hit rate ≈ 1/N_d (no fabricated edge).
4. Gate behavior: injected early clicks are flagged `DESCENT_CONTAMINATED` at ≥90 % recall with ≤2 % false positives on clean finals.

---

## 8. Real-data evaluation vs. baseline

**Baseline** = departure-calibrated Eichberger. Preferred source: parameters/predictions **exported from the existing C++ pipeline** (file hook, same schema spirit as the Stage-A bypass). Fallback: reimplement the paper's §5 fit — outer trisection on φ over [0, 2π], inner robust linear fit of eq. (43) for `(η, b²−Ω̄f²)` — consuming the legacy departure bins. Baseline strike prediction = its `(θf, tf)` plus constant offsets `(A₀_dir, C₀_b)` fitted as medians of training-set strike residuals. This gives the baseline the same descent add-on the candidate has; the *only* difference between the two models is the calibration channel (measured departure angles vs. strike clicks).

**Protocol:** repeated split-half (or 5-fold) by spin, stratified by direction, identical splits for both models and nulls.

**Metrics per model:** deflector hit rate with exact binomial CI; adjacent-hit rate (|Δd| ≤ 1 mod N_d); median `|r_θ|` in deflector units; median `|r_t|` and MAD; pocket-equivalent time error `|r_t| · ν_rotor · N_pockets` with nominal `ν_rotor = 0.33 rps` (or measured rotor clicks where present — reporting only). **Nulls:** (i) always call the pooled dominant deflector per direction; (ii) null (i) plus median strike time.

**Viability criterion:** candidate hit rate ≥ baseline − 1 binomial σ (non-inferiority) **and** candidate > null (i) by > 2σ; candidate median `|r_t|` ≤ baseline's. If the candidate merely matches the baseline, the strike-only protocol wins on collectability — that is a positive result; say so in the report.

**Diagnostics (both models):** per-direction deflector histograms, data vs. model-implied; `r_t` histogram with gridlines at `±k·T_rev` (lobe mass = hover fraction — report it); `r_θ` vs. `Ω₀` and `r_t` vs. `Ω₀` (unmodeled energy trends); objective profiles; Stage-A gate report (counts by flag); per-spin `marginal_exit` summary. Small-N caution: at ~10² spins use bootstrap CIs over spins and report them; the ~10³-spin video is the decisive test.

---

## 9. Data contract

JSONL, one spin per line, plus a dataset header object:

```json
{"n_deflectors": 8, "annotation_mode": "mechanical", "fps": 60.0, "reference_deflector": 0}

{"spin_id": "v3_0142", "direction": "cw",
 "clicks_s": [0.0, 1.093, 2.221, 3.396, 4.634, 5.958],
 "strike": {"t_s": 8.412, "deflector": 5},
 "legacy": {"departure_bin": 17},
 "rotor_clicks_s": null,
 "flags": []}
```

- `clicks_s`: lap clicks, `t=0` at first click, strictly increasing.
- `strike`: null for clean misses (spin then feeds Stage A only).
- `legacy.departure_bin`: 0–35, 10° bins clockwise from reference — **baseline/diagnostics only**; the candidate loader view must not expose it (enforce with an assertion or a separate dataclass).
- `deflector`: physical ID per §3.
- Optional Stage-A bypass file: `{"a": ..., "b": ..., "omega0": {"v3_0142": 6.91, ...}}`.

---

## 10. Module layout and deliverables

```
conventions.py   # wrap, travel-frame helper, constants (Ω̄f²=7.62, κ(a), Δ)
data.py          # schema, loaders, candidate/baseline views, assertions
model.py         # closed forms: t(θ), t(k·2π), g, g', tf; vectorized over spins
stage_a.py       # nested fit + two-pass gate + bypass hook
stage_b.py       # cache-grid + golden-section polish + inner median C₀
synth.py         # §7 generator, truth flags
baseline.py      # C++ import hook + eq.(42/43) fallback fit + descent offsets
eval.py          # splits, metrics, nulls, diagnostics plots
tests/           # §7 assertions: recovery, convention invariance, η=0 sanity, gate
run_viability.py # CLI: synth → real → report
```

Report output: one markdown results table (parameters, metrics, CIs, verdict per §8) plus the diagnostic figures. Float64 throughout; vectorize the forward model over spins per `(dir, η, φ)` cell; full grid at N=1000 should run in minutes.

## 11. Out of scope (v1) — with triggers for later

Rotor stage and scatter model (separate pipeline). Explicit hover mixture component — trigger: `r_t` lobe mass > ~10 %; the fix is an `a,b` refit excluding final laps of lobe-member spins, iterated with Stage B. Per-direction `C₀±` — trigger: per-direction `r_t` medians differ by > 2σ. Interval-censored categorical likelihood for the deflector channel (replaces the L1 angle surrogate). First-harmonic descent terms — trigger: structure in `r_θ` vs. `(θf+φ) mod 2π`. C++ port into `stator`.