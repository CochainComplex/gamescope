# 07 — Frames-only SOTA alignment: what we have, what's missing

Status: **design map / gap analysis**; Gaps E1, B, bounded frames-only A, and a
bounded frames-only form of D implemented (`scripts/framegen-net-eval.py`, the
B4 GPU scene-cut guard, the Extreme disocclusion reservoir, the causal
shading-persistence head, and the conservative bidir ML authority boundary). Maps
[`../research-framegen.md`](../research-framegen.md) onto the shipped pipeline and
proposals #01–#06, then sketches the genuine gaps that are worth building.

Every sketch here is constrained by the same three realities the rest of this
directory lives under: the estimator is **frames-only** (no engine motion
vectors, depth, or G-buffers — confirmed in `cs_framegen_motion_warp_accel.comp`,
which needs "no future frame, application motion vector, optical-flow API, or
vendor extension"), it runs on the **weak present GPU** (AMD) while the strong
card renders, and it must stay inside the **monotonic degradation ladder** and
the **never-degrade-HDR** rule. That rules out the whole engine-integrated
school (Mob-FGSR, ExtraNet, ExtraSS, LMV, DLSS/FSR) as *direct* ports; we can
only borrow the parts that survive without engine buffers.

Scope note: the motion-quality stack is `--framegen-mode motion` only. The
default `extrapolate` mode and the quality tiers below it are the low-latency
forward path; SOTA alignment is about the `motion` path.

## Part 1 — Where the shipped pipeline already meets the research

| Research idea (see research-framegen.md) | In tree today | Assessment |
|---|---|---|
| **Extrapolation as the games-appropriate default** (zero added latency); interpolation is opt-in for its ≥1-frame latency (§0, Recommendations) | `extrapolate` default; `motion` forward path; `GAMESCOPE_FRAMEGEN_BIDIR` opt-in interpolation that presents one interval late | **Aligned by design.** The latency taxonomy is baked into the mode/bidir split. |
| **Heuristic motion + a lightweight correction net** is the real-time frontier — the GFFE template (§3, Rec 1) | 3-level luma pyramid matcher → FB/agreement/adaptation → ~4.6k-param field refiner + causal shading-focus head → bounded screen-space disocclusion reservoir | **Same shape within frames-only limits.** The remaining major GFFE gaps are its depth/MV-backed layered world-space background and full feature-domain SCN, neither of which a compositor can reproduce cheaply. |
| **Quadratic / uniform-acceleration motion** (Mob-FGSR, KF1) | Ultra tier: `mvFieldHistory` gives a 2-field second derivative → quadratic through 3 causal positions, deadzoned + accel-capped + confidence-gated (`cs_framegen_motion_warp_accel.comp`) | **Implemented, frames-only variant.** Mob-FGSR does this in world space from depth+MVs; we do it in screen space on the *motion field*. Ours is the correct choice given no depth. |
| **Hybrid "prefer whichever source gives the better color match"** at block level (FSR 3, §4, Recommendations) | Per-pixel two-source agreement + FB round-trip + confidence blend to the bounded pixel-space fallback | **Conceptually equivalent** — a home-grown per-pixel version of FSR's block-level color-match arbitration. |
| **UI/HUD must be composited after generation** (KF, §5) | Proposal **#02** base-layer generation + late overlay/cursor composite (prototyped, `GAMESCOPE_FRAMEGEN_BASE=1`) | **This *is* the research's UI recommendation.** Already prototyped. |
| **Heavy split net once per pair, cheap per-output-frame work** (DLSS 4, KF5) | Field estimation + net run **once per real frame**; only the warp runs per generated slot | **Already structurally satisfied** (split net confirmed verbatim in NVIDIA's write-up). The expensive stage is amortized across the multiplier exactly as DLSS 4 splits it — no gap here, contrary to first impression. (DLSS 4 *interpolates*; the amortization structure maps, not the mode.) |
| Edge-aware upsampling of a low-res field at fg/bg boundaries | Extreme tier `reconstructField()` guided reconstruction | **Local contribution** not named in the research; solves the bilinear-average-vector-at-boundaries problem the research only flags. |

## Part 2 — Genuine gaps, ranked by value ÷ (cost · risk)

### Gap E (do first) — Structural/temporal validation harness · cheap · offline
All tuning today grades on the photometric B4 probe (resid/bad/killed), an
L1-class scalar. The research is explicit that PSNR/L1/SSIM **under-measure the
exact artifacts we care about** — ghosting, flicker, disocclusion shimmer — and
that LPIPS, DISTS, and especially **FvVDP** are the field-standard temporal/
perceptual metrics (§6, Caveats).

**Ground-truth correction:** `GAMESCOPE_FRAMEGEN_RECORD` (`GSFD` files) captures
**field-resolution *luma* + the motion fields** — the exact tensors the net
trains on — **not** the full-resolution *colour* output frame. LPIPS/DISTS/FvVDP
are colour-domain, full-res metrics, so they *cannot* be computed from the
existing captures. This splits Gap E honestly:

- **E1 — implemented** (`scripts/framegen-net-eval.py`). Grades the *field* the
  captures represent with the structural + temporal view L1 misses: SSIM, an
  edge/gradient-structure error (catches boundary smear), `bad%`, and a
  temporal-stability σ, in both field directions, for the neutral (Stage B)
  field and — with `--net <blob>` — the refined field, printing the deltas. It
  reuses the trainer's verified GSFD parser and net so the two never drift; a
  zero-head blob reproduces the neutral metrics exactly (its correctness oracle).
  numpy-only, CPU, zero runtime cost, no HDR/ladder impact.
- **E2 — follow-up (needs a small capture extension).** True colour-domain
  LPIPS/DISTS/FvVDP require dumping generated + ground-truth **full-res colour**
  pairs (a new `RECORD` mode in `rendervulkan.cpp`). The evaluator would then
  grow an optional metric path (torch/LPIPS if importable, graceful otherwise).

E1 already **unblocks measuring** the other gaps structurally; E2 is what lets us
grade the final colour frame the way the literature does.

### Gap B — Content-based scene-cut detection · implemented
The B4 probe now accumulates four-bin luminance histograms for both real frames
over a 3×3 screen partition. A tiny same-pipeline finalize dispatch declares a
cut only when at least seven regions, global histogram total variation,
motion-compensated residual, and unreliable-field coverage all cross
conservative thresholds. This conjunction is important: camera motion can hurt
pixel correspondence while preserving regional distributions, and particle
chaos can kill most field confidence without being a cut. The verdict is
encoded into the final field after its FB diagnostic has been consumed. Causal
warps duplicate the newest real endpoint; bidir chooses the nearest endpoint;
Ultra/Extreme history receives zero confidence so the cut cannot seed a false
acceleration. No CPU round trip, future frame, vendor API, or per-generated-frame
stats read is added. Measured B4 cost is 20–39 µs per real at 1080p–4K on the
Radeon 890M, and a 1M-asteroid camera-motion stress run produced no false cuts.

### Gap A — Disocclusion background reservoir · implemented, Extreme only
**The single biggest frames-only quality gap.** On a disocclusion (revealed
background) the pipeline can only kill confidence and fall back to bounded
pixel-space extrapolation → hold-then-ghost. GFFE's core contribution is
**hierarchical background collection**: it maintains an *accumulated multi-layer
color+depth background buffer* from history frames the renderer would otherwise
discard, then world-space-projects it to fill revealed regions, with an explicit
out-of-screen / static / dynamic disocclusion taxonomy (KF3, §5, Rec 1);
Decoupled Motion Prediction (ACM MM 2025) does the learned-inpainting version
(OccNet). **Cross-check caveat:** GFFE is "G-buffer-free" only for the *future*
frame — it still consumes rendered-frame **depth and MVs**, on which its layered
background buffer and world-space projection depend. We have neither, so our
version is necessarily coarser.

The implemented frames-only form deliberately does **not** retain another
output-ring image: that would increase the pin set and break the five-slot ring's
"always one free composite target" proof while a dedicated-queue batch is still
reading. Instead, two internally owned 1/8-resolution luma images retain the
two-interval-old evidence needed by both the current batch and same-interval JIT
refills. Each image carries a real-frame ID; skipped estimates, invalidations,
scene cuts, and non-consecutive history cannot consume it.

At a full-resolution pixel, the Extreme warp enters the reservoir search only
when the centre field has both low confidence and a large local FB round-trip
error. It tests the eight adjacent field hypotheses, rejects candidates that are
not a distinct, locally FB-consistent layer, reprojects each through the retained
preceding field, rejects any path leaving the screen, and requires photometric
agreement across **current color → previous color → two-interval-old luma**. The
winning layer supplies its color from the **current** real frame, never from
history; this uses the reservoir as causal evidence without creating a stale
color trail. Trust is capped and blended back into the established bounded
fallback. `GAMESCOPE_FRAMEGEN_RESERVOIR=0` provides live A/B attribution.

This recovers one-field-neighbourhood static/dynamic boundary disocclusions. It
still cannot recover content that is out of screen, hidden for many frames, or
correctly depth-order arbitrary overlapping motions. Those require the engine
depth/MVs that GFFE actually uses. The luma ping-pong copy costs 4–9 us per real
frame at 1080p–4K on Radeon 890M; the conditional full-res search is charged to
the Extreme rung and disappears as soon as the deadline ladder drops to Ultra.

### Rejected experiment — temporal global-camera prior
Two causal variants used the preceding checked field as a global translation
hint: filling low-confidence vectors, then only biasing the coarsest matcher's
small-motion tie-break. Both improved mouse/camera responsiveness in live 1M-
asteroid testing, but both reduced image quality through visible jitter and
artifacts relative to the established local matcher. The complete experiment
was removed. A single global translation is not a reliable substitute for
local motion under parallax, rotation, acceleration, particles, disocclusion,
or independently moving layers, and applying the same prior to forward and
reverse matching can correlate errors that the FB check is meant to reject.
Do not reintroduce this path without E2 full-color temporal metrics and an
independent validation signal that cannot reinforce the proposal it grades.

### Corrected contract — bidirectional ML is a confidence veto
The learned flow head had the same validation circularity in a subtler form.
Its self-supervised endpoint loss can improve `warp(source, flow) ≈ endpoint`,
but repeated texture and aperture ambiguity leave multiple vectors with nearly
the same endpoint residual. Those vectors trace different trajectories at
`t=0.25/0.5/0.75`; applying a persisted endpoint-trained correction to both
directions produced heavy artifacts in the 1M-asteroid x4 bidir test even while
the B4 endpoint statistics looked acceptable.

Bidir now preserves both FB-checked vectors exactly and treats ML only as a
confidence veto: negative confidence corrections survive, positive corrections
and the flow/shading outputs are scaled to zero. Online learning remains in
situ, but recomputes its loss on the raw checked field and writes gradients only
for W3's confidence row/bias. The flow rows, shading row, and shared trunk are
frozen, so the endpoint objective cannot reshape geometry indirectly. The old
path is retained only as `GAMESCOPE_FRAMEGEN_NET_BIDIR_FLOW=1` for controlled
attribution.

Live ordering was unambiguous: persisted full-flow profile (artifact-heavy) <
no net (very good) < neutral net (better) < confidence-only learned profile
(best). After 1812 confidence-only steps, binary comparison against the protected
starting profile showed all 4499 non-confidence floats bit-identical; only 139
of 144 confidence-row weights and its bias moved. The original user profile was
never written during the experiment.

### Gap D — Color-domain shading correction · bounded frames-only form implemented
Non-geometric motion — lagging shadows, reflections, specular — is
*unrepresentable* as a motion vector, so no field refinement can fix it. GFFE's
Shading Correction Network repairs this with a flow-based feature warp and a
focus-mask-blended color refinement (§3, verified against Appendix D).

The implemented compositor-safe analog reuses the existing CNN's formerly
reserved fourth output as a **zero-neutral persistence focus**, written to one
1/8-resolution image. It does not ask the net to hallucinate RGB. During online
training, the two-interval-old luma reservoir and preceding field reproject a
third causal frame: the head learns whether an aligned `older→previous` luma
trend actually persisted into the now-known current real frame. A two-frame
loss was rejected because it would reward repeating every mismatch. The new
loss updates only output row four/bias, never the shared trunk or established
flow/confidence heads; reverse tiles, cuts, stale IDs, refills, lower tiers and
bidir provide exactly zero shading gradient.

Extreme's existing full-resolution warp then extrapolates the aligned
`current-previous` RGB trend under the learned focus. It is phase-scaled, capped
to 8% of local magnitude per interval, confidence/history/acceleration gated,
restricted to moderate changes, finite-checked, and never `[0,1]` clamped. A
zero/v1/v2 head is bit-neutral; legacy blobs have their previously undefined
fourth row forcibly zeroed before use. `GAMESCOPE_FRAMEGEN_SHADING=0` disables
both supervision and correction for isolated A/B.

This is deliberately below a full SCN: no output-resolution feature network,
no invented color, and no claim that it can handle arbitrary view-dependent
effects. E2 full-color capture and LPIPS/DISTS/FvVDP remain required to tune and
validate the final color result perceptually. The bounded form preserves the
split-net cost model and degradation safety while providing a causal mechanism
for persistent lighting/color trends.

## Part 3 — Annotations to existing proposals (no new work)

- **#03 dGPU optical-flow donor** — the research adds a caveat worth recording in
  that doc: DLSS 4 **abandoned the fixed-function Optical Flow Accelerator** for
  an AI flow network (§4, confirmed verbatim — *"replacing hardware optical flow
  with a very efficient AI model"*). The `VK_NV_optical_flow` donor is therefore
  a *stopgap*; the strategic direction is a **learned flow front-end** (SEA-RAFT /
  NeuFlow v2, §2) — which our Stage-C field net already partially is. Build #03 as
  a measurement baseline, not the endpoint. (Cross-check nuance: NVIDIA confirms
  engine MVs are wrong on *specular/reflections/UI*; the survey's stronger claim
  that learned flow fails on the *same* set — plus transparency/particles — is
  reasoned from the shared root cause, not a single cited source. It does not
  change this recommendation.)
- **#02 base-layer late composite** = the research's UI-compositing recommendation
  verbatim; no change, just the mapping.
- **#04 / bidir** already encode the FSR-3 "≥60 fps pre-generation" spirit via the
  ladder; a soft advisory when the measured base rate sits below ~40 fps (where
  the research says artifacts dominate) would be a cheap, honest addition.

## Part 4 — Explicitly out of scope (and why — so we don't chase them)

- **Engine-buffer methods** (Mob-FGSR, ExtraNet, ExtraSS, LMV): need depth /
  G-buffers / engine MVs for the generated frame. We are a compositor with color
  frames only. Not portable.
- **GFFE's adaptive rendering window** (extend the render frustum for out-of-screen
  disocclusions): requires engine cooperation we don't have. The background
  reservoir (Gap A) recovers static/dynamic disocclusions but *not* out-of-screen
  ones — an accepted, documented limitation of the frames-only regime.
- **Diffusion / transformer VFI** (MoMo, LDMVFI, Framer, VFIMamba): 0.1–0.7 s/frame,
  offline-only (§1, Caveats). Orders of magnitude outside a real-time budget on
  any GPU, let alone the weak present card.

## Recommended order

`E → B → A → D`. **E1, B, bounded frames-only A, and bounded frames-only D are
done**; E2 (colour-frame capture) is the next measurement step, while a full
feature-domain SCN remains the long-horizon ceiling. #03 stays a baseline.

## Cross-check provenance (2026-07-11)

Every mapping above was verified against **primary sources** (arXiv PDFs, ACM
DOIs, NVIDIA Research/Developer pages, AMD GPUOpen SDK docs), not just the
survey. Load-bearing confirmations:

- **GFFE** (arXiv 2406.18551, DOI 10.1145/3687923) — hierarchical background
  collection, the out-of-screen/static/dynamic disocclusion taxonomy, and the
  flow-based SCN with focus mask (Table 5: 6.62 ms/1080p, SCN 2.30 ms on a 4070
  Ti Super; Appendix D for the SCN internals).
- **Mob-FGSR** (SIGGRAPH 2024, DOI 10.1145/3641519.3657424) — *"under the
  assumption of quadratic motion… uniform acceleration,"* from color+depth+MVs →
  our ultra tier's screen-space analog.
- **DLSS 4** (research.nvidia.com/labs/adlr/DLSS4) — the split network (*"one half
  runs once for every input frame pair… re-used; the other much smaller half runs
  once for every generated output frame"*) and the OFA→AI-model replacement.
- **FSR 3** (GPUOpen FidelityFX Optical Flow / Frame Interpolation) — 8×8 blocks,
  24×24 search, SAD, 7-iteration luminance pyramid, 9-section luminance-histogram
  scene-change detection, and the OF-vs-MV color-match arbitration our per-pixel
  agreement mirrors.

Two corrections were pushed upstream into
[`research-framegen.md`](../research-framegen.md): (1) Decoupled Motion Prediction
(ACM MM 2025) is a *contemporaneous, independent* G-buffer-free method from the
ExtraNet lineage (Jie Guo et al.), **not** an authored "successor" to GFFE; (2)
the "engine MVs and optical flow fail on the same cases" claim is NVIDIA-confirmed
only for the engine-MV half (specular/reflections/UI). Neither changes a
recommendation here.
