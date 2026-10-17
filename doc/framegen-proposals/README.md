# Frame Generation Proposals and Design Records

Index of the numbered design proposals (01-07) that the shipped frame-generation
pipeline grew out of, followed by pointers to where current behavior is documented. The
proposals are historical design records. Each carries its own authoritative
`Status:` line plus an implementation-notes section listing divergences from the
original draft. When a proposal and the code disagree, the code and
[framegen-architecture.md](../framegen-architecture.md) win.

Audience: developers deciding what to build next, or tracing why a mechanism has its current shape.
Related: [Documentation index](../README.md) · [User guide](../framegen-howto.md) · [Configuration reference](../framegen-reference.md) · [Architecture](../framegen-architecture.md) · [Maintenance](../framegen-maintenance.md) · [Testing](../framegen-testing.md)

## Contents

- [Proposal index](#proposal-index)
- [Proposal summaries](#proposal-summaries)
- [Where the shipped behavior is documented](#where-the-shipped-behavior-is-documented)

## Proposal index

The proposals are design records written before implementation. Command-line flags they sketch (for example `--framegen-base-layer` or `--framegen-vrr-hybrid`) were never added; the shipped switches are listed in the table below and in the [Configuration reference](../framegen-reference.md).

Status legend:

| Label | Meaning |
|---|---|
| Shipped default | On unless you turn it off. |
| Shipped opt-in | Built and tested, but you must switch it on. |
| Prototype | Built, but not fully validated (the Status column says what is missing). |
| Not built | A design only. Skip it unless you are about to build it. |

| # | Proposal | Status | Opt-in switch | Read this if... |
|---|---|---|---|---|
| 01 | [VRR hybrid: timed mid-interval flips](01-vrr-hybrid-timed-flips.md) | Prototype implemented; not validated on a VRR panel | `GAMESCOPE_FRAMEGEN_VRR_HYBRID=1` (dedicated queue, active VRR) | you are touching VRR hybrid; skip otherwise |
| 02 | [Base-layer generation with late overlay](02-base-layer-generation-late-overlay.md) | Prototype implemented; validated nested on the dual-GPU split | `GAMESCOPE_FRAMEGEN_BASE=1` | you are touching base-layer generation; skip otherwise |
| 03 | [dGPU optical-flow donor](03-dgpu-optical-flow-donor.md) | Not built (exploratory) | none | never needed to understand today's code |
| 04 | [Timestamp-driven adaptive degradation](04-timestamp-adaptive-degradation.md) | Shipped default, with deliberate control-policy divergences | always on (`GAMESCOPE_FRAMEGEN_RECOVER=0` disables recovery; recovery has no effect with bidir or VRR hybrid) | you want to understand the degradation ladder |
| 05 | [Tile classification, indirect dispatch, SDMA fill](05-tile-classification-indirect-sdma.md) | Not built (design draft, deferred Tier-2 item T2.5b) | none | never needed to understand today's code |
| 06 | [JIT phase: display-clock pacing](06-jit-phase-display-clock.md) | Shipped default (default causal pacing, dedicated queue); `GAMESCOPE_FRAMEGEN_JIT` is a logged no-op; `GAMESCOPE_FRAMEGEN_CLASSIC=1` selects the old path | default behavior | you want to understand default pacing |
| 07 | [Frames-only SOTA alignment](07-frames-only-sota-alignment.md) | Design map and gap analysis; E1/E2, B, bounded A and bounded D implemented, broad E2 corpus and external DISTS/FvVDP grading open | `GAMESCOPE_FRAMEGEN_RECORD_COLOR` (E2 capture) | you want the research map and the list of known gaps |

## Proposal summaries

One line per proposal. The table above carries the status; follow the link for the design.

1. [VRR hybrid](01-vrr-hybrid-timed-flips.md): present real frames immediately under VRR and schedule one generated frame with a timer-armed mid-interval flip, keeping the panel above its low-framerate-compensation floor. Prototype; built on #06's estimator.
2. [Base-layer generation](02-base-layer-generation-late-overlay.md): generate on the pre-upscale game layer, upscale through the existing FSR pass, and composite overlays and cursor on top (live layer stack, with fallback to output-space generation for unsupported scenes). Prototype.
3. [dGPU optical-flow donor](03-dgpu-optical-flow-donor.md): offload motion estimation to the NVIDIA render GPU's `VK_NV_optical_flow` accelerator and ship a small flow field across PCIe, with the on-GPU estimator as fallback. Not built.
4. [Adaptive degradation ladder](04-timestamp-adaptive-degradation.md): measure generation GPU time with timestamp queries and step down before deadlines are missed. The rungs shed motion pipeline passes first (`guided -> predict -> learned -> checked -> warp`), then fall back to extrapolation, then (classic, shared-queue and bidir paths only; in forward mode the multiplier removes no work) step the multiplier down toward x2. There is no dormant rung: it never stops generating. Recovery is automatic and hysteretic: it climbs back to the richest rung whose measured cost fits the slot budget. Implemented.
5. [Tile classification, indirect dispatch, SDMA fill](05-tile-classification-indirect-sdma.md): a cheap change-detection pass drives `vkCmdDispatchIndirect` so generation runs only over moving tiles, with static tiles filled on the transfer engine. Depends on transfer-queue discovery that does not exist yet. Not built.
6. [JIT phase: display-clock pacing](06-jit-phase-display-clock.md): plan one generated slot at a time against the KMS pageflip clock. An online alpha-beta filter learns pre-vblank acquire-ready cadence, trend and late error, and admits a disposable generated backup only when the next real frame may miss the compositor wake deadline. Shipped as the default causal pacing.
7. [Frames-only SOTA alignment](07-frames-only-sota-alignment.md): maps the [research survey](../research-framegen.md) onto the shipped pipeline and proposals #01 to #06 and lists the frames-only gaps worth building. Gap E1's evaluator (`scripts/framegen-net-eval.py`), Gap B's scene-cut guard, Gap A's bounded Guided resolver, Gap D's bounded causal shading-focus form and Gap E2's paired full-colour capture (`GAMESCOPE_FRAMEGEN_RECORD_COLOR`) are implemented. Broad E2 corpus capture and external DISTS/FvVDP grading remain open.

## Where the shipped behavior is documented

This directory is the index of design records only. Current behavior lives
elsewhere, so there is a single place to keep each topic correct:

- Pipeline, algorithms and prior art: [framegen-architecture.md](../framegen-architecture.md)
  (the prior-art table is section 3.6; the research survey is [research-framegen.md](../research-framegen.md)).
- Every flag, environment variable, default, HUD row and metric:
  [framegen-reference.md](../framegen-reference.md). Mode compatibility
  (for example `GAMESCOPE_FRAMEGEN_BIDIR` with `VRR_HYBRID` or `BASE`) is stated there.
- Offline training and evaluation of the learned refiner (`scripts/framegen-net-train.py`,
  `scripts/framegen-net-eval.py`) and the E2 held-out colour capture
  (`GAMESCOPE_FRAMEGEN_RECORD_COLOR`, `scripts/framegen-color-eval.py`):
  [framegen-howto.md](../framegen-howto.md), [framegen-testing.md](../framegen-testing.md) and the architecture document.
- Deadline scheduling is the default pacing; `GAMESCOPE_FRAMEGEN_CLASSIC=1`
  selects the classic batch path and `GAMESCOPE_FRAMEGEN_JIT` is a logged no-op.

One measurement from the original notes is kept here because no other page
records it. On the vkmark harness (B4 probe metrics; lower resid/bad is better,
lower killed means more of the field keeps real motion vectors): no net gave
resid 0.0071 / bad 4.85% / killed 80.1%; online from a neutral prior gave
0.0062 / 2.88% / 70.1%; offline blob plus online tracking gave 0.0064 / 3.85% / 77.5%.

BIDIR and VRR-hybrid are exclusive: BIDIR is ignored with a logged line when
VRR-hybrid pacing is requested. BIDIR and BASE may be combined: base-layer scenes
use causal generation, and bidir runs only in scenes that fall back to
output-space generation.
