# Frame Generation Maintenance and Publication Guide

This guide defines the contracts that a frame-generation change must preserve.
It complements the algorithm description in
[`framegen-architecture.md`](framegen-architecture.md). The numbered proposal
documents are design history; they are not a substitute for checking the current
symbols below.

## Source map

| Area | Primary source | Ownership |
|---|---|---|
| CLI and environment | `src/main.cpp` | parsing, validation, debug controls |
| Framegen types and quality policy | `src/framegen/types.hpp`, `src/framegen/policy.hpp` | mode/quality vocabulary and pure degradation-ladder resolution |
| Temporal, scheduling, and dispatch policy | `src/framegen/temporal.hpp`, `src/framegen/scheduling.hpp`, `src/framegen/dispatch_policy.hpp` | pure phase math, cadence/deadline planning, and capability-to-strategy selection |
| Self-supervised adaptation policy | `src/framegen/adaptation.hpp` | B4 counter decoding, EMA state, and next-batch threshold derivation |
| Numeric and setting contracts | `src/framegen/numeric.hpp`, `src/framegen/settings.hpp` | fast-math-safe fp32 classification and strict scalar/path parsing |
| Shader and ML contracts | `src/framegen/push_constants.hpp`, `src/framegen/net_layout.hpp`, `src/framegen/net_profile.hpp` | CPU/GLSL ABI, tensor layout, and GSFR validation/migration |
| Atomic capture/profile output | `src/framegen/atomic_file.hpp`, `src/framegen/atomic_file.cpp` | unique same-directory staging, checked buffered close, cleanup, and atomic rename |
| Vulkan scheduling and algorithms | `src/rendervulkan.cpp`, `src/rendervulkan.hpp` | history, resources, dispatch recording, and ML execution |
| Queue and timestamp execution | `src/framegen/device.cpp`, `src/rendervulkan.hpp` | framegen submission, completion, command-buffer retirement, and GPU-time accounting |
| Presentation choice | `src/steamcompmgr.cpp` | real/generated/repeat arbitration and timed flips |
| Backend present | `src/Backends/DRMBackend.cpp`, `src/Backends/WaylandBackend.cpp` | final generated-frame substitution |
| Algorithms | `src/shaders/cs_framegen_*.comp` | extrapolation, motion estimation, validation, warp, ML |
| Offline ML tools | `scripts/framegen-net-*.py` | GSFD parsing, training, GSFR evaluation |
| Contract tests | `tests/test_framegen.cpp` | degradation, temporal, cadence/deadline admission, adaptation, dispatch, ABI encoding, net layout, GSFR compatibility, and atomic-output contracts |

Use symbol names in documentation and reviews. Numeric source-line references
become wrong whenever the hot path is reorganized.

The `src/framegen` policy and contract headers are deliberately stateless and
header-only: they add no link boundary to command recording. In particular,
`scheduling.hpp` owns deterministic cadence confidence, the bounded online
source-cadence filter, fixed-slot deadline admission, gap/slot planning,
keep-up thresholds, and deadline-ladder transitions; it does not own clocks,
timestamp provenance, history, or the decision to submit. `device.cpp` is a
narrow exception for `CVulkanDevice` queue/timestamp methods; the class remains
the sole owner of that state. `adaptation.hpp` similarly owns only deterministic
counter decoding and threshold math. `rendervulkan.cpp` retains the mapped-memory
completion gate, readback sequence ownership, scene lifetime, logging, and the
dispatches that consume those thresholds. Keep mutable algorithm resources,
history, and dispatch ordering together until they can move behind one explicit
owner with a proven lifetime. Splitting tightly coupled dispatch helpers merely
to reduce line count is not an architectural improvement and can hide ordering
or ownership bugs.

## Non-negotiable runtime contracts

1. **Real content has priority.** In causal modes, a newly latched game frame
   discards every stale generated candidate before presentation. Bidirectional
   mode is the explicit exception: its generated entries and real endpoints form
   one ordered, bounded presentation timeline.
2. **Normal presentation never waits for framegen.** Readiness uses
   `hasCompletedFramegen`; an incomplete candidate becomes a hardware repeat or
   is discarded. `waitFramegen` is reserved for exceptional reset/teardown paths
   that are about to release referenced resources.
3. **Generated frames are compositor-owned.** They must not create a client
   commit, `frame_done`, presentation feedback, or an anti-lag/Reflex event.
4. **Queue ordering is explicit.** Dedicated-queue work waits for the real
   composite's scratch-timeline value and signals the framegen timeline. The
   shared-queue fallback relies on in-order execution and is capped to one
   generated result per batch to bound head-of-line work.
5. **At most one generation batch is in flight.** This is the ownership rule
   behind the descriptor ring, timestamp query ring, motion intermediates, and
   `genReadA`/`genReadB` pins. Supporting concurrent batches requires explicit
   per-submission ownership for all four; increasing a ring size alone is not a
   correctness fix.
6. **Output images are acquired by ownership, not guessed age.** Real composite,
   generated, and late-composite targets are writable only when
   `CVulkanTexture::IsInUse()` is false. Completed cross-queue read pins are
   released before real-target acquisition. Real pressure first runs one
   non-blocking backend event-progress pass, then invalidates speculative
   history and retries; persistent pressure skips safely. This extra poll is
   exceptional-path only. Generated pool pressure shortens or skips a batch.
   No path rewrites a pending or scanned-out image.
7. **Every framegen submission is covered by reset lifetime.** Update
   `lastFramegenWorkSeqNo` for work that retains framegen textures, including
   descriptor-free copies. Reset must retire that sequence before releasing the
   resources.
8. **Classic fixed-slot mode suppresses VRR and tearing.** VRR hybrid is an
   explicit, dedicated-queue-only timing regime. Do not let a shader-quality
   change implicitly select a present mode.
9. **JIT prediction is admission-only.** It may spend idle presentation-GPU work
   on a disposable backup or skip that work. It must not delay a real frame,
   wait for a generated frame, or override real/generated/repeat arbitration.
10. **Storage-image formats match their views.** Generic output shaders use
   formatless write-only storage images and require the enabled Vulkan
   `shaderStorageImageWriteWithoutFormat` feature. This is what permits the same
   output path to bind RGBA/BGRA 8-bit, RGB10A2, 16-bit UNORM, and floating-point
   views without lying in SPIR-V. Explicitly formatted intermediate shaders must
   continue to match their concrete image views. R16F luma additionally requires
   enabled `shaderStorageImageExtendedFormats`; devices without it use the
   existing RGBA16F luma fallback.

## Temporal and algorithm contracts

- A motion-field vector is **displacement over one observed real-frame
  interval**, in field texels. It is not pixels per second and not a normalized
  velocity.
- Ultra/Extreme acceleration compares consecutive displacement fields only when
  frame IDs, quality, mode, and timestamps are consecutive. The preceding field
  is normalized by `currentDt / historyDt`; the irregular-sample quadratic term
  uses `currentDt / (currentDt + historyDt)`. Missing timestamps or an excessive
  interval ratio disables acceleration instead of guessing.
- `src/framegen/temporal.hpp` owns only those arithmetic transformations. The
  renderer owns timestamp provenance, validity gates, scene history, and the
  decision to submit. Do not move a guard into pure math unless every caller
  preserves the same early-exit and fallback behavior.
- `src/framegen/scheduling.hpp` has the same boundary for cadence admission and
  deadline degradation. The renderer supplies monotonic timestamps, measured
  GPU cost/sample identity, queue capability, and mutable ladder state. A helper
  result must not acquire resources, record commands, wait, or select a flip.
- Acquire-ready cadence and composite-time fallback are distinct timestamp
  provenances. Reset the predictor when provenance changes, timestamps stop
  increasing, or the scene gap expires. Never mix display-quantized and
  pre-vblank intervals in one learned state.
- Causal slot `phase` lies after the newest real frame. The shader coefficient is
  derived from that phase and `--framegen-strength`, then bounded by the forward
  cap. Bidirectional `phase` lies in `[0, 1]` between checked real endpoints and
  does not use the causal strength scale.
- Forward/backward consistency and full-resolution source agreement reduce
  confidence. They do not rewrite a surviving motion vector into an unrelated
  fallback trajectory.
- Conservative bidirectional ML cannot change checked geometry, raise
  confidence, or emit causal shading focus. Its authority is a confidence veto.
- Lower quality tiers must stay independently usable. Features added to Ultra or
  Extreme must not allocate, dispatch, or change results in Low through High
  unless that cross-tier change is deliberate and separately validated.
- A scene cut, focus change, layer/EOTF change, long gap, resource shape change,
  or base/output-space transition invalidates incompatible history. The first
  real frame after invalidation primes history and does not generate.

## C++ and shader ABI

Framegen push constants bypass the shared upload arena so dedicated-queue work
cannot race its host allocator. The C++ structures from `FramegenPushData_t`
through `FramegenMotionNetOptPush_t` are copied verbatim into GLSL blocks.
`src/framegen/push_constants.hpp` has compile-time size, semantic-member offset,
standard-layout, and trivially-copyable assertions for every block.

When changing a push block:

1. Change the C++ structure and every matching GLSL variant in the same commit.
2. Preserve scalar types and byte offsets; do not substitute C++ `bool` for a
   shader integer or float flag.
3. Keep the block at or below `k_uFramegenPushConstantSize`.
4. Build all shaders, including paired, fp16, bidirectional, training, and
   optimizer variants. A successful C++ compile alone does not prove the GLSL
   member order matches.

Other duplicated ABI constants require the same treatment:

- Net shape `12 -> 16 -> 16 -> 4`, parameter count `4644`, and the 2048-wide
  served-weight layout are shared by C++, inference/training/optimizer shaders,
  and the Python trainer/evaluator.
- GSFR magic/version/layer metadata are shared by runtime profile loading and
  the Python tools. `src/framegen/net_profile.hpp` is the runtime source of truth
  for metadata construction, strict validation, finite-weight checks, and the
  v1/v2 shading-head migration. A format change requires an explicit
  compatibility policy and matching tool updates. Serialized finite-weight
  checks use `src/framegen/numeric.hpp` to inspect IEEE-754 bits because the
  production C++ build uses `-ffast-math`; do not replace them with
  `std::isfinite`. The same rule applies to GPU readbacks, measured capture
  phases, CLI input, and environment floats.
- Framegen scalar environment settings use the complete-string parsers in
  `src/framegen/settings.hpp`. Keep range policy at each caller: some settings
  clamp a valid finite number while others fall back when it is out of range.
  Malformed, overflowing, NaN, and infinite values must never reach dispatch
  math.
- GSFD and GSCF capture versions are public analysis-file formats. Reject unknown
  versions rather than partially interpreting them.
- B4's 96-counter layout is shared by the stats, apply, training, and CPU
  readback paths. `src/framegen/adaptation.hpp` is the CPU source of truth for
  the image width, counter interpretation, EMA, and derived thresholds. A shader
  layout change requires matching decoder and boundary-test changes.
- `k_uFramegenDescriptorSets` must cover the maximum dispatch count of one legal
  batch. The current worst case is 25 and the ring has 30 entries. Recording
  counts framegen dispatches and fails before a descriptor can wrap within one
  batch; this capacity does not permit a second batch in flight.

`src/framegen/dispatch_policy.hpp` maps observed capability booleans and PCI
vendor ID to an immutable strategy. Vulkan format probing, `ShaderType` mapping,
format-keyed caching, and debug reporting stay in `framegen_dispatch_for_format`.
Unknown vendors use the capability-derived fp16/LDS path. A new hardware
override requires matched microbenchmark evidence and must keep output math
equivalent to the generic variant.

## Resource and cache identity

`previousReal` and `currentReal` retain output-ring images by reference. Real
target selection must continue to use `CVulkanTexture::IsInUse()` so logical
history, unfinished generation reads, and backend scanout ownership share one
correctness check. `genReadA/B` must be released as soon as `genReadSeqNo`
completes, but never before. The generated output pool is disjoint from that
ring. Both pool capacities are sized from the requested multiplier, not the
current degraded multiplier: generated output uses `2·multiplier`, while the
real composite ring uses 8/10/12 slots at x2/x3/x4 to absorb the deeper host
queue caused by higher presentation cadence. These are capacity estimates;
`IsInUse()` is still the safety condition if a backend retains more.

On nested Wayland, `wl_buffer.release` is an aggregate transition to reusable
storage. A repeated attach while the same buffer is still compositor-owned must
not add another lifetime reference: one acquire transition is paired with the
one release transition. Do not replace this with per-commit counting unless a
different protocol supplies explicit per-use completion tokens.

The finalized motion field is reusable only when all identity keys match:

- current real-frame ID;
- motion quality tier;
- causal versus bidirectional mode;
- required forward and reverse field availability.

A reused field runs only per-slot warps. It must not repeat estimation, the B4
probe, learned inference, capture, or online training. Clear the identity on
history invalidation, resource rebuild, held-out color probing, or any operation
that overwrites the working field. Acceleration history additionally requires a
consecutive preceding pair and matching interval metadata.

The timestamp query pool has device/process lifetime. Each batch selects one
unowned pair from its depth-4 ring; a slot is not reset until its previous
query-to-sequence association has been consumed. A temporarily full association
ring skips that optional timing sample instead of overwriting unread data. Scene
invalidation clears associations and costs so a late pre-cut result cannot seed
the new scene; the one-batch completion gate still orders any discarded old query
before a new reset. Resize or format changes do not recreate the pool.

## In-situ learning

The per-pair order is intentional:

1. consume completed readbacks/profile work from the preceding batch;
2. estimate and consistency-check raw fields;
3. capture raw training inputs when requested;
4. run inference with the currently served weights;
5. run B4 cut/trust analysis;
6. record training, with scene-cut gating, and publish optimizer output for the
   next batch;
7. warp generated slots from the finalized field.

Training never consumes its own refined output as ground truth. Same-pair JIT or
idle refills do not train again. A cut zeros the affected gradient work before
Adam consumes it. Non-finite served weights are rejected and the optimizer is
reinitialized from the finite prior.

Profile I/O stays off the compositor hot path. Checkpoints use the owned writer
thread and atomic replace; reset/exit joins it before the final flush. The
renderer owns profile state, logging, and worker synchronization;
`net_profile.hpp` owns the stateless serialized-format checks; and
`AtomicOutputFile` owns only the output transaction. It creates a unique staging
file beside the destination, checks every write and the buffered close, then
renames. Destruction before commit removes the staging file and preserves the
old destination. Concurrent instances can race over which complete snapshot is
newest, but cannot share or corrupt a partial staging file. GSFD/GSCF capture
uses the same primitive and remains a measurement-only synchronous path. Keep
the three disk/readback consumers explicitly out of line: they are polled from
`framegen_submit_planned`, and GCC otherwise folds cold string and file machinery
into the Vulkan command recorder even when capture is disabled.

The per-step served-weight readback uses two reusable vectors. GPU data is copied
into the scratch vector, every value is checked with the fast-math-safe finite
test, and only then are scratch and live snapshots swapped. Do not overwrite the
live vector in place: the last healthy snapshot is both the reset warm start and
the only data eligible for persistence after optimizer divergence. During tests,
point `GAMESCOPE_FRAMEGEN_NET_PROFILE` at a disposable copy. Never use a known-good
profile as the writable test destination.

## Validation by change class

**Documentation, launchers, comments, and compile-time guards:**

```sh
bash -n env-gamescope-local.sh test-framegen.sh \
  scripts/framegen-launch-common.sh watch-framegen.sh run-framegen-native.sh
shellcheck -x env-gamescope-local.sh test-framegen.sh \
  scripts/framegen-launch-common.sh watch-framegen.sh run-framegen-native.sh
PYTHONPYCACHEPREFIX=/tmp/gamescope-pycache \
  python3 -m py_compile scripts/framegen-net-train.py \
    scripts/framegen-net-eval.py scripts/framegen-color-eval.py
c++ -std=c++20 -Wall -Wextra -Werror -Isrc \
  tests/test_framegen.cpp src/framegen/atomic_file.cpp \
  -o /tmp/gamescope-framegen-contracts
/tmp/gamescope-framegen-contracts
c++ -std=c++20 -O3 -ffast-math -Wall -Wextra -Werror -Isrc \
  tests/test_framegen.cpp src/framegen/atomic_file.cpp \
  -o /tmp/gamescope-framegen-contracts-fastmath
/tmp/gamescope-framegen-contracts-fastmath
git diff --check
```

**C++, shader, resource, queue, or scheduling changes:**

```sh
GAMESCOPE_BUILD_DIR=build-perf ./env-gamescope-local.sh \
  ninja -C build-perf
GAMESCOPE_BUILD_DIR=build-perf ./env-gamescope-local.sh \
  meson test -C build-perf --print-errorlogs
GAMESCOPE_BUILD_DIR=build-perf QUALITY=extreme \
  ./test-framegen.sh bench vendor:device
```

Also run a nested functional test and a native DRM/KMS pacing test. Check debug
logs for real/generated/repeat cadence, degradation changes, scene cuts, output
pool pressure, late drops, and routing. Nested success proves execution and
import; it does not prove native vblank smoothness.

**Shader math, phase, confidence, reconstruction, or ML changes:** perform all of
the above, then compare held-out GSCF/GSFD metrics and run matched visual A/Bs at
x2, x3, and x4. Include fast camera motion, disocclusion, translucency, HUD/text,
low source rate, a scene cut, and presenter overload. Report quality and timing;
an image-quality gain that introduces cadence misses is not a net improvement.

## Publication checklist

- No absolute home paths, private benchmark locations, serials, credentials, or
  machine-specific GPU IDs are defaults in tracked files.
- Build trees, wrap downloads, captures, learned profiles, logs, and benchmark
  binaries are not committed. Meson wrap directories stay ignored; wrap files
  and submodule declarations are the reproducible dependency boundary.
- New files are covered by the repository's BSD-2-Clause license unless they
  carry a different explicit license. Do not redistribute third-party benchmark
  assets with this repository.
- `git diff --check`, the local build, and the full test suite pass. Review the
  final diff specifically for push-layout, queue wait/signal, cache-key, history
  invalidation, and flip-priority changes.
- Documentation distinguishes measured results from hypotheses and nested tests
  from native DRM/KMS validation.
