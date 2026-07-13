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
| Framegen types and policy | `src/framegen/types.hpp`, `src/framegen/policy.hpp` | mode/quality vocabulary and pure degradation-ladder resolution |
| Shader and ML contracts | `src/framegen/push_constants.hpp`, `src/framegen/net_layout.hpp` | CPU/GLSL ABI and serialized/GPU tensor layout |
| Vulkan execution | `src/rendervulkan.cpp`, `src/rendervulkan.hpp` | history, resources, dispatch, queues, timestamps, ML execution |
| Presentation choice | `src/steamcompmgr.cpp` | real/generated/repeat arbitration and timed flips |
| Backend present | `src/Backends/DRMBackend.cpp`, `src/Backends/WaylandBackend.cpp` | final generated-frame substitution |
| Algorithms | `src/shaders/cs_framegen_*.comp` | extrapolation, motion estimation, validation, warp, ML |
| Offline ML tools | `scripts/framegen-net-*.py` | GSFD parsing, training, GSFR evaluation |
| Contract tests | `tests/test_framegen.cpp` | exhaustive pure-policy, ABI encoding, and net-layout checks |

Use symbol names in documentation and reviews. Numeric source-line references
become wrong whenever the hot path is reorganized.

The `src/framegen` headers are deliberately stateless and header-only: they add
no link boundary to command recording. Keep mutable Vulkan resources, history,
and submission ordering together until they can move behind one explicit owner
with a proven lifetime. Splitting tightly coupled dispatch helpers merely to
reduce line count is not an architectural improvement and can hide ordering or
ownership bugs.

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
6. **Output images are acquired by ownership, not guessed age.** A generated or
   late-composite target is writable only when `CVulkanTexture::IsInUse()` is
   false. Pool pressure shortens or skips a batch; it never rewrites a pending or
   scanned-out image.
7. **Every framegen submission is covered by reset lifetime.** Update
   `lastFramegenWorkSeqNo` for work that retains framegen textures, including
   descriptor-free copies. Reset must retire that sequence before releasing the
   resources.
8. **Classic fixed-slot mode suppresses VRR and tearing.** VRR hybrid is an
   explicit, dedicated-queue-only timing regime. Do not let a shader-quality
   change implicitly select a present mode.

## Temporal and algorithm contracts

- A motion-field vector is **displacement over one observed real-frame
  interval**, in field texels. It is not pixels per second and not a normalized
  velocity.
- Ultra/Extreme acceleration compares consecutive displacement fields only when
  frame IDs, quality, mode, and timestamps are consecutive. The preceding field
  is normalized by `currentDt / historyDt`; the irregular-sample quadratic term
  uses `currentDt / (currentDt + historyDt)`. Missing timestamps or an excessive
  interval ratio disables acceleration instead of guessing.
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
  the Python tools. A format change requires an explicit compatibility policy.
- GSFD and GSCF capture versions are public analysis-file formats. Reject unknown
  versions rather than partially interpreting them.
- B4's 96-counter layout is shared by the stats, apply, training, and CPU
  readback paths.
- `k_uFramegenDescriptorSets` must cover the maximum dispatch count of one legal
  batch. Its current value is 30; it does not permit a second batch in flight.

## Resource and cache identity

`previousReal` and `currentReal` retain output-ring images by reference. Ring
advance must continue to skip history and in-flight generation reads. The
generated output pool is disjoint from that ring and remains sized from the user
multiplier, not the current degraded multiplier.

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

The timestamp query pool has device/process lifetime. Each batch resets and uses
one pair from its depth-4 ring. Scene invalidation clears query-to-rung
associations and costs so a late pre-cut result cannot seed the new scene; resize
or format changes do not recreate the pool.

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
thread and atomic replace; reset/exit joins it before the final flush. During
testing, point `GAMESCOPE_FRAMEGEN_NET_PROFILE` at a disposable copy. Never use a
known-good profile as the writable test destination.

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
  tests/test_framegen.cpp -o /tmp/gamescope-framegen-contracts
/tmp/gamescope-framegen-contracts
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
