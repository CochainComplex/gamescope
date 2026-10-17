# Scripts

This directory holds the launch, test, and analysis helpers for this fork's frame generation, and the Lua scripting/configuration system inherited from upstream gamescope (unrelated to frame generation; see [Lua scripts and configs](#lua-scripts-and-configs)).

Players need only the first sections: [If something is wrong](#if-something-is-wrong), [For players](#for-players) and [Hybrid laptops](#hybrid-laptops-gamescope-hybridsh). The rest is developer tooling.

Jump to: [If something is wrong](#if-something-is-wrong), [Frame-generation helpers](#frame-generation-helpers), [Hybrid laptops](#hybrid-laptops-gamescope-hybridsh), [Showcase and analysis tools](#framegen-showcase-and-analysis), [Hybrid launcher reference](#hybrid-launcher-reference), [Lua scripts and configs](#lua-scripts-and-configs).

## If something is wrong

Run these in order:

1. `./scripts/framegen-doctor.sh`: checks the build tools, Vulkan, and which GPU is which. It prints the `export RENDER_DEV=...` and `export PRESENT_DEV=...` lines to copy.
2. `scripts/gamescope-hybrid.sh --report -- vkcube`: hybrid laptops only. A dry run that shows what it detected and the command it would run, and runs nothing.
3. `scripts/watch-framegen.sh`: shows generation working in a nested window. It needs `RENDER_DEV` and `PRESENT_DEV` set (step 1 prints them) and runs vkmark unless you pass a client after `--`.

## Frame-generation helpers

All scripts for this fork's frame generation, in one place. Testing workflow and verdicts: [doc/framegen-testing.md](../doc/framegen-testing.md).

### For players

| Script | When to run it, and what it does | Details |
|---|---|---|
| `framegen-doctor.sh` | First, and whenever something is wrong. Standalone pre-build and Vulkan/topology preflight; also prints the hybrid-laptop report; prints the `export RENDER_DEV=...!` and `export PRESENT_DEV=...` lines to copy for `run-framegen-native.sh` and the other launchers | [Hybrid laptops](#hybrid-laptops-gamescope-hybridsh) |
| `run-framegen-native.sh` | You want to play from a text console (the real smoothness test). Run frame generation directly on the DRM/KMS backend (from a text VT) | [User guide: native mode](../doc/framegen-howto.md#native-mode-from-a-text-console) |
| `gamescope-hybrid.sh` | Hybrid laptop (iGPU plus dGPU). Hybrid-laptop launcher | [Hybrid laptops](#hybrid-laptops-gamescope-hybridsh) |
| `watch-framegen.sh` | You want to see generation working in a nested window with a test workload (vkmark by default). Watch frame generation in a nested desktop window | [Testing: overview](../doc/framegen-testing.md#overview-and-quick-commands) |
| `env-gamescope-local.sh` | You run gamescope from this checkout and it needs the checkout's runtime (for example the WSI layer). Source it, or run `scripts/env-gamescope-local.sh <cmd>`, to load this checkout's runtime; `GAMESCOPE_BUILD_DIR` selects the build, `GAMESCOPE_LOCAL_PREFIX` the dependency prefix (default `$HOME/.local/gamescope-wayland`) | Script header |

### For developers and testers

Test, A/B and training tools, plus the shared launcher library. Terms: GSFD is the capture format that `GAMESCOPE_FRAMEGEN_RECORD` writes; "parity" means checking that the offline numpy port gives the same result as the GPU.

| Script | Purpose | Details |
|---|---|---|
| `test-framegen.sh` | Drive compositor-side frame generation for testing; `gpus` lists the GPUs and their `vendor:device` IDs | [Maintenance: validation](../doc/framegen-maintenance.md#validation-by-change-class) |
| `framegen-launch-common.sh` | Library of shared argument and environment checks; sourced by the launchers, not run directly | n/a |
| `framegen-nested-smoke.py` | Nested runtime smoke and fault-injection harness | [Testing: smoke harness](../doc/framegen-testing.md#smoke-harness-setup-and-options) |
| `framegen-showcase.sh` | Visible showcase and end-to-end A/B judge (`--baseline`, `--compare`) | [Testing: showcase](../doc/framegen-testing.md#end-to-end-showcase-and-baseline-compare) |
| `framegen-showcase-report.py` | Summary and compare logic behind the showcase | [Showcase and analysis tools](#framegen-showcase-and-analysis) |
| `framegen-headless-host.py` | Runs a command nested in a private headless GNOME Shell | [Showcase and analysis tools](#framegen-showcase-and-analysis) |
| `framegen-hindsight-eval.py` | Offline prediction-error split of a `GAMESCOPE_FRAMEGEN_RECORD` capture | [Testing: S1/S2](../doc/framegen-testing.md#offline-prediction-error-split-s1s2) |
| `framegen-capture.sh` | Records a client as raw luma through a headless gamescope's PipeWire stream, for the offline gates | [Testing: offline gates](../doc/framegen-testing.md#offline-gates-on-captured-content) |
| `framegen-field-gate.py` | Per-frame offline gate for motion-field changes on captures; `--parity` checks the port against the GPU | [Testing: offline gates](../doc/framegen-testing.md#offline-gates-on-captured-content) |
| `framegen-temporal-gate.py` | Temporal offline gate: replays the displayed sequence with ground truth at the in-between phases | [Testing: offline gates](../doc/framegen-testing.md#offline-gates-on-captured-content) |
| `framegen-net-train.py` | Trains the learned refiner from a capture and exports a weights blob | [Showcase and analysis tools](#framegen-showcase-and-analysis) |
| `framegen-net-eval.py` | Offline perceptual/temporal evaluator of a capture and of a weights blob | [Testing: refiner and colour evaluators](../doc/framegen-testing.md#offline-refiner-and-colour-evaluators) |
| `framegen-color-eval.py` | Evaluates full-resolution held-out captures | [Testing: refiner and colour evaluators](../doc/framegen-testing.md#offline-refiner-and-colour-evaluators) |

## Hybrid laptops: `gamescope-hybrid.sh`

Three modes, so that gamescope stays on the display GPU and games are steered to the dGPU:

| Mode | Gamescope | Its child | Use for |
|------|-----------|-----------|---------|
| host (default) | pinned to the display GPU | unsteered, with the caller's environment | Steam or a gaming session; steer each game with `--child` |
| `--steer-child` | pinned to the display GPU | steered to the game GPU | one game wrapped in its own gamescope |
| `--child` | not started | steered | a game's Steam launch option |

```bash
scripts/gamescope-hybrid.sh --report -- vkcube                  # dry run: detection, decision, command
scripts/gamescope-hybrid.sh -W 2560 -H 1600 -r 120 --experimental-framegen --framegen-mode motion --framegen-multiplier 2 -- steam -gamepadui   # host: Steam on the iGPU
#   then, per game, Properties > Launch Options (absolute path):
/opt/gamescope/scripts/gamescope-hybrid.sh --child -- %command%
#   or, from a desktop Steam, one gamescope per game:
/opt/gamescope/scripts/gamescope-hybrid.sh --steer-child -W 2560 -H 1600 --experimental-framegen --framegen-mode motion --framegen-multiplier 2 -- %command%
```

In one sentence: the launcher finds the GPU your screen is on, runs gamescope there, and sends the game to the other GPU. If it cannot split, it runs plain gamescope and tells you why.

**From a build tree.** The launcher finds `build/src/gamescope` of its own checkout, but it does not set `VK_LAYER_PATH`, so the Gamescope WSI layer from that build is not found. Run the launcher through the environment script, in a terminal or in front of a Steam launch option: `./scripts/env-gamescope-local.sh /path/to/gamescope/scripts/gamescope-hybrid.sh ...`. Or install the build first (`meson install -C build/ --skip-subprojects`). With `GAMESCOPE_BUILD_DIR` set, use the same value for the environment script; the launcher itself looks only in `build/` or at `GAMESCOPE_BIN`.

Replace `/opt/gamescope` with your checkout. Other launchers take the same per-game prefix: `/opt/gamescope/scripts/gamescope-hybrid.sh --child --` as Lutris's "Command prefix" or Heroic's "Wrapper". Use an absolute path in launch options (Steam runs them with its own working directory and `PATH`), and never wrap the Steam client itself with `--child` or `--steer-child`. Detection rules, environment steering, fail-open behavior, measurements and open items are in the [Hybrid launcher reference](#hybrid-launcher-reference). User-level setup: [doc/framegen-howto.md](../doc/framegen-howto.md).

## Framegen showcase and analysis

Showcase and A/B tooling, then offline analysis and refiner training tools. Options, variables, run indices and gates are documented once, in [doc/framegen-testing.md](../doc/framegen-testing.md#end-to-end-showcase-and-baseline-compare).

- `framegen-showcase.sh`: runs a fixed table of ten framegen configurations against one client (GravityMark from `GM_DIR` by default, or `CLIENT`) and judges a session against a saved baseline: `--baseline FILE`, `--compare FILE` (exit 0 pass, 1 regression, 2 not comparable), `--logs DIR`, `--list` (prints the run table and the numeric run indices). A full default session takes about 30 minutes; try `REPS=1 scripts/framegen-showcase.sh --baseline base.json 1 3` first.
- `framegen-showcase-report.py`: the summary and compare logic behind the showcase (standard library only). `framegen-showcase-report.py summarize --out results.json <logs>` keeps the steady metrics windows of one session; `framegen-showcase-report.py compare <baseline> <candidate>` applies the regression gates with the same exit codes.
- `framegen-headless-host.py`: runs a command nested in a private headless GNOME Shell (virtual monitor `--mode WxH@Hz`, private bus and runtime directory) and kills the host's and the command's process groups on exit, also on SIGTERM, so showcase runs happen in the background. `OUT` defaults to `$XDG_RUNTIME_DIR/framegen-showcase/<time>`, outside the private runtime directory, so the results survive. Reuses `framegen-nested-smoke.py`'s host.

  ```bash
  scripts/framegen-headless-host.py --mode 2560x1440@60 -- env HZ=60 scripts/framegen-showcase.sh 6 7
  ```

  `6 7` are run-table indices (see `--list`). On a hybrid machine the host must render on the present device; see the EGL note in [doc/framegen-testing.md](../doc/framegen-testing.md#end-to-end-showcase-and-baseline-compare).
- `framegen-capture.sh NAME W H WARMUP_S SECONDS -- client...`: records what a client draws as raw 8-bit luma (`NAME-WxH.gray`) through a headless gamescope's PipeWire stream. `RATE` sets the output refresh (default 60); for ground truth at the in-between phases use `RATE=120` with a vsynced client. Fails on a failed or partial capture.
- `framegen-field-gate.py --size WxH [--arms a,b,...] [--domain coded|linear] NAME...`: numpy port of the production field chain and warps, scored per frame on captured triplets; matches on code values like production by default; arm suffix `@xf` selects the crossfade fallback. `--parity DIR` compares the port with `GAMESCOPE_FRAMEGEN_RECORD` samples and fails below 99.9% identical vectors or above a mean confidence error of 0.01.
- `framegen-temporal-gate.py --size WxH --k 2,3 [--arms hold,checked/8,...] NAME...`: replays the displayed sequence on 120 Hz captures, every k-th frame real, and reports per-frame error, flicker, edge flicker, lost and false edges. Offline numbers guide; blind A/Bs decide (see the testing guide).
- `framegen-hindsight-eval.py`: offline analyser for `GAMESCOPE_FRAMEGEN_RECORD` (GSFD) captures: splits the predictive path's error on moving texels into the next pair's residual and the continuation gap. Field-resolution luma only, not a perceptual score. `framegen-hindsight-eval.py --data <capture dir>`; needs `numpy` and `framegen-net-train.py` next to it.
- `framegen-net-train.py`: consumes `GAMESCOPE_FRAMEGEN_RECORD` captures and exports a weights blob for `GAMESCOPE_FRAMEGEN_NET`. `framegen-net-train.py --data <capture dir> --out weights.bin`; `--init --out neutral.bin` writes an untrained blob; `--steps`, `--batch`, `--crop`, `--lr`, `--seed` tune training (`--help`). numpy only. Workflow: [AI refiner section of the user guide](../doc/framegen-howto.md#f-ai-refiner-in-game-learning).
- `framegen-net-eval.py`: offline perceptual/temporal evaluator for the same captures. `framegen-net-eval.py --data <capture dir> [--net weights.bin]`; with `--net` it also reports neutral against refined deltas.
- `framegen-color-eval.py`: evaluates full-resolution held-out captures from `GAMESCOPE_FRAMEGEN_RECORD_COLOR`. `framegen-color-eval.py --data <gscf dir> [--per-frame] [--lpips]`.

Output meaning for the two evaluators: [doc/framegen-testing.md](../doc/framegen-testing.md#offline-refiner-and-colour-evaluators).

## Hybrid launcher reference

Reference for `gamescope-hybrid.sh`: a short player section, the launcher message table, then developer detail on how detection works. Measurements were taken on the AMD 890M + NVIDIA POC unless stated; they are collected in [Measurements and open items](#measurements-and-open-items).

### Launcher tips for players

- Run `scripts/gamescope-hybrid.sh --report -- vkcube` first. It prints what the launcher detected and the command it would run, and runs nothing.
- Normally you set nothing. Set `PRESENT_DEV` (the GPU your screen is plugged into) and `RENDER_DEV` (the game GPU) only when the launcher says `no split` because you have two discrete GPUs, or when it picked the wrong card. `./scripts/framegen-doctor.sh` prints both values.
- A `MESA_VK_DEVICE_SELECT` or `DRI_PRIME` you set yourself is kept, with a warning. The game then follows it, not the launcher's split.
- Laptop with the lid closed: the built-in panel does not count as a display unless you name it with `-O`.
- `--child` never blocks a game. Gamescope options written before it are ignored with one notice. Write `--child -- %command%`.
- Never wrap the Steam client itself with `--child` or `--steer-child`. Run Steam in host mode and steer each game.
- If the launcher cannot split, it runs plain gamescope on the display GPU, so you lose the split but not the game. The reason is on stderr, and the table below says what to do.

### Launcher messages

The launcher prints one line before it runs, `warning: ...` lines after it, and a reason line when it cannot split. `--report` shows the same text and the command, and runs nothing.

| Launcher says | Means | What to do |
|---|---|---|
| `no split: <reason>; gamescope on <GPU>, child unsteered` (for `--child`: `not steering: <reason>; running the game unchanged`) | No GPU split is possible (one GPU, display on the dGPU, two GPUs with the same ID, two discrete GPUs). It runs plain gamescope on the display GPU. | Read the reason after `no split:`. With two discrete GPUs set `PRESENT_DEV` (display GPU) and `RENDER_DEV` (game GPU). |
| `caller-set MESA_VK_DEVICE_SELECT=... is not the game GPU ...; the game follows ..., not this split` | A `MESA_VK_DEVICE_SELECT` or `DRI_PRIME` you set names another GPU, so it was kept. | Unset it, or accept that the game uses the GPU you named. |
| `--child takes no gamescope options; ignored: -f; running: <word>` | You put gamescope options before `--child`. They were skipped and `<word>` runs as the game. | Write `--child -- %command%`. Put `--report` and `--help` before `--child`. |
| `NVIDIA <version> < 555: no explicit sync ...` | The NVIDIA driver is older than 555. | Upgrade the driver for explicit sync. |
| `kernel <version> < 6.8: no explicit sync on the DRM backend` | The kernel is older than 6.8 (DRM backend only). | Upgrade the kernel for explicit sync. |
| `NVIDIA GPU ... has no DRM card node: load nvidia-drm with modeset=1` | The NVIDIA driver exposes no DRM node, so gamescope cannot use that GPU. | Boot with `nvidia-drm.modeset=1`. |
| `--steer-child wraps the Steam client ...` (or `--child`) | You wrapped the whole Steam client, so all of Steam renders on the game GPU. | Run Steam in host mode and steer each game with `--child`. |
| `<connector> is connected on several GPUs ...; set PRESENT_DEV to pick another` | An `-O` connector name exists on more than one GPU. The first in search order was used. | Set `PRESENT_DEV`. |
| `caller-set VK_... ... cannot use <gpu>` | Your Vulkan driver variables hide a GPU's driver from the game or from gamescope. | Remove or widen the variable. |


### Developer detail (how detection works)

#### Display GPU

- On DRM it is the owner of the `-O` connector when that is connected, else of a connected eDP/LVDS/DSI panel, else of any connected connector. The search order is the integrated GPU first, then a panel owner, then PCI order.
- Lid state comes from `/proc/acpi/button/lid/*/state`. While the lid is closed, panels count only when `-O` names them.
- Connector names are per GPU. An `-O` name connected on several GPUs takes the first in the search order, with a warning; set `PRESENT_DEV` to choose.
- Lid closed, no `-O`, an external connector picked: gamescope gets `-O <that connector>` (shown in `--report`), because it would otherwise prefer the panel.
- Lid closed and only the panel connected: there is no split. The launcher says so, names the panel for `-O`, and keeps the fallback gamescope on the panel's GPU.
- Nested, it is the host compositor's GPU. The launcher asks `wayland-info` (linux-dmabuf main device), else takes the first entry of `KWIN_DRM_DEVICES`, `AQ_DRM_DEVICES` or `WLR_DRM_DEVICES`, or uses `xrandr --current --listproviders` on X11. Without those it assumes the panel owner, lid or not, and the report says so.
- Inside gamescope a `--child` sees X11 only (gamescope sets `DISPLAY` and drops `WAYLAND_DISPLAY` unless `--expose-wayland`), and Xwayland lists no xrandr provider. So it assumes the panel owner too, which is the GPU gamescope picked whenever that gamescope has a split.
- Gamescope gets `--prefer-vk-device` for the display GPU.
- The launcher reads `--backend` and `-O`/`--prefer-output` also in getopt_long's abbreviated forms (`--ba`, `--prefer-o`). It refuses `--prefer-vk-device` (`--prefer-v`) in favour of `PRESENT_DEV`.
- `PRESENT_DEV` and `RENDER_DEV` override detection. On DRM an explicit `PRESENT_DEV` uses its own connected connector; nested, it must be the host compositor's GPU when that is known.
- iGPU versus dGPU comes from PCI placement. Intel's iGPU is `00:02.0`. An AMD APU's GPU shares its PCI device with AMD SoC functions (vendor `0x1022`). Everything else is discrete. `boot_vga` is not used.
- An NVIDIA GPU without a DRM card node is reported with a hint to load `nvidia-drm` with `modeset=1`. When it is the only other GPU, that is also the reason for no split.

#### Game steering

Set in the game's environment only, each key only if you have not set it yourself:

- always `MESA_VK_DEVICE_SELECT=vendor:device!`;
- NVIDIA driver: `__NV_PRIME_RENDER_OFFLOAD=1 __VK_LAYER_NV_optimus=NVIDIA_only __GLX_VENDOR_LIBRARY_NAME=nvidia` (NVIDIA README, "PRIME Render Offload");
- Mesa dGPU (amdgpu, i915/xe, nouveau): `DRI_PRIME=pci-DDDD_BB_DD_F` (Mesa environment variables).
- A `MESA_VK_DEVICE_SELECT` or `DRI_PRIME` you set that names another GPU is kept, with a warning: the game follows it, not the split. In host mode it reaches gamescope's child (Steam) unchanged, like the rest of your environment.
- `--child` and `--steer-child` also warn when your Vulkan variables hide the game GPU's driver from the game: `VK_ICD_FILENAMES`, `VK_DRIVER_FILES` or `VK_LOADER_DRIVERS_SELECT` without its manifest, `VK_LOADER_DRIVERS_DISABLE` matching it, or `DXVK_FILTER_DEVICE_NAME`/`VKD3D_FILTER_DEVICE_NAME` (they pick a GPU by name). The same check against the display GPU's driver covers gamescope.
- Existing launch options that pick the dGPU themselves (`prime-run %command%`, `DRI_PRIME=1 %command%`) keep working, since nothing reaches the child to stop them. `--child` does the same with the detected GPU and checks your variables.

#### Only gamescope's own processes are filtered

Gamescope, Xwayland and gamescopereaper never get the steering keys. With a split they get `gamescope_upscale_preemptive=false` and filters that keep the game GPU's user-space drivers out:

- `VK_LOADER_DRIVERS_DISABLE` (Vulkan loader 1.3.234+) with the game GPU's driver manifests, minus the display GPU's own:
  - `*nvidia*,*nouveau*` for a GPU bound to `nvidia` (`nvidia_icd.json`, and Mesa's NVK `nouveau_icd.json`, which would otherwise probe it too);
  - `*radeon*,*amd*` for `amdgpu`, `*intel*` for `i915`/`xe`, `*nouveau*` for `nouveau`.
  - If you set `VK_LOADER_DRIVERS_DISABLE` yourself, gamescope gets your list plus the missing globs.
- For an NVIDIA game GPU, `__EGL_VENDOR_LIBRARY_FILENAMES` lists the non-NVIDIA glvnd EGL vendors from `/etc/glvnd/egl_vendor.d` and `/usr/share/glvnd/egl_vendor.d`.
  - It is skipped, with a warning, when you set `__EGL_VENDOR_LIBRARY_FILENAMES` or `__EGL_VENDOR_LIBRARY_DIRS`.
  - Without it, Xwayland (glamor) loads NVIDIA's EGL vendor library and opens `/dev/nvidia*` on every start.
  - GBM needs nothing: Mesa's libgbm picks its backend from the device's DRM driver, so the iGPU never loads `nvidia-drm_gbm.so`.
- **Gamescope's child gets the caller's environment back exactly.** Each of those keys is unset again, or restored to your value, and so is `PATH` (see below). Steam and anything it starts therefore see the dGPU as without this launcher. The Steam client itself may open the dGPU (CEF, its Vulkan probes), as under any PRIME setup. That cost is idle power only and is unmeasured. Per game, steer with `--child`.
- **Known limitation: the filters cover an NVIDIA game GPU only.** For AMD, Intel Arc or nouveau dGPUs gamescope's Mesa drivers may open, and wake, the dGPU. To be measured.
  - Same-driver pairs (AMD iGPU + AMD dGPU, Intel iGPU + Arc) get no Vulkan filter, because the loader filters whole drivers and RADV and ANV open every GPU they drive during `vkEnumeratePhysicalDevices`.
  - One Mesa EGL vendor serves both GPUs, so there is no EGL filter for them.

#### Fail-open

- No split is possible with one GPU, MUX in dGPU-only mode, the display on the dGPU while an iGPU exists, two GPUs with the same `vendor:device`, or two discrete GPUs without explicit `PRESENT_DEV` and `RENDER_DEV`.
- Then the gamescope modes run plain gamescope on the display GPU with the child unsteered. `--child` runs the game unchanged, except with the display on the dGPU, where it steers the game to that same GPU (no cross-device copy).
- Fail-open sets no driver filters: a fallback gamescope pinned to the iGPU (lid closed, say) may still load the dGPU's driver.
- The reason goes to stderr (one short line; `--report` adds the explanation) and `--report` exits 3.
- `--child` never blocks a game and reads no gamescope options. Words before it (`-f`, `-W 1280 720`, `--steer-child`, ...) are ignored with one notice, never a usage error.
  - Use `--child -- %command%`. A `--` right after it is dropped and the rest runs verbatim without being scanned, so Steam's `steam-launch-wrapper -- reaper SteamLaunch AppId=N -- game` stays intact.
  - Without that `--`, the words after `--child` are skipped only until one names a command (on `PATH`, or an executable path), and everything from there runs. An option value that happens to name a command (`cvt`, say) would run instead of the game. The notice says what was ignored and what runs (`ignored: -f; running: <word>`). If no word names a command, the words run as given.
  - `--report` and `--help` go before `--child`.
  - Without `framegen-launch-common.sh`, `--child` says so once and runs the game unsteered.
- In the gamescope modes a forgotten `--` is a usage error (every gamescope option takes at most one value, so two words in a row mean the command started).

#### Detection and binary lookup

- Detection is read-only. It reads sysfs (`SYSFS_ROOT`, default `/sys`) and, when nested, asks the host compositor (`wayland-info`, `xrandr --current`). It never runs `nvidia-smi`, `vulkaninfo` or `glxinfo`.
- The report's PCIe cap is the minimum over the dGPU's upstream bridges, plus the dGPU's own link only while it is runtime-active. It warns below NVIDIA 555 and, on DRM, below kernel 6.8 (no explicit sync).
- Gamescope is `GAMESCOPE_BIN`, else `build/src/gamescope` of the checkout the script is in, else `gamescope` on `PATH`. The launcher prints the one it runs. It does not set `VK_LAYER_PATH` (see [From a build tree](#hybrid-laptops-gamescope-hybridsh)).
- Gamescope starts its child through `gamescopereaper` on `PATH`. So the directory of the `gamescopereaper` next to gamescope (or next to its symlink target, also for a symlink found on `PATH`) is prepended to gamescope's `PATH` only, unless `PATH` already finds that reaper. `--report` shows the child's restored `PATH` as `PATH="$PATH"`. Verified on the POC: the checkout's build, started without `scripts/env-gamescope-local.sh` and with no gamescope installed, runs its child.
- `-h` is gamescope's `--nested-height`; the launcher's help is `--help`.
- `scripts/framegen-doctor.sh` prints the same report under `HYBRID`, then PASS (split ready), WARN (no split on a laptop-shaped machine) or INFO (one GPU, a desktop). Fixture tests: `bash tests/gamescope_hybrid_test.sh` (no GPU needed).

### Measurements and open items

- **Measured on the AMD 890M + NVIDIA POC** (host mode, headless backend, `strace -f -e trace=openat` and `/proc/<pid>/fd` and `/proc/<pid>/maps` sampling): after startup, gamescope, Xwayland, gamescopereaper and the child hold no NVIDIA device node or library open; gamescope opens only the iGPU's render node (`renderD128`) and reads the `nvidia_icd.json` and `nouveau_icd.json` manifests without loading either driver; nothing opens `/dev/nvidia*`. Still opened: at startup, Xwayland's Mesa EGL device probe opens the dGPU's render node (`renderD129`, twice) and primary node (`card2`, once) and closes them again (the `libEGL warning: egl: failed to create dri2 screen` lines). No environment variable prevents that probe. Whether these short opens resume a suspended dGPU is not measured.
- **Open measurements:** dGPU `power/runtime_status` and `runtime_active_time` across an idle host-mode session with Steam, on the POC from a session where nothing else holds the dGPU (a TTY without a desktop: is it `suspended` before start, and does Xwayland's startup EGL probe resume it?), on an AMD + AMD laptop (Framework 16) and on the 7900 XT + 5700 XT desktop (the AMD/Arc limitation above); how often the Steam client itself opens the dGPU under host mode (`steamwebhelper`'s fds and maps with Steam idle).

## Lua scripts and configs

Not related to frame generation. This section is the inherited upstream gamescope Lua configuration system. You do not need it to use frame generation.

### Health warning

Gamescope scripting/configuration is currently experimental and subject to change massively.

Scripts and configs working between revisions is not guaranteed to work, it should at least not crash... probably.

### The basics

Gamescope uses Lua for its configuration and scripting system.

Scripts ending in `.lua` are executed recursively in alphabetical order from the following directories:
 - `/usr/share/gamescope`
 - `/etc/gamescope`
 - `$XDG_CONFIG_DIR/gamescope`

You can develop easily without overriding your installation by setting `script_use_local_scripts` which will eliminate `/usr/share/gamescope` and `/etc/gamescope` from being read, and instead read from `../config` of where Gamescope is run instead of those.

When errors are encountered, it will simply output that to the terminal. There is no visual indicator of this currently.

Things should mostly fail-safe, unless you actually made an egregious mistake in your config like setting the refresh rate to 0 or the colorimetry to all 0, 0 or something.

### Making modifications as a user

If you wish to make modifications that will persist as a user, simply make a new `.lua` file in `$XDG_CONFIG_DIR/gamescope` which is usually `$HOME/.config/gamescope` with what you want to change.

For example, to make the Steam Deck LCD use spec colorimetry instead of the measured colorimetry you could create the following file `~/.config/gamescope/my_deck_lcd_colorimetry.lua` with the following contents:

```lua
local steamdeck_lcd_colorimetry_spec = {
    r = { x = 0.602, y = 0.355 },
    g = { x = 0.340, y = 0.574 },
    b = { x = 0.164, y = 0.121 },
    w = { x = 0.3070, y = 0.3220 }
}

gamescope.config.known_displays.steamdeck_lcd.colorimetry = steamdeck_lcd_colorimetry_spec
```

and it would override that.

You could also place this in `/etc/gamescope` if you really want it to apply to all users/system-wide, but that would need root privileges.

### Features

Being able to set known displays (`gamescope.config.known_displays`)

The ability to set convars.

Hooks

### Examples

A script that will enable composite debug and force composition on and off every 60 frames.

```lua
my_counter = 0

gamescope.convars.composite_debug.value = 3

gamescope.hook("OnPostPaint", function()
    my_counter = my_counter + 1

    if my_counter > 60 then
        gamescope.convars.composite_force.value = not gamescope.convars.composite_force.value
        my_counter = 0
        warn("Changed composite_force to "..tostring(gamescope.convars.composite_force.value)..".")
    end
end)
```

### Hot reloading

Coming soon...
