# Gamescope Script/Config Files

## ⚠️ Health Warning ⚠️

Gamescope scripting/configuration is currently experimental and subject to change massively.

Scripts and configs working between revisions is not guaranteed to work, it should at least not crash... probably.

## The Basics

Gamescope uses Lua for it's configuration and scripting system.

Scripts ending in `.lua` are executed recursively in alphabetical order from the following directories:
 - `/usr/share/gamescope`
 - `/etc/gamescope`
 - `$XDG_CONFIG_DIR/gamescope`

You can develop easily without overriding your installation by setting `script_use_local_scripts` which will eliminate `/usr/share/gamescope` and `/etc/gamescope` from being read, and instead read from `../config` of where Gamescope is run instead of those.

When errors are encountered, it will simply output that to the terminal. There is no visual indicator of this currently.

Things should mostly fail-safe, unless you actually made an egregious mistake in your config like setting the refresh rate to 0 or the colorimetry to all 0, 0 or something.

# Making modifications as a user

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

You could also place this in `/etc/gamescope` if you really want it to apply to all users/system-wide, but that would need root privelages.

# Features

Being able to set known displays (`gamescope.config.known_displays`)

The ability to set convars.

Hooks

# Examples

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

# Hot Reloading?

Coming soon...

# Hybrid laptops: `gamescope-hybrid.sh`

On a hybrid laptop (iGPU + dGPU), `gamescope-hybrid.sh` runs gamescope (compositing, frame generation, present) on the GPU that drives the display and steers **only games** to the other GPU. Every PCI pairing is detected the same way: AMD iGPU + NVIDIA, AMD iGPU + AMD dGPU, Intel iGPU + NVIDIA, Intel iGPU + Arc, and a desktop with two discrete GPUs (explicit IDs).

Three modes, so that gamescope stays on the display GPU and games are steered to the dGPU:

| Mode | Gamescope | Its child | Use for |
|------|-----------|-----------|---------|
| host (default) | pinned to the display GPU | unsteered, with the caller's environment | Steam or a gaming session; steer each game with `--child` |
| `--steer-child` | pinned to the display GPU | steered to the game GPU | one game wrapped in its own gamescope |
| `--child` | not started | steered | a game's Steam launch option |

```bash
scripts/gamescope-hybrid.sh --report -- vkcube                  # dry run: detection, decision, command
scripts/gamescope-hybrid.sh -W 2560 -H 1600 -r 120 -- steam -gamepadui   # host: Steam on the iGPU
#   then, per game, Properties > Launch Options (absolute path):
/opt/gamescope/scripts/gamescope-hybrid.sh --child -- %command%
#   or, from a desktop Steam, one gamescope per game:
/opt/gamescope/scripts/gamescope-hybrid.sh --steer-child -W 2560 -H 1600 -- %command%
```

Replace `/opt/gamescope` with your checkout. Other launchers take the same per-game prefix: `/opt/gamescope/scripts/gamescope-hybrid.sh --child --` as Lutris's "Command prefix" or Heroic's "Wrapper". Steam runs launch options with its own working directory and `PATH`, so a relative path or a name only your shell's `PATH` knows is not found: use an absolute path (a symlink to the script works too). To wrap a single game in its own gamescope, use `--steer-child`; host mode leaves a game started without `--child` wherever its own defaults put it. Never wrap the Steam client with `--steer-child` or `--child` (the launcher warns for `steam`, `steam.sh`, `steam-runtime`, `com.valvesoftware.Steam` and gamescope's `-e`/`--steam`): all of Steam would render on the dGPU and keep it awake. With Flatpak Steam, launch options run inside the sandbox: the script and `framegen-launch-common.sh` must be somewhere the sandbox can read (e.g. `flatpak override --user --filesystem=/opt/gamescope:ro com.valvesoftware.Steam`). Untested here.

- **Display GPU.** On DRM it is the owner of the `-O` connector when that is connected, else of a connected eDP/LVDS/DSI panel, else of any connected connector, searching the integrated GPU first, then a panel owner, then PCI order. While the lid is closed (`/proc/acpi/button/lid/*/state`) panels count only when `-O` names them. Connector names are per GPU: an `-O` name connected on several GPUs takes the first in that order, with a warning (set `PRESENT_DEV` to choose). With the lid closed, no `-O` and an external connector picked, gamescope gets `-O <that connector>` (shown in `--report`), since it would otherwise prefer the panel. With the lid closed and only the panel connected there is no split; the launcher says so, names the panel for `-O`, and keeps the fallback gamescope on the panel's GPU. Nested, it is the host compositor's GPU: `wayland-info` (linux-dmabuf main device), else the first entry of `KWIN_DRM_DEVICES`, `AQ_DRM_DEVICES` or `WLR_DRM_DEVICES`, or `xrandr --current --listproviders` on X11; without those the panel owner is assumed, lid or not, and the report says so. Inside gamescope a `--child` sees X11 only (gamescope sets `DISPLAY` and drops `WAYLAND_DISPLAY` unless `--expose-wayland`) and Xwayland lists no xrandr provider, so it assumes the panel owner too, which is the GPU gamescope picked whenever that gamescope has a split. Gamescope gets `--prefer-vk-device` for it. The launcher reads `--backend` and `-O`/`--prefer-output` also in getopt_long's abbreviated forms (`--ba`, `--prefer-o`), and refuses `--prefer-vk-device` (`--prefer-v`) in favour of `PRESENT_DEV`.
- **Game steering**, set in the game's environment only, each key only if you have not set it yourself:
  - always `MESA_VK_DEVICE_SELECT=vendor:device!`;
  - NVIDIA driver: `__NV_PRIME_RENDER_OFFLOAD=1 __VK_LAYER_NV_optimus=NVIDIA_only __GLX_VENDOR_LIBRARY_NAME=nvidia` (NVIDIA README, "PRIME Render Offload");
  - Mesa dGPU (amdgpu, i915/xe, nouveau): `DRI_PRIME=pci-DDDD_BB_DD_F` (Mesa environment variables).
  - A `MESA_VK_DEVICE_SELECT` or `DRI_PRIME` you set that names another GPU is kept, with a warning: the game follows it, not the split. In host mode it reaches gamescope's child (Steam) unchanged, like the rest of your environment.
  - `--child` and `--steer-child` also warn when your Vulkan variables hide the game GPU's driver from the game: `VK_ICD_FILENAMES`, `VK_DRIVER_FILES` or `VK_LOADER_DRIVERS_SELECT` without its manifest, `VK_LOADER_DRIVERS_DISABLE` matching it, or `DXVK_FILTER_DEVICE_NAME`/`VKD3D_FILTER_DEVICE_NAME` (they pick a GPU by name). The same check against the display GPU's driver covers gamescope.
  - Existing launch options that pick the dGPU themselves (`prime-run %command%`, `DRI_PRIME=1 %command%`) keep working, since nothing reaches the child to stop them; `--child` does the same with the detected GPU and checks your variables.
- **Only gamescope's own processes are filtered.** Gamescope, Xwayland and gamescopereaper never get the steering keys and, with a split, get `gamescope_upscale_preemptive=false` and filters that keep the game GPU's user-space drivers out:
  - `VK_LOADER_DRIVERS_DISABLE` (Vulkan loader 1.3.234+) with the game GPU's driver manifests: `*nvidia*,*nouveau*` for a GPU bound to `nvidia` (`nvidia_icd.json`, and Mesa's NVK `nouveau_icd.json`, which would otherwise probe it too), `*radeon*,*amd*` for `amdgpu`, `*intel*` for `i915`/`xe`, `*nouveau*` for `nouveau`, minus the display GPU's own. If you set `VK_LOADER_DRIVERS_DISABLE` yourself, gamescope gets your list plus the missing globs.
  - For an NVIDIA game GPU, `__EGL_VENDOR_LIBRARY_FILENAMES` listing the non-NVIDIA glvnd EGL vendors from `/etc/glvnd/egl_vendor.d` and `/usr/share/glvnd/egl_vendor.d`. It is skipped, with a warning, when you set `__EGL_VENDOR_LIBRARY_FILENAMES` or `__EGL_VENDOR_LIBRARY_DIRS`. Without it, Xwayland (glamor) loads NVIDIA's EGL vendor library and opens `/dev/nvidia*` on every start. GBM needs nothing: Mesa's libgbm picks its backend from the device's DRM driver, so the iGPU never loads `nvidia-drm_gbm.so`.
  - **Gamescope's child gets the caller's environment back exactly**: each of those keys is unset again, or restored to your value, and so is `PATH` (see below). Steam and anything it starts therefore see the dGPU as without this launcher: the Steam client itself may open the dGPU (CEF, its Vulkan probes), as under any PRIME setup. That cost is idle power only and is unmeasured. Per game, steer with `--child`.
  - **Measured on the AMD 890M + NVIDIA POC** (host mode, headless backend, `strace -f -e trace=openat` and `/proc/<pid>/fd` and `/proc/<pid>/maps` sampling): after startup, gamescope, Xwayland, gamescopereaper and the child hold no NVIDIA device node or library open; gamescope opens only the iGPU's render node (`renderD128`) and reads the `nvidia_icd.json` and `nouveau_icd.json` manifests without loading either driver; nothing opens `/dev/nvidia*`. Still opened: at startup, Xwayland's Mesa EGL device probe opens the dGPU's render node (`renderD129`, twice) and primary node (`card2`, once) and closes them again (the `libEGL warning: egl: failed to create dri2 screen` lines). No environment variable prevents that probe. Whether these short opens resume a suspended dGPU is not measured.
  - **Known limitation: the filters cover an NVIDIA game GPU only.** For AMD, Intel Arc or nouveau dGPUs gamescope's Mesa drivers may open, and wake, the dGPU: same-driver pairs (AMD iGPU + AMD dGPU, Intel iGPU + Arc) get no Vulkan filter at all, because the loader filters whole drivers and RADV and ANV open every GPU they drive during `vkEnumeratePhysicalDevices`; and one Mesa EGL vendor serves both GPUs, so there is no EGL filter for them. To be measured.
- **Fail-open.** With no possible split (one GPU, MUX in dGPU-only mode, the display on the dGPU while an iGPU exists, two GPUs with the same `vendor:device`, two discrete GPUs without explicit `PRESENT_DEV` and `RENDER_DEV`), the gamescope modes run plain gamescope on the display GPU with the child unsteered, and `--child` runs the game unchanged, except with the display on the dGPU, where it steers the game to that same GPU (no cross-device copy). Fail-open sets no driver filters: a fallback gamescope pinned to the iGPU (lid closed, say) may still load the dGPU's driver. `--child` never blocks a game and reads no gamescope options; words before it (`-f`, `-W 1280 720`, `--steer-child`, ...) are ignored with one notice, never a usage error. Use `--child -- %command%`: a `--` right after it is dropped and the rest runs verbatim without being scanned, so Steam's `steam-launch-wrapper -- reaper SteamLaunch AppId=N -- game` stays intact. Without that `--`, the words after `--child` are skipped only until one names a command (on `PATH`, or an executable path) and everything from there runs, so an option value that happens to name a command (`cvt`, say) would run instead of the game. The notice says what was ignored and what runs (`ignored: -f; running: <word>`). If no word names a command, the words run as given. `--report` and `--help` go before `--child`. Without `framegen-launch-common.sh`, `--child` says so once and runs the game unsteered. In the gamescope modes a forgotten `--` is a usage error (every gamescope option takes at most one value, so two words in a row mean the command started). The reason goes to stderr (one short line; `--report` adds the explanation) and `--report` exits 3.
- `PRESENT_DEV` and `RENDER_DEV` override detection. On DRM an explicit `PRESENT_DEV` uses its own connected connector; nested, it must be the host compositor's GPU when that is known.
- **iGPU vs dGPU** comes from PCI placement: Intel's iGPU is `00:02.0`; an AMD APU's GPU shares its PCI device with AMD SoC functions (vendor `0x1022`); everything else is discrete. `boot_vga` is not used. An NVIDIA GPU without a DRM card node is reported with a hint to load `nvidia-drm` with `modeset=1`; when it is the only other GPU, that is also the reason for no split.
- **Read-only detection.** Detection reads sysfs (`SYSFS_ROOT`, default `/sys`) and, when nested, asks the host compositor (`wayland-info`, `xrandr --current`); it never runs `nvidia-smi`, `vulkaninfo` or `glxinfo`. The report's PCIe cap is the minimum over the dGPU's upstream bridges, plus the dGPU's own link only while it is runtime-active. It warns below NVIDIA 555 and, on DRM, below kernel 6.8 (no explicit sync).
- Gamescope is `GAMESCOPE_BIN`, else `build/src/gamescope` of the checkout the script is in, else `gamescope` on `PATH`; the launcher prints the one it runs. Gamescope starts its child through `gamescopereaper` on `PATH`, so the directory of the `gamescopereaper` next to gamescope (or next to its symlink target, also for a symlink found on `PATH`) is prepended to gamescope's `PATH` only, unless `PATH` already finds that reaper; `--report` shows the child's restored `PATH` as `PATH="$PATH"` (verified on the POC: the checkout's build, started without `scripts/env-gamescope-local.sh` and with no gamescope installed, runs its child). The checkout's build runs without the rest of `scripts/env-gamescope-local.sh` (no `VK_LAYER_PATH` for the WSI layer): if it needs that environment, start the launcher through `scripts/env-gamescope-local.sh` or set `GAMESCOPE_BIN` to an installed gamescope. `-h` is gamescope's `--nested-height`; the launcher's help is `--help`.
- `scripts/framegen-doctor.sh` prints the same report under `HYBRID`, then PASS (split ready), WARN (no split on a laptop-shaped machine) or INFO (one GPU, a desktop). Fixture tests: `bash tests/gamescope_hybrid_test.sh` (no GPU needed).
- **Open measurements:** dGPU `power/runtime_status` and `runtime_active_time` across an idle host-mode session with Steam, on the POC from a session where nothing else holds the dGPU (a TTY without a desktop: is it `suspended` before start, and does Xwayland's startup EGL probe resume it?), on an AMD + AMD laptop (Framework 16) and on the 7900 XT + 5700 XT desktop (the AMD/Arc limitation above); how often the Steam client itself opens the dGPU under host mode (`steamwebhelper`'s fds and maps with Steam idle).
