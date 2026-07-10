#!/usr/bin/env bash
# Watch compositor-side FRAME GENERATION live (nested inside your GNOME session).
#
# Dual-GPU split (the target topology):
#   * the CLIENT renders on the NVIDIA card (RENDER_DEV)
#   * gamescope does compositing + frame generation + present on the AMD card
#     (PRESENT_DEV) via a cross-GPU dma-buf share.
# vkmark's GPU-heavy "desktop" scene stands in for a GPU-bound game so real frames
# arrive slower than the display and framegen has real gaps to fill.
#
# NOTE: nested inside GNOME the PACING is parent-jittered — good for *seeing*
# generation happen, but the true smoothness test is run-framegen-native.sh (DRM).
#
# Usage:  ./watch-framegen.sh [mode] [multiplier] [windows] [WxH] [refresh]
#   mode        motion | extrapolate | blend        (default motion)
#   multiplier  2 | 3 | 4                            (default 3)
#   windows     vkmark desktop load; more = lower base fps.
#               On the fast NVIDIA card you need a lot: ~1000@4K -> ~40fps,
#               ~1500@4K -> ~27fps. (default 1000)
#   WxH         render size                          (default 3840x2160)
#   refresh     gamescope vblank rate                (default 120)
#
# Env:  RENDER_DEV (default 10de:2db9 NVIDIA)  PRESENT_DEV (default 1002:150e AMD)
#       APP=...  override the client command (a real game, GravityMark, etc.)
#
# Watch for smooth motion in the moving windows, and the log line:
#   framegen: generated N frame(s) ... mode=motion(xM) ... gpu=X.XXms
set -u
cd /home/awarth/Devstuff/dummy/gamescope || exit 1

source "$PWD/env-gamescope-local.sh"

RENDER_DEV=${RENDER_DEV:-10de:2db9}     # client renders here (NVIDIA)
PRESENT_DEV=${PRESENT_DEV:-1002:150e}   # framegen + present here (AMD)
mode=${1:-motion}; mult=${2:-3}; wins=${3:-1000}; res=${4:-3840x2160}; rhz=${5:-120}
W=${res%x*}; H=${res#*x}
APP=${APP:-vkmark --size $res -b desktop:windows=${wins}:duration=600}

echo "# NESTED  framegen: mode=$mode x$mult  render=$RENDER_DEV  present=$PRESENT_DEV"
echo "# ${W}x${H} @ ${rhz}Hz  client: $APP"

# gamescope on the AMD present card; the client is pinned to NVIDIA via a
# per-process env so it doesn't override gamescope's --prefer-vk-device.
gamescope --expose-wayland --backend wayland --prefer-vk-device "$PRESENT_DEV" \
  -W "$W" -H "$H" -r "$rhz" \
  --experimental-framegen --framegen-mode "$mode" --framegen-multiplier "$mult" \
  --framegen-strength 0.5 --framegen-debug \
  -- env MESA_VK_DEVICE_SELECT="$RENDER_DEV" $APP
