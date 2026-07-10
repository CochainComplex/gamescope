#!/usr/bin/env bash
# ============================================================================
# Run compositor-side FRAME GENERATION on the NATIVE DRM/KMS backend.
#
# This is bare-metal gamescope: it takes over the physical display directly
# (like a Steam Deck session), so the vblank pacing is REGULAR — real display
# flips, not the parent-jittered timing you get nested inside GNOME. This is the
# environment where the wobble fixes actually show and where "smooth, low
# latency" is measurable.
#
# ---------------------------------------------------------------------------
# HOW TO RUN (must NOT be inside your GNOME session):
#   1. Switch to a free virtual terminal:            Ctrl+Alt+F3
#   2. Log in as your user at the text prompt.
#   3. cd /home/awarth/Devstuff/dummy/gamescope
#      ./run-framegen-native.sh [mode] [mult] [windows] [WxH] [refresh]
#   4. Ctrl+C stops it. Return to GNOME with Ctrl+Alt+F2 (or F1).
#
# It CANNOT run from inside GNOME (GNOME already holds the display / DRM master).
#
# Args (all optional):
#   mode        motion | extrapolate | blend        (default motion)
#   mult        2 | 3 | 4                            (default 2)
#   windows     vkmark desktop load; more = lower base fps
#               (600 -> ~58fps, 900 -> ~40, 1200 -> ~30 @1440p on the 890M)
#   WxH         output size                          (default 2560x1440)
#   refresh     display refresh to request           (default 120)
#
# Dual-GPU split (the target topology):
#   RENDER_DEV   card the CLIENT/game renders on (the powerful one).
#                Default 10de:2db9 = NVIDIA RTX PRO 500 (this laptop's dGPU).
#                On the dual-AMD desktop set this to the 7900 XT.
#   PRESENT_DEV  card that drives the display + runs framegen + presents.
#                Default 1002:150e = AMD 890M (drives this laptop's panel).
#                On the dual-AMD desktop set this to the 5700 XT the monitor is on.
#   The NVIDIA-rendered frame is shared to AMD over dma-buf; gamescope composites,
#   generates the in-between frames, and flips — all on the AMD/present card.
#   APP=...      override the client command (e.g. a real game or GravityMark)
# ============================================================================
set -u
cd /home/awarth/Devstuff/dummy/gamescope || exit 1

# Newer wayland/pixman/seatd libs this build links against, plus the local WSI layer.
source "$PWD/env-gamescope-local.sh"

RENDER_DEV=${RENDER_DEV:-10de:2db9}     # client renders here (NVIDIA)
PRESENT_DEV=${PRESENT_DEV:-1002:150e}   # framegen + display here (AMD)
mode=${1:-motion}; mult=${2:-2}; wins=${3:-600}; res=${4:-2560x1440}; rhz=${5:-120}
W=${res%x*}; H=${res#*x}
APP=${APP:-vkmark --size $res -b desktop:windows=${wins}:duration=600}

# Refuse to run inside a compositor (it would just fail to get DRM master).
if [ -n "${WAYLAND_DISPLAY:-}" ] || [ -n "${DISPLAY:-}" ]; then
  echo "!! WAYLAND_DISPLAY/DISPLAY is set — you're inside GNOME/X." >&2
  echo "!! Switch to a text VT first (Ctrl+Alt+F3, log in), then run this." >&2
  echo "!! (Set FORCE=1 to override, but DRM master will almost certainly fail.)" >&2
  [ "${FORCE:-0}" = 1 ] || exit 1
fi

LOGFILE=${LOGFILE:-/tmp/framegen-native.log}
echo "# NATIVE DRM  framegen: mode=$mode x$mult"
echo "# render(client)=$RENDER_DEV   present+framegen(display)=$PRESENT_DEV"
echo "# ${W}x${H} @ ${rhz}Hz requested  |  client: $APP"
echo "# full output is ALSO saved to: $LOGFILE"
echo "# watch the log for:  framegen: generated N frame(s) ... mode=${mode}(x${mult}) ... gpu=X.XXms"
echo

# gamescope composites/generates/presents on the AMD (PRESENT_DEV). The client is
# launched via `env MESA_VK_DEVICE_SELECT=...` so ONLY it is pinned to the NVIDIA
# (RENDER_DEV) — gamescope keeps its own --prefer-vk-device untouched.
# Output is tee'd to $LOGFILE so a failure can be inspected afterwards.
gamescope --backend drm --prefer-vk-device "$PRESENT_DEV" \
  -W "$W" -H "$H" -r "$rhz" \
  --expose-wayland \
  --experimental-framegen --framegen-mode "$mode" --framegen-multiplier "$mult" \
  --framegen-strength 0.5 --framegen-debug \
  -- env MESA_VK_DEVICE_SELECT="$RENDER_DEV" $APP 2>&1 | tee "$LOGFILE"
