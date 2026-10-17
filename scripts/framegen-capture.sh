#!/usr/bin/env bash
# Record what a client draws, as consecutive full-resolution luma frames, for
# the offline field gate (framegen-field-gate.py). gamescope runs headless
# (exact W x H, no window, framegen off) and its PipeWire stream is written as
# raw 8-bit luma by GStreamer. The whole process group is killed on exit.
#
#   framegen-capture.sh NAME W H WARMUP_S SECONDS -- client command...
#
# Writes $OUT/NAME-WxH.gray (frames of W*H bytes) and NAME-WxH.log.
# Environment: OUT (default ${XDG_CACHE_HOME:-$HOME/.cache}/gamescope-framegen-captures),
# PRESENT_DEV (gamescope --prefer-vk-device), GAMESCOPE_BIN (default build/src/gamescope),
# RATE (output refresh, default 60). For ground truth at in-between phases
# (framegen-temporal-gate.py) the client must render faster than RATE, so every
# captured frame is a distinct, evenly spaced moment.
# Needs gst-launch-1.0 with pipewiresrc and a gamescope built with PipeWire.
set -u
if (($# < 7)) || [[ $6 != -- ]]; then
	echo "usage: $0 NAME W H WARMUP_S SECONDS -- command..." >&2
	exit 2
fi
name=$1 W=$2 H=$3 warm=$4 secs=$5
shift 6
REPO=$(cd "$(dirname "$0")/.." && pwd)
OUT=${OUT:-${XDG_CACHE_HOME:-$HOME/.cache}/gamescope-framegen-captures}
GAMESCOPE_BIN=${GAMESCOPE_BIN:-$REPO/build/src/gamescope}
mkdir -p "$OUT"
log="$OUT/$name-${W}x$H.log" raw="$OUT/$name-${W}x$H.gray"
rm -f "$raw"
echo "capture $name ${W}x$H@${RATE:-60}: warm-up ${warm}s, record ${secs}s: $*"

device=()
[[ -n "${PRESENT_DEV:-}" ]] && device=(--prefer-vk-device "$PRESENT_DEV")
setsid "$REPO/scripts/env-gamescope-local.sh" "$GAMESCOPE_BIN" "${device[@]}" \
	-W "$W" -H "$H" -w "$W" -h "$H" -r "${RATE:-60}" --backend headless -- env -u WAYLAND_DISPLAY "$@" >"$log" 2>&1 &
gs=$!
# SIGINT gamescope alone first: signalling the group kills Xwayland before
# gamescope, which then aborts on the X11 I/O error.
cleanup() { kill -INT "$gs" 2>/dev/null; timeout 10 tail --pid="$gs" -f /dev/null; kill -KILL -- -"$gs" 2>/dev/null; }
trap cleanup EXIT

node=''
for _ in $(seq 100); do
	node=$(grep -o 'stream available on node ID: [0-9]*' "$log" | grep -o '[0-9]*$' | head -1)
	[[ -n "$node" ]] && break
	timeout 0.2 tail --pid="$gs" -f /dev/null
done
[[ -n "$node" ]] || { echo "framegen-capture: no PipeWire node (see $log)" >&2; exit 1; }
timeout "$warm" tail --pid="$gs" -f /dev/null
timeout -s INT "$secs" gst-launch-1.0 -e -q pipewiresrc path="$node" always-copy=true \
	! videoconvert ! "video/x-raw,format=GRAY8,width=$W,height=$H" ! filesink location="$raw"
rc=$?
# 124 is the expected end (timeout's SIGINT); anything else is a failed capture.
((rc == 0 || rc == 124)) || { echo "framegen-capture: gst-launch failed (rc=$rc, see $log)" >&2; exit 1; }
size=$(stat -c %s "$raw" 2>/dev/null || echo 0)
((size > 0 && size % (W * H) == 0)) || { echo "framegen-capture: $raw holds no whole frames ($size bytes)" >&2; exit 1; }
echo "$name ${W}x$H: $((size / (W * H))) frames -> $raw"
