#!/usr/bin/env bash
#
# test-framegen.sh — drive gamescope compositor-side frame generation for testing.
#
# Usage:
#   ./test-framegen.sh gpus                             List GPUs and their vendor:device ids.
#   ./test-framegen.sh bench [gpu]                      GPU-only shader microbenchmark (no display).
#   ./test-framegen.sh run   [gpu] [mode] [res] [app]   Nested end-to-end run + cadence summary.
#
#   gpu  : nvidia | amd | <vendor:device hex>   (default: nvidia)   e.g. 1002:150e
#   mode : extrapolate | motion | blend         (default: extrapolate)
#   res  : 1080 | 1440 | 2160                    (default: 1080)
#   app  : command to run inside gamescope       (default: vkcube)
#
# Env overrides:
#   REFRESH=144        display refresh the compositor targets
#   LIMIT=2            game fps = REFRESH / LIMIT (the gap framegen fills; 2 => x2)
#   DEBUG_EVERY=1      log every Nth framegen event (1 = exact counts, higher = less spam)
#   DURATION=          seconds to auto-stop a 'run' (empty = until you close the window)
#
# Examples:
#   ./test-framegen.sh bench amd
#   ./test-framegen.sh run nvidia motion 1440
#   DURATION=12 ./test-framegen.sh run 1002:150e extrapolate 1080 vkgears
#
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# This checkout is linked against the manually built Wayland/Pixman stack.
# Keep every benchmark/run on that ABI; invoking the binary directly against
# distro libraries can fail at load time or, worse, mix incompatible symbols.
export GAMESCOPE_BUILD_DIR="${GAMESCOPE_BUILD_DIR:-build-perf}"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/env-gamescope-local.sh"

# Locate the gamescope binary (prefer the perf build).
BIN=""
for cand in "$SCRIPT_DIR/$GAMESCOPE_BUILD_DIR/src/gamescope" "$SCRIPT_DIR/build-perf/src/gamescope" "$SCRIPT_DIR/build/src/gamescope"; do
    [[ -x "$cand" ]] && { BIN="$cand"; break; }
done
if [[ -z "$BIN" ]]; then
    echo "error: gamescope binary not found. Build it first (ninja -C build-perf src/gamescope)." >&2
    exit 1
fi

# Map a friendly GPU name to a vendor:device id via lspci. On a dual-AMD box
# (both vendor 1002) pass the explicit id instead — see 'gpus'.
resolve_gpu() {
    local gpu="${1:-nvidia}" id=""
    case "$gpu" in
        nvidia) id=$(lspci -nn | grep -iE 'vga|3d controller|display controller' | grep -oiE '10de:[0-9a-f]{4}' | head -1) ;;
        amd)    id=$(lspci -nn | grep -iE 'vga|3d controller|display controller' | grep -oiE '1002:[0-9a-f]{4}' | head -1) ;;
        intel)  id=$(lspci -nn | grep -iE 'vga|3d controller|display controller' | grep -oiE '8086:[0-9a-f]{4}' | head -1) ;;
        [0-9a-fA-F]*:[0-9a-fA-F]*) id="$gpu" ;;
        *) echo "error: unknown gpu '$gpu' (use nvidia|amd|intel|<vendor:device>)" >&2; exit 1 ;;
    esac
    if [[ -z "$id" ]]; then
        echo "error: no GPU matching '$gpu' found (try './test-framegen.sh gpus')" >&2
        exit 1
    fi
    echo "$id"
}

res_to_dims() {
    case "${1:-1080}" in
        1080) echo "1920 1080" ;;
        1440) echo "2560 1440" ;;
        2160) echo "3840 2160" ;;
        *) echo "error: res must be 1080|1440|2160" >&2; exit 1 ;;
    esac
}

cmd_gpus() {
    echo "GPUs (lspci):"
    lspci -nn | grep -iE 'vga|3d controller|display controller' | sed 's/^/  /'
    echo
    echo "Vulkan devices:"
    if command -v vulkaninfo >/dev/null 2>&1; then
        vulkaninfo --summary 2>/dev/null | grep -E 'deviceName|driverName|deviceType' | sed 's/^/  /'
    else
        echo "  (vulkaninfo not installed)"
    fi
    echo
    echo "Pass a vendor:device id (e.g. 1002:150e) as the [gpu] arg to pin a specific card."
}

cmd_bench() {
    local id; id=$(resolve_gpu "${1:-nvidia}")
    echo "# framegen GPU microbenchmark — device $id"
    echo "# (bandwidth-bound passes; compare extrapolate fp16 vs fp32 and motion warp)"
    GAMESCOPE_FRAMEGEN_BENCHMARK=1 "$BIN" --backend headless --prefer-vk-device "$id" -- true 2>/dev/null
}

summarize() {
    local log="$1"
    echo
    echo "==================== framegen cadence summary ===================="
    if grep -q 'framegen: enabled' "$log"; then
        grep 'framegen: enabled' "$log" | tail -1 | sed 's/.*framegen:/  framegen:/'
    else
        echo "  framegen never enabled — no base frames were composited"
        echo "  (window didn't render? try a different app, or check the log below)"
    fi
    local real gen_slot repeat batch presented tooslow
    real=$(grep -c 'vblank slot=real' "$log" || true)
    gen_slot=$(grep -c 'vblank slot=generated' "$log" || true)
    repeat=$(grep -c 'vblank slot=repeat' "$log" || true)
    batch=$(grep -c 'framegen: generated ' "$log" || true)
    presented=$(grep -c 'framegen: presented generated' "$log" || true)
    tooslow=$(grep -c 'generation_too_slow' "$log" || true)
    printf '  %-26s %s\n' "vblank slots: real"        "$real"
    printf '  %-26s %s\n' "vblank slots: generated"   "$gen_slot"
    printf '  %-26s %s\n' "vblank slots: repeat"      "$repeat"
    printf '  %-26s %s\n' "generation batches"        "$batch"
    printf '  %-26s %s\n' "generated frames presented" "$presented"
    printf '  %-26s %s\n' "discarded (too slow)"      "$tooslow"
    echo "-----------------------------------------------------------------"
    if [[ "${presented:-0}" -gt 0 ]]; then
        echo "  RESULT: PASS — synthetic frames were generated and presented."
    else
        echo "  RESULT: no generated frames presented."
        echo "    - Ensure the game runs BELOW the display rate (LIMIT>1); with no gap"
        echo "      there are no empty vblanks to fill."
        echo "    - 'motion' mode needs ~8 stable frames before it activates."
        echo "    - Full log: $log"
    fi
    echo "================================================================="
}

cmd_run() {
    local id mode res app w h
    local -a app_cmd
    id=$(resolve_gpu "${1:-nvidia}")
    mode="${2:-extrapolate}"
    res="${3:-1080}"
    app="${4:-vkcube}"
    read -r -a app_cmd <<< "$app"
    read -r w h < <(res_to_dims "$res")
    local refresh="${REFRESH:-144}" limit="${LIMIT:-2}" every="${DEBUG_EVERY:-1}"

    case "$mode" in extrapolate|motion|blend) ;; *) echo "error: mode must be extrapolate|motion|blend" >&2; exit 1 ;; esac
    if ! command -v "${app%% *}" >/dev/null 2>&1; then
        echo "error: test app '${app%% *}' not found in PATH" >&2; exit 1
    fi

    local log; log="$(mktemp -t framegen-run.XXXXXX.log)"
    echo "# device=$id  mode=$mode  res=${w}x${h}  refresh=${refresh}  game=~$((refresh/limit))fps  app=$app"
    echo "# log: $log"
    echo "# close the gamescope window (or Ctrl+C) to stop and see the summary."
    [[ -n "${DURATION:-}" ]] && echo "# auto-stop after ${DURATION}s"
    echo

    local -a cmd=(
        "$BIN" --backend wayland --prefer-vk-device "$id"
        -W "$w" -H "$h" -r "$refresh" --framerate-limit "$limit"
        --experimental-framegen --framegen-mode "$mode" --framegen-strength 0.5
        --framegen-debug -- "${app_cmd[@]}"
    )

    if [[ -n "${DURATION:-}" ]]; then
        GAMESCOPE_FRAMEGEN_DEBUG_EVERY="$every" timeout "$DURATION" "${cmd[@]}" 2>&1 | tee "$log" || true
    else
        GAMESCOPE_FRAMEGEN_DEBUG_EVERY="$every" "${cmd[@]}" 2>&1 | tee "$log" || true
    fi

    summarize "$log"
}

case "${1:-}" in
    gpus)  cmd_gpus ;;
    bench) shift; cmd_bench "$@" ;;
    run)   shift; cmd_run "$@" ;;
    ""|-h|--help|help)
        sed -n '2,32p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
        ;;
    *) echo "error: unknown subcommand '$1' (use gpus|bench|run)" >&2; exit 1 ;;
esac
