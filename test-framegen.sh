#!/usr/bin/env bash
#
# test-framegen.sh — drive gamescope compositor-side frame generation for testing.
#
# Usage:
#   ./test-framegen.sh gpus                            List GPUs and their vendor:device IDs.
#   ./test-framegen.sh bench [gpu]                     GPU-only shader microbenchmark (no display).
#   ./test-framegen.sh run [gpu] [mode] [res] [-- app args]
#                                                       Nested run + cadence summary.
#
#   gpu  : auto | nvidia | amd | intel | <vendor:device>   (default: auto)
#   mode : extrapolate | motion | blend         (default: extrapolate)
#   res  : 1080 | 1440 | 2160                    (default: 1080)
#   app  : command to run inside gamescope       (default: vkcube)
#
# Env overrides:
#   REFRESH=144        display refresh the compositor targets
#   LIMIT=2            game fps = REFRESH / LIMIT (the gap framegen fills; 2 => x2)
#   DEBUG_EVERY=1      log every Nth framegen event (1 = exact counts, higher = less spam)
#   DURATION=          seconds to auto-stop a 'run' (empty = until you close the window)
#   QUALITY=high       motion tier: low | medium | high | ultra | extreme
#
# Examples:
#   ./test-framegen.sh bench amd
#   ./test-framegen.sh run auto motion 1440
#   DURATION=12 ./test-framegen.sh run 1002:1234 extrapolate 1080 -- vkgears
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Always use the checkout's environment wrapper. It supplies locally compiled
# dependencies when GAMESCOPE_LOCAL_PREFIX is set and keeps the selected build's
# WSI layer ahead of system paths.
export GAMESCOPE_BUILD_DIR="${GAMESCOPE_BUILD_DIR:-build}"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/env-gamescope-local.sh"

# Use exactly the requested local build. Falling through to a system binary or a
# different build tree makes benchmark and cadence comparisons meaningless.
BIN="${GAMESCOPE_BIN:-$SCRIPT_DIR/$GAMESCOPE_BUILD_DIR/src/gamescope}"

require_binary() {
    if [[ ! -x "$BIN" ]]; then
        echo "error: gamescope binary not found at '$BIN'." >&2
        echo "       Set GAMESCOPE_BUILD_DIR or GAMESCOPE_BIN, then build it with ninja." >&2
        return 1
    fi
}

# Map a friendly GPU name to a vendor:device id via lspci. On a dual-AMD box
# (both vendor 1002) pass the explicit id instead — see 'gpus'.
resolve_gpu() {
    local gpu="${1:-auto}" id=""
    case "$gpu" in
        auto)   id=$(lspci -nn | grep -iE 'vga|3d controller|display controller' | grep -oiE '[0-9a-f]{4}:[0-9a-f]{4}' | head -1 || true) ;;
        nvidia) id=$(lspci -nn | grep -iE 'vga|3d controller|display controller' | grep -oiE '10de:[0-9a-f]{4}' | head -1 || true) ;;
        amd)    id=$(lspci -nn | grep -iE 'vga|3d controller|display controller' | grep -oiE '1002:[0-9a-f]{4}' | head -1 || true) ;;
        intel)  id=$(lspci -nn | grep -iE 'vga|3d controller|display controller' | grep -oiE '8086:[0-9a-f]{4}' | head -1 || true) ;;
        [0-9a-fA-F][0-9a-fA-F][0-9a-fA-F][0-9a-fA-F]:[0-9a-fA-F][0-9a-fA-F][0-9a-fA-F][0-9a-fA-F]) id="${gpu,,}" ;;
        *) echo "error: unknown gpu '$gpu' (use auto|nvidia|amd|intel|<vendor:device>)" >&2; return 1 ;;
    esac
    if [[ -z "$id" ]]; then
        echo "error: no GPU matching '$gpu' found (try './test-framegen.sh gpus')" >&2
        return 1
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
    lspci -nn | grep -iE 'vga|3d controller|display controller' | sed 's/^/  /' || true
    echo
    echo "Vulkan devices:"
    if command -v vulkaninfo >/dev/null 2>&1; then
        vulkaninfo --summary 2>/dev/null | grep -E 'deviceName|driverName|deviceType' | sed 's/^/  /' || true
    else
        echo "  (vulkaninfo not installed)"
    fi
    echo
    echo "Pass a vendor:device ID as the [gpu] argument to pin a specific card."
}

cmd_bench() {
    require_binary
    local id quality="${QUALITY:-high}"; id=$(resolve_gpu "${1:-auto}")
    case "$quality" in low|medium|high|ultra|extreme) ;; *) echo "error: QUALITY must be low|medium|high|ultra|extreme" >&2; exit 1 ;; esac
    echo "# framegen GPU microbenchmark — device $id"
    echo "# quality=$quality (compare extrapolate variants and the selected motion pipeline)"
    GAMESCOPE_FRAMEGEN_BENCHMARK=1 "$BIN" --backend headless --prefer-vk-device "$id" \
        --framegen-quality "$quality" -- true 2>/dev/null
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
        echo "    - Shared-queue motion mode needs a short stable-cadence warm-up."
        echo "    - Full log: $log"
    fi
    echo "================================================================="
}

cmd_run() {
    require_binary
    local id mode res app w h
    local -a app_cmd
    id=$(resolve_gpu "${1:-auto}")
    mode="${2:-extrapolate}"
    res="${3:-1080}"
    if [[ "${4:-}" == "--" ]]; then
        app_cmd=( "${@:5}" )
    elif (( $# > 3 )); then
        app_cmd=( "${@:4}" )
        if (( ${#app_cmd[@]} == 1 )) && [[ "${app_cmd[0]}" == *[[:space:]]* ]]; then
            # Compatibility with the old one-string app argument.
            read -r -a app_cmd <<< "${app_cmd[0]}"
        fi
    else
        app_cmd=( vkcube )
    fi
    (( ${#app_cmd[@]} > 0 )) || { echo "error: app command is empty" >&2; return 1; }
    printf -v app '%q ' "${app_cmd[@]}"
    app="${app% }"
    read -r w h < <(res_to_dims "$res")
    local refresh="${REFRESH:-144}" limit="${LIMIT:-2}" every="${DEBUG_EVERY:-1}" quality="${QUALITY:-high}"

    case "$mode" in extrapolate|motion|blend) ;; *) echo "error: mode must be extrapolate|motion|blend" >&2; exit 1 ;; esac
    case "$quality" in low|medium|high|ultra|extreme) ;; *) echo "error: QUALITY must be low|medium|high|ultra|extreme" >&2; exit 1 ;; esac
    [[ "$refresh" =~ ^[1-9][0-9]*$ ]] || { echo "error: REFRESH must be a positive integer" >&2; return 1; }
    [[ "$limit" =~ ^[1-9][0-9]*$ ]] || { echo "error: LIMIT must be a positive integer" >&2; return 1; }
    [[ "$every" =~ ^[1-9][0-9]*$ ]] || { echo "error: DEBUG_EVERY must be a positive integer" >&2; return 1; }
    if [[ -n "${DURATION:-}" && ! "$DURATION" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
        echo "error: DURATION must be a non-negative number of seconds" >&2; return 1
    fi
    if ! command -v "${app_cmd[0]}" >/dev/null 2>&1 && [[ ! -x "${app_cmd[0]}" ]]; then
        echo "error: test app '${app_cmd[0]}' not found" >&2; exit 1
    fi

    local log; log="$(mktemp -t framegen-run.XXXXXX.log)"
    echo "# device=$id  mode=$mode/$quality  res=${w}x${h}  refresh=${refresh}  game=~$((refresh/limit))fps  app=$app"
    echo "# log: $log"
    echo "# close the gamescope window (or Ctrl+C) to stop and see the summary."
    [[ -n "${DURATION:-}" ]] && echo "# auto-stop after ${DURATION}s"
    echo

    local -a cmd=(
        "$BIN" --backend wayland --prefer-vk-device "$id"
        -W "$w" -H "$h" -r "$refresh" --framerate-limit "$limit"
        --experimental-framegen --framegen-mode "$mode" --framegen-quality "$quality" --framegen-strength 0.5
        --framegen-debug -- "${app_cmd[@]}"
    )

    local run_status
    local -a pipeline_status
    set +e
    if [[ -n "${DURATION:-}" ]]; then
        GAMESCOPE_FRAMEGEN_DEBUG_EVERY="$every" timeout "$DURATION" "${cmd[@]}" 2>&1 | tee "$log"
        pipeline_status=( "${PIPESTATUS[@]}" )
    else
        GAMESCOPE_FRAMEGEN_DEBUG_EVERY="$every" "${cmd[@]}" 2>&1 | tee "$log"
        pipeline_status=( "${PIPESTATUS[@]}" )
    fi
    set -e

    run_status=${pipeline_status[0]}
    if (( run_status == 0 && pipeline_status[1] != 0 )); then
        run_status=${pipeline_status[1]}
    fi

    summarize "$log"
    if [[ -n "${DURATION:-}" && "$run_status" -eq 124 ]]; then
        return 0
    fi
    return "$run_status"
}

case "${1:-}" in
    gpus)  cmd_gpus ;;
    bench) shift; cmd_bench "$@" ;;
    run)   shift; cmd_run "$@" ;;
    ""|-h|--help|help)
        sed -n '3,26p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
        ;;
    *) echo "error: unknown subcommand '$1' (use gpus|bench|run)" >&2; exit 1 ;;
esac
