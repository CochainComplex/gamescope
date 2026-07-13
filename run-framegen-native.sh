#!/usr/bin/env bash
# Run compositor-side frame generation directly on the DRM/KMS backend.
set -euo pipefail

usage()
{
	cat <<'EOF'
Usage (from a text VT, outside a graphical session):
  RENDER_DEV=vendor:device! PRESENT_DEV=vendor:device \
    ./run-framegen-native.sh [mode] [multiplier] [windows] [WxH] [refresh] [-- client [args...]]

Arguments:
  mode        motion | extrapolate | blend  (default: motion)
  multiplier  2 | 3 | 4                     (default: 2)
  windows     vkmark desktop workload       (default: 600)
  WxH         output size                    (default: 2560x1440)
  refresh     display refresh                (default: 120)

The client renders on RENDER_DEV. Gamescope composites, generates, and scans
out on PRESENT_DEV, which must drive the selected display. Pass a client after
`--`; otherwise vkmark is used. APP remains available for compatibility, but
cannot preserve complex shell quoting.

Switch to a free text VT (for example Ctrl+Alt+F3), log in, and run this script.
The graphical session normally owns DRM master, so this script refuses to run
when DISPLAY or WAYLAND_DISPLAY is set unless FORCE=1 is explicitly provided.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
	usage
	exit 0
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env-gamescope-local.sh
source "$SCRIPT_DIR/env-gamescope-local.sh"
# shellcheck source=scripts/framegen-launch-common.sh
source "$SCRIPT_DIR/scripts/framegen-launch-common.sh"

framegen_parse_launch_args 2 600 2560x1440 120 "$@"
mode="$FRAMEGEN_MODE"
multiplier="$FRAMEGEN_MULTIPLIER"
windows="$FRAMEGEN_WINDOWS"
resolution="$FRAMEGEN_RESOLUTION"
refresh="$FRAMEGEN_REFRESH"
app_args=( "${FRAMEGEN_APP_ARGS[@]}" )

framegen_require_device_ids
framegen_validate_launch_args "$mode" "$multiplier" "$windows" "$resolution" "$refresh"
framegen_resolve_gamescope_binary
framegen_build_app_command "$resolution" "$windows" "${app_args[@]}"
framegen_format_command

if [[ -n "${WAYLAND_DISPLAY:-}" || -n "${DISPLAY:-}" ]]; then
	printf '%s\n' 'error: DISPLAY/WAYLAND_DISPLAY is set; run native DRM mode from a text VT' >&2
	printf '%s\n' '       Set FORCE=1 only when the current session does not own DRM master.' >&2
	[[ "${FORCE:-0}" == "1" ]] || exit 1
fi

LOGFILE="${LOGFILE:-/tmp/framegen-native.log}"
printf '# NATIVE DRM framegen: mode=%s x%s\n' "$mode" "$multiplier"
printf '# render=%s present+framegen=%s\n' "$RENDER_DEV" "$PRESENT_DEV"
printf '# %sx%s @ %s Hz; client: %s\n' \
	"$FRAMEGEN_WIDTH" "$FRAMEGEN_HEIGHT" "$refresh" "$FRAMEGEN_APP_DISPLAY"
printf '# log: %s\n\n' "$LOGFILE"

"$FRAMEGEN_GAMESCOPE_BIN" \
	--backend drm --prefer-vk-device "$PRESENT_DEV" \
	-W "$FRAMEGEN_WIDTH" -H "$FRAMEGEN_HEIGHT" -r "$refresh" \
	--expose-wayland \
	--experimental-framegen --framegen-mode "$mode" \
	--framegen-multiplier "$multiplier" --framegen-strength 0.5 \
	--framegen-debug -- \
	env MESA_VK_DEVICE_SELECT="$RENDER_DEV" "${FRAMEGEN_APP_CMD[@]}" \
	2>&1 | tee "$LOGFILE"
