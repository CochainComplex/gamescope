#!/usr/bin/env bash
# Watch compositor-side frame generation in a nested desktop window.
set -euo pipefail

usage()
{
	cat <<'EOF'
Usage:
  RENDER_DEV=vendor:device! PRESENT_DEV=vendor:device \
    ./watch-framegen.sh [mode] [multiplier] [windows] [WxH] [refresh] [-- client [args...]]

Arguments:
  mode        motion | extrapolate | blend  (default: motion)
  multiplier  2 | 3 | 4                     (default: 3)
  windows     vkmark desktop workload       (default: 1000)
  WxH         render/output size             (default: 3840x2160)
  refresh     nested target refresh          (default: 120)

The client renders on RENDER_DEV. Gamescope composites, generates, and presents
on PRESENT_DEV. Pass a client after `--`; otherwise vkmark is used. APP remains
available for compatibility, but cannot preserve complex shell quoting.

This nested path is useful for visual and functional checks. Its cadence also
contains the parent compositor's jitter; use run-framegen-native.sh from a text
VT for direct DRM/KMS pacing tests.
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

framegen_parse_launch_args 3 1000 3840x2160 120 "$@"
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

printf '# NESTED framegen: mode=%s x%s render=%s present=%s\n' \
	"$mode" "$multiplier" "$RENDER_DEV" "$PRESENT_DEV"
printf '# %sx%s @ %s Hz; client: %s\n' \
	"$FRAMEGEN_WIDTH" "$FRAMEGEN_HEIGHT" "$refresh" "$FRAMEGEN_APP_DISPLAY"

"$FRAMEGEN_GAMESCOPE_BIN" \
	--expose-wayland --backend wayland --prefer-vk-device "$PRESENT_DEV" \
	-W "$FRAMEGEN_WIDTH" -H "$FRAMEGEN_HEIGHT" -r "$refresh" \
	--experimental-framegen --framegen-mode "$mode" \
	--framegen-multiplier "$multiplier" --framegen-strength 0.5 \
	--framegen-debug -- \
	env MESA_VK_DEVICE_SELECT="$RENDER_DEV" "${FRAMEGEN_APP_CMD[@]}"
