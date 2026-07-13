#!/usr/bin/env bash
# Shared argument and environment checks for the frame-generation launchers.
# shellcheck disable=SC2034 # FRAMEGEN_* variables are this sourced library's API.

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
	printf 'error: %s is a library; source it from a framegen launcher\n' "${BASH_SOURCE[0]}" >&2
	exit 2
fi

framegen_die()
{
	printf 'error: %s\n' "$*" >&2
	return 1
}

framegen_require_device_ids()
{
	if [[ -z "${RENDER_DEV:-}" || -z "${PRESENT_DEV:-}" ]]; then
		framegen_die \
			"set RENDER_DEV and PRESENT_DEV to PCI vendor:device IDs (run ./test-framegen.sh gpus)"
		return
	fi

	if [[ ! "$RENDER_DEV" =~ ^[[:xdigit:]]{4}:[[:xdigit:]]{4}!?$ ]]; then
		framegen_die "RENDER_DEV must be vendor:device or vendor:device!, got '$RENDER_DEV'"
		return
	fi

	if [[ ! "$PRESENT_DEV" =~ ^[[:xdigit:]]{4}:[[:xdigit:]]{4}$ ]]; then
		framegen_die "PRESENT_DEV must be vendor:device without a trailing !, got '$PRESENT_DEV'"
		return
	fi
}

framegen_parse_launch_args()
{
	local default_multiplier="$1"
	local default_windows="$2"
	local default_resolution="$3"
	local default_refresh="$4"
	shift 4

	local separator_seen=0
	local -a launch_args=()
	FRAMEGEN_APP_ARGS=()
	while (( $# > 0 )); do
		if (( !separator_seen )) && [[ "$1" == "--" ]]; then
			separator_seen=1
		elif (( separator_seen )); then
			FRAMEGEN_APP_ARGS+=( "$1" )
		else
			launch_args+=( "$1" )
		fi
		shift
	done

	if (( ${#launch_args[@]} > 5 )); then
		framegen_die "expected at most five launcher arguments before --"
		return
	fi
	if (( separator_seen && ${#FRAMEGEN_APP_ARGS[@]} == 0 )); then
		framegen_die "-- must be followed by a client command"
		return
	fi

	FRAMEGEN_MODE="${launch_args[0]:-motion}"
	FRAMEGEN_MULTIPLIER="${launch_args[1]:-$default_multiplier}"
	FRAMEGEN_WINDOWS="${launch_args[2]:-$default_windows}"
	FRAMEGEN_RESOLUTION="${launch_args[3]:-$default_resolution}"
	FRAMEGEN_REFRESH="${launch_args[4]:-$default_refresh}"
}

framegen_validate_launch_args()
{
	local mode="$1"
	local multiplier="$2"
	local windows="$3"
	local resolution="$4"
	local refresh="$5"

	case "$mode" in
		motion|extrapolate|blend) ;;
		*) framegen_die "mode must be motion, extrapolate, or blend"; return ;;
	esac

	case "$multiplier" in
		2|3|4) ;;
		*) framegen_die "multiplier must be 2, 3, or 4"; return ;;
	esac

	if [[ ! "$windows" =~ ^[1-9][0-9]*$ ]]; then
		framegen_die "windows must be a positive integer, got '$windows'"
		return
	fi

	if [[ ! "$resolution" =~ ^([1-9][0-9]*)x([1-9][0-9]*)$ ]]; then
		framegen_die "resolution must be WIDTHxHEIGHT, got '$resolution'"
		return
	fi
	FRAMEGEN_WIDTH="${BASH_REMATCH[1]}"
	FRAMEGEN_HEIGHT="${BASH_REMATCH[2]}"

	if [[ ! "$refresh" =~ ^[1-9][0-9]*$ ]]; then
		framegen_die "refresh must be a positive integer, got '$refresh'"
		return
	fi
}

framegen_resolve_gamescope_binary()
{
	FRAMEGEN_GAMESCOPE_BIN="${GAMESCOPE_BIN:-$GAMESCOPE_REPO/$GAMESCOPE_BUILD_DIR/src/gamescope}"
	if [[ ! -x "$FRAMEGEN_GAMESCOPE_BIN" ]]; then
		framegen_die \
			"gamescope binary not found at '$FRAMEGEN_GAMESCOPE_BIN'; set GAMESCOPE_BUILD_DIR or GAMESCOPE_BIN"
		return
	fi
}

framegen_build_app_command()
{
	local resolution="$1"
	local windows="$2"
	shift 2

	if (( $# > 0 )); then
		FRAMEGEN_APP_CMD=( "$@" )
	elif [[ -n "${APP:-}" ]]; then
		# Compatibility with the original launcher. For exact quoting, pass the
		# client command after `--` instead of putting shell words in APP.
		read -r -a FRAMEGEN_APP_CMD <<< "$APP"
	else
		FRAMEGEN_APP_CMD=(
			vkmark --size "$resolution"
			-b "desktop:windows=${windows}:duration=600"
		)
	fi

	if (( ${#FRAMEGEN_APP_CMD[@]} == 0 )); then
		framegen_die "client command is empty"
		return
	fi

	if ! command -v "${FRAMEGEN_APP_CMD[0]}" >/dev/null 2>&1 \
		&& [[ ! -x "${FRAMEGEN_APP_CMD[0]}" ]]; then
		framegen_die "client executable '${FRAMEGEN_APP_CMD[0]}' was not found"
		return
	fi
}

framegen_format_command()
{
	printf -v FRAMEGEN_APP_DISPLAY '%q ' "${FRAMEGEN_APP_CMD[@]}"
	FRAMEGEN_APP_DISPLAY="${FRAMEGEN_APP_DISPLAY% }"
}
