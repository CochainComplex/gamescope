#!/usr/bin/env bash
# Load the local runtime used by this gamescope checkout.
#
# Source this before running gamescope manually:
#   source ./env-gamescope-local.sh
#
# Or run one command inside the environment:
#   ./env-gamescope-local.sh gamescope --help
#
# Overrides:
#   GAMESCOPE_LOCAL_PREFIX=/path/to/local/prefix
#   GAMESCOPE_BUILD_DIR=build-perf

_gamescope_env_prepend()
{
	local var="$1"
	local dir="$2"

	[ -n "$dir" ] || return 0
	[ -d "$dir" ] || return 0

	local cur="${!var:-}"
	case ":$cur:" in
		*":$dir:"*) ;;
		*) export "$var=$dir${cur:+:$cur}" ;;
	esac
}

_gamescope_env_script="${BASH_SOURCE[0]}"
_gamescope_env_dir="$(cd "$(dirname "$_gamescope_env_script")" && pwd)"

export GAMESCOPE_REPO="${GAMESCOPE_REPO:-$_gamescope_env_dir}"
export GAMESCOPE_LOCAL_PREFIX="${GAMESCOPE_LOCAL_PREFIX:-$HOME/.local/gamescope-wayland}"
export GAMESCOPE_BUILD_DIR="${GAMESCOPE_BUILD_DIR:-build}"

_gamescope_env_build="$GAMESCOPE_REPO/$GAMESCOPE_BUILD_DIR"

_gamescope_env_prepend LD_LIBRARY_PATH "$_gamescope_env_build/layer"
_gamescope_env_prepend LD_LIBRARY_PATH "$GAMESCOPE_LOCAL_PREFIX/lib"
_gamescope_env_prepend LD_LIBRARY_PATH "$GAMESCOPE_LOCAL_PREFIX/lib/x86_64-linux-gnu"

_gamescope_env_prepend PATH "$_gamescope_env_build/src"
_gamescope_env_prepend PATH "$GAMESCOPE_LOCAL_PREFIX/bin"
_gamescope_env_prepend PKG_CONFIG_PATH "$GAMESCOPE_LOCAL_PREFIX/lib/pkgconfig"
_gamescope_env_prepend PKG_CONFIG_PATH "$GAMESCOPE_LOCAL_PREFIX/lib/x86_64-linux-gnu/pkgconfig"
_gamescope_env_prepend CMAKE_PREFIX_PATH "$GAMESCOPE_LOCAL_PREFIX"
export GAMESCOPE_SCRIPT_PATH="$GAMESCOPE_REPO/scripts"

# Let ad-hoc Vulkan clients find the local gamescope WSI layer JSON as well.
_gamescope_env_prepend VK_LAYER_PATH "$_gamescope_env_build/layer"

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
	if [[ $# -gt 0 ]]; then
		exec "$@"
	fi

	printf '%s\n' "Loaded gamescope local environment for this process only."
	printf '%s\n' "To keep it in your shell, run:"
	printf '%s\n' "  source ./env-gamescope-local.sh"
	printf '%s\n' ""
	printf '%s\n' "Current values:"
	printf '  GAMESCOPE_LOCAL_PREFIX=%s\n' "$GAMESCOPE_LOCAL_PREFIX"
	printf '  GAMESCOPE_BUILD_DIR=%s\n' "$GAMESCOPE_BUILD_DIR"
	printf '  GAMESCOPE_SCRIPT_PATH=%s\n' "$GAMESCOPE_SCRIPT_PATH"
fi

unset _gamescope_env_script _gamescope_env_dir _gamescope_env_build
