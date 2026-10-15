#!/usr/bin/env bash
# Hybrid-laptop launcher: gamescope composites, generates and presents on the
# GPU that drives the display; only games are steered to the other GPU.
set -euo pipefail

usage()
{
	cat <<'EOF'
Usage:
  gamescope-hybrid.sh [--report] [gamescope options] -- command [args...]
  gamescope-hybrid.sh [--report] --steer-child [gamescope options] -- game [args...]
  gamescope-hybrid.sh [--report] --child [--] game [args...]

Host mode (the default) starts gamescope pinned to the display GPU
(--prefer-vk-device), with its child unsteered. Use it for Steam or a gaming
session, and steer each game with its own Steam launch option:
  /absolute/path/to/gamescope-hybrid.sh --child -- %command%
Other launchers take the same prefix (Lutris "Command prefix", Heroic
"Wrapper").
--steer-child also steers gamescope's direct child to the game GPU: use it to
wrap one game in its own gamescope, never the Steam client (all of Steam would
then render on the dGPU).
--child starts no gamescope and only steers the game; it reads no gamescope
options, and words before it are ignored with one notice. Write it as
--child -- %command%: a -- right after it is dropped and the rest runs
verbatim, unscanned (a Steam %command% keeps its own --). Without that --, the
words after --child are skipped only until one names a command (on PATH, or an
executable path), so an option value that happens to name one (cvt, say) runs
instead of the game; the notice names the word that runs. If no word names a
command, the words run as given. Without framegen-launch-common.sh, --child
runs the game unsteered.
--report prints the detection and the command, and runs nothing.
--help prints this. Both go before --child. Every other option, -h included,
goes to gamescope.

The display GPU is the owner of the -O connector when it is connected, else of
a connected eDP/LVDS/DSI panel (DRM backend: not while the lid is closed), else
of any connected connector, integrated GPU first. Nested, it is the host
compositor's GPU (wayland-info, KWIN/AQ/WLR_DRM_DEVICES, xrandr), else the
panel owner. PRESENT_DEV and RENDER_DEV (vendor:device) override detection;
two discrete GPUs need both.

Only gamescope's own processes (gamescope, Xwayland, gamescopereaper) are
filtered: their environment never carries steering variables and, with a
split, gets VK_LOADER_DRIVERS_DISABLE with the game GPU's Vulkan driver
manifests (unless the display GPU uses the same driver) and, for an NVIDIA game
GPU, __EGL_VENDOR_LIBRARY_FILENAMES (the non-NVIDIA glvnd EGL vendors).
Gamescope's child gets the caller's environment back exactly (--steer-child
adds the steering variables), so Steam itself may still open the dGPU.
Variables the caller already set win, with a warning where they defeat this.
Measured with an NVIDIA game GPU: once started, gamescope's processes hold no
NVIDIA device node or library open; at startup Xwayland's Mesa EGL probe still
opens the dGPU's DRM nodes briefly. An AMD, Intel or nouveau game GPU may be
opened by the Mesa drivers gamescope loads.

Fail-open: when no split is possible (one GPU, the display on the dGPU, two
GPUs with the same vendor:device, two discrete GPUs without explicit IDs, an
unusable PRESENT_DEV/RENDER_DEV), the gamescope modes run gamescope on the
display GPU with the child unsteered, and --child runs the game unchanged,
except with the display on the dGPU, where it keeps the game on that GPU.
Fail-open sets no driver filters: gamescope may load the other GPU's driver.
The reason goes to stderr.

Exit codes: 2 usage, 127 gamescope (or the launcher's library) not found,
else the command's own. --report exits 0 with a split, 3 without one.

Detection reads sysfs and, nested, asks the host compositor (wayland-info,
xrandr --current); it never runs nvidia-smi, vulkaninfo or glxinfo, which
would wake a suspended dGPU. Gamescope is GAMESCOPE_BIN, else build/src/gamescope
of the checkout this script is in, else `gamescope` on PATH.
EOF
}

# Resolved through symlinks, so a link on PATH (or in a Steam launch option)
# still finds the library next to the real script.
SCRIPT_PATH=$(readlink -f -- "${BASH_SOURCE[0]}")
LIBRARY=${SCRIPT_PATH%/*}/framegen-launch-common.sh
# Without it, main still parses the arguments: --child runs the game unchanged.
HAVE_LIBRARY=0
if [[ -r "$LIBRARY" ]]; then
	# shellcheck source=scripts/framegen-launch-common.sh
	source "$LIBRARY"
	HAVE_LIBRARY=1
fi

REPORT=0
HELP=0
MODE=host
BACKEND=auto
OUTPUTS=""
GAMESCOPE=""
REAPER_DIR=""
USAGE_ERROR=""
STEER_INDEX=-1
GAMESCOPE_ARGS=()
GAME=()
SKIPPED=()
RESTORED=()

say()
{
	printf 'gamescope-hybrid: %s\n' "$*" >&2
}

# Keeps the first usage problem; main decides what to do with it.
usage_error()
{
	[[ -n "$USAGE_ERROR" ]] || USAGE_ERROR=$1
}

# Every gamescope option takes at most one value, so a word that is neither an
# option nor an option's value means the -- before the command was forgotten.
# The scan goes on after that error, since a later --child drops it: --child
# takes over the rest, and the words before it, except --report, are ignored.
parse_args()
{
	local -a argv=( "$@" )
	local -a before=()
	local i arg value_ok=0
	for (( i = 0; i < ${#argv[@]}; i++ )); do
		arg=${argv[i]}
		case "$arg" in
			--report) REPORT=1 value_ok=0; continue ;;
			# Ignored after a usage error, as when the scan stopped there.
			--help) [[ -n "$USAGE_ERROR" ]] || HELP=1; value_ok=0 ;;
			--)
				GAME=( "${argv[@]:i+1}" )
				return ;;
			--child)
				MODE=child SKIPPED=( "${before[@]}" ) GAMESCOPE_ARGS=() USAGE_ERROR=""
				parse_child "${argv[@]:i+1}"
				return ;;
			--steer-child) MODE=steer-child value_ok=0 ;;
			-*=*) GAMESCOPE_ARGS+=( "$arg" ); value_ok=0 ;;
			-*) GAMESCOPE_ARGS+=( "$arg" ); value_ok=1 ;;
			*)
				(( value_ok )) || usage_error "missing -- before '$arg'"
				GAMESCOPE_ARGS+=( "$arg" ); value_ok=0 ;;
		esac
		before+=( "$arg" )
	done
}

# --child's game: after an immediate --, the rest; else from the first word
# that names a command exec can run, with the words before it ignored. When
# none does, the words as given.
parse_child()
{
	local i
	if [[ "${1:-}" == -- ]]; then
		GAME=( "${@:2}" )
		return
	fi
	for (( i = 1; i <= $#; i++ )); do
		if [[ "${!i}" != -* ]] && type -P -- "${!i}" >/dev/null; then
			SKIPPED+=( "${@:1:i-1}" )
			GAME=( "${@:i}" )
			return
		fi
	done
	GAME=( "$@" )
}

# Does ARG name gamescope's long option NAME, also as one of getopt_long's
# abbreviations of at least MIN letters?
is_long_opt()
{
	local name=$1 min=$2 arg=${3%%=*}
	[[ "$arg" == --* ]] || return 1
	arg=${arg#--}
	(( ${#arg} >= min )) && [[ "$name" == "$arg"* ]]
}

# Reads --backend and -O from the gamescope options; the launcher owns
# --prefer-vk-device.
scan_gamescope_args()
{
	local -a args=( "${GAMESCOPE_ARGS[@]}" )
	local i arg value
	for (( i = 0; i < ${#args[@]}; i++ )); do
		arg=${args[i]}
		if [[ "$arg" == --*=* ]]; then
			value=${arg#*=}
		else
			value=${args[i + 1]:-}
		fi
		if is_long_opt prefer-vk-device 8 "$arg"; then
			usage_error "do not pass --prefer-vk-device; set PRESENT_DEV instead"
		elif is_long_opt backend 2 "$arg"; then
			BACKEND=${value:-auto}
		elif is_long_opt prefer-output 8 "$arg" || [[ "$arg" == -O ]]; then
			OUTPUTS=$value
		elif [[ "$arg" == -O?* ]]; then
			OUTPUTS=${arg#-O}
			continue
		else
			continue
		fi
		[[ "$arg" == *=* ]] || (( ++i ))
	done
}

check_usage()
{
	# With a game, --child ignores --help (in SKIPPED) and runs it.
	if (( HELP )) && [[ "$MODE" != child || ${#GAME[@]} == 0 ]]; then
		usage
		exit 0
	fi
	if (( !REPORT && ${#GAME[@]} == 0 )); then
		usage_error "missing -- command [args...]"
	fi
}

# GAMESCOPE_BIN, else this checkout's build, else PATH; fails when not found.
# Gamescope starts its child through gamescopereaper on PATH: the directory of
# the reaper next to gamescope, else next to its symlink target, goes first on
# gamescope's PATH unless PATH already finds that reaper (build_launch gives the
# child the caller's PATH back).
resolve_gamescope()
{
	local name=${GAMESCOPE_BIN:-gamescope} build=${SCRIPT_PATH%/*/*}/build/src/gamescope real dir
	[[ -n "${GAMESCOPE_BIN:-}" || ! -f "$build" || ! -x "$build" ]] || name=$build
	GAMESCOPE=$name
	if [[ "$name" != */* ]]; then
		GAMESCOPE=$(type -P -- "$name") || { GAMESCOPE=$name; return 1; }
	fi
	[[ -f "$GAMESCOPE" && -x "$GAMESCOPE" ]] || return 1
	[[ "$GAMESCOPE" == /* ]] || GAMESCOPE=$PWD/$GAMESCOPE
	real=$(readlink -f -- "$GAMESCOPE")
	for dir in "${GAMESCOPE%/*}" "${real%/*}"; do
		[[ -x "$dir/gamescopereaper" ]] || continue
		[[ "$(type -P gamescopereaper)" == "$dir/gamescopereaper" ]] || REAPER_DIR=$dir
		break
	done
	return 0
}

# Colon-separated glvnd EGL vendor files that are not NVIDIA's, in glvnd's
# search order (FRAMEGEN_EGL_VENDOR_DIRS replaces the dirs, for tests).
egl_vendors()
{
	local dirs=${FRAMEGEN_EGL_VENDOR_DIRS:-/etc/glvnd/egl_vendor.d:/usr/share/glvnd/egl_vendor.d}
	local dir file content list=""
	local -a dir_list=()
	IFS=: read -r -a dir_list <<< "$dirs"
	for dir in "${dir_list[@]}"; do
		for file in "$dir"/*.json; do
			[[ -f "$file" && -r "$file" ]] || continue
			content=$(<"$file")
			[[ "${content,,}" != *nvidia* ]] || continue
			list+="${list:+:}$file"
		done
	done
	[[ -n "$list" ]] && printf '%s' "$list"
}

# Vulkan driver manifest stems (space-separated) of the drivers that can drive
# GPU index I: nvidia_icd.json; radeon_icd.*.json (RADV) and amd_icd64.json,
# amd_pro_icd64.json (AMDVLK, AMD's proprietary driver); intel_icd.*.json and
# intel_hasvk_icd.*.json (ANV); nouveau_icd.*.json (NVK).
vk_stems()
{
	case "${FRAMEGEN_HYBRID_GPU_DRIVER[$1]}" in
		nvidia) printf nvidia ;;
		amdgpu) printf 'radeon amd' ;;
		i915|xe) printf intel ;;
		nouveau) printf nouveau ;;
		*) return 1 ;;
	esac
}

# VK_LOADER_DRIVERS_DISABLE globs that keep the loader off the game GPU: its
# driver's manifests, except the display GPU's own. NVK also probes a GPU bound
# to nvidia, so nvidia adds nouveau. Fails when nothing can be filtered (both
# GPUs on one driver: RADV and ANV open every GPU they drive).
loader_filter()
{
	local game display stem list=""
	local -a stems=()
	game=$(vk_stems "$FRAMEGEN_HYBRID_RENDER") || return 1
	display=$(vk_stems "$FRAMEGEN_HYBRID_PRESENT") || display=""
	read -r -a stems <<< "$game"
	[[ "$game" != nvidia ]] || stems+=( nouveau )
	for stem in "${stems[@]}"; do
		[[ " $display " == *" $stem "* ]] || list+="${list:+,}*$stem*"
	done
	[[ -n "$list" ]] && printf '%s' "$list"
}

# Does a VK_LOADER_DRIVERS_* glob list (comma-separated) match a manifest of STEM?
globs_match()
{
	local glob stem=$2
	local -a globs=()
	IFS=, read -r -a globs <<< "$1"
	for glob in "${globs[@]}"; do
		# shellcheck disable=SC2053 # the right-hand side is the caller's glob
		[[ "${stem}_icd.json" == $glob || "${stem}_icd.x86_64.json" == $glob \
			|| "${stem}_icd64.json" == $glob || "${stem}_pro_icd64.json" == $glob ]] && return 0
	done
	return 1
}

# Does a VK_ICD_FILENAMES/VK_DRIVER_FILES list name a manifest of one of STEMS
# (space-separated)? A directory entry may hold one, so it counts.
files_match()
{
	local entry stem
	local -a entries=() stems=()
	IFS=: read -r -a entries <<< "$1"
	read -r -a stems <<< "$2"
	for entry in "${entries[@]}"; do
		[[ "$entry" == *.json ]] || return 0
		for stem in "${stems[@]}"; do
			[[ "${entry##*/}" != *"$stem"* ]] || return 0
		done
	done
	return 1
}

# Warns when caller-set Vulkan driver variables hide GPU index I's driver from
# WHO (the game, or gamescope).
check_vk_drivers()
{
	local i=$1 who=$2 stems names key stem selected=0 disabled=0
	local gpu=${FRAMEGEN_HYBRID_GPU_ID[$1]}
	local -a list=()
	stems=$(vk_stems "$i") || return 0
	names=${stems// //}
	read -r -a list <<< "$stems"
	for key in VK_ICD_FILENAMES VK_DRIVER_FILES; do
		[[ -n "${!key:-}" ]] && ! files_match "${!key}" "$stems" \
			&& FRAMEGEN_HYBRID_WARNINGS+=( "caller-set $key names no $names driver manifest: $who cannot use $gpu" )
	done
	for stem in "${list[@]}"; do
		if globs_match "${VK_LOADER_DRIVERS_SELECT:-}" "$stem"; then selected=1; fi
		if globs_match "${VK_LOADER_DRIVERS_DISABLE:-}" "$stem"; then (( ++disabled )); fi
	done
	[[ -n "${VK_LOADER_DRIVERS_SELECT:-}" ]] && (( !selected )) \
		&& FRAMEGEN_HYBRID_WARNINGS+=( "caller-set VK_LOADER_DRIVERS_SELECT=$VK_LOADER_DRIVERS_SELECT skips the $names driver: $who cannot use $gpu" )
	(( disabled == ${#list[@]} )) \
		&& FRAMEGEN_HYBRID_WARNINGS+=( "caller-set VK_LOADER_DRIVERS_DISABLE=$VK_LOADER_DRIVERS_DISABLE disables the $names driver: $who cannot use $gpu" )
	return 0
}

# Steering-mode checks of the game's own GPU choice.
check_game_env()
{
	local key
	check_vk_drivers "$STEER_INDEX" "the game"
	for key in DXVK_FILTER_DEVICE_NAME VKD3D_FILTER_DEVICE_NAME; do
		[[ -z "${!key:-}" ]] \
			|| FRAMEGEN_HYBRID_WARNINGS+=( "caller-set $key=${!key} picks the game's GPU by name; the game follows it, not this split" )
	done
}

# Keeps the game GPU's user-space drivers out of gamescope's own processes
# (gamescope, Xwayland, gamescopereaper): a Vulkan driver or NVIDIA's EGL vendor
# library opens the dGPU, which wakes it from RTD3. Adds to GAMESCOPE_ENV; a
# caller-set VK_LOADER_DRIVERS_DISABLE gets the missing globs appended.
filter_game_gpu()
{
	local globs glob value=${VK_LOADER_DRIVERS_DISABLE:-} missing="" vendors
	local -a list=()
	if globs=$(loader_filter); then
		IFS=, read -r -a list <<< "$globs"
		for glob in "${list[@]}"; do
			[[ ",$value," == *",$glob,"* ]] || missing+=",$glob"
		done
		value+=$missing
		[[ -z "$missing" ]] || GAMESCOPE_ENV+=( "VK_LOADER_DRIVERS_DISABLE=${value#,}" )
	fi
	[[ "${FRAMEGEN_HYBRID_GPU_DRIVER[FRAMEGEN_HYBRID_RENDER]}" == nvidia \
		&& "${FRAMEGEN_HYBRID_GPU_DRIVER[FRAMEGEN_HYBRID_PRESENT]}" != nvidia ]] || return 0
	for glob in __EGL_VENDOR_LIBRARY_FILENAMES __EGL_VENDOR_LIBRARY_DIRS; do
		if [[ -n "${!glob+set}" ]]; then
			FRAMEGEN_HYBRID_WARNINGS+=( "caller-set $glob skips the EGL filter: Xwayland may load NVIDIA's EGL and wake the dGPU" )
			return
		fi
	done
	if vendors=$(egl_vendors); then
		GAMESCOPE_ENV+=( "__EGL_VENDOR_LIBRARY_FILENAMES=$vendors" )
	else
		FRAMEGEN_HYBRID_WARNINGS+=( "no non-NVIDIA glvnd EGL vendor found: Xwayland may load NVIDIA's EGL and wake the dGPU" )
	fi
}

# The Steam client, by its launcher name, a Flatpak run of it, or gamescope's
# own --steam (-e).
is_steam_client()
{
	local word
	case "${GAME[0]:-}" in
		steam|*/steam|steam.sh|*/steam.sh|steam-runtime|*/steam-runtime) return 0 ;;
	esac
	for word in "${GAME[@]}"; do
		[[ "$word" != com.valvesoftware.Steam ]] || return 0
	done
	for word in "${GAMESCOPE_ARGS[@]}"; do
		[[ "$word" != -e && "$word" != --steam ]] || return 0
	done
	return 1
}

# Fills GAMESCOPE_ENV (set for gamescope's own processes only), CHILD_ENV (the
# steering added for the game) and LAUNCH. Gamescope's child runs with the
# caller's environment: every key gamescope's differs in is put back or unset.
build_launch()
{
	local split=$1 key entry
	local -a game=( "${GAME[@]}" ) keys=() restore=() wrap=() child=()
	(( ${#game[@]} > 0 )) || game=( '<command>' )
	GAMESCOPE_ENV=() CHILD_ENV=()
	if (( STEER )); then
		CHILD_ENV=( "${FRAMEGEN_HYBRID_CHILD_ENV[@]}" )
		check_game_env
		if (( split )) && is_steam_client; then
			FRAMEGEN_HYBRID_WARNINGS+=( "--$MODE wraps the Steam client: all of Steam renders on the game GPU and keeps it awake; run Steam in host mode and steer each game with --child" )
		fi
	fi
	if [[ "$MODE" == child ]]; then
		LAUNCH=( "${game[@]}" )
		return
	fi
	if (( split )); then
		# Pre-emptive upscale off removes the cross-device OPAQUE_FD wait (design §3.2).
		[[ -n "${gamescope_upscale_preemptive+set}" ]] \
			|| GAMESCOPE_ENV+=( gamescope_upscale_preemptive=false )
		filter_game_gpu
		check_vk_drivers "$FRAMEGEN_HYBRID_PRESENT" gamescope
	fi
	for entry in "${GAMESCOPE_ENV[@]}"; do
		keys+=( "${entry%%=*}" )
	done
	# The reaper's directory is on gamescope's PATH only.
	[[ -z "$REAPER_DIR" ]] || keys+=( PATH )
	# Gamescope's environment drops these; a steered child has them in CHILD_ENV.
	for key in "${FRAMEGEN_HYBRID_STEER_KEYS[@]}"; do
		[[ "$STEER" == 0 && -n "${!key+set}" ]] || continue
		keys+=( "$key" )
	done
	for key in "${keys[@]}"; do
		if [[ -n "${!key+set}" ]]; then
			restore+=( "$key=${!key}" )
			RESTORED+=( "$key" )
		else
			wrap+=( -u "$key" )
		fi
	done
	CHILD_ENV=( "${restore[@]}" "${CHILD_ENV[@]}" )
	child=( "${game[@]}" )
	if (( ${#wrap[@]} + ${#CHILD_ENV[@]} > 0 )); then
		# env takes an argv[0] containing = for one more assignment.
		[[ "${game[0]}" != *=* ]] || child=( sh -c 'exec "$@"' sh "${game[@]}" )
		child=( env "${wrap[@]}" -- "${CHILD_ENV[@]}" "${child[@]}" )
	fi
	LAUNCH=( "$GAMESCOPE" )
	[[ -z "$FRAMEGEN_HYBRID_PIN" ]] || LAUNCH+=( --prefer-vk-device "$FRAMEGEN_HYBRID_PIN" )
	LAUNCH+=( "${GAMESCOPE_ARGS[@]}" -- "${child[@]}" )
}

quote_all()
{
	(( $# == 0 )) || printf ' %q' "$@"
}

# LAUNCH, quoted; a key restored to the caller's value prints as KEY="$KEY",
# which the caller's shell expands to the same word (a whole PATH stays out).
quote_launch()
{
	local word key
	for word in "${LAUNCH[@]}"; do
		key=${word%%=*}
		if [[ "$word" == ?*=* && " ${RESTORED[*]} " == *" $key "* && "$word" == "$key=${!key}" ]]; then
			printf ' %s="$%s"' "$key" "$key"
		else
			printf ' %q' "$word"
		fi
	done
}

# One line: what this mode does with this detection (the report's and
# stderr's only fallback text).
describe()
{
	local gpu=${FRAMEGEN_HYBRID_PIN:-the default GPU} game
	if [[ -n "$FRAMEGEN_HYBRID_REFUSAL" ]]; then
		case "$MODE" in
			child)
				if (( STEER )); then
					printf 'no split: %s; game kept on the display GPU %s' "$FRAMEGEN_HYBRID_REFUSAL" "$gpu"
				else
					printf 'not steering: %s; running the game unchanged' "$FRAMEGEN_HYBRID_REFUSAL"
				fi ;;
			*) printf 'no split: %s; gamescope on %s, child unsteered' "$FRAMEGEN_HYBRID_REFUSAL" "$gpu" ;;
		esac
		return
	fi
	game="${RENDER_DEV%!} (${FRAMEGEN_HYBRID_GPU_DRIVER[FRAMEGEN_HYBRID_RENDER]})"
	case "$MODE" in
		child) printf 'game on %s; display GPU %s' "$game" "$PRESENT_DEV" ;;
		steer-child) printf 'gamescope on %s, game on %s' "$PRESENT_DEV" "$game" ;;
		host) printf 'gamescope on %s, child unsteered (steer each game with --child; game GPU %s)' \
			"$PRESENT_DEV" "$game" ;;
	esac
}

# Who the report's steering env is for.
steer_label()
{
	case "$MODE" in
		child) printf 'the game' ;;
		steer-child) printf "gamescope's child" ;;
		host) printf 'games run with --child; not the child of this gamescope' ;;
	esac
}

main()
{
	local rc=0 found=1 warning entry connector
	parse_args "$@"
	check_usage
	scan_gamescope_args
	if (( ${#SKIPPED[@]} > 0 )); then
		say "--child takes no gamescope options; ignored:$(quote_all "${SKIPPED[@]}")${GAME[0]+; running:$(quote_all "${GAME[0]}")}"
	fi
	if (( !HAVE_LIBRARY )); then
		if [[ "$MODE" == child ]] && (( !REPORT && ${#GAME[@]} > 0 )); then
			say "$LIBRARY is missing; running the game unchanged"
			exec -- "${GAME[@]}"
		fi
		say "$LIBRARY is missing"
		exit 127
	fi
	if [[ -n "$USAGE_ERROR" ]]; then
		say "$USAGE_ERROR (see --help)"
		exit 2
	fi
	if [[ "$MODE" != child ]]; then
		resolve_gamescope || found=0
		if (( !found && !REPORT )); then
			say "gamescope not found: '$GAMESCOPE' (set GAMESCOPE_BIN to its path; a local build is build/src/gamescope)"
			exit 127
		fi
	fi

	framegen_detect_hybrid_devices "$BACKEND" "$OUTPUTS" || rc=$?
	# Lid closed: gamescope must drive the external connector chosen here, not
	# the panel it would prefer.
	connector=${FRAMEGEN_HYBRID_SOURCE#connector }
	if [[ "$MODE" != child && -z "$OUTPUTS" && "$FRAMEGEN_HYBRID_SOURCE" == "connector $connector" ]] \
		&& (( FRAMEGEN_HYBRID_LID_CLOSED )) && ! _framegen_is_panel "$connector"; then
		GAMESCOPE_ARGS+=( -O "$connector" )
	fi
	STEER=0 STEER_INDEX=$FRAMEGEN_HYBRID_RENDER
	if (( rc == 0 )) && [[ "$MODE" != host ]]; then STEER=1; fi
	if [[ "$MODE" == child ]] && (( FRAMEGEN_HYBRID_REVERSE )); then
		# Same GPU as the display: no cross-device copy for the game either.
		framegen_hybrid_steer_env "$FRAMEGEN_HYBRID_PRESENT"
		STEER=1 STEER_INDEX=$FRAMEGEN_HYBRID_PRESENT
	fi
	build_launch "$(( rc == 0 ))"

	if (( REPORT )); then
		framegen_hybrid_report "$(steer_label)"
		printf 'mode %s: %s\n' "$MODE" "$(describe)"
		[[ "$MODE" == child ]] || printf 'gamescope: %s%s\n' "$GAMESCOPE" "$( (( found )) || printf ' (not found)')"
		printf 'command:'
		if [[ "$MODE" == child ]]; then
			quote_all "${CHILD_ENV[@]}"
		else
			[[ -z "$REAPER_DIR" ]] || printf " PATH=%q:\$PATH" "$REAPER_DIR"
			quote_all "${GAMESCOPE_ENV[@]}"
		fi
		quote_launch
		printf '\n'
		exit "$rc"
	fi

	say "$(describe)"
	for warning in "${FRAMEGEN_HYBRID_WARNINGS[@]}"; do
		say "warning: $warning"
	done
	if [[ "$MODE" == child ]]; then
		for entry in "${CHILD_ENV[@]}"; do
			export "${entry?}"
		done
		exec -- "${LAUNCH[@]}"
	fi
	say "running $GAMESCOPE"
	for entry in "${FRAMEGEN_HYBRID_STEER_KEYS[@]}"; do
		unset "$entry"
	done
	for entry in "${GAMESCOPE_ENV[@]}"; do
		export "${entry?}"
	done
	[[ -z "$REAPER_DIR" ]] || PATH=$REAPER_DIR${PATH:+:$PATH}
	exec "${LAUNCH[@]}"
}

main "$@"
