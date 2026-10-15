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

FRAMEGEN_DEVICE_ID_RE='^[[:xdigit:]]{4}:[[:xdigit:]]{4}$'

# Checks the variable NAME: RENDER_DEV is vendor:device with one optional
# trailing !, PRESENT_DEV is vendor:device. On failure sets
# FRAMEGEN_DEVICE_ID_ERROR and returns 1.
framegen_check_device_id()
{
	local name=$1 value=${!1:-}
	FRAMEGEN_DEVICE_ID_ERROR=""
	if [[ "$name" == RENDER_DEV ]]; then
		[[ "${value%!}" =~ $FRAMEGEN_DEVICE_ID_RE ]] && return 0
		FRAMEGEN_DEVICE_ID_ERROR="RENDER_DEV must be vendor:device or vendor:device!, got '$value'"
	else
		[[ "$value" =~ $FRAMEGEN_DEVICE_ID_RE ]] && return 0
		FRAMEGEN_DEVICE_ID_ERROR="$name must be vendor:device without a trailing !, got '$value'"
	fi
	return 1
}

framegen_require_device_ids()
{
	local name
	if [[ -z "${RENDER_DEV:-}" || -z "${PRESENT_DEV:-}" ]]; then
		framegen_die \
			"set RENDER_DEV and PRESENT_DEV to PCI vendor:device IDs (run ./scripts/test-framegen.sh gpus)"
		return
	fi
	for name in RENDER_DEV PRESENT_DEV; do
		if ! framegen_check_device_id "$name"; then
			framegen_die "$FRAMEGEN_DEVICE_ID_ERROR"
			return
		fi
	done
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

# ---------------------------------------------------------------------------
# Hybrid (iGPU + dGPU) topology, for scripts/gamescope-hybrid.sh and the doctor.
#
# Rule: gamescope composites, generates and presents on the GPU that drives the
# display (DRM) or the host compositor (nested); only the game is steered to
# the other GPU. Detection reads sysfs under ${SYSFS_ROOT:-/sys} (and the lid
# state under ${PROCFS_ROOT:-/proc}) and nothing else that touches a GPU: never
# nvidia-smi, vulkaninfo or glxinfo, which wake a runtime-suspended (RTD3)
# dGPU. boot_vga is not used: APUs often lack it.

FRAMEGEN_HYBRID_STEER_KEYS=(
	MESA_VK_DEVICE_SELECT DRI_PRIME
	__NV_PRIME_RENDER_OFFLOAD __VK_LAYER_NV_optimus __GLX_VENDOR_LIBRARY_NAME
)
FRAMEGEN_PCI_SLOT_RE='^[[:xdigit:]]{4}:[[:xdigit:]]{2}:[[:xdigit:]]{2}\.[0-7]$'

# First line of a sysfs attribute; fails when it is missing or unreadable.
_framegen_read_attr()
{
	local value=""
	[[ -r "$1" ]] || return 1
	IFS= read -r value <"$1" || [[ -n "$value" ]] || return 1
	printf '%s' "$value"
}

_framegen_is_panel()
{
	[[ "$1" == eDP-* || "$1" == LVDS-* || "$1" == DSI-* ]]
}

# integrated | discrete, from PCI placement only (vendor-neutral, no wake-up):
# Intel's iGPU is always bus 0 device 2 function 0; an AMD APU's GPU shares its
# PCI device with AMD SoC functions (vendor 0x1022: PSP, USB, audio); NVIDIA
# and every other vendor are discrete.
_framegen_hybrid_kind()
{
	local vendor=$1 slot=$2 dev=$3 sibling
	if [[ "$vendor" == 0x8086 && "$slot" == ????:00:02.0 ]]; then
		printf integrated
		return
	fi
	if [[ "$vendor" == 0x1002 ]]; then
		for sibling in "${dev%/*}/${slot%.*}".[0-7]; do
			if [[ "$(_framegen_read_attr "$sibling/vendor")" == 0x1022 ]]; then
				printf integrated
				return
			fi
		done
	fi
	printf discrete
}

_framegen_hybrid_reset()
{
	FRAMEGEN_HYBRID_GPU_CARD=() FRAMEGEN_HYBRID_GPU_SLOT=() FRAMEGEN_HYBRID_GPU_ID=()
	FRAMEGEN_HYBRID_GPU_CLASS=() FRAMEGEN_HYBRID_GPU_DRIVER=() FRAMEGEN_HYBRID_GPU_KIND=()
	FRAMEGEN_HYBRID_GPU_CONNS=() FRAMEGEN_HYBRID_GPU_RENDER=() FRAMEGEN_HYBRID_GPU_PATH=()
	FRAMEGEN_HYBRID_GPU_PM=() FRAMEGEN_HYBRID_ORDER=()
	FRAMEGEN_HYBRID_HOST="" FRAMEGEN_HYBRID_SOURCE="" FRAMEGEN_HYBRID_HAS_PANEL=0
	FRAMEGEN_HYBRID_LID_CLOSED=0 FRAMEGEN_HYBRID_PRESENT=-1 FRAMEGEN_HYBRID_RENDER=-1
	FRAMEGEN_HYBRID_INDEX=-1 FRAMEGEN_HYBRID_REFUSAL="" FRAMEGEN_HYBRID_REFUSAL_DETAIL=""
	FRAMEGEN_HYBRID_REVERSE=0 FRAMEGEN_HYBRID_NO_DRM_SLOT=""
	FRAMEGEN_HYBRID_PIN="" FRAMEGEN_HYBRID_PCIE="" FRAMEGEN_HYBRID_CHILD_ENV=()
	FRAMEGEN_HYBRID_CALLER_KEYS=() FRAMEGEN_HYBRID_WARNINGS=()
}

# Fills the FRAMEGEN_HYBRID_GPU_* arrays with every PCI display-class DRM card.
# Non-PCI cards (simpledrm, evdi, udl) are skipped.
_framegen_hybrid_enumerate()
{
	local sys=${SYSFS_ROOT:-/sys} card name dev slot vendor device class node conn
	local conns status driver pm
	for card in "$sys"/class/drm/card*; do
		name=${card##*/}
		[[ "$name" =~ ^card[0-9]+$ ]] || continue
		dev=$(readlink -e -- "$card/device") || continue
		slot=${dev##*/}
		[[ "$slot" =~ $FRAMEGEN_PCI_SLOT_RE ]] || continue
		vendor=$(_framegen_read_attr "$dev/vendor") || continue
		device=$(_framegen_read_attr "$dev/device") || continue
		class=$(_framegen_read_attr "$dev/class") || continue
		[[ "$class" == 0x03* ]] || continue

		conns=""
		for status in "$card/$name"-*/status; do
			conn=${status%/status}
			conn=${conn##*/"$name"-}
			if _framegen_is_panel "$conn"; then FRAMEGEN_HYBRID_HAS_PANEL=1; fi
			[[ "$(_framegen_read_attr "$status")" == connected ]] || continue
			conns+="${conns:+ }$conn"
		done
		node=""
		for status in "$dev"/drm/renderD*; do
			[[ -e "$status" ]] && node=${status##*/}
		done
		driver=$(readlink -- "$dev/driver" 2>/dev/null) || driver=none
		pm=$(_framegen_read_attr "$dev/power/runtime_status") || pm=unknown

		FRAMEGEN_HYBRID_GPU_CARD+=( "$name" )
		FRAMEGEN_HYBRID_GPU_SLOT+=( "$slot" )
		FRAMEGEN_HYBRID_GPU_ID+=( "${vendor#0x}:${device#0x}" )
		FRAMEGEN_HYBRID_GPU_CLASS+=( "$class" )
		FRAMEGEN_HYBRID_GPU_DRIVER+=( "${driver##*/}" )
		FRAMEGEN_HYBRID_GPU_KIND+=( "$(_framegen_hybrid_kind "$vendor" "$slot" "$dev")" )
		FRAMEGEN_HYBRID_GPU_CONNS+=( "$conns" )
		FRAMEGEN_HYBRID_GPU_RENDER+=( "${node:--}" )
		FRAMEGEN_HYBRID_GPU_PATH+=( "$dev" )
		FRAMEGEN_HYBRID_GPU_PM+=( "$pm" )
	done
}

# Connector search order: integrated GPUs, then a GPU with a connected panel,
# then PCI order (not card numbers, which follow driver probe order).
_framegen_hybrid_sort()
{
	local i rank
	while read -r _ _ i; do
		FRAMEGEN_HYBRID_ORDER+=( "$i" )
	done < <(
		for i in "${!FRAMEGEN_HYBRID_GPU_ID[@]}"; do
			rank=2
			[[ " ${FRAMEGEN_HYBRID_GPU_CONNS[i]}" =~ \ (eDP|LVDS|DSI)- ]] && rank=1
			[[ "${FRAMEGEN_HYBRID_GPU_KIND[i]}" == integrated ]] && rank=0
			printf '%s %s %s\n' "$rank" "${FRAMEGEN_HYBRID_GPU_SLOT[i]}" "$i"
		done | LC_ALL=C sort
	)
}

_framegen_lid_closed()
{
	local state
	for state in "${PROCFS_ROOT:-/proc}"/acpi/button/lid/*/state; do
		[[ "$(_framegen_read_attr "$state")" == *closed ]] && return 0
	done
	return 1
}

# Warns about a GPU bound to nvidia without a DRM card node (nvidia-drm not
# loaded): it is invisible to the split and to gamescope. The first one's slot
# goes to FRAMEGEN_HYBRID_NO_DRM_SLOT.
_framegen_hybrid_check_nvidia_drm()
{
	local dev slot
	for dev in "${SYSFS_ROOT:-/sys}"/bus/pci/devices/*; do
		[[ "$(_framegen_read_attr "$dev/class")" == 0x03* ]] || continue
		[[ "$(readlink -- "$dev/driver" 2>/dev/null)" == */nvidia ]] || continue
		slot=${dev##*/}
		_framegen_hybrid_index_of_slot "$slot" >/dev/null && continue
		[[ -n "$FRAMEGEN_HYBRID_NO_DRM_SLOT" ]] || FRAMEGEN_HYBRID_NO_DRM_SLOT=$slot
		FRAMEGEN_HYBRID_WARNINGS+=( "NVIDIA GPU $slot is bound to nvidia but has no DRM card node: load nvidia-drm with modeset=1 (kernel parameter nvidia-drm.modeset=1)" )
	done
}

# Prints the GPU index whose vendor:device (trailing ! ignored) matches; fails otherwise.
_framegen_hybrid_index_of_id()
{
	local wanted=${1%!} i
	wanted=${wanted,,}
	for i in "${!FRAMEGEN_HYBRID_GPU_ID[@]}"; do
		if [[ "${FRAMEGEN_HYBRID_GPU_ID[i]}" == "$wanted" ]]; then
			printf '%s' "$i"
			return
		fi
	done
	return 1
}

# Prints the GPU index behind a DRM node name (card1, renderD128).
_framegen_hybrid_index_of_node()
{
	local sys=${SYSFS_ROOT:-/sys} path i
	path=$(readlink -e -- "$sys/class/drm/$1/device") || return 1
	for i in "${!FRAMEGEN_HYBRID_GPU_PATH[@]}"; do
		if [[ "${FRAMEGEN_HYBRID_GPU_PATH[i]}" == "$path" ]]; then
			printf '%s' "$i"
			return
		fi
	done
	return 1
}

_framegen_hybrid_index_of_slot()
{
	local i
	for i in "${!FRAMEGEN_HYBRID_GPU_SLOT[@]}"; do
		if [[ "${FRAMEGEN_HYBRID_GPU_SLOT[i]}" == "${1,,}" ]]; then
			printf '%s' "$i"
			return
		fi
	done
	return 1
}

# Validates the variable NAME (RENDER_DEV or PRESENT_DEV) and sets
# FRAMEGEN_HYBRID_INDEX to its GPU; refuses otherwise.
_framegen_hybrid_resolve_id()
{
	local name=$1
	if ! framegen_check_device_id "$name"; then
		_framegen_hybrid_refuse "$FRAMEGEN_DEVICE_ID_ERROR"
		return
	fi
	if ! FRAMEGEN_HYBRID_INDEX=$(_framegen_hybrid_index_of_id "${!name}"); then
		_framegen_hybrid_refuse "$name=${!name} is not a PCI GPU on this system"
		return
	fi
}

# Does connector CONN match one -O entry? gamescope matches exact names and a
# lone "*"; "panel" is this helper's own eDP/LVDS/DSI class. With the lid
# closed a panel matches only by its exact name.
_framegen_hybrid_conn_matches()
{
	local usable=1
	if (( FRAMEGEN_HYBRID_LID_CLOSED )) && _framegen_is_panel "$1"; then usable=0; fi
	case "$2" in
		"$1") return 0 ;;
		'*') (( usable )) ;;
		panel) (( usable )) && _framegen_is_panel "$1" ;;
		*) return 1 ;;
	esac
}

# The GPU that drives the target connector: the first connected entry of the
# -O list, else a connected panel (eDP/LVDS/DSI), else any connected one, in
# _framegen_hybrid_sort order. ONLY (a GPU index) restricts the search.
_framegen_hybrid_connector_owner()
{
	local outputs=$1 only=${2:--1} entry named i k conn
	local -a entries=() conns=()
	IFS=, read -r -a entries <<< "$outputs"
	named=${#entries[@]}
	entries+=( panel '*' )
	for k in "${!entries[@]}"; do
		entry=${entries[k]}
		[[ -n "$entry" ]] || continue
		for i in "${FRAMEGEN_HYBRID_ORDER[@]}"; do
			(( only < 0 || i == only )) || continue
			read -r -a conns <<< "${FRAMEGEN_HYBRID_GPU_CONNS[i]}"
			for conn in "${conns[@]}"; do
				if _framegen_hybrid_conn_matches "$conn" "$entry"; then
					FRAMEGEN_HYBRID_PRESENT=$i
					FRAMEGEN_HYBRID_SOURCE="connector $conn"
					(( only >= 0 || k >= named )) || _framegen_hybrid_check_output_name "$conn"
					return
				fi
			done
		done
	done
	return 1
}

# Connector names are per card: an -O name connected on several GPUs makes
# the display GPU a guess (the search order's first). Say so.
_framegen_hybrid_check_output_name()
{
	local conn=$1 i ids=""
	for i in "${!FRAMEGEN_HYBRID_GPU_CONNS[@]}"; do
		[[ " ${FRAMEGEN_HYBRID_GPU_CONNS[i]} " == *" $conn "* ]] && ids+="${ids:+, }${FRAMEGEN_HYBRID_GPU_ID[i]}"
	done
	[[ "$ids" == *,* ]] || return 0
	FRAMEGEN_HYBRID_WARNINGS+=( "$conn is connected on several GPUs ($ids); using ${FRAMEGEN_HYBRID_GPU_ID[FRAMEGEN_HYBRID_PRESENT]}, set PRESENT_DEV to pick another" )
}

# Host compositor's GPU from linux-dmabuf main_device (wayland-info prints it
# as a dev_t, e.g. "main device: 0xE280 (...)").
_framegen_hybrid_wayland_host()
{
	local out dev minor node idx
	local re='main device: 0x([[:xdigit:]]+)'
	command -v wayland-info >/dev/null 2>&1 || return 1
	out=$(timeout 5 wayland-info 2>/dev/null) || true
	[[ "$out" =~ $re ]] || return 1
	dev=$(( 16#${BASH_REMATCH[1]} ))
	minor=$(( (dev & 0xff) | ((dev >> 12) & 0xfff00) ))
	if (( minor >= 128 )); then node=renderD$minor; else node=card$minor; fi
	idx=$(_framegen_hybrid_index_of_node "$node") || return 1
	FRAMEGEN_HYBRID_PRESENT=$idx
	FRAMEGEN_HYBRID_SOURCE="wayland-info main_device $node"
}

# A compositor pinned to a GPU list: KWin, Hyprland (aquamarine) and wlroots
# take colon-separated DRM device paths, the first being the primary GPU.
_framegen_hybrid_env_host()
{
	local key first node idx
	for key in KWIN_DRM_DEVICES AQ_DRM_DEVICES WLR_DRM_DEVICES; do
		[[ -n "${!key:-}" ]] || continue
		first=${!key%%:*}
		node=$(readlink -f -- "$first" 2>/dev/null) || node=$first
		idx=$(_framegen_hybrid_index_of_node "${node##*/}") || continue
		FRAMEGEN_HYBRID_PRESENT=$idx
		FRAMEGEN_HYBRID_SOURCE="$key first entry ${node##*/}"
		return
	done
	return 1
}

# X11 host's primary provider (provider 0) from xrandr.
_framegen_hybrid_x11_host()
{
	local out name idx=""
	local re=$'Provider 0:[^\n]*name:([^\n]*)'
	local pci_re='pci:([[:xdigit:]]{4}:[[:xdigit:]]{2}:[[:xdigit:]]{2}\.[0-7])'
	command -v xrandr >/dev/null 2>&1 || return 1
	# --current: no output re-probe by the X server.
	out=$(timeout 5 xrandr --current --listproviders 2>/dev/null) || true
	[[ "$out" =~ $re ]] || return 1
	name=${BASH_REMATCH[1]}
	if [[ "$name" =~ $pci_re ]]; then
		idx=$(_framegen_hybrid_index_of_slot "${BASH_REMATCH[1]}") || return 1
	elif [[ "$name" == NVIDIA-[0-9]* ]]; then
		# NVIDIA-0 is an NVIDIA-primary X screen; NVIDIA-G0 is only an offload sink.
		for idx in "${!FRAMEGEN_HYBRID_GPU_DRIVER[@]}"; do
			[[ "${FRAMEGEN_HYBRID_GPU_DRIVER[idx]}" == nvidia ]] && break
			idx=""
		done
	fi
	[[ -n "$idx" ]] || return 1
	FRAMEGEN_HYBRID_PRESENT=$idx
	FRAMEGEN_HYBRID_SOURCE="xrandr provider 0 '$name'"
}

# No split. FRAMEGEN_HYBRID_PIN keeps the display GPU when it is known (IDs
# are unique by then), so a fallback can still pin gamescope to it.
_framegen_hybrid_refuse()
{
	FRAMEGEN_HYBRID_REFUSAL=$1
	if (( FRAMEGEN_HYBRID_PRESENT >= 0 )); then
		FRAMEGEN_HYBRID_PIN=${FRAMEGEN_HYBRID_GPU_ID[FRAMEGEN_HYBRID_PRESENT]}
	fi
	return 3
}

# drm when gamescope will own KMS (gamescope's own auto rule); otherwise the
# nested host type.
_framegen_hybrid_host_kind()
{
	local backend=${1:-auto}
	case "$backend" in
		drm) printf drm; return ;;
		headless) printf nested; return ;;
	esac
	if [[ -n "${WAYLAND_DISPLAY:-}" ]]; then
		printf wayland
	elif [[ -n "${DISPLAY:-}" ]]; then
		printf x11
	elif [[ "$backend" == auto ]]; then
		printf drm
	else
		printf nested
	fi
}

# With the lid closed, the owner of a connected panel (which then is the only
# kind of connector connected); fails otherwise.
_framegen_hybrid_closed_panel_owner()
{
	local rc=0
	(( FRAMEGEN_HYBRID_LID_CLOSED )) || return 1
	FRAMEGEN_HYBRID_LID_CLOSED=0
	_framegen_hybrid_connector_owner "" || rc=$?
	FRAMEGEN_HYBRID_LID_CLOSED=1
	return "$rc"
}

# EXPLICIT is PRESENT_DEV's GPU index, or -1. Needs FRAMEGEN_HYBRID_HOST.
_framegen_hybrid_pick_present()
{
	local outputs=$1 explicit=$2 confirmed=1 hint=""
	if [[ "$FRAMEGEN_HYBRID_HOST" == drm ]]; then
		# gamescope drives the connector of the GPU it presents on.
		_framegen_hybrid_connector_owner "$outputs" "$explicit" && return
		if (( explicit >= 0 )); then
			_framegen_hybrid_refuse "no connected connector on PRESENT_DEV=$PRESENT_DEV"
		elif _framegen_hybrid_closed_panel_owner; then
			# The fallback gamescope stays on the panel's GPU.
			_framegen_hybrid_refuse "lid closed and no external display connected; name the panel with -O ${FRAMEGEN_HYBRID_SOURCE#connector } to use it"
		else
			_framegen_hybrid_refuse "no connected connector on any PCI GPU"
		fi
		return
	fi
	case "$FRAMEGEN_HYBRID_HOST" in
		wayland) _framegen_hybrid_wayland_host || _framegen_hybrid_env_host \
			|| { confirmed=0; hint="; install wayland-info"; } ;;
		x11) _framegen_hybrid_x11_host || { confirmed=0; hint="; xrandr provider unknown"; } ;;
		*) confirmed=0 ;;
	esac
	if (( confirmed )); then
		if (( explicit >= 0 && explicit != FRAMEGEN_HYBRID_PRESENT )); then
			_framegen_hybrid_refuse "PRESENT_DEV=$PRESENT_DEV does not drive the display ($FRAMEGEN_HYBRID_SOURCE is on ${FRAMEGEN_HYBRID_GPU_ID[FRAMEGEN_HYBRID_PRESENT]})"
		fi
		return
	fi
	if (( explicit >= 0 )); then
		FRAMEGEN_HYBRID_PRESENT=$explicit
		FRAMEGEN_HYBRID_SOURCE="PRESENT_DEV (host GPU not confirmed$hint)"
		return
	fi
	if _framegen_hybrid_connector_owner "$outputs"; then
		FRAMEGEN_HYBRID_SOURCE="assumed $FRAMEGEN_HYBRID_SOURCE (host GPU not confirmed$hint)"
		return
	fi
	_framegen_hybrid_refuse "host GPU unknown and no connected connector on any PCI GPU"
}

_framegen_hybrid_pick_render()
{
	local i idx=-1 candidates=0 discrete=0
	if [[ -n "${RENDER_DEV:-}" ]]; then
		_framegen_hybrid_resolve_id RENDER_DEV || return
		FRAMEGEN_HYBRID_RENDER=$FRAMEGEN_HYBRID_INDEX
		return
	fi
	for i in "${!FRAMEGEN_HYBRID_GPU_ID[@]}"; do
		(( i != FRAMEGEN_HYBRID_PRESENT )) || continue
		(( ++candidates ))
		if [[ "${FRAMEGEN_HYBRID_GPU_KIND[i]}" == discrete ]]; then
			(( ++discrete ))
			idx=$i
		elif (( discrete == 0 )); then
			idx=$i
		fi
	done
	if (( candidates == 0 )) && [[ -n "$FRAMEGEN_HYBRID_NO_DRM_SLOT" ]]; then
		_framegen_hybrid_refuse "NVIDIA GPU $FRAMEGEN_HYBRID_NO_DRM_SLOT has no DRM node: load nvidia-drm with modeset=1"
		return
	fi
	if (( candidates == 0 )); then
		_framegen_hybrid_refuse "only one PCI GPU (${FRAMEGEN_HYBRID_GPU_ID[FRAMEGEN_HYBRID_PRESENT]}): nothing to split (MUX in dGPU-only mode, or no second GPU)"
		return
	fi
	if (( discrete > 1 )); then
		_framegen_hybrid_refuse "several discrete render candidates; set RENDER_DEV=vendor:device!"
		return
	fi
	FRAMEGEN_HYBRID_RENDER=$idx
}

# Refuses the pairs the split must not run on (design §2).
_framegen_hybrid_check_topology()
{
	local p=$FRAMEGEN_HYBRID_PRESENT r=$FRAMEGEN_HYBRID_RENDER explicit=$1
	local pk=${FRAMEGEN_HYBRID_GPU_KIND[FRAMEGEN_HYBRID_PRESENT]}
	local rk=${FRAMEGEN_HYBRID_GPU_KIND[FRAMEGEN_HYBRID_RENDER]}
	if (( p == r )); then
		_framegen_hybrid_refuse "RENDER_DEV and PRESENT_DEV name the same GPU (${FRAMEGEN_HYBRID_GPU_ID[p]})"
		return
	fi
	if [[ "$pk" == discrete && "$rk" == integrated ]]; then
		FRAMEGEN_HYBRID_REVERSE=1
		FRAMEGEN_HYBRID_REFUSAL_DETAIL="the game GPU ${FRAMEGEN_HYBRID_GPU_ID[r]} is integrated (MUX in dGPU-only mode, a port wired to the dGPU, or a dGPU-primary host); real frames would go to the iGPU and generated frames come back over PCIe"
		_framegen_hybrid_refuse "reverse topology: the display ($FRAMEGEN_HYBRID_SOURCE) is on the discrete GPU ${FRAMEGEN_HYBRID_GPU_ID[p]}"
		return
	fi
	if [[ "$pk" == discrete && "$rk" == discrete ]] && (( !explicit )); then
		_framegen_hybrid_refuse "two discrete GPUs (${FRAMEGEN_HYBRID_GPU_ID[p]}, ${FRAMEGEN_HYBRID_GPU_ID[r]}): set PRESENT_DEV (display GPU) and RENDER_DEV (game GPU) explicitly"
		return
	fi
}

# Child-only steering to the GPU with index I, in FRAMEGEN_HYBRID_CHILD_ENV. A
# key the caller already set is passed through untouched (and listed in
# FRAMEGEN_HYBRID_CALLER_KEYS), with a warning when it names another GPU.
framegen_hybrid_steer_env()
{
	local r=$1 key slot
	local -A want=( [MESA_VK_DEVICE_SELECT]="${FRAMEGEN_HYBRID_GPU_ID[r]}!" )
	FRAMEGEN_HYBRID_CHILD_ENV=() FRAMEGEN_HYBRID_CALLER_KEYS=()
	if [[ "${FRAMEGEN_HYBRID_GPU_DRIVER[r]}" == nvidia ]]; then
		# NVIDIA README, "PRIME Render Offload".
		want[__NV_PRIME_RENDER_OFFLOAD]=1
		want[__VK_LAYER_NV_optimus]=NVIDIA_only
		want[__GLX_VENDOR_LIBRARY_NAME]=nvidia
	else
		# Mesa envvars: DRI_PRIME=pci-0000_02_00_0 (GL and Vulkan).
		slot=${FRAMEGEN_HYBRID_GPU_SLOT[r]}
		want[DRI_PRIME]="pci-${slot//[:.]/_}"
	fi
	for key in "${FRAMEGEN_HYBRID_STEER_KEYS[@]}"; do
		if [[ -n "${!key+set}" ]]; then
			FRAMEGEN_HYBRID_CHILD_ENV+=( "$key=${!key}" )
			FRAMEGEN_HYBRID_CALLER_KEYS+=( "$key" )
		elif [[ -n "${want[$key]+set}" ]]; then
			FRAMEGEN_HYBRID_CHILD_ENV+=( "$key=${want[$key]}" )
		fi
	done
	_framegen_hybrid_check_caller_keys "$r"
}

# A caller-set selector that names another GPU than index I wins: say so.
# Only the vendor:device and pci-DDDD_BB_DD_F forms name a GPU.
_framegen_hybrid_check_caller_keys()
{
	local key value named
	local id=${FRAMEGEN_HYBRID_GPU_ID[$1]} slot=${FRAMEGEN_HYBRID_GPU_SLOT[$1]}
	for key in MESA_VK_DEVICE_SELECT DRI_PRIME; do
		value=${!key:-}
		named=${value%!}
		named=${named,,}
		if [[ "$named" =~ $FRAMEGEN_DEVICE_ID_RE ]]; then
			[[ "$named" != "$id" ]] || continue
		elif [[ "$named" == pci-* ]]; then
			[[ "$named" != "pci-${slot//[:.]/_}" ]] || continue
		else
			continue
		fi
		FRAMEGEN_HYBRID_WARNINGS+=( "caller-set $key=$value is not the game GPU $id; the game follows $key, not this split" )
	done
}

# Gen of a "16.0 GT/s PCIe" attribute; fails for anything unknown.
_framegen_pcie_gen()
{
	case "${1%% *}" in
		2.5) printf 1 ;; 5.0) printf 2 ;; 8.0) printf 3 ;;
		16.0) printf 4 ;; 32.0) printf 5 ;; 64.0) printf 6 ;;
		*) return 1 ;;
	esac
}

# Link cap of the render GPU: min over its upstream bridges, plus the endpoint
# itself only while it is runtime-active (reading a suspended one is pointless).
_framegen_hybrid_pcie_cap()
{
	local i=$1 path="" part gen width gen_min=99 width_min=99999 gen_dev="" width_dev="" note=""
	local -a parts=()
	IFS=/ read -r -a parts <<< "${FRAMEGEN_HYBRID_GPU_PATH[i]}"
	for part in "${parts[@]}"; do
		[[ -n "$part" ]] || continue
		path+="/$part"
		[[ "$part" =~ $FRAMEGEN_PCI_SLOT_RE ]] || continue
		if [[ "$path" == "${FRAMEGEN_HYBRID_GPU_PATH[i]}" && "${FRAMEGEN_HYBRID_GPU_PM[i]}" != active ]]; then
			note="; endpoint not read (runtime_status ${FRAMEGEN_HYBRID_GPU_PM[i]})"
			continue
		fi
		if gen=$(_framegen_pcie_gen "$(_framegen_read_attr "$path/max_link_speed")") && (( gen < gen_min )); then
			gen_min=$gen gen_dev=$part
		fi
		if width=$(_framegen_read_attr "$path/max_link_width") && [[ "$width" =~ ^[0-9]+$ ]] \
			&& (( width < width_min )); then
			width_min=$width width_dev=$part
		fi
	done
	if [[ -z "$gen_dev" || -z "$width_dev" ]]; then
		FRAMEGEN_HYBRID_PCIE="unknown$note"
	elif [[ "$gen_dev" == "$width_dev" ]]; then
		FRAMEGEN_HYBRID_PCIE="Gen$gen_min x$width_min ($gen_dev)$note"
	else
		FRAMEGEN_HYBRID_PCIE="Gen$gen_min ($gen_dev) x$width_min ($width_dev)$note"
	fi
}

# Explicit-sync floors (design §3.4): NVIDIA >= 555, and kernel >= 6.8 on DRM.
_framegen_hybrid_collect_warnings()
{
	local r=$FRAMEGEN_HYBRID_RENDER version major minor
	if [[ "${FRAMEGEN_HYBRID_GPU_DRIVER[r]}" == nvidia ]] \
		&& version=$(_framegen_read_attr "${SYSFS_ROOT:-/sys}/module/nvidia/version"); then
		major=${version%%.*}
		if [[ "$major" =~ ^[0-9]+$ ]] && (( major < 555 )); then
			FRAMEGEN_HYBRID_WARNINGS+=( "NVIDIA $version < 555: no explicit sync, gamescope falls back to implicit dma-buf polling" )
		fi
	fi
	if [[ "$FRAMEGEN_HYBRID_HOST" == drm ]]; then
		IFS=. read -r major minor _ <<< "$(uname -r)"
		minor=${minor%%[!0-9]*}
		if [[ "$major" =~ ^[0-9]+$ && "$minor" =~ ^[0-9]+$ ]] \
			&& (( major < 6 || (major == 6 && minor < 8) )); then
			FRAMEGEN_HYBRID_WARNINGS+=( "kernel $(uname -r) < 6.8: no explicit sync on the DRM backend" )
		fi
	fi
}

# Detects the topology and decides the split. BACKEND is gamescope's --backend
# (auto when unset); OUTPUTS its -O list. Fills PRESENT_DEV and RENDER_DEV when
# unset (RENDER_DEV gets one trailing !) and FRAMEGEN_HYBRID_PIN. Returns 3
# with FRAMEGEN_HYBRID_REFUSAL set (FRAMEGEN_HYBRID_PIN too when the display
# GPU is known; FRAMEGEN_HYBRID_REVERSE=1 when it is the discrete one) when the
# split must not run. FRAMEGEN_HYBRID_REFUSAL is one short line for stderr;
# FRAMEGEN_HYBRID_REFUSAL_DETAIL, when set, is the report's explanation.
framegen_detect_hybrid_devices()
{
	local backend=${1:-auto} outputs=${2:-} explicit=-1 both=0 i
	local -A seen=()
	_framegen_hybrid_reset
	FRAMEGEN_HYBRID_HOST=$(_framegen_hybrid_host_kind "$backend")
	if [[ -n "${PRESENT_DEV:-}" && -n "${RENDER_DEV:-}" ]]; then both=1; fi
	_framegen_hybrid_enumerate
	_framegen_hybrid_sort
	_framegen_hybrid_check_nvidia_drm
	# The lid matters only where gamescope picks the connector itself (DRM);
	# nested, the panel owner merely stands in for the host compositor's GPU.
	if [[ "$FRAMEGEN_HYBRID_HOST" == drm ]] && _framegen_lid_closed; then FRAMEGEN_HYBRID_LID_CLOSED=1; fi
	if (( ${#FRAMEGEN_HYBRID_GPU_ID[@]} == 0 )); then
		_framegen_hybrid_refuse "no PCI GPU under ${SYSFS_ROOT:-/sys}/class/drm"
		return
	fi
	for i in "${!FRAMEGEN_HYBRID_GPU_ID[@]}"; do
		if [[ -n "${seen[${FRAMEGEN_HYBRID_GPU_ID[i]}]:-}" ]]; then
			_framegen_hybrid_refuse "two GPUs share ${FRAMEGEN_HYBRID_GPU_ID[i]}; --prefer-vk-device and MESA_VK_DEVICE_SELECT cannot tell them apart"
			return
		fi
		seen[${FRAMEGEN_HYBRID_GPU_ID[i]}]=1
	done
	if [[ -n "${PRESENT_DEV:-}" ]]; then
		_framegen_hybrid_resolve_id PRESENT_DEV || return
		explicit=$FRAMEGEN_HYBRID_INDEX
	fi
	_framegen_hybrid_pick_present "$outputs" "$explicit" || return
	_framegen_hybrid_pick_render || return
	_framegen_hybrid_check_topology "$both" || return
	PRESENT_DEV=${PRESENT_DEV:-${FRAMEGEN_HYBRID_GPU_ID[FRAMEGEN_HYBRID_PRESENT]}}
	RENDER_DEV=${RENDER_DEV:-${FRAMEGEN_HYBRID_GPU_ID[FRAMEGEN_HYBRID_RENDER]}!}
	FRAMEGEN_HYBRID_PIN=$PRESENT_DEV
	framegen_hybrid_steer_env "$FRAMEGEN_HYBRID_RENDER"
	_framegen_hybrid_pcie_cap "$FRAMEGEN_HYBRID_RENDER"
	_framegen_hybrid_collect_warnings
}

# True for a laptop-shaped machine: an integrated and a discrete GPU and a
# built-in panel connector. The doctor warns about a refused split only there.
framegen_hybrid_laptop_shaped()
{
	local kinds=" ${FRAMEGEN_HYBRID_GPU_KIND[*]} "
	[[ "$kinds" == *" integrated "* && "$kinds" == *" discrete "* ]] && (( FRAMEGEN_HYBRID_HAS_PANEL ))
}

_framegen_hybrid_role()
{
	if (( $1 == FRAMEGEN_HYBRID_PRESENT )); then printf present
	elif (( $1 == FRAMEGEN_HYBRID_RENDER )); then printf render
	else printf -- -
	fi
}

# Human-readable result of framegen_detect_hybrid_devices. LABEL says who gets
# the steering env.
framegen_hybrid_report()
{
	local label=$1 i warning
	printf 'sysfs %s, host %s%s\n' "${SYSFS_ROOT:-/sys}" "${FRAMEGEN_HYBRID_HOST:-?}" \
		"$( (( !FRAMEGEN_HYBRID_LID_CLOSED )) || printf ', lid closed: panels ignored unless named by -O')"
	for i in "${!FRAMEGEN_HYBRID_GPU_ID[@]}"; do
		printf '  %-7s %s %s class %s %-7s %-10s %-10s pm %-9s role %-7s connected: %s\n' \
			"${FRAMEGEN_HYBRID_GPU_CARD[i]}" "${FRAMEGEN_HYBRID_GPU_SLOT[i]}" \
			"${FRAMEGEN_HYBRID_GPU_ID[i]}" "${FRAMEGEN_HYBRID_GPU_CLASS[i]}" \
			"${FRAMEGEN_HYBRID_GPU_DRIVER[i]}" "${FRAMEGEN_HYBRID_GPU_KIND[i]}" \
			"${FRAMEGEN_HYBRID_GPU_RENDER[i]}" "${FRAMEGEN_HYBRID_GPU_PM[i]}" \
			"$(_framegen_hybrid_role "$i")" "${FRAMEGEN_HYBRID_GPU_CONNS[i]:--}"
	done
	[[ -z "$FRAMEGEN_HYBRID_SOURCE" ]] || printf 'display: %s\n' "$FRAMEGEN_HYBRID_SOURCE"
	if [[ -n "$FRAMEGEN_HYBRID_REFUSAL" ]]; then
		printf 'decision: no split: %s%s\n' "$FRAMEGEN_HYBRID_REFUSAL" \
			"${FRAMEGEN_HYBRID_REFUSAL_DETAIL:+; $FRAMEGEN_HYBRID_REFUSAL_DETAIL}"
	else
		printf 'decision: split\n'
		printf '  PRESENT_DEV=%s (gamescope: --prefer-vk-device %s)\n' "$PRESENT_DEV" "$PRESENT_DEV"
		printf '  RENDER_DEV=%s (%s, game only)\n' "$RENDER_DEV" \
			"${FRAMEGEN_HYBRID_GPU_DRIVER[FRAMEGEN_HYBRID_RENDER]}"
		printf '  steering env (%s): %s\n' "$label" "${FRAMEGEN_HYBRID_CHILD_ENV[*]}"
		(( ${#FRAMEGEN_HYBRID_CALLER_KEYS[@]} == 0 )) \
			|| printf '  caller-set, passed through: %s\n' "${FRAMEGEN_HYBRID_CALLER_KEYS[*]}"
		printf '  pcie cap: %s\n' "$FRAMEGEN_HYBRID_PCIE"
	fi
	for warning in "${FRAMEGEN_HYBRID_WARNINGS[@]}"; do
		printf 'warning: %s\n' "$warning"
	done
}

# The doctor's HYBRID section: the report, indented, then one PASS, WARN or
# INFO line. Returns 0 split ready, 1 no split on a laptop-shaped machine
# (WARN), 2 not applicable: one GPU, a desktop (INFO).
framegen_hybrid_doctor()
{
	local rc=0
	framegen_detect_hybrid_devices auto "" || rc=$?
	framegen_hybrid_report games | sed 's/^/      /'
	if (( rc == 0 )); then
		printf 'PASS  hybrid split ready: scripts/gamescope-hybrid.sh [gamescope options] -- steam (and --child -- %%command%% per game)\n'
		return 0
	fi
	if framegen_hybrid_laptop_shaped; then
		printf 'WARN  hybrid laptop, but no split: %s\n' "$FRAMEGEN_HYBRID_REFUSAL"
		return 1
	fi
	printf 'INFO  hybrid split not applicable: %s\n' "$FRAMEGEN_HYBRID_REFUSAL"
	return 2
}
