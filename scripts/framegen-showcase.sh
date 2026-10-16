#!/usr/bin/env bash
# Visible framegen showcase and end-to-end judge. Every run prints its exact
# settings before it starts and the HUD is on, so what is on screen is always
# known. Each run is repeated REPS times (rep-major, so drift spreads over all
# runs); the numbers come only from the steady 5 s metrics windows.
#
#   framegen-showcase.sh [--list] [--baseline FILE] [--compare FILE] [--logs DIR] [RUN...]
#     --list            print the run table and settings, then exit
#     --baseline FILE   also save this session's results.json as FILE
#     --compare FILE    judge this session against FILE: exit 0 pass, 1 regression, 2 not comparable
#     --logs DIR        run nothing; summarise (and compare) the run*-r*.log already in DIR
#     RUN...            table indices to run (default: all)
#
# Setup or usage errors exit 2.
#
# Environment (every value is written to each log header):
#   REPS (3)               whole repetitions
#   SECONDS_PER_RUN (45)   at least 5 * (WARMUP_WINDOWS + 2)
#   WARMUP_WINDOWS (1)     leading 5 s windows treated as client startup; raise
#                          it for games with loading screens
#   BACKEND (wayland)      use drm from a text VT for the native verdict
#   W, H, HZ (2560, 1440, 120)
#   RENDER_DEV             client device as MESA_VK_DEVICE_SELECT (vid:did[!]);
#                          empty: the client's default
#   PRESENT_DEV            gamescope --prefer-vk-device; empty: gamescope's default
#   CLIENT                 shell command run inside gamescope (default:
#                          GravityMark from GM_DIR with ASTEROIDS asteroids)
#   GM_DIR                 GravityMark's bin directory (default:
#                          $HOME/GravityMark/bin)
#   CLIENT_PROC            the client's process name (first 15 characters), for
#                          the render device check; GravityMark.x64 by default
#   GAMESCOPE_BIN          binary under test (default: build/src/gamescope); its
#                          build dir also provides the WSI layer and the reaper
#   OUT                    session directory (default: a new timestamped one)
#   DEBUG=1                adds --framegen-debug to framegen runs: diagnosis
#                          only, logging perturbs pacing, never compared
#   EXTRA_ENV              extra gamescope env for every run (the treatment of
#                          an opt-in A/B on one binary)
#   GAMESCOPE_LOCAL_PREFIX passed through to scripts/env-gamescope-local.sh only
#                          when set (its own default applies otherwise); not
#                          written to the log header
#
# Hybrid example: RENDER_DEV='<dgpu vid:did>!' PRESENT_DEV='<igpu vid:did>'
# (render on the dGPU; composite, generate and present on the iGPU).
set -u

REPO=$(cd "$(dirname "$0")/.." && pwd)
REPORT="$REPO/scripts/framegen-showcase-report.py"
REPS=${REPS:-3}
DUR=${SECONDS_PER_RUN:-45}
WARMUP=${WARMUP_WINDOWS:-1}
BACKEND=${BACKEND:-wayland}
W=${W:-2560} H=${H:-1440} HZ=${HZ:-120}
# A nested session paces against its host compositor (present lead, latch
# phase), so it records the host's active modes as context.
host_display()
{
	[[ $BACKEND == drm ]] && { echo native; return; }
	local modes
	modes=$(xrandr --current 2>/dev/null | awk '
		/ connected/ { out = $1; next }
		out && /\*/ { for (i = 2; i <= NF; i++) if ($i ~ /\*/) { sub(/[*+]+$/, "", $i); printf "%s%s %s@%s", sep, out, $1, $i; sep = " " } out = "" }')
	echo "${modes:-unknown}"
}
HOST_DISPLAY=$(host_display)
# A locked host session stops presenting nested windows, so every metric
# would read 0.
host_locked()
{
	[[ $BACKEND == drm ]] && return 1
	local session
	session=$(loginctl show-user "$(id -un)" -p Display --value 2>/dev/null)
	[[ -n "$session" && "$(loginctl show-session "$session" -p LockedHint --value 2>/dev/null)" == yes ]]
}
RENDER_DEV=${RENDER_DEV:-}
PRESENT_DEV=${PRESENT_DEV:-}
ASTEROIDS=${ASTEROIDS:-500000}
GM_DIR=${GM_DIR:-$HOME/GravityMark/bin}
if [[ -n "${CLIENT:-}" ]]; then
	CLIENT_PROC=${CLIENT_PROC:-}
else
	CLIENT="cd '$GM_DIR' && exec ./GravityMark.x64 -vk -temporal 0 -benchmark 0 -fps 1 -asteroids $ASTEROIDS -width $W -height $H -vsync 0"
	CLIENT_PROC=${CLIENT_PROC:-GravityMark.x64}
	CHECK_GRAVITYMARK=1
fi
GAMESCOPE_BIN=$(realpath -m "${GAMESCOPE_BIN:-$REPO/build/src/gamescope}")
BUILD_DIR=$(realpath -m "$(dirname "$GAMESCOPE_BIN")/..")
OUT=${OUT:-${XDG_RUNTIME_DIR:-/tmp}/framegen-showcase/$(date +%Y%m%d-%H%M%S)}
DEBUG=${DEBUG:-0}
DEBUG_FLAG=
[[ "$DEBUG" == 1 ]] && DEBUG_FLAG=--framegen-debug
EXTRA_ENV=${EXTRA_ENV:-}

# name | kind | gamescope framegen flags | extra gamescope env
RUNS=(
	"off|no framegen (baseline)||"
	"extrapolate x2|predictive, no added latency|--experimental-framegen --framegen-mode extrapolate --framegen-multiplier 2|"
	"motion x2|predictive, no added latency|--experimental-framegen --framegen-mode motion --framegen-multiplier 2|"
	"motion x3|predictive, no added latency|--experimental-framegen --framegen-mode motion --framegen-multiplier 3|"
	"bidir x2|interpolation, real frames delayed|--experimental-framegen --framegen-mode motion --framegen-multiplier 2|GAMESCOPE_FRAMEGEN_BIDIR=1"
	"best low-latency|predictive, checked pipeline|--experimental-framegen --framegen-mode motion --framegen-multiplier 2 --framegen-pipeline checked|"
	"best quality|interpolation, checked pipeline|--experimental-framegen --framegen-mode motion --framegen-multiplier 2 --framegen-pipeline checked|GAMESCOPE_FRAMEGEN_BIDIR=1"
	"learned x2 + net|predictive, online-trained refiner|--experimental-framegen --framegen-mode motion --framegen-multiplier 2 --framegen-pipeline learned|GAMESCOPE_FRAMEGEN_NET_ONLINE=1 GAMESCOPE_FRAMEGEN_NET_EVERY=2"
	"guided x2|predictive, guided pipeline|--experimental-framegen --framegen-mode motion --framegen-multiplier 2 --framegen-pipeline guided|"
)

die() { echo "framegen-showcase: $*" >&2; exit 2; }

print_table()
{
	echo
	local i=1 r name kind flags extra
	for r in "${RUNS[@]}"; do
		IFS='|' read -r name kind flags extra <<<"$r"
		printf '  %2d  %-19s %-36s %s %s\n' "$i" "$name" "$kind" "${flags:-(framegen off)}" "$extra"
		i=$((i + 1))
	done
	printf '\nBackend %s  Output %sx%s@%s  Render %s  Present %s\n' \
		"$BACKEND" "$W" "$H" "$HZ" "${RENDER_DEV:-(client default)}" "${PRESENT_DEV:-(gamescope default)}"
	printf 'Client: %s\nBinary: %s\n%s reps x %ss per run, %s warm-up window(s)  Logs: %s\n\n' \
		"$CLIENT" "$GAMESCOPE_BIN" "$REPS" "$DUR" "$WARMUP" "$OUT"
}

# Prints pid:comm for every descendant of $1.
descendants()
{
	local child comm
	for child in $(pgrep -P "$1"); do
		comm=$(cat "/proc/$child/comm" 2>/dev/null) || continue
		printf '%s:%s\n' "$child" "$comm"
		descendants "$child"
	done
}

# Prints seen, not_seen or unverified: did process $1 submit work on RENDER_DEV?
# The method follows the device driver's capability: DRM fdinfo engine time
# where the driver reports it, the driver's own process list otherwise.
render_gpu_check()
{
	local pid=$1 id=${RENDER_DEV%!} dev driver='' addrs=() f
	if [[ -z "$RENDER_DEV" || -z "$pid" ]]; then
		echo unverified
		return
	fi
	for dev in /sys/bus/pci/devices/*; do
		[[ "$(cat "$dev/vendor" 2>/dev/null):$(cat "$dev/device" 2>/dev/null)" == "0x${id%%:*}:0x${id#*:}" ]] || continue
		addrs+=("${dev##*/}")
		[[ -n "$driver" ]] || driver=$(basename "$(readlink "$dev/driver")" 2>/dev/null)
	done
	if ((${#addrs[@]} == 0)); then
		echo unverified
		return
	fi
	# The proprietary nvidia driver has no DRM fdinfo usage statistics; its
	# process list names each process's device by PCI address (8-digit domain).
	if [[ "$driver" == nvidia ]]; then
		if ! command -v nvidia-smi >/dev/null; then
			echo unverified
		elif nvidia-smi --query-compute-apps=pid,gpu_bus_id --format=csv,noheader 2>/dev/null \
			| awk -F', *' -v pid="$pid" -v devs=" ${addrs[*]} " '
				$1 == pid && index(devs, " " substr(tolower($2), length($2) - 11) " ") { found = 1 }
				END { exit !found }'; then
			echo seen
		else
			echo not_seen
		fi
		return
	fi
	# Opening a device during Vulkan enumeration accrues no engine time.
	for f in /proc/"$pid"/fdinfo/*; do
		awk -v devs=" ${addrs[*]} " '
			$1 == "drm-pdev:" && index(devs, " " $2 " ") { dev = 1 }
			($1 ~ /^drm-engine-/ && $1 !~ /^drm-engine-capacity-/ || $1 ~ /^drm-cycles-/) && $2 > 0 { busy = 1 }
			END { exit !(dev && busy) }' "$f" 2>/dev/null && { echo seen; return; }
	done
	echo not_seen
}

# run_one INDEX REP LOG
run_one()
{
	local index=$1 rep=$2 log=$3 name kind flags run_env key gs rc tree='' latest render='' entry pid
	IFS='|' read -r name kind flags run_env <<<"${RUNS[index - 1]}"
	local -A header=(
		[run]="$name" [index]="$index" [rep]="$rep" [kind]="$kind" [flags]="$flags" [run_env]="$run_env"
		[backend]="$BACKEND" [size]="${W}x${H}" [hz]="$HZ" [render]="$RENDER_DEV" [present]="$PRESENT_DEV"
		[client]="$CLIENT" [warmup]="$WARMUP" [extra_env]="$EXTRA_ENV" [debug]="$DEBUG" [seconds]="$DUR"
		[bin]="$GAMESCOPE_BIN" [host_display]="$HOST_DISPLAY"
	)
	: >"$log"
	for key in run index rep kind flags run_env backend size hz render present client warmup extra_env debug seconds bin host_display; do
		printf 'framegen-showcase: %s=%s\n' "$key" "${header[$key]}" >>"$log"
	done

	printf '\n===== RUN %d/%d (rep %d/%d): %s — %s =====\n' "$index" "${#RUNS[@]}" "$rep" "$REPS" "$name" "$kind"
	printf '      gamescope %s on %s, %sx%s@%s Hz: %s %s %s\n      client on %s: %s\n      log: %s\n' \
		"$BACKEND" "${PRESENT_DEV:-(default)}" "$W" "$H" "$HZ" "${flags:-(framegen off)}" "$run_env" "$EXTRA_ENV" \
		"${RENDER_DEV:-(default)}" "$CLIENT" "$log"

	[[ -n "$flags" ]] && flags+=" $DEBUG_FLAG"
	local present_args=() client_env=()
	[[ -n "$PRESENT_DEV" ]] && present_args=(--prefer-vk-device "$PRESENT_DEV")
	[[ -n "$RENDER_DEV" ]] && client_env=("MESA_VK_DEVICE_SELECT=$RENDER_DEV")
	# shellcheck disable=SC2086 # flags and the env lists are word lists by design
	timeout -s INT -k 5 $((DUR + 5)) env \
		GAMESCOPE_BUILD_DIR="$(realpath -m --relative-to="$REPO" "$BUILD_DIR")" \
		GAMESCOPE_FRAMEGEN_METRICS=1 GAMESCOPE_FRAMEGEN_HUD=2 GAMESCOPE_FRAMEGEN_HUD_SCALE=2 $run_env $EXTRA_ENV \
		"$REPO/scripts/env-gamescope-local.sh" "$GAMESCOPE_BIN" --backend "$BACKEND" --expose-wayland \
		"${present_args[@]}" -W "$W" -H "$H" -r "$HZ" $flags -- env "${client_env[@]}" sh -c "$CLIENT" \
		>>"$log" 2>&1 &
	gs=$!

	# Track this run's process tree for cleanup, and check the render device
	# once the client has produced its first metrics window.
	while kill -0 "$gs" 2>/dev/null; do
		latest=$(descendants "$gs")
		[[ -n "$latest" ]] && tree=$latest
		if [[ -z "$render" ]] && grep -q 'framegen-metrics: w=1 ' "$log"; then
			pid=
			while IFS= read -r entry; do
				[[ -n "$CLIENT_PROC" && "${entry#*:}" == "${CLIENT_PROC:0:15}" ]] && { pid=${entry%%:*}; break; }
			done <<<"$tree"
			render=$(render_gpu_check "$pid")
			echo "framegen-showcase: render_gpu=$render" >>"$log"
			case $render in
			seen) echo "      client is rendering on ${RENDER_DEV}" ;;
			not_seen) echo "      WARNING: client NOT seen on ${RENDER_DEV}" ;;
			*) echo "      render device not verified" ;;
			esac
		fi
		sleep 1
	done
	wait "$gs"
	rc=$?
	[[ -n "$render" ]] || echo "framegen-showcase: render_gpu=unverified" >>"$log"
	echo "framegen-showcase: exit=$rc" >>"$log"
	((rc == 124)) || echo "      ended early (rc=$rc)"

	# Only processes this run started, and only while they still are what the
	# snapshot saw (the reaper's parent-death signal normally got there first).
	while IFS= read -r entry; do
		[[ -n "$entry" ]] || continue
		pid=${entry%%:*}
		[[ "$(cat "/proc/$pid/comm" 2>/dev/null)" == "${entry#*:}" ]] && kill -9 "$pid" 2>/dev/null
	done <<<"$tree"
	sleep 2
}

LIST=0 BASELINE='' COMPARE='' LOGS_DIR=''
SELECT=()
while (($#)); do
	case $1 in
	--list) LIST=1 ;;
	--baseline) BASELINE=${2:?--baseline needs a file}; shift ;;
	--compare) COMPARE=${2:?--compare needs a file}; shift ;;
	--logs) LOGS_DIR=${2:?--logs needs a directory}; shift ;;
	-h | --help) sed -n '2,/^set -u/{/^set -u/d;s/^# \{0,1\}//;p}' "$0"; exit 0 ;;
	*[!0-9]* | '') die "unknown argument: $1" ;;
	*)
		((10#$1 >= 1 && 10#$1 <= ${#RUNS[@]})) || die "no run $1 (1-${#RUNS[@]})"
		[[ " ${SELECT[*]} " == *" $((10#$1)) "* ]] || SELECT+=($((10#$1)))
		;;
	esac
	shift
done

for pair in "REPS:$REPS" "SECONDS_PER_RUN:$DUR" "WARMUP_WINDOWS:$WARMUP" "W:$W" "H:$H" "HZ:$HZ"; do
	[[ "${pair#*:}" =~ ^[0-9]+$ ]] || die "${pair%%:*} must be a whole number, not '${pair#*:}'"
done
((REPS >= 1)) || die "REPS must be at least 1"
((DUR >= 5 * (WARMUP + 2))) ||
	die "SECONDS_PER_RUN=$DUR leaves fewer than 2 steady windows; it must be at least $((5 * (WARMUP + 2))) with WARMUP_WINDOWS=$WARMUP"
if [[ "$DEBUG" == 1 && (-n "$BASELINE" || -n "$COMPARE") ]]; then
	die "DEBUG=1 perturbs pacing; its numbers are never a baseline or compared"
fi

if ((LIST)); then
	print_table
	exit 0
fi

if [[ -n "$LOGS_DIR" ]]; then
	shopt -s nullglob
	logs=("$LOGS_DIR"/run*-r*.log)
	shopt -u nullglob
	((${#logs[@]})) || die "no run*-r*.log in $LOGS_DIR"
	results="$LOGS_DIR/results.json"
else
	[[ -x "$GAMESCOPE_BIN" ]] || die "gamescope binary not executable: $GAMESCOPE_BIN (set GAMESCOPE_BIN)"
	[[ -d "$BUILD_DIR/layer" ]] || echo "framegen-showcase: warning: no $BUILD_DIR/layer; the WSI layer will not come from this build" >&2
	if [[ -n "${CHECK_GRAVITYMARK:-}" && ! -x "$GM_DIR/GravityMark.x64" ]]; then
		die "GravityMark not found in $GM_DIR (set GM_DIR, or CLIENT for another client)"
	fi
	((${#SELECT[@]})) || mapfile -t SELECT < <(seq 1 "${#RUNS[@]}")
	print_table
	mkdir -p "$OUT" || die "cannot create $OUT"
	logs=()
	for ((rep = 1; rep <= REPS; rep++)); do
		for i in "${SELECT[@]}"; do
			host_locked && die "the host session is locked: its nested windows are not presented"
			logs+=("$OUT/run$i-r$rep.log")
			run_one "$i" "$rep" "$OUT/run$i-r$rep.log"
		done
	done
	results="$OUT/results.json"
	echo
fi

python3 "$REPORT" summarize --out "$results" "${logs[@]}" || exit 2
echo "results: $results"
if [[ -n "$BASELINE" ]]; then
	cp "$results" "$BASELINE" || die "cannot write $BASELINE"
	echo "baseline saved: $BASELINE"
fi
if [[ -n "$COMPARE" ]]; then
	echo
	exec python3 "$REPORT" compare "$COMPARE" "$results"
fi
