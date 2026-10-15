#!/usr/bin/env bash
# Fixture tests for scripts/gamescope-hybrid.sh and the hybrid helpers in
# scripts/framegen-launch-common.sh. Needs no GPU: every topology is a fake
# sysfs tree under tests/fixtures/sysfs-hybrid/ (the lid state under
# tests/fixtures/proc-lid-closed/), the glvnd EGL vendor dirs are built below,
# and wayland-info, xrandr, uname, gamescope and gamescopereaper are stubs.
# shellcheck disable=SC2016 # stub bodies expand when the stub runs, not here
set -euo pipefail

ROOT=$(CDPATH='' cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
LAUNCHER=$ROOT/scripts/gamescope-hybrid.sh
FIXTURES=$ROOT/tests/fixtures/sysfs-hybrid
LID_CLOSED=$ROOT/tests/fixtures/proc-lid-closed
TMP=$(mktemp -d "${TMPDIR:-/tmp}/gamescope-hybrid-test.XXXXXX")
trap 'rm -rf -- "$TMP"' EXIT

FAKE=$TMP/bin
REAPER=$TMP/reaper
WAKE_MARKER=$TMP/woke-the-dgpu
REAL_UNAME=$(command -v uname)
mkdir -p "$FAKE" "$REAPER"

# stub NAME BODY [DIR]
stub()
{
	local dir=${3:-$FAKE}
	printf '#!/usr/bin/env bash\n%s\n' "$2" >"$dir/$1"
	chmod +x "$dir/$1"
}

# Tools that wake a suspended dGPU: the launcher must never run them.
for tool in nvidia-smi vulkaninfo glxinfo; do
	stub "$tool" "echo $tool >>'$WAKE_MARKER'; exit 1"
done
stub wayland-info '[[ -n "${FAKE_WAYLAND_MAIN:-}" ]] || exit 1
printf "interface: '\''zwp_linux_dmabuf_v1'\'', version: 4, name: 7\n\tmain device: %s\n" "$FAKE_WAYLAND_MAIN"'
# Without FAKE_XRANDR_PROVIDER0 it answers as Xwayland does: no provider.
stub xrandr 'printf "%s\n" "$*" >"$CAPTURE/xrandr.args"
if [[ -z "${FAKE_XRANDR_PROVIDER0:-}" ]]; then echo "Providers: number : 0"; exit 0; fi
printf "Providers: number : 2\nProvider 0: id: 0x48 cap: 0xf, Source Output, Sink Output, Sink Offload crtcs: 4 outputs: 5 associated providers: 1 name:%s\nProvider 1: id: 0x1f8 cap: 0x2, Sink Output crtcs: 4 outputs: 0 associated providers: 1 name:NVIDIA-G0\n" "$FAKE_XRANDR_PROVIDER0"'
stub uname "if [[ \"\${1:-}\" == -r && -n \"\${FAKE_UNAME_R:-}\" ]]; then echo \"\$FAKE_UNAME_R\"; else exec '$REAL_UNAME' \"\$@\"; fi"
# Fake gamescope: records its own env and argv, then runs what follows `--`
# through gamescopereaper on PATH, as gamescope does. Like gamescope, it gives
# the child DISPLAY and no WAYLAND_DISPLAY unless --expose-wayland is passed.
stub gamescope 'env >"$CAPTURE/gamescope.env"; printf "%s\n" "$@" >"$CAPTURE/gamescope.args"
wayland=( -u WAYLAND_DISPLAY )
while (( $# )) && [[ "$1" != -- ]]; do
	[[ "$1" != --expose-wayland ]] || wayland=( WAYLAND_DISPLAY=gamescope-0 )
	shift
done
shift; exec env "${wayland[@]}" DISPLAY=:99 gamescopereaper -- "$@"'
stub gamescopereaper '[[ "${1:-}" != -- ]] || shift; exec "$@"' "$REAPER"
stub fake-game 'env >"$CAPTURE/game.env"; printf "%s\n" "$0" "$@" >"$CAPTURE/game.args"'
# A game whose argv[0] contains "=": an `env` wrapper would take it for an assignment.
cp "$FAKE/fake-game" "$FAKE/odd=name"
cp "$FAKE/fake-game" "$FAKE/steam"
# Steam's %command%: steam-launch-wrapper -- reaper SteamLaunch AppId=N -- game.
stub steam-launch-wrapper 'printf "%s\n" "${0##*/}" "$@" >"$CAPTURE/wrapper.args"; shift; exec "$@"'
stub reaper 'while [[ "$1" != -- ]]; do shift; done; shift; exec "$@"'

# glvnd EGL vendors as on the POC (NVIDIA's sorts first), plus one in /etc.
EGL_ETC=$TMP/glvnd/etc
EGL_SHARE=$TMP/glvnd/share
mkdir -p "$EGL_ETC" "$EGL_SHARE"
egl_vendor() { printf '{ "file_format_version" : "1.0.0", "ICD" : { "library_path" : "%s" } }\n' "$2" >"$1"; }
egl_vendor "$EGL_SHARE/10_nvidia.json" libEGL_nvidia.so.0
egl_vendor "$EGL_SHARE/50_mesa.json" libEGL_mesa.so.0
egl_vendor "$EGL_ETC/60_custom.json" libEGL_custom.so.0
EGL_DIRS=$EGL_ETC:$EGL_SHARE
EGL_LIST=$EGL_ETC/60_custom.json:$EGL_SHARE/50_mesa.json
EGL_KEEP="__EGL_VENDOR_LIBRARY_FILENAMES=$EGL_LIST"
# A fixed kernel, so the explicit-sync floor never depends on the test host.
KERNEL=6.12.0-fixture

PASS=0
FAIL=0
OUT=""
RC=0
RUN_CMD=( bash "$LAUNCHER" )

# run FIXTURE [VAR=value...] -- launcher args. Clean environment every time;
# the lid reads as open unless PROCFS_ROOT is overridden.
run()
{
	local fixture=$1
	shift
	local -a vars=()
	while [[ "$1" != -- ]]; do vars+=( "$1" ); shift; done
	shift
	rm -rf "$TMP/capture"; mkdir -p "$TMP/capture"
	RC=0
	OUT=$(env -i PATH="$FAKE:$REAPER:/usr/bin:/bin" HOME="$TMP" CAPTURE="$TMP/capture" \
		SYSFS_ROOT="$fixture" PROCFS_ROOT="$TMP/no-proc" FAKE_UNAME_R="$KERNEL" \
		GAMESCOPE_BIN="$FAKE/gamescope" FRAMEGEN_EGL_VENDOR_DIRS="$EGL_DIRS" \
		"${vars[@]}" "${RUN_CMD[@]}" "$@" 2>&1) || RC=$?
}

captured()
{
	cat "$TMP/capture/$1" 2>/dev/null || true
}

check()
{
	local name=$1 want_rc=$2 pattern
	shift 2
	local ok=1
	if [[ "$RC" != "$want_rc" ]]; then
		printf '  rc %s, want %s\n' "$RC" "$want_rc"
		ok=0
	fi
	for pattern in "$@"; do
		if [[ "$pattern" == '!'* ]]; then
			if grep -qF -- "${pattern#!}" <<< "$OUT"; then
				printf '  unexpected: %s\n' "${pattern#!}"
				ok=0
			fi
		elif ! grep -qF -- "$pattern" <<< "$OUT"; then
			printf '  missing: %s\n' "$pattern"
			ok=0
		fi
	done
	if (( ok )); then
		PASS=$((PASS + 1))
		printf 'ok   %s\n' "$name"
	else
		FAIL=$((FAIL + 1))
		printf 'FAIL %s\n%s\n' "$name" "$OUT"
	fi
}

# expect_eq NAME WANT GOT
expect_eq()
{
	if [[ "$2" == "$3" ]]; then
		PASS=$((PASS + 1)); printf 'ok   %s\n' "$1"
	else
		FAIL=$((FAIL + 1)); printf 'FAIL %s\n  want: %s\n  got:  %s\n' "$1" "$2" "$3"
	fi
}

# The argv a stub recorded in FILE, on one line, without the program's directory.
argv_of()
{
	captured "$1" | tr '\n' ' ' | sed 's|^[^ ]*/||; s/ $//'
}

# expect_no FILE NAME: the stub that writes FILE must not have run.
expect_no()
{
	if [[ -e "$TMP/capture/$1" ]]; then
		FAIL=$((FAIL + 1)); printf 'FAIL %s\n' "$2"
	else
		PASS=$((PASS + 1)); printf 'ok   %s\n' "$2"
	fi
}

NV_KEYS=( '__NV_PRIME_RENDER_OFFLOAD=1' '__VK_LAYER_NV_optimus=NVIDIA_only' '__GLX_VENDOR_LIBRARY_NAME=nvidia' )
NO_NV_KEYS=( '!__NV_PRIME_RENDER_OFFLOAD' '!__VK_LAYER_NV_optimus' '!__GLX_VENDOR_LIBRARY_NAME' )
NO_STEER=( '!MESA_VK_DEVICE_SELECT' '!DRI_PRIME' "${NO_NV_KEYS[@]}" )

# --- AMD iGPU (890M) + NVIDIA 0x0302: the POC -------------------------------
POC=$FIXTURES/amd-nvidia-poc
run "$POC" -- --report --backend drm
check "poc drm: split, NV keys, cap Gen4 x4 at the root port" 0 \
	'decision: split' 'PRESENT_DEV=1002:150e' 'RENDER_DEV=10de:2db9!' \
	'MESA_VK_DEVICE_SELECT=10de:2db9!' "${NV_KEYS[@]}" '!DRI_PRIME' \
	'display: connector eDP-1' 'pcie cap: Gen4 x4 (0000:00:03.1)' \
	'--prefer-vk-device 1002:150e' 'gamescope_upscale_preemptive=false' '!warning:' 'mode host:'

run "$POC" WAYLAND_DISPLAY=wayland-0 FAKE_WAYLAND_MAIN='0xE280 (/dev/dri/card1 or /dev/dri/renderD128)' -- --report
check "poc nested wayland: host GPU confirmed by main_device" 0 \
	'host wayland' 'display: wayland-info main_device renderD128' 'PRESENT_DEV=1002:150e' '!assumed'

run "$POC" WAYLAND_DISPLAY=wayland-0 FAKE_WAYLAND_MAIN='0xE281' -- --report
check "poc nested wayland, host composites on the dGPU (T3): no split" 3 \
	'no split: reverse topology' 'mode host: no split: reverse topology' 'gamescope on 10de:2db9, child unsteered' \
	'!fallback:'

run "$POC" WAYLAND_DISPLAY=wayland-0 -- --report
check "poc nested wayland without wayland-info: assumed" 0 \
	'display: assumed connector eDP-1' 'install wayland-info' 'decision: split'

run "$POC" WAYLAND_DISPLAY=wayland-0 AQ_DRM_DEVICES=/dev/dri/card1:/dev/dri/card2 -- --report
check "poc nested Hyprland: AQ_DRM_DEVICES first entry confirms the iGPU" 0 \
	'display: AQ_DRM_DEVICES first entry card1' 'decision: split' '!assumed'
run "$POC" WAYLAND_DISPLAY=wayland-0 KWIN_DRM_DEVICES=/dev/dri/card2:/dev/dri/card1 -- --report
check "poc nested KWin pinned to the dGPU: no split" 3 \
	'KWIN_DRM_DEVICES first entry card2' 'no split: reverse topology'

run "$POC" DISPLAY=:0 FAKE_XRANDR_PROVIDER0='modesetting' -- --report --backend sdl
check "poc X11 modesetting host: assumed panel owner" 0 'host x11' 'assumed connector eDP-1' 'decision: split'

run "$POC" DISPLAY=:0 FAKE_XRANDR_PROVIDER0='AMD Radeon 890M Graphics @ pci:0000:c7:00.0' -- --report
check "poc X11 amdgpu-DDX host: confirmed by PCI slot" 0 "display: xrandr provider 0" 'PRESENT_DEV=1002:150e'

run "$POC" DISPLAY=:0 FAKE_XRANDR_PROVIDER0='NVIDIA-0' -- --report
check "poc NVIDIA-primary X11 host: no split" 3 "xrandr provider 0 'NVIDIA-0'" 'no split: reverse topology'

run "$POC" RENDER_DEV='10de:2db9!' PRESENT_DEV=1002:150e -- --report --backend drm
check "explicit RENDER_DEV with !: kept, ! not doubled" 0 \
	'RENDER_DEV=10de:2db9! ' 'MESA_VK_DEVICE_SELECT=10de:2db9! ' '!!!'

run "$POC" RENDER_DEV=10de:2db9 -- --report --backend drm
check "explicit RENDER_DEV without !: kept, child gets one !" 0 \
	'RENDER_DEV=10de:2db9 (' 'MESA_VK_DEVICE_SELECT=10de:2db9! '

run "$POC" RENDER_DEV='10de:2db9!!' -- --report --backend drm
check "malformed RENDER_DEV: shared message" 3 "RENDER_DEV must be vendor:device or vendor:device!, got '10de:2db9!!'"
run "$POC" PRESENT_DEV='1002:150e!' -- --report --backend drm
check "PRESENT_DEV with !: shared message" 3 "PRESENT_DEV must be vendor:device without a trailing !, got '1002:150e!'"

run "$POC" MESA_VK_DEVICE_SELECT=1002:150e -- --report --backend drm
check "caller MESA_VK_DEVICE_SELECT on another GPU: passed through, warned" 0 \
	'MESA_VK_DEVICE_SELECT=1002:150e ' '!MESA_VK_DEVICE_SELECT=10de' \
	'caller-set, passed through: MESA_VK_DEVICE_SELECT' "${NV_KEYS[@]}" \
	'warning: caller-set MESA_VK_DEVICE_SELECT=1002:150e is not the game GPU 10de:2db9'
run "$POC" MESA_VK_DEVICE_SELECT='10DE:2DB9!' DRI_PRIME=1 -- --report --backend drm
check "caller selectors naming the game GPU (or no GPU): no mismatch warning" 0 '!is not the game GPU'
run "$FIXTURES/amd-amd" DRI_PRIME=pci-0000_c4_00_0 -- --report --backend drm
check "caller DRI_PRIME pci form on the iGPU: warned" 0 'warning: caller-set DRI_PRIME=pci-0000_c4_00_0'

run "$POC" PRESENT_DEV=10de:2db9 -- --report --backend drm
check "explicit PRESENT_DEV without a connected connector: no split" 3 \
	'no connected connector on PRESENT_DEV=10de:2db9'

run "$POC" FAKE_UNAME_R=6.6.30-generic -- --report --backend drm
check "kernel < 6.8 on DRM: explicit-sync warning" 0 'warning: kernel 6.6.30-generic < 6.8'
run "$POC" FAKE_UNAME_R=6.6.30-generic WAYLAND_DISPLAY=wayland-0 -- --report
check "kernel < 6.8 nested: no DRM warning" 0 '!warning: kernel'

run "$POC" PROCFS_ROOT="$LID_CLOSED" -- --report --backend drm
check "poc lid closed, nothing external: said so, fallback kept on the panel's GPU" 3 'lid closed' \
	'no split: lid closed and no external display connected; name the panel with -O eDP-1 to use it' \
	'--prefer-vk-device 1002:150e' '!no connected connector'
run "$POC" PROCFS_ROOT="$LID_CLOSED" WAYLAND_DISPLAY=wayland-0 -- --report
check "poc lid closed, nested, host GPU unconfirmed: panel owner stands in, split" 0 \
	'display: assumed connector eDP-1' 'decision: split' '!lid closed' '!reverse'
cp -a "$POC" "$TMP/poc-dp"
echo connected >"$TMP/poc-dp/devices/pci0000:00/0000:00:08.1/0000:c7:00.0/drm/card1/card1-DP-1/status"
run "$TMP/poc-dp" PROCFS_ROOT="$LID_CLOSED" -- --report --backend drm
check "poc lid closed, external DP on the iGPU: gamescope told to use it (-O), not the panel" 0 \
	'display: connector DP-1' 'decision: split' '--prefer-vk-device 1002:150e --backend drm -O DP-1 --'
run "$TMP/poc-dp" PROCFS_ROOT="$LID_CLOSED" -- --report --backend drm -O DP-1
check "poc lid closed, -O given: not added twice" 0 '--backend drm -O DP-1 --' '!-O DP-1 -O'
run "$TMP/poc-dp" -- --report --backend drm
check "poc lid open: the panel, no -O added" 0 'display: connector eDP-1' '!-O DP-1'
run "$TMP/poc-dp" PROCFS_ROOT="$LID_CLOSED" -- --child -- fake-game
OUT=$(captured game.args | tr '\n' ' ')
check "poc lid closed, --child: no -O for the game" 0 'fake-game' '!-O'

# --- Intel iGPU + NVIDIA 0x0302, simpledrm leftover, dGPU in RTD3 ------------
IN=$FIXTURES/intel-nvidia
run "$IN" -- --report --backend drm
check "intel+nvidia: split with NV keys; simpledrm ignored; suspended endpoint not read" 0 \
	'PRESENT_DEV=8086:a7a0' 'RENDER_DEV=10de:28e0!' "${NV_KEYS[@]}" '!DRI_PRIME' '!card0' \
	'pcie cap: Gen4 x8 (0000:00:01.0); endpoint not read (runtime_status suspended)' \
	'warning: NVIDIA 550.120 < 555'

cp -a "$IN" "$TMP/intel-nvidia-active"
echo active >"$TMP/intel-nvidia-active/devices/pci0000:00/0000:00:01.0/0000:01:00.0/power/runtime_status"
run "$TMP/intel-nvidia-active" -- --report --backend drm
check "intel+nvidia, dGPU active: endpoint link read" 0 'pcie cap: Gen1 x1 (0000:01:00.0)'

run "$IN" -- --report --backend drm -O HDMI-A-1
check "-O HDMI-A-1 wired to the dGPU: no split" 3 'display (connector HDMI-A-1) is on the discrete GPU 10de:28e0' \
	'gamescope on 10de:28e0, child unsteered' '/gamescope --prefer-vk-device 10de:28e0 --backend drm -O HDMI-A-1 --'
run "$IN" -- --report --backend=drm --prefer-output=HDMI-A-1,eDP-1
check "--prefer-output= form honoured" 3 'connector HDMI-A-1'
run "$IN" -- --report --backend drm -ODP-9,eDP-1
check "-O list: first connected entry wins" 0 'display: connector eDP-1'
run "$IN" PROCFS_ROOT="$LID_CLOSED" -- --report --backend drm
check "intel+nvidia lid closed, docked on dGPU HDMI: panel skipped, no split, fallback told -O HDMI-A-1" 3 \
	'display: connector HDMI-A-1' 'no split: reverse topology' '--prefer-vk-device 10de:28e0 --backend drm -O HDMI-A-1 --'
run "$IN" PROCFS_ROOT="$LID_CLOSED" WAYLAND_DISPLAY=wayland-0 -- --report
check "intel+nvidia lid closed, docked, nested (T3): lid ignored, iGPU panel owner, split" 0 \
	'display: assumed connector eDP-1' 'decision: split' '--prefer-vk-device 8086:a7a0' '!lid closed'

run "$FIXTURES/intel-nvidia-no-drm" -- --report --backend drm
check "NVIDIA bound without nvidia-drm: named, modeset=1 suggested" 3 \
	'no split: NVIDIA GPU 0000:01:00.0 has no DRM node: load nvidia-drm with modeset=1' '!only one PCI GPU' \
	'warning: NVIDIA GPU 0000:01:00.0 is bound to nvidia but has no DRM card node' 'nvidia-drm.modeset=1' \
	'mode host: no split: NVIDIA GPU 0000:01:00.0 has no DRM node'

# --- AMD iGPU + AMD dGPU (Framework 16): dGPU is card0 ------------------------
AA=$FIXTURES/amd-amd
run "$AA" -- --report --backend drm
check "amd+amd: DRI_PRIME pci form, no NV keys, no ICD filter" 0 \
	'PRESENT_DEV=1002:15bf' 'MESA_VK_DEVICE_SELECT=1002:7480!' 'DRI_PRIME=pci-0000_03_00_0' "${NO_NV_KEYS[@]}" \
	'!VK_LOADER_DRIVERS_DISABLE'
run "$AA" -- --report --backend drm -O DP-1
check "amd+amd, -O on the dGPU USB-C: no split" 3 'no split: reverse topology'
run "$AA" -- --report --backend drm -O '*'
check "amd+amd, -O '*': integrated GPU first, not card order" 0 'display: connector eDP-1' 'decision: split'
run "$AA" PROCFS_ROOT="$LID_CLOSED" -- --report --backend drm -O eDP-1
check "lid closed but -O names the panel: honoured" 0 'display: connector eDP-1'

# --- Intel iGPU + Intel Arc dGPU --------------------------------------------
run "$FIXTURES/intel-arc" -- --report --backend drm
check "intel+arc: DRI_PRIME pci form" 0 \
	'PRESENT_DEV=8086:7d55' 'MESA_VK_DEVICE_SELECT=8086:5693!' 'DRI_PRIME=pci-0000_03_00_0' "${NO_NV_KEYS[@]}"

# --- Reverse, single-GPU and ambiguous topologies -----------------------------
run "$FIXTURES/amd-nvidia-mux-dgpu" -- --report --backend drm
check "MUX dGPU mode, panel on the dGPU: no split, explained in the report" 3 \
	'is on the discrete GPU 10de:2560; the game GPU 1002:1681 is integrated' '--prefer-vk-device 10de:2560' \
	'generated frames come back over PCIe'
run "$FIXTURES/dgpu-only" -- --report --backend drm
check "dGPU-only (iGPU hidden): no split, report shows the pass-through" 3 'only one PCI GPU (10de:2560)' \
	'mode host: no split' "command: $FAKE/gamescope --prefer-vk-device 10de:2560 --backend drm --" \
	'!gamescope_upscale_preemptive' '!VK_LOADER_DRIVERS_DISABLE'

DD=$FIXTURES/desktop-two-discrete
run "$DD" -- --report --backend drm
check "two discrete without IDs: no split" 3 'two discrete GPUs' 'set PRESENT_DEV'
run "$DD" PRESENT_DEV=1002:731f RENDER_DEV='1002:744c!' -- --report --backend drm
check "two discrete with explicit IDs: split" 0 \
	'decision: split' 'MESA_VK_DEVICE_SELECT=1002:744c!' 'DRI_PRIME=pci-0000_03_00_0'

cp -a "$DD" "$TMP/two-monitors"
echo connected >"$TMP/two-monitors/devices/pci0000:00/0000:00:01.1/0000:03:00.0/drm/card1/card1-DP-1/status"
run "$TMP/two-monitors" PRESENT_DEV=1002:744c RENDER_DEV='1002:731f!' -- --report --backend drm
check "monitors on both GPUs: explicit PRESENT_DEV (card1) picks its own connector" 0 \
	'decision: split' 'PRESENT_DEV=1002:744c' 'display: connector DP-1' 'MESA_VK_DEVICE_SELECT=1002:731f!'

DDI=$FIXTURES/desktop-two-discrete-igpu
run "$DDI" -- --report --backend drm
check "two discrete + unrelated iGPU: explicit IDs required, not reverse" 3 \
	'two discrete GPUs (1002:731f, 1002:744c)' '!reverse topology'
run "$DDI" PRESENT_DEV=1002:731f RENDER_DEV='1002:744c!' -- --report --backend drm
check "two discrete + unrelated iGPU, explicit IDs: split" 0 'decision: split' 'MESA_VK_DEVICE_SELECT=1002:744c!'

run "$FIXTURES/identical-ids" -- --report --backend drm
check "identical vendor:device: no split, nothing pinned" 3 'two GPUs share 1002:744c' \
	'gamescope on the default GPU, child unsteered' "command: $FAKE/gamescope --backend drm --"

# --- Report wording per mode ---------------------------------------------------
run "$POC" -- --report --backend drm
check "host report: steering env labelled for --child games, gamescope resolved" 0 \
	'steering env (games run with --child; not the child of this gamescope): MESA_VK_DEVICE_SELECT=10de:2db9!' \
	"gamescope: $FAKE/gamescope"
run "$POC" -- --report --steer-child --backend drm -- fake-game
check "steer-child report: steering env labelled for gamescope's child" 0 "steering env (gamescope's child):"
run "$POC" -- --report --child -- fake-game
check "child report: steering env labelled for the game, no gamescope line" 0 'steering env (the game):' '!gamescope: /'

# --- -O names connected on several GPUs ----------------------------------------
cp -a "$FIXTURES/amd-amd" "$TMP/amd-amd-dp"
IGPU_DRM=$TMP/amd-amd-dp/devices/pci0000:00/0000:00:08.1/0000:c4:00.0/drm/card1
mkdir -p "$IGPU_DRM/card1-DP-1"
echo connected >"$IGPU_DRM/card1-DP-1/status"
run "$TMP/amd-amd-dp" -- --report --backend drm -O DP-1
check "-O DP-1 connected on both GPUs: iGPU first, warned" 0 'display: connector DP-1' 'PRESENT_DEV=1002:15bf' \
	'warning: DP-1 is connected on several GPUs (1002:7480, 1002:15bf); using 1002:15bf, set PRESENT_DEV'
run "$TMP/amd-amd-dp" PRESENT_DEV=1002:15bf -- --report --backend drm -O DP-1
check "-O DP-1 on both GPUs, explicit PRESENT_DEV: no warning" 0 'display: connector DP-1' '!several GPUs'
run "$TMP/amd-amd-dp" -- --report --backend drm
check "no -O: the panel wins, no name warning" 0 'display: connector eDP-1' '!several GPUs'

# --- getopt_long abbreviations of the options the launcher reads --------------
run "$IN" WAYLAND_DISPLAY=wayland-0 -- --report --ba drm --prefer-o HDMI-A-1
check "--ba and --prefer-o abbreviate --backend and --prefer-output" 3 'host drm' 'display: connector HDMI-A-1'
run "$IN" WAYLAND_DISPLAY=wayland-0 -- --report --backe=drm --prefer-out=HDMI-A-1
check "--backe= and --prefer-out= forms" 3 'host drm' 'display: connector HDMI-A-1'
run "$POC" -- --prefer-v 1002:150e -- fake-game
check "--prefer-v abbreviates --prefer-vk-device: refused" 2 'gamescope-hybrid: do not pass --prefer-vk-device'
run "$POC" -- --prefer-vk=1002:150e -- fake-game
check "--prefer-vk= form refused" 2 'set PRESENT_DEV instead'
run "$IN" WAYLAND_DISPLAY=wayland-0 -- --report --b drm --prefer- HDMI-A-1
check "ambiguous --b and --prefer- are not read" 0 'host wayland' '!connector HDMI-A-1'

# --- Doctor HYBRID section: PASS, WARN only on a laptop-shaped machine, INFO --
for case in amd-nvidia-poc:PASS:0 amd-nvidia-mux-dgpu:WARN:1 dgpu-only:INFO:2 \
	desktop-two-discrete:INFO:2 desktop-two-discrete-igpu:INFO:2; do
	IFS=: read -r fixture level want <<< "$case"
	RC=0
	OUT=$(env -i PATH="$FAKE:/usr/bin:/bin" FAKE_UNAME_R="$KERNEL" SYSFS_ROOT="$FIXTURES/$fixture" \
		PROCFS_ROOT="$TMP/no-proc" bash -c 'source "$1"; framegen_hybrid_doctor' _ \
		"$ROOT/scripts/framegen-launch-common.sh") || RC=$?
	check "doctor: $fixture is $level, returns $want" "$want" "$level  hybrid" '      decision: '
done
check "doctor INFO line names the reason" 2 'INFO  hybrid split not applicable: two discrete GPUs'

# --- Real launches against stubs --------------------------------------------
run "$POC" MESA_VK_DEVICE_SELECT=1002:150e -- --steer-child --backend drm -- fake-game --flag
GS_ENV=$(captured gamescope.env)
GS_ARGS=$(captured gamescope.args)
GAME_ENV=$(captured game.env)
STDERR=$OUT
OUT=$GS_ENV
check "steer-child: gamescope env has no steering keys, preemptive upscale off, NVIDIA and NVK ICDs and EGL filtered" 0 \
	'gamescope_upscale_preemptive=false' 'VK_LOADER_DRIVERS_DISABLE=*nvidia*,*nouveau*' "$EGL_KEEP" \
	"${NO_STEER[@]}" '!__VK_LAYER_NV' '!__NV_' '!PRESENT_DEV' '!RENDER_DEV'
OUT=$(head -n 2 <<< "$GS_ARGS" | tr '\n' ' ')
check "steer-child: gamescope pinned to the display GPU" 0 '--prefer-vk-device 1002:150e'
OUT=$GAME_ENV
check "steer-child: game steered, caller value kept" 0 'MESA_VK_DEVICE_SELECT=1002:150e' "${NV_KEYS[@]}"
OUT=$STDERR
check "steer-child: caller selector mismatch warned on stderr, binary printed" 0 \
	'gamescope-hybrid: warning: caller-set MESA_VK_DEVICE_SELECT=1002:150e is not the game GPU' \
	"gamescope-hybrid: running $FAKE/gamescope" '!10de:2db9!'

run "$POC" MESA_VK_DEVICE_SELECT=1002:150e -- --backend drm -- fake-game
STDERR=$OUT
OUT=$(captured gamescope.env)
check "host mode: gamescope env clean, ICD and EGL filtered" 0 '!MESA_VK_DEVICE_SELECT' \
	'VK_LOADER_DRIVERS_DISABLE=*nvidia*,*nouveau*' "$EGL_KEEP"
OUT=$STDERR
check "host mode: one caller-key warning (the mismatch); child said to be unsteered" 0 \
	'gamescope-hybrid: warning: caller-set MESA_VK_DEVICE_SELECT=1002:150e is not the game GPU 10de:2db9' \
	"!reaches gamescope's child" \
	'gamescope-hybrid: gamescope on 1002:150e, child unsteered (steer each game with --child; game GPU 10de:2db9 (nvidia))'
run "$AA" -- --report --backend drm
check "host mode without filters (AMD + AMD): child unsteered" 0 \
	'mode host: gamescope on 1002:15bf, child unsteered (steer each game with --child'

run "$FIXTURES/amd-nvidia-mux-dgpu" MESA_VK_DEVICE_SELECT=1002:150e -- --steer-child --backend drm -- fake-game
OUT=$(captured game.env)
check "steer-child fallback: the caller's key reaches the child unchanged" 0 'MESA_VK_DEVICE_SELECT=1002:150e'

run "$POC" VK_LOADER_DRIVERS_DISABLE='*lvp*' gamescope_upscale_preemptive=true \
	__EGL_VENDOR_LIBRARY_DIRS=/opt/egl -- --backend drm -- fake-game
STDERR=$OUT
OUT=$(captured gamescope.env)
check "caller's VK_LOADER_DRIVERS_DISABLE gets the game GPU's globs for gamescope; upscale and EGL dirs kept" 0 \
	'gamescope_upscale_preemptive=true' 'VK_LOADER_DRIVERS_DISABLE=*lvp*,*nvidia*,*nouveau*' \
	'__EGL_VENDOR_LIBRARY_DIRS=/opt/egl' '!__EGL_VENDOR_LIBRARY_FILENAMES'
OUT=$STDERR
check "caller-set EGL variable skipping the EGL filter: warned" 0 \
	"gamescope-hybrid: warning: caller-set __EGL_VENDOR_LIBRARY_DIRS skips the EGL filter"
run "$POC" VK_LOADER_DRIVERS_DISABLE='*nouveau*,*nvidia*' -- --report --backend drm
check "caller's VK_LOADER_DRIVERS_DISABLE already covering the globs: unchanged for gamescope" 0 \
	'command: gamescope_upscale_preemptive=false __EGL_VENDOR_LIBRARY_FILENAMES=' '!\*nvidia\*\,\*nouveau\*' \
	'!disables the'
run "$POC" __EGL_VENDOR_LIBRARY_FILENAMES=/opt/egl/50_mesa.json -- --backend drm -- fake-game
OUT=$(captured gamescope.env)$(captured game.env)
check "caller's __EGL_VENDOR_LIBRARY_FILENAMES kept for both" 0 '!_custom' \
	'__EGL_VENDOR_LIBRARY_FILENAMES=/opt/egl/50_mesa.json'
run "$POC" __EGL_VENDOR_LIBRARY_FILENAMES=/opt/egl/50_mesa.json -- --report --backend drm
check "caller's __EGL_VENDOR_LIBRARY_FILENAMES: filter skipped, warned" 0 \
	'warning: caller-set __EGL_VENDOR_LIBRARY_FILENAMES skips the EGL filter'

run "$POC" FRAMEGEN_EGL_VENDOR_DIRS="$EGL_SHARE/none:$TMP/nvidia-only" -- --report --backend drm
check "no EGL vendor outside NVIDIA's: not filtered, warned" 0 '!__EGL_VENDOR' 'VK_LOADER_DRIVERS_DISABLE=\*nvidia\*' \
	"warning: no non-NVIDIA glvnd EGL vendor found"
run "$AA" -- --report --backend drm
check "AMD iGPU + AMD dGPU (one driver): neither loader nor EGL filtered" 0 '!VK_LOADER_DRIVERS_DISABLE' '!__EGL_VENDOR'
run "$FIXTURES/intel-arc" -- --report --backend drm
check "Intel iGPU + Arc (one driver): not filtered" 0 '!VK_LOADER_DRIVERS_DISABLE' '!__EGL_VENDOR'
run "$IN" -- --report --backend drm
check "Intel iGPU + NVIDIA: NVIDIA and NVK filtered" 0 'VK_LOADER_DRIVERS_DISABLE=\*nvidia\*\,\*nouveau\*'

# A dGPU rebound to another driver: FIXTURE DGPU-PATH DRIVER VENDOR DEVICE GLOBS.
for case in "$IN devices/pci0000:00/0000:00:01.0/0000:01:00.0 amdgpu 0x1002 0x7480 *radeon*,*amd*" \
	"$POC devices/pci0000:00/0000:00:03.1/0000:c6:00.0 i915 0x8086 0x5693 *intel*" \
	"$IN devices/pci0000:00/0000:00:01.0/0000:01:00.0 nouveau 0x10de 0x28e0 *nouveau*"; do
	read -r fixture dgpu driver vendor device globs <<< "$case"
	rebound=$TMP/rebound-$driver
	cp -a "$fixture" "$rebound"
	ln -sfn "../../../../bus/pci/drivers/$driver" "$rebound/$dgpu/driver"
	echo "$vendor" >"$rebound/$dgpu/vendor"
	echo "$device" >"$rebound/$dgpu/device"
	run "$rebound" -- --report --backend drm
	check "${fixture##*/} with the dGPU on $driver: only $globs filtered, no EGL list" 0 \
		"$(printf '%q ' "VK_LOADER_DRIVERS_DISABLE=$globs")" '!__EGL_VENDOR' '!nvidia\*'
done
# AMDVLK and AMD's proprietary driver (amd_icd64.json, amd_pro_icd64.json) drive
# an amdgpu GPU too: disabling one AMD driver, or naming only AMDVLK, is fine.
AMD_DGPU=$TMP/rebound-amdgpu
run "$AMD_DGPU" VK_LOADER_DRIVERS_DISABLE='*amd*' -- --child -- fake-game
check "amdgpu game GPU, AMDVLK disabled (RADV left): no warning" 0 '!disables the' 'game on 1002:7480'
run "$AMD_DGPU" VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/amd_icd64.json -- --child -- fake-game
check "amdgpu game GPU, VK_ICD_FILENAMES naming AMDVLK only: no warning" 0 '!names no'
run "$AMD_DGPU" VK_LOADER_DRIVERS_SELECT='amd_pro_icd64.json' -- --child -- fake-game
check "amdgpu game GPU, VK_LOADER_DRIVERS_SELECT naming AMD's driver: no warning" 0 '!skips the'
run "$AMD_DGPU" VK_LOADER_DRIVERS_DISABLE='*radeon*,*amd*' -- --child -- fake-game
check "amdgpu game GPU, every AMD driver disabled: warned" 0 \
	'warning: caller-set VK_LOADER_DRIVERS_DISABLE=*radeon*,*amd* disables the radeon/amd driver: the game cannot use 1002:7480'
run "$AMD_DGPU" VK_ICD_FILENAMES=/x/intel_icd.x86_64.json -- --child -- fake-game
check "amdgpu game GPU, VK_ICD_FILENAMES without an AMD driver: warned" 0 \
	'warning: caller-set VK_ICD_FILENAMES names no radeon/amd driver manifest'
cp -a "$POC" "$TMP/nouveau-igpu"
ln -sfn ../../../../bus/pci/drivers/nouveau "$TMP/nouveau-igpu/devices/pci0000:00/0000:00:08.1/0000:c7:00.0/driver"
run "$TMP/nouveau-igpu" -- --report --backend drm
check "display GPU on nouveau: NVK not filtered for an NVIDIA game GPU" 0 'VK_LOADER_DRIVERS_DISABLE=\*nvidia\* '

run "$POC" -- --backend drm -h 720 -W 1280 -- fake-game
OUT=$(captured gamescope.args | tr '\n' ' ')
check "-h is gamescope's --nested-height, not help" 0 '-h 720 -W 1280 --'

for mode in '' --steer-child; do
	run "$POC" -- ${mode:+"$mode"} --backend drm -- odd=name --x
	OUT=$(captured game.args | tr '\n' ' ')
	check "${mode:-host mode}: argv[0] with = survives the env wrapper" 0 'odd=name --x'
done

run "$POC" -- --steer-child --backend drm -- steam -gamepadui
check "--steer-child wrapping the Steam client: warned" 0 'gamescope-hybrid: warning: --steer-child wraps the Steam client'
run "$POC" -- --steer-child --backend drm -e -- fake-game
check "--steer-child with gamescope's -e (Steam integration): warned" 0 'warning: --steer-child wraps the Steam client'
run "$POC" -- --child -- steam -gamepadui
check "--child wrapping the Steam client: warned the same way" 0 'gamescope-hybrid: warning: --child wraps the Steam client'
run "$POC" -- --report --child -- flatpak run com.valvesoftware.Steam
check "--child wrapping Flatpak Steam: warned" 0 'warning: --child wraps the Steam client'
run "$POC" -- --backend drm -- steam -gamepadui
check "host mode Steam: no Steam warning" 0 '!wraps the Steam client'
run "$FIXTURES/dgpu-only" -- --child -- steam
check "--child Steam without a split: no Steam warning" 0 '!wraps the Steam client'

run "$POC" -- --child -- fake-game
OUT=$(captured game.env)
check "--child: game steered, no gamescope started" 0 'MESA_VK_DEVICE_SELECT=10de:2db9!' "${NV_KEYS[@]}"
expect_no gamescope.args "--child started no gamescope"

run "$POC" -- --child -- odd=name --x
OUT=$(captured game.args | tr '\n' ' ')
check "--child execs the game directly (argv[0] with = survives)" 0 'odd=name --x'

# Gamescope gives its child DISPLAY only: Xwayland lists no xrandr provider, so
# a --child inside it cannot confirm the host GPU and takes the panel owner.
run "$POC" -- --backend drm -- bash "$LAUNCHER" --report --child -- fake-game
check "--child inside host-mode gamescope: X11, xrandr provider unknown, panel owner" 0 'host x11' \
	'display: assumed connector eDP-1 (host GPU not confirmed; xrandr provider unknown)' \
	'mode child: game on 10de:2db9 (nvidia)'
OUT=$(captured xrandr.args)
check "xrandr asked with --current (no output re-probe)" 0 '--current --listproviders'
run "$POC" -- --backend drm --expose-wayland -- bash "$LAUNCHER" --report --child -- fake-game
check "--child inside host-mode gamescope --expose-wayland: wayland path, same decision" 0 'host wayland' \
	'display: assumed connector eDP-1 (host GPU not confirmed; install wayland-info)' \
	'mode child: game on 10de:2db9 (nvidia)'

run "$POC" -- --backend drm -- bash "$LAUNCHER" --child -- fake-game
OUT=$(captured game.env)
check "--child inside host-mode gamescope: no filter reaches the game, game steered" 0 \
	'!VK_LOADER_DRIVERS_DISABLE' '!__EGL_VENDOR' 'MESA_VK_DEVICE_SELECT=10de:2db9!' "${NV_KEYS[@]}" \
	'!gamescope_upscale_preemptive'

# Caller-set Vulkan driver variables that hide the game GPU from the game.
run "$POC" VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/radeon_icd.x86_64.json -- --child -- fake-game
check "--child: VK_ICD_FILENAMES without the game GPU's driver warned" 0 \
	'warning: caller-set VK_ICD_FILENAMES names no nvidia driver manifest: the game cannot use 10de:2db9'
run "$POC" VK_DRIVER_FILES=/x/radeon_icd.x86_64.json:/x/nvidia_icd.json -- --child -- fake-game
check "--child: VK_DRIVER_FILES with the game GPU's driver: no warning" 0 '!warning:'
run "$POC" VK_DRIVER_FILES=/x/radeon_icd.x86_64.json -- --steer-child --backend drm -- fake-game
check "--steer-child: VK_DRIVER_FILES without the game GPU's driver warned" 0 \
	'warning: caller-set VK_DRIVER_FILES names no nvidia driver manifest: the game cannot use 10de:2db9'
run "$POC" VK_LOADER_DRIVERS_SELECT='*radeon*' -- --child -- fake-game
check "--child: VK_LOADER_DRIVERS_SELECT skipping the game GPU's driver warned" 0 \
	"warning: caller-set VK_LOADER_DRIVERS_SELECT=*radeon* skips the nvidia driver: the game cannot use 10de:2db9"
run "$POC" VK_LOADER_DRIVERS_DISABLE='nvidia_icd.json' -- --child -- fake-game
check "--child: VK_LOADER_DRIVERS_DISABLE matching the game GPU's driver warned" 0 \
	"warning: caller-set VK_LOADER_DRIVERS_DISABLE=nvidia_icd.json disables the nvidia driver"
run "$POC" DXVK_FILTER_DEVICE_NAME=Radeon -- --child -- fake-game
check "--child: DXVK_FILTER_DEVICE_NAME warned" 0 \
	"warning: caller-set DXVK_FILTER_DEVICE_NAME=Radeon picks the game's GPU by name"
run "$POC" VK_ICD_FILENAMES=/x/nvidia_icd.json -- --report --backend drm
check "host mode: VK_ICD_FILENAMES without the display GPU's driver warned for gamescope" 0 \
	'warning: caller-set VK_ICD_FILENAMES names no radeon/amd driver manifest: gamescope cannot use 1002:150e' \
	'!the game cannot'

run "$FIXTURES/dgpu-only" -- --child -- fake-game --x
STDERR=$OUT
OUT=$(captured game.env)
check "--child on one GPU fails open: game runs unsteered" 0 "${NO_STEER[@]}"
OUT=$STDERR
check "--child fail-open reason printed once" 0 'gamescope-hybrid: not steering: only one PCI GPU' 'running the game unchanged'
if (( $(grep -c 'only one PCI GPU' <<< "$STDERR") == 1 )); then PASS=$((PASS + 1)); printf 'ok   reason once\n'
else FAIL=$((FAIL + 1)); printf 'FAIL reason not printed exactly once\n%s\n' "$STDERR"; fi
expect_no gamescope.args "--child fail-open started no gamescope"

run "$FIXTURES/amd-nvidia-mux-dgpu" -- --child -- fake-game
STDERR=$OUT
OUT=$(captured game.env)
check "--child, reverse topology: game kept on the display GPU (NVIDIA)" 0 \
	'MESA_VK_DEVICE_SELECT=10de:2560!' "${NV_KEYS[@]}" '!DRI_PRIME'
OUT=$STDERR
check "--child, reverse topology: said once, short (detail only in --report)" 0 'no split: reverse topology' \
	'game kept on the display GPU 10de:2560' '!generated frames come back' '!the game GPU 1002:1681 is integrated'
run "$AA" WAYLAND_DISPLAY=wayland-0 KWIN_DRM_DEVICES=/dev/dri/card0 -- --report --child -- fake-game
check "--child, reverse topology on an AMD dGPU: Mesa keys for the display GPU" 3 \
	'command: MESA_VK_DEVICE_SELECT=1002:7480\! DRI_PRIME=pci-0000_03_00_0 fake-game'

run "$FIXTURES/amd-nvidia-mux-dgpu" -- --steer-child --backend drm -- fake-game
STDERR=$OUT
OUT=$(captured gamescope.args | tr '\n' ' ')
check "steer-child on a MUX dGPU-only laptop: plain gamescope on the dGPU" 0 '--prefer-vk-device 10de:2560 --backend drm -- fake-game'
OUT=$(captured gamescope.env)$(captured game.env)
check "steer-child fallback: nothing steered or filtered" 0 "${NO_STEER[@]}" '!VK_LOADER_DRIVERS_DISABLE' \
	'!gamescope_upscale_preemptive' '!__EGL_VENDOR'
OUT=$STDERR
check "steer-child fallback: info line" 0 'gamescope-hybrid: no split: reverse topology' 'gamescope on 10de:2560, child unsteered'

run "$FIXTURES/identical-ids" -- --backend drm -- fake-game
OUT=$(captured gamescope.args | tr '\n' ' ')
check "ambiguous IDs: plain gamescope, not pinned" 0 '!--prefer-vk-device' '--backend drm -- fake-game'

# --- Invocation ----------------------------------------------------------------
ln -s "$LAUNCHER" "$FAKE/gamescope-hybrid"
RUN_CMD=( gamescope-hybrid )
run "$POC" -- --report --backend drm
check "launcher symlinked onto PATH finds its library" 0 'decision: split'
mkdir -p "$TMP/cdpath/scripts"
RUN_CMD=( bash -c 'cd -- "$0" && exec bash scripts/gamescope-hybrid.sh "$@"' "$ROOT" )
run "$POC" CDPATH="$TMP/cdpath" -- --report --backend drm
check "relative invocation with CDPATH set" 0 'decision: split'

# Gamescope: GAMESCOPE_BIN, else this checkout's build, else PATH.
CHECKOUT=$TMP/checkout
mkdir -p "$CHECKOUT/scripts" "$CHECKOUT/build/src"
cp "$LAUNCHER" "$ROOT/scripts/framegen-launch-common.sh" "$CHECKOUT/scripts/"
RUN_CMD=( bash "$CHECKOUT/scripts/gamescope-hybrid.sh" )
run "$POC" GAMESCOPE_BIN= -- --report --backend drm
check "no build in the checkout: gamescope from PATH" 0 "gamescope: $FAKE/gamescope"
cp "$FAKE/gamescope" "$CHECKOUT/build/src/gamescope"
cp "$REAPER/gamescopereaper" "$CHECKOUT/build/src/gamescopereaper"
# No gamescopereaper on PATH: only the one next to the checkout's build.
run "$POC" GAMESCOPE_BIN= PATH="$FAKE:/usr/bin:/bin" -- --backend drm -- fake-game
check "the checkout's build/src/gamescope before PATH, printed" 0 \
	"gamescope-hybrid: running $CHECKOUT/build/src/gamescope"
OUT=$(captured game.args | tr '\n' ' ')
check "the checkout's build finds its gamescopereaper: the game ran" 0 'fake-game'
OUT=$(captured gamescope.env)
check "the checkout's build dir is first on gamescope's PATH" 0 "PATH=$CHECKOUT/build/src:$FAKE:/usr/bin:/bin"
# A symlink on PATH to the build, whose directory has no gamescopereaper.
mkdir -p "$TMP/linked" "$TMP/plain/scripts"
ln -s "$CHECKOUT/build/src/gamescope" "$TMP/linked/gamescope"
cp "$LAUNCHER" "$ROOT/scripts/framegen-launch-common.sh" "$TMP/plain/scripts/"
RUN_CMD=( bash "$TMP/plain/scripts/gamescope-hybrid.sh" )
run "$POC" GAMESCOPE_BIN= PATH="$TMP/linked:$FAKE:/usr/bin:/bin" -- --backend drm -- fake-game
OUT=$(captured game.args | tr '\n' ' ')
check "gamescope symlinked on PATH: the reaper next to its target is found, the game ran" 0 'fake-game'
OUT=$(captured gamescope.env)
check "gamescope symlinked on PATH: the target's dir is on gamescope's PATH" 0 \
	"PATH=$CHECKOUT/build/src:$TMP/linked:$FAKE:/usr/bin:/bin"
RUN_CMD=( bash "$CHECKOUT/scripts/gamescope-hybrid.sh" )
run "$POC" GAMESCOPE_BIN= -- --report --backend drm
check "the report shows the PATH prepend for the reaper, and the caller's PATH only by name" 0 \
	"command: PATH=$CHECKOUT/build/src:\$PATH gamescope_upscale" ' PATH="$PATH" ' "!$REAPER:/usr/bin"
run "$POC" -- --report --backend drm
check "GAMESCOPE_BIN before the checkout's build" 0 "gamescope: $FAKE/gamescope" '!PATH='

# The game's environment is the caller's in every mode, PATH included (the
# reaper's directory is on gamescope's PATH only): gamescope adds DISPLAY, a
# steering mode the steering keys. Compared with the game run directly.
game_env()
{
	local skip='SHLVL\|_\|DISPLAY\|WAYLAND_DISPLAY'
	[[ -z "$1" ]] || skip+='\|MESA_VK_DEVICE_SELECT\|DRI_PRIME\|__NV_PRIME_RENDER_OFFLOAD\|__VK_LAYER_NV_optimus\|__GLX_VENDOR_LIBRARY_NAME'
	captured game.env | grep -v "^\($skip\)=" | sort
}
# same_env NAME FIXTURE MODE [VAR=value...], with the launcher in RUN_CMD.
same_env()
{
	local name=$1 fixture=$2 mode=$3 want got
	local -a launcher=( "${RUN_CMD[@]}" ) args=( --backend drm )
	shift 3
	[[ "$mode" != --child ]] || args=()
	RUN_CMD=()
	run "$fixture" "$@" -- fake-game
	want=$(game_env "$mode")
	RUN_CMD=( "${launcher[@]}" )
	run "$fixture" "$@" -- ${mode:+"$mode"} "${args[@]}" -- fake-game
	got=$(game_env "$mode")
	if [[ -n "$want" && "$got" == "$want" ]]; then
		PASS=$((PASS + 1)); printf 'ok   %s\n' "$name"
	else
		FAIL=$((FAIL + 1)); printf 'FAIL %s\n%s\n' "$name" "$(diff <(printf '%s\n' "$want") <(printf '%s\n' "$got"))"
	fi
}
same_env "host mode, reaper's dir on gamescope's PATH: the child's env is the caller's" "$POC" '' \
	GAMESCOPE_BIN= PATH="$FAKE:/usr/bin:/bin"
RUN_CMD=( bash "$LAUNCHER" )
same_env "host mode: the child's env is the caller's" "$POC" ''
same_env "host mode: caller-set filter, upscale, EGL and steering values reach the child unchanged" "$POC" '' \
	VK_LOADER_DRIVERS_DISABLE='*lvp*' gamescope_upscale_preemptive=true __EGL_VENDOR_LIBRARY_DIRS=/opt/egl \
	MESA_VK_DEVICE_SELECT=1002:150e DRI_PRIME=1
same_env "host fallback: the child's env is the caller's" "$FIXTURES/dgpu-only" '' MESA_VK_DEVICE_SELECT=10de:2560
same_env "--steer-child: the game's env is the caller's plus the steering" "$POC" --steer-child
same_env "--child: the game's env is the caller's plus the steering" "$POC" --child

run "$POC" GAMESCOPE_BIN=/nonexistent/gamescope -- --backend drm -- fake-game
check "missing gamescope binary: clear error, nothing run" 127 \
	"gamescope-hybrid: gamescope not found: '/nonexistent/gamescope'" 'set GAMESCOPE_BIN'
expect_no game.env "missing gamescope ran nothing"

# --- Usage -------------------------------------------------------------------
run "$POC" -- --backend drm
check "usage: a command is required" 2 'gamescope-hybrid: missing -- command' '(see --help)'
run "$POC" -- --help
check "--help prints usage" 0 'Usage:' '--steer-child'
run "$POC" -- --child
check "usage: --child without a game: nothing to run" 2 'gamescope-hybrid: missing -- command'
run "$POC" -- --help --child
check "--help --child without a game prints usage" 0 'Usage:'
run "$POC" -- --report --child -W 1280 -- fake-game
check "--report --child with gamescope options: report, words ignored, nothing run" 0 \
	'gamescope-hybrid: --child takes no gamescope options; ignored: -W 1280' \
	'command: MESA_VK_DEVICE_SELECT=10de:2db9\! __NV_PRIME_RENDER_OFFLOAD=1 __VK_LAYER_NV_optimus=NVIDIA_only __GLX_VENDOR_LIBRARY_NAME=nvidia fake-game'
expect_no game.args "--report --child ran nothing"
run "$POC" -- --backend drm -W 1280 fake-game --x
check "host mode, forgotten --: usage error" 2 "gamescope-hybrid: missing -- before 'fake-game' (see --help)"
expect_no gamescope.args "host mode, forgotten --: no gamescope"
run "$POC" -- --backend drm stray --help -- fake-game
check "host mode, --help after a stray word: the usage error stands" 2 "missing -- before 'stray'"
run "$POC" -- --steer-child -W 1280 -H 720 steam -gamepadui
check "steer-child, forgotten -- after option values: usage error" 2 "missing -- before 'steam'"
run "$POC" -- --backend drm -f fake-game
check "host mode, a flag took the command as its value: usage error" 2 'gamescope-hybrid: missing -- command'
expect_no gamescope.args "host mode, flag took the command: no gamescope"

# --child reads no gamescope options: the game is what follows an immediate --,
# else everything from the first word naming a command; the words before it are
# ignored with one notice that names the word that runs.
# child_runs NAME WANT_ARGV NOTICE launcher-args... (NOTICE: what follows 'ignored:')
child_runs()
{
	local name=$1 want=$2 ignored=$3 stderr
	shift 3
	run "$POC" -- "$@"
	stderr=$OUT
	expect_eq "--child $name: runs '$want'" "$want" "$(argv_of game.args)"
	OUT=$(captured game.env)
	check "--child $name: steered" 0 'MESA_VK_DEVICE_SELECT=10de:2db9!' "${NV_KEYS[@]}"
	expect_eq "--child $name: notice" "${ignored:+gamescope-hybrid: --child takes no gamescope options; ignored: $ignored}" \
		"$(grep -F ignored: <<< "$stderr" || true)"
}
CHAIN='steam-launch-wrapper -- reaper SteamLaunch AppId=1 -- fake-game --x'
read -r -a chain <<< "$CHAIN"
child_runs "-f before Steam's chain" 'fake-game --x' '-f; running: steam-launch-wrapper' --child -f "${chain[@]}"
expect_eq "--child -f before Steam's chain: the chain runs intact" "$CHAIN" "$(argv_of wrapper.args)"
child_runs "-- before Steam's chain" 'fake-game --x' '' --child -- "${chain[@]}"
expect_eq "--child -- before Steam's chain: the chain runs intact" "$CHAIN" "$(argv_of wrapper.args)"
child_runs "%command% without --" 'fake-game SteamLaunch AppId=1 -- proton run' '' \
	--child fake-game SteamLaunch AppId=1 -- proton run
child_runs "-f game -windowed" 'fake-game -windowed' '-f; running: fake-game' --child -f fake-game -windowed
child_runs "-W 1280 game" 'fake-game' '-W 1280; running: fake-game' --child -W 1280 fake-game
child_runs "--prefer-vk-device" 'fake-game' '--prefer-vk-device 1002:150e --; running: fake-game' \
	--child --prefer-vk-device 1002:150e -- fake-game
child_runs "after options" 'fake-game' '-W 1280 --steer-child; running: fake-game' -W 1280 --steer-child --child -- fake-game
# A stray word before --child is not a usage error: --child never blocks a game.
child_runs "after an option's stray word" 'fake-game' '-W 1280 720; running: fake-game' \
	-W 1280 720 --child -- fake-game
child_runs "after a stray word" 'fake-game' '--backend=sdl stray; running: fake-game' \
	--backend=sdl stray --child -- fake-game
child_runs "--help with a game" 'fake-game --x' '--steer-child --help --; running: fake-game' --child --steer-child --help -- fake-game --x
run "$POC" -- --child -W 1280 no-such-game --x
check "--child, no word names a command: the words run as given (and fail)" 127 'exec: -W: not found'

# --child never blocks the game, not even without the launcher's library.
mkdir -p "$TMP/no-library"
cp "$LAUNCHER" "$TMP/no-library/"
RUN_CMD=( bash "$TMP/no-library/gamescope-hybrid.sh" )
run "$POC" -- --child -- fake-game --x
STDERR=$OUT
OUT=$(captured game.args | tr '\n' ' ')
check "--child without its library: game run unchanged" 0 'fake-game --x'
OUT=$(captured game.env)
check "--child without its library: game unsteered" 0 "${NO_STEER[@]}"
OUT=$STDERR
check "--child without its library: said so" 0 'framegen-launch-common.sh is missing; running the game unchanged'
run "$POC" -- --child -f fake-game --x
expect_eq "--child without its library, -f before the game: game run" 'fake-game --x' "$(argv_of game.args)"
run "$POC" -- --report --child -- fake-game
check "--report --child without its library: problem printed, nothing run" 127 'framegen-launch-common.sh is missing' \
	'!running the game'
expect_no game.args "--report --child without its library ran nothing"
run "$POC" -- --help --child
check "--help --child without its library: usage" 0 'Usage:'
run "$POC" -- --backend drm -- fake-game
check "gamescope mode without the library: not found" 127 'framegen-launch-common.sh is missing'
RUN_CMD=( bash "$LAUNCHER" )

# --- Doctor: a laptop without a split makes the whole doctor exit non-zero ---
DOCTOR_BIN=$TMP/doctor-bin
mkdir -p "$DOCTOR_BIN"
stub vulkaninfo 'exit 1' "$DOCTOR_BIN"
RC=0
OUT=$(env -i PATH="$DOCTOR_BIN:/usr/bin:/bin" HOME="$TMP" CDPATH="$TMP/cdpath" FAKE_UNAME_R="$KERNEL" \
	SYSFS_ROOT="$FIXTURES/amd-nvidia-mux-dgpu" PROCFS_ROOT="$TMP/no-proc" \
	bash -c 'cd -- "$0" && exec bash scripts/framegen-doctor.sh' "$ROOT" 2>&1) || RC=$?
if [[ "$RC" != 0 ]]; then RC=nonzero; fi
check "doctor on a MUX laptop, run relative with CDPATH set: HYBRID WARN, non-zero exit" nonzero \
	'WARN  hybrid laptop, but no split: reverse topology'

# --- Never wake the dGPU -----------------------------------------------------
if [[ -e "$WAKE_MARKER" ]]; then
	FAIL=$((FAIL + 1))
	printf 'FAIL launcher ran a dGPU-waking tool: %s\n' "$(tr '\n' ' ' <"$WAKE_MARKER")"
else
	PASS=$((PASS + 1))
	printf 'ok   never ran nvidia-smi, vulkaninfo or glxinfo\n'
fi

printf '\n%d passed, %d failed\n' "$PASS" "$FAIL"
(( FAIL == 0 ))
