#!/usr/bin/env bash
# Standalone pre-build and Vulkan/topology preflight for the framegen fork.

set -uo pipefail
LC_ALL=C

if (( $# )); then
	if [[ "$1" == "-h" || "$1" == "--help" ]]; then
		printf 'usage: %s\n' "${0##*/}"
		exit 0
	fi
	printf 'unknown option: %s\n' "$1" >&2
	exit 2
fi

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
DOCTOR_TMP=$(mktemp -d "${TMPDIR:-/tmp}/framegen-doctor.XXXXXX") || exit 2
trap 'rm -rf -- "$DOCTOR_TMP"' EXIT

BUILD_OK=1
WARNINGS=0
HARD_FAILURE=0
FIRST_FIX=""

pass() { printf 'PASS  %s\n' "$1"; }
warn() { printf 'WARN  %s\n' "$1"; WARNINGS=1; }
build_fail() {
	printf 'FAIL  %s\n' "$1"
	BUILD_OK=0
	HARD_FAILURE=1
	[[ -n "$FIRST_FIX" ]] || FIRST_FIX=$2
}
version_ge() {
	awk -v have="$1" -v need="$2" 'BEGIN {
		n=split(have,h,"."); m=split(need,w,"."); z=n>m?n:m
		for(i=1;i<=z;i++){a=h[i]+0;b=w[i]+0;if(a>b)exit 0;if(a<b)exit 1} exit 0
	}'
}
pkg_check() {
	local pc=$1 label=$2 min=${3:-} version
	if ! version=$(pkg-config --modversion "$pc" 2>/dev/null); then
		build_fail "$label: pkg-config module '$pc' missing" "install $label development files"
		return
	fi
	if [[ -n "$min" ]] && ! pkg-config --atleast-version="$min" "$pc"; then
		build_fail "$label $version (need >= $min)" "upgrade $label to >= $min"
		return
	fi
	pass "$label $version${min:+ (>= $min)}"
}

printf 'BUILD\n'

# Source: meson.build project(meson_version); vendored wlroots raises the effective floor.
if command -v meson >/dev/null 2>&1; then
	MESON_VERSION=$(meson --version 2>/dev/null | sed -n '1p')
	if version_ge "$MESON_VERSION" 1.3; then
		pass "meson $MESON_VERSION (top-level >= 0.58; vendored wlroots >= 1.3)"
	else
		build_fail "meson $MESON_VERSION (vendored wlroots needs >= 1.3)" "upgrade meson to >= 1.3"
	fi
else
	build_fail "meson missing (need >= 0.58; current vendored wlroots needs >= 1.3)" "install meson >= 1.3"
fi

# Source: meson.build (Meson/Ninja build) and dependency() lookups throughout.
if command -v ninja >/dev/null 2>&1; then
	pass "ninja $(ninja --version 2>/dev/null | sed -n '1p')"
else
	build_fail "ninja missing" "install ninja"
fi
if command -v pkg-config >/dev/null 2>&1; then
	pass "pkg-config $(pkg-config --version 2>/dev/null | sed -n '1p')"
else
	build_fail "pkg-config missing" "install pkg-config"
fi

# Source: src/meson.build find_program('glslang', 'glslangValidator').
GLSLANG=""
command -v glslangValidator >/dev/null 2>&1 && GLSLANG=glslangValidator
[[ -n "$GLSLANG" ]] || { command -v glslang >/dev/null 2>&1 && GLSLANG=glslang; }
if [[ -n "$GLSLANG" ]]; then
	pass "$GLSLANG present"
else
	build_fail "glslang/glslangValidator missing" "install glslang"
fi

# Source: meson.build subproject() calls; uninitialized git submodules have '-' status.
if command -v git >/dev/null 2>&1 && SUBMODULES=$(git -C "$ROOT" submodule status --recursive 2>/dev/null); then
	if awk '/^-/ { bad=1 } END { exit !bad }' <<< "$SUBMODULES"; then
		build_fail "git submodules uninitialized (run: git submodule update --init --recursive)" "initialize git submodules"
	else
		pass "git submodules initialized"
	fi
else
	build_fail "cannot inspect git submodules" "install git and run inside the checkout"
fi

if command -v pkg-config >/dev/null 2>&1; then
	# Source: subprojects/wlroots/meson.build wayland_server and pixman dependencies.
	pkg_check wayland-server wayland-server 1.23.1
	pkg_check pixman-1 pixman-1 0.43.0
	if ! pkg-config --atleast-version=1.23.1 wayland-server 2>/dev/null \
		|| ! pkg-config --atleast-version=0.43.0 pixman-1 2>/dev/null; then
		printf '      FIX: build Wayland and pixman into one PREFIX; pass '\''-Dpkg_config_path=PREFIX/lib/pkgconfig:PREFIX/share/pkgconfig'\''.\n'
	fi

	# Source: subprojects/wlroots/meson.build and backend/{drm,libinput,session}/meson.build.
	pkg_check libdrm libdrm 2.4.122
	pkg_check xkbcommon xkbcommon
	pkg_check libinput libinput 1.19.0
	pkg_check libseat libseat 0.2.0
	if DISPLAYINFO_VERSION=$(pkg-config --modversion libdisplay-info 2>/dev/null); then
		if pkg-config --atleast-version=0.4.0 libdisplay-info; then
			build_fail "libdisplay-info $DISPLAYINFO_VERSION (need < 0.4.0)" "use libdisplay-info < 0.4.0"
		else
			pass "libdisplay-info $DISPLAYINFO_VERSION (< 0.4.0)"
		fi
	else
		build_fail "libdisplay-info: pkg-config module missing" "install libdisplay-info development files"
	fi
	pkg_check hwdata hwdata

	# Source: src/meson.build dependency('SDL2') and top-level dependency('vulkan').
	pkg_check sdl2 sdl2
	if VK_VERSION=$(pkg-config --modversion vulkan 2>/dev/null); then
		VK_INCLUDEDIR=$(pkg-config --variable=includedir vulkan 2>/dev/null)
		if [[ -r "${VK_INCLUDEDIR:-/usr/include}/vulkan/vulkan.h" ]]; then
			pass "Vulkan headers present (vulkan.pc $VK_VERSION)"
		else
			build_fail "Vulkan loader found, but vulkan/vulkan.h missing" "install Vulkan headers"
		fi
	else
		build_fail "Vulkan headers/loader pkg-config module missing" "install Vulkan headers and loader development files"
	fi

	# Source: meson.build enable_tests gate and tests/meson.build catch2-with-main.
	if CATCH_VERSION=$(pkg-config --modversion catch2-with-main 2>/dev/null); then
		pass "catch2-with-main $CATCH_VERSION"
	else
		warn "catch2-with-main missing; tests disabled, -Denable_tests=false"
	fi
fi

printf '\nGPUS\n'

declare -a G_API G_DRIVER G_VENDOR G_DEVICE G_TYPE G_NAME G_MEMFD G_DMABUF
declare -a G_MODIFIER G_FOREIGN G_ROBUST G_TIMELINE G_WRITE G_FP16 G_EXTFMT
declare -a G_LDS G_QCOUNT G_TIMESTAMP G_PRIORITY G_RENDER_OK G_PRESENT_OK G_ATTACHED
GPU_COUNT=0
VULKAN_OK=0

# Source: src/rendervulkan.cpp selectPhysDev/createDevice/BInit feature and extension gates.
if command -v vulkaninfo >/dev/null 2>&1; then
	# One invocation only. Surface variables are removed so a TTY/SSH run still inventories devices.
	env -u DISPLAY -u WAYLAND_DISPLAY -u MESA_VK_DEVICE_SELECT -u DRI_PRIME \
		-u __VK_LAYER_NV_optimus vulkaninfo >"$DOCTOR_TMP/vulkaninfo" 2>&1
	VULKAN_RC=$?
	awk -F '[[:space:]]*=[[:space:]]*' '
	function clean(s){gsub(/\t/," ",s);return s}
	function key(s){gsub(/^[[:space:]]+|[[:space:]]+$/,"",s);return s}
	function reset(){api=drv=vendor=device=type=name="-";mem=dmabuf=mod=foreign=robust=timeline=write=fp16=extfmt=prio=0;lds="-";gc=gt=cc=ct="-";qf=qc=qt="-";inq=inext=0}
	function commitq(){if(!inq)return;if(qf~/QUEUE_COMPUTE_BIT/){if(qf!~/QUEUE_GRAPHICS_BIT/&&cc=="-"){cc=qc;ct=qt}else if(qf~/QUEUE_GRAPHICS_BIT/&&gc=="-"){gc=qc;gt=qt}}}
	function flush(){if(!seen)return;commitq();if(tolower(vendor)=="0x8086"||cc=="-"){qc=gc;qt=gt}else{qc=cc;qt=ct};printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%s\t%s\t%s\t%d\n",id,api,clean(drv),vendor,device,type,clean(name),mem,dmabuf,mod,foreign,robust,timeline,write,fp16,extfmt,lds,qc,qt,prio}
	/^GPU[0-9]+:$/ {flush();reset();seen=1;id=$0;sub(/^GPU/,"",id);sub(/:$/,"",id);next}
	!seen {next}
	{k=key($1)}
	k=="apiVersion"&&api=="-" {split($2,a," ");api=a[1]}
	k=="driverVersion"&&drv=="-" {split($2,a," ");drv=a[1]}
	k=="vendorID"&&vendor=="-" {vendor=tolower($2)}
	k=="deviceID"&&device=="-" {device=tolower($2)}
	k=="deviceType"&&type=="-" {type=$2}
	k=="deviceName"&&name=="-" {name=$2}
	k=="driverName" {drv=$2}
	k=="maxComputeSharedMemorySize"&&lds=="-" {lds=$2}
	/^Device Extensions:/ {inext=1;next}
	/^VkQueueFamilyProperties:/ {inext=0;next}
	inext&&/^[[:space:]]*VK_KHR_external_memory_fd[[:space:]]*:/ {mem=1}
	inext&&/^[[:space:]]*VK_EXT_external_memory_dma_buf[[:space:]]*:/ {dmabuf=1}
	inext&&/^[[:space:]]*VK_EXT_image_drm_format_modifier[[:space:]]*:/ {mod=1}
	inext&&/^[[:space:]]*VK_EXT_queue_family_foreign[[:space:]]*:/ {foreign=1}
	inext&&/^[[:space:]]*VK_EXT_robustness2[[:space:]]*:/ {robust=1}
	inext&&/^[[:space:]]*VK_(KHR|EXT)_global_priority[[:space:]]*:/ {prio=1}
	/^[[:space:]]*queueProperties\[[0-9]+\]:/ {commitq();inq=1;qf="";qc=qt="-";next}
	inq&&k=="queueFlags" {qf=$2}
	inq&&k=="queueCount" {qc=$2}
	inq&&k=="timestampValidBits" {qt=$2}
	k=="timelineSemaphore"&&$2=="true" {timeline=1}
	k=="shaderStorageImageWriteWithoutFormat"&&$2=="true" {write=1}
	k=="shaderFloat16"&&$2=="true" {fp16=1}
	k=="shaderStorageImageExtendedFormats"&&$2=="true" {extfmt=1}
	END {flush()}
	' "$DOCTOR_TMP/vulkaninfo" >"$DOCTOR_TMP/gpus"
	while IFS=$'\t' read -r idx api drv vendor device type name mem dmabuf modifier foreign robust timeline write fp16 extfmt lds queues timestamps priority; do
		[[ -n "$idx" ]] || continue
		G_API[idx]=$api; G_DRIVER[idx]=$drv; G_VENDOR[idx]=${vendor#0x}; G_DEVICE[idx]=${device#0x}
		G_TYPE[idx]=$type; G_NAME[idx]=$name; G_MEMFD[idx]=$mem; G_DMABUF[idx]=$dmabuf
		G_MODIFIER[idx]=$modifier; G_FOREIGN[idx]=$foreign; G_ROBUST[idx]=$robust
		G_TIMELINE[idx]=$timeline; G_WRITE[idx]=$write; G_FP16[idx]=$fp16; G_EXTFMT[idx]=$extfmt
		G_LDS[idx]=$lds; G_QCOUNT[idx]=$queues; G_TIMESTAMP[idx]=$timestamps; G_PRIORITY[idx]=$priority
		(( GPU_COUNT++ ))
	done <"$DOCTOR_TMP/gpus"
	(( GPU_COUNT > 0 )) && VULKAN_OK=1
	if (( ! VULKAN_OK )); then
		printf 'FAIL  vulkaninfo found no physical devices (exit %d)\n' "$VULKAN_RC"
	fi
else
	printf 'FAIL  vulkaninfo missing; install vulkan-tools to check runtime\n'
fi

declare -a S_CARD S_ID S_CONNECTORS
SYS_COUNT=0
for CARD_PATH in /sys/class/drm/card*; do
	[[ -e "$CARD_PATH" && ${CARD_PATH##*/} =~ ^card[0-9]+$ ]] || continue
	[[ -r "$CARD_PATH/device/vendor" && -r "$CARD_PATH/device/device" ]] || continue
	read -r SYS_VENDOR <"$CARD_PATH/device/vendor"
	read -r SYS_DEVICE <"$CARD_PATH/device/device"
	S_CARD[SYS_COUNT]=${CARD_PATH##*/}
	S_ID[SYS_COUNT]="${SYS_VENDOR#0x}:${SYS_DEVICE#0x}"
	CONNS=""
	for STATUS in "$CARD_PATH"/"${CARD_PATH##*/}"-*/status; do
		[[ -r "$STATUS" ]] || continue
		read -r STATE <"$STATUS"
		[[ "$STATE" == connected ]] || continue
		CONNECTOR=${STATUS%/status}; CONNECTOR=${CONNECTOR##*/}
		CONNS+="${CONNS:+,}${CONNECTOR#"${CARD_PATH##*/}"-}"
	done
	S_CONNECTORS[SYS_COUNT]=${CONNS:--}
	(( SYS_COUNT++ ))
done

gpu_for_id() {
	local wanted=$1 i
	for ((i=0;i<GPU_COUNT;i++)); do
		[[ "${G_VENDOR[i]}:${G_DEVICE[i]}" == "$wanted" ]] && { printf '%s' "$i"; return; }
	done
	printf '%s' -1
}
add_missing() { local -n list=$1; list+=("$2"); }
report_gpu() {
	local i=$1 card=$2 attached=$3 connectors=$4 missing=() present_missing=() optional=()
	printf '%s  %s\n' "$card" "${G_VENDOR[i]}:${G_DEVICE[i]}"
	printf '      %s | driver %s | apiVersion %s | deviceType %s\n' "${G_NAME[i]}" "${G_DRIVER[i]}" "${G_API[i]}" "${G_TYPE[i]}"
	if (( attached )); then pass "display: $connectors (must be PRESENT)"; else printf '      display: none (RENDER candidate)\n'; fi
	version_ge "${G_API[i]}" 1.2 || add_missing missing 'Vulkan >= 1.2'
	[[ ${G_MEMFD[i]} == 1 ]] || add_missing missing VK_KHR_external_memory_fd
	[[ ${G_DMABUF[i]} == 1 ]] || add_missing missing VK_EXT_external_memory_dma_buf
	[[ ${G_TIMELINE[i]} == 1 ]] || add_missing missing timelineSemaphore
	[[ ${G_TYPE[i]} != PHYSICAL_DEVICE_TYPE_CPU ]] || add_missing missing 'hardware GPU'
	if (( ${#missing[@]} )); then
		printf 'FAIL  RENDER role: missing %s\n' "$(IFS=', '; echo "${missing[*]}")"
		G_RENDER_OK[i]=0
	else
		pass "RENDER role: Vulkan/dma-buf export/timeline ready"
		G_RENDER_OK[i]=1
	fi
	((${#missing[@]})) && present_missing+=("${missing[@]}")
	[[ ${G_ROBUST[i]} == 1 ]] || add_missing present_missing VK_EXT_robustness2
	[[ ${G_WRITE[i]} == 1 ]] || add_missing present_missing shaderStorageImageWriteWithoutFormat
	if (( ${#present_missing[@]} )); then
		printf 'FAIL  PRESENT role: missing %s\n' "$(IFS=', '; echo "${present_missing[*]}")"
		G_PRESENT_OK[i]=0
	else
		pass "PRESENT role: required extensions/features ready"
		G_PRESENT_OK[i]=1
	fi
	if [[ ${G_MODIFIER[i]} != 1 || ${G_FOREIGN[i]} != 1 ]]; then
		warn "legacy EXTERNAL fallback; cross-vendor unreliable (need DRM modifier + foreign queue)"
	else
		pass "PRESENT dma-buf path: DRM modifiers + foreign queue"
	fi
	if [[ ! ${G_QCOUNT[i]} =~ ^[0-9]+$ ]] || (( G_QCOUNT[i] < 2 )); then
		warn "compute queueCount ${G_QCOUNT[i]}; shared-queue fallback, reduced modes (no JIT-era dedicated pacing)"
	else
		pass "compute queueCount ${G_QCOUNT[i]}; dedicated pacing queue ready"
	fi
	if [[ ! ${G_TIMESTAMP[i]} =~ ^[0-9]+$ ]] || (( G_TIMESTAMP[i] == 0 )); then
		warn "timestampValidBits ${G_TIMESTAMP[i]}; no live GPU-time ladder"
	else
		pass "timestampValidBits ${G_TIMESTAMP[i]}; live GPU-time ladder ready"
	fi
	if version_ge "${G_API[i]}" 1.3; then
		pass "Vulkan ${G_API[i]}; sync2 tracker path"
	else
		warn "Vulkan ${G_API[i]}; legacy barrier path instead of sync2 tracker"
	fi
	[[ ${G_FP16[i]} == 1 ]] || add_missing optional 'shaderFloat16 (no fp16 variants)'
	[[ ${G_EXTFMT[i]} == 1 ]] || add_missing optional 'shaderStorageImageExtendedFormats (no compact R16F luma)'
	if [[ ! ${G_LDS[i]} =~ ^[0-9]+$ ]] || (( G_LDS[i] < 27904 )); then
		add_missing optional 'shared memory <27904 (net disabled; Stage B still works)'
	fi
	[[ ${G_PRIORITY[i]} == 1 ]] || add_missing optional 'KHR/EXT_global_priority (no realtime queue under CAP_SYS_NICE)'
	if (( ${#optional[@]} )); then warn "optional: $(IFS='; '; echo "${optional[*]}")"; else pass "optional: fp16, R16F luma, learned net, global priority"; fi
	printf '\n'
}

declare -a SEEN_GPU
for ((s=0;s<SYS_COUNT;s++)); do
	i=$(gpu_for_id "${S_ID[s]}")
	if (( i < 0 )); then
		printf '%s  %s\n' "${S_CARD[s]}" "${S_ID[s]}"
		[[ ${S_CONNECTORS[s]} == - ]] || pass "display: ${S_CONNECTORS[s]} (must be PRESENT)"
		printf 'FAIL  Vulkan device not enumerated; check driver and /dev/dri access\n\n'
		continue
	fi
	SEEN_GPU[i]=1
	G_ATTACHED[i]=0; [[ ${S_CONNECTORS[s]} != - ]] && G_ATTACHED[i]=1
	report_gpu "$i" "${S_CARD[s]}" "${G_ATTACHED[i]}" "${S_CONNECTORS[s]}"
done
for ((i=0;i<GPU_COUNT;i++)); do
	[[ ${SEEN_GPU[i]:-0} == 1 ]] && continue
	G_ATTACHED[i]=0
	report_gpu "$i" "GPU$i (no DRM card mapping)" 0 -
done
(( SYS_COUNT > 0 )) || printf 'FAIL  no DRM cards visible under /sys/class/drm\n\n'

printf 'TOPOLOGY\n'
PRESENT=-1
for ((i=0;i<GPU_COUNT;i++)); do
	if [[ ${G_ATTACHED[i]:-0} == 1 && ${G_PRESENT_OK[i]:-0} == 1 ]]; then PRESENT=$i; break; fi
done
RENDER=-1; BEST=-1
if (( PRESENT >= 0 )); then
	for ((i=0;i<GPU_COUNT;i++)); do
		[[ ${G_RENDER_OK[i]:-0} == 1 && i -ne PRESENT ]] || continue
		SCORE=1; [[ ${G_TYPE[i]} == PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU ]] && SCORE=2
		[[ ${G_TYPE[i]} == PHYSICAL_DEVICE_TYPE_DISCRETE_GPU ]] && SCORE=3
		(( SCORE > BEST )) && { BEST=$SCORE; RENDER=$i; }
	done
fi
DUAL_OK=0; SINGLE_OK=0
if (( PRESENT >= 0 )); then
	SINGLE_OK=1
	if (( RENDER >= 0 )); then
		DUAL_OK=1
		pass "dual-GPU route ready"
		printf '      export RENDER_DEV=%s:%s!\n' "${G_VENDOR[RENDER]}" "${G_DEVICE[RENDER]}"
		printf '      export PRESENT_DEV=%s:%s\n' "${G_VENDOR[PRESENT]}" "${G_DEVICE[PRESENT]}"
	else
		RENDER=$PRESENT
		warn "single-GPU only: works, but generation competes with the game (see README's one-GPU exception)"
		printf '      export RENDER_DEV=%s:%s!\n' "${G_VENDOR[RENDER]}" "${G_DEVICE[RENDER]}"
		printf '      export PRESENT_DEV=%s:%s\n' "${G_VENDOR[PRESENT]}" "${G_DEVICE[PRESENT]}"
	fi
else
	printf 'FAIL  no display-attached GPU satisfies the PRESENT role\n'
fi

printf '\nVERDICT\n'
(( BUILD_OK )) && pass "can build: yes" || printf 'FAIL  can build: no\n'
(( DUAL_OK )) && pass "can run dual-GPU: yes" || printf '%s  can run dual-GPU: no\n' "$([[ $SINGLE_OK == 1 ]] && echo WARN || echo FAIL)"
(( SINGLE_OK )) && pass "can run single-GPU: yes" || printf 'FAIL  can run single-GPU: no\n'
if (( ! BUILD_OK )); then
	printf 'FAIL  fix first: %s\n' "$FIRST_FIX"
elif (( PRESENT < 0 )); then
	printf 'FAIL  fix first: restore Vulkan driver/device access on the display GPU\n'
elif (( ! DUAL_OK )); then
	printf 'WARN  fix first: add or enable a second dma-buf-capable Vulkan GPU for isolated generation\n'
else
	pass "fix first: nothing required"
fi

if (( ! BUILD_OK || ! SINGLE_OK )); then HARD_FAILURE=1; fi
(( HARD_FAILURE )) && exit 2
(( WARNINGS )) && exit 1
exit 0
