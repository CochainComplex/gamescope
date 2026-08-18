// Initialize Vulkan and composite stuff with a compute queue

#include <cassert>
#include <cerrno>
#include <fcntl.h>
#include <random>
#include <unistd.h>
#include <stdio.h>
#include <string.h>
#include <inttypes.h>
#include <sys/stat.h>
#include <algorithm>
#include <array>
#include <bitset>
#include <cmath>
#include <limits>
#include <thread>
#include <dlfcn.h>
#include "vulkan_include.h"
#include "Utils/Algorithm.h"

#if defined(__linux__)
#include <sys/sysmacros.h>
#endif

// Used to remove the config struct alignment specified by the NIS header
#define NIS_ALIGNED(x)
// NIS_Config needs to be included before the X11 headers because of conflicting defines introduced by X11
#include "shaders/NVIDIAImageScaling/NIS/NIS_Config.h"

#include <drm_fourcc.h>
#include "hdmi.h"
#if HAVE_DRM
#include "drm_include.h"
#endif
#include "wlr_begin.hpp"
#include <wlr/render/drm_format_set.h>
#include "wlr_end.hpp"

#include "rendervulkan.hpp"
#include "main.hpp"
#include "steamcompmgr.hpp"
#include "vblankmanager.hpp"
#include "log.hpp"
#include "Utils/Process.h"
#include "framegen/adaptation.hpp"
#include "framegen/atomic_file.hpp"
#include "framegen/deadline.hpp"
#include "framegen/dispatch_policy.hpp"
#include "framegen/hud.hpp"
#include "framegen/metrics.hpp"
#include "framegen/net_layout.hpp"
#include "framegen/net_profile.hpp"
#include "framegen/policy.hpp"
#include "framegen/push_constants.hpp"
#include "framegen/scheduling.hpp"
#include "framegen/settings.hpp"
#include "framegen/temporal.hpp"
#include "GamescopeVersion.h"

#include "cs_composite_blit.h"
#include "cs_composite_blur.h"
#include "cs_composite_blur_cond.h"
#include "cs_composite_rcas.h"
#include "cs_easu.h"
#include "cs_easu_fp16.h"
#include "cs_framegen_blend.h"
#include "cs_framegen_extrapolate.h"
#include "cs_framegen_extrapolate_direct.h"
#include "cs_framegen_extrapolate_fp16.h"
#include "cs_framegen_extrapolate_pair.h"
#include "cs_framegen_extrapolate_pair_fp16.h"
#include "cs_framegen_hud.h"
#include "cs_framegen_motion_luma_pair.h"
#include "cs_framegen_motion_luma_pair_rgba.h"
#include "cs_framegen_motion_pyramid.h"
#include "cs_framegen_motion_pyramid_rgba.h"
#include "cs_framegen_motion_match.h"
#include "cs_framegen_motion_match_refine.h"
#include "cs_framegen_motion_fbcheck.h"
#include "cs_framegen_motion_warp.h"
#include "cs_framegen_motion_warp_accel.h"
#include "cs_framegen_motion_bidir.h"
#include "cs_framegen_motion_stats.h"
#include "cs_framegen_motion_stats_apply.h"
#include "cs_framegen_motion_net.h"
#include "cs_framegen_motion_net_train.h"
#include "cs_framegen_motion_net_opt.h"
#include "cs_gaussian_blur_horizontal.h"
#include "cs_nis.h"
#include "cs_nis_fp16.h"
#include "cs_rgb_to_nv12.h"

#define A_CPU
#include "shaders/ffx_a.h"
#include "shaders/ffx_fsr1.h"

#include "reshade_effect_manager.hpp"

extern bool g_bWasPartialComposite;
extern bool g_bAllowDeferredBackend;

namespace
{

constexpr size_t k_uInitialSubmissionCapacity = 16;
constexpr size_t k_uInitialTrackedTextureCapacity = VKR_SAMPLER_SLOTS * 4 + VKR_TARGET_SLOTS;

// Vulkan submits normally carry only the compositor timeline, plus at most one
// explicit-sync acquire/release point. Keep that path off the allocator while
// retaining an unbounded fallback for future multi-semaphore callers.
template<typename T, size_t InlineCapacity>
class InlineSubmitArray
{
public:
	T *storage( size_t count )
	{
		if ( count <= InlineCapacity ) [[likely]]
			return m_inline.data();
		m_overflow = std::make_unique_for_overwrite<T[]>( count );
		return m_overflow.get();
	}

private:
	std::array<T, InlineCapacity> m_inline;
	std::unique_ptr<T[]> m_overflow;
};

} // namespace

static uint32_t framegen_hud_level()
{
	static const uint32_t s_uLevel = []()
	{
		const auto parsed = gamescope::framegen::parse_uint32_setting(
			getenv( "GAMESCOPE_FRAMEGEN_HUD" ), true );
		return parsed.has_value() && ( *parsed == 1u || *parsed == 2u )
			? *parsed : 0u;
	}();
	return s_uLevel;
}

enum class FramegenHudCorner : uint32_t
{
	TopLeft,
	TopRight,
	BottomLeft,
	BottomRight,
	BottomCenter,
};

static uint32_t framegen_hud_scale( uint32_t uOutputHeight )
{
	static const uint32_t s_uOverride = []()
	{
		const auto parsed = gamescope::framegen::parse_uint32_setting(
			getenv( "GAMESCOPE_FRAMEGEN_HUD_SCALE" ), false );
		return parsed.has_value() && *parsed >= 1u && *parsed <= 6u
			? *parsed : 0u;
	}();
	return s_uOverride != 0u ? s_uOverride
		: std::clamp( uOutputHeight / 720u, 2u, 4u );
}

static FramegenHudCorner framegen_hud_corner()
{
	static const FramegenHudCorner s_eCorner = []()
	{
		const char *pszCorner = getenv( "GAMESCOPE_FRAMEGEN_HUD_CORNER" );
		if ( pszCorner != nullptr && strcmp( pszCorner, "tl" ) == 0 )
			return FramegenHudCorner::TopLeft;
		if ( pszCorner != nullptr && strcmp( pszCorner, "tr" ) == 0 )
			return FramegenHudCorner::TopRight;
		if ( pszCorner != nullptr && strcmp( pszCorner, "bl" ) == 0 )
			return FramegenHudCorner::BottomLeft;
		if ( pszCorner != nullptr && strcmp( pszCorner, "br" ) == 0 )
			return FramegenHudCorner::BottomRight;
		return FramegenHudCorner::BottomCenter;
	}();
	return s_eCorner;
}

static constexpr size_t k_nFramegenHudUploadSlots = 2u;
static constexpr size_t k_nFramegenHudPersistentBytes =
	k_nFramegenHudUploadSlots * sizeof( gamescope::framegen::FramegenHudUniform_t );
static constexpr size_t k_nFramegenHudPersistentPad =
	( ( k_nFramegenHudPersistentBytes + 4095u ) / 4096u ) * 4096u;
static_assert( k_nFramegenHudPersistentBytes == 3'168u );
static_assert( k_nFramegenHudPersistentPad == 4'096u );
static_assert( CVulkanDevice::upload_buffer_persistent_pad
	>= k_nFramegenHudPersistentPad );
struct FramegenHudGpuState_t
{
	struct UploadSlot_t
	{
		void *pMapped = nullptr;
		uint32_t offset = 0u;
		uint64_t lastSubmitSeqNo = 0u;
		uint32_t logicalWidthPixels = 0u;
		uint32_t logicalHeightPixels = 0u;
	};

	std::array<UploadSlot_t, k_nFramegenHudUploadSlots> uploadSlots;
	gamescope::framegen::FramegenHudText_t text;
	uint64_t nextRebuildNs = 0u;
	int activeUploadSlot = -1;
	bool textValid = false;
	bool dirty = false;
	bool hdr = false;
	bool hdrValid = false;
};
static FramegenHudGpuState_t g_framegenHud;

static void framegen_hud_init_upload_buffer( CVulkanDevice *pDevice )
{
	if ( framegen_hud_level() == 0u )
		return;
	for ( auto &slot : g_framegenHud.uploadSlots )
	{
		const auto [pMapped, offset] = pDevice->reservePersistentUniformBufferData(
			sizeof( gamescope::framegen::FramegenHudUniform_t ) );
		slot.pMapped = pMapped;
		slot.offset = offset;
	}
}

static int framegen_hud_record( CVulkanCmdBuffer *pCmdBuffer,
	const gamescope::Rc<CVulkanTexture> &pTarget, const struct FrameInfo_t *pFrameInfo );
static void framegen_hud_note_submit( int nSlot, uint64_t ulSeqNo );

static bool framegen_backend_supported()
{
	return GetBackend() != nullptr && GetBackend()->SupportsFramegen();
}

static constexpr mat3x4 g_rgb2yuv_srgb_to_bt601_limited = {{
  { 0.257f, 0.504f, 0.098f, 0.0625f },
  { -0.148f, -0.291f, 0.439f, 0.5f },
  { 0.439f, -0.368f, -0.071f, 0.5f },
}};

static constexpr mat3x4 g_rgb2yuv_srgb_to_bt601 = {{
  { 0.299f, 0.587f, 0.114f, 0.0f },
  { -0.169f, -0.331f, 0.500f, 0.5f },
  { 0.500f, -0.419f, -0.081f, 0.5f },
}};

static constexpr mat3x4 g_rgb2yuv_srgb_to_bt709_limited = {{
  { 0.1826f, 0.6142f, 0.0620f, 0.0625f },
  { -0.1006f, -0.3386f, 0.4392f, 0.5f },
  { 0.4392f, -0.3989f, -0.0403f, 0.5f },
}};

static constexpr mat3x4 g_rgb2yuv_srgb_to_bt709_full = {{
  { 0.2126f, 0.7152f, 0.0722f, 0.0f },
  { -0.1146f, -0.3854f, 0.5000f, 0.5f },
  { 0.5000f, -0.4542f, -0.0458f, 0.5f },
}};

static const mat3x4& colorspace_to_conversion_from_srgb_matrix(EStreamColorspace colorspace) {
	switch (colorspace) {
		default:
		case k_EStreamColorspace_BT601:			return g_rgb2yuv_srgb_to_bt601_limited;
		case k_EStreamColorspace_BT601_Full:	return g_rgb2yuv_srgb_to_bt601;
		case k_EStreamColorspace_BT709:			return g_rgb2yuv_srgb_to_bt709_limited;
		case k_EStreamColorspace_BT709_Full:	return g_rgb2yuv_srgb_to_bt709_full;
	}
}

PFN_vkGetInstanceProcAddr g_pfn_vkGetInstanceProcAddr;
PFN_vkCreateInstance g_pfn_vkCreateInstance;

static VkResult vulkan_load_module()
{
	static VkResult s_result = []()
	{
		void* pModule = dlopen( "libvulkan.so.1", RTLD_NOW | RTLD_LOCAL );
		if ( !pModule )
			pModule = dlopen( "libvulkan.so", RTLD_NOW | RTLD_LOCAL );
		if ( !pModule )
			return VK_ERROR_INITIALIZATION_FAILED;

		g_pfn_vkGetInstanceProcAddr = (PFN_vkGetInstanceProcAddr)dlsym( pModule, "vkGetInstanceProcAddr" );
		if ( !g_pfn_vkGetInstanceProcAddr )
			return VK_ERROR_INITIALIZATION_FAILED;

		g_pfn_vkCreateInstance = (PFN_vkCreateInstance) g_pfn_vkGetInstanceProcAddr( nullptr, "vkCreateInstance" );
		if ( !g_pfn_vkCreateInstance )
			return VK_ERROR_INITIALIZATION_FAILED;

		return VK_SUCCESS;
	}();

	return s_result;
}

VulkanOutput_t g_output;

// Size of the compute push-constant range reserved for frame generation. Large
// enough for the extrapolate params and the motion-pass params, well under the
// Vulkan-guaranteed 128-byte minimum maxPushConstantsSize.
static constexpr uint32_t k_uFramegenPushConstantSize = 64;

// Shared-memory requirements for cs_framegen_motion_net.comp and
// cs_framegen_motion_net_train.comp; these byte counts must track their LDS
// declarations.
static constexpr uint32_t k_uFramegenMotionNetLdsBytes = 27'904;
static constexpr uint32_t k_uFramegenMotionNetTrainLdsBytes = 27'136;

static bool framegen_net_lds_supported()
{
	return g_device.maxComputeSharedMemorySize() >= k_uFramegenMotionNetLdsBytes
		&& g_device.maxComputeSharedMemorySize() >= k_uFramegenMotionNetTrainLdsBytes;
}

uint32_t g_uCompositeDebug = 0u;
gamescope::ConVar<uint32_t> cv_composite_debug{ "composite_debug", 0, "Debug composition flags" };

static std::map< VkFormat, std::map< uint64_t, VkDrmFormatModifierPropertiesEXT > > DRMModifierProps = {};
static std::unordered_map<uint32_t, std::vector<uint64_t>> s_SampledModifierFormats = {};
static struct wlr_drm_format_set sampledShmFormats = {};
static struct wlr_drm_format_set sampledDRMFormats = {};

std::span<const uint64_t> GetSupportedSampleModifiers( uint32_t uDrmFormat )
{
	auto iter = s_SampledModifierFormats.find( uDrmFormat );
	if ( iter == s_SampledModifierFormats.end() )
		return std::span<const uint64_t>{};

	return std::span<const uint64_t>{ iter->second.begin(), iter->second.end() };
}

static LogScope vk_log("vulkan");

static void vk_errorf(VkResult result, const char *fmt, ...) {
	static char buf[1024];
	va_list args;
	va_start(args, fmt);
	vsnprintf(buf, sizeof(buf), fmt, args);
	va_end(args);

	vk_log.errorf("%s (VkResult: %d)", buf, result);
}

[[noreturn]] void vulkan_check_fatal( VkResult result, const char *expression )
{
	vk_errorf( result, "%s failed!", expression );
	abort();
}

static const char *vk_device_type_name( VkPhysicalDeviceType eType )
{
	switch ( eType )
	{
		case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU: return "integrated";
		case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU:  return "discrete";
		case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU:   return "virtual";
		case VK_PHYSICAL_DEVICE_TYPE_CPU:           return "cpu";
		default:                                    return "other";
	}
}

static const char *vk_image_tiling_name( VkImageTiling eTiling )
{
	switch ( eTiling )
	{
		case VK_IMAGE_TILING_OPTIMAL:                 return "optimal";
		case VK_IMAGE_TILING_LINEAR:                  return "linear";
		case VK_IMAGE_TILING_DRM_FORMAT_MODIFIER_EXT: return "drm_format_modifier";
		default:                                      return "unknown";
	}
}

#if HAVE_DRM
static void debug_log_drm_device( const char *pszPrefix, drmDevice *pDrmDevice )
{
	if ( !g_bDebugDualGpuRoute || !pDrmDevice )
		return;

	const char *pszPrimaryNode = ( pDrmDevice->available_nodes & ( 1 << DRM_NODE_PRIMARY ) )
		? pDrmDevice->nodes[ DRM_NODE_PRIMARY ]
		: "(none)";
	const char *pszRenderNode = ( pDrmDevice->available_nodes & ( 1 << DRM_NODE_RENDER ) )
		? pDrmDevice->nodes[ DRM_NODE_RENDER ]
		: "(none)";

	if ( pDrmDevice->bustype == DRM_BUS_PCI && pDrmDevice->businfo.pci )
	{
		drmPciBusInfoPtr pPci = pDrmDevice->businfo.pci;
		vk_log.infof( "dual-gpu-route: %s DRM device pci %04x:%02x:%02x.%u primary %s render %s",
			pszPrefix,
			unsigned( pPci->domain ),
			unsigned( pPci->bus ),
			unsigned( pPci->dev ),
			unsigned( pPci->func ),
			pszPrimaryNode,
			pszRenderNode );
	}
	else
	{
		vk_log.infof( "dual-gpu-route: %s DRM device bus type %d primary %s render %s",
			pszPrefix,
			pDrmDevice->bustype,
			pszPrimaryNode,
			pszRenderNode );
	}
}
#endif

// For when device is up and it would be totally fatal to fail
#define vk_check( x ) \
	do \
	{ \
		VkResult check_res = VK_SUCCESS; \
		if ( ( check_res = ( x ) ) != VK_SUCCESS ) \
		{ \
			vulkan_check_fatal( check_res, #x ); \
		} \
	} while ( 0 )

template<typename Target, typename Base>
Target *pNextFind(const Base *base, VkStructureType sType)
{
	for ( ; base; base = (const Base *)base->pNext )
	{
		if (base->sType == sType)
			return (Target *) base;
	}
	return nullptr;
}

#define VK_STRUCTURE_TYPE_WSI_IMAGE_CREATE_INFO_MESA (VkStructureType)1000001002
#define VK_STRUCTURE_TYPE_WSI_MEMORY_ALLOCATE_INFO_MESA (VkStructureType)1000001003

struct wsi_image_create_info {
	VkStructureType sType;
	const void *pNext;
	bool scanout;

	uint32_t modifier_count;
	const uint64_t *modifiers;
};

struct wsi_memory_allocate_info {
    VkStructureType sType;
    const void *pNext;
    bool implicit_sync;
};

// DRM doesn't always have 32bit floating point formats, so add our own if necessary

#ifndef DRM_FORMAT_ABGR32323232F
#define DRM_FORMAT_ABGR32323232F fourcc_code('A', 'B', '8', 'F')
#endif

#ifndef DRM_FORMAT_R16F
#define DRM_FORMAT_R16F fourcc_code('R', '1', '6', 'F')
#endif

#ifndef DRM_FORMAT_R32F
#define DRM_FORMAT_R32F fourcc_code('R', '3', '2', 'F')
#endif

// Internal-only, for the framegen adaptation stats image: R32_UINT is the one
// format Vulkan guarantees storage-image atomics on.
#ifndef DRM_FORMAT_R32UI
#define DRM_FORMAT_R32UI fourcc_code('R', '3', '2', 'U')
#endif

struct {
	uint32_t DRMFormat;
	VkFormat vkFormat;
	VkFormat vkFormatSrgb;
	uint32_t bpp;
	bool bHasAlpha;
	bool internal;
} s_DRMVKFormatTable[] = {
	{ DRM_FORMAT_ARGB8888, VK_FORMAT_B8G8R8A8_UNORM, VK_FORMAT_B8G8R8A8_SRGB, 4, true, false },
	{ DRM_FORMAT_XRGB8888, VK_FORMAT_B8G8R8A8_UNORM, VK_FORMAT_B8G8R8A8_SRGB, 4, false, false },
	{ DRM_FORMAT_ABGR8888, VK_FORMAT_R8G8B8A8_UNORM, VK_FORMAT_R8G8B8A8_SRGB, 4, true, false },
	{ DRM_FORMAT_XBGR8888, VK_FORMAT_R8G8B8A8_UNORM, VK_FORMAT_R8G8B8A8_SRGB, 4, false, false },
	{ DRM_FORMAT_RGB565, VK_FORMAT_R5G6B5_UNORM_PACK16, VK_FORMAT_R5G6B5_UNORM_PACK16, 2, false, false },
	{ DRM_FORMAT_NV12, VK_FORMAT_G8_B8R8_2PLANE_420_UNORM, VK_FORMAT_G8_B8R8_2PLANE_420_UNORM, 0, false, false },
	{ DRM_FORMAT_ABGR16161616F, VK_FORMAT_R16G16B16A16_SFLOAT, VK_FORMAT_R16G16B16A16_SFLOAT, 8, true, false },
	{ DRM_FORMAT_XBGR16161616F, VK_FORMAT_R16G16B16A16_SFLOAT, VK_FORMAT_R16G16B16A16_SFLOAT, 8, false, false },
	{ DRM_FORMAT_ABGR16161616, VK_FORMAT_R16G16B16A16_UNORM, VK_FORMAT_R16G16B16A16_UNORM, 8, true, false },
	{ DRM_FORMAT_XBGR16161616, VK_FORMAT_R16G16B16A16_UNORM, VK_FORMAT_R16G16B16A16_UNORM, 8, false, false },
	{ DRM_FORMAT_ABGR2101010, VK_FORMAT_A2B10G10R10_UNORM_PACK32, VK_FORMAT_A2B10G10R10_UNORM_PACK32, 4, true, false },
	{ DRM_FORMAT_XBGR2101010, VK_FORMAT_A2B10G10R10_UNORM_PACK32, VK_FORMAT_A2B10G10R10_UNORM_PACK32, 4, false, false },
	{ DRM_FORMAT_ARGB2101010, VK_FORMAT_A2R10G10B10_UNORM_PACK32, VK_FORMAT_A2R10G10B10_UNORM_PACK32, 4, true, false },
	{ DRM_FORMAT_XRGB2101010, VK_FORMAT_A2R10G10B10_UNORM_PACK32, VK_FORMAT_A2R10G10B10_UNORM_PACK32, 4, false, false },

	{ DRM_FORMAT_R8, VK_FORMAT_R8_UNORM, VK_FORMAT_R8_UNORM, 1, false, true },
	{ DRM_FORMAT_R16, VK_FORMAT_R16_UNORM, VK_FORMAT_R16_UNORM, 2, false, true },
	{ DRM_FORMAT_GR88, VK_FORMAT_R8G8_UNORM, VK_FORMAT_R8G8_UNORM, 2, false, true },
	{ DRM_FORMAT_GR1616, VK_FORMAT_R16G16_UNORM, VK_FORMAT_R16G16_UNORM, 4, false, true },
	{ DRM_FORMAT_ABGR32323232F, VK_FORMAT_R32G32B32A32_SFLOAT, VK_FORMAT_R32G32B32A32_SFLOAT, 16,true, true },
	{ DRM_FORMAT_R16F, VK_FORMAT_R16_SFLOAT, VK_FORMAT_R16_SFLOAT, 2, false, true },
	{ DRM_FORMAT_R32F, VK_FORMAT_R32_SFLOAT, VK_FORMAT_R32_SFLOAT, 4, false, true },
	{ DRM_FORMAT_R32UI, VK_FORMAT_R32_UINT, VK_FORMAT_R32_UINT, 4, false, true },
	{ DRM_FORMAT_INVALID, VK_FORMAT_UNDEFINED, VK_FORMAT_UNDEFINED, false, true },
};

uint32_t VulkanFormatToDRM( VkFormat vkFormat, std::optional<bool> obHasAlphaOverride )
{
	for ( int i = 0; s_DRMVKFormatTable[i].vkFormat != VK_FORMAT_UNDEFINED; i++ )
	{
		if ( ( s_DRMVKFormatTable[i].vkFormat == vkFormat || s_DRMVKFormatTable[i].vkFormatSrgb == vkFormat ) && ( !obHasAlphaOverride || s_DRMVKFormatTable[i].bHasAlpha == *obHasAlphaOverride ) )
		{
			return s_DRMVKFormatTable[i].DRMFormat;
		}
	}
	
	return DRM_FORMAT_INVALID;
}

VkFormat DRMFormatToVulkan( uint32_t nDRMFormat, bool bSrgb )
{
	for ( int i = 0; s_DRMVKFormatTable[i].vkFormat != VK_FORMAT_UNDEFINED; i++ )
	{
		if ( s_DRMVKFormatTable[i].DRMFormat == nDRMFormat )
		{
			return bSrgb ? s_DRMVKFormatTable[i].vkFormatSrgb : s_DRMVKFormatTable[i].vkFormat;
		}
	}
	
	return VK_FORMAT_UNDEFINED;
}

bool DRMFormatHasAlpha( uint32_t nDRMFormat )
{
	for ( int i = 0; s_DRMVKFormatTable[i].vkFormat != VK_FORMAT_UNDEFINED; i++ )
	{
		if ( s_DRMVKFormatTable[i].DRMFormat == nDRMFormat )
		{
			return s_DRMVKFormatTable[i].bHasAlpha;
		}
	}
	
	return false;
}

uint32_t DRMFormatGetBPP( uint32_t nDRMFormat )
{
	for ( int i = 0; s_DRMVKFormatTable[i].vkFormat != VK_FORMAT_UNDEFINED; i++ )
	{
		if ( s_DRMVKFormatTable[i].DRMFormat == nDRMFormat )
		{
			return s_DRMVKFormatTable[i].bpp;
		}
	}

	return false;
}

bool CVulkanDevice::BInit(VkInstance instance, VkSurfaceKHR surface)
{
	assert(instance);
	assert(!m_bInitialized);

	g_output.surface = surface;

	m_instance = instance;
	#define VK_FUNC(x) vk.x = (PFN_vk##x) g_pfn_vkGetInstanceProcAddr(instance, "vk"#x);
	VULKAN_INSTANCE_FUNCTIONS
	#undef VK_FUNC

	if (!selectPhysDev(surface))
		return false;
	if (!createDevice())
		return false;
	if (!createLayouts())
		return false;
	if (!createPools())
		return false;
	if (!createShaders())
		return false;
	if (!createScratchResources())
		return false;
	framegen_hud_init_upload_buffer( this );

	m_bInitialized = true;
	m_unusedCmdBufs.reserve( k_uInitialSubmissionCapacity );
	m_pendingCmdBufs.reserve( k_uInitialSubmissionCapacity );
	m_pendingFramegenCmdBufs.reserve( k_uInitialSubmissionCapacity );
	m_framegenQuerySlotBySeqNo.reserve( k_uInitialSubmissionCapacity );

	// Frame generation is opt-in, so pay its fixed pipeline creation cost at
	// compositor startup instead of on the first generated frame. A lazy miss in
	// pipeline() calls vkCreateComputePipelines on the compositor thread and is a
	// visible first-use hitch, especially for the larger ML shaders.
	if ( vulkan_framegen_is_enabled() )
		compileFramegenPipelines();

	std::thread piplelineThread([this](){compileAllPipelines();});
	piplelineThread.detach();

	g_reshadeManager.init(this);

	return true;
}

extern bool env_to_bool(const char *env);

static const char *vulkan_queue_family_quirk_force_general( const VkPhysicalDeviceProperties &props )
{
	struct VendorQueueQuirk_t
	{
		uint32_t vendorID;
		const char *reason;
	};

	static constexpr VendorQueueQuirk_t s_Quirks[] = {
		{
			0x8086,
			"Intel compute-only queue interop performance quirk (drm/xe#4452)",
		},
	};

	for ( const VendorQueueQuirk_t &quirk : s_Quirks )
	{
		if ( props.vendorID == quirk.vendorID )
			return quirk.reason;
	}

	return nullptr;
}

bool CVulkanDevice::selectPhysDev(VkSurfaceKHR surface)
{
	uint32_t deviceCount = 0;
	vk.EnumeratePhysicalDevices(instance(), &deviceCount, nullptr);
	std::vector<VkPhysicalDevice> physDevs(deviceCount);
	vk.EnumeratePhysicalDevices(instance(), &deviceCount, physDevs.data());
	if (deviceCount < physDevs.size())
		physDevs.resize(deviceCount);

	bool bTryComputeOnly = true;

	// In theory vkBasalt might want to filter out compute-only queue families to force our hand here
	const char *pchEnableVkBasalt = getenv( "ENABLE_VKBASALT" );
	if ( pchEnableVkBasalt != nullptr && pchEnableVkBasalt[0] == '1' )
	{
		bTryComputeOnly = false;
	}

	for (auto cphysDev : physDevs)
	{
		VkPhysicalDeviceProperties deviceProperties;
		vk.GetPhysicalDeviceProperties(cphysDev, &deviceProperties);

		if (deviceProperties.apiVersion < VK_API_VERSION_1_2)
			continue;

		uint32_t queueFamilyCount = 0;
		vk.GetPhysicalDeviceQueueFamilyProperties(cphysDev, &queueFamilyCount, nullptr);
		std::vector<VkQueueFamilyProperties> queueFamilyProperties(queueFamilyCount);
		vk.GetPhysicalDeviceQueueFamilyProperties(cphysDev, &queueFamilyCount, queueFamilyProperties.data());

		uint32_t generalIndex = ~0u;
		uint32_t computeOnlyIndex = ~0u;
		for (uint32_t i = 0; i < queueFamilyCount; ++i) {
			const VkQueueFlags generalBits = VK_QUEUE_COMPUTE_BIT | VK_QUEUE_GRAPHICS_BIT;
			if ((queueFamilyProperties[i].queueFlags & generalBits) == generalBits )
				generalIndex = std::min(generalIndex, i);
			else if (bTryComputeOnly && queueFamilyProperties[i].queueFlags & VK_QUEUE_COMPUTE_BIT)
				computeOnlyIndex = std::min(computeOnlyIndex, i);
		}

		if ( g_bDebugDualGpuRoute )
		{
			vk_log.infof( "dual-gpu-route: Vulkan candidate '%s' vendor:device %04x:%04x type %s api %u.%u.%u general queue %d compute-only queue %d%s",
				deviceProperties.deviceName,
				deviceProperties.vendorID,
				deviceProperties.deviceID,
				vk_device_type_name( deviceProperties.deviceType ),
				VK_API_VERSION_MAJOR( deviceProperties.apiVersion ),
				VK_API_VERSION_MINOR( deviceProperties.apiVersion ),
				VK_API_VERSION_PATCH( deviceProperties.apiVersion ),
				generalIndex == ~0u ? -1 : int( generalIndex ),
				computeOnlyIndex == ~0u ? -1 : int( computeOnlyIndex ),
				(g_preferVendorID == deviceProperties.vendorID && g_preferDeviceID == deviceProperties.deviceID) ? " preferred" : "" );
		}

		if (generalIndex != ~0u || computeOnlyIndex != ~0u)
		{
			// Select the device if it's the first one or the preferred one
			if (!m_physDev ||
			    (g_preferVendorID == deviceProperties.vendorID && g_preferDeviceID == deviceProperties.deviceID))
			{
				// if we have a surface, check that the queue family can actually present on it
				if (surface) {
					VkBool32 canPresent = false;
					vk.GetPhysicalDeviceSurfaceSupportKHR( cphysDev, generalIndex, surface, &canPresent );
					if ( !canPresent )
					{
						vk_log.infof( "physical device %04x:%04x queue doesn't support presenting on our surface, testing next one..", deviceProperties.vendorID, deviceProperties.deviceID );
						continue;
					}
					if (computeOnlyIndex != ~0u)
					{
						vk.GetPhysicalDeviceSurfaceSupportKHR( cphysDev, computeOnlyIndex, surface, &canPresent );
						if ( !canPresent )
						{
							vk_log.infof( "physical device %04x:%04x compute queue doesn't support presenting on our surface, using graphics queue", deviceProperties.vendorID, deviceProperties.deviceID );
							computeOnlyIndex = ~0u;
						}
					}
				}

				m_queueFamily = computeOnlyIndex == ~0u ? generalIndex : computeOnlyIndex;
				m_generalQueueFamily = generalIndex;
				m_physDev = cphysDev;

				if ( const char *pszReason = vulkan_queue_family_quirk_force_general( deviceProperties ) )
				{
					vk_log.infof( "%s; forcing general queue family instead of compute-only queue", pszReason );
					m_queueFamily = generalIndex;
				}
				else if ( env_to_bool( getenv( "GAMESCOPE_FORCE_GENERAL_QUEUE" ) ) )
					m_queueFamily = generalIndex;
			}
		}
	}

	if (!m_physDev)
	{
		vk_log.errorf("failed to find physical device");
		return false;
	}

	VkPhysicalDeviceProperties props;
	vk.GetPhysicalDeviceProperties( m_physDev, &props );
	m_uMaxComputeSharedMemorySize = props.limits.maxComputeSharedMemorySize;
	vk_log.infof( "selecting physical device '%s': queue family %x (general queue family %x)", props.deviceName, m_queueFamily, m_generalQueueFamily );

	m_uFramegenOtherDeviceCount = 0u;
	m_framegenOtherDeviceName = {};
	for ( auto cphysDev : physDevs )
	{
		if ( cphysDev == m_physDev )
			continue;

		VkPhysicalDeviceProperties otherProperties;
		vk.GetPhysicalDeviceProperties( cphysDev, &otherProperties );
		if ( otherProperties.deviceType == VK_PHYSICAL_DEVICE_TYPE_CPU )
			continue;

		m_uFramegenOtherDeviceCount++;
		if ( m_uFramegenOtherDeviceCount == 1u )
		{
			std::snprintf( m_framegenOtherDeviceName.data(),
				m_framegenOtherDeviceName.size(), "%s", otherProperties.deviceName );
		}
	}

	// Record how many queues the chosen compositor family exposes, so
	// createDevice can request a second (frame-generation) queue when the
	// hardware/driver exposes it.
	{
		uint32_t nQueueFamilies = 0;
		vk.GetPhysicalDeviceQueueFamilyProperties( m_physDev, &nQueueFamilies, nullptr );
		std::vector<VkQueueFamilyProperties> queueFamilyProperties( nQueueFamilies );
		vk.GetPhysicalDeviceQueueFamilyProperties( m_physDev, &nQueueFamilies, queueFamilyProperties.data() );
		if ( m_queueFamily < nQueueFamilies )
			m_queueCount = queueFamilyProperties[ m_queueFamily ].queueCount;

		// Pick the family the dedicated frame-generation queue will live on.
		// Preference order, so nothing changes on hardware that already had a
		// dedicated queue (AMD/NVIDIA: the compute family exposes many queues):
		//  1. queue index 1 of the compositor family, when it exposes >= 2.
		//  2. queue index 0 of a *different* compute-capable family, preferring a
		//     compute-only one. This is the case that unlocks Intel ANV/Xe, whose
		//     general family commonly exposes a single queue and which is quirked
		//     away from the compute-only family for imported-image interop.
		//  3. queue index 1 of the general family, if it has a spare queue.
		if ( m_queueCount >= 2 )
		{
			m_framegenQueueFamily = m_queueFamily;
			m_framegenQueueIndex = 1;
		}
		else
		{
			uint32_t nComputeOnly = ~0u;
			uint32_t nComputeAny = ~0u;
			for ( uint32_t i = 0; i < nQueueFamilies; ++i )
			{
				if ( i == m_queueFamily || i == m_generalQueueFamily )
					continue;
				if ( !( queueFamilyProperties[ i ].queueFlags & VK_QUEUE_COMPUTE_BIT ) )
					continue;
				if ( queueFamilyProperties[ i ].queueCount == 0 )
					continue;

				if ( !( queueFamilyProperties[ i ].queueFlags & VK_QUEUE_GRAPHICS_BIT ) )
					nComputeOnly = std::min( nComputeOnly, i );
				else
					nComputeAny = std::min( nComputeAny, i );
			}

			const uint32_t nOther = nComputeOnly != ~0u ? nComputeOnly : nComputeAny;
			if ( nOther != ~0u )
			{
				m_framegenQueueFamily = nOther;
				m_framegenQueueIndex = 0;
			}
			else if ( m_generalQueueFamily != m_queueFamily
				&& m_generalQueueFamily < nQueueFamilies
				&& queueFamilyProperties[ m_generalQueueFamily ].queueCount >= 2 )
			{
				m_framegenQueueFamily = m_generalQueueFamily;
				m_framegenQueueIndex = 1;
			}
		}
	}

	if ( g_bDebugDualGpuRoute )
	{
		vk_log.infof( "dual-gpu-route: compositor Vulkan device '%s' vendor:device %04x:%04x type %s queue family %u (queues %u) general queue family %u",
			props.deviceName,
			props.vendorID,
			props.deviceID,
			vk_device_type_name( props.deviceType ),
			m_queueFamily,
			m_queueCount,
			m_generalQueueFamily );
	}

	return true;
}

bool CVulkanDevice::createDevice()
{
	uint32_t supportedExtensionCount;
	vk.EnumerateDeviceExtensionProperties( physDev(), NULL, &supportedExtensionCount, NULL );

	m_supportedExts.resize(supportedExtensionCount);
	vk.EnumerateDeviceExtensionProperties( physDev(), NULL, &supportedExtensionCount, m_supportedExts.data() );

	if ( !GetBackend()->ValidPhysicalDevice( physDev() ) ) {
		vk_log.errorf( "not a valid physical device" );
		return false;
	}

	vk.GetPhysicalDeviceMemoryProperties( physDev(), &m_memoryProperties );

	bool hasDrmProps = vulkan_has_drm_props();
	bool supportsForeignQueue = false;
	bool supportsHDRMetadata = false;
	const char *pszGlobalPriorityExtension = nullptr;
	for (const auto& ext : m_supportedExts) {
		if ( strcmp(ext.extensionName, VK_EXT_IMAGE_DRM_FORMAT_MODIFIER_EXTENSION_NAME) == 0 )
			m_bSupportsModifiers = true;

		if ( strcmp(ext.extensionName, VK_EXT_QUEUE_FAMILY_FOREIGN_EXTENSION_NAME) == 0 )
			supportsForeignQueue = true;

		if ( strcmp(ext.extensionName, VK_EXT_HDR_METADATA_EXTENSION_NAME) == 0 )
			supportsHDRMetadata = true;

		if ( strcmp(ext.extensionName, VK_KHR_GLOBAL_PRIORITY_EXTENSION_NAME) == 0 )
		{
			m_bSupportsGlobalPriority = true;
			pszGlobalPriorityExtension = VK_KHR_GLOBAL_PRIORITY_EXTENSION_NAME;
		}
		else if ( strcmp(ext.extensionName, VK_EXT_GLOBAL_PRIORITY_EXTENSION_NAME) == 0 )
		{
			m_bSupportsGlobalPriority = true;
			if ( pszGlobalPriorityExtension == nullptr )
				pszGlobalPriorityExtension = VK_EXT_GLOBAL_PRIORITY_EXTENSION_NAME;
		}
	}

	vk_log.infof( "physical device %s DRM format modifiers", m_bSupportsModifiers ? "supports" : "does not support" );

	if ( !hasDrmProps ) {
		// This could happen when e.g. running the lavapipe driver
		// (without an actual physical device)
		vk_log.warnf( "physical device doesn't support VK_EXT_physical_device_drm" );
	} else {
#if HAVE_DRM
		VkPhysicalDeviceDrmPropertiesEXT drmProps = {
			.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRM_PROPERTIES_EXT,
		};
		VkPhysicalDeviceProperties2 props2 = {
			.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
			.pNext = &drmProps,
		};
		vk.GetPhysicalDeviceProperties2( physDev(), &props2 );

		if ( g_bDebugDualGpuRoute )
		{
			vk_log.infof( "dual-gpu-route: compositor Vulkan DRM props primary %s %" PRId64 ":%" PRId64 " render %s %" PRId64 ":%" PRId64,
				drmProps.hasPrimary ? "yes" : "no",
				drmProps.primaryMajor,
				drmProps.primaryMinor,
				drmProps.hasRender ? "yes" : "no",
				drmProps.renderMajor,
				drmProps.renderMinor );
		}

		if ( !GetBackend()->UsesVulkanSwapchain() && !drmProps.hasPrimary ) {
			vk_log.errorf( "physical device has no primary node" );
			return false;
		}
		if ( !drmProps.hasRender ) {
			vk_log.errorf( "physical device has no render node" );
			return false;
		}

		dev_t renderDevId = makedev( drmProps.renderMajor, drmProps.renderMinor );
		drmDevice *drmDev = nullptr;
		if (drmGetDeviceFromDevId(renderDevId, 0, &drmDev) != 0) {
			vk_log.errorf( "drmGetDeviceFromDevId() failed" );
			return false;
		}
		assert(drmDev->available_nodes & (1 << DRM_NODE_RENDER));
		const char *drmRenderName = drmDev->nodes[DRM_NODE_RENDER];
		debug_log_drm_device( "compositor Vulkan", drmDev );

		m_drmRendererFd = open( drmRenderName, O_RDWR | O_CLOEXEC );
		drmFreeDevice(&drmDev);
		if ( m_drmRendererFd < 0 ) {
			vk_log.errorf_errno( "failed to open DRM render node" );
			return false;
		}

		if ( drmProps.hasPrimary ) {
			m_bHasDrmPrimaryDevId = true;
			m_drmPrimaryDevId = makedev( drmProps.primaryMajor, drmProps.primaryMinor );
		}
#else
		vk_log.warnf( "built without DRM support" );
#endif
	}

	if ( m_bSupportsModifiers && !supportsForeignQueue ) {
		vk_log.infof( "The vulkan driver does not support foreign queues,"
		              " disabling modifier support.");
		m_bSupportsModifiers = false;
	}

	{
		VkPhysicalDeviceVulkan12Features vulkan12Features = {
			.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
		};
		VkPhysicalDeviceVulkan13Features vulkan13Features = {
			.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,
			.pNext = &vulkan12Features,
		};
		VkPhysicalDeviceProperties deviceProperties;
		vk.GetPhysicalDeviceProperties( physDev(), &deviceProperties );
		const bool bVulkan13 = deviceProperties.apiVersion >= VK_API_VERSION_1_3;
		VkPhysicalDeviceFeatures2 features2 = {
			.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
			.pNext = bVulkan13 ? static_cast<void *>( &vulkan13Features ) : &vulkan12Features,
		};
		vk.GetPhysicalDeviceFeatures2( physDev(), &features2 );

		if ( !vulkan12Features.scalarBlockLayout )
		{
			vk_log.errorf( "physical device does not support scalarBlockLayout, required by gamescope shaders" );
			return false;
		}

		if ( !vulkan12Features.timelineSemaphore )
		{
			vk_log.errorf( "physical device does not support timelineSemaphore, required by gamescope synchronization" );
			return false;
		}

		m_bSupportsShaderFloat16 = vulkan12Features.shaderFloat16;
		m_bSupportsFp16 = m_bSupportsShaderFloat16 && features2.features.shaderInt16;
		m_bSupportsStorageImageExtendedFormats = features2.features.shaderStorageImageExtendedFormats;
		m_bSupportsSync2 = bVulkan13 && vulkan13Features.synchronization2;

		// Generic compute outputs are bound to the backend-selected image format
		// (8/10/16-bit integer or float). Declaring one fixed SPIR-V image format
		// for all of them is invalid even when the byte size happens to match.
		if ( !features2.features.shaderStorageImageWriteWithoutFormat )
		{
			vk_log.errorf( "physical device does not support shaderStorageImageWriteWithoutFormat, required by gamescope output shaders" );
			return false;
		}
	}

	// Queue 0 carries real composites; queue 1 is disposable speculative
	// frame-generation work. Keep the latter at the lowest relative priority so
	// a long motion pass cannot compete equally for compute/memory resources and
	// add jitter to a real frame. The global priority is family-wide, but Vulkan
	// still applies these relative priorities between queues in that family.
	float queuePriorities[2] = { 1.0f, 0.0f };

	VkDeviceQueueGlobalPriorityCreateInfoEXT queueCreateInfoEXT = {
		.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_GLOBAL_PRIORITY_CREATE_INFO_EXT,
		.pNext = nullptr,
		.globalPriority = VK_QUEUE_GLOBAL_PRIORITY_REALTIME_EXT
	};

	// Request a second queue in the compositor's (compute) family for frame
	// generation, when framegen is enabled, the family exposes one, and it isn't
	// disabled. Vulkan has no per-queue global priority within a family, so both
	// queues inherit this create-info's REALTIME priority. Their relative
	// priorities above still favor real composites; the second queue also removes
	// FIFO head-of-line blocking. Gated on framegen so a session that never uses
	// it requests exactly the single REALTIME queue it did before.
	const bool bWantFramegenQueue = g_bExperimentalFramegen && framegen_backend_supported()
		&& m_framegenQueueFamily != ~0u && !env_to_bool( getenv( "GAMESCOPE_FRAMEGEN_SINGLE_QUEUE" ) );
	if ( !bWantFramegenQueue )
	{
		m_framegenQueueFamily = ~0u;
		m_framegenQueueIndex = 0;
	}

	// The framegen queue is normally index 1 of one of the two families we
	// already create; when it lives on a third, compute-capable family (the
	// compositor family only exposed one queue) that family gets its own
	// create-info with a single low-priority queue.
	const bool bFramegenOwnFamily = bWantFramegenQueue
		&& m_framegenQueueFamily != m_queueFamily
		&& m_framegenQueueFamily != m_generalQueueFamily;
	const uint32_t nComputeQueues = bWantFramegenQueue && m_framegenQueueFamily == m_queueFamily ? 2u : 1u;
	const uint32_t nGeneralQueues = bWantFramegenQueue && m_framegenQueueFamily == m_generalQueueFamily
		&& m_generalQueueFamily != m_queueFamily ? 2u : 1u;

	VkDeviceQueueCreateInfo queueCreateInfos[3] =
	{
		{
			.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
			.pNext = gamescope::Process::HasCapSysNice() && m_bSupportsGlobalPriority ? &queueCreateInfoEXT : nullptr,
			.queueFamilyIndex = m_queueFamily,
			.queueCount = nComputeQueues,
			.pQueuePriorities = queuePriorities
		},
		{
			.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
			.pNext = gamescope::Process::HasCapSysNice() && m_bSupportsGlobalPriority ? &queueCreateInfoEXT : nullptr,
			.queueFamilyIndex = m_generalQueueFamily,
			.queueCount = nGeneralQueues,
			.pQueuePriorities = queuePriorities
		},
		{
			.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
			.pNext = gamescope::Process::HasCapSysNice() && m_bSupportsGlobalPriority ? &queueCreateInfoEXT : nullptr,
			.queueFamilyIndex = bFramegenOwnFamily ? m_framegenQueueFamily : 0u,
			.queueCount = 1,
			// Speculative work: lowest relative priority, like queue index 1 of
			// the compositor family in the same-family case.
			.pQueuePriorities = &queuePriorities[1]
		},
	};

	std::vector< const char * > enabledExtensions;

	if ( GetBackend()->UsesVulkanSwapchain() )
	{
		enabledExtensions.push_back( VK_KHR_SWAPCHAIN_EXTENSION_NAME );
		enabledExtensions.push_back( VK_KHR_SWAPCHAIN_MUTABLE_FORMAT_EXTENSION_NAME );

		enabledExtensions.push_back( VK_KHR_PRESENT_ID_EXTENSION_NAME );
		enabledExtensions.push_back( VK_KHR_PRESENT_WAIT_EXTENSION_NAME );
	}

	if ( m_bSupportsModifiers )
	{
		enabledExtensions.push_back( VK_EXT_IMAGE_DRM_FORMAT_MODIFIER_EXTENSION_NAME );
		enabledExtensions.push_back( VK_EXT_QUEUE_FAMILY_FOREIGN_EXTENSION_NAME );
	}

	enabledExtensions.push_back( VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME );
	enabledExtensions.push_back( VK_EXT_EXTERNAL_MEMORY_DMA_BUF_EXTENSION_NAME );

	enabledExtensions.push_back( VK_KHR_EXTERNAL_SEMAPHORE_FD_EXTENSION_NAME );

	if ( m_bSupportsGlobalPriority )
		enabledExtensions.push_back( pszGlobalPriorityExtension );

	enabledExtensions.push_back( VK_EXT_ROBUSTNESS_2_EXTENSION_NAME );
#if 0
	enabledExtensions.push_back( VK_KHR_MAINTENANCE_5_EXTENSION_NAME );
#endif

	if ( supportsHDRMetadata )
		enabledExtensions.push_back( VK_EXT_HDR_METADATA_EXTENSION_NAME );

	for ( auto& extension : GetBackend()->GetDeviceExtensions( physDev() ) )
		enabledExtensions.push_back( extension );

	uint32_t devExtPropCount = 0;
	vk.EnumerateDeviceExtensionProperties( physDev(), nullptr, &devExtPropCount, nullptr );
	std::vector<VkExtensionProperties> devExtProp( devExtPropCount );
	vk.EnumerateDeviceExtensionProperties( physDev(), nullptr, &devExtPropCount, devExtProp.data() );
	bool anyMissing = false;
	for ( auto& requiredExt : enabledExtensions ) {
		bool extFound = false;
		for ( auto & availableExt : devExtProp ) {
			if ( strcmp( requiredExt, availableExt.extensionName ) == 0 ) {
				extFound = true;
				break;
			}
		}
		if ( !extFound ) {
			vk_log.errorf( "Missing required extension: %s", requiredExt );
			anyMissing = true;
		}
	}
	if ( anyMissing )
		return false;

#if 0
	VkPhysicalDeviceMaintenance5FeaturesKHR maintenance5 = {
		.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_5_FEATURES_KHR,
		.maintenance5 = VK_TRUE,
	};
#endif

	VkPhysicalDeviceVulkan13Features features13 = {
		.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,
#if 0
		.pNext = &maintenance5,
#endif
		.synchronization2 = m_bSupportsSync2,
		.dynamicRendering = VK_TRUE,
	};

	VkPhysicalDevicePresentWaitFeaturesKHR presentWaitFeatures = {
		.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PRESENT_WAIT_FEATURES_KHR,
		.pNext = &features13,
		.presentWait = VK_TRUE,
	};

	VkPhysicalDevicePresentIdFeaturesKHR presentIdFeatures = {
		.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PRESENT_ID_FEATURES_KHR,
		.pNext = &presentWaitFeatures,
		.presentId = VK_TRUE,
	};

	VkPhysicalDeviceFeatures2 features2 = {
		.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
		.pNext = &presentIdFeatures,
		.features = {
			.shaderInt16 = m_bSupportsFp16,
		},
	};
	features2.features.shaderStorageImageWriteWithoutFormat = VK_TRUE;
	features2.features.shaderStorageImageExtendedFormats = m_bSupportsStorageImageExtendedFormats;

	VkDeviceCreateInfo deviceCreateInfo = {
		.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
		.pNext = &features2,
		.queueCreateInfoCount = ( m_queueFamily == m_generalQueueFamily ? 1u : 2u ) + ( bFramegenOwnFamily ? 1u : 0u ),
		.pQueueCreateInfos = queueCreateInfos,
		.enabledExtensionCount = (uint32_t)enabledExtensions.size(),
		.ppEnabledExtensionNames = enabledExtensions.data(),
	};

	VkPhysicalDeviceVulkan12Features vulkan12Features = {
		.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
		.pNext = std::exchange(features2.pNext, &vulkan12Features),
		.shaderFloat16 = m_bSupportsShaderFloat16,
		.scalarBlockLayout = VK_TRUE,
		.timelineSemaphore = VK_TRUE,
	};

	VkPhysicalDeviceSamplerYcbcrConversionFeatures ycbcrFeatures = {
		.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SAMPLER_YCBCR_CONVERSION_FEATURES,
		.pNext = std::exchange(features2.pNext, &ycbcrFeatures),
		.samplerYcbcrConversion = VK_TRUE,
	};

	VkPhysicalDeviceRobustness2FeaturesEXT robustness2Features = {
		.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ROBUSTNESS_2_FEATURES_EXT,
		.pNext = std::exchange(features2.pNext, &robustness2Features),
		.nullDescriptor = VK_TRUE,
	};

	// queueCreateInfoCount above counts a contiguous prefix; when the compositor
	// and general families are the same we only submit entry [0], so move the
	// framegen family's create-info into slot [1].
	if ( bFramegenOwnFamily && m_queueFamily == m_generalQueueFamily )
		queueCreateInfos[1] = queueCreateInfos[2];

	// Drop the global-priority chain from the submitted create-infos, starting at
	// nFirst. Naming individual queues by index is not robust here: the framegen
	// entry is compacted from [2] into [1] above when the compositor and general
	// families coincide, so only [0] (always the compositor family) has a fixed
	// meaning. Entries past queueCreateInfoCount are not submitted at all.
	const auto clearQueuePriorities = [&]( uint32_t nFirst )
	{
		for ( uint32_t i = nFirst; i < deviceCreateInfo.queueCreateInfoCount; i++ )
			queueCreateInfos[i].pNext = nullptr;
	};

	VkResult res = vk.CreateDevice(physDev(), &deviceCreateInfo, nullptr, &m_device);
	if ( res == VK_ERROR_NOT_PERMITTED_KHR && gamescope::Process::HasCapSysNice() && m_bSupportsGlobalPriority )
	{
		fprintf(stderr, "vkCreateDevice failed with high-priority queues. Falling back to regular priority for the secondary queues.\n");
		clearQueuePriorities( 1 );
		res = vk.CreateDevice(physDev(), &deviceCreateInfo, nullptr, &m_device);

		if ( res == VK_ERROR_NOT_PERMITTED_KHR && gamescope::Process::HasCapSysNice() )
		{
			fprintf(stderr, "vkCreateDevice still failed. Falling back to regular priority for all queues.\n");
			clearQueuePriorities( 0 );
			res = vk.CreateDevice(physDev(), &deviceCreateInfo, nullptr, &m_device);
		}
	}

	if ( res != VK_SUCCESS )
	{
		vk_errorf( res, "vkCreateDevice failed" );
		return false;
	}

	#define VK_FUNC(x) vk.x = (PFN_vk##x) vk.GetDeviceProcAddr(device(), "vk"#x);
	VULKAN_DEVICE_FUNCTIONS
	#undef VK_FUNC

	vk.GetDeviceQueue(device(), m_queueFamily, 0, &m_queue);
	if ( m_queueFamily == m_generalQueueFamily )
		m_generalQueue = m_queue;
	else
		vk.GetDeviceQueue(device(), m_generalQueueFamily, 0, &m_generalQueue);

	if ( bWantFramegenQueue )
	{
		vk.GetDeviceQueue( device(), m_framegenQueueFamily, m_framegenQueueIndex, &m_framegenQueue );
		m_bHasFramegenQueue = m_framegenQueue != VK_NULL_HANDLE;
		if ( m_bHasFramegenQueue )
		{
			vk_log.infof( "framegen: dedicated queue family %u index %u (compositor family %u)",
				m_framegenQueueFamily, m_framegenQueueIndex, m_queueFamily );
		}
		else
		{
			m_framegenQueueFamily = ~0u;
		}
	}
	else if ( g_bExperimentalFramegen && framegen_backend_supported() )
	{
		vk_log.infof( "framegen: no dedicated queue (compositor family %u exposes %u queue(s), no other compute-capable family usable); generation shares the composite queue",
			m_queueFamily, m_queueCount );
	}

	return true;
}

static VkSamplerYcbcrModelConversion colorspaceToYCBCRModel( EStreamColorspace colorspace )
{
	switch (colorspace)
	{
		default:
		case k_EStreamColorspace_Unknown:
			return VK_SAMPLER_YCBCR_MODEL_CONVERSION_YCBCR_709;

		case k_EStreamColorspace_BT601:
		case k_EStreamColorspace_BT601_Full:
			return VK_SAMPLER_YCBCR_MODEL_CONVERSION_YCBCR_601;

		case k_EStreamColorspace_BT709:
		case k_EStreamColorspace_BT709_Full:
			return VK_SAMPLER_YCBCR_MODEL_CONVERSION_YCBCR_709;
	}
}

static VkSamplerYcbcrRange colorspaceToYCBCRRange( EStreamColorspace colorspace )
{
	switch (colorspace)
	{
		default:
		case k_EStreamColorspace_Unknown:
			return VK_SAMPLER_YCBCR_RANGE_ITU_FULL;

		case k_EStreamColorspace_BT709:
		case k_EStreamColorspace_BT601:
			return VK_SAMPLER_YCBCR_RANGE_ITU_NARROW;

		case k_EStreamColorspace_BT601_Full:
		case k_EStreamColorspace_BT709_Full:
			return VK_SAMPLER_YCBCR_RANGE_ITU_FULL;
	}
}

bool CVulkanDevice::createLayouts()
{
	VkFormatProperties nv12Properties;
	vk.GetPhysicalDeviceFormatProperties(physDev(), VK_FORMAT_G8_B8R8_2PLANE_420_UNORM, &nv12Properties);
	bool cosited = nv12Properties.optimalTilingFeatures & VK_FORMAT_FEATURE_COSITED_CHROMA_SAMPLES_BIT;

	VkSamplerYcbcrConversionCreateInfo ycbcrSamplerConversionCreateInfo = 
	{
		.sType = VK_STRUCTURE_TYPE_SAMPLER_YCBCR_CONVERSION_CREATE_INFO,
		.format = VK_FORMAT_G8_B8R8_2PLANE_420_UNORM,
		.ycbcrModel = colorspaceToYCBCRModel( g_ForcedNV12ColorSpace ),
		.ycbcrRange = colorspaceToYCBCRRange( g_ForcedNV12ColorSpace ),
		.xChromaOffset = cosited ? VK_CHROMA_LOCATION_COSITED_EVEN : VK_CHROMA_LOCATION_MIDPOINT,
		.yChromaOffset = cosited ? VK_CHROMA_LOCATION_COSITED_EVEN : VK_CHROMA_LOCATION_MIDPOINT,
		.chromaFilter = VK_FILTER_LINEAR,
		.forceExplicitReconstruction = VK_FALSE,
	};

	vk.CreateSamplerYcbcrConversion( device(), &ycbcrSamplerConversionCreateInfo, nullptr, &m_ycbcrConversion );

	VkSamplerYcbcrConversionInfo ycbcrSamplerConversionInfo = {
		.sType = VK_STRUCTURE_TYPE_SAMPLER_YCBCR_CONVERSION_INFO,
		.conversion = m_ycbcrConversion,
	};

	VkSamplerCreateInfo ycbcrSamplerInfo = {
		.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
		.pNext = &ycbcrSamplerConversionInfo,
		.magFilter = VK_FILTER_LINEAR,
		.minFilter = VK_FILTER_LINEAR,
		.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
		.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
		.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
		.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_BLACK,
	};
	
	vk.CreateSampler( device(), &ycbcrSamplerInfo, nullptr, &m_ycbcrSampler );

	// Create an array of our ycbcrSampler to fill up
	std::array<VkSampler, VKR_SAMPLER_SLOTS> ycbcrSamplers;
	for (auto& sampler : ycbcrSamplers)
		sampler = m_ycbcrSampler;

	std::array<VkDescriptorSetLayoutBinding, 7 > layoutBindings = {
		VkDescriptorSetLayoutBinding {
			.binding = 0,
			.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
			.descriptorCount = 1,
			.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
		},
		VkDescriptorSetLayoutBinding {
			.binding = 1,
			.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
			.descriptorCount = 1,
			.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
		},
		VkDescriptorSetLayoutBinding {
			.binding = 2,
			.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
			.descriptorCount = 1,
			.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
		},
		VkDescriptorSetLayoutBinding {
			.binding = 3,
			.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
			.descriptorCount = VKR_SAMPLER_SLOTS,
			.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
		},
		VkDescriptorSetLayoutBinding {
			.binding = 4,
			.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
			.descriptorCount = VKR_SAMPLER_SLOTS,
			.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
			.pImmutableSamplers = ycbcrSamplers.data(),
		},
		VkDescriptorSetLayoutBinding {
			.binding = 5,
			.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
			.descriptorCount = VKR_LUT3D_COUNT,
			.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
		},
		VkDescriptorSetLayoutBinding {
			.binding = 6,
			.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
			.descriptorCount = VKR_LUT3D_COUNT,
			.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
		},
	};

	VkDescriptorSetLayoutCreateInfo descriptorSetLayoutCreateInfo =
	{
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
		.bindingCount = (uint32_t)layoutBindings.size(),
		.pBindings = layoutBindings.data()
	};

	VkResult res = vk.CreateDescriptorSetLayout(device(), &descriptorSetLayoutCreateInfo, 0, &m_descriptorSetLayout);
	if ( res != VK_SUCCESS )
	{
		vk_errorf( res, "vkCreateDescriptorSetLayout failed" );
		return false;
	}

	// A small compute push-constant range used by the frame-generation shaders
	// for their per-slot parameters. Push constants are recorded directly into
	// the command buffer, so framegen needs no slice of the shared upload arena
	// — which is what lets its work migrate to a dedicated queue without racing
	// the composite path's bump allocator. Shaders that don't declare a
	// push_constant block simply ignore the range.
	VkPushConstantRange framegenPushConstantRange = {
		.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
		.offset = 0,
		.size = k_uFramegenPushConstantSize,
	};

	VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
		.setLayoutCount = 1,
		.pSetLayouts = &m_descriptorSetLayout,
		.pushConstantRangeCount = 1,
		.pPushConstantRanges = &framegenPushConstantRange,
	};

	res = vk.CreatePipelineLayout(device(), &pipelineLayoutCreateInfo, nullptr, &m_pipelineLayout);
	if ( res != VK_SUCCESS )
	{
		vk_errorf( res, "vkCreatePipelineLayout failed" );
		return false;
	}

	return true;
}

bool CVulkanDevice::createPools()
{
	VkCommandPoolCreateInfo commandPoolCreateInfo = {
		.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
		.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
		.queueFamilyIndex = m_queueFamily,
	};

	VkResult res = vk.CreateCommandPool(device(), &commandPoolCreateInfo, nullptr, &m_commandPool);
	if ( res != VK_SUCCESS )
	{
		vk_errorf( res, "vkCreateCommandPool failed" );
		return false;
	}

	VkCommandPoolCreateInfo generalCommandPoolCreateInfo = {
		.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
		.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
		.queueFamilyIndex = m_generalQueueFamily,
	};

	res = vk.CreateCommandPool(device(), &generalCommandPoolCreateInfo, nullptr, &m_generalCommandPool);
	if ( res != VK_SUCCESS )
	{
		vk_errorf( res, "vkCreateCommandPool failed" );
		return false;
	}

	// Command buffers are family-scoped: when the framegen queue lives on a
	// different family it needs its own pool (and its own recycle list, see
	// commandBuffer()/framegenGarbageCollect()).
	if ( framegenFamilySplit() )
	{
		VkCommandPoolCreateInfo framegenCommandPoolCreateInfo = {
			.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
			.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
			.queueFamilyIndex = m_framegenQueueFamily,
		};

		res = vk.CreateCommandPool(device(), &framegenCommandPoolCreateInfo, nullptr, &m_framegenCommandPool);
		if ( res != VK_SUCCESS )
		{
			vk_errorf( res, "vkCreateCommandPool failed (framegen family)" );
			return false;
		}
	}

	VkPhysicalDeviceImageFormatInfo2 imageFormatInfo = {
		.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_FORMAT_INFO_2,
		.format = VK_FORMAT_G8_B8R8_2PLANE_420_UNORM,
		.type = VK_IMAGE_TYPE_2D,
		.tiling = VK_IMAGE_TILING_OPTIMAL,
		.usage = VK_IMAGE_USAGE_SAMPLED_BIT,
	};

	VkSamplerYcbcrConversionImageFormatProperties ycbcrProps = {
		.sType = VK_STRUCTURE_TYPE_SAMPLER_YCBCR_CONVERSION_IMAGE_FORMAT_PROPERTIES,
	};

	VkImageFormatProperties2 imageFormatProps = {
		.sType = VK_STRUCTURE_TYPE_IMAGE_FORMAT_PROPERTIES_2,
		.pNext = &ycbcrProps,
	};

	res = vk.GetPhysicalDeviceImageFormatProperties2( physDev(), &imageFormatInfo, &imageFormatProps );

	// Reserve extra sets for the separate framegen descriptor ring when framegen
	// is enabled, so the pool covers both rings. Non-framegen sessions allocate
	// exactly what they did before.
	const uint32_t nTotalSets = uint32_t(m_descriptorSets.size())
		+ ( g_bExperimentalFramegen && framegen_backend_supported() ? k_uFramegenDescriptorSets : 0u );

	VkDescriptorPoolSize poolSizes[3] {
		{
			VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
			nTotalSets,
		},
		{
			VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
			nTotalSets * 2,
		},
		{
			VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
			nTotalSets * (((ycbcrProps.combinedImageSamplerDescriptorCount + 1) * VKR_SAMPLER_SLOTS) + (2 * VKR_LUT3D_COUNT)),
		},
	};

	VkDescriptorPoolCreateInfo descriptorPoolCreateInfo = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
		.maxSets = nTotalSets,
		.poolSizeCount = sizeof(poolSizes) / sizeof(poolSizes[0]),
		.pPoolSizes = poolSizes,
	};
	
	res = vk.CreateDescriptorPool(device(), &descriptorPoolCreateInfo, nullptr, &m_descriptorPool);
	if ( res != VK_SUCCESS )
	{
		vk_errorf( res, "vkCreateDescriptorPool failed" );
		return false;
	}

	return true;
}

bool CVulkanDevice::createShaders()
{
	struct ShaderInfo_t
	{
		const uint32_t* spirv;
		uint32_t size;
	};

	std::array<ShaderInfo_t, SHADER_TYPE_COUNT> shaderInfos;
#define SHADER(type, array) shaderInfos[SHADER_TYPE_##type] = {array , sizeof(array)}
	SHADER(BLIT, cs_composite_blit);
	SHADER(BLUR, cs_composite_blur);
	SHADER(BLUR_COND, cs_composite_blur_cond);
	SHADER(BLUR_FIRST_PASS, cs_gaussian_blur_horizontal);
	SHADER(RCAS, cs_composite_rcas);
	if (m_bSupportsFp16)
	{
		SHADER(EASU, cs_easu_fp16);
		SHADER(NIS, cs_nis_fp16);
	}
	else
	{
		SHADER(EASU, cs_easu);
		SHADER(NIS, cs_nis);
	}
	SHADER(RGB_TO_NV12, cs_rgb_to_nv12);
	SHADER(FRAMEGEN_BLEND, cs_framegen_blend);
	SHADER(FRAMEGEN_EXTRAPOLATE, cs_framegen_extrapolate);
	SHADER(FRAMEGEN_EXTRAPOLATE_DIRECT, cs_framegen_extrapolate_direct);
	if (m_bSupportsShaderFloat16)
		SHADER(FRAMEGEN_EXTRAPOLATE_FP16, cs_framegen_extrapolate_fp16);
	else
		SHADER(FRAMEGEN_EXTRAPOLATE_FP16, cs_framegen_extrapolate);
	SHADER(FRAMEGEN_EXTRAPOLATE_PAIR, cs_framegen_extrapolate_pair);
	if (m_bSupportsShaderFloat16)
		SHADER(FRAMEGEN_EXTRAPOLATE_PAIR_FP16, cs_framegen_extrapolate_pair_fp16);
	else
		SHADER(FRAMEGEN_EXTRAPOLATE_PAIR_FP16, cs_framegen_extrapolate_pair);
	SHADER(FRAMEGEN_HUD, cs_framegen_hud);
	SHADER(FRAMEGEN_MOTION_LUMA_PAIR, cs_framegen_motion_luma_pair);
	SHADER(FRAMEGEN_MOTION_LUMA_PAIR_RGBA, cs_framegen_motion_luma_pair_rgba);
	SHADER(FRAMEGEN_MOTION_PYRAMID, cs_framegen_motion_pyramid);
	SHADER(FRAMEGEN_MOTION_PYRAMID_RGBA, cs_framegen_motion_pyramid_rgba);
	SHADER(FRAMEGEN_MOTION_MATCH, cs_framegen_motion_match);
	SHADER(FRAMEGEN_MOTION_MATCH_REFINE, cs_framegen_motion_match_refine);
	SHADER(FRAMEGEN_MOTION_FBCHECK, cs_framegen_motion_fbcheck);
	SHADER(FRAMEGEN_MOTION_WARP, cs_framegen_motion_warp);
	SHADER(FRAMEGEN_MOTION_WARP_ACCEL, cs_framegen_motion_warp_accel);
	SHADER(FRAMEGEN_MOTION_BIDIR, cs_framegen_motion_bidir);
	SHADER(FRAMEGEN_MOTION_BIDIR_TRACE, cs_framegen_motion_bidir);
	SHADER(FRAMEGEN_MOTION_STATS, cs_framegen_motion_stats);
	SHADER(FRAMEGEN_MOTION_STATS_APPLY, cs_framegen_motion_stats_apply);
	SHADER(FRAMEGEN_MOTION_NET, cs_framegen_motion_net);
	SHADER(FRAMEGEN_MOTION_NET_TRAIN, cs_framegen_motion_net_train);
	SHADER(FRAMEGEN_MOTION_NET_OPT, cs_framegen_motion_net_opt);
#undef SHADER

	for (uint32_t i = 0; i < shaderInfos.size(); i++)
	{
		VkShaderModuleCreateInfo shaderCreateInfo = {
			.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
			.codeSize = shaderInfos[i].size,
			.pCode = shaderInfos[i].spirv,
		};

		VkResult res = vk.CreateShaderModule(device(), &shaderCreateInfo, nullptr, &m_shaderModules[i]);
		if ( res != VK_SUCCESS )
		{
			vk_errorf( res, "vkCreateShaderModule failed" );
			return false;
		}
	}

	return true;
}

bool CVulkanDevice::createScratchResources()
{
	std::vector<VkDescriptorSetLayout> descriptorSetLayouts(m_descriptorSets.size(), m_descriptorSetLayout);
	
	VkDescriptorSetAllocateInfo descriptorSetAllocateInfo = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
		.descriptorPool = m_descriptorPool,
		.descriptorSetCount = (uint32_t)descriptorSetLayouts.size(),
		.pSetLayouts = descriptorSetLayouts.data(),
	};
	
	VkResult res = vk.AllocateDescriptorSets(device(), &descriptorSetAllocateInfo, m_descriptorSets.data());
	if ( res != VK_SUCCESS )
	{
		vk_log.errorf( "vkAllocateDescriptorSets failed" );
		return false;
	}

	// Separate framegen descriptor ring (see CVulkanDevice::descriptorSet).
	if ( g_bExperimentalFramegen && framegen_backend_supported() )
	{
		m_framegenDescriptorSets.resize( k_uFramegenDescriptorSets );
		std::vector<VkDescriptorSetLayout> framegenLayouts( k_uFramegenDescriptorSets, m_descriptorSetLayout );
		VkDescriptorSetAllocateInfo framegenAllocInfo = {
			.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
			.descriptorPool = m_descriptorPool,
			.descriptorSetCount = (uint32_t)framegenLayouts.size(),
			.pSetLayouts = framegenLayouts.data(),
		};
		res = vk.AllocateDescriptorSets( device(), &framegenAllocInfo, m_framegenDescriptorSets.data() );
		if ( res != VK_SUCCESS )
		{
			vk_log.errorf( "vkAllocateDescriptorSets (framegen) failed" );
			return false;
		}
	}

	// Make and map upload buffer
	VkPhysicalDeviceProperties deviceProperties = {};
	vk.GetPhysicalDeviceProperties( physDev(), &deviceProperties );
	const VkDeviceSize uniformAlignment = std::max<VkDeviceSize>(
		16, deviceProperties.limits.minUniformBufferOffsetAlignment );
	if ( uniformAlignment > UINT32_MAX )
	{
		vk_log.errorf( "uniform-buffer offset alignment is too large: %" PRIu64,
			(uint64_t)uniformAlignment );
		return false;
	}
	m_uniformBufferOffsetAlignment = (uint32_t)uniformAlignment;
	
	VkBufferCreateInfo bufferCreateInfo = {
		.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
		.size = upload_buffer_size + upload_buffer_persistent_pad,
		.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
	};

	res = vk.CreateBuffer( device(), &bufferCreateInfo, nullptr, &m_uploadBuffer );
	if ( res != VK_SUCCESS )
	{
		vk_errorf( res, "vkCreateBuffer failed" );
		return false;
	}
	
	VkMemoryRequirements memRequirements;
	vk.GetBufferMemoryRequirements(device(), m_uploadBuffer, &memRequirements);
	
	uint32_t memTypeIndex =  findMemoryType(VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT|VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT, memRequirements.memoryTypeBits );
	if ( memTypeIndex == ~0u )
	{
		vk_log.errorf( "findMemoryType failed" );
		return false;
	}
	
	VkMemoryAllocateInfo allocInfo = {
		.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
		.allocationSize = memRequirements.size,
		.memoryTypeIndex = memTypeIndex,
	};
	
	vk.AllocateMemory( device(), &allocInfo, nullptr, &m_uploadBufferMemory);
	
	vk.BindBufferMemory( device(), m_uploadBuffer, m_uploadBufferMemory, 0 );

	res = vk.MapMemory( device(), m_uploadBufferMemory, 0, VK_WHOLE_SIZE, 0, (void**)&m_uploadBufferData );
	if ( res != VK_SUCCESS )
	{
		vk_errorf( res, "vkMapMemory failed" );
		return false;
	}

	VkSemaphoreTypeCreateInfo timelineCreateInfo = {
		.sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO,
		.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE,
	};

	VkSemaphoreCreateInfo semCreateInfo = {
		.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
		.pNext = &timelineCreateInfo,
	};

	res = vk.CreateSemaphore( device(), &semCreateInfo, NULL, &m_scratchTimelineSemaphore );
	if ( res != VK_SUCCESS )
	{
		vk_errorf( res, "vkCreateSemaphore failed" );
		return false;
	}

	// Dedicated timeline for the frame-generation queue: framegen signals its
	// own monotonic counter here instead of the shared scratch timeline (whose
	// single counter cannot tolerate a second queue signalling out of order).
	if ( m_bHasFramegenQueue )
	{
		m_framegenTimeline = CreateTimelineSemaphore( 0, false );
		if ( m_framegenTimeline == nullptr )
		{
			vk_log.errorf( "failed to create frame generation timeline semaphore; disabling dedicated framegen queue" );
			m_bHasFramegenQueue = false;
		}
	}

	// Best-effort timestamp query-pool ring to measure live framegen GPU time,
	// feeding the deadline-driven degradation ladder. A dedicated queue measures
	// isolated generation work; the shared-queue fallback measures the actual
	// submission span, including unavoidable same-queue interference. Both are
	// useful deadline signals. If the family cannot timestamp, framegen simply
	// runs without measurement and the ladder stays on the full pipeline.
	{
		uint32_t nQueueFamilyCount = 0;
		vk.GetPhysicalDeviceQueueFamilyProperties( physDev(), &nQueueFamilyCount, nullptr );
		std::vector<VkQueueFamilyProperties> queueFamilyProps( nQueueFamilyCount );
		vk.GetPhysicalDeviceQueueFamilyProperties( physDev(), &nQueueFamilyCount, queueFamilyProps.data() );

		VkPhysicalDeviceProperties physProps = {};
		vk.GetPhysicalDeviceProperties( physDev(), &physProps );

		// Timestamps are written on whichever queue records the framegen batch.
		const uint32_t nTimestampFamily = m_bHasFramegenQueue ? m_framegenQueueFamily : m_queueFamily;
		const bool bTimestampsUsable = nTimestampFamily < nQueueFamilyCount
			&& queueFamilyProps[ nTimestampFamily ].timestampValidBits > 0
			&& physProps.limits.timestampPeriod != 0.0f;

		if ( bTimestampsUsable )
		{
			// A batch is admitted only when the previous one has finished (the
			// oversubscription guard), so at most one is ever in flight; a small
			// ring still gives a late readback slack before a slot is reused.
			m_uFramegenQueryRingDepth = 4;
			const VkQueryPoolCreateInfo queryPoolInfo = {
				.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
				.queryType = VK_QUERY_TYPE_TIMESTAMP,
				.queryCount = m_uFramegenQueryRingDepth * 2,
			};
			if ( vk.CreateQueryPool( device(), &queryPoolInfo, nullptr, &m_framegenQueryPool ) == VK_SUCCESS )
			{
				m_uFramegenTimestampValidBits = queueFamilyProps[ nTimestampFamily ].timestampValidBits;
				m_flFramegenTimestampPeriodNs = physProps.limits.timestampPeriod;
				vk_log.infof( "frame generation: measuring GPU time via timestamp queries (period %.2f ns, %u valid bits, %s queue)",
					m_flFramegenTimestampPeriodNs, m_uFramegenTimestampValidBits,
					m_bHasFramegenQueue ? "dedicated" : "shared" );
			}
			else
			{
				m_framegenQueryPool = VK_NULL_HANDLE;
				m_uFramegenQueryRingDepth = 0;
			}
		}

		// Second, independent ring for the dual-GPU staging copy. That copy is
		// always recorded on the composite queue family, which is not necessarily
		// the family the pool above was created for, so it needs its own support
		// check and pool. If this family cannot timestamp, staging copies simply
		// run unmeasured and copy_ms_* stays 0.0.
		const bool bStagingTimestampsUsable = m_queueFamily < nQueueFamilyCount
			&& queueFamilyProps[ m_queueFamily ].timestampValidBits > 0
			&& physProps.limits.timestampPeriod != 0.0f;
		if ( bStagingTimestampsUsable )
		{
			// At most one staging submission per composite, and its result is
			// harvested on the next metrics drain; four slots give several frames
			// of slack before round-robin comes back to an unread one.
			m_uStagingQueryRingDepth = 4;
			const VkQueryPoolCreateInfo queryPoolInfo = {
				.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
				.queryType = VK_QUERY_TYPE_TIMESTAMP,
				.queryCount = m_uStagingQueryRingDepth * 2,
			};
			if ( vk.CreateQueryPool( device(), &queryPoolInfo, nullptr, &m_stagingQueryPool ) == VK_SUCCESS )
			{
				m_uStagingTimestampValidBits = queueFamilyProps[ m_queueFamily ].timestampValidBits;
				m_flStagingTimestampPeriodNs = physProps.limits.timestampPeriod;
			}
			else
			{
				m_stagingQueryPool = VK_NULL_HANDLE;
				m_uStagingQueryRingDepth = 0;
			}
		}
	}

	return true;
}

VkSampler CVulkanDevice::sampler( SamplerState key )
{
	if ( m_samplerCache.count(key) != 0 )
		return m_samplerCache[key];

	VkSampler ret = VK_NULL_HANDLE;

	VkSamplerCreateInfo samplerCreateInfo = {
		.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
		.magFilter = key.bNearest ? VK_FILTER_NEAREST : VK_FILTER_LINEAR,
		.minFilter = key.bNearest ? VK_FILTER_NEAREST : VK_FILTER_LINEAR,
		.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
		.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
		.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
		.borderColor = VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK,
		.unnormalizedCoordinates = key.bUnnormalized,
	};

	vk.CreateSampler( device(), &samplerCreateInfo, nullptr, &ret );

	m_samplerCache[key] = ret;

	return ret;
}

VkPipeline CVulkanDevice::compilePipeline(uint32_t layerCount, uint32_t ycbcrMask, ShaderType type, uint32_t blur_layer_count, uint32_t composite_debug, uint32_t colorspace_mask, uint32_t output_eotf, bool itm_enable)
{
	// Keep these IDs aligned with descriptor_set.h and the framegen shaders.
	// Slot 6 is the bidir trace variant; ITM remains slot 7.
	const std::array<VkSpecializationMapEntry, 8> specializationEntries = {{
		{
			.constantID = 0,
			.offset     = sizeof(uint32_t) * 0,
			.size       = sizeof(uint32_t)
		},
		{
			.constantID = 1,
			.offset     = sizeof(uint32_t) * 1,
			.size       = sizeof(uint32_t)
		},
		{
			.constantID = 2,
			.offset     = sizeof(uint32_t) * 2,
			.size       = sizeof(uint32_t)
		},
		{
			.constantID = 3,
			.offset     = sizeof(uint32_t) * 3,
			.size       = sizeof(uint32_t)
		},
		{
			.constantID = 4,
			.offset     = sizeof(uint32_t) * 4,
			.size       = sizeof(uint32_t)
		},

		{
			.constantID = 5,
			.offset     = sizeof(uint32_t) * 5,
			.size       = sizeof(uint32_t)
		},

		{
			.constantID = 6,
			.offset     = sizeof(uint32_t) * 6,
			.size       = sizeof(uint32_t)
		},
		{
			.constantID = 7,
			.offset     = sizeof(uint32_t) * 7,
			.size       = sizeof(uint32_t)
		},
	}};

	struct {
		uint32_t layerCount;
		uint32_t ycbcrMask;
		uint32_t debug;
		uint32_t blur_layer_count;
		uint32_t colorspace_mask;
		uint32_t output_eotf;
		uint32_t endpoint_trace;
		uint32_t itm_enable;
	} specializationData = {
		.layerCount   = layerCount,
		.ycbcrMask    = ycbcrMask,
		.debug        = composite_debug,
		.blur_layer_count = blur_layer_count,
		.colorspace_mask = colorspace_mask,
		.output_eotf = output_eotf,
		.endpoint_trace = type == SHADER_TYPE_FRAMEGEN_MOTION_BIDIR_TRACE,
		.itm_enable = itm_enable,
	};

	VkSpecializationInfo specializationInfo = {
		.mapEntryCount = uint32_t(specializationEntries.size()),
		.pMapEntries   = specializationEntries.data(),
		.dataSize      = sizeof(specializationData),
		.pData		   = &specializationData,
	};

	VkComputePipelineCreateInfo computePipelineCreateInfo = {
		.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
		.stage = {
			.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
			.stage = VK_SHADER_STAGE_COMPUTE_BIT,
			.module = m_shaderModules[type],
			.pName = "main",
			.pSpecializationInfo = &specializationInfo
		},
		.layout = m_pipelineLayout,
	};

	VkPipeline result;

	VkResult res = vk.CreateComputePipelines(device(), VK_NULL_HANDLE, 1, &computePipelineCreateInfo, nullptr, &result);
	if (res != VK_SUCCESS) {
		vk_errorf( res, "vkCreateComputePipelines failed" );
		return VK_NULL_HANDLE;
	}

	return result;
}

void CVulkanDevice::compileAllPipelines()
{
	pthread_setname_np( pthread_self(), "gamescope-shdr" );

	std::vector<PipelineInfo_t> pipelineInfos = {
		PipelineInfo_t{ SHADER_TYPE_BLIT, k_nMaxLayers, k_nMaxYcbcrMask_ToPreCompile, 1 },
		PipelineInfo_t{ SHADER_TYPE_BLUR, k_nMaxLayers, k_nMaxYcbcrMask_ToPreCompile, k_nMaxBlurLayers },
		PipelineInfo_t{ SHADER_TYPE_BLUR_COND, k_nMaxLayers, k_nMaxYcbcrMask_ToPreCompile, k_nMaxBlurLayers },
		PipelineInfo_t{ SHADER_TYPE_BLUR_FIRST_PASS, 1, 2, 1 },
		PipelineInfo_t{ SHADER_TYPE_RCAS, k_nMaxLayers, k_nMaxYcbcrMask_ToPreCompile, 1 },
		PipelineInfo_t{ SHADER_TYPE_EASU, 1, 1, 1 },
		PipelineInfo_t{ SHADER_TYPE_NIS, 1, 1, 1 },
		PipelineInfo_t{ SHADER_TYPE_RGB_TO_NV12, 1, 1, 1 },
	};

	for (auto& info : pipelineInfos) {
		for (uint32_t layerCount = 1; layerCount <= info.layerCount; layerCount++) {
			for (uint32_t ycbcrMask = 0; ycbcrMask < info.ycbcrMask; ycbcrMask++) {
				for (uint32_t blur_layers = 1; blur_layers <= info.blurLayerCount; blur_layers++) {
					if (ycbcrMask >= (1u << (layerCount + 1)))
						continue;
					if (blur_layers > layerCount)
						continue;

					VkPipeline newPipeline = compilePipeline(layerCount, ycbcrMask, info.shaderType, blur_layers, info.compositeDebug, info.colorspaceMask, info.outputEOTF, info.itmEnable);
					{
						std::lock_guard<std::mutex> lock(m_pipelineMutex);
						PipelineInfo_t key = {info.shaderType, layerCount, ycbcrMask, blur_layers, info.compositeDebug};
						auto result = m_pipelineMap.emplace(std::make_pair(key, newPipeline));
						if (!result.second)
							vk.DestroyPipeline(device(), newPipeline, nullptr);
					}
				}
			}
		}
	}
}

void CVulkanDevice::compileFramegenPipelines()
{
	static constexpr ShaderType pipelines[] = {
		SHADER_TYPE_FRAMEGEN_BLEND,
		SHADER_TYPE_FRAMEGEN_EXTRAPOLATE,
		SHADER_TYPE_FRAMEGEN_EXTRAPOLATE_DIRECT,
		SHADER_TYPE_FRAMEGEN_EXTRAPOLATE_PAIR,
		SHADER_TYPE_FRAMEGEN_MOTION_LUMA_PAIR_RGBA,
		SHADER_TYPE_FRAMEGEN_MOTION_PYRAMID_RGBA,
		SHADER_TYPE_FRAMEGEN_MOTION_MATCH,
		SHADER_TYPE_FRAMEGEN_MOTION_MATCH_REFINE,
		SHADER_TYPE_FRAMEGEN_MOTION_FBCHECK,
		SHADER_TYPE_FRAMEGEN_MOTION_WARP,
		SHADER_TYPE_FRAMEGEN_MOTION_WARP_ACCEL,
		SHADER_TYPE_FRAMEGEN_MOTION_BIDIR,
		SHADER_TYPE_FRAMEGEN_MOTION_BIDIR_TRACE,
		SHADER_TYPE_FRAMEGEN_MOTION_STATS,
		SHADER_TYPE_FRAMEGEN_MOTION_STATS_APPLY,
	};

	for ( ShaderType type : pipelines )
		pipeline( type );
	if ( framegen_hud_level() != 0u )
		pipeline( SHADER_TYPE_FRAMEGEN_HUD );

	// pipeline() caches compile failures, including VK_NULL_HANDLE. The runtime
	// uses this same gate, so unsupported net shaders can never be cached or bound.
	if ( framegen_net_lds_supported() )
	{
		pipeline( SHADER_TYPE_FRAMEGEN_MOTION_NET );
		pipeline( SHADER_TYPE_FRAMEGEN_MOTION_NET_TRAIN );
		pipeline( SHADER_TYPE_FRAMEGEN_MOTION_NET_OPT );
	}

	// R16F storage images require shaderStorageImageExtendedFormats. Keep the
	// always-valid RGBA16F fallback warm on every device, and only create the
	// compact luma pipelines when that feature was enabled at device creation.
	if ( m_bSupportsStorageImageExtendedFormats )
	{
		pipeline( SHADER_TYPE_FRAMEGEN_MOTION_LUMA_PAIR );
		pipeline( SHADER_TYPE_FRAMEGEN_MOTION_PYRAMID );
	}

	if ( m_bSupportsShaderFloat16 )
	{
		pipeline( SHADER_TYPE_FRAMEGEN_EXTRAPOLATE_FP16 );
		pipeline( SHADER_TYPE_FRAMEGEN_EXTRAPOLATE_PAIR_FP16 );
	}
}

extern bool g_bSteamIsActiveWindow;

VkPipeline CVulkanDevice::pipeline(ShaderType type, uint32_t layerCount, uint32_t ycbcrMask, uint32_t blur_layers, uint32_t colorspace_mask, uint32_t output_eotf, bool itm_enable)
{
	uint32_t effective_debug = g_uCompositeDebug;
	if ( g_bSteamIsActiveWindow )
		effective_debug &= ~(CompositeDebugFlag::Heatmap | CompositeDebugFlag::Heatmap_MSWCG | CompositeDebugFlag::Heatmap_Hard);

	std::lock_guard<std::mutex> lock(m_pipelineMutex);
	PipelineInfo_t key = {type, layerCount, ycbcrMask, blur_layers, effective_debug, colorspace_mask, output_eotf, itm_enable};
	auto search = m_pipelineMap.find(key);
	if (search == m_pipelineMap.end())
	{
		VkPipeline result = compilePipeline(layerCount, ycbcrMask, type, blur_layers, effective_debug, colorspace_mask, output_eotf, itm_enable);
		m_pipelineMap[key] = result;
		return result;
	}
	else
	{
		return search->second;
	}
}


int32_t CVulkanDevice::findMemoryType( VkMemoryPropertyFlags properties, uint32_t requiredTypeBits )
{
	for ( uint32_t i = 0; i < m_memoryProperties.memoryTypeCount; i++ )
	{
		if ( ( ( 1u << i ) & requiredTypeBits ) == 0 )
			continue;
		
		if ( ( properties & m_memoryProperties.memoryTypes[ i ].propertyFlags ) != properties )
			continue;
		
		return i;
	}
	
	return -1;
}

std::unique_ptr<CVulkanCmdBuffer> CVulkanDevice::commandBuffer( bool bFramegenQueue )
{
	// Only a cross-family framegen queue needs its own pool/recycle list; in the
	// common same-family case this is the exact same path as before.
	const bool bSplit = bFramegenQueue && framegenFamilySplit();
	std::vector<std::unique_ptr<CVulkanCmdBuffer>> &unusedCmdBufs =
		bSplit ? m_unusedFramegenCmdBufs : m_unusedCmdBufs;

	std::unique_ptr<CVulkanCmdBuffer> cmdBuffer;
	if (unusedCmdBufs.empty())
	{
		VkCommandBuffer rawCmdBuffer;
		VkCommandBufferAllocateInfo commandBufferAllocateInfo = {
			.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
			.commandPool = bSplit ? m_framegenCommandPool : m_commandPool,
			.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
			.commandBufferCount = 1
		};

		VkResult res = vk.AllocateCommandBuffers( device(), &commandBufferAllocateInfo, &rawCmdBuffer );
		if ( res != VK_SUCCESS )
		{
			vk_errorf( res, "vkAllocateCommandBuffers failed" );
			return nullptr;
		}

		cmdBuffer = std::make_unique<CVulkanCmdBuffer>(this, rawCmdBuffer,
			bSplit ? m_framegenQueue : queue(),
			bSplit ? m_framegenQueueFamily : queueFamily());
	}
	else
	{
		cmdBuffer = std::move(unusedCmdBufs.back());
		unusedCmdBufs.pop_back();
	}

	cmdBuffer->begin();
	return cmdBuffer;
}

uint64_t CVulkanDevice::submitInternal( CVulkanCmdBuffer* cmdBuffer )
{
	cmdBuffer->end();

	// The seq no of the last submission.
	const uint64_t lastSubmissionSeqNo = m_submissionSeqNo++;

	// This is the seq no of the command buffer we are going to submit.
	const uint64_t nextSeqNo = lastSubmissionSeqNo + 1;

	static constexpr size_t k_uInlineSubmitSemaphores = 8;
	const auto &externalSignals = cmdBuffer->GetExternalSignals();
	const auto &externalWaits = cmdBuffer->GetExternalDependencies();
	const size_t signalCount = 1 + externalSignals.size();
	const size_t waitCount = externalWaits.size();

	InlineSubmitArray<VkSemaphore, k_uInlineSubmitSemaphores> signalSemaphores;
	InlineSubmitArray<uint64_t, k_uInlineSubmitSemaphores> signalPoints;
	InlineSubmitArray<VkSemaphore, k_uInlineSubmitSemaphores> waitSemaphores;
	InlineSubmitArray<uint64_t, k_uInlineSubmitSemaphores> waitPoints;
	InlineSubmitArray<VkPipelineStageFlags, k_uInlineSubmitSemaphores> waitStageFlags;
	VkSemaphore *pSignalSemaphores = signalSemaphores.storage( signalCount );
	uint64_t *pSignalPoints = signalPoints.storage( signalCount );
	VkSemaphore *pWaitSemaphores = waitSemaphores.storage( waitCount );
	uint64_t *pWaitPoints = waitPoints.storage( waitCount );
	VkPipelineStageFlags *pWaitStageFlags = waitStageFlags.storage( waitCount );

	pSignalSemaphores[0] = m_scratchTimelineSemaphore;
	pSignalPoints[0] = nextSeqNo;
	for ( size_t i = 0; i < externalSignals.size(); i++ )
	{
		pSignalSemaphores[i + 1] = externalSignals[i].pTimelineSemaphore->pVkSemaphore;
		pSignalPoints[i + 1] = externalSignals[i].ulPoint;
	}

	for ( size_t i = 0; i < externalWaits.size(); i++ )
	{
		pWaitSemaphores[i] = externalWaits[i].pTimelineSemaphore->pVkSemaphore;
		pWaitPoints[i] = externalWaits[i].ulPoint;
		// TRANSFER is required as well as the shading stages: a non-device-local
		// client import is first read by vkCmdCopyImage in the dual-gpu staging
		// path (and in the ReShade pre-stage submission), so without it the
		// transfer read may legally run before the client's acquire point is
		// signalled and copy a stale or partially written image.
		pWaitStageFlags[i] = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT;
	}

	VkTimelineSemaphoreSubmitInfo timelineInfo = {
		.sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO,
		// no need to ensure order of cmd buffer submission, we only have one queue
		.waitSemaphoreValueCount = static_cast<uint32_t>( waitCount ),
		.pWaitSemaphoreValues = pWaitPoints,
		.signalSemaphoreValueCount = static_cast<uint32_t>( signalCount ),
		.pSignalSemaphoreValues = pSignalPoints,
	};

	VkCommandBuffer rawCmdBuffer = cmdBuffer->rawBuffer();

	VkSubmitInfo submitInfo = {
		.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
		.pNext = &timelineInfo,
		.waitSemaphoreCount = static_cast<uint32_t>( waitCount ),
		.pWaitSemaphores = pWaitSemaphores,
		.pWaitDstStageMask = pWaitStageFlags,
		.commandBufferCount = 1,
		.pCommandBuffers = &rawCmdBuffer,
		.signalSemaphoreCount = static_cast<uint32_t>( signalCount ),
		.pSignalSemaphores = pSignalSemaphores,
	};

	vk_check( vk.QueueSubmit( cmdBuffer->queue(), 1, &submitInfo, VK_NULL_HANDLE ) );

	return nextSeqNo;
}

uint64_t CVulkanDevice::submit( std::unique_ptr<CVulkanCmdBuffer> cmdBuffer)
{
	uint64_t nextSeqNo = submitInternal(cmdBuffer.get());
	m_pendingCmdBufs.emplace_back(nextSeqNo, std::move(cmdBuffer));
	return nextSeqNo;
}

void CVulkanDevice::garbageCollect( void )
{
	uint64_t currentSeqNo;
	vk_check( vk.GetSemaphoreCounterValue(device(), m_scratchTimelineSemaphore, &currentSeqNo) );
	cache_timeline_completion( m_submissionCompletedSeqNo, currentSeqNo );

	resetCmdBuffers(currentSeqNo);
}

VulkanTimelineSemaphore_t::~VulkanTimelineSemaphore_t()
{
	if ( pVkSemaphore != VK_NULL_HANDLE )
	{
		pDevice->vk.DestroySemaphore( pDevice->device(), pVkSemaphore, nullptr );
		pVkSemaphore = VK_NULL_HANDLE;
	}
}

int VulkanTimelineSemaphore_t::GetFd() const
{
	const VkSemaphoreGetFdInfoKHR semaphoreGetInfo =
	{
		.sType = VK_STRUCTURE_TYPE_SEMAPHORE_GET_FD_INFO_KHR,
		.semaphore = pVkSemaphore,
		.handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT,
	};

	int32_t nFd = -1;
	VkResult res = VK_SUCCESS;
	if ( ( res = pDevice->vk.GetSemaphoreFdKHR( pDevice->device(), &semaphoreGetInfo, &nFd ) ) != VK_SUCCESS )
	{
		vk_errorf( res, "vkGetSemaphoreFdKHR failed" );
		return -1;
	}

	return nFd;
}

std::shared_ptr<VulkanTimelineSemaphore_t> CVulkanDevice::CreateTimelineSemaphore( uint64_t ulStartPoint, bool bShared )
{
	std::shared_ptr<VulkanTimelineSemaphore_t> pSemaphore = std::make_unique<VulkanTimelineSemaphore_t>();
	pSemaphore->pDevice = this;

	VkSemaphoreCreateInfo createInfo =
	{
		.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
	};

	VkSemaphoreTypeCreateInfo typeInfo =
	{
		.sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO,
		.pNext = std::exchange( createInfo.pNext, &typeInfo ),
		.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE,
		.initialValue = ulStartPoint,
	};

	VkExportSemaphoreCreateInfo exportInfo =
	{
		.sType = VK_STRUCTURE_TYPE_EXPORT_SEMAPHORE_CREATE_INFO,
		.pNext = bShared ? std::exchange( createInfo.pNext, &exportInfo ) : nullptr,
		// This is a syncobj fd for any drivers using syncobj.
		.handleTypes = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT,
	};

	VkResult res;
	if ( ( res = vk.CreateSemaphore( m_device, &createInfo, nullptr, &pSemaphore->pVkSemaphore ) ) != VK_SUCCESS )
	{
		vk_errorf( res, "vkCreateSemaphore failed" );
		return nullptr;
	}

	return pSemaphore;
}

std::shared_ptr<VulkanTimelineSemaphore_t> CVulkanDevice::ImportTimelineSemaphore( gamescope::CTimeline *pTimeline )
{
	std::shared_ptr<VulkanTimelineSemaphore_t> pSemaphore = std::make_unique<VulkanTimelineSemaphore_t>();
	pSemaphore->pDevice = this;

	const VkSemaphoreTypeCreateInfo typeInfo =
	{
		.sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO,
		.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE,
	};

	const VkSemaphoreCreateInfo createInfo =
	{
		.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
		.pNext = &typeInfo,
	};

	VkResult res;
	if ( ( res = vk.CreateSemaphore( m_device, &createInfo, nullptr, &pSemaphore->pVkSemaphore ) ) != VK_SUCCESS )
	{
		vk_errorf( res, "vkCreateSemaphore failed" );
		return nullptr;
	}

    // "Importing a semaphore payload from a file descriptor transfers
    // ownership of the file descriptor from the application to the Vulkan
    // implementation. The application must not perform any operations on
    // the file descriptor after a successful import."
	//
	// Thus, we must dup.

	VkImportSemaphoreFdInfoKHR importFdInfo =
	{
		.sType = VK_STRUCTURE_TYPE_IMPORT_SEMAPHORE_FD_INFO_KHR,
		.pNext = nullptr,
		.semaphore = pSemaphore->pVkSemaphore,
		.flags = 0, // not temporary
		.handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT,
		.fd = dup( pTimeline->GetSyncobjFd() ),
	};
	if ( ( res = vk.ImportSemaphoreFdKHR( m_device, &importFdInfo ) ) != VK_SUCCESS )
	{
		vk_errorf( res, "vkImportSemaphoreFdKHR failed" );
		return nullptr;
	}

	return pSemaphore;
}

void CVulkanCmdBuffer::AddDependency( std::shared_ptr<VulkanTimelineSemaphore_t> pTimelineSemaphore, uint64_t ulPoint )
{
	m_ExternalDependencies.emplace_back( std::move( pTimelineSemaphore ), ulPoint );
}

void CVulkanCmdBuffer::AddSignal( std::shared_ptr<VulkanTimelineSemaphore_t> pTimelineSemaphore, uint64_t ulPoint )
{
	m_ExternalSignals.emplace_back( std::move( pTimelineSemaphore ), ulPoint );
}

void CVulkanDevice::wait(uint64_t sequence, bool reset)
{
	if (m_submissionSeqNo == sequence)
		m_uploadBufferOffset = 0;

	VkSemaphoreWaitInfo waitInfo = {
		.sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO,
		.semaphoreCount = 1,
		.pSemaphores = &m_scratchTimelineSemaphore,
		.pValues = &sequence,
	} ;

	vk_check( vk.WaitSemaphores( device(), &waitInfo, ~0ull ) );
	cache_timeline_completion( m_submissionCompletedSeqNo, sequence );

	if (reset)
		resetCmdBuffers(sequence);
}

void CVulkanDevice::waitIdle(bool reset)
{
	wait(m_submissionSeqNo, reset);

	// The dedicated framegen queue signals its own timeline, so a composite-queue
	// wait does not cover it. Drain it too and recycle its command buffers, so
	// teardown/remake paths (which call waitIdle before freeing the output ring
	// and framegen pools those buffers still reference) never free GPU-in-flight
	// resources. Not on any per-frame path.
	if ( m_bHasFramegenQueue )
		waitFramegen( m_framegenSeqNo );
	// Dedicated buffers and either queue mode's completed timing association are
	// now safe to recycle/consume. This is still outside every per-frame path.
	framegenGarbageCollect();
}

bool CVulkanDevice::hasCompleted(uint64_t sequence)
{
	if ( sequence == 0
		|| m_submissionCompletedSeqNo.load( std::memory_order_relaxed ) >= sequence )
		return true;

	uint64_t currentSeqNo = 0;
	vk_check( vk.GetSemaphoreCounterValue(device(), m_scratchTimelineSemaphore, &currentSeqNo) );
	cache_timeline_completion( m_submissionCompletedSeqNo, currentSeqNo );
	return currentSeqNo >= sequence;
}

void CVulkanDevice::resetCmdBuffers(uint64_t sequence)
{
	// Submission sequence numbers are monotonic, so a contiguous vector is both
	// cheaper and more predictable than allocating one tree node per frame.
	// Retire every tracked command buffer covered by the observed timeline value;
	// untracked submitInternal callers can legitimately leave gaps in the keys.
	auto completedEnd = m_pendingCmdBufs.begin();
	while ( completedEnd != m_pendingCmdBufs.end() && completedEnd->first <= sequence )
	{
		completedEnd->second->reset();
		m_unusedCmdBufs.push_back(std::move(completedEnd->second));
		++completedEnd;
	}

	m_pendingCmdBufs.erase(m_pendingCmdBufs.begin(), completedEnd);
}

CVulkanCmdBuffer::CVulkanCmdBuffer(CVulkanDevice *parent, VkCommandBuffer cmdBuffer, VkQueue queue, uint32_t queueFamily)
	: m_cmdBuffer(cmdBuffer), m_device(parent), m_queue(queue), m_queueFamily(queueFamily)
{
	m_textureRefs.reserve( k_uInitialTrackedTextureCapacity );
	m_textureState.reserve( k_uInitialTrackedTextureCapacity );
	m_imageBarriers.reserve( VKR_SAMPLER_SLOTS + VKR_TARGET_SLOTS );
	m_imageUses.reserve( VKR_SAMPLER_SLOTS + VKR_TARGET_SLOTS );
	m_imageBarriers2.reserve( VKR_SAMPLER_SLOTS + VKR_TARGET_SLOTS );
}

CVulkanCmdBuffer::~CVulkanCmdBuffer()
{
	m_device->vk.FreeCommandBuffers(m_device->device(), m_device->commandPool(), 1, &m_cmdBuffer);
}

void CVulkanCmdBuffer::reset()
{
	vk_check( m_device->vk.ResetCommandBuffer(m_cmdBuffer, 0) );
	m_textureRefs.clear();
	m_textureState.clear();

	m_ExternalDependencies.clear();
	m_ExternalSignals.clear();

	m_bFramegen = false;
	m_uFramegenDispatchCount = 0;
}

void CVulkanCmdBuffer::begin()
{
	VkCommandBufferBeginInfo commandBufferBeginInfo = {
		.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
		.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT
	};

	vk_check( m_device->vk.BeginCommandBuffer(m_cmdBuffer, &commandBufferBeginInfo) );

	clearState();
}

void CVulkanCmdBuffer::end()
{
	insertBarrier(true);
	vk_check( m_device->vk.EndCommandBuffer(m_cmdBuffer) );
}

void CVulkanCmdBuffer::bindTexture(uint32_t slot, gamescope::Rc<CVulkanTexture> texture)
{
	m_boundTextures[slot] = texture.get();
	if (texture)
		m_textureRefs.emplace_back(std::move(texture));
}

void CVulkanCmdBuffer::bindColorMgmtLuts(uint32_t slot, gamescope::Rc<CVulkanTexture> lut1d, gamescope::Rc<CVulkanTexture> lut3d)
{
	m_shaperLut[slot] = lut1d.get();
	m_lut3D[slot] = lut3d.get();

	if (lut1d != nullptr)
		m_textureRefs.emplace_back(std::move(lut1d));
	if (lut3d != nullptr)
		m_textureRefs.emplace_back(std::move(lut3d));
}

void CVulkanCmdBuffer::setTextureSrgb(uint32_t slot, bool srgb)
{
	m_useSrgb[slot] = srgb;
}

void CVulkanCmdBuffer::setSamplerNearest(uint32_t slot, bool nearest)
{
	m_samplerState[slot].bNearest = nearest;
}

void CVulkanCmdBuffer::setSamplerUnnormalized(uint32_t slot, bool unnormalized)
{
	m_samplerState[slot].bUnnormalized = unnormalized;
}

void CVulkanCmdBuffer::bindTarget(gamescope::Rc<CVulkanTexture> target)
{
	m_target = target.get();
	m_target2 = nullptr;
	if (target)
		m_textureRefs.emplace_back(std::move(target));
}

void CVulkanCmdBuffer::bindTarget2(gamescope::Rc<CVulkanTexture> target)
{
	m_target2 = target.get();
	if (target)
		m_textureRefs.emplace_back(std::move(target));
}

void CVulkanCmdBuffer::clearState()
{
	for (auto& texture : m_boundTextures)
		texture = nullptr;

	for (auto& sampler : m_samplerState)
		sampler = {};

	for (auto& lut : m_shaperLut)
		lut = nullptr;

	for (auto& lut : m_lut3D)
		lut = nullptr;

	m_target = nullptr;
	m_target2 = nullptr;
	m_renderBufferOffset = 0;
	m_renderBufferSize = 0;
	m_useSrgb.reset();
}

void CVulkanCmdBuffer::bindUploadedConstants(uint32_t offset, uint32_t size)
{
	m_renderBufferOffset = offset;
	m_renderBufferSize = size;
}

template<class PushData, class... Args>
void CVulkanCmdBuffer::uploadConstants(Args&&... args)
{
	PushData data(std::forward<Args>(args)...);

	auto [ptr, offset] = m_device->uploadUniformBufferData(sizeof(data));
	m_renderBufferOffset = offset;
	m_renderBufferSize = sizeof(data);
	memcpy(ptr, &data, sizeof(data));
}

template<class PushData, class... Args>
void CVulkanCmdBuffer::pushConstants(Args&&... args)
{
	PushData data(std::forward<Args>(args)...);
	static_assert( sizeof(PushData) <= k_uFramegenPushConstantSize, "framegen push constants exceed reserved range" );
	// Recorded straight into the command buffer, so this needs no upload arena
	// slice and cannot race the composite path's bump allocator across queues.
	m_device->vk.CmdPushConstants( m_cmdBuffer, m_device->pipelineLayout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(data), &data );
}

void CVulkanCmdBuffer::bindPipeline(VkPipeline pipeline)
{
	m_device->vk.CmdBindPipeline(m_cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
}

void CVulkanCmdBuffer::dispatch(uint32_t x, uint32_t y, uint32_t z)
{
	if ( m_bFramegen )
	{
		const uint32_t uCapacity = m_device->framegenDescriptorSetCapacity();
		if ( m_uFramegenDispatchCount >= uCapacity ) [[unlikely]]
		{
			vk_log.errorf( "framegen descriptor ring exhausted: dispatch %u exceeds capacity %u",
				m_uFramegenDispatchCount + 1, uCapacity );
			abort();
		}
		m_uFramegenDispatchCount++;
	}

	assert(m_target != nullptr);
	emitPreDispatchBarriers();

	VkDescriptorSet descriptorSet = m_device->descriptorSet( m_bFramegen );

	std::array<VkWriteDescriptorSet, 7> writeDescriptorSets;
	std::array<VkDescriptorImageInfo, VKR_SAMPLER_SLOTS> imageDescriptors = {};
	std::array<VkDescriptorImageInfo, VKR_SAMPLER_SLOTS> ycbcrImageDescriptors = {};
	std::array<VkDescriptorImageInfo, VKR_TARGET_SLOTS> targetDescriptors = {};
	std::array<VkDescriptorImageInfo, VKR_LUT3D_COUNT> shaperLutDescriptor = {};
	std::array<VkDescriptorImageInfo, VKR_LUT3D_COUNT> lut3DDescriptor = {};
	VkDescriptorBufferInfo scratchDescriptor = {};

	writeDescriptorSets[0] = {
		.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
		.dstSet = descriptorSet,
		.dstBinding = 0,
		.dstArrayElement = 0,
		.descriptorCount = 1,
		.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
		.pBufferInfo = &scratchDescriptor,
	};

	writeDescriptorSets[1] = {
		.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
		.dstSet = descriptorSet,
		.dstBinding = 1,
		.dstArrayElement = 0,
		.descriptorCount = 1,
		.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
		.pImageInfo = &targetDescriptors[0],
	};

	writeDescriptorSets[2] = {
		.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
		.dstSet = descriptorSet,
		.dstBinding = 2,
		.dstArrayElement = 0,
		.descriptorCount = 1,
		.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
		.pImageInfo = &targetDescriptors[1],
	};

	writeDescriptorSets[3] = {
		.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
		.dstSet = descriptorSet,
		.dstBinding = 3,
		.dstArrayElement = 0,
		.descriptorCount = imageDescriptors.size(),
		.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
		.pImageInfo = imageDescriptors.data(),
	};

	writeDescriptorSets[4] = {
		.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
		.dstSet = descriptorSet,
		.dstBinding = 4,
		.dstArrayElement = 0,
		.descriptorCount = ycbcrImageDescriptors.size(),
		.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
		.pImageInfo = ycbcrImageDescriptors.data(),
	};

	writeDescriptorSets[5] = {
		.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
		.dstSet = descriptorSet,
		.dstBinding = 5,
		.dstArrayElement = 0,
		.descriptorCount = shaperLutDescriptor.size(),
		.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
		.pImageInfo = shaperLutDescriptor.data(),
	};

	writeDescriptorSets[6] = {
		.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
		.dstSet = descriptorSet,
		.dstBinding = 6,
		.dstArrayElement = 0,
		.descriptorCount = lut3DDescriptor.size(),
		.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
		.pImageInfo = lut3DDescriptor.data(),
	};

	scratchDescriptor.buffer = m_device->m_uploadBuffer;
	scratchDescriptor.offset = m_renderBufferOffset;
	// VK_WHOLE_SIZE makes the effective UBO range extend to the end of the
	// 8 MiB upload arena, exceeding maxUniformBufferRange on devices such as
	// NVIDIA's 64 KiB implementation. Bind only the constants this dispatch uses.
	// Push-constant-only framegen shaders still need a valid, non-zero descriptor
	// because all pipelines share this descriptor-set layout.
	scratchDescriptor.range = m_renderBufferSize != 0 ? m_renderBufferSize : 16u;

	for (uint32_t i = 0; i < VKR_SAMPLER_SLOTS; i++)
	{
		imageDescriptors[i].sampler = m_device->sampler(m_samplerState[i]);
		imageDescriptors[i].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
		ycbcrImageDescriptors[i].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
		if (m_boundTextures[i] == nullptr)
			continue;

		VkImageView view = m_useSrgb[i] ? m_boundTextures[i]->srgbView() : m_boundTextures[i]->linearView();

		if (m_boundTextures[i]->format() == VK_FORMAT_G8_B8R8_2PLANE_420_UNORM)
			ycbcrImageDescriptors[i].imageView = view;
		else
			imageDescriptors[i].imageView = view;
	}

	for (uint32_t i = 0; i < VKR_LUT3D_COUNT; i++)
	{
		SamplerState linearState;
		linearState.bNearest = false;
		linearState.bUnnormalized = false;
		SamplerState nearestState; // TODO(Josh): Probably want to do this when I bring in tetrahedral interpolation.
		nearestState.bNearest = true;
		nearestState.bUnnormalized = false;

		shaperLutDescriptor[i].sampler = m_device->sampler(linearState);
		shaperLutDescriptor[i].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
		// TODO(Josh): I hate the fact that srgbView = view *as* raw srgb and treat as linear.
		// I need to change this, it's so utterly stupid and confusing.
		shaperLutDescriptor[i].imageView = m_shaperLut[i] ? m_shaperLut[i]->srgbView() : VK_NULL_HANDLE;

		lut3DDescriptor[i].sampler = m_device->sampler(nearestState);
		lut3DDescriptor[i].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
		lut3DDescriptor[i].imageView = m_lut3D[i] ? m_lut3D[i]->srgbView() : VK_NULL_HANDLE;
	}

	if (!m_target->isYcbcr())
	{
		targetDescriptors[0].imageView = m_target->srgbView();
		targetDescriptors[0].imageLayout = VK_IMAGE_LAYOUT_GENERAL;

		// Optional second RGBA target (framegen paired shader). Reuses the chroma
		// descriptor slot, which is otherwise unused for non-YCbCr compute.
		if (m_target2)
		{
			targetDescriptors[1].imageView = m_target2->srgbView();
			targetDescriptors[1].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
		}
	}
	else
	{
		targetDescriptors[0].imageView = m_target->lumaView();
		targetDescriptors[0].imageLayout = VK_IMAGE_LAYOUT_GENERAL;

		targetDescriptors[1].imageView = m_target->chromaView();
		targetDescriptors[1].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
	}

	m_device->vk.UpdateDescriptorSets(m_device->device(), writeDescriptorSets.size(), writeDescriptorSets.data(), 0, nullptr);

	m_device->vk.CmdBindDescriptorSets(m_cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_device->pipelineLayout(), 0, 1, &descriptorSet, 0, nullptr);

	m_device->vk.CmdDispatch(m_cmdBuffer, x, y, z);

	markDirty(m_target);
	if (m_target2)
		markDirty(m_target2);
}

void CVulkanCmdBuffer::copyImage(gamescope::Rc<CVulkanTexture> src, gamescope::Rc<CVulkanTexture> dst)
{
	assert(src->width() == dst->width());
	assert(src->height() == dst->height());
	prepareSrcImage(src.get());
	prepareDestImage(dst.get());
	if ( !m_device->supportsSync2() )
	{
		insertBarrier();
	}
	else
	{
		m_imageUses.clear();
		addImageUse( src.get(), VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_READ_BIT, false );
		addImageUse( dst.get(), VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT, true );
		emitSync2Barriers();
	}

	VkImageCopy region = {
		.srcSubresource = {
			.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
			.layerCount = 1
		},
		.dstSubresource = {
			.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
			.layerCount = 1
		},
		.extent = {
			.width = src->width(),
			.height = src->height(),
			.depth = 1
		},
	};

	m_device->vk.CmdCopyImage(m_cmdBuffer, src->vkImage(), VK_IMAGE_LAYOUT_GENERAL, dst->vkImage(), VK_IMAGE_LAYOUT_GENERAL, 1, &region);

	markDirty(dst.get());
	m_textureRefs.emplace_back(std::move(src));
	m_textureRefs.emplace_back(std::move(dst));
}

void CVulkanCmdBuffer::copyBufferToImage(VkBuffer buffer, VkDeviceSize offset, uint32_t stride, gamescope::Rc<CVulkanTexture> dst)
{
	prepareDestImage(dst.get());
	if ( !m_device->supportsSync2() )
	{
		insertBarrier();
	}
	else
	{
		m_imageUses.clear();
		addImageUse( dst.get(), VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT, true );
		emitSync2Barriers();
	}
	VkBufferImageCopy region = {
		.bufferOffset = offset,
		.bufferRowLength = stride,
		.imageSubresource = {
			.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
			.layerCount = 1,
		},
		.imageExtent = {
			.width = dst->width(),
			.height = dst->height(),
			.depth = dst->depth(),
		},
	};

	m_device->vk.CmdCopyBufferToImage(m_cmdBuffer, buffer, dst->vkImage(), VK_IMAGE_LAYOUT_GENERAL, 1, &region);

	markDirty(dst.get());

	m_textureRefs.emplace_back(std::move(dst));
}

void CVulkanCmdBuffer::addImageUse( CVulkanTexture *image, VkPipelineStageFlags2 stage, VkAccessFlags2 access, bool bWrite )
{
	for ( ImageUse &use : m_imageUses )
	{
		if ( use.pTexture != image )
			continue;

		use.stage |= stage;
		use.access |= access;
		use.bWrite |= bWrite;
		return;
	}

	m_imageUses.push_back( ImageUse{
		.pTexture = image,
		.stage = stage,
		.access = access,
		.bWrite = bWrite,
	} );
}

void CVulkanCmdBuffer::emitExternalAcquireBarriers()
{
	m_imageBarriers.clear();

	const uint32_t externalQueue = m_device->supportsModifiers() ? VK_QUEUE_FAMILY_FOREIGN_EXT : VK_QUEUE_FAMILY_EXTERNAL_KHR;
	const VkImageSubresourceRange subResRange = {
		.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
		.levelCount = 1,
		.layerCount = 1,
	};
	const VkAccessFlags writeBits = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
	const VkAccessFlags readBits = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT;

	for ( TrackedTextureState &tracked : m_textureState )
	{
		CVulkanTexture *image = tracked.pTexture;
		TextureState &state = tracked.state;
		if ( !state.needsImport )
			continue;

		if ( image->queueFamily == VK_QUEUE_FAMILY_IGNORED )
			image->queueFamily = m_queueFamily;

		m_imageBarriers.push_back( VkImageMemoryBarrier{
			.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
			.srcAccessMask = state.dirty ? writeBits : 0u,
			.dstAccessMask = readBits | writeBits,
			.oldLayout = state.discarded ? VK_IMAGE_LAYOUT_UNDEFINED : VK_IMAGE_LAYOUT_GENERAL,
			.newLayout = VK_IMAGE_LAYOUT_GENERAL,
			.srcQueueFamilyIndex = externalQueue,
			// A concurrent image has no owning family to acquire into; the
			// non-external side must be VK_QUEUE_FAMILY_IGNORED
			// (VUID-VkImageMemoryBarrier-image-04071).
			.dstQueueFamilyIndex = image->concurrentSharing() ? VK_QUEUE_FAMILY_IGNORED : m_queueFamily,
			.image = image->vkImage(),
			.subresourceRange = subResRange,
		} );

		// Preserve the synchronization1 import acquire exactly; synchronization2
		// below only tracks uses recorded after ownership has reached this queue.
		state.discarded = false;
		state.dirty = false;
		state.needsImport = false;
	}

	if ( !m_imageBarriers.empty() )
	{
		m_device->vk.CmdPipelineBarrier( m_cmdBuffer,
			VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
			0, 0, nullptr, 0, nullptr, m_imageBarriers.size(), m_imageBarriers.data() );
	}
}

void CVulkanCmdBuffer::emitSync2Barriers()
{
	emitExternalAcquireBarriers();
	m_imageBarriers2.clear();

	const VkImageSubresourceRange subResRange = {
		.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
		.levelCount = 1,
		.layerCount = 1,
	};

	for ( const ImageUse &use : m_imageUses )
	{
		TextureState *state = findTextureState( use.pTexture );
		assert( state != nullptr );
		if ( use.pTexture->queueFamily == VK_QUEUE_FAMILY_IGNORED )
			use.pTexture->queueFamily = m_queueFamily;

		const bool bHasPriorUse = state->lastAccess != 0;
		const bool bWriteHazard = bHasPriorUse && state->bLastWasWrite;
		const bool bReadToWriteHazard = bHasPriorUse && !state->bLastWasWrite && use.bWrite;
		if ( state->discarded || bWriteHazard || bReadToWriteHazard )
		{
			m_imageBarriers2.push_back( VkImageMemoryBarrier2{
				.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
				.srcStageMask = state->discarded ? VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT : state->lastStage,
				.srcAccessMask = !state->discarded && bWriteHazard ? state->lastAccess : 0u,
				.dstStageMask = use.stage,
				.dstAccessMask = use.access,
				.oldLayout = state->discarded ? VK_IMAGE_LAYOUT_UNDEFINED : VK_IMAGE_LAYOUT_GENERAL,
				.newLayout = VK_IMAGE_LAYOUT_GENERAL,
				.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
				.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
				.image = use.pTexture->vkImage(),
				.subresourceRange = subResRange,
			} );
		}
	}

	if ( !m_imageBarriers2.empty() )
	{
		const VkDependencyInfo dependencyInfo = {
			.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
			.imageMemoryBarrierCount = static_cast<uint32_t>( m_imageBarriers2.size() ),
			.pImageMemoryBarriers = m_imageBarriers2.data(),
		};
		m_device->vk.CmdPipelineBarrier2( m_cmdBuffer, &dependencyInfo );
	}

	for ( const ImageUse &use : m_imageUses )
	{
		TextureState *state = findTextureState( use.pTexture );
		assert( state != nullptr );
		const bool bHasPriorUse = state->lastAccess != 0;
		// A clean first use needs no barrier: submission order plus the producing
		// command buffer's 3d951b7 flush barrier supplies cross-buffer visibility.
		// Keep consecutive read stages accumulated so a later WAR waits on every
		// outstanding reader even though read -> read itself needs no barrier.
		if ( bHasPriorUse && !state->bLastWasWrite && !use.bWrite )
		{
			state->lastStage |= use.stage;
			state->lastAccess |= use.access;
		}
		else
		{
			state->lastStage = use.stage;
			state->lastAccess = use.access;
		}
		state->bLastWasWrite = use.bWrite;
		state->discarded = false;
		// Keep dirty latched: insertBarrier(true) must publish every image written
		// in this buffer, including writes already consumed by a narrower RAW here.
	}
}

void CVulkanCmdBuffer::emitPreDispatchBarriers()
{
	for ( CVulkanTexture *src : m_boundTextures )
	{
		if ( src )
			prepareSrcImage( src );
	}
	prepareDestImage( m_target );
	if ( m_target2 )
		prepareDestImage( m_target2 );

	if ( !m_device->supportsSync2() )
	{
		insertBarrier();
		return;
	}

	m_imageUses.clear();
	for ( CVulkanTexture *src : m_boundTextures )
	{
		if ( src )
			addImageUse( src, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT, false );
	}
	const VkAccessFlags2 storageAccess = VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
	addImageUse( m_target, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, storageAccess, true );
	if ( m_target2 )
		addImageUse( m_target2, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, storageAccess, true );
	emitSync2Barriers();
}

void CVulkanCmdBuffer::prepareSrcImage(CVulkanTexture *image)
{
	auto [state, inserted] = trackTexture( image );
	// no need to reimport if the image didn't change
	if (!inserted)
		return;
	state->needsImport = image->externalImage();
	state->needsExport = image->externalImage();
}

void CVulkanCmdBuffer::prepareDestImage(CVulkanTexture *image)
{
	auto [state, inserted] = trackTexture( image );
	// no need to discard if the image is already image/in the correct layout
	if (!inserted)
		return;
	state->discarded = true;
	state->needsExport = image->externalImage();
	state->needsPresentLayout = image->outputImage();
}

void CVulkanCmdBuffer::discardImage(CVulkanTexture *image)
{
	auto [state, inserted] = trackTexture( image );
	if (!inserted)
		return;
	state->discarded = true;
}

void CVulkanCmdBuffer::markDirty(CVulkanTexture *image)
{
	TextureState *state = findTextureState( image );
	// image should have been prepared already
	assert(state != nullptr);
	state->dirty = true;
}

std::pair<TextureState *, bool> CVulkanCmdBuffer::trackTexture( CVulkanTexture *image )
{
	if ( TextureState *state = findTextureState( image ) )
		return { state, false };

	m_textureState.push_back( TrackedTextureState{ image, TextureState{} } );
	return { &m_textureState.back().state, true };
}

TextureState *CVulkanCmdBuffer::findTextureState( CVulkanTexture *image )
{
	for ( TrackedTextureState &tracked : m_textureState )
	{
		if ( tracked.pTexture == image )
			return &tracked.state;
	}
	return nullptr;
}

void CVulkanCmdBuffer::insertBarrier(bool flush)
{
	m_imageBarriers.clear();

	uint32_t externalQueue = m_device->supportsModifiers() ? VK_QUEUE_FAMILY_FOREIGN_EXT : VK_QUEUE_FAMILY_EXTERNAL_KHR;

	VkImageSubresourceRange subResRange =
	{
		.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
		.levelCount = 1,
		.layerCount = 1
	};

	for ( TrackedTextureState &tracked : m_textureState )
	{
		CVulkanTexture *image = tracked.pTexture;
		TextureState& state = tracked.state;
		assert(!flush || !state.needsImport);

		bool isExport = flush && state.needsExport;
		bool isPresent = flush && state.needsPresentLayout;

		if (!state.discarded && !state.dirty && !state.needsImport && !isExport && !isPresent)
			continue;

		const VkAccessFlags write_bits = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
		const VkAccessFlags read_bits = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT;
		// Same-queue submission order supplies execution order, not memory
		// visibility, so dirty internal images need a destination scope at flush.
		const bool bInternalFlush = flush && state.dirty && !isExport && !isPresent;

		if (image->queueFamily == VK_QUEUE_FAMILY_IGNORED)
			image->queueFamily = m_queueFamily;

		// A VK_SHARING_MODE_CONCURRENT image has no owning family to transfer, so
		// naming one is invalid: both indices must be VK_QUEUE_FAMILY_IGNORED
		// unless the other side is EXTERNAL/FOREIGN, in which case only the
		// non-external side must be IGNORED (VUID-VkImageMemoryBarrier-image-04071
		// / -04072). Release/acquire against FOREIGN stays a real transfer for
		// exclusive images.
		const bool bConcurrent = image->concurrentSharing();
		const uint32_t srcQueueFamily = state.needsImport ? externalQueue
			: bConcurrent ? VK_QUEUE_FAMILY_IGNORED : image->queueFamily;
		const uint32_t dstQueueFamily = isExport ? externalQueue
			: bConcurrent ? VK_QUEUE_FAMILY_IGNORED : m_queueFamily;

		VkImageMemoryBarrier memoryBarrier =
		{
			.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
			.srcAccessMask = state.dirty ? write_bits : 0u,
			.dstAccessMask = !flush || bInternalFlush ? read_bits | write_bits : 0u,
			.oldLayout = state.discarded ? VK_IMAGE_LAYOUT_UNDEFINED : VK_IMAGE_LAYOUT_GENERAL,
			.newLayout = isPresent ? GetBackend()->GetPresentLayout() : VK_IMAGE_LAYOUT_GENERAL,
			.srcQueueFamilyIndex = srcQueueFamily,
			.dstQueueFamilyIndex = dstQueueFamily,
			.image = image->vkImage(),
			.subresourceRange = subResRange
		};

		m_imageBarriers.push_back(memoryBarrier);

		state.discarded = false;
		state.dirty = false;
		state.needsImport = false;
	}

	// TODO replace VK_PIPELINE_STAGE_ALL_COMMANDS_BIT
	m_device->vk.CmdPipelineBarrier(m_cmdBuffer, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
									0, 0, nullptr, 0, nullptr, m_imageBarriers.size(), m_imageBarriers.data());
}

CVulkanDevice g_device;

static bool allDMABUFsEqual( wlr_dmabuf_attributes *pDMA )
{
	if ( pDMA->n_planes == 1 )
		return true;

	struct stat first_stat;
	if ( fstat( pDMA->fd[0], &first_stat ) != 0 )
	{
		vk_log.errorf_errno( "fstat failed" );
		return false;
	}

	for ( int i = 1; i < pDMA->n_planes; ++i )
	{
		struct stat plane_stat;
		if ( fstat( pDMA->fd[i], &plane_stat ) != 0 )
		{
			vk_log.errorf_errno( "fstat failed" );
			return false;
		}
		if ( plane_stat.st_ino != first_stat.st_ino )
			return false;
	}

	return true;
}

static VkResult getModifierProps( const VkImageCreateInfo *imageInfo, uint64_t modifier, VkExternalImageFormatProperties *externalFormatProps)
{
	VkPhysicalDeviceImageDrmFormatModifierInfoEXT modifierFormatInfo = {
		.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_DRM_FORMAT_MODIFIER_INFO_EXT,
		.drmFormatModifier = modifier,
		.sharingMode = imageInfo->sharingMode,
	};

	VkPhysicalDeviceExternalImageFormatInfo externalImageFormatInfo = {
		.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_IMAGE_FORMAT_INFO,
		.pNext = &modifierFormatInfo,
		.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT,
	};

	VkPhysicalDeviceImageFormatInfo2 imageFormatInfo = {
		.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_FORMAT_INFO_2,
		.pNext = &externalImageFormatInfo,
		.format = imageInfo->format,
		.type = imageInfo->imageType,
		.tiling = VK_IMAGE_TILING_DRM_FORMAT_MODIFIER_EXT,
		.usage = imageInfo->usage,
		.flags = imageInfo->flags,
	};

	const VkImageFormatListCreateInfo *readonlyList = pNextFind<VkImageFormatListCreateInfo>(imageInfo, VK_STRUCTURE_TYPE_IMAGE_FORMAT_LIST_CREATE_INFO);
	VkImageFormatListCreateInfo formatList = {};
	if ( readonlyList != nullptr )
	{
		formatList = *readonlyList;
		formatList.pNext = std::exchange(imageFormatInfo.pNext, &formatList);
	}

	VkImageFormatProperties2 imageProps = {
		.sType = VK_STRUCTURE_TYPE_IMAGE_FORMAT_PROPERTIES_2,
		.pNext = externalFormatProps,
	};

	return g_device.vk.GetPhysicalDeviceImageFormatProperties2(g_device.physDev(), &imageFormatInfo, &imageProps);
}

static VkImageViewType VulkanImageTypeToViewType(VkImageType type)
{
	switch (type)
	{
		case VK_IMAGE_TYPE_1D: return VK_IMAGE_VIEW_TYPE_1D;
		case VK_IMAGE_TYPE_2D: return VK_IMAGE_VIEW_TYPE_2D;
		case VK_IMAGE_TYPE_3D: return VK_IMAGE_VIEW_TYPE_3D;
		default: abort();
	}
}

static void copy_render_origin( std::array<char, 16> &dst, const char *origin )
{
	std::snprintf( dst.data(), dst.size(), "%s", origin != nullptr ? origin : "unknown" );
}

static const char *dmabuf_modifier_vendor_name( uint8_t vendor )
{
	switch ( vendor )
	{
		case 0x01: return "Intel";
		case 0x02: return "AMD";
		case 0x03: return "NVIDIA";
		case 0x04: return "Samsung";
		case 0x05: return "Qcom";
		case 0x06: return "Vivante";
		case 0x07: return "Broadcom";
		case 0x08: return "ARM";
		default: return nullptr;
	}
}

static std::array<char, 16> dmabuf_render_origin( const wlr_dmabuf_attributes *pDMA )
{
	std::array<char, 16> result = {};
	copy_render_origin( result, "unknown" );
	if ( pDMA == nullptr )
	{
		copy_render_origin( result, "cpu" );
		return result;
	}

	if ( pDMA->modifier != DRM_FORMAT_MOD_INVALID
		&& pDMA->modifier != DRM_FORMAT_MOD_LINEAR )
	{
		const uint8_t vendor = static_cast<uint8_t>( pDMA->modifier >> 56u );
		const char *pszVendor = dmabuf_modifier_vendor_name( vendor );
		if ( pszVendor != nullptr )
			std::snprintf( result.data(), result.size(), "%s", pszVendor );
		else
			std::snprintf( result.data(), result.size(), "vendor0x%02x", vendor );
		return result;
	}

	copy_render_origin( result, "linear" );
	return result;
}

void CVulkanTexture::setRenderOrigin( const char *origin )
{
	copy_render_origin( m_renderOrigin, origin );
}

bool CVulkanTexture::BInit( uint32_t width, uint32_t height, uint32_t depth, uint32_t drmFormat, createFlags flags, wlr_dmabuf_attributes *pDMA /* = nullptr */,  uint32_t contentWidth /* = 0 */, uint32_t contentHeight /* =  0 */, CVulkanTexture *pExistingImageToReuseMemory, gamescope::OwningRc<gamescope::IBackendFb> pBackendFb )
{
	m_pBackendFb = std::move( pBackendFb );
	m_drmFormat = drmFormat;
	m_renderOrigin = dmabuf_render_origin( pDMA );
	VkResult res = VK_ERROR_INITIALIZATION_FAILED;

	VkImageTiling tiling = (flags.bMappable || flags.bLinear) ? VK_IMAGE_TILING_LINEAR : VK_IMAGE_TILING_OPTIMAL;
	VkImageUsageFlags usage = 0;
	VkMemoryPropertyFlags properties;

	if ( flags.bSampled == true )
	{
		usage |= VK_IMAGE_USAGE_SAMPLED_BIT;
	}

	if ( flags.bStorage == true )
	{
		usage |= VK_IMAGE_USAGE_STORAGE_BIT;
	}

	if ( flags.bColorAttachment == true )
	{
		usage |= VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
	}

	if ( flags.bFlippable == true )
	{
		flags.bExportable = true;
	}

	if ( flags.bTransferSrc == true )
	{
		usage |= VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
	}
	m_bTransferSrc = flags.bTransferSrc;

	if ( flags.bTransferDst == true )
	{
		usage |= VK_IMAGE_USAGE_TRANSFER_DST_BIT;
	}

	if ( flags.bMappable == true )
	{
		properties = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT;
	}
	else
	{
		properties = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
	}

	if ( flags.bOutputImage == true )
	{
		m_bOutputImage = true;	
	}

	m_bExternal = pDMA || flags.bExportable == true;

	// Possible extensions for below
	wsi_image_create_info wsiImageCreateInfo = {};
	VkExternalMemoryImageCreateInfo externalImageCreateInfo = {};
	VkImageDrmFormatModifierExplicitCreateInfoEXT modifierInfo = {};
	VkSubresourceLayout modifierPlaneLayouts[4] = {};
	VkImageDrmFormatModifierListCreateInfoEXT modifierListInfo = {};
	
	// Cross-family framegen (framegenFamilySplit(): the framegen queue lives on
	// a different queue family than the compositor) reads and writes
	// gamescope-owned images that the composite queue also touches - framegen
	// history/output/present rings, motion fields, and the composited output
	// images used as history in output-space mode. Rather than thread explicit
	// queue-family ownership-transfer barrier pairs through every one of those
	// hand-offs, declare exactly those images (flags.bFramegenShared) VK_SHARING_MODE_CONCURRENT
	// over the families involved: correct by construction, at the cost of losing
	// some drivers' compressed layouts on those images. Everything framegen never
	// touches - cursors, LUTs, mura, screenshot and PipeWire targets - stays
	// VK_SHARING_MODE_EXCLUSIVE and keeps compression. Client-imported dma-bufs
	// (pDMA) are never accessed on the framegen family either - their FOREIGN
	// acquire barriers only ever transfer ownership to the compositor family, and
	// imported-image interop on a secondary family is exactly what the Intel
	// quirk above is about.
	uint32_t nSharedQueueFamilies[3] = {};
	uint32_t nSharedQueueFamilyCount = 0;
	if ( pDMA == nullptr && flags.bFramegenShared && g_device.framegenFamilySplit() )
	{
		const auto addFamily = [&]( uint32_t nFamily )
		{
			for ( uint32_t i = 0; i < nSharedQueueFamilyCount; i++ )
			{
				if ( nSharedQueueFamilies[i] == nFamily )
					return;
			}
			nSharedQueueFamilies[ nSharedQueueFamilyCount++ ] = nFamily;
		};
		addFamily( g_device.queueFamily() );
		addFamily( g_device.generalQueueFamily() );
		addFamily( g_device.framegenQueueFamily() );
		if ( nSharedQueueFamilyCount < 2 )
			nSharedQueueFamilyCount = 0;
	}

	VkImageCreateInfo imageInfo = {
		.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
		.imageType = flags.imageType,
		.format = DRMFormatToVulkan(drmFormat, false),
		.extent = {
			.width = width,
			.height = height,
			.depth = depth,
		},
		.mipLevels = 1,
		.arrayLayers = 1,
		.samples = VK_SAMPLE_COUNT_1_BIT,
		.tiling = tiling,
		.usage = usage,
		.sharingMode = nSharedQueueFamilyCount != 0 ? VK_SHARING_MODE_CONCURRENT : VK_SHARING_MODE_EXCLUSIVE,
		.queueFamilyIndexCount = nSharedQueueFamilyCount,
		.pQueueFamilyIndices = nSharedQueueFamilyCount != 0 ? nSharedQueueFamilies : nullptr,
	};

	m_bConcurrentSharing = imageInfo.sharingMode == VK_SHARING_MODE_CONCURRENT;

	assert( imageInfo.format != VK_FORMAT_UNDEFINED );

	if ( g_bDebugDualGpuRoute && pDMA )
	{
		vk_log.infof( "dual-gpu-route: client dma-buf Vulkan import request %dx%d format 0x%" PRIX32 " modifier 0x%" PRIX64 " planes %d usage 0x%x sampled %s storage %s transfer-src %s transfer-dst %s",
			pDMA->width,
			pDMA->height,
			pDMA->format,
			pDMA->modifier,
			pDMA->n_planes,
			usage,
			flags.bSampled ? "yes" : "no",
			flags.bStorage ? "yes" : "no",
			flags.bTransferSrc ? "yes" : "no",
			flags.bTransferDst ? "yes" : "no" );

		for ( int i = 0; i < pDMA->n_planes; i++ )
		{
			vk_log.infof( "dual-gpu-route:   plane %d fd %d offset %u stride %u",
				i,
				pDMA->fd[i],
				pDMA->offset[i],
				pDMA->stride[i] );
		}
	}

	std::array<VkFormat, 2> formats = {
		DRMFormatToVulkan(drmFormat, false),
		DRMFormatToVulkan(drmFormat, true),
	};

	VkImageFormatListCreateInfo formatList = {
		.sType = VK_STRUCTURE_TYPE_IMAGE_FORMAT_LIST_CREATE_INFO,
		.viewFormatCount = (uint32_t)formats.size(),
		.pViewFormats = formats.data(),
	};

	if ( formats[0] != formats[1] )
	{
		formatList.pNext = std::exchange(imageInfo.pNext, &formatList);
		imageInfo.flags |= VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT;
	}

	if ( pDMA != nullptr )
	{
		assert( drmFormat == pDMA->format );
	}

	if ( g_device.supportsModifiers() && pDMA && pDMA->modifier != DRM_FORMAT_MOD_INVALID )
	{
		VkExternalImageFormatProperties externalImageProperties = {
			.sType = VK_STRUCTURE_TYPE_EXTERNAL_IMAGE_FORMAT_PROPERTIES,
		};

		res = getModifierProps( &imageInfo, pDMA->modifier, &externalImageProperties );
		if ( res != VK_SUCCESS && res != VK_ERROR_FORMAT_NOT_SUPPORTED ) {
			vk_errorf( res, "getModifierProps failed" );
			return false;
		}

		if ( g_bDebugDualGpuRoute )
		{
			vk_log.infof( "dual-gpu-route: client dma-buf modifier capability result %d external features 0x%x importable %s",
				res,
				externalImageProperties.externalMemoryProperties.externalMemoryFeatures,
				( res == VK_SUCCESS &&
				  ( externalImageProperties.externalMemoryProperties.externalMemoryFeatures & VK_EXTERNAL_MEMORY_FEATURE_IMPORTABLE_BIT ) )
					? "yes"
					: "no" );
		}

		if ( res == VK_SUCCESS &&
		     ( externalImageProperties.externalMemoryProperties.externalMemoryFeatures & VK_EXTERNAL_MEMORY_FEATURE_IMPORTABLE_BIT ) )
		{
			modifierInfo = {
				.sType = VK_STRUCTURE_TYPE_IMAGE_DRM_FORMAT_MODIFIER_EXPLICIT_CREATE_INFO_EXT,
				.pNext = std::exchange(imageInfo.pNext, &modifierInfo),
				.drmFormatModifier = pDMA->modifier,
				.drmFormatModifierPlaneCount = uint32_t(pDMA->n_planes),
				.pPlaneLayouts = modifierPlaneLayouts,
			};

			for ( int i = 0; i < pDMA->n_planes; ++i )
			{
				modifierPlaneLayouts[i].offset = pDMA->offset[i];
				modifierPlaneLayouts[i].rowPitch = pDMA->stride[i];
			}

			imageInfo.tiling = tiling = VK_IMAGE_TILING_DRM_FORMAT_MODIFIER_EXT;
		}
		else
		{
			vk_log.errorf( "dma-buf modifier 0x%" PRIX64 " for DRM format 0x%" PRIX32 " is not importable",
				pDMA->modifier, drmFormat );
			return false;
		}
	}
	else if ( pDMA && pDMA->modifier != DRM_FORMAT_MOD_INVALID )
	{
		if ( g_bDebugDualGpuRoute )
		{
			vk_log.infof( "dual-gpu-route: client dma-buf has modifier 0x%" PRIX64 " but compositor Vulkan modifier support is %s",
				pDMA->modifier,
				g_device.supportsModifiers() ? "enabled" : "disabled" );
		}

		// Without modifier support we can only create a plain (implicitly tiled)
		// image, which would interpret a tiled buffer as if it were linear.
		// Fail the import instead of silently sampling garbage.
		if ( pDMA->modifier != DRM_FORMAT_MOD_LINEAR )
		{
			vk_log.errorf( "dma-buf modifier 0x%" PRIX64 " cannot be imported without Vulkan DRM format modifier support",
				pDMA->modifier );
			return false;
		}
	}

	std::vector<uint64_t> modifiers = {};
	// TODO(JoshA): Move this code to backend for making flippable image.
	if ( GetBackend()->UsesModifiers() && flags.bFlippable && g_device.supportsModifiers() && !pDMA )
	{
		assert( drmFormat != DRM_FORMAT_INVALID );

		uint64_t linear = DRM_FORMAT_MOD_LINEAR;

		const uint64_t *possibleModifiers;
		size_t numPossibleModifiers;
		if ( flags.bLinear )
		{
			possibleModifiers = &linear;
			numPossibleModifiers = 1;
		}
		else
		{
			std::span<const uint64_t> modifiers = GetBackend()->GetSupportedModifiers( drmFormat );
			assert( !modifiers.empty() );
			possibleModifiers = modifiers.data();
			numPossibleModifiers = modifiers.size();
		}

		for ( size_t i = 0; i < numPossibleModifiers; i++ )
		{
			uint64_t modifier = possibleModifiers[i];

			VkExternalImageFormatProperties externalFormatProps = {
				.sType = VK_STRUCTURE_TYPE_EXTERNAL_IMAGE_FORMAT_PROPERTIES,
			};
			res = getModifierProps( &imageInfo, modifier, &externalFormatProps );
			if ( res == VK_ERROR_FORMAT_NOT_SUPPORTED )
				continue;
			else if ( res != VK_SUCCESS ) {
				vk_errorf( res, "getModifierProps failed" );
				return false;
			}

			if ( !( externalFormatProps.externalMemoryProperties.externalMemoryFeatures & VK_EXTERNAL_MEMORY_FEATURE_EXPORTABLE_BIT ) )
				continue;

			modifiers.push_back( modifier );
		}

		assert( modifiers.size() > 0 );

		modifierListInfo = {
			.sType = VK_STRUCTURE_TYPE_IMAGE_DRM_FORMAT_MODIFIER_LIST_CREATE_INFO_EXT,
			.pNext = std::exchange(imageInfo.pNext, &modifierListInfo),
			.drmFormatModifierCount = uint32_t(modifiers.size()),
			.pDrmFormatModifiers = modifiers.data(),
		};

		externalImageCreateInfo = {
			.sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO,
			.pNext = std::exchange(imageInfo.pNext, &externalImageCreateInfo),
			.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT,
		};

		imageInfo.tiling = tiling = VK_IMAGE_TILING_DRM_FORMAT_MODIFIER_EXT;
	}

	if ( flags.bFlippable == true && tiling != VK_IMAGE_TILING_DRM_FORMAT_MODIFIER_EXT )
	{
		// We want to scan-out the image
		wsiImageCreateInfo = {
			.sType = VK_STRUCTURE_TYPE_WSI_IMAGE_CREATE_INFO_MESA,
			.pNext = std::exchange(imageInfo.pNext, &wsiImageCreateInfo),
			.scanout = VK_TRUE,
		};
	}
	
	if ( pDMA != nullptr )
	{
		externalImageCreateInfo = {
			.sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO,
			.pNext = std::exchange(imageInfo.pNext, &externalImageCreateInfo),
			.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT,
		};
	}

	m_width = width;
	m_height = height;
	m_depth = depth;

	if (contentWidth && contentHeight)
	{
		m_contentWidth = contentWidth;
		m_contentHeight = contentHeight;
	}
	else
	{
		m_contentWidth = width;
		m_contentHeight = height;
	}

	m_format = imageInfo.format;

	res = g_device.vk.CreateImage(g_device.device(), &imageInfo, nullptr, &m_vkImage);
	if (res != VK_SUCCESS) {
		vk_errorf( res, "vkCreateImage failed" );
		return false;
	}
	
	VkMemoryRequirements memRequirements;
	g_device.vk.GetImageMemoryRequirements(g_device.device(), m_vkImage, &memRequirements);

	VkMemoryAllocateInfo allocInfo = {
		.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
		.allocationSize = memRequirements.size,
		.memoryTypeIndex = uint32_t(g_device.findMemoryType(properties, memRequirements.memoryTypeBits)),
	};

	m_size = allocInfo.allocationSize;

	VkDeviceMemory memoryHandle = VK_NULL_HANDLE;

	if ( pExistingImageToReuseMemory == nullptr )
	{
		// Possible pNexts
		// fd = -1 so the failure path below can tell "never imported" from a real FD.
		VkImportMemoryFdInfoKHR importMemoryInfo = { .fd = -1 };
		VkExportMemoryAllocateInfo memory_export_info = {};
		VkMemoryDedicatedAllocateInfo memory_dedicated_info = {};
		struct wsi_memory_allocate_info memory_wsi_info = {};

		if ( flags.bFlippable == true )
		{
			memory_wsi_info = {
				.sType = VK_STRUCTURE_TYPE_WSI_MEMORY_ALLOCATE_INFO_MESA,
				.pNext = std::exchange(allocInfo.pNext, &memory_wsi_info),
			};
		}

		if ( flags.bExportable == true || pDMA != nullptr )
		{
			memory_dedicated_info = {
				.sType = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO,
				.pNext = std::exchange(allocInfo.pNext, &memory_dedicated_info),
				.image = m_vkImage,
			};
		}
		
		if ( flags.bExportable == true && pDMA == nullptr )
		{
			// We'll export it to DRM
			memory_export_info = {
				.sType = VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO,
				.pNext = std::exchange(allocInfo.pNext, &memory_export_info),
				.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT,
			};
		}
		
		if ( pDMA != nullptr )
		{
			// TODO: multi-planar DISTINCT DMA-BUFs support (see vkBindImageMemory2
			// and VkBindImagePlaneMemoryInfo)
			// Only fd[0] is imported and bound to the whole image, so all planes
			// must live in the same dma-buf. In release builds the assert is gone,
			// hence the runtime check.
			if ( !allDMABUFsEqual( pDMA ) )
			{
				vk_log.errorf( "multi-fd dma-buf import unsupported" );
				return false;
			}

			// Importing memory from a FD transfers ownership of the FD
			int fd = dup( pDMA->fd[0] );
			if ( fd < 0 )
			{
				vk_log.errorf_errno( "dup failed" );
				return false;
			}

			VkMemoryFdPropertiesKHR memoryFdProperties = {
				.sType = VK_STRUCTURE_TYPE_MEMORY_FD_PROPERTIES_KHR,
			};
			res = g_device.vk.GetMemoryFdPropertiesKHR( g_device.device(),
				VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT, fd, &memoryFdProperties );
			if ( res != VK_SUCCESS )
			{
				vk_errorf( res, "vkGetMemoryFdPropertiesKHR failed" );
				close( fd );
				return false;
			}

			// The properties query does not consume fd; the successful import below
			// retains the existing ownership-transfer semantics.
			const uint32_t uImportMemoryTypeBits = memRequirements.memoryTypeBits & memoryFdProperties.memoryTypeBits;
			int32_t nMemoryTypeIndex = g_device.findMemoryType( properties, uImportMemoryTypeBits );
			if ( nMemoryTypeIndex < 0 )
			{
				// Imported memory is already backed; the property preference (e.g.
				// DEVICE_LOCAL) is a placement hint that a cross-vendor dma-buf may
				// legitimately not satisfy. Any type in the legal intersection is
				// spec-correct — only an empty intersection is a real failure.
				nMemoryTypeIndex = g_device.findMemoryType( 0, uImportMemoryTypeBits );
				if ( nMemoryTypeIndex < 0 )
				{
					vk_errorf( VK_ERROR_FEATURE_NOT_PRESENT, "no compatible memory type for dma-buf import" );
					close( fd );
					return false;
				}
				static bool s_bLoggedRelaxedImportType = false;
				if ( !s_bLoggedRelaxedImportType )
				{
					vk_log.infof( "dma-buf import: preferred memory properties 0x%x unavailable, using memory type %d from the imported FD's legal set",
						properties, nMemoryTypeIndex );
					s_bLoggedRelaxedImportType = true;
				}
			}
			allocInfo.memoryTypeIndex = uint32_t( nMemoryTypeIndex );

			// Memory already provided by pDMA
			importMemoryInfo = {
					.sType = VK_STRUCTURE_TYPE_IMPORT_MEMORY_FD_INFO_KHR,
					.pNext = std::exchange(allocInfo.pNext, &importMemoryInfo),
					.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT,
					.fd = fd,
			};
		}
		
		res = g_device.vk.AllocateMemory( g_device.device(), &allocInfo, nullptr, &memoryHandle );
		if ( res != VK_SUCCESS )
		{
			vk_errorf( res, "vkAllocateMemory failed" );
			// Ownership of the dup'd FD only transfers on a successful import.
			if ( importMemoryInfo.fd >= 0 )
				close( importMemoryInfo.fd );
			return false;
		}

		m_vkImageMemory = memoryHandle;
		m_bDeviceLocal = g_device.findMemoryType( VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
			1u << allocInfo.memoryTypeIndex ) >= 0;
	}
	else
	{
		vk_log.infof("%d vs %d!", (int)pExistingImageToReuseMemory->m_size, (int)m_size);
		assert(pExistingImageToReuseMemory->m_size >= m_size);

		memoryHandle = pExistingImageToReuseMemory->m_vkImageMemory;
		m_vkImageMemory = VK_NULL_HANDLE;
		m_bDeviceLocal = pExistingImageToReuseMemory->m_bDeviceLocal;
	}
	
	res = g_device.vk.BindImageMemory( g_device.device(), m_vkImage, memoryHandle, 0 );
	if ( res != VK_SUCCESS )
	{
		vk_errorf( res, "vkBindImageMemory failed" );
		return false;
	}

	if ( flags.bMappable == true )
	{
		assert( tiling != VK_IMAGE_TILING_DRM_FORMAT_MODIFIER_EXT );
		const VkImageSubresource image_subresource = {
			.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
		};
		VkSubresourceLayout image_layout;
		g_device.vk.GetImageSubresourceLayout(g_device.device(), m_vkImage, &image_subresource, &image_layout);

		m_unRowPitch = image_layout.rowPitch;

		if (isYcbcr())
		{
			const VkImageSubresource lumaSubresource = {
				.aspectMask = VK_IMAGE_ASPECT_PLANE_0_BIT,
			};
			VkSubresourceLayout lumaLayout;
			g_device.vk.GetImageSubresourceLayout(g_device.device(), m_vkImage, &lumaSubresource, &lumaLayout);

			m_lumaOffset = lumaLayout.offset;
			m_lumaPitch = lumaLayout.rowPitch;

			const VkImageSubresource chromaSubresource = {
				.aspectMask = VK_IMAGE_ASPECT_PLANE_1_BIT,
			};
			VkSubresourceLayout chromaLayout;
			g_device.vk.GetImageSubresourceLayout(g_device.device(), m_vkImage, &chromaSubresource, &chromaLayout);

			m_chromaOffset = chromaLayout.offset;
			m_chromaPitch = chromaLayout.rowPitch;
		}
	}
	
	if ( flags.bExportable == true )
	{
		// We assume we own the memory when doing this right now.
		// We could support the import scenario as well if needed (but we
		// already have a DMA-BUF in that case).
		assert( pDMA == nullptr );

		struct wlr_dmabuf_attributes dmabuf = {
			.width = int(width),
			.height = int(height),
			.format = drmFormat,
		};
		assert( dmabuf.format != DRM_FORMAT_INVALID );

		// TODO: disjoint planes support
		const VkMemoryGetFdInfoKHR memory_get_fd_info = {
			.sType = VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR,
			.memory = memoryHandle,
			.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT,
		};
		res = g_device.vk.GetMemoryFdKHR(g_device.device(), &memory_get_fd_info, &dmabuf.fd[0]);
		if ( res != VK_SUCCESS ) {
			vk_errorf( res, "vkGetMemoryFdKHR failed" );
			return false;
		}

		if ( tiling == VK_IMAGE_TILING_DRM_FORMAT_MODIFIER_EXT )
		{
			assert( g_device.vk.GetImageDrmFormatModifierPropertiesEXT != nullptr );

			VkImageDrmFormatModifierPropertiesEXT imgModifierProps = {
				.sType = VK_STRUCTURE_TYPE_IMAGE_DRM_FORMAT_MODIFIER_PROPERTIES_EXT,
			};

			res = g_device.vk.GetImageDrmFormatModifierPropertiesEXT( g_device.device(), m_vkImage, &imgModifierProps );
			if ( res != VK_SUCCESS ) {
				vk_errorf( res, "vkGetImageDrmFormatModifierPropertiesEXT failed" );
				return false;
			}
			dmabuf.modifier = imgModifierProps.drmFormatModifier;

			assert( DRMModifierProps.count( m_format ) > 0);
			assert( DRMModifierProps[ m_format ].count( dmabuf.modifier ) > 0);

			dmabuf.n_planes = DRMModifierProps[ m_format ][ dmabuf.modifier ].drmFormatModifierPlaneCount;

			const VkImageAspectFlagBits planeAspects[] = {
				VK_IMAGE_ASPECT_MEMORY_PLANE_0_BIT_EXT,
				VK_IMAGE_ASPECT_MEMORY_PLANE_1_BIT_EXT,
				VK_IMAGE_ASPECT_MEMORY_PLANE_2_BIT_EXT,
				VK_IMAGE_ASPECT_MEMORY_PLANE_3_BIT_EXT,
			};
			assert( dmabuf.n_planes <= 4 );

			for ( int i = 0; i < dmabuf.n_planes; i++ )
			{
				const VkImageSubresource subresource = {
					.aspectMask = planeAspects[i],
				};
				VkSubresourceLayout subresourceLayout = {};
				g_device.vk.GetImageSubresourceLayout( g_device.device(), m_vkImage, &subresource, &subresourceLayout );
				dmabuf.offset[i] = subresourceLayout.offset;
				dmabuf.stride[i] = subresourceLayout.rowPitch;
			}

			// Copy the first FD to all other planes
			for ( int i = 1; i < dmabuf.n_planes; i++ )
			{
				dmabuf.fd[i] = dup( dmabuf.fd[0] );
				if ( dmabuf.fd[i] < 0 ) {
					vk_log.errorf_errno( "dup failed" );
					return false;
				}
			}
		}
		else
		{
			const VkImageSubresource subresource = {
				.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
			};
			VkSubresourceLayout subresourceLayout = {};
			g_device.vk.GetImageSubresourceLayout( g_device.device(), m_vkImage, &subresource, &subresourceLayout );

			dmabuf.n_planes = 1;
			dmabuf.modifier = DRM_FORMAT_MOD_INVALID;
			dmabuf.offset[0] = 0;
			dmabuf.stride[0] = subresourceLayout.rowPitch;
		}

		m_dmabuf = dmabuf;
	}

	if ( flags.bFlippable == true )
	{
		m_pBackendFb = GetBackend()->ImportDmabufToBackend( &m_dmabuf );
	}

	bool bHasAlpha = pDMA ? DRMFormatHasAlpha( pDMA->format ) : true;

	if (!bHasAlpha )
	{
		// not compatible with with swizzles
		assert ( flags.bStorage == false );
	}

	if ( flags.bStorage || flags.bSampled || flags.bColorAttachment )
	{
		VkImageViewCreateInfo createInfo = {
			.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
			.image = m_vkImage,
			.viewType = VulkanImageTypeToViewType(flags.imageType),
			.format = DRMFormatToVulkan(drmFormat, false),
			.components = {
				.r = VK_COMPONENT_SWIZZLE_IDENTITY,
				.g = VK_COMPONENT_SWIZZLE_IDENTITY,
				.b = VK_COMPONENT_SWIZZLE_IDENTITY,
				.a = bHasAlpha ? VK_COMPONENT_SWIZZLE_IDENTITY : VK_COMPONENT_SWIZZLE_ONE,
			},
			.subresourceRange = {
				.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
				.levelCount = 1,
				.layerCount = 1,
			},
		};

		res = g_device.vk.CreateImageView(g_device.device(), &createInfo, nullptr, &m_srgbView);
		if ( res != VK_SUCCESS ) {
			vk_errorf( res, "vkCreateImageView failed" );
			return false;
		}

		if ( flags.bSampled )
		{
			VkImageViewUsageCreateInfo viewUsageInfo = {
				.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_USAGE_CREATE_INFO,
				.usage = usage & ~VK_IMAGE_USAGE_STORAGE_BIT,
			};
			createInfo.pNext = &viewUsageInfo;
			createInfo.format = DRMFormatToVulkan(drmFormat, true);
			res = g_device.vk.CreateImageView(g_device.device(), &createInfo, nullptr, &m_linearView);
			if ( res != VK_SUCCESS ) {
				vk_errorf( res, "vkCreateImageView failed" );
				return false;
			}
		}


		if ( isYcbcr() )
		{
			createInfo.pNext = NULL;
			createInfo.format = VK_FORMAT_R8_UNORM;

			createInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_PLANE_0_BIT;
			res = g_device.vk.CreateImageView(g_device.device(), &createInfo, nullptr, &m_lumaView);
			if ( res != VK_SUCCESS ) {
				vk_errorf( res, "vkCreateImageView failed" );
				return false;
			}

			createInfo.pNext = NULL;
			createInfo.format = VK_FORMAT_R8G8_UNORM;
			createInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_PLANE_1_BIT;
			res = g_device.vk.CreateImageView(g_device.device(), &createInfo, nullptr, &m_chromaView);
			if ( res != VK_SUCCESS ) {
				vk_errorf( res, "vkCreateImageView failed" );
				return false;
			}

			createInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		}
	}

	if ( flags.bMappable )
	{
		if (pExistingImageToReuseMemory)
		{
			m_pMappedData = pExistingImageToReuseMemory->m_pMappedData;
		}
		else
		{
			void *pData = nullptr;
			res = g_device.vk.MapMemory( g_device.device(), memoryHandle, 0, VK_WHOLE_SIZE, 0, &pData );
			if ( res != VK_SUCCESS )
			{
				vk_errorf( res, "vkMapMemory failed" );
				return false;
			}
			m_pMappedData = (uint8_t*)pData;
		}
	}
	
	m_bInitialized = true;

	if ( g_bDebugDualGpuRoute && pDMA )
	{
		vk_log.infof( "dual-gpu-route: client dma-buf Vulkan import success tiling %s allocation %" PRIu64 " bytes backend fb %s",
			vk_image_tiling_name( tiling ),
			uint64_t( m_size ),
			m_pBackendFb ? "yes" : "no" );
	}
	
	return true;
}

bool CVulkanTexture::BInitFromSwapchain( VkImage image, uint32_t width, uint32_t height, VkFormat format )
{
	m_drmFormat = VulkanFormatToDRM( format );
	m_vkImage = image;
	m_vkImageMemory = VK_NULL_HANDLE;
	m_width = width;
	m_height = height;
	m_depth = 1;
	m_format = format;
	m_contentWidth = width;
	m_contentHeight = height;
	m_bOutputImage = true;

	VkImageViewCreateInfo createInfo = {
		.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
		.image = image,
		.viewType = VK_IMAGE_VIEW_TYPE_2D,
		.format = ToLinearVulkanFormat( format ),
		.components = {
			.r = VK_COMPONENT_SWIZZLE_IDENTITY,
			.g = VK_COMPONENT_SWIZZLE_IDENTITY,
			.b = VK_COMPONENT_SWIZZLE_IDENTITY,
			.a = VK_COMPONENT_SWIZZLE_IDENTITY,
		},
		.subresourceRange = {
			.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
			.levelCount = 1,
			.layerCount = 1,
		},
	};

	VkResult res = g_device.vk.CreateImageView(g_device.device(), &createInfo, nullptr, &m_srgbView);
	if ( res != VK_SUCCESS ) {
		vk_errorf( res, "vkCreateImageView failed" );
		return false;
	}

	VkImageViewUsageCreateInfo viewUsageInfo = {
		.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_USAGE_CREATE_INFO,
		.usage = VK_IMAGE_USAGE_SAMPLED_BIT,
	};

	createInfo.pNext = &viewUsageInfo;
	createInfo.format = ToSrgbVulkanFormat( format );

	res = g_device.vk.CreateImageView(g_device.device(), &createInfo, nullptr, &m_linearView);
	if ( res != VK_SUCCESS ) {
		vk_errorf( res, "vkCreateImageView failed" );
		return false;
	}

	m_bInitialized = true;

	return true;
}

gamescope::Rc<CVulkanTexture> CVulkanTexture::AcquireDeviceLocalStagingImage()
{
	if ( !m_bExternal || m_bDeviceLocal || !m_bTransferSrc )
		return nullptr;

	const size_t uPoolSize = m_deviceLocalStagingImages.size();
	for ( size_t i = 0; i < uPoolSize; i++ )
	{
		const size_t uIndex = ( m_uDeviceLocalStagingCursor + i ) % uPoolSize;
		auto &pTexture = m_deviceLocalStagingImages[ uIndex ];
		if ( pTexture->IsInUse() )
			continue;

		m_uDeviceLocalStagingCursor = ( uIndex + 1 ) % uPoolSize;
		return pTexture;
	}

	gamescope::OwningRc<CVulkanTexture> pTexture = new CVulkanTexture();
	CVulkanTexture::createFlags flags;
	flags.bSampled = true;
	flags.bTransferDst = true;
	if ( !pTexture->BInit( m_width, m_height, m_depth, m_drmFormat, flags,
			nullptr, m_contentWidth, m_contentHeight ) )
	{
		vk_log.errorf( "dma-buf staging: failed to allocate %ux%u format 0x%" PRIX32,
			m_width, m_height, m_drmFormat );
		return nullptr;
	}
	if ( !pTexture->deviceLocal() )
	{
		vk_log.errorf( "dma-buf staging: %ux%u format 0x%" PRIX32 " did not allocate in DEVICE_LOCAL memory",
			m_width, m_height, m_drmFormat );
		return nullptr;
	}

	pTexture->setStreamColorspace( m_streamColorspace );
	pTexture->setRenderOrigin( renderOrigin() );
	pTexture->m_bDeviceLocalStagingImage = true;
	m_deviceLocalStagingImages.emplace_back( std::move( pTexture ) );
	m_uDeviceLocalStagingCursor = 0;

	vk_log.infof( "dma-buf staging: grew image pool for %p (%ux%u format 0x%" PRIX32 ") to %zu",
		this, m_width, m_height, m_drmFormat, m_deviceLocalStagingImages.size() );
	return m_deviceLocalStagingImages.back();
}

uint32_t CVulkanTexture::IncRef()
{
	uint32_t uRefCount = gamescope::RcObject::IncRef();
	if ( m_pBackendFb && !uRefCount )
	{
		m_pBackendFb->IncRef();
	}
	return uRefCount;
}
uint32_t CVulkanTexture::DecRef()
{
	// Need to pull it out as we could be destroyed in DecRef.
	gamescope::IBackendFb *pBackendFb = m_pBackendFb.get();

	uint32_t uRefCount = gamescope::RcObject::DecRef();
	if ( pBackendFb && !uRefCount )
	{
		pBackendFb->DecRef();
	}
	return uRefCount;
}

bool CVulkanTexture::IsInUse()
{
	if ( m_pBackendFb && m_pBackendFb->GetRefCount() != 0 )
		return true;

	return GetRefCount() != 0;
}

CVulkanTexture::CVulkanTexture( void )
{
}

CVulkanTexture::~CVulkanTexture( void )
{
	wlr_dmabuf_attributes_finish( &m_dmabuf );

	if ( m_pMappedData != nullptr && m_vkImageMemory )
	{
		g_device.vk.UnmapMemory( g_device.device(), m_vkImageMemory );
		m_pMappedData = nullptr;
	}

	if ( m_srgbView != VK_NULL_HANDLE )
	{
		g_device.vk.DestroyImageView( g_device.device(), m_srgbView, nullptr );
		m_srgbView = VK_NULL_HANDLE;
	}

	if ( m_linearView != VK_NULL_HANDLE )
	{
		g_device.vk.DestroyImageView( g_device.device(), m_linearView, nullptr );
		m_linearView = VK_NULL_HANDLE;
	}

	if ( m_pBackendFb != nullptr )
		m_pBackendFb = nullptr;

	if ( m_vkImageMemory != VK_NULL_HANDLE )
	{
		if ( m_vkImage != VK_NULL_HANDLE )
		{
			g_device.vk.DestroyImage( g_device.device(), m_vkImage, nullptr );
			m_vkImage = VK_NULL_HANDLE;
		}

		g_device.vk.FreeMemory( g_device.device(), m_vkImageMemory, nullptr );
		m_vkImageMemory = VK_NULL_HANDLE;
	}

	m_bInitialized = false;
}

int CVulkanTexture::memoryFence()
{
	const VkMemoryGetFdInfoKHR memory_get_fd_info = {
		.sType = VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR,
		.memory = m_vkImageMemory,
		.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT,
	};
	int fence = -1;
	VkResult res = g_device.vk.GetMemoryFdKHR(g_device.device(), &memory_get_fd_info, &fence);
	if ( res != VK_SUCCESS ) {
		fprintf( stderr, "vkGetMemoryFdKHR failed\n" );
	}

	return fence;
}

static bool is_image_format_modifier_supported(VkFormat format, uint32_t drmFormat, uint64_t modifier)
{
  VkPhysicalDeviceImageFormatInfo2 imageFormatInfo = {
    .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_FORMAT_INFO_2,
    .format = format,
    .type = VK_IMAGE_TYPE_2D,
    .tiling = VK_IMAGE_TILING_DRM_FORMAT_MODIFIER_EXT,
    .usage = VK_IMAGE_USAGE_SAMPLED_BIT,
  };

  std::array<VkFormat, 2> formats = {
    DRMFormatToVulkan(drmFormat, false),
    DRMFormatToVulkan(drmFormat, true),
  };

  VkImageFormatListCreateInfo formatList = {
    .sType = VK_STRUCTURE_TYPE_IMAGE_FORMAT_LIST_CREATE_INFO,
    .viewFormatCount = (uint32_t)formats.size(),
    .pViewFormats = formats.data(),
  };

  if ( formats[0] != formats[1] )
    {
      formatList.pNext = std::exchange(imageFormatInfo.pNext,
				       &formatList);
      imageFormatInfo.flags |= VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT;
    }

  VkPhysicalDeviceImageDrmFormatModifierInfoEXT modifierInfo = {
    .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_DRM_FORMAT_MODIFIER_INFO_EXT,
    .pNext = nullptr,
    .drmFormatModifier = modifier,
  };

  modifierInfo.pNext = std::exchange(imageFormatInfo.pNext, &modifierInfo);

  VkImageFormatProperties2 imageFormatProps = {
    .sType = VK_STRUCTURE_TYPE_IMAGE_FORMAT_PROPERTIES_2,
  };

  VkResult res = g_device.vk.GetPhysicalDeviceImageFormatProperties2( g_device.physDev(), &imageFormatInfo, &imageFormatProps );
  return res == VK_SUCCESS;
}

bool vulkan_init_format(VkFormat format, uint32_t drmFormat)
{
	// First, check whether the Vulkan format is supported
	VkPhysicalDeviceImageFormatInfo2 imageFormatInfo = {
		.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_FORMAT_INFO_2,
		.format = format,
		.type = VK_IMAGE_TYPE_2D,
		.tiling = VK_IMAGE_TILING_OPTIMAL,
		.usage = VK_IMAGE_USAGE_SAMPLED_BIT,
	};

	std::array<VkFormat, 2> formats = {
		DRMFormatToVulkan(drmFormat, false),
		DRMFormatToVulkan(drmFormat, true),
	};

	VkImageFormatListCreateInfo formatList = {
		.sType = VK_STRUCTURE_TYPE_IMAGE_FORMAT_LIST_CREATE_INFO,
		.viewFormatCount = (uint32_t)formats.size(),
		.pViewFormats = formats.data(),
	};

	if ( formats[0] != formats[1] )
	{
		formatList.pNext = std::exchange(imageFormatInfo.pNext,
						 &formatList);
		imageFormatInfo.flags |= VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT;
	}


	VkImageFormatProperties2 imageFormatProps = {
		.sType = VK_STRUCTURE_TYPE_IMAGE_FORMAT_PROPERTIES_2,
	};

	VkResult res = g_device.vk.GetPhysicalDeviceImageFormatProperties2( g_device.physDev(), &imageFormatInfo, &imageFormatProps );
	if ( res == VK_ERROR_FORMAT_NOT_SUPPORTED )
	{
		return false;
	}
	else if ( res != VK_SUCCESS )
	{
		vk_errorf( res, "vkGetPhysicalDeviceImageFormatProperties2 failed for DRM format 0x%" PRIX32, drmFormat );
		return false;
	}

	wlr_drm_format_set_add( &sampledShmFormats, drmFormat, DRM_FORMAT_MOD_LINEAR );

	if ( g_device.supportsModifiers() )
	{
		// Then, collect the list of modifiers supported for sampled usage
		VkDrmFormatModifierPropertiesListEXT modifierPropList = {
			.sType = VK_STRUCTURE_TYPE_DRM_FORMAT_MODIFIER_PROPERTIES_LIST_EXT,
		};
		VkFormatProperties2 formatProps = {
			.sType = VK_STRUCTURE_TYPE_FORMAT_PROPERTIES_2,
			.pNext = &modifierPropList,
		};

		g_device.vk.GetPhysicalDeviceFormatProperties2( g_device.physDev(), format, &formatProps );

		if ( modifierPropList.drmFormatModifierCount == 0 )
		{
			vk_errorf( res, "vkGetPhysicalDeviceFormatProperties2 returned zero modifiers for DRM format 0x%" PRIX32, drmFormat );
			return false;
		}

		std::vector<VkDrmFormatModifierPropertiesEXT> modifierProps(modifierPropList.drmFormatModifierCount);
		modifierPropList.pDrmFormatModifierProperties = modifierProps.data();
		g_device.vk.GetPhysicalDeviceFormatProperties2( g_device.physDev(), format, &formatProps );

		std::map< uint64_t, VkDrmFormatModifierPropertiesEXT > map = {};

		for ( size_t j = 0; j < modifierProps.size(); j++ )
		{
			map[ modifierProps[j].drmFormatModifier ] = modifierProps[j];

			uint64_t modifier = modifierProps[j].drmFormatModifier;

			if ( !is_image_format_modifier_supported( format, drmFormat, modifier ) )
				continue;

			if ( ( modifierProps[j].drmFormatModifierTilingFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT ) == 0 )
			{
				continue;
			}

			// The deferred backend exposes all sample-able formats as supported modifiers.
			if ( !g_bAllowDeferredBackend )
			{
				if ( GetBackend()->UsesModifiers() && !gamescope::Algorithm::Contains( GetBackend()->GetSupportedModifiers( drmFormat ), modifier ) )
					continue;
			}

			wlr_drm_format_set_add( &sampledDRMFormats, drmFormat, modifier );
			s_SampledModifierFormats[ drmFormat ].emplace_back( modifier );
		}

		DRMModifierProps[ format ] = map;
		return true;
	}
	else
	{
		if ( GetBackend()->UsesModifiers() && !GetBackend()->SupportsInvalidModifier( drmFormat ) )
			return false;

		wlr_drm_format_set_add( &sampledDRMFormats, drmFormat, DRM_FORMAT_MOD_INVALID );
		return false;
	}
}

bool vulkan_init_formats()
{
	for ( size_t i = 0; s_DRMVKFormatTable[i].DRMFormat != DRM_FORMAT_INVALID; i++ )
	{
		if (s_DRMVKFormatTable[i].internal)
			continue;

		VkFormat format = s_DRMVKFormatTable[i].vkFormat;
		VkFormat srgbFormat = s_DRMVKFormatTable[i].vkFormatSrgb;
		uint32_t drmFormat = s_DRMVKFormatTable[i].DRMFormat;

		vulkan_init_format(format, drmFormat);
		if (format != srgbFormat)
			vulkan_init_format(srgbFormat, drmFormat);
	}

	vk_log.infof( "supported DRM formats for sampling usage:" );
	for ( size_t i = 0; i < sampledDRMFormats.len; i++ )
	{
		uint32_t fmt = sampledDRMFormats.formats[ i ].format;
#if HAVE_DRM
		char *name = drmGetFormatName(fmt);
		vk_log.infof( "  %s (0x%" PRIX32 ")", name, fmt );
		free(name);
#endif
	}

	return true;
}

bool acquire_next_image( void )
{
	VkResult res = g_device.vk.AcquireNextImageKHR( g_device.device(), g_output.swapChain, UINT64_MAX, VK_NULL_HANDLE, g_output.acquireFence, &g_output.nOutImage );
	if ( res != VK_SUCCESS && res != VK_SUBOPTIMAL_KHR )
		return false;
	if ( g_device.vk.WaitForFences( g_device.device(), 1, &g_output.acquireFence, false, UINT64_MAX ) != VK_SUCCESS )
		return false;
	return g_device.vk.ResetFences( g_device.device(), 1, &g_output.acquireFence ) == VK_SUCCESS;
}


static std::atomic<uint64_t> g_currentPresentWaitId = {0u};
static std::mutex present_wait_lock;

extern void mangoapp_output_update( uint64_t vblanktime );
static void present_wait_thread_func( void )
{
	uint64_t present_wait_id = 0;

	while (true)
	{
		g_currentPresentWaitId.wait(present_wait_id);

		// Lock to make sure swapchain destruction is waited on and that
		// it's for this swapchain.
		{
			std::unique_lock lock(present_wait_lock);
			present_wait_id = g_currentPresentWaitId.load();

			if (present_wait_id != 0)
			{
				g_device.vk.WaitForPresentKHR( g_device.device(), g_output.swapChain, present_wait_id, 1'000'000'000lu );
				uint64_t vblanktime = get_time_in_nanos();
				GetVBlankTimer().MarkVBlank( vblanktime, true );
				mangoapp_output_update( vblanktime );
			}
		}
	}
}

void vulkan_update_swapchain_hdr_metadata( VulkanOutput_t *pOutput )
{
	if (!g_output.swapchainHDRMetadata)
		return;

	if ( !g_device.vk.SetHdrMetadataEXT )
	{
		static bool s_bWarned = false;
		if (!s_bWarned)
		{
			vk_log.errorf("Unable to forward HDR metadata with Vulkan as vkSetMetadataEXT is not supported.");
			s_bWarned = true;
		}
		return;
	}

	const hdr_metadata_infoframe &infoframe = g_output.swapchainHDRMetadata->View<hdr_output_metadata>().hdmi_metadata_type1;
	VkHdrMetadataEXT metadata =
	{
		.sType = VK_STRUCTURE_TYPE_HDR_METADATA_EXT,
		.displayPrimaryRed = VkXYColorEXT { color_xy_from_u16(infoframe.display_primaries[0].x), color_xy_from_u16(infoframe.display_primaries[0].y) },
		.displayPrimaryGreen = VkXYColorEXT { color_xy_from_u16(infoframe.display_primaries[1].x), color_xy_from_u16(infoframe.display_primaries[1].y), },
		.displayPrimaryBlue = VkXYColorEXT { color_xy_from_u16(infoframe.display_primaries[2].x), color_xy_from_u16(infoframe.display_primaries[2].y), },
		.whitePoint = VkXYColorEXT { color_xy_from_u16(infoframe.white_point.x), color_xy_from_u16(infoframe.white_point.y), },
		.maxLuminance = nits_from_u16(infoframe.max_display_mastering_luminance),
		.minLuminance = nits_from_u16_dark(infoframe.min_display_mastering_luminance),
		.maxContentLightLevel = nits_from_u16(infoframe.max_cll),
		.maxFrameAverageLightLevel = nits_from_u16(infoframe.max_fall),
	};
	g_device.vk.SetHdrMetadataEXT(g_device.device(), 1, &g_output.swapChain, &metadata);
}

void vulkan_present_to_window( void )
{
	static uint64_t s_lastPresentId = 0;

	uint64_t presentId = ++s_lastPresentId;
	
	auto feedback = steamcompmgr_get_base_layer_swapchain_feedback();
	if (feedback && feedback->hdr_metadata_blob)
	{
		if ( feedback->hdr_metadata_blob != g_output.swapchainHDRMetadata )
		{
			g_output.swapchainHDRMetadata = feedback->hdr_metadata_blob;
			vulkan_update_swapchain_hdr_metadata( &g_output );
		}
	}
	else if ( g_output.swapchainHDRMetadata != nullptr )
	{
		// Only way to clear hdr metadata for a swapchain in Vulkan
		// is to recreate the swapchain.
		g_output.swapchainHDRMetadata = nullptr;
		vulkan_remake_swapchain();
	}


	VkPresentIdKHR presentIdInfo = {
		.sType = VK_STRUCTURE_TYPE_PRESENT_ID_KHR,
		.swapchainCount = 1,
		.pPresentIds = &presentId,
	};

	VkPresentInfoKHR presentInfo = {
		.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
		.pNext = &presentIdInfo,
		.swapchainCount = 1,
		.pSwapchains = &g_output.swapChain,
		.pImageIndices = &g_output.nOutImage,
	};

	if ( g_device.vk.QueuePresentKHR( g_device.queue(), &presentInfo ) == VK_SUCCESS )
	{
		g_currentPresentWaitId = presentId;
		g_currentPresentWaitId.notify_all();
	}
	else
		vulkan_remake_swapchain();

	while ( !acquire_next_image() )
		vulkan_remake_swapchain();
}

gamescope::Rc<CVulkanTexture> vulkan_create_1d_lut(uint32_t size)
{
	CVulkanTexture::createFlags flags;
	flags.bSampled = true;
	flags.bTransferDst = true;
	flags.imageType = VK_IMAGE_TYPE_1D;

	auto texture = new CVulkanTexture();
	auto drmFormat = VulkanFormatToDRM( VK_FORMAT_R16G16B16A16_UNORM );
	bool bRes = texture->BInit( size, 1u, 1u, drmFormat, flags );
	assert( bRes );

	return texture;
}

gamescope::Rc<CVulkanTexture> vulkan_create_3d_lut(uint32_t width, uint32_t height, uint32_t depth)
{
	CVulkanTexture::createFlags flags;
	flags.bSampled = true;
	flags.bTransferDst = true;
	flags.imageType = VK_IMAGE_TYPE_3D;

	auto texture = new CVulkanTexture();
	auto drmFormat = VulkanFormatToDRM( VK_FORMAT_R16G16B16A16_UNORM );
	bool bRes = texture->BInit( width, height, depth, drmFormat, flags );
	assert( bRes );

	return texture;
}

void vulkan_update_luts(const gamescope::Rc<CVulkanTexture>& lut1d, const gamescope::Rc<CVulkanTexture>& lut3d, void* lut1d_data, void* lut3d_data)
{
	size_t lut1d_size = lut1d->width() * sizeof(uint16_t) * 4;
	size_t lut3d_size = lut3d->width() * lut3d->height() * lut3d->depth() * sizeof(uint16_t) * 4;

	auto [base_dst, base_offset] = g_device.uploadBufferData(lut1d_size + lut3d_size);

	void* lut1d_dst = base_dst;
	void *lut3d_dst = ((uint8_t*)base_dst) + lut1d_size;
	memcpy(lut1d_dst, lut1d_data, lut1d_size);
	memcpy(lut3d_dst, lut3d_data, lut3d_size);

	auto cmdBuffer = g_device.commandBuffer();
	cmdBuffer->copyBufferToImage(g_device.uploadBuffer(), base_offset, 0, lut1d);
	cmdBuffer->copyBufferToImage(g_device.uploadBuffer(), base_offset + lut1d_size, 0, lut3d);
	g_device.submit(std::move(cmdBuffer));
	g_device.waitIdle(); // TODO: Sync this better
}

gamescope::Rc<CVulkanTexture> vulkan_get_hacky_blank_texture()
{
	return g_output.temporaryHackyBlankImage.get();
}

gamescope::OwningRc<CVulkanTexture> vulkan_create_flat_texture( uint32_t width, uint32_t height, uint8_t r, uint8_t g, uint8_t b, uint8_t a )
{
	CVulkanTexture::createFlags flags;
	flags.bFlippable = true;
	flags.bSampled = true;
	flags.bTransferDst = true;

	gamescope::OwningRc<CVulkanTexture> texture = new CVulkanTexture();
	bool bRes = texture->BInit( width, height, 1u, VulkanFormatToDRM( VK_FORMAT_B8G8R8A8_UNORM ), flags );
	assert( bRes );

	auto [_dst, offset] = g_device.uploadBufferData( width * height * 4 );
	uint8_t *dst = (uint8_t *)_dst;
	for ( uint32_t i = 0; i < width * height * 4; i += 4 )
	{
		dst[i + 0] = b;
		dst[i + 1] = g;
		dst[i + 2] = r;
		dst[i + 3] = a;
	}

	auto cmdBuffer = g_device.commandBuffer();
	cmdBuffer->copyBufferToImage(g_device.uploadBuffer(), offset, 0, texture.get());
	g_device.submit(std::move(cmdBuffer));
	g_device.waitIdle();

	return texture;
}

gamescope::OwningRc<CVulkanTexture> vulkan_create_debug_blank_texture()
{
	// To match Steam's scaling, which is capped at 1080p
	int width = std::min<int>( g_nOutputWidth, 1920 );
	int height = std::min<int>( g_nOutputHeight, 1080 );

	return vulkan_create_flat_texture( width, height, 0, 0, 0, 0 );
}

bool vulkan_supports_hdr10()
{
	for ( auto& format : g_output.surfaceFormats )
	{
		if ( format.colorSpace == VK_COLOR_SPACE_HDR10_ST2084_EXT )
			return true;
	}

	return false;
}

extern bool g_bOutputHDREnabled;

bool vulkan_make_swapchain( VulkanOutput_t *pOutput )
{
	uint32_t imageCount = pOutput->surfaceCaps.minImageCount + 1;
	uint32_t formatCount = pOutput->surfaceFormats.size();
	uint32_t surfaceFormat = formatCount;
	VkColorSpaceKHR preferredColorSpace = g_bOutputHDREnabled ? VK_COLOR_SPACE_HDR10_ST2084_EXT : VK_COLOR_SPACE_SRGB_NONLINEAR_KHR;

	if ( surfaceFormat == formatCount )
	{
		for ( surfaceFormat = 0; surfaceFormat < formatCount; surfaceFormat++ )
		{
			if ( pOutput->surfaceFormats[ surfaceFormat ].format == VK_FORMAT_A2B10G10R10_UNORM_PACK32 &&
				 pOutput->surfaceFormats[ surfaceFormat ].colorSpace == preferredColorSpace )
				break;
		}
	}

	if ( surfaceFormat == formatCount )
	{
		for ( surfaceFormat = 0; surfaceFormat < formatCount; surfaceFormat++ )
		{
			if ( pOutput->surfaceFormats[ surfaceFormat ].format == VK_FORMAT_A2R10G10B10_UNORM_PACK32 &&
				 pOutput->surfaceFormats[ surfaceFormat ].colorSpace == preferredColorSpace )
				break;
		}
	}

	if ( surfaceFormat == formatCount )
	{
		for ( surfaceFormat = 0; surfaceFormat < formatCount; surfaceFormat++ )
		{
			if ( pOutput->surfaceFormats[ surfaceFormat ].format == VK_FORMAT_B8G8R8A8_UNORM &&
				 pOutput->surfaceFormats[ surfaceFormat ].colorSpace == preferredColorSpace )
				break;
		}
	}
	
	if ( surfaceFormat == formatCount )
		return false;

	VkFormat eVkFormat = pOutput->surfaceFormats[ surfaceFormat ].format;
	pOutput->uOutputFormat = VulkanFormatToDRM( pOutput->surfaceFormats[ surfaceFormat ].format );
	
	VkFormat formats[2] =
	{
		ToSrgbVulkanFormat( eVkFormat ),
		ToLinearVulkanFormat( eVkFormat ),
	};

	VkImageFormatListCreateInfo usageListInfo = {
		.sType = VK_STRUCTURE_TYPE_IMAGE_FORMAT_LIST_CREATE_INFO,
		.viewFormatCount = 2,
		.pViewFormats = formats,
	};

	vk_log.infof("Creating Gamescope nested swapchain with format %u and colorspace %u", eVkFormat, pOutput->surfaceFormats[surfaceFormat].colorSpace);

	VkSwapchainCreateInfoKHR createInfo = {
		.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
		.pNext = formats[0] != formats[1] ? &usageListInfo : nullptr,
		.flags = formats[0] != formats[1] ? VK_SWAPCHAIN_CREATE_MUTABLE_FORMAT_BIT_KHR : (VkSwapchainCreateFlagBitsKHR )0,
		.surface = pOutput->surface,
		.minImageCount = imageCount,
		.imageFormat = eVkFormat,
		.imageColorSpace = pOutput->surfaceFormats[surfaceFormat].colorSpace,
		.imageExtent = {
			.width = g_nOutputWidth,
			.height = g_nOutputHeight,
		},
		.imageArrayLayers = 1,
		.imageUsage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
		.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE,
		.preTransform = pOutput->surfaceCaps.currentTransform,
		.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
		.presentMode = VK_PRESENT_MODE_FIFO_KHR,
		.clipped = VK_TRUE,
	};

	if (g_device.vk.CreateSwapchainKHR( g_device.device(), &createInfo, nullptr, &pOutput->swapChain) != VK_SUCCESS ) {
		return false;
	}

	g_device.vk.GetSwapchainImagesKHR( g_device.device(), pOutput->swapChain, &imageCount, nullptr );
	std::vector<VkImage> swapchainImages( imageCount );
	g_device.vk.GetSwapchainImagesKHR( g_device.device(), pOutput->swapChain, &imageCount, swapchainImages.data() );

	pOutput->outputImages.resize(imageCount);

	for ( uint32_t i = 0; i < pOutput->outputImages.size(); i++ )
	{
		pOutput->outputImages[i] = new CVulkanTexture();

		if ( !pOutput->outputImages[i]->BInitFromSwapchain(swapchainImages[i], g_nOutputWidth, g_nOutputHeight, eVkFormat))
			return false;
	}

	VkFenceCreateInfo fenceInfo = {
		.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
	};

	g_device.vk.CreateFence( g_device.device(), &fenceInfo, nullptr, &pOutput->acquireFence );

	vulkan_update_swapchain_hdr_metadata(pOutput);

	return true;
}

bool vulkan_remake_swapchain( void )
{
	std::unique_lock lock(present_wait_lock);
	g_currentPresentWaitId = 0;
	g_currentPresentWaitId.notify_all();

	VulkanOutput_t *pOutput = &g_output;
	g_device.waitIdle();
	g_device.vk.QueueWaitIdle( g_device.queue() );
	vulkan_framegen_reset( "swapchain_remake" );

	pOutput->outputImages.clear();

	g_device.vk.DestroySwapchainKHR( g_device.device(), pOutput->swapChain, nullptr );

	// Delete screenshot/capture textures to be remade if needed
	for (auto& pScreenshotTexture : pOutput->pScreenshotTextures)
		pScreenshotTexture = nullptr;
	for (auto& pCaptureTexture : pOutput->pCaptureTextures)
		pCaptureTexture = nullptr;

	bool bRet = vulkan_make_swapchain( pOutput );
	assert( bRet ); // Something has gone horribly wrong!
	return bRet;
}

// The classic output ring is 3 (ping/pong plus one for partial composition).
// Frame generation retains the last two composited output images as its
// prediction history (zero-copy), so it needs a deeper ring for history,
// backend-owned commits, and the next real target. The framegen policy scales
// that capacity from 8 at x2 to 12 at x4: a higher generated presentation rate
// can build a deeper nested-compositor queue before wl_buffer.release catches
// up. Generated frames themselves remain in a disjoint pool. Capacity only
// absorbs bounded bursts; the ownership selector below is the safety rule.
static constexpr uint32_t k_uOutputRingSizeDefault = 3;

// Frame generation holds output-ring images as history while the backend may
// independently retain older scanout commits. Pick by actual ownership rather
// than assuming a fixed commit depth: writing a nested-Wayland wl_buffer before
// its release is a protocol violation and can appear as tearing or edge bleed.
static std::optional<uint32_t> framegen_find_available_output_image( uint32_t nFirst )
{
	const uint32_t nRing = (uint32_t)g_output.outputImages.size();
	if ( nRing == 0 )
		return std::nullopt;

	for ( uint32_t i = 0; i < nRing; i++ )
	{
		const uint32_t nIndex = ( nFirst + i ) % nRing;
		CVulkanTexture *pImage = g_output.outputImages[ nIndex ].get();
		if ( pImage != nullptr && !pImage->IsInUse() )
			return nIndex;
	}

	return std::nullopt;
}

static bool vulkan_make_output_images( VulkanOutput_t *pOutput )
{
	CVulkanTexture::createFlags outputImageflags;
	outputImageflags.bFlippable = true;
	outputImageflags.bStorage = true;
	outputImageflags.bTransferSrc = true; // for screenshots
	outputImageflags.bSampled = true; // for pipewire blits
	outputImageflags.bOutputImage = true;
	// Output-space frame generation retains these as history and samples them on
	// the framegen queue, which may be a different family than the compositor.
	outputImageflags.bFramegenShared = true;

	const uint32_t nRing = vulkan_framegen_is_enabled()
		? gamescope::framegen::output_ring_size_for_multiplier( g_nFramegenMultiplier )
		: k_uOutputRingSizeDefault;

	pOutput->outputImages.resize(nRing);
	pOutput->outputImagesPartialOverlay.resize(nRing);
	for ( uint32_t i = 0; i < nRing; i++ )
	{
		pOutput->outputImages[i] = nullptr;
		pOutput->outputImagesPartialOverlay[i] = nullptr;
	}

	uint32_t uDRMFormat = pOutput->uOutputFormat;

	for ( uint32_t i = 0; i < nRing; i++ )
	{
		pOutput->outputImages[i] = new CVulkanTexture();
		if ( !pOutput->outputImages[i]->BInit( g_nOutputWidth, g_nOutputHeight, 1u, uDRMFormat, outputImageflags ) )
		{
			vk_log.errorf( "failed to allocate buffer for KMS" );
			return false;
		}
	}

	// Oh no.
	pOutput->temporaryHackyBlankImage = vulkan_create_debug_blank_texture();

	// Partial composition aliases outputImagesPartialOverlay[i] onto
	// outputImages[i]'s VkDeviceMemory. Frame generation retains outputImages
	// slots as history, so a partial composite writing the aliased overlay
	// image would silently corrupt that history. Framegen forces full
	// composite anyway, so simply do not allocate the aliases while it is
	// active — this removes the aliasing hazard structurally.
	if ( pOutput->uOutputFormatOverlay != VK_FORMAT_UNDEFINED && !kDisablePartialComposition && !vulkan_framegen_is_enabled() )
	{
		uint32_t uPartialDRMFormat = pOutput->uOutputFormatOverlay;

		for ( uint32_t i = 0; i < nRing; i++ )
		{
			pOutput->outputImagesPartialOverlay[i] = new CVulkanTexture();
			if ( !pOutput->outputImagesPartialOverlay[i]->BInit( g_nOutputWidth, g_nOutputHeight, 1u, uPartialDRMFormat, outputImageflags, nullptr, 0, 0, pOutput->outputImages[i].get() ) )
			{
				vk_log.errorf( "failed to allocate buffer for KMS" );
				return false;
			}
		}
	}

	return true;
}

bool vulkan_remake_output_images()
{
	VulkanOutput_t *pOutput = &g_output;
	g_device.waitIdle();
	vulkan_framegen_reset( "output_images_remade" );

	pOutput->nOutImage = 0;

	// Delete screenshot/capture textures to be remade if needed
	for (auto& pScreenshotTexture : pOutput->pScreenshotTextures)
		pScreenshotTexture = nullptr;
	for (auto& pCaptureTexture : pOutput->pCaptureTextures)
		pCaptureTexture = nullptr;

	bool bRet = vulkan_make_output_images( pOutput );
	assert( bRet );
	return bRet;
}

bool vulkan_make_output()
{
	VulkanOutput_t *pOutput = &g_output;

	VkResult result;
	
	if ( GetBackend()->UsesVulkanSwapchain() )
	{
		result = g_device.vk.GetPhysicalDeviceSurfaceCapabilitiesKHR( g_device.physDev(), pOutput->surface, &pOutput->surfaceCaps );
		if ( result != VK_SUCCESS )
		{
			vk_errorf( result, "vkGetPhysicalDeviceSurfaceCapabilitiesKHR failed" );
			return false;
		}
		
		uint32_t formatCount = 0;
		result = g_device.vk.GetPhysicalDeviceSurfaceFormatsKHR( g_device.physDev(), pOutput->surface, &formatCount, nullptr );
		if ( result != VK_SUCCESS )
		{
			vk_errorf( result, "vkGetPhysicalDeviceSurfaceFormatsKHR failed" );
			return false;
		}
		
		if ( formatCount != 0 ) {
			pOutput->surfaceFormats.resize( formatCount );
			g_device.vk.GetPhysicalDeviceSurfaceFormatsKHR( g_device.physDev(), pOutput->surface, &formatCount, pOutput->surfaceFormats.data() );
			if ( result != VK_SUCCESS )
			{
				vk_errorf( result, "vkGetPhysicalDeviceSurfaceFormatsKHR failed" );
				return false;
			}
		}
		
		uint32_t presentModeCount = false;
		result = g_device.vk.GetPhysicalDeviceSurfacePresentModesKHR(g_device.physDev(), pOutput->surface, &presentModeCount, nullptr );
		if ( result != VK_SUCCESS )
		{
			vk_errorf( result, "vkGetPhysicalDeviceSurfacePresentModesKHR failed" );
			return false;
		}
		
		if ( presentModeCount != 0 ) {
			pOutput->presentModes.resize(presentModeCount);
			result = g_device.vk.GetPhysicalDeviceSurfacePresentModesKHR( g_device.physDev(), pOutput->surface, &presentModeCount, pOutput->presentModes.data() );
			if ( result != VK_SUCCESS )
			{
				vk_errorf( result, "vkGetPhysicalDeviceSurfacePresentModesKHR failed" );
				return false;
			}
		}
		
		if ( !vulkan_make_swapchain( pOutput ) )
			return false;

		while ( !acquire_next_image() )
			vulkan_remake_swapchain();
	}
	else
	{
		GetBackend()->GetPreferredOutputFormat( &pOutput->uOutputFormat, &pOutput->uOutputFormatOverlay );

		if ( pOutput->uOutputFormat == DRM_FORMAT_INVALID )
		{
			vk_log.errorf( "failed to find Vulkan format suitable for KMS" );
			return false;
		}

		if ( pOutput->uOutputFormatOverlay == DRM_FORMAT_INVALID )
		{
			vk_log.errorf( "failed to find Vulkan format suitable for KMS partial overlays" );
			return false;
		}

		if ( !vulkan_make_output_images( pOutput ) )
			return false;
	}

	return true;
}

static void update_tmp_images( uint32_t width, uint32_t height )
{
	if ( g_output.tmpOutput != nullptr
			&& width == g_output.tmpOutput->width()
			&& height == g_output.tmpOutput->height() )
	{
		return;
	}

	CVulkanTexture::createFlags createFlags;
	createFlags.bSampled = true;
	createFlags.bStorage = true;

	g_output.tmpOutput = new CVulkanTexture();
	bool bSuccess = g_output.tmpOutput->BInit( width, height, 1u, DRM_FORMAT_ARGB8888, createFlags, nullptr );

	if ( !bSuccess )
	{
		vk_log.errorf( "failed to create fsr output" );
		return;
	}
}


static bool init_nis_data()
{
	// Create the NIS images
	// Select between the FP16 or FP32 coefficients

	void* coefScaleData = g_device.supportsFp16() ? (void*) coef_scale_fp16 : (void*) coef_scale;

	void* coefUsmData = g_device.supportsFp16() ? (void*) coef_usm_fp16 : (void*) coef_usm;

	uint32_t nisFormat = g_device.supportsFp16() ? DRM_FORMAT_ABGR16161616F : DRM_FORMAT_ABGR32323232F;

	uint32_t width = kFilterSize / 4;
	uint32_t height = kPhaseCount;

	g_output.nisScalerImage = vulkan_create_texture_from_bits( width, height, width, height, nisFormat, {}, coefScaleData );
	g_output.nisUsmImage = vulkan_create_texture_from_bits( width, height, width, height, nisFormat, {}, coefUsmData );

	return true;
}

VkInstance vulkan_get_instance( void )
{
	static VkInstance s_pVkInstance = []() -> VkInstance
	{
		VkResult result = VK_ERROR_INITIALIZATION_FAILED;

		if ( ( result = vulkan_load_module() ) != VK_SUCCESS )
		{
			vk_errorf( result, "Failed to load vulkan module." );
			return nullptr;
		}

		auto instanceExtensions = GetBackend()->GetInstanceExtensions();

		const VkApplicationInfo appInfo = {
			.sType              = VK_STRUCTURE_TYPE_APPLICATION_INFO,
			.pApplicationName   = "gamescope",
			.applicationVersion = VK_MAKE_VERSION(1, 0, 0),
			.pEngineName        = "hopefully not just some code",
			.engineVersion      = VK_MAKE_VERSION(1, 0, 0),
			.apiVersion         = VK_API_VERSION_1_3,
		};

		const VkInstanceCreateInfo createInfo = {
			.sType                   = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
			.pApplicationInfo        = &appInfo,
			.enabledExtensionCount   = (uint32_t)instanceExtensions.size(),
			.ppEnabledExtensionNames = instanceExtensions.data(),
		};

		VkInstance instance = nullptr;
		result = g_pfn_vkCreateInstance(&createInfo, 0, &instance);
		if ( result != VK_SUCCESS )
		{
			vk_errorf( result, "vkCreateInstance failed" );
		}

		return instance;
	}();

	return s_pVkInstance;
}

bool vulkan_init( VkInstance instance, VkSurfaceKHR surface )
{
	static bool s_bInitted = false;
	if ( s_bInitted )
	{
		g_output.surface = surface;
		return true;
	}

	if (!g_device.BInit(instance, surface))
		return false;

	if (!init_nis_data())
		return false;

	if ( GetBackend()->UsesVulkanSwapchain() )
	{
		std::thread present_wait_thread( present_wait_thread_func );
		present_wait_thread.detach();
	}

	s_bInitted = true;

	return true;
}

gamescope::OwningRc<CVulkanTexture> vulkan_create_texture_from_dmabuf( struct wlr_dmabuf_attributes *pDMA, gamescope::OwningRc<gamescope::IBackendFb> pBackendFb )
{
	gamescope::OwningRc<CVulkanTexture> pTex = new CVulkanTexture();

	CVulkanTexture::createFlags texCreateFlags;
	texCreateFlags.bSampled = true;
	// A legal cross-device import can reside in host-visible memory. Give
	// composited client buffers a transfer-source usage so their one-time
	// device-local staging copy can use vkCmdCopyImage without changing pixels.
	// That usage is only needed on the staging path though, and some sampleable
	// modifiers do not support TRANSFER_SRC, so a rejected import is retried
	// without it. Such a texture is simply not stageable and the staging code
	// (AcquireDeviceLocalStagingImage) skips it.
	texCreateFlags.bTransferSrc = true;

	//fprintf(stderr, "pDMA->width: %d pDMA->height: %d pDMA->format: 0x%x pDMA->modifier: 0x%lx pDMA->n_planes: %d\n",
	//	pDMA->width, pDMA->height, pDMA->format, pDMA->modifier, pDMA->n_planes);
	
	if ( pTex->BInit( pDMA->width, pDMA->height, 1u, pDMA->format, texCreateFlags, pDMA, 0, 0, nullptr, pBackendFb ) == false )
	{
		if ( g_bDebugDualGpuRoute )
		{
			vk_log.infof( "dual-gpu-route: client dma-buf Vulkan import with transfer-src failed %dx%d format 0x%" PRIX32 " modifier 0x%" PRIX64 " planes %d, retrying sampled-only",
				pDMA->width,
				pDMA->height,
				pDMA->format,
				pDMA->modifier,
				pDMA->n_planes );
		}

		texCreateFlags.bTransferSrc = false;
		pTex = new CVulkanTexture();

		if ( pTex->BInit( pDMA->width, pDMA->height, 1u, pDMA->format, texCreateFlags, pDMA, 0, 0, nullptr, pBackendFb ) == false )
		{
			if ( g_bDebugDualGpuRoute )
			{
				vk_log.errorf( "dual-gpu-route: client dma-buf Vulkan import failed %dx%d format 0x%" PRIX32 " modifier 0x%" PRIX64 " planes %d backend fb %s",
					pDMA->width,
					pDMA->height,
					pDMA->format,
					pDMA->modifier,
					pDMA->n_planes,
					pBackendFb ? "yes" : "no" );
			}
			return nullptr;
		}
	}

	if ( g_bDebugDualGpuRoute )
	{
		vk_log.infof( "dual-gpu-route: client dma-buf Vulkan import completed %dx%d format 0x%" PRIX32 " modifier 0x%" PRIX64 " backend fb %s",
			pDMA->width,
			pDMA->height,
			pDMA->format,
			pDMA->modifier,
			pBackendFb ? "yes" : "no" );
	}
	
	return pTex;
}

gamescope::OwningRc<CVulkanTexture> vulkan_create_texture_from_bits( uint32_t width, uint32_t height, uint32_t contentWidth, uint32_t contentHeight, uint32_t drmFormat, CVulkanTexture::createFlags texCreateFlags, void *bits )
{
	gamescope::OwningRc<CVulkanTexture> pTex = new CVulkanTexture();

	texCreateFlags.bSampled = true;
	texCreateFlags.bTransferDst = true;

	if ( pTex->BInit( width, height, 1u, drmFormat, texCreateFlags, nullptr,  contentWidth, contentHeight) == false )
		return nullptr;

	size_t size = width * height * DRMFormatGetBPP(drmFormat);
	auto [ dst, offset ] = g_device.uploadBufferData(size);
	memcpy( dst, bits, size );

	auto cmdBuffer = g_device.commandBuffer();

	cmdBuffer->copyBufferToImage(g_device.uploadBuffer(), offset, 0, pTex.get());
	// TODO: Sync this copyBufferToImage.

	g_device.submit(std::move(cmdBuffer));
	g_device.waitIdle();

	return pTex;
}

static uint32_t s_frameId = 0;
static __attribute__((noinline)) void framegen_color_probe_consume();

void vulkan_garbage_collect( void )
{
	g_device.garbageCollect();
	g_device.framegenGarbageCollect();
	framegen_color_probe_consume();
}

static gamescope::Rc<CVulkanTexture> acquire_pooled_texture( auto& pool, uint32_t width, uint32_t height, bool exportable, uint32_t drmFormat, EStreamColorspace colorspace )
{
	for (auto& pTexture : pool)
	{
		// Evict a stale texture and reuse the slot
		if (pTexture && pTexture->GetRefCount() == 0 &&
			(width != pTexture->width() ||
			 height != pTexture->height() ||
			 drmFormat != pTexture->drmFormat()))
		{
			pTexture = nullptr;
		}

		if (pTexture == nullptr)
		{
			pTexture = new CVulkanTexture();

			CVulkanTexture::createFlags textureFlags;
			textureFlags.bMappable = true;
			textureFlags.bTransferDst = true;
			textureFlags.bStorage = true;
			if (exportable || drmFormat == DRM_FORMAT_NV12) {
				textureFlags.bExportable = true;
				textureFlags.bLinear = true; // TODO: support multi-planar DMA-BUF export via PipeWire
			}

			bool bSuccess = pTexture->BInit( width, height, 1u, drmFormat, textureFlags );
			pTexture->setStreamColorspace(colorspace);

			assert( bSuccess );
		}

		if (pTexture->GetRefCount() != 0 ||
			width != pTexture->width() ||
			height != pTexture->height() ||
			drmFormat != pTexture->drmFormat())
			continue;

		return pTexture.get();
	}

	return nullptr;
}

gamescope::Rc<CVulkanTexture> vulkan_acquire_screenshot_texture(uint32_t width, uint32_t height, bool exportable, uint32_t drmFormat, EStreamColorspace colorspace)
{
	auto texture = acquire_pooled_texture(g_output.pScreenshotTextures, width, height, exportable, drmFormat, colorspace);
	if (!texture)
		vk_log.errorf("Unable to acquire screenshot texture. Out of textures.");
	return texture;
}

gamescope::Rc<CVulkanTexture> vulkan_acquire_capture_texture(uint32_t width, uint32_t height, bool exportable, uint32_t drmFormat, EStreamColorspace colorspace)
{
	auto texture = acquire_pooled_texture(g_output.pCaptureTextures, width, height, exportable, drmFormat, colorspace);
	if (!texture)
		vk_log.errorf("Unable to acquire capture texture. Out of textures.");
	return texture;
}

// Internal display's native brightness.
float g_flInternalDisplayBrightnessNits = 500.0f;

float g_flHDRItmSdrNits = 100.f;
float g_flHDRItmTargetNits = 1000.f;

#pragma pack(push, 1)
struct BlitPushData_t
{
	vec2_t scale[k_nMaxLayers];
	vec2_t offset[k_nMaxLayers];
	float opacity[k_nMaxLayers];
	glm::mat3x4 ctm[k_nMaxLayers];
	uint32_t borderMask;
	uint32_t frameId;
	uint32_t blurRadius;

	uint32_t u_shaderFilter;
	uint32_t u_alphaMode;

    float u_linearToNits; // unset
    float u_nitsToLinear; // unset
    float u_itmSdrNits; // unset
    float u_itmTargetNits; // unset

	explicit BlitPushData_t(const struct FrameInfo_t *frameInfo)
	{
		u_shaderFilter = 0;
		u_alphaMode = 0;

		for (int i = 0; i < frameInfo->layerCount; i++) {
			const FrameInfo_t::Layer_t *layer = &frameInfo->layers[i];
			scale[i] = layer->scale;
			offset[i] = layer->offsetPixelCenter();
			opacity[i] = layer->opacity;
            if (layer->isScreenSize() || (layer->filter == GamescopeUpscaleFilter::LINEAR && layer->viewConvertsToLinearAutomatically()))
                u_shaderFilter |= ((uint32_t)GamescopeUpscaleFilter::FROM_VIEW) << (i * 4);
            else
                u_shaderFilter |= ((uint32_t)layer->filter) << (i * 4);

			u_alphaMode |= ((uint32_t)layer->eAlphaBlendingMode) << ( i * 4 );

			if (layer->ctm)
			{
				ctm[i] = layer->ctm->View<glm::mat3x4>();
			}
			else
			{
				ctm[i] = glm::mat3x4
				{
					1, 0, 0, 0,
					0, 1, 0, 0,
					0, 0, 1, 0
				};
			}
		}

		borderMask = frameInfo->borderMask();
		frameId = s_frameId++;
		blurRadius = frameInfo->blurRadius ? ( frameInfo->blurRadius * 2 ) - 1 : 0;

		u_linearToNits = g_flInternalDisplayBrightnessNits;
		u_nitsToLinear = 1.0f / g_flInternalDisplayBrightnessNits;
		u_itmSdrNits = g_flHDRItmSdrNits;
		u_itmTargetNits = g_flHDRItmTargetNits;
	}

	explicit BlitPushData_t(float blit_scale) {
		scale[0] = { blit_scale, blit_scale };
		offset[0] = { 0.5f, 0.5f };
		opacity[0] = 1.0f;
        u_shaderFilter = (uint32_t)GamescopeUpscaleFilter::LINEAR;
		u_alphaMode = 0;
		ctm[0] = glm::mat3x4
		{
			1, 0, 0, 0,
			0, 1, 0, 0,
			0, 0, 1, 0
		};
		borderMask = 0;
		frameId = s_frameId;

		u_linearToNits = g_flInternalDisplayBrightnessNits;
		u_nitsToLinear = 1.0f / g_flInternalDisplayBrightnessNits;
		u_itmSdrNits = g_flHDRItmSdrNits;
		u_itmTargetNits = g_flHDRItmTargetNits;
	}
};

struct CaptureConvertBlitData_t
{
	vec2_t scale[1];
	vec2_t offset[1];
	float opacity[1];
	glm::mat3x4 ctm[1];
	mat3x4 outputCTM;
	uint32_t borderMask;
	uint32_t halfExtent[2];

	explicit CaptureConvertBlitData_t(float blit_scale, const mat3x4 &color_matrix) {
		scale[0] = { blit_scale, blit_scale };
		offset[0] = { 0.0f, 0.0f };
		opacity[0] = 1.0f;
		borderMask = 0;
		ctm[0] = glm::mat3x4
		{
			1, 0, 0, 0,
			0, 1, 0, 0,
			0, 0, 1, 0
		};
		outputCTM = color_matrix;
	}
};

struct uvec4_t
{
	uint32_t  x;
	uint32_t  y;
	uint32_t  z;
	uint32_t  w;
};
struct uvec2_t
{
	uint32_t x;
	uint32_t y;
};

struct EasuPushData_t
{
	uvec4_t Const0;
	uvec4_t Const1;
	uvec4_t Const2;
	uvec4_t Const3;

	EasuPushData_t(uint32_t inputX, uint32_t inputY, uint32_t tempX, uint32_t tempY)
	{
		FsrEasuCon(&Const0.x, &Const1.x, &Const2.x, &Const3.x, inputX, inputY, inputX, inputY, tempX, tempY);
	}
};

struct RcasPushData_t
{
	uvec2_t u_layer0Offset;
	vec2_t u_scale[k_nMaxLayers - 1];
	vec2_t u_offset[k_nMaxLayers - 1];
	float u_opacity[k_nMaxLayers];
	glm::mat3x4 ctm[k_nMaxLayers];
	uint32_t u_borderMask;
	uint32_t u_frameId;
	uint32_t u_c1;

	uint32_t u_shaderFilter;
	uint32_t u_alphaMode;

    float u_linearToNits; // unset
    float u_nitsToLinear; // unset
    float u_itmSdrNits; // unset
    float u_itmTargetNits; // unset

	RcasPushData_t(const struct FrameInfo_t *frameInfo, float sharpness)
	{
		uvec4_t tmp;
		FsrRcasCon(&tmp.x, sharpness);
		u_layer0Offset.x = uint32_t(int32_t(frameInfo->layers[0].offset.x));
		u_layer0Offset.y = uint32_t(int32_t(frameInfo->layers[0].offset.y));
		u_borderMask = frameInfo->borderMask() >> 1u;
		u_frameId = s_frameId++;
		u_c1 = tmp.x;
		u_shaderFilter = 0;
		u_alphaMode = 0;

		for (int i = 0; i < frameInfo->layerCount; i++)
		{
			const FrameInfo_t::Layer_t *layer = &frameInfo->layers[i];

            if (i == 0 || layer->isScreenSize() || (layer->filter == GamescopeUpscaleFilter::LINEAR && layer->viewConvertsToLinearAutomatically()))
                u_shaderFilter |= ((uint32_t)GamescopeUpscaleFilter::FROM_VIEW) << (i * 4);
            else
                u_shaderFilter |= ((uint32_t)layer->filter) << (i * 4);

			u_alphaMode |= ((uint32_t)layer->eAlphaBlendingMode) << ( i * 4 );

			if (layer->ctm)
			{
				ctm[i] = layer->ctm->View<glm::mat3x4>();
			}
			else
			{
				ctm[i] = glm::mat3x4
				{
					1, 0, 0, 0,
					0, 1, 0, 0,
					0, 0, 1, 0
				};
			}

			u_opacity[i] = frameInfo->layers[i].opacity;
		}

		u_linearToNits = g_flInternalDisplayBrightnessNits;
		u_nitsToLinear = 1.0f / g_flInternalDisplayBrightnessNits;
		u_itmSdrNits = g_flHDRItmSdrNits;
		u_itmTargetNits = g_flHDRItmTargetNits;

		for (uint32_t i = 1; i < k_nMaxLayers; i++)
		{
			u_scale[i - 1] = frameInfo->layers[i].scale;
			u_offset[i - 1] = frameInfo->layers[i].offsetPixelCenter();
		}
	}
};

struct NisPushData_t
{
	NISConfig nisConfig;

	NisPushData_t(uint32_t inputX, uint32_t inputY, uint32_t tempX, uint32_t tempY, float sharpness)
	{
		NVScalerUpdateConfig(
			nisConfig, sharpness,
			0, 0,
			inputX, inputY,
			inputX, inputY,
			0, 0,
			tempX, tempY,
			tempX, tempY);
	}
};
#pragma pack(pop)

void bind_all_layers(CVulkanCmdBuffer* cmdBuffer, const struct FrameInfo_t *frameInfo)
{
	for ( int i = 0; i < frameInfo->layerCount; i++ )
	{
		const FrameInfo_t::Layer_t *layer = &frameInfo->layers[i];

		bool nearest = layer->isScreenSize()
                    || layer->filter == GamescopeUpscaleFilter::NEAREST
                    || (layer->filter == GamescopeUpscaleFilter::LINEAR && !layer->viewConvertsToLinearAutomatically());

		cmdBuffer->bindTexture(i, layer->tex);
		cmdBuffer->setTextureSrgb(i, layer->colorspace != GAMESCOPE_APP_TEXTURE_COLORSPACE_LINEAR);
		cmdBuffer->setSamplerNearest(i, nearest);
		cmdBuffer->setSamplerUnnormalized(i, true);
	}
	for (uint32_t i = frameInfo->layerCount; i < VKR_SAMPLER_SLOTS; i++)
	{
		cmdBuffer->bindTexture(i, nullptr);
	}
}

std::optional<uint64_t> vulkan_screenshot( const struct FrameInfo_t *frameInfo, gamescope::Rc<CVulkanTexture> pScreenshotTexture, gamescope::Rc<CVulkanTexture> pYUVOutTexture )
{
	EOTF outputTF = frameInfo->outputEncodingEOTF;
	if (!frameInfo->applyOutputColorMgmt)
		outputTF = EOTF_Count; //Disable blending stuff.

	auto cmdBuffer = g_device.commandBuffer();

	for (uint32_t i = 0; i < EOTF_Count; i++)
		cmdBuffer->bindColorMgmtLuts(i, frameInfo->shaperLut[i], frameInfo->lut3D[i]);

	cmdBuffer->bindPipeline( g_device.pipeline(SHADER_TYPE_BLIT, frameInfo->layerCount, frameInfo->ycbcrMask(), 0u, frameInfo->colorspaceMask(), outputTF ));
	bind_all_layers(cmdBuffer.get(), frameInfo);
	cmdBuffer->bindTarget(pScreenshotTexture);
	cmdBuffer->uploadConstants<BlitPushData_t>(frameInfo);

	const int pixelsPerGroup = 8;

	cmdBuffer->dispatch(div_roundup(currentOutputWidth, pixelsPerGroup), div_roundup(currentOutputHeight, pixelsPerGroup));

	if ( pYUVOutTexture != nullptr )
	{
		float scale = (float)pScreenshotTexture->width() / pYUVOutTexture->width();

		CaptureConvertBlitData_t constants( scale, colorspace_to_conversion_from_srgb_matrix( pYUVOutTexture->streamColorspace() ) );
		constants.halfExtent[0] = pYUVOutTexture->width() / 2.0f;
		constants.halfExtent[1] = pYUVOutTexture->height() / 2.0f;
		cmdBuffer->uploadConstants<CaptureConvertBlitData_t>(constants);

		for (uint32_t i = 0; i < EOTF_Count; i++)
			cmdBuffer->bindColorMgmtLuts(i, nullptr, nullptr);

		cmdBuffer->bindPipeline(g_device.pipeline( SHADER_TYPE_RGB_TO_NV12, 1, 0, 0, GAMESCOPE_APP_TEXTURE_COLORSPACE_SRGB, EOTF_Count ));
		cmdBuffer->bindTexture(0, pScreenshotTexture);
		cmdBuffer->setTextureSrgb(0, true);
		cmdBuffer->setSamplerNearest(0, false);
		cmdBuffer->setSamplerUnnormalized(0, true);
		for (uint32_t i = 1; i < VKR_SAMPLER_SLOTS; i++)
		{
			cmdBuffer->bindTexture(i, nullptr);
		}
		cmdBuffer->bindTarget(pYUVOutTexture);

		const int pixelsPerGroup = 8;

		// For ycbcr, we operate on 2 pixels at a time, so use the half-extent.
		const int dispatchSize = pixelsPerGroup * 2;

		cmdBuffer->dispatch(div_roundup(pYUVOutTexture->width(), dispatchSize), div_roundup(pYUVOutTexture->height(), dispatchSize));
	}

	uint64_t sequence = g_device.submit(std::move(cmdBuffer));
	return sequence;
}

extern std::string g_reshade_effect;
extern uint32_t g_reshade_technique_idx;

ReshadeEffectPipeline *g_pLastReshadeEffect = nullptr;

// Motion-estimation intermediates (low-resolution luma pyramids and the motion
// field). Allocated lazily by the motion dispatch, released on framegen reset.
// The matcher runs coarse-to-fine over three pyramid levels: full search only
// at the smallest, +/-1 seeded refinement below it (framegen_prepare_motion).
struct FramegenMotionResources_t
{
	// Finest (base low-res) level: what the warp's field scale is keyed to.
	gamescope::OwningRc<CVulkanTexture> lumaPrev;
	gamescope::OwningRc<CVulkanTexture> lumaCur;
	gamescope::OwningRc<CVulkanTexture> mvField;
	// Coarser levels, each a further 2x box downscale: [0] = /2, [1] = /4 of
	// the base low-res grid, with an intermediate motion field per level.
	gamescope::OwningRc<CVulkanTexture> lumaPrevCoarse[2];
	gamescope::OwningRc<CVulkanTexture> lumaCurCoarse[2];
	gamescope::OwningRc<CVulkanTexture> mvFieldCoarse[2];
	// Forward-backward consistency (only when enabled): the unchecked forward
	// field and the reverse (prev-anchored) field the check pass reads; the
	// checked result lands in mvField. The reverse chain reuses the coarse
	// fields above as scratch, so these are the only extra allocations.
	gamescope::OwningRc<CVulkanTexture> mvFieldFwd;
	gamescope::OwningRc<CVulkanTexture> mvFieldRev;
	// Bidirectional interpolation (B3, only when active): the CHECKED reverse
	// field the bidir warp gathers the previous frame along. The raw reverse
	// field above is a check INPUT and stays as-is; this is the symmetric
	// check's output (reverse vectors whose round trip through the forward
	// field does not close lose their confidence too).
	gamescope::OwningRc<CVulkanTexture> mvFieldRevChk;
	// Predict-pipeline causal acceleration: a retained copy of the preceding
	// interval's final checked forward field. It is reprojected through the
	// current field before differencing, so acceleration compares the same
	// moving content rather than the same screen coordinate. The frame ID gate
	// rejects stale history when shared-queue admission skipped an interval.
	gamescope::OwningRc<CVulkanTexture> mvFieldHistory;
	uint64_t uMotionHistoryFrameId = 0;
	uint64_t uMotionHistoryIntervalNs = 0;
	// The finalized field for the current real-frame pair remains resident in
	// mvField/mvFieldNet. Deadline slots and classic idle refills reuse it instead of estimating,
	// refining, probing and training on the same pair again. On the next
	// consecutive pair it is copied to mvFieldHistory before preparation
	// overwrites the working images.
	uint64_t uMotionFieldFrameId = 0;
	uint64_t uMotionFieldIntervalNs = 0;
	GamescopeFramegenPipeline eMotionFieldPipeline = GamescopeFramegenPipeline::Warp;
	bool bMotionFieldBidir = false;
	// Guided-pipeline frames-only disocclusion evidence. This low-resolution
	// image contains luma from two intervals ago: a batch samples it during
	// every warp, then refreshes it from lumaPrev after all warps complete. The
	// internally-owned copy avoids retaining a third output-ring slot, and luma
	// is sufficient because history validates a layer rather than supplying
	// the displayed color.
	gamescope::OwningRc<CVulkanTexture> lumaReservoir[2];
	uint64_t uLumaReservoirFrameId[2] = {};
	bool bLumaReservoirAllocTried = false;
	// Self-supervised adaptation (B4): the 96x1 R32_UINT counter image the
	// stats probe atomically accumulates into (applied to the motion field in
	// the same batch), and its host-mapped linear copy the CPU parses one batch
	// later to auto-calibrate thresholds. 384 bytes
	// each; allocated with the other motion intermediates so the warps can
	// bind the accumulator unconditionally.
	gamescope::OwningRc<CVulkanTexture> statsAccum;
	gamescope::OwningRc<CVulkanTexture> statsReadback;
	// Learned field refinement (Stage C, only with a loaded weights blob): the
	// net's refined copies of both checked fields (a conv reads an apron of
	// raw neighbors, so refinement can never be in place — see
	// framegen_motion_field()), the sampled weight texture and the host-
	// visible staging it is copied from once per (re)allocation.
	gamescope::OwningRc<CVulkanTexture> mvFieldNet;
	gamescope::OwningRc<CVulkanTexture> mvFieldRevNet;
	// Fourth CNN head: a zero-neutral focus mask for persistent non-geometric
	// color change. Kept at field resolution and sampled only by Guided's
	// existing full-resolution warp, so it adds no output-resolution pass.
	gamescope::OwningRc<CVulkanTexture> netShadingFocus;
	gamescope::OwningRc<CVulkanTexture> netWeightsGpu;
	gamescope::OwningRc<CVulkanTexture> netWeightsUpload;
	bool bNetWeightsUploaded = false;
	bool bNetAllocTried = false;
	// In-situ learning (C2, only with GAMESCOPE_FRAMEGEN_NET_ONLINE): the
	// immutable prior the fast weights decay toward, the optimizer state
	// (fast weights / Adam m / Adam v / served EMA, one row each), the
	// per-tile gradient slices the training pass writes and the optimizer
	// sums (no float atomics — not universal), and the mappable readback the
	// served weights are copied through every trained step — it feeds the
	// CPU-side health check (non-finite => re-init from the prior) and the
	// profile persistence (checkpoint + exit/reset flush). All
	// resolution-independent, allocated once. uNetTrainStep is the Adam step
	// counter (also the tile-placement seed); bNetStatePending requests the
	// state-initialization dispatch on the next online batch.
	gamescope::OwningRc<CVulkanTexture> netWeightsPrior;
	gamescope::OwningRc<CVulkanTexture> netState;
	gamescope::OwningRc<CVulkanTexture> netGradSlices;
	gamescope::OwningRc<CVulkanTexture> netProfileReadback;
	bool bNetStatePending = false;
	uint32_t uNetTrainStep = 0;
	// True while the last recorded batch routed the warps/probe through the
	// refined fields; the accessor below keys off it. Batches are strictly
	// one-in-flight, so a single flag is race-free.
	bool bNetActive = false;
	// Dataset capture (only with GAMESCOPE_FRAMEGEN_RECORD): host-mapped
	// linear copies of the raw training tensors, written to disk one batch
	// later under the same completion gate as the stats readback.
	gamescope::OwningRc<CVulkanTexture> recLumaPrev;
	gamescope::OwningRc<CVulkanTexture> recLumaCur;
	gamescope::OwningRc<CVulkanTexture> recField;
	gamescope::OwningRc<CVulkanTexture> recFieldRev;
	bool bRecAllocTried = false;
	uint32_t width = 0;
	uint32_t height = 0;
	uint32_t lumaFormat = DRM_FORMAT_INVALID;
};
static FramegenMotionResources_t g_framegenMotion;
static constexpr uint32_t k_uFramegenMotionDownscale = 8;

// Full-colour held-out validation (Gap E2). In capture mode, endpoint frames
// A/C and one exact intermediate reference B are retained; B is hidden from the
// estimator, and three paired predictions of B from A/C are copied beside exact
// B into host-visible images. The default offset/span is the consecutive A/B/C
// triplet; configurable spacing covers the lower phases used by slow x4 input.
// Predictions are never queued for presentation.
static constexpr uint32_t k_uFramegenColorProbeCandidates = 3;
static constexpr float k_flFramegenColorProbeStrengths[ k_uFramegenColorProbeCandidates ] = { 0.0f, 0.5f, 1.0f };

using gamescope::framegen::FramegenColorProbeSweep;

struct FramegenColorProbeResources_t
{
	gamescope::OwningRc<CVulkanTexture> generatedReadback[ k_uFramegenColorProbeCandidates ];
	gamescope::OwningRc<CVulkanTexture> referenceReadback;
	gamescope::Rc<CVulkanTexture> anchor;
	gamescope::Rc<CVulkanTexture> reference;
	uint32_t width = 0;
	uint32_t height = 0;
	uint32_t drmFormat = DRM_FORMAT_INVALID;
	uint32_t bytesPerPixel = 0;
	EOTF eotf = EOTF_Gamma22;
	uint64_t nextRealId = 0;
	uint64_t anchorId = 0;
	uint64_t referenceId = 0;
	uint64_t anchorTimeNs = 0;
	uint64_t referenceTimeNs = 0;
	uint64_t lastRealTimeNs = 0;

	uint64_t pendingSeqNo = 0;
	uint64_t pendingAnchorId = 0;
	uint64_t pendingReferenceId = 0;
	uint64_t pendingEndpointId = 0;
	uint64_t pendingAnchorTimeNs = 0;
	uint64_t pendingReferenceTimeNs = 0;
	uint64_t pendingEndpointTimeNs = 0;
	float pendingPhase = 0.0f;
	FramegenColorProbeSweep pendingSweep = FramegenColorProbeSweep::Occlusion;
};
static FramegenColorProbeResources_t g_framegenColorProbe;
static uint32_t g_uFramegenColorRecordCount = 0;

// Framegen shader dispatch is capability-based first: Vulkan tells us whether
// float16 arithmetic is legal and whether R16F works as a sampled+storage image.
// The one exception is the LDS-vs-direct extrapolation shader, which turns on
// texture-cache effectiveness — a property no capability bit exposes — so it uses
// a narrow, benchmark-backed vendor check (see framegen_dispatch_for_format). The
// selection is computed once and cached, so it never costs a per-dispatch branch.
struct FramegenDispatch_t
{
	uint32_t drmFormat = DRM_FORMAT_INVALID;
	ShaderType extrapolate = SHADER_TYPE_FRAMEGEN_EXTRAPOLATE;
	ShaderType extrapolatePair = SHADER_TYPE_FRAMEGEN_EXTRAPOLATE_PAIR;
	ShaderType motionLumaPair = SHADER_TYPE_FRAMEGEN_MOTION_LUMA_PAIR_RGBA;
	ShaderType motionPyramidPair = SHADER_TYPE_FRAMEGEN_MOTION_PYRAMID_RGBA;
	uint32_t motionLumaFormat = DRM_FORMAT_ABGR16161616F;
	bool useFp16 = false;
	bool useR16FLuma = false;
	bool motionSupported = false;
};
static FramegenDispatch_t g_framegenDispatch;

// Inter-frame change (gamma-encoded [0,1]) over which forward extrapolation is
// faded out. Below k_flFramegenSuppressLo we trust the full predicted step;
// above k_flFramegenSuppressHi we fall back to the current frame to avoid ghosting.
// The shader also rectifies the prediction against the local neighborhood of the
// current frame, which bounds any remaining overshoot, so this window can be a
// little wider than pure fade-out would allow.
static constexpr float k_flFramegenSuppressLo = 0.08f;
static constexpr float k_flFramegenSuppressHi = 0.40f;

struct FramegenPresentState_t
{
	const CVulkanTexture *pLastBaseTexture = nullptr;
	// steamcompmgr's commit id for that base layer. Authoritative for content
	// identity; the pointer above only covers compositor-owned layers that
	// carry no commit id.
	uint64_t ulLastBaseCommitID = 0;
	uint64_t ulLastPresentToken = 0;
	uint64_t ulLastSlotId = 0;
	uint64_t ulPreviousRealFrameId = 0;
	uint64_t ulCurrentRealFrameId = 0;
	uint64_t ulPreviousRealCompositeSeqNo = 0;
	uint64_t ulCurrentRealCompositeSeqNo = 0;
	gamescope::framegen::DisplayChainTimingState_t displayTiming;
	// acquireReadyTimeNs is optional per frame. Once one sample is absent or
	// invalid, keep this display-chain session on the always-available composite
	// clock instead of changing clock provenance on every subsequent sample.
	bool bSourceTimestampFallbackLatched = false;
	// A >250 ms content stall preserves the display-chain learner, but the first
	// correlated flip after it is discard-only evidence for the bias EMA.
	bool bPresentBiasHitchEpisode = false;
	bool bLoggedMinimumViableLeadActive = false;
	gamescope::FramegenPresentTag_t pendingTag = {};
	bool bTagPending = false;
};
static FramegenPresentState_t g_framegenPresentState;

static uint64_t framegen_commit_lead_override_ns();
static bool framegen_causal_deadline_enabled();
static gamescope::framegen::FixedRefreshPresentLead_t
framegen_native_present_lead( uint64_t ulPresentMarginNs,
	uint64_t ulVblankIntervalNs );

struct DisplayFeedback_t
{
	gamescope::FramegenPresentTag_t tag = {};
	uint64_t ulActualFlipNs = 0;
	uint64_t ulBackendSequence = 0;
	bool bPresented = false;
	bool bTimestampValid = false;
};

// Backend callbacks are producers and the compositor thread is the sole
// consumer. The fixed ring keeps the DRM page-flip hot path allocation-free;
// the mutex only protects mailbox indices and POD copies, never framegen state.
static constexpr size_t k_nDisplayFeedbackMailboxCapacity = 256;
struct DisplayFeedbackMailbox_t
{
	std::mutex mutex;
	std::array<DisplayFeedback_t, k_nDisplayFeedbackMailboxCapacity> records;
	size_t nRead = 0;
	size_t nCount = 0;
};
static DisplayFeedbackMailbox_t g_displayFeedbackMailbox;

static constexpr uint64_t k_ulFramegenMetricsWindowNs = 5'000'000'000ull;

// 250 us x 64 = 16 ms full scale: sub-vblank timing accuracy (deadline error,
// commit lead), which cannot plausibly run past the range without the sample
// already being rejected upstream.
using FramegenMetricsDistribution_t = gamescope::framegen::MetricsDistribution;
// 250 us x 256 = 64 ms full scale, same resolution: wall-clock presentation
// latency. A 60 Hz flip interval is 16.67 ms on its own, so the narrow
// distribution reported exactly 16.00 for these and hid the real tail.
using FramegenMetricsLatencyDistribution_t = gamescope::framegen::MetricsLatencyDistribution;

struct FramegenMetricsWindow_t
{
	uint64_t real = 0, generated = 0, delayedReal = 0, repeats = 0;
	uint64_t discards = 0, slowDrops = 0, admissionSkips = 0, resets = 0;
	uint64_t resetsCut = 0, resetsGrid = 0, resetsProvenance = 0, resetsHitch = 0;
	uint64_t resetsRing = 0, resetsChain = 0;
	// Feedback that arrived as "discarded" (window hidden/occluded at the host):
	// nonzero here with near-zero presented counts means the measurement
	// environment is invalid, not that the compositor idled.
	uint64_t feedbackDiscarded = 0;
	uint64_t deadlineCount = 0, deadlineHits = 0;
	uint32_t generatedLate = 0;
	// Real commits that became ready while an early generated commit was still
	// in flight on native KMS. Such a real frame cannot replace the reserved
	// atomic state and lands one vblank later than it would have without the
	// generated commit. Nonzero here means the early-commit lead is too long.
	uint32_t realWaitDuringGenerated = 0;
	double deadlineSignedSumMs = 0.0;
	// Dual-GPU staging traffic: bytes copied from the client's imported (non
	// device-local) base buffer into a device-local staging image, counted where
	// the copy is recorded. This is the throughput cost of the cross-GPU route.
	uint64_t copyBytes = 0;
	// Measured GPU time of those staging copies, from timestamp pairs around the
	// dedicated staging submission. Sum/count rather than a running average so a
	// window's copy_ms_avg is the true mean of the copies that landed in it; all
	// three stay 0 when the composite family cannot timestamp.
	uint64_t copyGpuNsSum = 0;
	uint64_t copyGpuSamples = 0;
	uint64_t copyGpuNsMax = 0;
	FramegenMetricsLatencyDistribution_t flipIntervals;
	FramegenMetricsDistribution_t deadlineErrors;
	FramegenMetricsDistribution_t realCommitLeads;
	// Source-ready -> actual-flip latency of REAL presents: how long a client
	// frame waited between being acquirable and actually reaching the screen.
	FramegenMetricsLatencyDistribution_t realSourceToFlip;
};

struct FramegenMetricsPendingEvents_t
{
	uint64_t repeats = 0, discards = 0, slowDrops = 0, admissionSkips = 0, resets = 0;
	uint64_t resetsCut = 0, resetsGrid = 0, resetsProvenance = 0, resetsHitch = 0;
	uint64_t resetsRing = 0, resetsChain = 0;
	uint64_t realWaits = 0;
	uint64_t copyBytes = 0;
	uint64_t copyGpuNsSum = 0;
	uint64_t copyGpuSamples = 0;
	uint64_t copyGpuNsMax = 0;
};
static FramegenMetricsPendingEvents_t g_framegenMetricsPendingEvents;

struct FramegenMetricsState_t
{
	std::array<FramegenMetricsWindow_t, 12> windows;
	FramegenMetricsWindow_t current;
	FramegenMetricsWindow_t total;
	uint64_t ulNextWindowNs = 0;
	uint64_t ulLastFlipNs = 0;
	uint64_t nClosedWindows = 0;
	size_t nNextWindow = 0;
};
static FramegenMetricsState_t g_framegenMetrics;

static void framegen_metrics_shutdown();

static bool framegen_metrics_enabled()
{
	static const bool s_bEnabled = []()
	{
		const bool bEnabled = env_to_bool( getenv( "GAMESCOPE_FRAMEGEN_METRICS" ) );
		if ( bEnabled )
			atexit( framegen_metrics_shutdown );
		return bEnabled;
	}();
	return s_bEnabled;
}

static void framegen_metrics_add_events( FramegenMetricsWindow_t &window,
	const FramegenMetricsPendingEvents_t &events )
{
	window.repeats += events.repeats;
	window.discards += events.discards;
	window.slowDrops += events.slowDrops;
	window.admissionSkips += events.admissionSkips;
	window.resets += events.resets;
	window.resetsCut += events.resetsCut;
	window.resetsGrid += events.resetsGrid;
	window.resetsProvenance += events.resetsProvenance;
	window.resetsHitch += events.resetsHitch;
	window.resetsRing += events.resetsRing;
	window.resetsChain += events.resetsChain;
	if ( events.realWaits != 0 )
	{
		// events.realWaits is 64-bit: compute the sum in 64-bit and saturate,
		// so a count past UINT32_MAX cannot wrap the subtraction and silently
		// stop accumulating.
		const uint64_t ulRealWaits = (uint64_t)window.realWaitDuringGenerated + events.realWaits;
		window.realWaitDuringGenerated = ulRealWaits > UINT32_MAX
			? UINT32_MAX : (uint32_t)ulRealWaits;
	}
	window.copyBytes += events.copyBytes;
	window.copyGpuNsSum += events.copyGpuNsSum;
	window.copyGpuSamples += events.copyGpuSamples;
	window.copyGpuNsMax = std::max( window.copyGpuNsMax, events.copyGpuNsMax );
}

static void framegen_metrics_flush_events()
{
	// Harvest whatever staging-copy timestamps have completed since the last
	// flush. Non-blocking by construction (see drainStagingCopyTiming), and the
	// results join the same pending-event batch as copy_bytes so both land in the
	// same window.
	uint64_t ulCopySumNs = 0, ulCopySamples = 0, ulCopyMaxNs = 0;
	if ( g_device.drainStagingCopyTiming( ulCopySumNs, ulCopySamples, ulCopyMaxNs ) )
	{
		g_framegenMetricsPendingEvents.copyGpuNsSum += ulCopySumNs;
		g_framegenMetricsPendingEvents.copyGpuSamples += ulCopySamples;
		g_framegenMetricsPendingEvents.copyGpuNsMax =
			std::max( g_framegenMetricsPendingEvents.copyGpuNsMax, ulCopyMaxNs );
	}

	framegen_metrics_add_events( g_framegenMetrics.current, g_framegenMetricsPendingEvents );
	framegen_metrics_add_events( g_framegenMetrics.total, g_framegenMetricsPendingEvents );
	g_framegenMetricsPendingEvents = {};
}

// Source-ready timestamps of in-flight real frames, keyed by the present tag's
// real frame id. The tag itself is a backend ABI struct, so the timestamp is
// carried alongside it here instead. Written from the compositor thread when a
// real present begins and read when its flip feedback is drained on the same
// thread; the ring is sized well past the display feedback mailbox depth so a
// late flip still finds its entry.
static constexpr size_t k_nFramegenSourceReadyRing = 64;
struct FramegenSourceReadyEntry_t
{
	uint64_t ulRealFrameId = 0;
	uint64_t ulSourceReadyNs = 0;
};
static std::array<FramegenSourceReadyEntry_t, k_nFramegenSourceReadyRing>
	g_framegenSourceReadyRing;

static void framegen_metrics_note_source_ready( uint64_t ulRealFrameId, uint64_t ulSourceReadyNs )
{
	if ( ulRealFrameId == 0 || ulSourceReadyNs == 0 )
		return;
	g_framegenSourceReadyRing[ ulRealFrameId % k_nFramegenSourceReadyRing ] =
		{ ulRealFrameId, ulSourceReadyNs };
}

static uint64_t framegen_metrics_find_source_ready( uint64_t ulRealFrameId )
{
	const FramegenSourceReadyEntry_t &entry =
		g_framegenSourceReadyRing[ ulRealFrameId % k_nFramegenSourceReadyRing ];
	return entry.ulRealFrameId == ulRealFrameId ? entry.ulSourceReadyNs : 0u;
}

// Share of presents that carried fresh content rather than repeating the
// previous scanout. Same definition as the HUD's "fill NN%".
static double framegen_metrics_fill_rate( const FramegenMetricsWindow_t &window )
{
	const uint64_t ulFresh = window.real + window.delayedReal + window.generated;
	const uint64_t ulPresented = ulFresh + window.repeats;
	return ulPresented != 0 ? (double)ulFresh / (double)ulPresented : 0.0;
}

static void framegen_metrics_log( const char *pszLabel,
	const FramegenMetricsWindow_t &window )
{
	const FramegenMetricsLatencyDistribution_t &flip = window.flipIntervals;
	const FramegenMetricsDistribution_t &deadline = window.deadlineErrors;
	const FramegenMetricsDistribution_t &lead = window.realCommitLeads;
	const FramegenMetricsLatencyDistribution_t &srcFlip = window.realSourceToFlip;
	const uint64_t ulVblankIntervalNs =
		g_framegenPresentState.displayTiming.key.intervalNs;
	const gamescope::framegen::FixedRefreshPresentLead_t viableLead =
		framegen_native_present_lead(
			ulVblankIntervalNs / 10u, ulVblankIntervalNs );
	vk_log.infof( "framegen-metrics: %s real=%" PRIu64 " gen=%" PRIu64
		" dreal=%" PRIu64 " rep=%" PRIu64
		" flip_ms_avg=%.3f flip_ms_min=%.3f flip_ms_max=%.3f"
		" flip_ms_sd=%.3f flip_ms_p95=%.3f bias_ms=%.3f dl_hit=%.3f"
		" dl_ms_avg=%.3f dl_ms_p95=%.3f dl_ms_worst=%.3f"
		" disc=%" PRIu64 " slow=%" PRIu64 " adm=%" PRIu64
		" resets=%" PRIu64 " fbdisc=%" PRIu64
		" resets_cut=%" PRIu64 " resets_grid=%" PRIu64
		" resets_prov=%" PRIu64 " resets_hitch=%" PRIu64
		" resets_ring=%" PRIu64 " resets_chain=%" PRIu64
		" lead_ms_min=%.3f lead_ms_avg=%.3f lead_ms_max=%.3f"
		" lead_ema=%.3f gen_late=%u lead_viable=%.3f real_wait=%u"
		// Appended fields. New fields go at the END of this line so existing
		// log parsers keep working; never reorder the ones above.
		" fill=%.3f copy_bytes=%" PRIu64 " copy_ms_avg=%.3f copy_ms_max=%.3f"
		" src_flip_ms_p50=%.3f src_flip_ms_p95=%.3f",
		pszLabel, window.real, window.generated, window.delayedReal, window.repeats,
		flip.average(), flip.n != 0 ? flip.min : 0.0, flip.max,
		flip.stddev(), flip.p95(),
		g_framegenPresentState.displayTiming.presentBias.emaNs / 1.0e6,
		window.deadlineCount != 0
			? (double)window.deadlineHits / window.deadlineCount : 0.0,
		window.deadlineCount != 0 ? window.deadlineSignedSumMs / window.deadlineCount : 0.0,
		deadline.p95(), deadline.max, window.discards, window.slowDrops,
		window.admissionSkips, window.resets,
		window.feedbackDiscarded, window.resetsCut, window.resetsGrid,
		window.resetsProvenance, window.resetsHitch, window.resetsRing,
		window.resetsChain,
		lead.n != 0 ? lead.min : 0.0, lead.average(), lead.max,
		g_framegenPresentState.displayTiming.presentLead.emaNs / 1.0e6,
		window.generatedLate,
		viableLead.ready ? viableLead.leadNs / 1.0e6 : 0.0,
		window.realWaitDuringGenerated,
		framegen_metrics_fill_rate( window ), window.copyBytes,
		// Real GPU time of the staging copies, from the timestamp pair around
		// their dedicated submission. Still exactly 0.0 when nothing was measured
		// (single-GPU route, or a composite family without timestamp support):
		// copy_ms_* remains "not measured", never an estimate.
		window.copyGpuSamples != 0
			? (double)window.copyGpuNsSum / (double)window.copyGpuSamples / 1.0e6 : 0.0,
		(double)window.copyGpuNsMax / 1.0e6,
		srcFlip.p50(), srcFlip.p95() );
}

static void framegen_metrics_close_windows( uint64_t ulNowNs, bool bLog )
{
	while ( ulNowNs >= g_framegenMetrics.ulNextWindowNs )
	{
		g_framegenMetrics.windows[ g_framegenMetrics.nNextWindow ] =
			g_framegenMetrics.current;
		g_framegenMetrics.nNextWindow = ( g_framegenMetrics.nNextWindow + 1 )
			% g_framegenMetrics.windows.size();
		g_framegenMetrics.nClosedWindows++;
		if ( bLog )
		{
			char szWindow[32];
			snprintf( szWindow, sizeof( szWindow ), "w=%" PRIu64,
				g_framegenMetrics.nClosedWindows );
			framegen_metrics_log( szWindow, g_framegenMetrics.current );
		}
		g_framegenMetrics.current = {};
		g_framegenMetrics.ulNextWindowNs += k_ulFramegenMetricsWindowNs;
	}
}

static void framegen_metrics_add_feedback( const DisplayFeedback_t &feedback )
{
	if ( !feedback.bPresented )
	{
		g_framegenMetrics.current.feedbackDiscarded++;
		g_framegenMetrics.total.feedbackDiscarded++;
		return;
	}
	auto addKind = [&]( FramegenMetricsWindow_t &window )
	{
		switch ( feedback.tag.eKind )
		{
			case gamescope::FramegenPresentKind_t::Real: window.real++; break;
			case gamescope::FramegenPresentKind_t::Generated: window.generated++; break;
			case gamescope::FramegenPresentKind_t::DelayedReal: window.delayedReal++; break;
		}
	};
	addKind( g_framegenMetrics.current );
	addKind( g_framegenMetrics.total );
	if ( !feedback.bTimestampValid )
		return;
	if ( feedback.tag.eKind == gamescope::FramegenPresentKind_t::Real )
	{
		// Latency of the real frame the client actually produced: from the
		// moment its buffer was acquirable to the moment it lit up the display.
		const uint64_t ulSourceReadyNs =
			framegen_metrics_find_source_ready( feedback.tag.ulRealFrameId );
		if ( ulSourceReadyNs != 0 && feedback.ulActualFlipNs > ulSourceReadyNs )
		{
			const uint64_t ulLatencyNs = feedback.ulActualFlipNs - ulSourceReadyNs;
			g_framegenMetrics.current.realSourceToFlip.add( ulLatencyNs );
			g_framegenMetrics.total.realSourceToFlip.add( ulLatencyNs );
		}
		gamescope::IBackend *pBackend = GetBackend();
		if ( pBackend != nullptr && pBackend->OwnsKMSPresentTiming() )
		{
			const gamescope::framegen::PresentLeadSample_t lead =
				gamescope::framegen::present_lead_sample(
					g_framegenPresentState.displayTiming.presentLead,
					feedback.tag.ulCommitSubmitNs,
					feedback.ulActualFlipNs,
					g_framegenPresentState.displayTiming.key.intervalNs,
					g_framegenPresentState.bPresentBiasHitchEpisode );
			if ( lead.valid )
			{
				g_framegenMetrics.current.realCommitLeads.add( lead.leadNs );
				g_framegenMetrics.total.realCommitLeads.add( lead.leadNs );
			}
		}
	}
	if ( g_framegenMetrics.ulLastFlipNs != 0
		&& feedback.ulActualFlipNs > g_framegenMetrics.ulLastFlipNs )
	{
		const uint64_t ulIntervalNs = feedback.ulActualFlipNs
			- g_framegenMetrics.ulLastFlipNs;
		g_framegenMetrics.current.flipIntervals.add( ulIntervalNs );
		g_framegenMetrics.total.flipIntervals.add( ulIntervalNs );
	}
	g_framegenMetrics.ulLastFlipNs = feedback.ulActualFlipNs;
	if ( feedback.tag.eKind == gamescope::FramegenPresentKind_t::Real
		|| feedback.tag.ulTargetFlipNs == 0 )
		return;
	const int nRefreshMhz = GetVBlankTimer().GetRefresh();
	const uint64_t ulVblankNs = nRefreshMhz > 0
		? 1'000'000'000'000ull / (uint64_t)nRefreshMhz : 8'333'333ull;
	const gamescope::framegen::DeadlineFeedbackSample_t sample =
		gamescope::framegen::deadline_feedback_sample(
			feedback.ulActualFlipNs, feedback.tag.ulTargetFlipNs,
			ulVblankNs );
	if ( !sample.valid )
		return;
	auto addDeadline = [&]( FramegenMetricsWindow_t &window )
	{
		window.deadlineCount++;
		window.deadlineHits += sample.hit;
		window.deadlineSignedSumMs += sample.signedErrorNs / 1.0e6;
		window.deadlineErrors.add( sample.absoluteErrorNs );
	};
	addDeadline( g_framegenMetrics.current );
	addDeadline( g_framegenMetrics.total );
	if ( feedback.tag.eKind == gamescope::FramegenPresentKind_t::Generated
		&& sample.signedErrorNs > static_cast<int64_t>( ulVblankNs / 2u ) )
	{
		if ( g_framegenMetrics.current.generatedLate != UINT32_MAX )
			g_framegenMetrics.current.generatedLate++;
		if ( g_framegenMetrics.total.generatedLate != UINT32_MAX )
			g_framegenMetrics.total.generatedLate++;
	}
}

static void framegen_metrics_shutdown()
{
	framegen_metrics_flush_events();
	char szTotal[32];
	snprintf( szTotal, sizeof( szTotal ), "TOTAL w=%" PRIu64,
		g_framegenMetrics.nClosedWindows );
	framegen_metrics_log( szTotal, g_framegenMetrics.total );
}

void vulkan_framegen_metrics_note_repeat()
{
	g_framegenMetricsPendingEvents.repeats++;
}

void vulkan_framegen_metrics_note_real_wait()
{
	g_framegenMetricsPendingEvents.realWaits++;
}

static void framegen_metrics_note_discard( uint64_t n ) { g_framegenMetricsPendingEvents.discards += n; }
static void framegen_metrics_note_slow_drop( uint64_t n ) { g_framegenMetricsPendingEvents.slowDrops += n; }
static void framegen_metrics_note_admission_skip() { g_framegenMetricsPendingEvents.admissionSkips++; }
enum class FramegenResetReason_t : uint8_t
{
	Cut,
	Grid,
	Provenance,
	Hitch,
	Ring,
};

static void framegen_metrics_note_reset( FramegenResetReason_t reason )
{
	g_framegenMetricsPendingEvents.resets++;
	switch ( reason )
	{
		case FramegenResetReason_t::Cut: g_framegenMetricsPendingEvents.resetsCut++; break;
		case FramegenResetReason_t::Grid: g_framegenMetricsPendingEvents.resetsGrid++; break;
		case FramegenResetReason_t::Provenance: g_framegenMetricsPendingEvents.resetsProvenance++; break;
		case FramegenResetReason_t::Hitch: g_framegenMetricsPendingEvents.resetsHitch++; break;
		case FramegenResetReason_t::Ring: g_framegenMetricsPendingEvents.resetsRing++; break;
	}
}

static void framegen_metrics_note_chain_reset( FramegenResetReason_t reason )
{
	framegen_metrics_note_reset( reason );
	g_framegenMetricsPendingEvents.resetsChain++;
}

static void framegen_apply_live_flip_feedback( const DisplayFeedback_t &feedback );
static void framegen_apply_bidir_flip_feedback( const DisplayFeedback_t &feedback );
static void framegen_apply_vrr_flip_feedback( const DisplayFeedback_t &feedback );

struct FramegenShadowDecision_t
{
	gamescope::framegen::CausalSlotPlan_t plan;
	uint64_t ulRealFrameId = 0;
	uint32_t nClassicGap = 0;
	uint32_t nClassicSlot = 0;
	bool bDiscardedByFeedback = false;
};

// Step 2 is deliberately one-way: this state reads live measurements, but no
// live pacing, submission, or presentation path reads it. Keeping the ring
// fixed-size also makes the shadow path allocation-free.
static constexpr size_t k_nFramegenShadowDecisionCapacity = 8;
struct FramegenDeadlineShadowState_t
{
	gamescope::framegen::RealAnchorState_t anchor;
	gamescope::framegen::PresentBiasState_t presentBias;
	std::array<FramegenShadowDecision_t, k_nFramegenShadowDecisionCapacity> decisions;
	size_t nNextDecision = 0;
	size_t nDecisionCount = 0;
	uint64_t ulGridEpoch = 0;
	uint64_t ulGridIntervalNs = 0;
	uint64_t ulDisplayChainGeneration = 0;
	bool bSourceProvenanceInitialized = false;
	bool bSourceTimestampsReliable = false;
};
static FramegenDeadlineShadowState_t g_framegenDeadlineShadow;

static void framegen_invalidate_deadline_shadow_content()
{
	const gamescope::framegen::PresentBiasState_t presentBias =
		g_framegenDeadlineShadow.presentBias;
	const uint64_t ulDisplayChainGeneration =
		g_framegenDeadlineShadow.ulDisplayChainGeneration;
	g_framegenDeadlineShadow = {};
	g_framegenDeadlineShadow.presentBias = presentBias;
	g_framegenDeadlineShadow.ulDisplayChainGeneration =
		ulDisplayChainGeneration;
}

static bool framegen_observe_display_chain( uint64_t ulIntervalNs,
	bool bSourceTimestampValid )
{
	gamescope::IBackend *pBackend = GetBackend();
	gamescope::IBackendConnector *pConnector = pBackend != nullptr
		? pBackend->GetCurrentConnector() : nullptr;
	const gamescope::framegen::DisplayChainKey_t physicalKey = {
		.backendId = static_cast<uint64_t>( reinterpret_cast<uintptr_t>( pBackend ) ),
		.connectorId = pConnector != nullptr ? pConnector->GetConnectorID() : 0u,
		.intervalNs = ulIntervalNs,
		.vrrActive = pConnector != nullptr && pConnector->IsVRRActive(),
	};
	const gamescope::framegen::DisplayChainKey_t oldKey =
		g_framegenPresentState.displayTiming.key;
	const bool bPhysicalChainChanged =
		g_framegenPresentState.displayTiming.initialized
		&& ( oldKey.backendId != physicalKey.backendId
			|| oldKey.connectorId != physicalKey.connectorId
			|| oldKey.intervalNs != physicalKey.intervalNs
			|| oldKey.vrrActive != physicalKey.vrrActive );
	if ( !g_framegenPresentState.displayTiming.initialized || bPhysicalChainChanged )
		g_framegenPresentState.bSourceTimestampFallbackLatched = !bSourceTimestampValid;
	else if ( !bSourceTimestampValid )
		g_framegenPresentState.bSourceTimestampFallbackLatched = true;

	const bool bSourceTimestampsReliable =
		!g_framegenPresentState.bSourceTimestampFallbackLatched;
	gamescope::framegen::DisplayChainKey_t key = physicalKey;
	key.sourceTimestampsReliable = bSourceTimestampsReliable;
	const gamescope::framegen::DisplayChainTimingTransition_t transition =
		gamescope::framegen::observe_display_chain(
			g_framegenPresentState.displayTiming, key );
	g_framegenPresentState.displayTiming = transition.state;
	if ( !transition.displayChainChanged )
		return bSourceTimestampsReliable;
	g_framegenPresentState.bLoggedMinimumViableLeadActive = false;

	std::string changedFields;
	const auto noteChanged = [&]( bool changed, const char *field )
	{
		if ( !changed )
			return;
		if ( !changedFields.empty() )
			changedFields += ',';
		changedFields += field;
	};
	noteChanged( oldKey.backendId != key.backendId, "backend" );
	noteChanged( oldKey.connectorId != key.connectorId, "connector" );
	noteChanged( oldKey.intervalNs != key.intervalNs, "interval" );
	noteChanged( oldKey.vrrActive != key.vrrActive, "vrr" );
	noteChanged( oldKey.sourceTimestampsReliable
		!= key.sourceTimestampsReliable, "source_ts" );
	vk_log.infof(
		"framegen: display-chain change fields=%s backend=%" PRIu64 "->%" PRIu64
		" connector=%" PRIu64 "->%" PRIu64 " interval_ns=%" PRIu64 "->%" PRIu64
		" vrr=%u->%u source_ts=%u->%u",
		changedFields.c_str(), oldKey.backendId, key.backendId,
		oldKey.connectorId, key.connectorId, oldKey.intervalNs, key.intervalNs,
		static_cast<unsigned>( oldKey.vrrActive ),
		static_cast<unsigned>( key.vrrActive ),
		static_cast<unsigned>( oldKey.sourceTimestampsReliable ),
		static_cast<unsigned>( key.sourceTimestampsReliable ) );

	// Feedback already queued by the old connector/provenance must not warm the
	// freshly reset learners. The compositor is the sole mailbox consumer.
	{
		std::scoped_lock lock( g_displayFeedbackMailbox.mutex );
		g_displayFeedbackMailbox.nRead = 0;
		g_displayFeedbackMailbox.nCount = 0;
	}
	framegen_metrics_note_chain_reset( bPhysicalChainChanged
		? FramegenResetReason_t::Grid
		: FramegenResetReason_t::Provenance );
	return bSourceTimestampsReliable;
}

static uint64_t framegen_next_present_slot_id()
{
	return ++g_framegenPresentState.ulLastSlotId;
}

static void framegen_select_present_tag( gamescope::FramegenPresentKind_t eKind,
	uint64_t ulRealFrameId, uint64_t ulSlotId, uint64_t ulCompositeSeqNo,
	uint64_t ulTargetFlipNs )
{
	if ( !g_framegenPresentState.bTagPending )
	{
		g_framegenPresentState.pendingTag = {
			.ulPresentToken = ++g_framegenPresentState.ulLastPresentToken,
			.ulRealFrameId = g_framegenPresentState.ulCurrentRealFrameId,
			.ulSlotId = 0,
			.ulCompositeSeqNo = 0,
			.ulTargetFlipNs = GetVBlankTimer().GetNextVBlank( 0 ),
			.ulCommitSubmitNs = 0,
			.eKind = gamescope::FramegenPresentKind_t::Real,
		};
		g_framegenPresentState.bTagPending = true;
	}

	gamescope::FramegenPresentTag_t &tag = g_framegenPresentState.pendingTag;
	tag.eKind = eKind;
	tag.ulRealFrameId = ulRealFrameId;
	tag.ulSlotId = ulSlotId;
	tag.ulCompositeSeqNo = ulCompositeSeqNo;
	tag.ulTargetFlipNs = ulTargetFlipNs != 0
		? ulTargetFlipNs : GetVBlankTimer().GetNextVBlank( 0 );
	tag.ulCommitSubmitNs = 0;
}

void vulkan_framegen_begin_present( const struct FrameInfo_t *pFrameInfo )
{
	const CVulkanTexture *pBaseTexture = pFrameInfo != nullptr && pFrameInfo->layerCount > 0
		? pFrameInfo->layers[ 0 ].tex.get() : nullptr;
	const gamescope::framegen::RealFrameIdentity_t identity = {
		.commitId = pFrameInfo != nullptr && pFrameInfo->layerCount > 0
			? pFrameInfo->layers[ 0 ].ulCommitID : 0u,
		.pTexture = pBaseTexture,
	};
	if ( g_framegenPresentState.ulCurrentRealFrameId == 0
		|| gamescope::framegen::is_new_real_frame_content(
			{ g_framegenPresentState.ulLastBaseCommitID,
				g_framegenPresentState.pLastBaseTexture }, identity ) )
	{
		g_framegenPresentState.ulPreviousRealFrameId = g_framegenPresentState.ulCurrentRealFrameId;
		g_framegenPresentState.ulPreviousRealCompositeSeqNo =
			g_framegenPresentState.ulCurrentRealCompositeSeqNo;
		g_framegenPresentState.ulCurrentRealFrameId++;
		g_framegenPresentState.ulCurrentRealCompositeSeqNo = 0;
	}
	if ( pBaseTexture != nullptr )
	{
		g_framegenPresentState.pLastBaseTexture = pBaseTexture;
		g_framegenPresentState.ulLastBaseCommitID = identity.commitId;
	}

	if ( framegen_metrics_enabled() && pFrameInfo != nullptr && pFrameInfo->layerCount > 0 )
	{
		framegen_metrics_note_source_ready( g_framegenPresentState.ulCurrentRealFrameId,
			pFrameInfo->layers[ 0 ].acquireReadyTimeNs );
	}

	g_framegenPresentState.pendingTag = {
		.ulPresentToken = ++g_framegenPresentState.ulLastPresentToken,
		.ulRealFrameId = g_framegenPresentState.ulCurrentRealFrameId,
		.ulSlotId = 0,
		.ulCompositeSeqNo = 0,
		.ulTargetFlipNs = GetVBlankTimer().GetNextVBlank( 0 ),
		.ulCommitSubmitNs = 0,
		.eKind = gamescope::FramegenPresentKind_t::Real,
	};
	g_framegenPresentState.bTagPending = true;
}

void vulkan_framegen_note_present_composite_seqno( uint64_t ulCompositeSeqNo )
{
	if ( g_framegenPresentState.bTagPending
		&& g_framegenPresentState.pendingTag.eKind == gamescope::FramegenPresentKind_t::Real )
	{
		g_framegenPresentState.pendingTag.ulCompositeSeqNo = ulCompositeSeqNo;
		g_framegenPresentState.ulCurrentRealCompositeSeqNo = ulCompositeSeqNo;
	}
}

gamescope::FramegenPresentTag_t vulkan_framegen_take_present_tag()
{
	if ( !g_framegenPresentState.bTagPending )
		framegen_select_present_tag( gamescope::FramegenPresentKind_t::Real,
			g_framegenPresentState.ulCurrentRealFrameId, 0, 0, 0 );

	g_framegenPresentState.bTagPending = false;
	return g_framegenPresentState.pendingTag;
}

void vulkan_framegen_publish_present_feedback( const gamescope::FramegenPresentTag_t &tag,
	uint64_t ulActualFlipNs, uint64_t ulBackendSequence, bool bPresented, bool bTimestampValid )
{
	std::scoped_lock lock( g_displayFeedbackMailbox.mutex );
	if ( g_displayFeedbackMailbox.nCount == k_nDisplayFeedbackMailboxCapacity )
	{
		g_displayFeedbackMailbox.nRead =
			( g_displayFeedbackMailbox.nRead + 1 ) % k_nDisplayFeedbackMailboxCapacity;
		g_displayFeedbackMailbox.nCount--;
	}

	const size_t nWrite = ( g_displayFeedbackMailbox.nRead + g_displayFeedbackMailbox.nCount )
		% k_nDisplayFeedbackMailboxCapacity;
	g_displayFeedbackMailbox.records[ nWrite ] = {
		.tag = tag,
		.ulActualFlipNs = ulActualFlipNs,
		.ulBackendSequence = ulBackendSequence,
		.bPresented = bPresented,
		.bTimestampValid = bTimestampValid,
	};
	g_displayFeedbackMailbox.nCount++;
}

void vulkan_framegen_drain_present_feedback()
{
	const bool bMetricsLoggingEnabled = framegen_metrics_enabled();
	const bool bMetricsCollectionEnabled = bMetricsLoggingEnabled
		|| framegen_hud_level() != 0u;
	if ( bMetricsCollectionEnabled )
	{
		const uint64_t ulNowNs = get_time_in_nanos();
		if ( g_framegenMetrics.ulNextWindowNs == 0 )
			g_framegenMetrics.ulNextWindowNs = ulNowNs + k_ulFramegenMetricsWindowNs;
		framegen_metrics_close_windows( ulNowNs, bMetricsLoggingEnabled );
		framegen_metrics_flush_events();
	}
	std::array<DisplayFeedback_t, k_nDisplayFeedbackMailboxCapacity> records;
	size_t nCount = 0;
	{
		std::scoped_lock lock( g_displayFeedbackMailbox.mutex );
		nCount = g_displayFeedbackMailbox.nCount;
		for ( size_t i = 0; i < nCount; i++ )
		{
			records[ i ] = g_displayFeedbackMailbox.records[ g_displayFeedbackMailbox.nRead ];
			g_displayFeedbackMailbox.nRead =
				( g_displayFeedbackMailbox.nRead + 1 ) % k_nDisplayFeedbackMailboxCapacity;
		}
		g_displayFeedbackMailbox.nCount = 0;
	}

	for ( size_t i = 0; i < nCount; i++ )
	{
		const DisplayFeedback_t &feedback = records[ i ];
		if ( feedback.tag.ulPresentToken == 0 )
			continue;
		if ( bMetricsCollectionEnabled )
			framegen_metrics_add_feedback( feedback );

		if ( feedback.bPresented && feedback.bTimestampValid )
		{
			gamescope::IBackend *pBackend = GetBackend();
			if ( pBackend != nullptr && pBackend->OwnsKMSPresentTiming() )
			{
				if ( feedback.tag.eKind == gamescope::FramegenPresentKind_t::Generated
					&& feedback.tag.ulTargetFlipNs != 0u
					&& feedback.tag.ulCommitSubmitNs != 0u )
				{
					const uint64_t ulVblankNs =
						g_framegenPresentState.displayTiming.key.intervalNs;
					const gamescope::framegen::DeadlineFeedbackSample_t sample =
						gamescope::framegen::deadline_feedback_sample(
							feedback.ulActualFlipNs,
							feedback.tag.ulTargetFlipNs, ulVblankNs );
					const bool bLate = sample.valid
						&& sample.signedErrorNs
							> static_cast<int64_t>( ulVblankNs / 2u );
					static uint64_t s_uGeneratedCommitFeedbackDebugLogCounter = 0;
					if ( sample.valid && FramegenDebugShouldLog(
						s_uGeneratedCommitFeedbackDebugLogCounter ) )
					{
						const double flLeadMs = static_cast<double>(
							gamescope::framegen::signed_ns_delta(
								feedback.ulActualFlipNs,
								feedback.tag.ulCommitSubmitNs ) ) / 1.0e6;
						vk_log.infof( "framegen: gen commit lead=%.2fms target_err=%+.2fms late=%d",
							flLeadMs, sample.signedErrorNs / 1.0e6,
							bLate ? 1 : 0 );
					}

					const uint64_t ulPresentMarginNs = ulVblankNs / 10u;
					const gamescope::framegen::FixedRefreshPresentLead_t before =
						framegen_native_present_lead(
							ulPresentMarginNs, ulVblankNs );
					if ( gamescope::framegen::minimum_viable_present_lead_needs_backoff(
						before, feedback.ulActualFlipNs,
						feedback.tag.ulTargetFlipNs, ulVblankNs )
						&& framegen_causal_deadline_enabled()
						&& !vulkan_framegen_vrr_hybrid_active()
						&& !vulkan_framegen_bidir_active() )
					{
						gamescope::framegen::DisplayChainTimingState_t &timing =
							g_framegenPresentState.displayTiming;
						timing.minimumViablePresentLead =
							gamescope::framegen::back_off_minimum_viable_present_lead(
								timing.minimumViablePresentLead,
								timing.presentLead, ulVblankNs );
						const gamescope::framegen::FixedRefreshPresentLead_t after =
							framegen_native_present_lead(
								ulPresentMarginNs, ulVblankNs );
						if ( after.leadNs > before.leadNs )
						{
							static uint64_t s_ulLastViableLeadBackoffLogNs = 0u;
							const uint64_t ulNowNs = get_time_in_nanos();
							if ( s_ulLastViableLeadBackoffLogNs == 0u
								|| ulNowNs - s_ulLastViableLeadBackoffLogNs
									>= 1'000'000'000u )
							{
								vk_log.infof(
									"framegen: minimum viable commit lead backed off %.3fms->%.3fms after late generated flip",
									before.leadNs / 1.0e6,
									after.leadNs / 1.0e6 );
								s_ulLastViableLeadBackoffLogNs = ulNowNs;
							}
						}
					}
				}

				// An early generated commit is scheduled from this estimate, so
				// feeding that controlled lead back into the learner would teach
				// it its own margin. Native KMS learns only from real commits and
				// rejects the first post-hitch/outlier observation.
				if ( feedback.tag.eKind != gamescope::FramegenPresentKind_t::Generated )
				{
					gamescope::framegen::DisplayChainTimingState_t &timing =
						g_framegenPresentState.displayTiming;
					const bool bWasActive = timing.minimumViablePresentLead.samples
						>= gamescope::framegen::k_uPresentLeadWarmupSamples;
					if ( feedback.tag.eKind == gamescope::FramegenPresentKind_t::Real )
					{
						timing.minimumViablePresentLead =
							gamescope::framegen::update_minimum_viable_present_lead(
								timing.minimumViablePresentLead,
								timing.presentLead,
								feedback.tag.ulCommitSubmitNs,
								feedback.ulActualFlipNs,
								timing.key.intervalNs,
								g_framegenPresentState.bPresentBiasHitchEpisode );
					}
					timing.presentLead =
						gamescope::framegen::update_present_lead(
							timing.presentLead,
							feedback.tag.ulCommitSubmitNs,
							feedback.ulActualFlipNs,
							timing.key.intervalNs,
							g_framegenPresentState.bPresentBiasHitchEpisode );
					const gamescope::framegen::FixedRefreshPresentLead_t viable =
						framegen_native_present_lead(
							timing.key.intervalNs / 10u,
							timing.key.intervalNs );
					if ( !bWasActive && viable.learned
						&& !g_framegenPresentState.bLoggedMinimumViableLeadActive )
					{
						vk_log.infof(
							"framegen: learned minimum viable commit lead active lead=%.3fms low=%.3fms habitual=%.3fms samples=%u",
							viable.leadNs / 1.0e6,
							timing.minimumViablePresentLead.lowWaterNs / 1.0e6,
							timing.presentLead.emaNs / 1.0e6,
							timing.minimumViablePresentLead.samples );
						g_framegenPresentState.bLoggedMinimumViableLeadActive = true;
					}
				}
			}
			else
			{
				// Preserve the established nested/host-compositor learner exactly.
				g_framegenPresentState.displayTiming.presentLead =
					gamescope::framegen::update_present_lead(
						g_framegenPresentState.displayTiming.presentLead,
						feedback.tag.ulCommitSubmitNs,
						feedback.ulActualFlipNs );
			}
		}

		if ( feedback.tag.eKind == gamescope::FramegenPresentKind_t::DelayedReal )
		{
			framegen_apply_bidir_flip_feedback( feedback );
			continue;
		}

		// A discarded real completion is still identity-bearing cancellation
		// evidence for VRR; no timestamp-based anchor correction is possible.
		if ( feedback.tag.eKind == gamescope::FramegenPresentKind_t::Real )
			framegen_apply_vrr_flip_feedback( feedback );
		if ( !feedback.bPresented || !feedback.bTimestampValid
			|| feedback.tag.eKind != gamescope::FramegenPresentKind_t::Real
			|| feedback.tag.ulTargetFlipNs == 0 )
			continue;

		static uint64_t s_uAnchorErrorDebugLogCounter = 0;
		if ( FramegenDebugShouldLog( s_uAnchorErrorDebugLogCounter ) )
		{
			const double flErrorMs = (double)( (long double)feedback.ulActualFlipNs
				- (long double)feedback.tag.ulTargetFlipNs ) / 1.0e6;
			vk_log.infof( "framegen: anchor error real=%" PRIu64 " err=%+.3fms",
				feedback.tag.ulRealFrameId, flErrorMs );
		}

		// Preserve the Step 2 shadow ring for A/B logging, then apply the same
		// correlated record to the live causal anchor below. Other policies apply
		// their typed feedback through the handlers above.
		const uint64_t ulArrivalGuardNs = std::max(
			gamescope::framegen::k_ulCadenceArrivalGuardMinNs,
			g_framegenDeadlineShadow.ulGridIntervalNs
				/ gamescope::framegen::k_uCadenceArrivalGuardDivisor );
		const gamescope::framegen::AnchorCorrection_t correction =
			gamescope::framegen::apply_flip_feedback(
				g_framegenDeadlineShadow.anchor,
				g_framegenDeadlineShadow.presentBias,
				feedback.tag.ulRealFrameId,
				feedback.ulActualFlipNs,
				ulArrivalGuardNs,
				g_framegenDeadlineShadow.ulGridIntervalNs,
				g_framegenPresentState.bPresentBiasHitchEpisode );
		if ( correction.matched )
		{
			g_framegenDeadlineShadow.anchor = correction.anchor;
			g_framegenDeadlineShadow.presentBias = correction.presentBias;
			if ( correction.discardProvisional )
			{
				for ( FramegenShadowDecision_t &decision : g_framegenDeadlineShadow.decisions )
				{
					if ( decision.ulRealFrameId == feedback.tag.ulRealFrameId
						&& decision.plan.provisional )
						decision.bDiscardedByFeedback = true;
				}
			}
		}

		framegen_apply_live_flip_feedback( feedback );
	}
}

struct FramegenHistory_t
{
	// The last two real frames, held as references straight into the
	// g_output.outputImages ring. Zero-copy: the composite already wrote these
	// images, so framegen never copies them — it keeps them alive and samples
	// them. Real-target selection checks actual texture/backend ownership, so
	// neither slot is recomposited while history or generation still reads it.
	// previousReal is the older of the two.
	gamescope::Rc<CVulkanTexture> previousReal;
	gamescope::Rc<CVulkanTexture> currentReal;

	// The two output-ring slots the most recent generation batch samples, kept
	// pinned (skipped by the ring advance) until that batch signals genReadSeqNo.
	// On the dedicated framegen queue a batch keeps reading its inputs after
	// history is logically invalidated (e.g. gpu_oversubscribed nulls
	// previousReal/currentReal); without this, a later composite on the realtime
	// queue could overwrite a slot the framegen queue is still reading — a
	// cross-queue write-after-read. We never make the composite wait (that would
	// delay a real frame); we just avoid reusing the slot until the read is done.
	gamescope::Rc<CVulkanTexture> genReadA;
	gamescope::Rc<CVulkanTexture> genReadB;
	gamescope::Rc<CVulkanTexture> genReadReference;
	uint64_t genReadSeqNo = 0;

	// Generated frames waiting for their empty vblanks, presented front-first,
	// one per vblank. Depth is multiplier-1 (x2 -> 1, x4 -> 3). Drained
	// wholesale the moment a real frame supersedes them — EXCEPT in bidir mode
	// (B3), where the queue IS the presentation timeline: interpolated slots
	// precede the real frame they lead up to (bReal entries), and a new real
	// frame appends behind them instead of superseding.
	struct PendingGenerated_t
	{
		gamescope::Rc<CVulkanTexture> tex;
		uint64_t seqNo = 0;
		uint64_t frameId = 0;
		uint64_t ulPresentRealFrameId = 0;
		uint64_t ulSlotId = 0;
		uint64_t ulCompositeSeqNo = 0;
		uint64_t ulTargetFlipNs = 0;
		uint64_t ulWakeDeadlineNs = 0;
		uint64_t ulAnchorRealFrameId = 0;
		float phase = 0.0f; // fraction of the real-frame interval, for logs
		bool bProvisional = false;
		// Bidir: this entry is a REAL frame riding the queue behind its
		// interpolations. Its composite completed at its own paint (seqNo 0 on
		// the framegen timeline = always ready) and it must never be dropped
		// by the generated-frame discard paths.
		bool bReal = false;
		// Output-space cursor split: both history endpoints this frame was
		// predicted from were cursor-free composites, so the live cursor still
		// has to be composited on top at present time. Carried per entry (not
		// read off the live history) so a frame planned before a fallback can
		// never end up with two cursors, or none.
		bool bCursorFree = false;
	};
	std::vector<PendingGenerated_t> pending;

	uint32_t width = 0;
	uint32_t height = 0;
	uint32_t drmFormat = DRM_FORMAT_INVALID;
	uint64_t previousFrameId = 0;
	uint64_t currentFrameId = 0;
	uint64_t previousPresentTimeNs = 0;
	uint64_t currentPresentTimeNs = 0;
	// Default display-clock pacing (#06). cadence learns source-buffer readiness
	// from acquire-fence completion times, before fixed-refresh quantization. Its
	// bounded trend predicts smooth rate changes and its one-sided late envelope
	// decides whether the next display slot needs a speculative backup. Backends
	// without a source timestamp fall back to composite time without mixing the
	// two clock bases in one model. ulCurrentRealVblankNs anchors the current real
	// frame on the display clock, so each generated slot is placed against the
	// exact vblank where it is intended to be shown.
	gamescope::framegen::CadencePredictorState cadence;
	uint64_t ulCurrentCadenceTimeNs = 0;
	bool bCadenceUsesSourceTime = false;
	uint64_t ulCurrentRealVblankNs = 0;
	// Step 3 causal fixed-refresh state. The provisional anchor is installed
	// when the real composite records; correlated flip feedback replaces it in
	// this same object and may invalidate pixels already generated from it.
	gamescope::framegen::RealAnchorState_t causalAnchor;
	uint64_t ulDeadlineGridEpoch = 0;
	uint64_t ulDeadlineGridIntervalNs = 0;
	uint64_t ulDeadlineDisplayChainGeneration = 0;
	uint64_t ulLastPlannedTargetNs = 0;
	bool bDeadlineProvenanceInitialized = false;
	bool bDeadlineSourceTimestampsReliable = false;
	bool bCausalDeadlineMissed = false;
	gamescope::framegen::DeadlineMissState_t causalDeadlineMisses;
	// Fixed-delay bidirectional timeline. Epoch feedback mutates only this
	// mapping; queued slot targets remain immutable values. Source records are
	// retained until their tagged delayed-real completion can correct the epoch.
	gamescope::framegen::BidirEpoch_t bidirEpoch;
	gamescope::framegen::BidirCutEpisodeState_t bidirCutEpisode;
	uint64_t ulBidirGridEpoch = 0;
	uint64_t ulBidirGridIntervalNs = 0;
	uint64_t ulBidirDisplayChainGeneration = 0;
	uint64_t ulBidirGridTargetNs = 0;
	uint64_t ulBidirGridWakeNs = 0;
	// Gap-limited pending depth the planner last admitted to (one real interval
	// of display slots plus its closing endpoint, capped by the ring ceiling).
	// The drain valve compares against this, not against the raw ceiling, so a
	// queue that is exactly as deep as planned still flips on its own targets.
	// 0 = no real interval measured yet in this epoch.
	size_t uBidirPendingTarget = 0;
	bool bBidirProvenanceInitialized = false;
	bool bBidirSourceTimestampsReliable = false;
	struct BidirFeedbackEndpoint_t
	{
		gamescope::framegen::BidirEndpoint_t endpoint;
		uint64_t ulEpoch = 0;
	};
	std::vector<BidirFeedbackEndpoint_t> bidirFeedbackEndpoints;
	gamescope::Rc<CVulkanTexture> bidirLastOutput;
	// A first exceptional ring reset retains the newest delayed endpoint for a
	// visible hold. If even that reset cannot free a composite target, the next
	// attempt releases the hold too, preventing a compositor-owned failure loop.
	uint32_t nBidirRingPressureFailures = 0;
	// VRR midpoint work is not submitted until the correlated Real feedback
	// establishes its non-grid target and lead-compensated wake deadline.
	uint64_t ulVrrAwaitingRealFrameId = 0;
	uint64_t ulVrrAwaitingCompositeSeqNo = 0;
	uint64_t ulVrrAwaitingCadenceNs = 0;
	uint64_t ulVrrPanelIntervalNs = 0;
	uint64_t ulVrrMidTargetNs = 0;
	uint64_t ulVrrMidWakeNs = 0;
	uint64_t lastCompositeSeqNo = 0;
	// Latest submission on the framegen execution path, including descriptor-free
	// base-history copies. Reset paths wait for this token before releasing pools
	// or history; lastGeneratedSeqNo remains generation-only for the headroom gate.
	uint64_t lastFramegenWorkSeqNo = 0;
	// Split-family path only: the base-history ingest copy runs on the COMPOSITE
	// queue (it reads the client's imported dma-buf), so its token lives on the
	// composite timeline and must be waited on separately.
	uint64_t lastBaseIngestSeqNo = 0;
	// Seq no (on the framegen timeline) of the most recent generation batch, for
	// the oversubscription guard: skip new generation while the previous batch
	// is still running rather than queue work in front of real frames.
	uint64_t lastGeneratedSeqNo = 0;
	// Slot planning relative to the current real frame. Real-frame submission
	// fills the first multiplier-1 slots; the dedicated-queue idle refill can
	// continue from this counter if the game stalls before the next real frame.
	uint32_t nLastGeneratedSlot = 0;
	uint32_t nLastGenerationGapVblanks = 0;
	// Rolling index into g_output.framegenOutputImages for the next generated
	// frame, advanced per generated frame independent of the real-frame nOutImage.
	uint32_t nNextOutputIndex = 0;
	// Identity of the base layer's texture at the last recorded frame. Only a
	// new base-layer commit counts as a real frame; overlay-only repaints
	// re-composite the same game content and must not disturb pacing.
	const CVulkanTexture *pLastBaseTexture = nullptr;
	uint64_t ulLastBaseCommitID = 0;
	// Base-layer mode (#02): history and generation run on the pre-upscale
	// game layer; previousReal/currentReal then point into baseHistory[] —
	// two internally-owned base-sized copies — instead of the output ring.
	// The client's buffer can't be retained zero-copy: its release back to
	// the game is keyed to commit lifetime, not to our texture refs, so a
	// held frame would be rewritten under our sampler. The copy is made on
	// the framegen queue the moment the frame records, while the commit is
	// still current.
	bool bBaseLayer = false;
	gamescope::OwningRc<CVulkanTexture> baseHistory[2];
	uint32_t nBaseHistoryNext = 0;
	// Output-space cursor split: whether previousReal / currentReal hold a
	// composite with NO cursor in it. True both when the split ran and when the
	// stack simply carried no cursor layer (hidden/grabbed pointer) - in both
	// cases prediction is cursor-free and the live cursor may be late-composited
	// onto generated frames. Shifted alongside previousReal/currentReal.
	bool bCursorFreePrevious = false;
	bool bCursorFreeCurrent = false;
	// Rolling index into g_output.framegenCursorHistoryImages.
	uint32_t nNextCursorHistoryIndex = 0;
	// Rolling index into g_output.framegenPresentImages for the next late
	// overlay composite target (base mode only).
	uint32_t nNextPresentIndex = 0;
	// Base-layer colorspace fingerprint: an SDR<->HDR flip of the game buffer
	// on an unchanged DRM format is a scene discontinuity for prediction.
	int nLastBaseColorspace = -1;
	// Scene fingerprint: prediction across a layer-count or output-encoding
	// change would smear the previous scene over the new one.
	int nLastLayerCount = -1;
	EOTF eLastEOTF = EOTF_Count;
	// Consecutive real frames slow enough to leave an empty vblank to fill.
	uint32_t nStableFrames = 0;
	// Deadline-driven degradation ladder (#04) position: 0 = full startup config,
	// each step sheds work (motion pipelines, then extrapolate, then a
	// multiplier notch). Never
	// reaches "stop generating" — that is left to the reactive pacing gate below.
	// nDegradeHold is a post-step cooldown so the new rung's cost folds into the
	// measurement before the next step decision. Recovery is scene-local, climbs
	// one adjacent rung only after sustained headroom, and exponentially backs off
	// when a probe falls back during probation.
	uint32_t nDegradeSteps = 0;
	uint32_t nDegradeHold = 0;
	gamescope::framegen::RecoveryState_t recovery;
	// Bidir (B3): set by framegen_record_real_frame when the composite that is
	// being presented right now queued its real frame behind interpolation
	// slots; consumed by vulkan_framegen_bidir_flip_texture to substitute the
	// pending front for the flip. Cleared at the top of every vulkan_composite
	// so a composite that never records (overlay-only, partial, screenshot)
	// can't inherit a stale verdict.
	bool bBidirQueuedReal = false;
	// Bidir (B3): this composite re-rendered the SAME game frame (overlay-only
	// repaint). While the queue is draining, its flip must show the queue front
	// — not the recomposite, whose (newest) game content is still queued and
	// would present out of order. The overlay update rides the next real frame.
	bool bBidirSameBaseComposite = false;
	// Self-supervised adaptation (B4): slow EMAs of the stats-probe readback
	// and the threshold values derived from them (applied to the NEXT batch's
	// push constants). -1 = no sample yet / default. The one-batch-in-flight
	// guarantee makes the mapped readback race-free: it is only parsed after
	// hasCompletedFramegen() admits the batch that wrote it.
	gamescope::framegen::AdaptationState adaptation;
	uint64_t ulAdaptStatsSeqNo = 0;      // batch whose probe wrote the readback
	uint64_t ulAdaptConsumedSeqNo = 0;   // last readback folded into the EMAs
	// Dataset capture (Stage C): batch whose copies filled the recorder
	// readbacks, and the last one flushed to disk — same one-batch-in-flight
	// race-freedom argument as the adapt pair above.
	uint64_t ulNetRecordSeqNo = 0;
	uint64_t ulNetRecordConsumedSeqNo = 0;
	// Online-learning profile dump (C2): batch whose copy filled the profile
	// readback, and the last one written to the profile file.
	uint64_t ulNetProfileSeqNo = 0;
	uint64_t ulNetProfileConsumedSeqNo = 0;
	bool valid = false;
};

static FramegenHistory_t g_framegenHistory;

struct FramegenImagePoolPressure_t
{
	size_t nTextureBusy = 0;
	size_t nBackendBusy = 0;
	uint64_t ulTextureRefs = 0;
	uint64_t ulBackendExternalRefs = 0;
};

static FramegenImagePoolPressure_t framegen_image_pool_pressure(
	const std::vector<gamescope::OwningRc<CVulkanTexture>> &images )
{
	FramegenImagePoolPressure_t pressure;
	for ( const auto &pImage : images )
	{
		if ( pImage == nullptr )
			continue;

		const uint32_t uTextureRefs = pImage->GetRefCount();
		const gamescope::IBackendFb *pBackendFb = pImage->GetBackendFb();
		const uint32_t uBackendRefs = pBackendFb != nullptr ? pBackendFb->GetRefCount() : 0;
		// The first backend-fb ref belongs to the first public texture ref. Any
		// excess is an external owner such as a pending KMS commit or unreleased
		// nested-Wayland attach.
		const uint32_t uBackendExternalRefs = uBackendRefs
			- std::min( uBackendRefs, uTextureRefs != 0 ? 1u : 0u );
		pressure.nTextureBusy += uTextureRefs != 0;
		pressure.nBackendBusy += uBackendExternalRefs != 0;
		pressure.ulTextureRefs += uTextureRefs;
		pressure.ulBackendExternalRefs += uBackendExternalRefs;
	}
	return pressure;
}

static void framegen_log_output_ring_pressure()
{
	size_t nHistoryRefs = 0;
	size_t nReadPinRefs = 0;
	size_t nPendingRealRefs = 0;
	const FramegenImagePoolPressure_t pressure = framegen_image_pool_pressure(
		g_output.outputImages );

	for ( const auto &pOwnedImage : g_output.outputImages )
	{
		CVulkanTexture *pImage = pOwnedImage.get();
		if ( pImage == nullptr )
			continue;

		nHistoryRefs += g_framegenHistory.previousReal.get() == pImage;
		nHistoryRefs += g_framegenHistory.currentReal.get() == pImage;
		nReadPinRefs += g_framegenHistory.genReadA.get() == pImage;
		nReadPinRefs += g_framegenHistory.genReadB.get() == pImage;
		nReadPinRefs += g_framegenHistory.genReadReference.get() == pImage;
		for ( const FramegenHistory_t::PendingGenerated_t &entry : g_framegenHistory.pending )
			nPendingRealRefs += entry.bReal && entry.tex.get() == pImage;
	}

	vk_log.infof( "framegen: real output-ring pressure pool=%zu texture-busy=%zu refs=%" PRIu64
		" backend-busy=%zu external-refs=%" PRIu64 " history-refs=%zu read-pins=%zu pending-real-refs=%zu pending=%zu; skipping live-buffer composite",
		g_output.outputImages.size(), pressure.nTextureBusy, pressure.ulTextureRefs,
		pressure.nBackendBusy, pressure.ulBackendExternalRefs, nHistoryRefs, nReadPinRefs,
		nPendingRealRefs, g_framegenHistory.pending.size() );
}
static bool g_bLoggedFramegenConfig = false;
static constexpr uint64_t k_ulFramegenMaxRealFrameGapNs = 250'000'000ull;
// Dedicated-queue idle refill may extrapolate beyond the originally expected
// next real frame if the game stalls. Cap the forward step so a long stall does
// not run prediction unbounded; after this point additional refills converge
// toward the capped prediction instead of accelerating away from real content.
static constexpr float k_flFramegenMaxForwardStrength = 1.5f;

static uint64_t framegen_commit_lead_override_ns()
{
	static const uint64_t s_ulOverrideNs = []()
	{
		const auto value = gamescope::framegen::parse_finite_float_setting(
			getenv( "GAMESCOPE_FRAMEGEN_COMMIT_LEAD_MS" ) );
		if ( !value.has_value() || *value <= 0.0f )
			return uint64_t{ 0u };

		constexpr double k_flNsPerMs = 1.0e6;
		const double flMs = *value;
		if ( flMs >= std::numeric_limits<uint64_t>::max() / k_flNsPerMs )
			return std::numeric_limits<uint64_t>::max();
		return std::max<uint64_t>( 1u,
			static_cast<uint64_t>( flMs * k_flNsPerMs + 0.5 ) );
	}();
	return s_ulOverrideNs;
}

static gamescope::framegen::FixedRefreshPresentLead_t
framegen_native_present_lead( uint64_t ulPresentMarginNs,
	uint64_t ulVblankIntervalNs )
{
	return gamescope::framegen::select_fixed_refresh_present_lead(
		g_framegenPresentState.displayTiming.presentLead,
		g_framegenPresentState.displayTiming.minimumViablePresentLead,
		ulPresentMarginNs, ulVblankIntervalNs,
		framegen_commit_lead_override_ns() );
}

static gamescope::framegen::FixedRefreshCommitPlan_t
framegen_plan_fixed_refresh_commit( uint64_t ulTargetFlipNs,
	uint64_t ulPresentMarginNs, uint64_t ulVblankIntervalNs )
{
	return gamescope::framegen::plan_fixed_refresh_commit(
		ulTargetFlipNs,
		g_framegenPresentState.displayTiming.presentLead,
		g_framegenPresentState.displayTiming.minimumViablePresentLead,
		ulPresentMarginNs, ulVblankIntervalNs,
		framegen_commit_lead_override_ns() );
}

static uint64_t framegen_predicted_interval_ns()
{
	return gamescope::framegen::predicted_cadence_interval_ns(
		g_framegenHistory.cadence );
}

static bool framegen_refill_idle();
static bool framegen_causal_submit( uint64_t ulCompositeSeqNo );
static bool framegen_vrr_hybrid_submit( uint64_t ulCompositeSeqNo, uint32_t nMaxDegradeSteps );
static void framegen_clear_vrr_midpoint_state( bool bClearPending );
static bool framegen_format_supports_sampled_storage( uint32_t drmFormat );
static gamescope::Rc<CVulkanTexture> framegen_base_present_composite( gamescope::Rc<CVulkanTexture> pGeneratedBase, uint64_t ulFramegenSeqNo, const struct FrameInfo_t *pPresentFrameInfo );
static gamescope::Rc<CVulkanTexture> framegen_cursor_present_composite( gamescope::Rc<CVulkanTexture> pGeneratedOutput, uint64_t ulFramegenSeqNo, const struct FrameInfo_t *pPresentFrameInfo );

static const char *framegen_color_record_dir()
{
	static const char *s_pszDir = gamescope::framegen::non_empty_setting(
		getenv( "GAMESCOPE_FRAMEGEN_RECORD_COLOR" ) );
	return s_pszDir;
}

static uint32_t framegen_color_record_uint( const char *pszName, uint32_t uDefault, bool bAllowZero )
{
	return gamescope::framegen::parse_uint32_setting(
		getenv( pszName ), bAllowZero ).value_or( uDefault );
}

static uint32_t framegen_color_record_max()
{
	// GSCF v2/v3 stores three generated candidates plus the exact reference. Eight
	// 1440p XB30 samples are already about 500 MiB (about 1 GiB at 4K), so keep
	// the safe default deliberately small; explicit experiments may raise it.
	static const uint32_t s_uMax = framegen_color_record_uint(
		"GAMESCOPE_FRAMEGEN_RECORD_COLOR_MAX", 8u, false );
	return s_uMax;
}

static uint32_t framegen_color_record_skip()
{
	static const uint32_t s_uSkip = framegen_color_record_uint(
		"GAMESCOPE_FRAMEGEN_RECORD_COLOR_SKIP", 0u, true );
	return s_uSkip;
}

static uint32_t framegen_color_record_span()
{
	// A span of N retains A and one exact reference until endpoint C arrives N
	// real frames later. Two output-ring pins leave three rotating composite
	// targets, so longer validation spans need no larger production ring.
	static const uint32_t s_uSpan = std::clamp( framegen_color_record_uint(
		"GAMESCOPE_FRAMEGEN_RECORD_COLOR_SPAN", 2u, false ), 2u, 16u );
	return s_uSpan;
}

static uint32_t framegen_color_record_offset()
{
	static const uint32_t s_uOffset = []()
	{
		const uint32_t uSpan = framegen_color_record_span();
		const uint32_t uOffset = framegen_color_record_uint(
			"GAMESCOPE_FRAMEGEN_RECORD_COLOR_OFFSET", 1u, false );
		return uOffset < uSpan ? uOffset : 1u;
	}();
	return s_uOffset;
}

static float framegen_color_record_phase_tolerance()
{
	// Frame-count spacing does not imply timestamp spacing when the source
	// cadence jitters. Leave broad, measured-phase capture as the default, but
	// let targeted sweeps reject intervals outside OFFSET/SPAN +/- tolerance.
	static const float s_flTolerance = []()
	{
		const auto value = gamescope::framegen::parse_finite_float_setting(
			getenv( "GAMESCOPE_FRAMEGEN_RECORD_COLOR_PHASE_TOLERANCE" ) );
		return value.has_value() ? std::clamp( *value, 0.0f, 1.0f ) : 1.0f;
	}();
	return s_flTolerance;
}

static FramegenColorProbeSweep framegen_color_record_sweep()
{
	static const FramegenColorProbeSweep s_eSweep = []()
	{
		const char *pszSweep = gamescope::framegen::non_empty_setting(
			getenv( "GAMESCOPE_FRAMEGEN_RECORD_COLOR_SWEEP" ) );
		if ( pszSweep == nullptr )
			return FramegenColorProbeSweep::Occlusion;
		if ( const auto parsed = gamescope::framegen::parse_color_probe_sweep_setting( pszSweep ) )
			return *parsed;
		vk_log.errorf( "framegen: unknown GAMESCOPE_FRAMEGEN_RECORD_COLOR_SWEEP='%s'; using occlusion", pszSweep );
		return FramegenColorProbeSweep::Occlusion;
	}();
	return s_eSweep;
}

// Step 5 deletes this temporary A/B escape hatch together with the classic
// batch planner. Shared-queue devices continue to use that conservative path
// regardless because one-slot speculation must not block a real composite.
static bool framegen_classic_enabled()
{
	static const bool s_bEnabled = env_to_bool( getenv( "GAMESCOPE_FRAMEGEN_CLASSIC" ) );
	return s_bEnabled;
}

// Absolute-deadline, one-slot causal scheduling is the production fixed-refresh
// policy whenever a dedicated framegen queue is available. Keep accepting the
// old opt-in for one migration interval so existing launchers do not break.
static bool framegen_causal_deadline_enabled()
{
	return g_device.hasFramegenQueue() && !framegen_classic_enabled();
}

// VRR hybrid (#01): instead of suppressing adaptive sync, present real frames
// immediately (full VRR latency win) and show the one generated frame via a
// timer-armed mid-interval flip. Requires the dedicated framegen queue for the
// same reason causal deadline scheduling does. "Requested" gates the allowVRR decision in steamcompmgr
// (VRR can only BECOME active if real flips carry allowVRR=true); "active"
// additionally requires the connector to actually be in VRR right now — when it
// is not (nested, panel without VRR, adaptive sync toggled off), every decision
// point falls back to the fixed-refresh paths, live, with no restart.
static bool framegen_vrr_hybrid_configured()
{
	static const bool s_bEnabled = env_to_bool( getenv( "GAMESCOPE_FRAMEGEN_VRR_HYBRID" ) );
	return s_bEnabled;
}

bool vulkan_framegen_vrr_hybrid_requested()
{
	return framegen_vrr_hybrid_configured() && g_device.hasFramegenQueue();
}

bool vulkan_framegen_vrr_hybrid_active()
{
	return vulkan_framegen_vrr_hybrid_requested()
		&& GetBackend() != nullptr
		&& GetBackend()->GetCurrentConnector() != nullptr
		&& GetBackend()->GetCurrentConnector()->IsVRRActive();
}

// Base-layer generation with late overlay composite (#02): generate on the
// pre-upscale game layer instead of the final composited output, then push the
// generated base through the real composite pipeline at present time with the
// CURRENT overlay stack on top. Overlays, HUDs and the cursor are therefore
// never extrapolated — the classic framegen UI-smear artifact class is removed
// by construction, and prediction runs in the game's own encoding (the correct
// space for motion) with the full shaper/3D-LUT color pipeline applied to
// generated frames exactly as to real ones. Opt-in prototype toggle.
static bool framegen_base_layer_enabled()
{
	static const bool s_bEnabled = env_to_bool( getenv( "GAMESCOPE_FRAMEGEN_BASE" ) );
	return s_bEnabled;
}

bool vulkan_framegen_base_layer_active()
{
	return framegen_base_layer_enabled() && g_framegenHistory.bBaseLayer;
}

// Per-frame dispatcher rung decision for #02 — deliberately a handful of
// compares plus one cached format-capability probe, so it costs nothing per
// frame. When the paint config can't take the base path (video underlay in
// layer 0, YCbCr game buffer, ReShade active — ReShade rewrites layer 0 inside
// vulkan_composite and would run twice and host-stall on the generated
// composite, or a base format without storage-image support), the recorder
// falls back LIVE to the output-space path; the switch is mediated by
// the dims/mode-keyed history reset, so the two paths never mix within a
// scene.
static bool framegen_base_layer_usable( const struct FrameInfo_t *pFrameInfo )
{
	if ( !framegen_base_layer_enabled()
		|| pFrameInfo->layerCount < 1
		|| pFrameInfo->layers[ 0 ].tex == nullptr
		|| pFrameInfo->layers[ 0 ].zpos != g_zposBase
		|| pFrameInfo->layers[ 0 ].tex->isYcbcr()
		|| !g_reshade_effect.empty() )
		return false;

	// The generation shaders write the generated base as a storage image in
	// the CLIENT's format (a raw copy into a swizzled format would swap
	// channels, so the format is kept). Probe storage support once per format.
	static uint32_t s_uCheckedDrmFormat = DRM_FORMAT_INVALID;
	static bool s_bFormatSupported = false;
	const uint32_t uBaseFormat = pFrameInfo->layers[ 0 ].tex->drmFormat();
	if ( uBaseFormat != s_uCheckedDrmFormat )
	{
		s_uCheckedDrmFormat = uBaseFormat;
		s_bFormatSupported = framegen_format_supports_sampled_storage( uBaseFormat );
		if ( g_bFramegenDebug && !s_bFormatSupported )
			vk_log.infof( "framegen: base-layer path unavailable for format 0x%" PRIX32 " (no sampled+storage support), using output-space generation", uBaseFormat );
	}
	return s_bFormatSupported;
}

// Output-space cursor cadence. gamescope has no hardware cursor plane: the
// pointer is a normal composited layer (zpos == g_zposCursor) and in
// output-space mode the finished composite — cursor baked in — is what becomes
// framegen history, so every generated frame shows a warped copy of a cursor
// position that is one real frame old. The fix is two halves that only make
// sense together: the real composite keeps the cursor OUT of the recorded
// history image, and each generated frame gets the CURRENT cursor composited
// on top at present time. GAMESCOPE_FRAMEGEN_CURSOR=0 disables both halves and
// restores the previous behaviour exactly (including generated-frame image
// usage flags, see framegen_create_output_texture).
static bool framegen_cursor_split_enabled()
{
	static const bool s_bEnabled = []()
	{
		const char *pszEnv = getenv( "GAMESCOPE_FRAMEGEN_CURSOR" );
		return pszEnv == nullptr || env_to_bool( pszEnv );
	}();
	return s_bEnabled;
}

// The cursor is painted last by steamcompmgr, but not unconditionally last:
// mura correction (Steam Deck OLED) appends a plane above it. Both halves of
// the split draw the cursor as the topmost layer, so they only engage when it
// really is the top of the stack; anything else keeps today's single-pass
// behaviour. Returns the layer index, or -1.
static int framegen_cursor_top_layer_index( const struct FrameInfo_t *pFrameInfo )
{
	if ( pFrameInfo == nullptr || pFrameInfo->layerCount < 2 )
		return -1;
	const int nLast = pFrameInfo->layerCount - 1;
	if ( pFrameInfo->layers[ nLast ].zpos != (int)g_zposCursor
		|| pFrameInfo->layers[ nLast ].tex == nullptr )
		return -1;
	return nLast;
}

// True when the stack carries a cursor layer anywhere. A composite with no
// cursor at all is already cursor-free history and needs no split.
static bool framegen_frame_has_cursor_layer( const struct FrameInfo_t *pFrameInfo )
{
	if ( pFrameInfo == nullptr )
		return false;
	for ( int i = 0; i < pFrameInfo->layerCount; i++ )
	{
		if ( pFrameInfo->layers[ i ].zpos == (int)g_zposCursor
			&& pFrameInfo->layers[ i ].tex != nullptr )
			return true;
	}
	return false;
}

// Overlay-only repaints — a HUD tick, or the pointer moving across an
// otherwise idle scene — re-composite unchanged game content and are NOT
// recorded as history: framegen_record_real_frame drops them on exactly this
// predicate. Testing it before compositing keeps the split's second pass off
// every such repaint, which is what makes a mouse sweep over a paused game free
// again. The shared predicate is reused rather than copied, and a disagreement
// would be harmless in both directions (one wasted pass, or one frame of
// today's baked-in cursor).
static bool framegen_composite_records_history( const struct FrameInfo_t *pFrameInfo )
{
	const CVulkanTexture *pBaseTexture = pFrameInfo->layerCount > 0
		? pFrameInfo->layers[ 0 ].tex.get() : nullptr;
	const uint64_t ulBaseCommitID = pFrameInfo->layerCount > 0
		? pFrameInfo->layers[ 0 ].ulCommitID : 0u;
	return pBaseTexture == nullptr
		|| gamescope::framegen::is_new_real_frame_content(
			{ g_framegenHistory.ulLastBaseCommitID, g_framegenHistory.pLastBaseTexture },
			{ ulBaseCommitID, pBaseTexture } );
}

// Bidirectional interpolation (B3): generated frames sit BETWEEN the two real
// frames — both are warped toward the slot's phase along the forward and the
// (already-computed) reverse motion field and blended by confidence, so
// occlusions take content from the frame that has it and unmatched regions get
// a phase-correct crossfade instead of a static hold. This is the structural
// fix for extrapolation's fallback judder (killed regions holding then jumping
// at base rate) and for translucent content (a crossfade is phase-correct for
// both motion layers). The price is intrinsic: the real frame can only be
// presented AFTER its interpolations, i.e. up to one real-frame interval of
// added latency — which is why it is opt-in. Motion mode only; the pacing
// VRR hybrid and base-layer paths remain mutually exclusive with it; bidir
// otherwise uses the shared absolute-deadline timeline.
static bool framegen_bidir_enabled()
{
	static const bool s_bEnabled = env_to_bool( getenv( "GAMESCOPE_FRAMEGEN_BIDIR" ) );
	return s_bEnabled;
}

// Experimental occlusion-side authority for bidirectional interpolation. The
// baseline warp already normalizes a one-sided field to the surviving gather,
// but its final validity is phase-weighted and therefore dissolves much of that
// valid gather back into the unwarped crossfade. This knob restores a bounded,
// phase-aware amount of the surviving side only when the opposite direction is
// clearly rejected. Zero leaves the established shader path exactly unchanged.
static float framegen_bidir_one_sided_strength()
{
	static const float s_flStrength = []()
	{
		const auto value = gamescope::framegen::parse_finite_float_setting(
			getenv( "GAMESCOPE_FRAMEGEN_BIDIR_OCCLUSION" ) );
		return value.has_value() ? std::clamp( *value, 0.0f, 1.0f ) : 0.0f;
	}();
	return s_flStrength;
}

// Guided-only intermediate-grid correction. Endpoint motion fields are
// defined on their respective real-frame grids, so sampling them directly at
// an intermediate output coordinate is only the first fixed-point iterate.
// This opt-in strength selects a separately specialized, symmetric,
// closure-checked warp. Cheaper pipelines retain the established behavior.
static float framegen_bidir_endpoint_trace_strength( GamescopeFramegenPipeline ePipeline )
{
	if ( ePipeline != GamescopeFramegenPipeline::Guided )
		return 0.0f;

	static const float s_flStrength = []()
	{
		const auto value = gamescope::framegen::parse_finite_float_setting(
			getenv( "GAMESCOPE_FRAMEGEN_BIDIR_TRACE" ) );
		return value.has_value() ? std::clamp( *value, 0.0f, 1.0f ) : 0.0f;
	}();
	return s_flStrength;
}

bool vulkan_framegen_bidir_active()
{
	return framegen_bidir_enabled()
		&& g_eFramegenMode == GamescopeFramegenMode::Motion
		&& !vulkan_framegen_vrr_hybrid_requested()
		&& !g_framegenHistory.bBaseLayer;
}

static bool framegen_color_probe_requested()
{
	return framegen_color_record_dir() != nullptr
		&& g_device.hasFramegenQueue()
		&& vulkan_framegen_bidir_active()
		&& !framegen_base_layer_enabled()
		&& !g_framegenHistory.bBaseLayer;
}

static bool framegen_color_probe_active()
{
	return framegen_color_probe_requested()
		&& g_uFramegenColorRecordCount < framegen_color_record_max();
}

bool vulkan_framegen_is_enabled()
{
	// Only enable on backends that actually present generated frames. Otherwise
	// generation would run and be discarded, and the forced-composite / no-VRR /
	// no-tearing tax would be paid for no benefit (Headless, OpenVR, SDL).
	if ( !framegen_backend_supported() )
		return false;

	return g_bExperimentalFramegen
		&& g_nFramegenMultiplier >= 2 && g_nFramegenMultiplier <= 4
		&& ( g_eFramegenMode == GamescopeFramegenMode::Extrapolate
			|| g_eFramegenMode == GamescopeFramegenMode::Blend
			|| g_eFramegenMode == GamescopeFramegenMode::Motion );
}

// Deadline-driven degradation ladder (#04). The startup mode/pipeline/multiplier are the
// cost ceiling; under GPU-time pressure the ladder sheds work immediately.
// Slow evidence-gated recovery may jump to the richest measured fitting rung,
// or re-probe one adjacent cold rung. It NEVER mutates
// the base globals (g_eFramegenMode / g_nFramegenMultiplier): those gate the whole
// feature (vulkan_framegen_is_enabled) and size the output pool, so the ladder only
// ever degrades within them and reads its result through framegen_effective_config.
using FramegenEffective_t = gamescope::framegen::EffectiveConfig;

static bool framegen_recovery_enabled()
{
	static const bool s_bEnabled = []()
	{
		const char *pszEnv = getenv( "GAMESCOPE_FRAMEGEN_RECOVER" );
		return pszEnv == nullptr || env_to_bool( pszEnv );
	}();
	return s_bEnabled;
}

static bool framegen_recovery_active_for_path()
{
	return framegen_recovery_enabled()
		&& !vulkan_framegen_vrr_hybrid_active()
		&& !vulkan_framegen_bidir_active();
}

static bool framegen_native_causal_deadline_path()
{
	gamescope::IBackend *pBackend = GetBackend();
	return framegen_causal_deadline_enabled()
		&& !vulkan_framegen_vrr_hybrid_active()
		&& !vulkan_framegen_bidir_active()
		&& pBackend != nullptr && pBackend->OwnsKMSPresentTiming();
}

static void framegen_recovery_reset_streak()
{
	if ( !framegen_recovery_active_for_path() )
		return;
	g_framegenHistory.recovery =
		gamescope::framegen::reset_ladder_recovery_streak(
			g_framegenHistory.recovery );
}

static void framegen_recovery_note_capacity_failure()
{
	if ( !framegen_recovery_active_for_path() )
		return;
	g_framegenHistory.recovery =
		gamescope::framegen::note_ladder_recovery_failure(
			g_framegenHistory.recovery );
}

static void framegen_recovery_note_degradation()
{
	if ( !framegen_recovery_active_for_path() )
		return;
	g_framegenHistory.recovery =
		gamescope::framegen::note_ladder_recovery_degradation(
			g_framegenHistory.recovery );
}

static void framegen_log_ladder_recovery( uint32_t nMaxDegradeSteps,
	uint32_t nPreviousRung, uint32_t uEvidenceDecisions )
{
	vk_log.infof(
		"framegen: ladder recovered to rung %u/%u (from %u) after %u decisions",
		g_framegenHistory.nDegradeSteps, nMaxDegradeSteps,
		nPreviousRung, uEvidenceDecisions );
}

static void framegen_log_ladder_recovery_blocked( uint32_t nMaxDegradeSteps,
	uint32_t nDegradeHold )
{
	vk_log.infof(
		"framegen: ladder recovery threshold reached at rung %u/%u but blocked"
		" (hold %u, probation %u, backoff %u/%u)",
		g_framegenHistory.nDegradeSteps, nMaxDegradeSteps, nDegradeHold,
		g_framegenHistory.recovery.probationRemaining,
		g_framegenHistory.recovery.decisionsSinceClimb,
		g_framegenHistory.recovery.backoffDecisions );
}

// Total number of rungs available from the startup config: motion walks through
// every cheaper pipeline and then to extrapolate, followed by one rung per
// multiplier notch down to x2. Deliberately does NOT include a "stop generating" rung: the
// ladder always keeps generating (at worst the cheapest config), so its GPU-time
// input never starves. The genuine "even the cheapest config overruns" case is
// left to the existing reactive pacing gate (bGpuHasHeadroom + nStableFrames),
// which is driven by live per-frame signals and so self-heals, unlike a
// measurement-frozen ladder rung would.
static uint32_t framegen_max_degrade_steps()
{
	return gamescope::framegen::max_degrade_steps(
		g_eFramegenMode, g_eFramegenPipeline, g_nFramegenMultiplier );
}

// Apply nDegradeSteps degradations to the startup ceiling. Motion (the priciest
// pass) is shed first, then the multiplier is stepped down to x2. nDegradeSteps
// is always clamped to framegen_max_degrade_steps(), so n is fully consumed and
// the result is always a still-generating config (never dormant).
static FramegenEffective_t framegen_effective_config( uint32_t nDegradeSteps )
{
	return gamescope::framegen::effective_config(
		g_eFramegenMode, g_eFramegenPipeline, g_nFramegenMultiplier, nDegradeSteps );
}

static size_t framegen_bidir_pending_hard_capacity( uint32_t multiplier )
{
	return gamescope::framegen::bidir_pending_hard_capacity(
		g_output.outputImages.size(), multiplier );
}

static size_t framegen_bidir_pending_hard_capacity()
{
	return framegen_bidir_pending_hard_capacity(
		framegen_effective_config(
			g_framegenHistory.nDegradeSteps ).multiplier );
}

// True when the queue is over-subscribed relative to what can still drain —
// either at the ring-derived ceiling, or deeper than the interval the planner
// last admitted for. Everything else flips on its own display target.
static bool framegen_bidir_queue_forces_drain()
{
	return gamescope::framegen::bidir_queue_forces_drain(
		g_framegenHistory.pending.size(),
		framegen_bidir_pending_hard_capacity(),
		g_framegenHistory.uBidirPendingTarget );
}

static bool framegen_output_matches( const gamescope::OwningRc<CVulkanTexture> &pTexture, uint32_t width, uint32_t height, uint32_t drmFormat )
{
	return pTexture != nullptr
		&& pTexture->width() == width
		&& pTexture->height() == height
		&& pTexture->drmFormat() == drmFormat;
}

static void framegen_release_completed_read_pins()
{
	if ( g_framegenHistory.genReadSeqNo == 0
		|| !g_device.hasCompletedFramegen( g_framegenHistory.genReadSeqNo ) )
		return;

	g_framegenHistory.genReadA = nullptr;
	g_framegenHistory.genReadB = nullptr;
	g_framegenHistory.genReadReference = nullptr;
	g_framegenHistory.genReadSeqNo = 0;
}

void vulkan_framegen_invalidate_history( const char *reason )
{
	// Completed generation no longer needs its zero-copy input pins. Release
	// them before the early-out and before dropping logical history; an active
	// batch deliberately keeps its pins until its timeline point signals.
	framegen_release_completed_read_pins();

	const bool bHitch = reason != nullptr
		&& ( strcmp( reason, "frame_gap" ) == 0
			|| strcmp( reason, "idle_frame_gap" ) == 0 );
	if ( bHitch )
		g_framegenPresentState.bPresentBiasHitchEpisode = true;
	const bool bRing = reason != nullptr
		&& strcmp( reason, "real_output_ring_pressure" ) == 0;
	if ( ( !bRing || !vulkan_framegen_bidir_active() )
		&& !g_framegenHistory.valid && g_framegenHistory.pending.empty()
		&& g_framegenHistory.previousReal == nullptr && g_framegenHistory.currentReal == nullptr
		&& g_framegenHistory.cadence.intervalNs == 0
		&& g_framegenHistory.ulCurrentCadenceTimeNs == 0 )
		return;
	const bool bCut = reason != nullptr
		&& ( strcmp( reason, "base_colorspace_change" ) == 0
			|| strcmp( reason, "output_eotf_change" ) == 0
			|| strcmp( reason, "layer_count_change" ) == 0 );
	framegen_metrics_note_reset( bCut ? FramegenResetReason_t::Cut
		: bHitch ? FramegenResetReason_t::Hitch
		: bRing ? FramegenResetReason_t::Ring
		: FramegenResetReason_t::Grid );

	if ( g_bFramegenDebug )
		vk_log.infof( "framegen: history valid=false reason=%s", reason ? reason : "unknown" );

	gamescope::Rc<CVulkanTexture> pRingPressureHold;
	if ( bRing && vulkan_framegen_bidir_active() )
	{
		if ( g_framegenHistory.nBidirRingPressureFailures != UINT32_MAX )
			g_framegenHistory.nBidirRingPressureFailures++;
		if ( g_framegenHistory.nBidirRingPressureFailures == 1u )
		{
			uint64_t ulNewestRealFrameId = 0u;
			for ( const FramegenHistory_t::PendingGenerated_t &entry :
				g_framegenHistory.pending )
			{
				if ( entry.bReal && entry.tex != nullptr
					&& entry.ulPresentRealFrameId >= ulNewestRealFrameId )
				{
					ulNewestRealFrameId = entry.ulPresentRealFrameId;
					pRingPressureHold = entry.tex;
				}
			}
			if ( pRingPressureHold == nullptr )
				pRingPressureHold = g_framegenHistory.currentReal != nullptr
					? g_framegenHistory.currentReal
					: g_framegenHistory.bidirLastOutput;
		}
	}

	g_framegenHistory.valid = false;
	// Bidir (B3): queued REAL frames are actual content the user has not seen
	// yet — dropping one at a scene change would skip a painted frame (a
	// visible hitch), so only the interpolations (now-stale predictions) go.
	// Their ring slots stay protected: the composite ring advance pins slots
	// referenced by pending real entries. Everything else clears as before.
	if ( vulkan_framegen_bidir_active() && !bRing )
	{
		std::erase_if( g_framegenHistory.pending,
			[]( const FramegenHistory_t::PendingGenerated_t &entry ) { return !entry.bReal; } );
	}
	else
	{
		g_framegenHistory.pending.clear();
	}
	g_framegenHistory.nStableFrames = 0;
	g_framegenHistory.previousPresentTimeNs = 0;
	g_framegenHistory.currentPresentTimeNs = 0;
	// The cadence predictor re-seeds from the first post-prime interval: a scene
	// change is exactly where its period, drift, uncertainty, or timestamp
	// provenance may be stale.
	g_framegenHistory.cadence = {};
	g_framegenHistory.ulCurrentCadenceTimeNs = 0;
	g_framegenHistory.bCadenceUsesSourceTime = false;
	g_framegenHistory.ulCurrentRealVblankNs = 0;
	g_framegenHistory.causalAnchor = {};
	g_framegenHistory.ulDeadlineGridEpoch = 0;
	g_framegenHistory.ulDeadlineGridIntervalNs = 0;
	g_framegenHistory.ulDeadlineDisplayChainGeneration = 0;
	g_framegenHistory.ulLastPlannedTargetNs = 0;
	g_framegenHistory.bDeadlineProvenanceInitialized = false;
	g_framegenHistory.bDeadlineSourceTimestampsReliable = false;
	g_framegenHistory.bCausalDeadlineMissed = false;
	g_framegenHistory.causalDeadlineMisses = {};
	// deadline bias and VRR present lead live in displayTiming and intentionally
	// survive this content reset. The shadow retains its independent bias while
	// dropping only anchors/decisions derived from the invalidated content.
	framegen_invalidate_deadline_shadow_content();
	g_framegenHistory.bidirEpoch = {};
	g_framegenHistory.bidirCutEpisode = {};
	g_framegenHistory.ulBidirGridEpoch = 0;
	g_framegenHistory.ulBidirGridIntervalNs = 0;
	g_framegenHistory.ulBidirDisplayChainGeneration = 0;
	g_framegenHistory.ulBidirGridTargetNs = 0;
	g_framegenHistory.ulBidirGridWakeNs = 0;
	g_framegenHistory.bBidirProvenanceInitialized = false;
	g_framegenHistory.bBidirSourceTimestampsReliable = false;
	g_framegenHistory.bidirFeedbackEndpoints.clear();
	if ( bRing )
		g_framegenHistory.bidirLastOutput = pRingPressureHold;
	g_framegenHistory.ulVrrAwaitingRealFrameId = 0;
	g_framegenHistory.ulVrrAwaitingCompositeSeqNo = 0;
	g_framegenHistory.ulVrrAwaitingCadenceNs = 0;
	g_framegenHistory.ulVrrPanelIntervalNs = 0;
	g_framegenHistory.ulVrrMidTargetNs = 0;
	g_framegenHistory.ulVrrMidWakeNs = 0;
	g_framegenHistory.lastCompositeSeqNo = 0;
	g_framegenHistory.nLastGeneratedSlot = 0;
	g_framegenHistory.nLastGenerationGapVblanks = 0;
	// Re-probe the configured pipeline from the top on a fresh scene and restore the base recovery
	// delay; neither stale rung costs nor a prior scene's failed probes carry over.
	g_framegenHistory.nDegradeSteps = 0;
	g_framegenHistory.nDegradeHold = 0;
	g_framegenHistory.recovery = {};
	// Forget learned per-rung costs so the new scene is measured afresh rather than
	// inheriting the old scene's "this rung overruns" verdicts.
	g_device.framegenResetRungCosts();
	// B4: the adaptation EMAs describe the old scene's content (noise floor,
	// round-trip ambiguity); a new scene re-calibrates from the defaults.
	g_framegenHistory.adaptation = {};
	g_framegenHistory.ulAdaptStatsSeqNo = 0;
	g_framegenHistory.ulAdaptConsumedSeqNo = 0;
	// Stage C dataset capture: a pending, never-flushed batch of tensors dies
	// with the scene (the sample counter itself persists — it names files).
	g_framegenHistory.ulNetRecordSeqNo = 0;
	g_framegenHistory.ulNetRecordConsumedSeqNo = 0;
	// C2: a pending profile snapshot dies with the scene too; the learned
	// weights themselves persist — they describe the game, not one shot.
	g_framegenHistory.ulNetProfileSeqNo = 0;
	g_framegenHistory.ulNetProfileConsumedSeqNo = 0;
	// Temporal acceleration must never cross a scene/cadence discontinuity.
	g_framegenMotion.uMotionHistoryFrameId = 0;
	g_framegenMotion.uMotionHistoryIntervalNs = 0;
	g_framegenMotion.uMotionFieldFrameId = 0;
	g_framegenMotion.uMotionFieldIntervalNs = 0;
	g_framegenMotion.uLumaReservoirFrameId[0] = 0;
	g_framegenMotion.uLumaReservoirFrameId[1] = 0;
	g_framegenColorProbe.anchor = nullptr;
	g_framegenColorProbe.reference = nullptr;
	g_framegenColorProbe.anchorId = 0;
	g_framegenColorProbe.referenceId = 0;
	g_framegenColorProbe.anchorTimeNs = 0;
	g_framegenColorProbe.referenceTimeNs = 0;
	g_framegenColorProbe.lastRealTimeNs = 0;
	g_framegenHistory.pLastBaseTexture = nullptr;
	g_framegenHistory.ulLastBaseCommitID = 0;
	// Release the retained output-ring slots so a ring rebuild is never blocked
	// by history holding a reference to an old image, and so the next real frame
	// re-primes cleanly.
	g_framegenHistory.previousReal = nullptr;
	g_framegenHistory.currentReal = nullptr;
}

static __attribute__((noinline)) void framegen_net_profile_consume();
static void framegen_net_profile_flush();
static void framegen_net_profile_join_writer();

void vulkan_framegen_reset( const char *reason )
{
	framegen_metrics_note_reset( FramegenResetReason_t::Grid );
	if ( g_bFramegenDebug )
		vk_log.infof( "framegen: reset history reason=%s", reason ? reason : "unknown" );

	// A base-history copy deliberately does not participate in the generation
	// headroom gate, but it still reads/writes resources owned by this state. A
	// live resize or base-mode transition can reach reset without a global device
	// drain; retire the latest framegen-path submission before dropping its pools,
	// descriptor-ring invariant, or output-ring read pins. This wait is confined
	// to exceptional reset paths and never runs in steady-state presentation.
	if ( g_framegenHistory.lastFramegenWorkSeqNo != 0
		&& !g_device.hasCompletedFramegen( g_framegenHistory.lastFramegenWorkSeqNo ) )
		g_device.waitFramegen( g_framegenHistory.lastFramegenWorkSeqNo );
	if ( g_framegenHistory.lastBaseIngestSeqNo != 0
		&& !g_device.hasCompleted( g_framegenHistory.lastBaseIngestSeqNo ) )
		g_device.wait( g_framegenHistory.lastBaseIngestSeqNo );
	g_device.framegenGarbageCollect();

	// C2: persist unsaved learning before the state textures go away. The latest
	// health-checked served weights remain cached process-wide and seed the new
	// state, so resize/format resets do not throw away in-session adaptation.
	framegen_net_profile_consume();
	framegen_net_profile_flush();
	framegen_color_probe_consume();

	g_framegenHistory = {};
	// Resource/history rebuilds are not display-chain changes. The next real
	// observation resets the timing learners only if its display-chain key differs.
	framegen_invalidate_deadline_shadow_content();
	// The per-rung costs live on g_device and survive the history reset; forget
	// them too (as invalidate_history does) so the ladder re-probes the
	// new workload from the configured pipeline instead of stepping on the old scene's stale
	// over-deadline measurements.
	g_device.framegenResetRungCosts();
	g_output.framegenOutputImages.clear();
	g_output.framegenPresentImages.clear();
	g_output.framegenCursorHistoryImages.clear();
	g_framegenMotion = {};
	g_framegenColorProbe = {};
}

void vulkan_framegen_shutdown()
{
	if ( !vulkan_framegen_is_enabled() )
		return;

	// Framegen owns output/present ring images, history copies and pending
	// generated presents. Their CVulkanTexture destructors release backend FBs
	// (drmModeRmFB on native KMS). As file-scope statics they would otherwise
	// outlive the backend, which steamcompmgr_exit destroys — so drop them here,
	// while the backend and its KMS fd are still valid.

	// Step 1: the net-profile writer thread, unconditionally and before any
	// other teardown. It is the one framegen worker that can still be running,
	// and it logs through the file-scope vk_log. The atexit flush is far too
	// late — it runs after the backend is gone.
	framegen_net_profile_join_writer();

	// Step 2: drain the device. The two token waits this used to do
	// (lastFramegenWorkSeqNo, lastBaseIngestSeqNo) cover only the framegen and
	// base-ingest submissions. They do NOT cover genReadSeqNo — which pins
	// output-ring slots for the in-flight batch — nor the composite-timeline
	// points (ulCompositeSeqNo) that every pending bidir queue entry carries.
	// Destroying those images while the device may still reference them is a
	// textbook double-free/heap-corruption source. waitIdle retires the scratch
	// timeline to its last submitted value and, when one exists, the dedicated
	// framegen timeline too, then recycles the command buffers. Teardown is not
	// a hot path; there is no reason to be surgical here.
	if ( g_device.device() != VK_NULL_HANDLE )
		g_device.waitIdle();

	// Every readback is now guaranteed complete, so the final checkpoint is
	// consumable rather than skipped.
	framegen_net_profile_consume();
	framegen_net_profile_flush();
	framegen_color_probe_consume();

	// Step 3: release the presentation timeline and the retained ring slots
	// explicitly, and before the images they point into. In bidir the pending
	// queue IS the timeline and its bReal entries reference composite output
	// images, so it is by far the largest late-release set; making the order
	// explicit beats leaving it implicit in the aggregate assignment below.
	g_framegenHistory.pending.clear();
	g_framegenHistory.bidirLastOutput = nullptr;
	g_framegenHistory.bidirFeedbackEndpoints.clear();
	g_framegenHistory.previousReal = nullptr;
	g_framegenHistory.currentReal = nullptr;
	g_framegenHistory.genReadA = nullptr;
	g_framegenHistory.genReadB = nullptr;
	g_framegenHistory.genReadReference = nullptr;
	g_framegenHistory.genReadSeqNo = 0;

	g_framegenHistory = {};
	g_framegenMotion = {};
	g_framegenColorProbe = {};
	g_output.framegenOutputImages.clear();
	g_output.framegenPresentImages.clear();
	g_output.framegenCursorHistoryImages.clear();
	g_device.framegenGarbageCollect();
	// framegen_net_profile_consume above can re-arm the writer; its flush joins
	// again and writes inline, so this is normally a no-op. Repeating it makes
	// "no framegen thread survives this function" true by construction.
	framegen_net_profile_join_writer();
}

bool vulkan_framegen_has_pending_generated_frame()
{
	return vulkan_framegen_is_enabled() && !g_framegenHistory.pending.empty();
}

uint64_t vulkan_framegen_fixed_refresh_commit_deadline_ns()
{
	// Early commits exist to catch the KMS latch on direct scanout. A nested
	// parent compositor owns latch timing itself - the one-vblank defect never
	// occurs there, and committing ahead of the parent's frame callbacks only
	// disturbs its pacing. Native DRM only.
	if ( GetBackend() == nullptr || !GetBackend()->OwnsKMSPresentTiming() )
		return 0u;
	if ( !framegen_causal_deadline_enabled()
		|| vulkan_framegen_vrr_hybrid_active()
		|| vulkan_framegen_bidir_active()
		|| g_framegenHistory.ulDeadlineGridIntervalNs == 0u
		|| !vulkan_framegen_has_pending_generated_frame() )
		return 0u;

	const FramegenHistory_t::PendingGenerated_t &front =
		g_framegenHistory.pending.front();
	if ( front.bReal || front.ulAnchorRealFrameId == 0u
		|| front.ulTargetFlipNs == 0u )
		return 0u;

	// Match the established VRR compensation margin. The causal planner uses
	// this same bounded deadline for native generation admission, so the work
	// gate and the commit timer describe one immutable D.
	const uint64_t ulMarginNs =
		g_framegenHistory.ulDeadlineGridIntervalNs / 10u;
	const gamescope::framegen::FixedRefreshCommitPlan_t plan =
		framegen_plan_fixed_refresh_commit(
			front.ulTargetFlipNs,
			ulMarginNs,
			g_framegenHistory.ulDeadlineGridIntervalNs );
	return plan.earlyCommit ? plan.commitDeadlineNs : 0u;
}

bool vulkan_framegen_real_arrival_blocks_early_commit()
{
	// Native KMS only, and only where an early commit would actually be taken.
	// The generated atomic commit reserves KMS state and then blocks until its
	// page flip; a real client frame becoming ready inside that window cannot
	// replace the reservation and lands a vblank late. Conservative by
	// construction: no cadence model, no prediction, or an overdue prediction
	// all leave generation alone.
	if ( GetBackend() == nullptr || !GetBackend()->OwnsKMSPresentTiming() )
		return false;
	if ( !vulkan_framegen_has_pending_generated_frame()
		|| g_framegenHistory.ulDeadlineGridIntervalNs == 0u )
		return false;

	const FramegenHistory_t::PendingGenerated_t &front =
		g_framegenHistory.pending.front();
	if ( front.bReal || front.ulTargetFlipNs == 0u )
		return false;

	const uint64_t ulMarginNs =
		g_framegenHistory.ulDeadlineGridIntervalNs / 10u;
	const gamescope::framegen::FixedRefreshCommitPlan_t plan =
		framegen_plan_fixed_refresh_commit(
			front.ulTargetFlipNs, ulMarginNs,
			g_framegenHistory.ulDeadlineGridIntervalNs );
	if ( !plan.earlyCommit )
		return false;

	const uint64_t ulSourceReadyNs = g_framegenHistory.causalAnchor.sourceReadyNs;
	const uint64_t ulPredictedIntervalNs = framegen_predicted_interval_ns();
	if ( ulSourceReadyNs == 0u || ulPredictedIntervalNs == 0u )
		return false;

	const uint64_t ulNowNs = get_time_in_nanos();
	const uint64_t ulPredictedRealReadyNs = gamescope::framegen::saturating_add_ns(
		ulSourceReadyNs, ulPredictedIntervalNs );
	const bool bBlocked = gamescope::framegen::real_arrival_blocks_early_generated_commit(
		ulPredictedRealReadyNs, ulNowNs, plan.advanceNs,
		g_framegenHistory.cadence.samples
			>= gamescope::framegen::k_uCadencePredictorMinSamples );
	if ( bBlocked )
	{
		static uint64_t s_uRealArrivalSkipDebugLogCounter = 0;
		if ( FramegenDebugShouldLog( s_uRealArrivalSkipDebugLogCounter ) )
			vk_log.infof( "framegen: skipping early generated commit — next real predicted in %.3f ms, commit lead %.3f ms",
				( ulPredictedRealReadyNs - ulNowNs ) / 1.0e6,
				plan.advanceNs / 1.0e6 );
	}
	return bBlocked;
}

bool vulkan_framegen_generated_frame_due()
{
	if ( !vulkan_framegen_has_pending_generated_frame() )
		return false;
	const FramegenHistory_t::PendingGenerated_t &front =
		g_framegenHistory.pending.front();
	if ( front.ulTargetFlipNs == 0u )
		return true;
	// The non-grid timer already represents W_mid; its target is deliberately
	// still in the future when the compositor initiates the commit.
	if ( g_framegenHistory.ulVrrMidWakeNs != 0u
		&& front.ulTargetFlipNs == g_framegenHistory.ulVrrMidTargetNs )
		return get_time_in_nanos() >= g_framegenHistory.ulVrrMidWakeNs;
	// The Step 3 causal/default and classic fallback admission semantics are
	// unchanged. Their queue front already represents the next opportunity.
	if ( !vulkan_framegen_bidir_active() )
		return true;
	return front.ulTargetFlipNs <= GetVBlankTimer().GetNextVBlank( 0 )
		|| framegen_bidir_queue_forces_drain();
}

bool vulkan_framegen_generated_frame_ready()
{
	if ( !vulkan_framegen_has_pending_generated_frame() )
		return false;
	// Peek the front slot's completion without consuming it, so the present
	// decision can choose a hardware repeat over a wasted full recomposite when
	// generation hasn't finished by its vblank. A queued real frame (bidir) is
	// always ready: its composite completed at its own paint.
	return g_framegenHistory.pending.front().bReal
		|| g_device.hasCompletedFramegen( g_framegenHistory.pending.front().seqNo );
}

size_t vulkan_framegen_pending_queue_debug( double *pflFrontTargetDeltaMs )
{
	if ( pflFrontTargetDeltaMs != nullptr )
	{
		const uint64_t ulTargetNs = g_framegenHistory.pending.empty()
			? 0u : g_framegenHistory.pending.front().ulTargetFlipNs;
		*pflFrontTargetDeltaMs = ulTargetNs != 0u
			? gamescope::framegen::signed_ns_delta(
				ulTargetNs, GetVBlankTimer().GetNextVBlank( 0 ) ) / 1.0e6
			: 0.0;
	}
	return g_framegenHistory.pending.size();
}

gamescope::Rc<CVulkanTexture> vulkan_framegen_consume_generated_frame( const struct FrameInfo_t *pPresentFrameInfo )
{
	if ( !vulkan_framegen_has_pending_generated_frame() )
		return nullptr;

	// Bidir (B3): a paint that carries a NEW game frame must go through the
	// composite path — that is where history records and the interpolation
	// batch is planned — so the front is NOT consumed here. The backend's flip
	// substitution (vulkan_framegen_bidir_flip_texture) presents the front
	// after the composite has recorded, keeping the delayed timeline intact.
	// Repaints of unchanged base content (overlay ticks, repeat-vblank fills)
	// still consume normally below.
	if ( vulkan_framegen_bidir_active() && pPresentFrameInfo
		&& pPresentFrameInfo->layerCount > 0
		&& pPresentFrameInfo->layers[ 0 ].tex != nullptr
		&& pPresentFrameInfo->layers[ 0 ].tex.get() != g_framegenHistory.pLastBaseTexture )
		return nullptr;
	// Overlay-driven composites can enter the backend between scheduled bidir
	// opportunities. Leave a future queue front untouched; flip substitution
	// below the composite will explicitly hold the last valid output instead.
	if ( vulkan_framegen_bidir_active()
		&& !vulkan_framegen_generated_frame_due() )
		return nullptr;

	FramegenHistory_t::PendingGenerated_t front = g_framegenHistory.pending.front();

	// Generation was submitted in its own command buffer (and, when a dedicated
	// framegen queue exists, on its own queue) so the real frame's present never
	// waited on it. It normally completes well before the vblank it fills; if the
	// GPU is behind, presenting now would stall scanout on compute work, so drop
	// this frame instead — the display simply repeats the last scanned-out frame.
	// Never stall the present path for a generated frame. Queued real frames
	// (bidir) are always complete and never take this path.
	if ( !front.bReal && !g_device.hasCompletedFramegen( front.seqNo ) )
	{
		g_framegenHistory.pending.erase( g_framegenHistory.pending.begin() );
		framegen_metrics_note_slow_drop( 1 );
		// Native causal scheduling applies miss hysteresis at the next rung
		// decision. Other paths retain their established immediate reset.
		if ( !framegen_native_causal_deadline_path() )
			framegen_recovery_reset_streak();
		g_framegenHistory.bCausalDeadlineMissed |=
			front.ulAnchorRealFrameId != 0u;
		static uint64_t s_uTooSlowDebugLogCounter = 0;
		if ( FramegenDebugShouldLog( s_uTooSlowDebugLogCounter ) )
			vk_log.infof( "framegen: discarded generated frame id=%" PRIu64 ".%02u reason=generation_too_slow",
				front.frameId, (unsigned)( front.phase * 100.0f ) );
		if ( g_framegenHistory.pending.empty()
			&& framegen_causal_deadline_enabled()
			&& !vulkan_framegen_vrr_hybrid_active()
			&& !vulkan_framegen_bidir_active() )
			framegen_causal_submit( g_framegenHistory.lastCompositeSeqNo );
		return nullptr;
	}

	g_framegenHistory.pending.erase( g_framegenHistory.pending.begin() );

	// The readiness check above is only the non-blocking deadline gate. Do not
	// follow it with vkWaitSemaphores on the presentation path: the base-layer
	// composite carries the device-side dependency below, and even an
	// already-signalled host wait is an avoidable driver round trip at every
	// generated flip. Framegen uses push constants, so it owns no slice of the
	// shared upload arena that would need recycling here.

	// Base-layer mode (#02): the pending slot holds a pre-upscale BASE frame;
	// composite it through the real pipeline with the live layer stack (fresh
	// overlays, latest cursor) before it can be flipped. Output-space mode returns
	// the scanout-ready generated output directly.
	gamescope::Rc<CVulkanTexture> pResult = front.tex;
	if ( g_framegenHistory.bBaseLayer )
		pResult = framegen_base_present_composite( front.tex, front.seqNo, pPresentFrameInfo );
	// Output-space cursor split: this frame was predicted from cursor-free
	// history, so the pointer it should be showing is the LIVE one, composited
	// now, not a motion-warped copy of where it was a real frame ago. Returns
	// front.tex untouched when the live stack has no cursor layer.
	else if ( front.bCursorFree && !front.bReal )
		pResult = framegen_cursor_present_composite( front.tex, front.seqNo, pPresentFrameInfo );

	if ( pResult != nullptr )
	{
		if ( vulkan_framegen_bidir_active() )
			g_framegenHistory.bidirLastOutput = pResult;
		framegen_select_present_tag(
			front.bReal ? gamescope::FramegenPresentKind_t::DelayedReal
				: gamescope::FramegenPresentKind_t::Generated,
			front.ulPresentRealFrameId, front.ulSlotId,
			front.ulCompositeSeqNo, front.ulTargetFlipNs );
		static uint64_t s_uPresentedDebugLogCounter = 0;
		if ( FramegenDebugShouldLog( s_uPresentedDebugLogCounter ) )
			vk_log.infof( "framegen: presented %s frame id=%" PRIu64 ".%02u",
				front.bReal ? "delayed real" : "generated", front.frameId, (unsigned)( front.phase * 100.0f ) );
	}

	if ( g_framegenHistory.pending.empty() )
	{
		if ( vulkan_framegen_vrr_hybrid_active() )
		{
			// One mid-interval flip per real frame; the next is planned when
			// the next real frame records. NO forward-extrapolated stall
			// insurance here: an insurance flip at phase ~1.0 would land
			// exactly where the next real frame is expected, and the panel's
			// minimum flip spacing could then delay that real frame. A stall
			// is left to the panel's own LFC instead.
			framegen_clear_vrr_midpoint_state( false );
		}
		else if ( framegen_causal_deadline_enabled()
			&& !vulkan_framegen_bidir_active() )
		{
			// Every drained causal slot advances through the same absolute-grid
			// planner used by real arrivals, feedback, and repeat ticks.
			framegen_causal_submit( g_framegenHistory.lastCompositeSeqNo );
		}
		else if ( !vulkan_framegen_bidir_active() )
		{
			// Bidir never refills a stall with forward extrapolation: the
			// timeline only ever shows content between two REAL frames, so a
			// stall is an honest hold on the newest real frame (like no-FG)
			// rather than a speculative prediction on a different timeline.
			framegen_refill_idle();
		}
	}

	return pResult;
}

void vulkan_framegen_discard_generated_frame( const char *reason )
{
	if ( g_framegenHistory.pending.empty() )
		return;

	// Bidir (B3): only the interpolations are discardable predictions; a queued
	// REAL frame is painted content the user has not seen and stays queued (the
	// present decision will show it on the next vblank it wins).
	const size_t nBefore = g_framegenHistory.pending.size();
	if ( reason != nullptr && strcmp( reason, "generation_too_slow" ) == 0
		&& std::ranges::any_of( g_framegenHistory.pending,
			[]( const FramegenHistory_t::PendingGenerated_t &entry ) {
				return entry.ulAnchorRealFrameId != 0u;
			} ) )
	{
		g_framegenHistory.bCausalDeadlineMissed = true;
	}
	if ( vulkan_framegen_bidir_active() )
	{
		std::erase_if( g_framegenHistory.pending,
			[]( const FramegenHistory_t::PendingGenerated_t &entry ) { return !entry.bReal; } );
	}
	else
	{
		// A real frame supersedes the whole batch: every queued generated frame
		// is a stale prediction now and would inject a vblank of latency if
		// shown after the real frame. Drop them all.
		g_framegenHistory.pending.clear();
	}
	if ( g_framegenHistory.ulVrrMidTargetNs != 0u )
		framegen_clear_vrr_midpoint_state( false );

	const size_t nDiscarded = nBefore - g_framegenHistory.pending.size();
	if ( nDiscarded > 0u && reason != nullptr
		&& strcmp( reason, "generation_too_slow" ) == 0
		&& !framegen_native_causal_deadline_path() )
		framegen_recovery_reset_streak();
	if ( reason != nullptr && strcmp( reason, "generation_too_slow" ) == 0 )
		framegen_metrics_note_slow_drop( nDiscarded );
	static uint64_t s_uDiscardDebugLogCounter = 0;
	if ( nDiscarded > 0 && FramegenDebugShouldLog( s_uDiscardDebugLogCounter ) )
		vk_log.infof( "framegen: discarded %zu generated frame(s) reason=%s",
			nDiscarded, reason ? reason : "unknown" );
}

// Bidir (B3) flip substitution — see rendervulkan.hpp. In steady state the
// queue at a real-frame paint reads [prevReal, interp(s), thisReal], so the
// substituted front is the PREVIOUS real frame: complete long ago, flipped
// exactly one measured interval after its own composite. The interpolations
// then win the repeat vblanks in between via the normal consume path.
// Present the bidir queue front in place of pFallback, or hold the previous
// real frame when the front's GPU work is still running (normally only the
// timeline bootstrap, where the interpolation was submitted microseconds ago
// in this very paint — the previous real is already on screen, so that flip is
// a visual no-op and the delayed timeline simply starts one vblank later).
static gamescope::Rc<CVulkanTexture> framegen_bidir_take_front( const gamescope::Rc<CVulkanTexture> &pFallback )
{
	const uint64_t ulOpportunityNs = g_framegenPresentState.bTagPending
		? g_framegenPresentState.pendingTag.ulTargetFlipNs
		: GetVBlankTimer().GetNextVBlank( 0 );
	while ( !g_framegenHistory.pending.empty() )
	{
		FramegenHistory_t::PendingGenerated_t front =
			g_framegenHistory.pending.front();
		const bool bForceDrain = framegen_bidir_queue_forces_drain();
		if ( front.ulTargetFlipNs != 0u
			&& front.ulTargetFlipNs > ulOpportunityNs && !bForceDrain )
			break;

		// Interpolation is disposable and never slides to a later display
		// opportunity. Endpoints remain ordered and may catch up after a hitch.
		// A frame dropped here was generated and then never shown, exactly like
		// one shed at plan time, so it counts into disc= too — no generated
		// frame may vanish uncounted.
		if ( !front.bReal
			&& ( ( front.ulTargetFlipNs != 0u
					&& front.ulTargetFlipNs < ulOpportunityNs )
				|| !g_device.hasCompletedFramegen( front.seqNo ) ) )
		{
			g_framegenHistory.pending.erase(
				g_framegenHistory.pending.begin() );
			framegen_metrics_note_discard( 1 );
			continue;
		}

		g_framegenHistory.pending.erase( g_framegenHistory.pending.begin() );
		// hasCompletedFramegen is the image-readiness proof. Blocking waits remain
		// reserved for reset and teardown, never a steady-state flip.
		static uint64_t s_uFlipDebugLogCounter = 0;
		if ( FramegenDebugShouldLog( s_uFlipDebugLogCounter ) )
			vk_log.infof( "framegen: presented %s frame id=%" PRIu64 ".%02u (bidir flip substitution)",
				front.bReal ? "delayed real" : "generated", front.frameId, (unsigned)( front.phase * 100.0f ) );
		framegen_select_present_tag(
			front.bReal ? gamescope::FramegenPresentKind_t::DelayedReal
				: gamescope::FramegenPresentKind_t::Generated,
			front.ulPresentRealFrameId, front.ulSlotId,
			front.ulCompositeSeqNo, front.ulTargetFlipNs );
		g_framegenHistory.bidirLastOutput = front.tex;
		return front.tex;
	}

	if ( g_framegenHistory.bidirLastOutput != nullptr )
	{
		// This is an explicit hold, not endpoint feedback. Classify it as
		// generated so it can update backend lead without moving the epoch.
		framegen_select_present_tag( gamescope::FramegenPresentKind_t::Generated,
			g_framegenPresentState.ulCurrentRealFrameId,
			framegen_next_present_slot_id(),
			g_framegenPresentState.ulCurrentRealCompositeSeqNo,
			ulOpportunityNs );
		return g_framegenHistory.bidirLastOutput;
	}
	return pFallback;
}

gamescope::Rc<CVulkanTexture> vulkan_framegen_bidir_flip_texture( gamescope::Rc<CVulkanTexture> pComposite )
{
	if ( !vulkan_framegen_is_enabled() || !vulkan_framegen_bidir_active() )
		return pComposite;

	if ( g_framegenHistory.bBidirQueuedReal )
	{
		// This composite recorded a new real frame and queued it behind its
		// interpolations; present the queue front in its place.
		g_framegenHistory.bBidirQueuedReal = false;
		return framegen_bidir_take_front( pComposite );
	}

	if ( g_framegenHistory.bBidirSameBaseComposite && !g_framegenHistory.pending.empty() )
	{
		// Overlay-only recomposite while the queue drains: its game content is
		// the newest frame, which is still QUEUED — flipping it now would show
		// content out of order. Present the queue front instead; the overlay
		// update rides the next real frame's (delayed) composite.
		return framegen_bidir_take_front( pComposite );
	}

	// Prime/cut/hitch holds use the same deadline selector. In particular, a
	// zero-generated interval never falls through to the newest live composite.
	if ( g_framegenHistory.bidirLastOutput != nullptr
		|| g_framegenHistory.bidirEpoch.valid )
		return framegen_bidir_take_front( pComposite );

	// This composite is presenting LIVE (framegen dormant, prime frame after a
	// scene change, game keeping up with refresh). Anything still pending
	// belongs to the abandoned delayed timeline and would present BACKWARD in
	// content time after this flip — drop it.
	if ( !g_framegenHistory.pending.empty() )
	{
		static uint64_t s_uSnapDebugLogCounter = 0;
		if ( FramegenDebugShouldLog( s_uSnapDebugLogCounter ) )
			vk_log.infof( "framegen: bidir timeline snapped to live, dropped %zu pending frame(s)",
				g_framegenHistory.pending.size() );
		g_framegenHistory.pending.clear();
	}
	g_framegenHistory.bidirLastOutput = pComposite;
	return pComposite;
}

static bool framegen_create_output_texture( gamescope::OwningRc<CVulkanTexture> *ppTexture, uint32_t width, uint32_t height, uint32_t drmFormat )
{
	CVulkanTexture::createFlags createFlags;
	createFlags.bFlippable = true;
	createFlags.bStorage = true;
	// An incompatible/ignored capture request must not alter production image
	// usage or modifier selection. Only the active E2 path copies these images.
	createFlags.bTransferSrc = framegen_color_probe_requested();
	// Output-space cursor split only: the present-time cursor composite samples
	// the generated frame as its base layer. Kept behind the same switch as the
	// feature so GAMESCOPE_FRAMEGEN_CURSOR=0 leaves usage flags and modifier
	// selection byte-for-byte as before. The output ring itself already runs
	// flippable+storage+sampled, so the combination is not new here.
	createFlags.bSampled = framegen_cursor_split_enabled();
	createFlags.bOutputImage = true;
	createFlags.bFramegenShared = true;

	*ppTexture = new CVulkanTexture();
	return ( *ppTexture )->BInit( width, height, 1u, drmFormat, createFlags );
}

// Base-mode generated frames (#02): written by the generation shaders
// (storage) and sampled by the late overlay composite as layer 0. Deliberately
// NOT flippable — they are inputs to the present-time composite, never
// scanout buffers themselves.
static bool framegen_create_base_texture( gamescope::OwningRc<CVulkanTexture> *ppTexture, uint32_t width, uint32_t height, uint32_t drmFormat )
{
	CVulkanTexture::createFlags createFlags;
	createFlags.bStorage = true;
	createFlags.bSampled = true;
	createFlags.bFramegenShared = true;

	*ppTexture = new CVulkanTexture();
	return ( *ppTexture )->BInit( width, height, 1u, drmFormat, createFlags );
}

// Base-mode history (#02): the copy target for the client's base buffer and
// the sampling source for every generation shader.
static bool framegen_create_base_history_texture( gamescope::OwningRc<CVulkanTexture> *ppTexture, uint32_t width, uint32_t height, uint32_t drmFormat )
{
	CVulkanTexture::createFlags createFlags;
	createFlags.bSampled = true;
	createFlags.bTransferDst = true;
	createFlags.bFramegenShared = true;

	*ppTexture = new CVulkanTexture();
	return ( *ppTexture )->BInit( width, height, 1u, drmFormat, createFlags );
}

static bool framegen_ensure_resources( uint32_t width, uint32_t height, uint32_t drmFormat, bool bBaseLayer )
{
	if ( g_framegenHistory.width != width || g_framegenHistory.height != height
		|| g_framegenHistory.drmFormat != drmFormat || g_framegenHistory.bBaseLayer != bBaseLayer )
	{
		// The mode is part of the reset key: flipping base<->output-space without a
		// reset would mix owned-copy history with output-ring history and
		// mislabel pending frames (a base-sized, non-flippable image must
		// never reach drm_prepare directly, and vice versa).
		vulkan_framegen_reset( g_framegenHistory.bBaseLayer != bBaseLayer ? "base_layer_toggle" : "resize_or_format_change" );
		g_framegenHistory.width = width;
		g_framegenHistory.height = height;
		g_framegenHistory.drmFormat = drmFormat;
		g_framegenHistory.bBaseLayer = bBaseLayer;
	}

	// Generated-frame pool: 2*multiplier distinct images so the (multiplier-1)
	// frames in flight plus any still being scanned out (output-space) or still
	// being read by a late composite (base mode) never alias. History
	// (previousReal/currentReal) is NOT allocated here in output-space mode — it is
	// retained by reference from the output ring in framegen_record_real_frame;
	// in base mode it lives in the two owned baseHistory images below.
	const size_t nPool = (size_t)2 * (size_t)g_nFramegenMultiplier;
	if ( g_output.framegenOutputImages.size() != nPool )
		g_output.framegenOutputImages.resize( nPool );

	for ( auto &pImage : g_output.framegenOutputImages )
	{
		if ( framegen_output_matches( pImage, width, height, drmFormat ) )
			continue;

		const bool bCreated = bBaseLayer
			? framegen_create_base_texture( &pImage, width, height, drmFormat )
			: framegen_create_output_texture( &pImage, width, height, drmFormat );
		if ( !bCreated )
		{
			vulkan_framegen_reset( "generated_allocation_failed" );
			return false;
		}
	}

	if ( bBaseLayer )
	{
		for ( auto &pImage : g_framegenHistory.baseHistory )
		{
			if ( framegen_output_matches( pImage, width, height, drmFormat ) )
				continue;

			if ( !framegen_create_base_history_texture( &pImage, width, height, drmFormat ) )
			{
				vulkan_framegen_reset( "history_allocation_failed" );
				return false;
			}
		}
	}

	return true;
}

// Late-composite scanout targets (#02): sized to the CURRENT output, which can
// change independently of the base layer (display mode switch), so this is
// (re)checked at consume time rather than at record time.
static bool framegen_ensure_present_pool()
{
	if ( g_output.outputImages.empty() || g_output.outputImages[ 0 ] == nullptr )
		return false;

	const uint32_t uWidth = g_output.outputImages[ 0 ]->width();
	const uint32_t uHeight = g_output.outputImages[ 0 ]->height();
	const uint32_t uFormat = g_output.outputImages[ 0 ]->drmFormat();

	const size_t nPool = 3;
	if ( g_output.framegenPresentImages.size() != nPool )
		g_output.framegenPresentImages.resize( nPool );

	for ( auto &pImage : g_output.framegenPresentImages )
	{
		if ( framegen_output_matches( pImage, uWidth, uHeight, uFormat ) )
			continue;

		if ( !framegen_create_output_texture( &pImage, uWidth, uHeight, uFormat ) )
		{
			g_output.framegenPresentImages.clear();
			return false;
		}
	}

	return true;
}

// Rotating acquire shared by every framegen-owned image pool. Selection is by
// actual Vulkan/backend ownership (IsInUse covers history pins, the read pins
// held by an in-flight generation batch, unretired command buffers and backend
// scanout refs) rather than by blindly cycling: writing an image somebody still
// reads is a correctness bug, and skipping the frame's extra pass is not.
static gamescope::Rc<CVulkanTexture> framegen_acquire_pool_image(
	std::vector<gamescope::OwningRc<CVulkanTexture>> &pool, uint32_t &nNext )
{
	const uint32_t nSize = (uint32_t)pool.size();
	for ( uint32_t nProbe = 0; nProbe < nSize; nProbe++ )
	{
		const uint32_t idx = nNext % nSize;
		nNext++;
		CVulkanTexture *pCandidate = pool[ idx ].get();
		if ( pCandidate != nullptr && !pCandidate->IsInUse() )
			return pCandidate;
	}
	return nullptr;
}

// Cursor-free history targets for the output-space cursor split. Sampled by
// every generation shader (as previousReal/currentReal) and by the second
// composite pass, written as a storage image by the first pass, and copied by
// the held-out colour probe, so: storage + sampled + transferSrc. Never
// flippable — a cursor-free frame is a prediction input, not a scanout buffer.
// bFramegenShared is mandatory: the framegen queue may live on another family
// (see the split-family rules around framegen_create_base_history_texture).
static bool framegen_create_cursor_history_texture( gamescope::OwningRc<CVulkanTexture> *ppTexture, uint32_t width, uint32_t height, uint32_t drmFormat )
{
	CVulkanTexture::createFlags createFlags;
	createFlags.bStorage = true;
	createFlags.bSampled = true;
	createFlags.bTransferSrc = true;
	createFlags.bFramegenShared = true;

	*ppTexture = new CVulkanTexture();
	return ( *ppTexture )->BInit( width, height, 1u, drmFormat, createFlags );
}

// Four is the smallest depth that covers the steady state plus one frame of
// slack: two images are held as previousReal/currentReal, one may still be
// pinned as a read input of an in-flight generation batch that history has
// already moved past, and one is being written by this composite. If they are
// somehow all busy the caller simply does not split this frame — the composite
// then bakes the cursor exactly as it does today, and PendingGenerated_t
// ::bCursorFree keeps the present side in step.
static constexpr size_t k_nFramegenCursorHistoryPool = 4;

static gamescope::Rc<CVulkanTexture> framegen_acquire_cursor_history_image( uint32_t width, uint32_t height, uint32_t drmFormat )
{
	if ( g_output.framegenCursorHistoryImages.size() != k_nFramegenCursorHistoryPool )
		g_output.framegenCursorHistoryImages.resize( k_nFramegenCursorHistoryPool );

	for ( auto &pImage : g_output.framegenCursorHistoryImages )
	{
		if ( framegen_output_matches( pImage, width, height, drmFormat ) )
			continue;
		// A live image of the wrong size can still be referenced by history or
		// an in-flight batch; dropping this slot's owning ref is safe (the Rc
		// holders keep it alive), the reallocation below just gives us a fresh
		// one for the new geometry.
		if ( !framegen_create_cursor_history_texture( &pImage, width, height, drmFormat ) )
		{
			g_output.framegenCursorHistoryImages.clear();
			return nullptr;
		}
	}

	return framegen_acquire_pool_image( g_output.framegenCursorHistoryImages,
		g_framegenHistory.nNextCursorHistoryIndex );
}

// Two-layer present-time composite used by both halves of the output-space
// cursor split: an already-output-space base image with the live cursor drawn
// on top.
//
// HDR contract. Layer 0 is the finished composite (real frame) or the
// generated frame, both of which already left the full shaper + 3D LUT + output
// transfer function pipeline. It is therefore declared PASSTHRU (the shader's
// apply_layer_color_mgmt returns immediately, and the passthru degamma is the
// identity) and applyOutputColorMgmt is cleared so encodeOutputColor is the
// identity too — the output encoding is applied exactly once, never twice, and
// no value is dragged through a transfer function it has already been through.
// The cursor layer is copied VERBATIM (colorspace, ctm, alpha mode, opacity,
// filter, scale/offset) and the frame's shaper/3D LUTs stay bound, so it gets
// precisely the colour management a single-pass composite would give it. The
// one difference from single-pass is that the cursor's own alpha is blended in
// output space instead of blend space: nothing is clipped and no range is lost
// (that is what would degrade HDR), only the ramp of a semi-transparent cursor
// edge differs — and it differs the same way on real and generated frames,
// which is what matters for temporal consistency. It is also what a hardware
// cursor plane does on KMS, where the cursor blends after the CRTC pipeline.
static void framegen_build_cursor_overlay_frame_info( struct FrameInfo_t *pOut,
	const gamescope::Rc<CVulkanTexture> &pBase,
	const struct FrameInfo_t::Layer_t &cursorLayer,
	const struct FrameInfo_t *pSource )
{
	*pOut = FrameInfo_t{};
	pOut->applyOutputColorMgmt = false;
	pOut->outputEncodingEOTF = pSource->outputEncodingEOTF;
	pOut->allowVRR = pSource->allowVRR;
	for ( uint32_t i = 0; i < EOTF_Count; i++ )
	{
		pOut->shaperLut[ i ] = pSource->shaperLut[ i ];
		pOut->lut3D[ i ] = pSource->lut3D[ i ];
	}

	pOut->layerCount = 2;

	FrameInfo_t::Layer_t &base = pOut->layers[ 0 ];
	base.tex = pBase;
	base.zpos = g_zposBase;
	base.scale = vec2_t{ 1.0f, 1.0f };
	base.offset = vec2_t{ 0.0f, 0.0f };
	base.opacity = 1.0f;
	base.filter = GamescopeUpscaleFilter::NEAREST;
	base.blackBorder = false;
	base.applyColorMgmt = false;
	base.colorspace = GAMESCOPE_APP_TEXTURE_COLORSPACE_PASSTHRU;
	base.eAlphaBlendingMode = ALPHA_BLENDING_MODE_PREMULTIPLIED;

	pOut->layers[ 1 ] = cursorLayer;
	// Compositor-owned layers carry no commit back-pointers; the import staging
	// pass keys off these, and a stale pointer here would be a lifetime bug.
	pOut->layers[ 1 ].pCommitTexture = nullptr;
	pOut->layers[ 1 ].pStagedCopyCount = nullptr;
}

// Second half of the output-space cursor split: draw the CURRENT cursor onto a
// generated frame at present time. Mirrors framegen_base_present_composite —
// same present-image pool, same realtime-queue submission, same host wait
// before the image can be flipped — and, critically, the same pOutputOverride
// contract: a non-null override keeps this composite out of
// framegen_record_real_frame (no history poisoning) and off the output ring.
// Returns the generated frame unchanged when there is nothing to draw, so a
// frame with no cursor costs literally nothing.
static gamescope::Rc<CVulkanTexture> framegen_cursor_present_composite( gamescope::Rc<CVulkanTexture> pGeneratedOutput, uint64_t ulFramegenSeqNo, const struct FrameInfo_t *pPresentFrameInfo )
{
	const int nCursorLayer = framegen_cursor_top_layer_index( pPresentFrameInfo );
	if ( nCursorLayer < 0 )
		return pGeneratedOutput;

	if ( !framegen_ensure_present_pool() || g_output.framegenPresentImages.empty()
		|| g_output.framegenPresentImages[ 0 ] == nullptr )
		return pGeneratedOutput;

	// The present pool tracks the CURRENT output; a generated frame planned
	// before a mode switch no longer matches it. Presenting the raw generated
	// frame is what happens today, so that is the safe answer.
	if ( g_output.framegenPresentImages[ 0 ]->width() != pGeneratedOutput->width()
		|| g_output.framegenPresentImages[ 0 ]->height() != pGeneratedOutput->height() )
		return pGeneratedOutput;

	gamescope::Rc<CVulkanTexture> pTarget = framegen_acquire_pool_image(
		g_output.framegenPresentImages, g_framegenHistory.nNextPresentIndex );
	if ( pTarget == nullptr )
	{
		static uint64_t s_uCursorPoolPressureDebugLogCounter = 0;
		if ( FramegenDebugShouldLog( s_uCursorPoolPressureDebugLogCounter ) )
			vk_log.infof( "framegen: cursor late-composite pool pressure pool=%zu", g_output.framegenPresentImages.size() );
		return pGeneratedOutput;
	}

	FrameInfo_t cursorFrameInfo;
	framegen_build_cursor_overlay_frame_info( &cursorFrameInfo, pGeneratedOutput,
		pPresentFrameInfo->layers[ nCursorLayer ], pPresentFrameInfo );

	auto pCmdBuffer = g_device.commandBuffer();
	g_device.addFramegenDependency( pCmdBuffer.get(), ulFramegenSeqNo );
	std::optional<uint64_t> oSeqNo = vulkan_composite( &cursorFrameInfo, nullptr, false, pTarget, false, std::move( pCmdBuffer ) );
	if ( !oSeqNo )
		return pGeneratedOutput;
	// Same wait the real composition path performs before its flip: the commit
	// must never scan out a half-written image. One two-layer full-screen blit.
	vulkan_wait( *oSeqNo, true );

	static uint64_t s_uCursorCompositeDebugLogCounter = 0;
	if ( FramegenDebugShouldLog( s_uCursorCompositeDebugLogCounter ) )
		vk_log.infof( "framegen: cursor late composite %ux%u",
			pTarget->width(), pTarget->height() );

	return pTarget;
}

// #02 late overlay composite: turn a generated BASE frame into a scanout-ready
// output image by running it through the same composite pipeline a real frame
// uses — FSR EASU/RCAS (or NIS/blit), shaper + 3D LUTs, and every CURRENT
// overlay layer blended on top, cursor at its latest position. Overlays are
// therefore pixel-perfect on generated frames: never extrapolated, always
// re-composited fresh. Runs on the realtime queue at present time like any
// composite; the vblank pacing already budgets for compositing
// (UpdateWasCompositing), and pOutputOverride both keeps this composite out of
// framegen_record_real_frame (no history poisoning) and off the output ring.
static gamescope::Rc<CVulkanTexture> framegen_base_present_composite( gamescope::Rc<CVulkanTexture> pGeneratedBase, uint64_t ulFramegenSeqNo, const struct FrameInfo_t *pPresentFrameInfo )
{
	if ( pPresentFrameInfo == nullptr || pPresentFrameInfo->layerCount < 1
		|| pPresentFrameInfo->layers[ 0 ].tex == nullptr
		|| pPresentFrameInfo->layers[ 0 ].tex->width() != pGeneratedBase->width()
		|| pPresentFrameInfo->layers[ 0 ].tex->height() != pGeneratedBase->height() )
	{
		// The live frame's base no longer matches the prediction (the paint
		// config changed between plan and present). The supersede/invalidate
		// paths normally catch this first; dropping to a hardware repeat is
		// the safe answer if one slips through.
		static uint64_t s_uMismatchDebugLogCounter = 0;
		if ( FramegenDebugShouldLog( s_uMismatchDebugLogCounter ) )
			vk_log.infof( "framegen: discarded generated frame reason=base_config_mismatch" );
		return nullptr;
	}

	if ( !framegen_ensure_present_pool() )
		return nullptr;

	// The parent compositor or KMS may retain more than a fixed number of old
	// commits. Select by actual Vulkan/backend ownership rather than blindly
	// cycling the three-image pool; if all targets are acquired, repeating the
	// last scanout is preferable to compositing into a live buffer.
	gamescope::Rc<CVulkanTexture> pTarget = framegen_acquire_pool_image(
		g_output.framegenPresentImages, g_framegenHistory.nNextPresentIndex );
	if ( pTarget == nullptr )
	{
		static uint64_t s_uPresentPressureDebugLogCounter = 0;
		if ( FramegenDebugShouldLog( s_uPresentPressureDebugLogCounter ) )
			vk_log.infof( "framegen: late-composite pool pressure pool=%zu", g_output.framegenPresentImages.size() );
		return nullptr;
	}

	FrameInfo_t generatedFrameInfo = *pPresentFrameInfo;
	generatedFrameInfo.layers[ 0 ].tex = pGeneratedBase;

	auto pCmdBuffer = g_device.commandBuffer();
	g_device.addFramegenDependency( pCmdBuffer.get(), ulFramegenSeqNo );
	std::optional<uint64_t> oSeqNo = vulkan_composite( &generatedFrameInfo, nullptr, false, pTarget, false, std::move( pCmdBuffer ) );
	if ( !oSeqNo )
		return nullptr;
	// Same wait the real composition path performs before its flip: the commit
	// must never scan out a half-written image. EASU/RCAS + overlays is well
	// under a millisecond on anything that runs the FSR path at this
	// resolution in the first place.
	vulkan_wait( *oSeqNo, true );

	static uint64_t s_uLateCompositeDebugLogCounter = 0;
	if ( FramegenDebugShouldLog( s_uLateCompositeDebugLogCounter ) )
		vk_log.infof( "framegen: late composite base=%ux%u output=%ux%u layers=%d fsr=%d",
			pGeneratedBase->width(), pGeneratedBase->height(),
			pTarget->width(), pTarget->height(),
			pPresentFrameInfo->layerCount,
			pPresentFrameInfo->useFSRLayer0 ? 1 : 0 );

	return pTarget;
}

// Refresh base history for a new real frame (#02): rotate the two owned images
// and copy the client's base buffer into the older one, on the framegen queue
// so the real frame's composite and present never wait on it. Same-queue
// ALL_COMMANDS barriers order this copy against the previous generation
// batch's reads of the target (WAR) and order the next batch's reads against
// the copy (RAW), across command buffers. The copy uses no descriptors and no
// timestamp slots, so it is exempt from the one-batch-in-flight machinery and
// must NOT bump lastGeneratedSeqNo (the headroom gate would see a perpetually
// busy queue). Its separate lastFramegenWorkSeqNo still makes reset lifetime-
// safe. The client texture is only referenced by this immediately submitted
// copy — never retained across frames — so its commit-keyed buffer lifetime
// is respected.
static bool framegen_base_record_copy( gamescope::Rc<CVulkanTexture> pBaseFrame, uint64_t ulCompositeSeqNo )
{
	const uint32_t nTarget = g_framegenHistory.nBaseHistoryNext & 1u;
	gamescope::Rc<CVulkanTexture> pTarget = g_framegenHistory.baseHistory[ nTarget ];
	if ( pTarget == nullptr )
		return false;

	// pBaseFrame is the CLIENT's imported dma-buf. On the split-family path the
	// framegen queue lives on a family that must never touch imported images
	// (that is exactly what the Intel interop quirk is about, and the FOREIGN
	// acquire barriers only ever transfer ownership to the compositor family),
	// so record the ingest copy on the composite queue instead. Only the
	// gamescope-owned baseHistory target is then read cross-family, and that one
	// is VK_SHARING_MODE_CONCURRENT. The cost is one copy back on the composite
	// queue; the generation batch itself still runs on the framegen queue.
	const bool bSplitFamily = g_device.framegenFamilySplit();

	auto pCmdBuffer = g_device.commandBuffer( !bSplitFamily );
	if ( !bSplitFamily )
		pCmdBuffer->markFramegen();
	pCmdBuffer->copyImage( std::move( pBaseFrame ), pTarget );
	// The real composite carries the client's acquire dependency. Waiting for
	// its timeline point makes that readiness chain visible to this queue before
	// it reads the same client image, and also orders the composite's image-state
	// transitions ahead of the copy. On the composite queue that ordering is
	// implicit (in-order submission on the same queue).
	if ( bSplitFamily )
	{
		// Write-after-read across queues: this copy overwrites the older history
		// slot, which the last generation batch on the framegen queue sampled as
		// previousReal. Nothing else orders the two queues, so take an explicit
		// framegen-timeline wait. Zero-cost elsewhere - the non-split path never
		// reaches here, and addFramegenDependency is a no-op without a dedicated
		// queue.
		if ( g_framegenHistory.lastFramegenWorkSeqNo != 0 )
			g_device.addFramegenDependency( pCmdBuffer.get(), g_framegenHistory.lastFramegenWorkSeqNo );
		g_framegenHistory.lastBaseIngestSeqNo = g_device.submit( std::move( pCmdBuffer ) );
	}
	else
		g_framegenHistory.lastFramegenWorkSeqNo =
			g_device.submitFramegen( std::move( pCmdBuffer ), ulCompositeSeqNo, -1, 0, 0 );

	g_framegenHistory.nBaseHistoryNext = nTarget ^ 1u;
	g_framegenHistory.previousReal = g_framegenHistory.currentReal;
	g_framegenHistory.currentReal = pTarget;
	return true;
}

static bool framegen_is_float_drm_format( uint32_t drmFormat )
{
	// Float (scRGB) targets carry HDR highlights above 1.0 and wide-gamut
	// negatives; fp16 arithmetic can band those, so the extrapolate shader stays
	// fp32 for them (see the fp16 shader's precision note); 32F formats are float
	// targets a fortiori. 16-bit UNORM targets need the same treatment for a
	// different reason: fp16's 11-bit
	// mantissa cannot represent 16-bit-deep content, so the fp16 path would
	// band it. Unreachable in output-space mode (scanout formats are
	// 8/10-bit), but base-layer mode (#02) generates in the CLIENT's format,
	// and 16-bit UNORM swapchains do occur (e.g. the NVIDIA WSI path here).
	switch ( drmFormat )
	{
		case DRM_FORMAT_ABGR32323232F:
		case DRM_FORMAT_ABGR16161616F:
		case DRM_FORMAT_XBGR16161616F:
		case DRM_FORMAT_ABGR16161616:
		case DRM_FORMAT_XBGR16161616:
			return true;
		default:
			return false;
	}
}

// Bind the two real-frame history textures into sampler slots 0 (previous) and
// 1 (current) with bilinear, normalized, sRGB-alias sampling — the common setup
// for every generation shader.
static void framegen_bind_history( CVulkanCmdBuffer *pCmdBuffer )
{
	pCmdBuffer->bindTexture( 0, g_framegenHistory.previousReal );
	pCmdBuffer->setTextureSrgb( 0, true );
	pCmdBuffer->setSamplerUnnormalized( 0, false );
	pCmdBuffer->setSamplerNearest( 0, false );
	pCmdBuffer->bindTexture( 1, g_framegenHistory.currentReal );
	pCmdBuffer->setTextureSrgb( 1, true );
	pCmdBuffer->setSamplerUnnormalized( 1, false );
	pCmdBuffer->setSamplerNearest( 1, false );
}

static void framegen_bind_extrapolate( CVulkanCmdBuffer *pCmdBuffer, ShaderType shader, const gamescope::Rc<CVulkanTexture> &pTarget, float flStrength )
{
	// The extrapolate variants read the effective per-slot coefficient from push
	// constants (no upload arena, so this is safe on the dedicated framegen queue).
	pCmdBuffer->pushConstants<FramegenPushData_t>( flStrength, k_flFramegenSuppressLo, k_flFramegenSuppressHi );

	pCmdBuffer->bindPipeline( g_device.pipeline( shader ) );
	pCmdBuffer->bindTarget( pTarget );
	framegen_bind_history( pCmdBuffer );

	const int pixelsPerGroup = 8;
	pCmdBuffer->dispatch( div_roundup( pTarget->width(), pixelsPerGroup ), div_roundup( pTarget->height(), pixelsPerGroup ) );
}

// Paired extrapolation: one dispatch writes two generated frames, sharing the
// two full-resolution history reads instead of repeating them per slot.
static void framegen_bind_extrapolate_pair( CVulkanCmdBuffer *pCmdBuffer, ShaderType shader,
	const gamescope::Rc<CVulkanTexture> &pTarget0, const gamescope::Rc<CVulkanTexture> &pTarget1,
	float flStrength0, float flStrength1 )
{
	pCmdBuffer->pushConstants<FramegenPairPushData_t>( flStrength0, flStrength1, k_flFramegenSuppressLo, k_flFramegenSuppressHi );

	pCmdBuffer->bindPipeline( g_device.pipeline( shader ) );
	pCmdBuffer->bindTarget( pTarget0 );
	pCmdBuffer->bindTarget2( pTarget1 );
	framegen_bind_history( pCmdBuffer );

	const int pixelsPerGroup = 8;
	pCmdBuffer->dispatch( div_roundup( pTarget0->width(), pixelsPerGroup ), div_roundup( pTarget0->height(), pixelsPerGroup ) );
}

// Blend mode (debug): crossfade the two real frames by this slot's temporal
// placement, so x3/x4 emits graded frames rather than identical 0.5 duplicates.
static void framegen_bind_blend( CVulkanCmdBuffer *pCmdBuffer, const gamescope::Rc<CVulkanTexture> &pTarget, float flPhase )
{
	pCmdBuffer->pushConstants<FramegenBlendPushData_t>( flPhase );

	pCmdBuffer->bindPipeline( g_device.pipeline( SHADER_TYPE_FRAMEGEN_BLEND ) );
	pCmdBuffer->bindTarget( pTarget );
	framegen_bind_history( pCmdBuffer );

	const int pixelsPerGroup = 8;
	pCmdBuffer->dispatch( div_roundup( pTarget->width(), pixelsPerGroup ), div_roundup( pTarget->height(), pixelsPerGroup ) );
}

// ---- Learned field refinement (Stage C) --------------------------------------
// A tiny convolutional net (12->16->16->4, 3x3 kernels, ~4.6k parameters)
// refines the causal checked motion field once per real frame: a bounded flow
// residual (at most +-2 field texels, tanh-limited in the shader), additive
// confidence recalibration, and a zero-neutral shading-persistence focus.
// Bidir runs the same network in both directions, but by default treats it as
// a conservative confidence veto: endpoint photometric supervision cannot
// prove that a learned vector is a valid in-between trajectory. The checked
// geometry therefore stays untouched and confidence can only go down.
// The causal path predicts corrections on top of the
// Stage-B field — a zero-initialized head IS Stage B, so the failure floor is
// the current behavior. The fourth head never predicts pixels; Guided uses it
// to gate a separately bounded analytic color trend. B4 closes the field safety loop:
// the stats probe grades the REFINED field, so a net that mispredicts is
// clamped by the same-batch trust factor and shows up in the adapt log lines.
// Trained offline, self-supervised, on tensors captured by the recorder below
// (scripts/framegen-net-train.py); enabled by pointing GAMESCOPE_FRAMEGEN_NET
// at a weights blob. The forward path already computes both checked fields for
// consistency, so learned prediction does not require bidir or its latency.
static const char *framegen_net_weights_path()
{
	static const char *s_pszPath = gamescope::framegen::non_empty_setting(
		getenv( "GAMESCOPE_FRAMEGEN_NET" ) );
	return s_pszPath;
}

// Blob layout (little-endian, written by the trainer): magic 'GSFR', version,
// layer count, then per-layer (c_in, c_out, k) dims, then all fp32 weights in
// [c_out][ky][kx][c_in] order (c_in contiguous, for the shader's vec4 dots)
// followed by the biases, layer by layer — the exact flat order the shader
// indexes. The version field tracks feature/training semantics. V3 activates
// the formerly reserved fourth output as a shading-persistence focus gate.
// Older blobs keep the same tensor layout and are accepted only after that
// unconstrained legacy row is explicitly zeroed; incompatible future layouts
// must fail validation rather than run with silently mismatched channels.
// The numeric CPU contract is centralized in framegen/net_layout.hpp.

// In-situ learning (C2): GAMESCOPE_FRAMEGEN_NET_ONLINE=1 keeps training the
// refiner on the framegen GPU while it serves — every real frame is a fresh
// labeled example, so the model tracks the current scene instead of the
// content some offline blob was fit to. Works with or without a starting
// blob (the prior is then a neutral zero-head initialization).
// GAMESCOPE_FRAMEGEN_NET_PROFILE=<path> makes the learning persistent
// per-game: loaded as the prior at startup when the file exists, and the
// served weights are written back there periodically while training.
static bool framegen_net_online_enabled()
{
	static const bool s_bEnabled = env_to_bool( getenv( "GAMESCOPE_FRAMEGEN_NET_ONLINE" ) );
	return s_bEnabled;
}

static float framegen_net_online_lr()
{
	static const float s_flLr = []()
	{
		const auto value = gamescope::framegen::parse_finite_float_setting(
			getenv( "GAMESCOPE_FRAMEGEN_NET_LR" ) );
		return value.has_value() && *value > 0.0f && *value <= 0.1f
			? *value : 3e-4f;
	}();
	return s_flLr;
}

static uint32_t framegen_net_online_every()
{
	static const uint32_t s_uEvery = gamescope::framegen::parse_uint32_setting(
		getenv( "GAMESCOPE_FRAMEGEN_NET_EVERY" ), false ).value_or( 1u );
	return s_uEvery;
}

static const char *framegen_net_profile_path()
{
	static const char *s_pszPath = gamescope::framegen::non_empty_setting(
		getenv( "GAMESCOPE_FRAMEGEN_NET_PROFILE" ) );
	return s_pszPath;
}

// Served-weights EMA, decay-toward-prior and profile-dump cadence. The decay
// gives the fast weights a ~2000-step (<1 minute) memory horizon: the model
// can only stay away from its safe prior by continually re-earning the
// distance on fresh frames — "a model that works temporarily for the scene".
static constexpr uint32_t k_uFramegenNetTrainTiles = 16;   // gradient tiles per step (split across directions)
static constexpr float k_flFramegenNetEmaAlpha = 0.0625f;  // served = EMA_1/16(fast)
static constexpr float k_flFramegenNetDecay = 5e-4f;       // per-step pull toward the prior
static constexpr uint32_t k_uFramegenNetProfileInterval = 1024; // steps between profile checkpoints

// The served weights, cached CPU-side from the last completed readback. File
// scope, NOT part of g_framegenMotion: it must survive vulkan_framegen_reset
// (and Vulkan teardown entirely) so unsaved learning can still be flushed to
// the profile when a session or a mode/resolution config ends. Progress is a
// trained-step counter that — unlike uNetTrainStep, which restarts with each
// state re-init — increases monotonically for the whole process, so the save
// cadence and the "anything unsaved?" checks survive resets.
static std::vector<float> g_framegenNetLiveWeights;
// Ping-pong with the live snapshot after validation. Online learning therefore
// allocates at most two 18.6 kB vectors instead of allocating and freeing one
// on every completed real-frame training step.
static std::vector<float> g_framegenNetReadbackWeights;
static bool g_bFramegenNetProfileLoaded = false;
static uint64_t g_ulFramegenNetProgress = 0;                     // trained steps, all-time
static uint64_t g_ulFramegenNetLiveProgress = 0;                 // progress at the cached readback
static std::atomic<uint64_t> g_ulFramegenNetSavedProgress = { 0 }; // progress at the last successful write
static std::atomic<bool> g_bFramegenNetWriteInFlight = { false };
// Periodic profile I/O is off the render thread, but the worker remains owned:
// reset/exit joins it before touching the same temp path or destroying process
// state. A detached writer plus a bounded poll can outlive a forced shutdown
// and race libc/static teardown.
static std::thread g_framegenNetWriteThread;

static bool framegen_net_parse_blob( const char *pszPath, std::vector<float> &weights, bool bLogMissing )
{
	FILE *pFile = fopen( pszPath, "rb" );
	if ( pFile == nullptr )
	{
		if ( bLogMissing )
			vk_log.errorf( "framegen: net weights '%s' unreadable", pszPath );
		return false;
	}

	gamescope::framegen::NetProfileMetadata metadata = {};
	bool bOk = fread( metadata.data(), sizeof( uint32_t ), metadata.size(), pFile ) == metadata.size();
	const uint32_t uVersion = bOk
		? gamescope::framegen::net_profile_metadata_version( metadata ) : 0u;
	bOk = uVersion != 0u;
	if ( bOk )
	{
		weights.resize( k_uFramegenNetFloats );
		bOk = fread( weights.data(), sizeof( float ), weights.size(), pFile ) == weights.size();
		bOk = bOk && gamescope::framegen::validate_and_migrate_net_profile_weights(
			uVersion, weights );
	}
	fclose( pFile );

	if ( !bOk )
	{
		weights.clear();
		vk_log.errorf( "framegen: net weights '%s' malformed (want 3 finite fp32 conv layers 12->16->16->4, k=3)", pszPath );
	}
	else if ( uVersion < k_uFramegenNetVersion )
	{
		vk_log.infof( "framegen: net weights '%s' use legacy v%u training semantics; accepting as a bounded prior under v%u with the formerly-reserved shading head zeroed",
			pszPath, uVersion, k_uFramegenNetVersion );
	}
	return bOk;
}

// The refiner's starting weights, first match wins: a saved per-game profile,
// the GAMESCOPE_FRAMEGEN_NET blob, or (online mode only) a synthesized
// neutral init — He-random hidden layers, zero head, i.e. exactly Stage B
// until learning moves it. Empty = refiner disabled.
static const std::vector<float> &framegen_net_weights()
{
	static const std::vector<float> s_weights = []() -> std::vector<float>
	{
		std::vector<float> weights;
		if ( framegen_net_profile_path() != nullptr && framegen_net_parse_blob( framegen_net_profile_path(), weights, false ) )
		{
			g_bFramegenNetProfileLoaded = true;
			vk_log.infof( "framegen: net prior loaded from profile '%s'", framegen_net_profile_path() );
			return weights;
		}
		if ( framegen_net_weights_path() != nullptr )
		{
			if ( framegen_net_parse_blob( framegen_net_weights_path(), weights, true ) )
			{
				vk_log.infof( "framegen: net weights loaded from '%s' (%u floats)", framegen_net_weights_path(), k_uFramegenNetFloats );
				return weights;
			}
			// Malformed blob (already logged loudly). Online mode still gets
			// its neutral prior below; without online learning there is
			// nothing safe to serve, so the refiner stays disabled.
		}
		if ( framegen_net_online_enabled() )
		{
			weights.assign( k_uFramegenNetFloats, 0.0f );
			std::mt19937 rng( 7u );
			std::normal_distribution<float> dist1( 0.0f, std::sqrt( 2.0f / 108.0f ) );
			std::normal_distribution<float> dist2( 0.0f, std::sqrt( 2.0f / 144.0f ) );
			for ( uint32_t i = 0; i < k_uFramegenNetLayer1Weights; i++ )
				weights[ i ] = dist1( rng );
			for ( uint32_t i = k_uFramegenNetLayer2Offset;
				i < k_uFramegenNetLayer2Offset + k_uFramegenNetLayer2Weights; i++ )
				weights[ i ] = dist2( rng );
			vk_log.infof( "framegen: net starting from a neutral prior (no blob/profile); online learning will shape it" );
		}
		return weights;
	}();
	return s_weights;
}

static bool framegen_net_requested( GamescopeFramegenPipeline ePipeline )
{
	if ( !framegen_net_lds_supported() )
	{
		static bool s_bLoggedUnsupported = false;
		if ( !s_bLoggedUnsupported && ( framegen_net_weights_path() != nullptr
			|| framegen_net_online_enabled() || framegen_net_profile_path() != nullptr ) )
		{
			vk_log.infof( "framegen: learned refinement requires %u bytes of compute shared memory but the device exposes %u; falling back to Stage B",
				k_uFramegenMotionNetLdsBytes, g_device.maxComputeSharedMemorySize() );
			s_bLoggedUnsupported = true;
		}
		return false;
	}

	return ePipeline >= GamescopeFramegenPipeline::Learned
		&& !framegen_net_weights().empty();
}

// Dataset capture: GAMESCOPE_FRAMEGEN_RECORD=<dir> dumps the raw field-res
// training tensors (both lumas + both checked fields, pre-refinement,
// pre-trust) to one file per real frame, up to GAMESCOPE_FRAMEGEN_RECORD_MAX
// samples (default 1000 — mind the disk; ~0.6 MB per 1080p sample).
static const char *framegen_record_dir()
{
	static const char *s_pszDir = gamescope::framegen::non_empty_setting(
		getenv( "GAMESCOPE_FRAMEGEN_RECORD" ) );
	return s_pszDir;
}

static uint32_t framegen_record_max()
{
	static const uint32_t s_uMax = gamescope::framegen::parse_uint32_setting(
		getenv( "GAMESCOPE_FRAMEGEN_RECORD_MAX" ), false ).value_or( 1000u );
	return s_uMax;
}

static uint32_t g_uFramegenRecordCount = 0;

// The field the warps and the adaptation probe consume: the net's refined copy
// when the refiner ran this batch, the checked Stage-B field otherwise.
static const gamescope::OwningRc<CVulkanTexture> &framegen_motion_field()
{
	return g_framegenMotion.bNetActive ? g_framegenMotion.mvFieldNet : g_framegenMotion.mvField;
}

static const gamescope::OwningRc<CVulkanTexture> &framegen_motion_field_rev()
{
	return g_framegenMotion.bNetActive ? g_framegenMotion.mvFieldRevNet : g_framegenMotion.mvFieldRevChk;
}

static bool framegen_create_intermediate( gamescope::OwningRc<CVulkanTexture> *ppTexture, uint32_t width, uint32_t height, uint32_t drmFormat )
{
	CVulkanTexture::createFlags createFlags;
	createFlags.bStorage = true;
	createFlags.bSampled = true;
	// Dataset capture copies the lumas and fields out to mapped readbacks.
	// Predict and Guided also rotate the final field into a retained transfer destination.
	createFlags.bTransferSrc = framegen_record_dir() != nullptr
		|| g_eFramegenPipeline >= GamescopeFramegenPipeline::Predict;
	createFlags.bTransferDst = g_eFramegenPipeline >= GamescopeFramegenPipeline::Predict;
	createFlags.bFramegenShared = true;

	*ppTexture = new CVulkanTexture();
	return ( *ppTexture )->BInit( width, height, 1u, drmFormat, createFlags );
}

static bool framegen_create_luma_reservoir( gamescope::OwningRc<CVulkanTexture> *ppTexture,
	uint32_t width, uint32_t height, uint32_t drmFormat )
{
	CVulkanTexture::createFlags createFlags;
	createFlags.bSampled = true;
	createFlags.bTransferDst = true;
	createFlags.bFramegenShared = true;

	*ppTexture = new CVulkanTexture();
	return ( *ppTexture )->BInit( width, height, 1u, drmFormat, createFlags );
}

static bool framegen_format_supports_sampled_storage( uint32_t drmFormat )
{
	VkFormat format = DRMFormatToVulkan( drmFormat, false );
	if ( format == VK_FORMAT_UNDEFINED )
		return false;

	VkFormatProperties props = {};
	g_device.vk.GetPhysicalDeviceFormatProperties( g_device.physDev(), format, &props );
	const VkFormatFeatureFlags needed = VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT | VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT;
	return ( props.optimalTilingFeatures & needed ) == needed;
}

static const FramegenDispatch_t &framegen_dispatch_for_format( uint32_t drmFormat )
{
	if ( g_framegenDispatch.drmFormat == drmFormat )
		return g_framegenDispatch;

	FramegenDispatch_t dispatch;
	dispatch.drmFormat = drmFormat;

	const bool bSupportsShaderFloat16 = g_device.supportsShaderFloat16();
	const bool bFloatOutput = framegen_is_float_drm_format( drmFormat );
	const bool bR16FLumaSupported = g_device.supportsStorageImageExtendedFormats()
		&& framegen_format_supports_sampled_storage( DRM_FORMAT_R16F );
	const bool bMotionSupported = framegen_format_supports_sampled_storage( DRM_FORMAT_ABGR16161616F );
	VkPhysicalDeviceProperties physProps = {};
	g_device.vk.GetPhysicalDeviceProperties( g_device.physDev(), &physProps );
	const gamescope::framegen::DispatchPolicy policy = gamescope::framegen::select_dispatch_policy(
		bSupportsShaderFloat16, bFloatOutput,
		bR16FLumaSupported, bMotionSupported, physProps.vendorID );

	dispatch.useFp16 = policy.useFp16;
	dispatch.extrapolate = policy.preferDirectExtrapolate
		? SHADER_TYPE_FRAMEGEN_EXTRAPOLATE_DIRECT
		: ( policy.useFp16 ? SHADER_TYPE_FRAMEGEN_EXTRAPOLATE_FP16 : SHADER_TYPE_FRAMEGEN_EXTRAPOLATE );
	// The direct-pair path has not been benchmark-qualified, so x3/x4 retains
	// the independently selected LDS variant and precision.
	dispatch.extrapolatePair = policy.pairUseFp16
		? SHADER_TYPE_FRAMEGEN_EXTRAPOLATE_PAIR_FP16 : SHADER_TYPE_FRAMEGEN_EXTRAPOLATE_PAIR;

	dispatch.useR16FLuma = policy.useR16FLuma;
	dispatch.motionLumaFormat = dispatch.useR16FLuma ? DRM_FORMAT_R16F : DRM_FORMAT_ABGR16161616F;
	dispatch.motionLumaPair = dispatch.useR16FLuma ? SHADER_TYPE_FRAMEGEN_MOTION_LUMA_PAIR : SHADER_TYPE_FRAMEGEN_MOTION_LUMA_PAIR_RGBA;
	dispatch.motionPyramidPair = dispatch.useR16FLuma ? SHADER_TYPE_FRAMEGEN_MOTION_PYRAMID : SHADER_TYPE_FRAMEGEN_MOTION_PYRAMID_RGBA;
	dispatch.motionSupported = policy.motionSupported;

	// LDS-vs-direct extrapolation is a memory-strategy choice that capability bits
	// cannot express: it turns on the GPU's texture-cache effectiveness, not on a
	// feature flag. The LDS apron (staging the current frame's neighbour cross into
	// shared memory) only pays off on cache-poor / bandwidth-starved parts; on
	// large-cache GPUs those neighbours are already cache hits, so the apron's
	// cooperative load + barrier is pure overhead. Measured on this tree: the
	// direct (no-LDS) fp32 shader is ~30-37% faster than either LDS variant on
	// NVIDIA and also beats fp16 there. So NVIDIA selects the direct shader; every
	// other vendor keeps the LDS path (and fp16 where capable) until benchmarked on
	// that part with the framegen microbench. This is the one selection that a
	// narrow vendor check earns — decided once here and cached, so no per-dispatch
	// cost. Extend the predicate as parts are measured.
	g_framegenDispatch = dispatch;
	if ( g_bFramegenDebug )
	{
		const char *pszExtrap = dispatch.extrapolate == SHADER_TYPE_FRAMEGEN_EXTRAPOLATE_DIRECT ? "fp32-direct"
			: ( dispatch.useFp16 ? "fp16-lds" : "fp32-lds" );
		vk_log.infof( "framegen: dispatch profile target=0x%" PRIX32 " extrapolate=%s motion=%s luma=0x%" PRIX32,
			drmFormat,
			pszExtrap,
			dispatch.motionSupported ? "yes" : "no",
			dispatch.motionLumaFormat );
	}

	return g_framegenDispatch;
}

// Forward-backward consistency (B2): also estimate the reverse (prev-anchored)
// flow and kill the confidence of forward vectors whose round trip through it
// does not close. Default on — it targets the disocclusion/mislock fizzle
// class directly; GAMESCOPE_FRAMEGEN_FB=0 restores the unchecked field for
// A/B comparison.
static bool framegen_fbcheck_enabled( GamescopeFramegenPipeline ePipeline )
{
	if ( ePipeline < GamescopeFramegenPipeline::Checked )
		return false;

	static const bool s_bEnabled = []()
	{
		const char *pszEnv = getenv( "GAMESCOPE_FRAMEGEN_FB" );
		return pszEnv == nullptr || env_to_bool( pszEnv );
	}();
	return s_bEnabled;
}

// Base round-trip tolerance in low-res texels (~4 full-res px of slack at the
// default downscale before any confidence loss). GAMESCOPE_FRAMEGEN_FB_TOL
// overrides it for tuning: larger = more forgiving = fewer kills.
static float framegen_fbcheck_tol_base()
{
	static const float s_flTolBase = []()
	{
		const char *pszEnv = getenv( "GAMESCOPE_FRAMEGEN_FB_TOL" );
		if ( pszEnv != nullptr && *pszEnv != '\0' )
		{
			const auto value = gamescope::framegen::parse_finite_float_setting( pszEnv );
			if ( value.has_value() )
				return std::clamp( *value, 0.05f, 8.0f );
			vk_log.errorf( "framegen: GAMESCOPE_FRAMEGEN_FB_TOL is not a finite number; using 0.75" );
		}
		return 0.75f;
	}();
	return s_flTolBase;
}
// Tolerance growth per texel of round-trip motion (quarter-pel quantization at
// both ends of long vectors is legitimate error, not inconsistency).
static constexpr float k_flFramegenFBTolSlope = 0.05f;

// Two-source agreement window for the warp (normalized gamma-encoded color
// distance between the current-frame and previous-frame projections of the
// same flow). Lo leaves room for sub-pixel resample differences on high-
// frequency detail; Hi is a clearly-different-content kill.
// GAMESCOPE_FRAMEGEN_AGREE=0 disables the test for A/B attribution.
static constexpr float k_flFramegenAgreeLo = 0.12f;
static constexpr float k_flFramegenAgreeHi = 0.45f;
static bool framegen_agreement_enabled( GamescopeFramegenPipeline ePipeline )
{
	if ( ePipeline < GamescopeFramegenPipeline::Checked )
		return false;

	static const bool s_bEnabled = []()
	{
		const char *pszEnv = getenv( "GAMESCOPE_FRAMEGEN_AGREE" );
		return pszEnv == nullptr || env_to_bool( pszEnv );
	}();
	return s_bEnabled;
}

// Guided-only three-frame disocclusion evidence. Default on as part of the
// Guided pipeline contract; the environment switch exists for live A/B cost
// and artifact attribution without changing any cheaper pipeline rung.
static bool framegen_reservoir_enabled( GamescopeFramegenPipeline ePipeline )
{
	if ( ePipeline != GamescopeFramegenPipeline::Guided )
		return false;

	static const bool s_bEnabled = []()
	{
		const char *pszEnv = getenv( "GAMESCOPE_FRAMEGEN_RESERVOIR" );
		return pszEnv == nullptr || env_to_bool( pszEnv );
	}();
	return s_bEnabled;
}

// Guided-only learned non-geometric color trend. The switch isolates the
// final correction for live A/B while leaving the field net and every queue /
// descriptor / timing decision identical.
static bool framegen_shading_enabled( GamescopeFramegenPipeline ePipeline )
{
	if ( ePipeline != GamescopeFramegenPipeline::Guided )
		return false;

	static const bool s_bEnabled = []()
	{
		const char *pszEnv = getenv( "GAMESCOPE_FRAMEGEN_SHADING" );
		return pszEnv == nullptr || env_to_bool( pszEnv );
	}();
	return s_bEnabled;
}

// ---- Self-supervised online adaptation (B4) ----------------------------------
// Every real frame is ground truth for the field just estimated from it. A
// field-res probe pass measures how well that field predicts the actual frame
// (per-texel warped-luma residual) and reduces the verdicts to a few counters.
// Two consumers:
//   GPU, same batch: the warps read a global "field trust" factor — the
//   fraction of texels that PASSED every consistency check yet still
//   mispredicted (regimes where the per-texel checks lie: lighting flashes,
//   particle chaos, stroboscopic content) — and fade the motion term toward
//   the safe fallback. Continuous, zero-latency, self-recovering: this is the
//   pipeline-driven degradation #04's monotonic ladder must not provide (a
//   discrete pipeline rung would either oscillate or, held monotonic, let one
//   bad interval degrade the whole scene).
//   CPU, next batch: the readback auto-calibrates the FB tolerance and the
//   agreement window to the content (see framegen_adapt_consume).
// Default on in motion mode; GAMESCOPE_FRAMEGEN_ADAPT=0 restores B3 behavior
// bit-exactly for A/B attribution.
static bool framegen_adapt_enabled( GamescopeFramegenPipeline ePipeline )
{
	if ( ePipeline < GamescopeFramegenPipeline::Learned )
		return false;

	static const bool s_bEnabled = []()
	{
		const char *pszEnv = getenv( "GAMESCOPE_FRAMEGEN_ADAPT" );
		return pszEnv == nullptr || env_to_bool( pszEnv );
	}();
	return s_bEnabled;
}

static const FramegenMetricsWindow_t &framegen_metrics_last_closed_window()
{
	static const FramegenMetricsWindow_t s_empty;
	if ( g_framegenMetrics.nClosedWindows == 0u )
		return s_empty;
	const size_t nLast = ( g_framegenMetrics.nNextWindow
		+ g_framegenMetrics.windows.size() - 1u ) % g_framegenMetrics.windows.size();
	return g_framegenMetrics.windows[nLast];
}

template <typename ValueFn>
static std::array<double, gamescope::framegen::k_nFramegenHudSparklineSamples>
framegen_metrics_window_history( ValueFn value )
{
	std::array<double, gamescope::framegen::k_nFramegenHudSparklineSamples> result = {};
	const size_t nCount = static_cast<size_t>( std::min<uint64_t>(
		g_framegenMetrics.nClosedWindows, g_framegenMetrics.windows.size() ) );
	const size_t nOldest = ( g_framegenMetrics.nNextWindow
		+ g_framegenMetrics.windows.size() - nCount )
		% g_framegenMetrics.windows.size();
	const size_t nDestination = result.size() - nCount;
	for ( size_t i = 0u; i < nCount; i++ )
	{
		result[nDestination + i] = value( g_framegenMetrics.windows[
			( nOldest + i ) % g_framegenMetrics.windows.size() ] );
	}
	return result;
}

static const char *framegen_hud_device_name()
{
	static const std::array<char, VK_MAX_PHYSICAL_DEVICE_NAME_SIZE> s_name = []()
	{
		std::array<char, VK_MAX_PHYSICAL_DEVICE_NAME_SIZE> result = {};
		VkPhysicalDeviceProperties properties = {};
		g_device.vk.GetPhysicalDeviceProperties( g_device.physDev(), &properties );
		std::snprintf( result.data(), result.size(), "%s", properties.deviceName );
		return result;
	}();
	return s_name.data();
}

static const CVulkanTexture *framegen_hud_base_client_texture( const struct FrameInfo_t *pFrameInfo )
{
	if ( pFrameInfo == nullptr )
		return nullptr;
	for ( int i = 0; i < pFrameInfo->layerCount; i++ )
	{
		const FrameInfo_t::Layer_t &layer = pFrameInfo->layers[i];
		if ( layer.zpos == g_zposBase && layer.pCommitTexture != nullptr )
		{
			const gamescope::Rc<CVulkanTexture> &pClientTexture = *layer.pCommitTexture;
			if ( pClientTexture != nullptr )
				return pClientTexture.get();
		}
	}
	return nullptr;
}

static gamescope::framegen::FramegenHudSnapshot_t framegen_hud_snapshot(
	const struct FrameInfo_t *pFrameInfo )
{
	const CVulkanTexture *pBaseClientTexture = framegen_hud_base_client_texture( pFrameInfo );
	const FramegenMetricsWindow_t &window = framegen_metrics_last_closed_window();
	const uint32_t uHitPercent = window.deadlineCount != 0u
		? static_cast<uint32_t>( std::min<uint64_t>( 100u,
			( window.deadlineHits * 100u + window.deadlineCount / 2u )
				/ window.deadlineCount ) )
		: 0u;
	const int nRefreshMilliHz = GetVBlankTimer().GetRefresh();
	const bool bNetActive = g_framegenMotion.bNetActive;
	const FramegenEffective_t effective = framegen_effective_config(
		g_framegenHistory.nDegradeSteps );
	const double flBiasTenths = std::clamp(
		g_framegenPresentState.displayTiming.presentBias.emaNs / 100'000.0,
		-9'999.0, 9'999.0 );
	const double flPacingSdTenths = std::clamp(
		window.flipIntervals.stddev() * 10.0, 0.0, 9'999.0 );
	const auto deadlineHitHistory = framegen_metrics_window_history(
		[]( const FramegenMetricsWindow_t &sample )
		{
			return sample.deadlineCount != 0u
				? static_cast<double>( sample.deadlineHits ) * 100.0
					/ static_cast<double>( sample.deadlineCount )
				: 0.0;
		} );
	const auto pacingSdMsHistory = framegen_metrics_window_history(
		[]( const FramegenMetricsWindow_t &sample )
		{
			return sample.flipIntervals.stddev();
		} );
	return {
		.version = gamescope::k_szGamescopeHudVersion,
		.deviceName = framegen_hud_device_name(),
		.otherDeviceName = g_device.framegenOtherDeviceName(),
		.renderOrigin = pBaseClientTexture != nullptr
			? pBaseClientTexture->renderOrigin() : nullptr,
		.mode = effective.mode,
		.pipeline = effective.pipeline,
		.multiplier = effective.multiplier,
		.refreshMilliHz = nRefreshMilliHz > 0
			? static_cast<uint32_t>( nRefreshMilliHz ) : 0u,
		.vrrRequested = framegen_vrr_hybrid_configured(),
		.vrrActive = vulkan_framegen_vrr_hybrid_active(),
		.bidirRequested = framegen_bidir_enabled(),
		.bidirActive = vulkan_framegen_bidir_active(),
		.baseLayer = vulkan_framegen_base_layer_active(),
		.clientBuffersStaged = pBaseClientTexture != nullptr
			&& pBaseClientTexture->deviceLocalStagingImage(),
		.netRequested = framegen_net_weights_path() != nullptr
			|| framegen_net_online_enabled() || framegen_net_profile_path() != nullptr,
		.netActive = bNetActive,
		.netOnline = bNetActive && framegen_net_online_enabled(),
		.adapt = effective.mode == GamescopeFramegenMode::Motion
			&& framegen_adapt_enabled( effective.pipeline ),
		.real = window.real,
		.delayedReal = window.delayedReal,
		.generated = window.generated,
		.repeats = window.repeats,
		.ladderSteps = g_framegenHistory.nDegradeSteps,
		.ladderMaxSteps = gamescope::framegen::max_degrade_steps(
			g_eFramegenMode, g_eFramegenPipeline, 2 ),
		.ladderHold = g_framegenHistory.nDegradeHold,
		.ladderRecoveryStreak = g_framegenHistory.recovery.streak,
		.ladderRecoveryBackoffDecisions =
			g_framegenHistory.recovery.backoffDecisions,
		.ladderRecoveryProbationRemaining =
			g_framegenHistory.recovery.probationRemaining,
		.ladderRecoveryDecisionsSinceClimb =
			g_framegenHistory.recovery.decisionsSinceClimb,
		.biasTenthsMs = static_cast<int32_t>( std::llround( flBiasTenths ) ),
		.deadlineHitPercent = uHitPercent,
		.pacingSdTenthsMs = static_cast<uint32_t>( std::llround( flPacingSdTenths ) ),
		.deadlineHitHistory = deadlineHitHistory,
		.pacingSdMsHistory = pacingSdMsHistory,
		.resets = window.resets,
		.ringResets = window.resetsRing,
		.netTrainedSteps = g_ulFramegenNetProgress,
		.netProfileLoaded = g_bFramegenNetProfileLoaded,
	};
}

static int framegen_hud_record( CVulkanCmdBuffer *pCmdBuffer,
	const gamescope::Rc<CVulkanTexture> &pTarget, const struct FrameInfo_t *pFrameInfo )
{
	const uint32_t uLevel = framegen_hud_level();
	if ( uLevel == 0u || pTarget == nullptr )
		return -1;

	const uint64_t ulNowNs = get_time_in_nanos();
	const bool bHdr = pFrameInfo->outputEncodingEOTF == EOTF_PQ;
	if ( !g_framegenHud.hdrValid || g_framegenHud.hdr != bHdr )
	{
		g_framegenHud.hdr = bHdr;
		g_framegenHud.hdrValid = true;
		g_framegenHud.dirty = true;
	}
	if ( !g_framegenHud.textValid || ulNowNs >= g_framegenHud.nextRebuildNs )
	{
		const gamescope::framegen::FramegenHudText_t text =
			gamescope::framegen::format_framegen_hud(
				uLevel, framegen_hud_snapshot( pFrameInfo ) );
		if ( !g_framegenHud.textValid || text != g_framegenHud.text )
		{
			g_framegenHud.text = text;
			g_framegenHud.dirty = true;
		}
		g_framegenHud.textValid = true;
		g_framegenHud.nextRebuildNs = ulNowNs + 1'000'000'000ull;
	}

	// Empty text is a CPU-side no-op: do not bind the pipeline or consume a
	// scratch descriptor merely to let the shader discover there is no work.
	if ( g_framegenHud.text.lineCount == 0u )
		return -1;

	if ( g_framegenHud.dirty )
	{
		int nUploadSlot = -1;
		for ( size_t i = 0u; i < g_framegenHud.uploadSlots.size(); i++ )
		{
			if ( static_cast<int>( i ) == g_framegenHud.activeUploadSlot )
				continue;
			const auto &candidate = g_framegenHud.uploadSlots[i];
			if ( candidate.lastSubmitSeqNo == 0u
				|| g_device.hasCompleted( candidate.lastSubmitSeqNo ) )
			{
				nUploadSlot = static_cast<int>( i );
				break;
			}
		}

		if ( nUploadSlot >= 0 )
		{
			auto &slot = g_framegenHud.uploadSlots[nUploadSlot];
			const gamescope::framegen::FramegenHudUniform_t uniform =
				gamescope::framegen::make_framegen_hud_uniform(
					g_framegenHud.text, g_framegenHud.hdr );
			memcpy( slot.pMapped, &uniform, sizeof( uniform ) );
			slot.logicalWidthPixels = uniform.widthChars * 8u + 8u;
			slot.logicalHeightPixels = uniform.lineCount * 8u + 8u;
			g_framegenHud.activeUploadSlot = nUploadSlot;
			g_framegenHud.dirty = false;
		}
	}

	if ( g_framegenHud.activeUploadSlot < 0 )
		return -1;
	const int nSlot = g_framegenHud.activeUploadSlot;
	const auto &slot = g_framegenHud.uploadSlots[nSlot];
	if ( slot.logicalWidthPixels == 0u || slot.logicalHeightPixels == 0u )
		return -1;
	const uint32_t uScale = framegen_hud_scale( pTarget->height() );
	const uint32_t uWidthPixels = slot.logicalWidthPixels * uScale;
	const uint32_t uHeightPixels = slot.logicalHeightPixels * uScale;
	const uint32_t uMargin = 8u * uScale;
	const FramegenHudCorner eCorner = framegen_hud_corner();
	const bool bRight = eCorner == FramegenHudCorner::TopRight
		|| eCorner == FramegenHudCorner::BottomRight;
	const bool bBottom = eCorner == FramegenHudCorner::BottomLeft
		|| eCorner == FramegenHudCorner::BottomRight
		|| eCorner == FramegenHudCorner::BottomCenter;
	uint32_t uOffsetX = pTarget->width() > uWidthPixels + uMargin ? uMargin : 0u;
	if ( eCorner == FramegenHudCorner::BottomCenter )
		uOffsetX = pTarget->width() > uWidthPixels
			? ( pTarget->width() - uWidthPixels ) / 2u : 0u;
	else if ( bRight )
		uOffsetX = pTarget->width() > uWidthPixels + uMargin
			? pTarget->width() - uWidthPixels - uMargin : 0u;
	const uint32_t uOffsetY = bBottom
		? ( pTarget->height() > uHeightPixels + uMargin
			? pTarget->height() - uHeightPixels - uMargin : 0u )
		: ( pTarget->height() > uHeightPixels + uMargin ? uMargin : 0u );

	pCmdBuffer->clearState();
	pCmdBuffer->bindPipeline( g_device.pipeline( SHADER_TYPE_FRAMEGEN_HUD ) );
	pCmdBuffer->pushConstants<gamescope::framegen::FramegenHudPush_t>(
		uOffsetX, uOffsetY, uScale );
	pCmdBuffer->bindTexture( 0u, pTarget );
	pCmdBuffer->setTextureSrgb( 0u, true );
	pCmdBuffer->setSamplerNearest( 0u, true );
	pCmdBuffer->bindTarget( pTarget );
	pCmdBuffer->bindUploadedConstants( slot.offset,
		sizeof( gamescope::framegen::FramegenHudUniform_t ) );
	pCmdBuffer->dispatch( div_roundup( uWidthPixels, 8u ),
		div_roundup( uHeightPixels, 8u ) );
	return nSlot;
}

static void framegen_hud_note_submit( int nSlot, uint64_t ulSeqNo )
{
	if ( nSlot >= 0 && static_cast<size_t>( nSlot ) < g_framegenHud.uploadSlots.size() )
		g_framegenHud.uploadSlots[nSlot].lastSubmitSeqNo = ulSeqNo;
}

// An explicit GAMESCOPE_FRAMEGEN_FB_TOL is a manual tuning decision; the
// auto-calibration keeps its hands off it.
static bool framegen_fbcheck_tol_pinned()
{
	static const bool s_bPinned = []()
	{
		const char *pszEnv = getenv( "GAMESCOPE_FRAMEGEN_FB_TOL" );
		return pszEnv != nullptr && *pszEnv != '\0';
	}();
	return s_bPinned;
}

FramegenMotionStatsPush_t::FramegenMotionStatsPush_t( bool bClearOnly )
	: clearOnly( bClearOnly ? 1u : 0u )
	, badThresh( gamescope::framegen::k_flAdaptationBadResidual )
	, staticMvMax( 0.25f )
	, minConfSurvive( 0.25f )
{
}

static float framegen_adapt_fbcheck_tol()
{
	if ( g_framegenHistory.adaptation.fbTolerance > 0.0f )
		return g_framegenHistory.adaptation.fbTolerance;
	return framegen_fbcheck_tol_base();
}

static float framegen_adapt_agree_lo()
{
	return gamescope::framegen::active_agreement_lo(
		g_framegenHistory.adaptation, k_flFramegenAgreeLo );
}

static float framegen_adapt_agree_hi()
{
	return gamescope::framegen::active_agreement_hi(
		g_framegenHistory.adaptation, k_flFramegenAgreeHi );
}

static float framegen_effective_fbcheck_tol( GamescopeFramegenPipeline ePipeline )
{
	return framegen_adapt_enabled( ePipeline )
		? framegen_adapt_fbcheck_tol() : framegen_fbcheck_tol_base();
}

static float framegen_effective_agree_lo( GamescopeFramegenPipeline ePipeline )
{
	return framegen_adapt_enabled( ePipeline )
		? framegen_adapt_agree_lo() : k_flFramegenAgreeLo;
}

static float framegen_effective_agree_hi( GamescopeFramegenPipeline ePipeline )
{
	return framegen_adapt_enabled( ePipeline )
		? framegen_adapt_agree_hi() : k_flFramegenAgreeHi;
}

// Parse the completed batch's stats readback and fold it into the adaptation
// EMAs, then derive the threshold values the next batch records with. Called
// at batch-planning time: the same hasCompletedFramegen() gate that admits a
// new batch guarantees the mapped memory is no longer being written.
static void framegen_adapt_consume( GamescopeFramegenPipeline ePipeline )
{
	if ( !framegen_adapt_enabled( ePipeline ) )
		return;

	FramegenHistory_t &h = g_framegenHistory;
	if ( h.ulAdaptStatsSeqNo == 0 || h.ulAdaptStatsSeqNo == h.ulAdaptConsumedSeqNo
		|| g_framegenMotion.statsReadback == nullptr
		|| g_framegenMotion.statsReadback->mappedData() == nullptr
		|| !g_device.hasCompletedFramegen( h.ulAdaptStatsSeqNo ) )
		return;
	h.ulAdaptConsumedSeqNo = h.ulAdaptStatsSeqNo;

	std::array<uint32_t, gamescope::framegen::k_uAdaptationStatsCount> stats;
	memcpy( stats.data(), g_framegenMotion.statsReadback->mappedData(), sizeof( stats ) );
	const auto measurement = gamescope::framegen::decode_adaptation_stats( stats );
	if ( !measurement.has_value() )
		return;

	const bool bSceneCut = measurement->sceneCut != 0u;
	if ( vulkan_framegen_bidir_active() )
	{
		const gamescope::framegen::BidirCutTransition_t cut =
			gamescope::framegen::observe_bidir_scene_cut(
				h.bidirCutEpisode, bSceneCut );
		h.bidirCutEpisode = cut.state;
		if ( cut.discardInterpolations )
		{
			// CPU readback arrives one batch later: discard every still-pending
			// interpolation and retain scheduled real endpoints. A classification
			// that remains asserted keeps suppressing pixels without repeatedly
			// tearing down the display-clock translation.
			std::erase_if( h.pending,
				[]( const FramegenHistory_t::PendingGenerated_t &entry ) {
					return !entry.bReal;
				} );
		}
		if ( cut.invalidateEpoch )
		{
			framegen_metrics_note_reset( FramegenResetReason_t::Cut );
			vk_log.infof( "framegen: content scene cut detected (%u/9 sections, histogram distance %.2f); presenting a real endpoint",
				measurement->changedSections,
				gamescope::framegen::scene_histogram_distance( *measurement ) );
			h.bidirEpoch = {};
			h.bidirFeedbackEndpoints.clear();
		}
	}
	else if ( bSceneCut )
	{
		framegen_metrics_note_reset( FramegenResetReason_t::Cut );
		vk_log.infof( "framegen: content scene cut detected (%u/9 sections, histogram distance %.2f); presenting a real endpoint",
			measurement->changedSections,
			gamescope::framegen::scene_histogram_distance( *measurement ) );
	}

	// Slow EMA (1/8), hysteretic mode selection, and bounded tolerance slew:
	// threshold moves must be calmer than the per-frame signal, or the
	// adaptation itself becomes a flicker source. FB tolerance loosens
	// ONLY on ambiguity-without-error — round trips fail
	// while the field demonstrably predicts the real frame (periodic textures:
	// fences, grilles, tiled detail, where many vectors are equally valid and
	// the kill would reintroduce fizzle). High round-trip error WITH high
	// residual is genuine mislocking and keeps the strict tolerance; the field
	// trust handles it. Agreement widens only from a measured temporal-noise
	// floor. The policy helper owns those arithmetic contracts; this function
	// retains readback lifetime, completion, and logging ownership.
	const float flTolBase = framegen_fbcheck_tol_base();
	const bool bTolPinned = framegen_fbcheck_tol_pinned();
	gamescope::framegen::update_adaptation_state(
		h.adaptation, *measurement, flTolBase, bTolPinned );

	static uint64_t s_uAdaptDebugLogCounter = 0;
	if ( FramegenDebugShouldLog( s_uAdaptDebugLogCounter ) )
	{
		vk_log.infof( "framegen: adapt resid=%.3f bad=%.1f%% killed=%.1f%% noise=%.4f fbP75=%.2f mv=%.1f scene=%u(%u/9 hist=%.2f) -> fbTol=%.2f relax=%s agree=%.2f/%.2f",
			measurement->residual, measurement->badFraction * 100.0f,
			measurement->killedFraction * 100.0f,
			h.adaptation.noiseEma, h.adaptation.fbP75Ema, measurement->motionMean,
			measurement->sceneCut, measurement->changedSections,
			gamescope::framegen::scene_histogram_distance( *measurement ),
			framegen_adapt_fbcheck_tol(),
			h.adaptation.fbRelaxationActive ? "on" : "off",
			framegen_adapt_agree_lo(), framegen_adapt_agree_hi() );
	}
}

// Dataset capture, CPU half: flush the completed batch's recorder readbacks
// to disk. Same race-freedom argument as framegen_adapt_consume — the
// hasCompletedFramegen() gate that admits a new batch guarantees the mapped
// memory is quiescent. Runs on the composite thread; one buffered ~0.6 MB
// write per real frame is capture-tool territory, not a production path.
// Capture stop: release the staging readbacks so a finished (or failed)
// capture holds no memory for the rest of the session. The GPU-side copies
// already stop via the count gate in framegen_submit_planned, the completion
// gate in the consumer means none are in flight when this runs, and the
// realloc path is gated on the count too — a later reset can't resurrect them.
static void framegen_record_release( const char *pszReason )
{
	g_framegenMotion.recLumaPrev = nullptr;
	g_framegenMotion.recLumaCur = nullptr;
	g_framegenMotion.recField = nullptr;
	g_framegenMotion.recFieldRev = nullptr;
	vk_log.infof( "framegen: dataset capture stopped (%s), readbacks released", pszReason );
}

// Capture I/O is deliberately out of line: this function is polled by the
// Vulkan batch recorder, and inlining its cold path bloats the production
// command-recording footprint even when no GSFD capture is configured.
static __attribute__((noinline)) void framegen_record_consume()
{
	FramegenHistory_t &h = g_framegenHistory;
	if ( h.ulNetRecordSeqNo == 0 || h.ulNetRecordSeqNo == h.ulNetRecordConsumedSeqNo
		|| g_framegenMotion.recField == nullptr
		|| !g_device.hasCompletedFramegen( h.ulNetRecordSeqNo ) )
		return;
	h.ulNetRecordConsumedSeqNo = h.ulNetRecordSeqNo;

	if ( g_uFramegenRecordCount >= framegen_record_max() )
		return; // belt-and-braces: copies stop and readbacks are freed at the cap

	// The trainer globs *.bin, so only publish a sample after every tightly
	// packed plane has reached a unique same-directory staging file.
	char szName[ 32 ];
	snprintf( szName, sizeof( szName ), "fg_%06u.bin", g_uFramegenRecordCount );
	std::string path = framegen_record_dir();
	path += '/';
	path += szName;
	gamescope::framegen::AtomicOutputFile output( path );
	if ( !output.is_open() )
	{
		vk_log.errorf( "framegen: dataset capture can't write '%s' (%s)",
			path.c_str(), strerror( output.error() ) );
		g_uFramegenRecordCount = ~0u; // trip every future count gate
		framegen_record_release( "write failure" );
		return;
	}

	// Header + tightly packed planes (rows re-packed from the readbacks'
	// rowPitch): lumaPrev, lumaCur, field (rgba16f), fieldRev (rgba16f).
	// The trainer (scripts/framegen-net-train.py) parses exactly this.
	const uint32_t uLumaBpp = g_framegenMotion.lumaFormat == DRM_FORMAT_R16F ? 2u : 8u;
	const uint32_t uHeader[ 8 ] = {
		0x44465347u /* 'GSFD' */, 1u,
		g_framegenMotion.width, g_framegenMotion.height,
		uLumaBpp, 8u /* field bpp */, 1u /* flags: has reverse field */, 0u,
	};
	const uint64_t ulSeq = h.ulNetRecordConsumedSeqNo;
	bool bOk = output.write( uHeader, sizeof( uHeader ) )
		&& output.write( &ulSeq, sizeof( ulSeq ) );

	const auto writePlane = [&]( CVulkanTexture *pTex, uint32_t uBytesPerRow )
	{
		if ( pTex == nullptr || pTex->rowPitch() < uBytesPerRow
			|| pTex->width() != g_framegenMotion.width
			|| pTex->height() != g_framegenMotion.height )
			return false;
		const uint8_t *pData = pTex->mappedData();
		if ( pData == nullptr )
			return false;
		for ( uint32_t y = 0; bOk && y < pTex->height(); y++ )
			bOk = output.write( pData + (size_t)y * pTex->rowPitch(), uBytesPerRow );
		return bOk;
	};
	bOk = bOk && writePlane( g_framegenMotion.recLumaPrev.get(), g_framegenMotion.width * uLumaBpp )
		&& writePlane( g_framegenMotion.recLumaCur.get(), g_framegenMotion.width * uLumaBpp )
		&& writePlane( g_framegenMotion.recField.get(), g_framegenMotion.width * 8u )
		&& writePlane( g_framegenMotion.recFieldRev.get(), g_framegenMotion.width * 8u );
	if ( bOk )
		bOk = output.commit();

	if ( !bOk )
	{
		const int error = output.error() != 0 ? output.error() : EIO;
		vk_log.errorf( "framegen: dataset capture write failed at '%s' (%s)",
			path.c_str(), strerror( error ) );
		g_uFramegenRecordCount = ~0u;
		framegen_record_release( "write failure" );
		return;
	}
	if ( g_uFramegenRecordCount == 0 )
		vk_log.infof( "framegen: dataset capture writing %ux%u field-res samples to '%s'", g_framegenMotion.width, g_framegenMotion.height, framegen_record_dir() );
	g_uFramegenRecordCount++;
	if ( g_uFramegenRecordCount >= framegen_record_max() )
		framegen_record_release( "sample cap reached" );
}

static void framegen_color_probe_release( const char *pszReason )
{
	for ( auto &pReadback : g_framegenColorProbe.generatedReadback )
		pReadback = nullptr;
	g_framegenColorProbe.referenceReadback = nullptr;
	g_framegenColorProbe.anchor = nullptr;
	g_framegenColorProbe.reference = nullptr;
	g_framegenColorProbe.lastRealTimeNs = 0;
	g_framegenColorProbe.pendingSeqNo = 0;
	g_framegenHistory.previousReal = nullptr;
	g_framegenHistory.currentReal = nullptr;
	g_framegenHistory.genReadA = nullptr;
	g_framegenHistory.genReadB = nullptr;
	g_framegenHistory.genReadReference = nullptr;
	g_framegenHistory.genReadSeqNo = 0;
	vk_log.infof( "framegen: full-colour held-out capture stopped (%s), readbacks released", pszReason );
}

static bool framegen_color_probe_prepare( uint32_t width, uint32_t height, uint32_t drmFormat, EOTF eotf )
{
	FramegenColorProbeResources_t &c = g_framegenColorProbe;
	if ( c.referenceReadback != nullptr && c.width == width && c.height == height
		&& c.drmFormat == drmFormat && c.eotf == eotf )
	{
		bool bComplete = true;
		for ( const auto &pReadback : c.generatedReadback )
			bComplete = bComplete && pReadback != nullptr;
		if ( bComplete )
			return true;
	}

	// A format transition cannot destroy staging images still owned by a GPU
	// copy. The probe simply skips this triplet and retries after completion.
	if ( c.pendingSeqNo != 0 && !g_device.hasCompletedFramegen( c.pendingSeqNo ) )
		return false;

	for ( auto &pReadback : c.generatedReadback )
		pReadback = nullptr;
	c.referenceReadback = nullptr;
	c.width = width;
	c.height = height;
	c.drmFormat = drmFormat;
	c.bytesPerPixel = DRMFormatGetBPP( drmFormat );
	c.eotf = eotf;
	const bool bSupportedFormat = drmFormat == DRM_FORMAT_ARGB8888
		|| drmFormat == DRM_FORMAT_XRGB8888
		|| drmFormat == DRM_FORMAT_ABGR8888
		|| drmFormat == DRM_FORMAT_XBGR8888
		|| drmFormat == DRM_FORMAT_ARGB2101010
		|| drmFormat == DRM_FORMAT_XRGB2101010
		|| drmFormat == DRM_FORMAT_ABGR2101010
		|| drmFormat == DRM_FORMAT_XBGR2101010
		|| drmFormat == DRM_FORMAT_ABGR16161616
		|| drmFormat == DRM_FORMAT_XBGR16161616
		|| drmFormat == DRM_FORMAT_ABGR16161616F
		|| drmFormat == DRM_FORMAT_XBGR16161616F
		|| drmFormat == DRM_FORMAT_ABGR32323232F;
	if ( !bSupportedFormat || c.bytesPerPixel == 0 )
		return false;

	const auto makeReadback = [=]( gamescope::OwningRc<CVulkanTexture> *ppTex )
	{
		CVulkanTexture::createFlags flags;
		flags.bMappable = true;
		flags.bTransferDst = true;
		flags.bFramegenShared = true;
		*ppTex = new CVulkanTexture();
		return ( *ppTex )->BInit( width, height, 1u, drmFormat, flags )
			&& ( *ppTex )->mappedData() != nullptr;
	};

	bool bAllocated = true;
	for ( auto &pReadback : c.generatedReadback )
		bAllocated = bAllocated && makeReadback( &pReadback );
	if ( !bAllocated || !makeReadback( &c.referenceReadback ) )
	{
		for ( auto &pReadback : c.generatedReadback )
			pReadback = nullptr;
		c.referenceReadback = nullptr;
		return false;
	}
	return true;
}

// E2 CPU half. GSCF v3 stores the three paired full-resolution candidates, the
// swept parameter, and their exact real reference. Rows are tightly repacked
// from the linear Vulkan images; the evaluator interprets the recorded DRM
// fourcc and output EOTF. The unchanged header size preserves v1/v2 parsing.
static void framegen_color_probe_consume()
{
	FramegenColorProbeResources_t &c = g_framegenColorProbe;
	if ( c.pendingSeqNo == 0 || c.generatedReadback[ 0 ] == nullptr || c.referenceReadback == nullptr
		|| !g_device.hasCompletedFramegen( c.pendingSeqNo ) )
		return;

	char szName[ 32 ];
	snprintf( szName, sizeof( szName ), "color_%06u.gscf", g_uFramegenColorRecordCount );
	std::string path = framegen_color_record_dir();
	path += '/';
	path += szName;
	gamescope::framegen::AtomicOutputFile output( path );
	if ( !output.is_open() )
	{
		vk_log.errorf( "framegen: full-colour capture can't write '%s' (%s)",
			path.c_str(), strerror( output.error() ) );
		g_uFramegenColorRecordCount = ~0u;
		framegen_color_probe_release( "write failure" );
		return;
	}

	const uint32_t uHeader[ 12 ] = {
		0x46435347u /* 'GSCF' */, 3u,
		c.width, c.height, c.drmFormat, c.bytesPerPixel, (uint32_t)c.eotf,
		3u /* flags: exact reference + paired candidates */, 4u /* planes */,
		(uint32_t)c.pendingSweep, 0u, 0u,
	};
	const uint64_t uMetadata[ 8 ] = {
		c.pendingSeqNo, c.pendingAnchorId, c.pendingReferenceId, c.pendingEndpointId,
		c.pendingAnchorTimeNs, c.pendingReferenceTimeNs, c.pendingEndpointTimeNs, 0u,
	};
	const float flMetadata[ 4 ] = {
		c.pendingPhase,
		k_flFramegenColorProbeStrengths[ 0 ],
		k_flFramegenColorProbeStrengths[ 1 ],
		k_flFramegenColorProbeStrengths[ 2 ],
	};
	bool bOk = output.write( uHeader, sizeof( uHeader ) )
		&& output.write( uMetadata, sizeof( uMetadata ) )
		&& output.write( flMetadata, sizeof( flMetadata ) );

	const auto writePlane = [&]( CVulkanTexture *pTex )
	{
		const uint32_t uBytesPerRow = c.width * c.bytesPerPixel;
		if ( pTex == nullptr || pTex->rowPitch() < uBytesPerRow
			|| pTex->width() != c.width || pTex->height() != c.height )
			return false;
		const uint8_t *pData = pTex->mappedData();
		if ( pData == nullptr )
			return false;
		for ( uint32_t y = 0; bOk && y < c.height; y++ )
			bOk = output.write( pData + (size_t)y * pTex->rowPitch(), uBytesPerRow );
		return bOk;
	};
	// Plane order is generated strength 0/0.5/1, then exact real reference.
	for ( const auto &pReadback : c.generatedReadback )
		bOk = bOk && writePlane( pReadback.get() );
	bOk = bOk && writePlane( c.referenceReadback.get() );
	if ( bOk )
		bOk = output.commit();

	if ( !bOk )
	{
		const int error = output.error() != 0 ? output.error() : EIO;
		vk_log.errorf( "framegen: full-colour capture write failed at '%s' (%s)",
			path.c_str(), strerror( error ) );
		g_uFramegenColorRecordCount = ~0u;
		framegen_color_probe_release( "write failure" );
		return;
	}

	c.pendingSeqNo = 0;
	// The synchronous file write above can take multiple display intervals.
	// Never reuse an anchor/reference timestamp captured before that stall: doing
	// so silently moves a requested 1/6 probe toward the midpoint. The GPU batch
	// is complete here, so every read/input pin can be released and the next real
	// frame starts a fresh, uncontaminated held-out sequence.
	c.anchor = nullptr;
	c.reference = nullptr;
	c.anchorId = 0;
	c.referenceId = 0;
	c.anchorTimeNs = 0;
	c.referenceTimeNs = 0;
	c.lastRealTimeNs = 0;
	g_framegenHistory.previousReal = nullptr;
	g_framegenHistory.currentReal = nullptr;
	g_framegenHistory.genReadA = nullptr;
	g_framegenHistory.genReadB = nullptr;
	g_framegenHistory.genReadReference = nullptr;
	g_framegenHistory.genReadSeqNo = 0;
	if ( g_uFramegenColorRecordCount == 0 )
		vk_log.infof( "framegen: full-colour held-out capture writing %ux%u paired %s GSCF samples to '%s'",
			c.width, c.height, gamescope::framegen::color_probe_sweep_name( c.pendingSweep ),
			framegen_color_record_dir() );
	g_uFramegenColorRecordCount++;
	if ( g_uFramegenColorRecordCount >= framegen_color_record_max() )
		framegen_color_probe_release( "sample cap reached" );
}

// Common sampler binding for the motion passes' non-sRGB intermediates (luma
// levels and motion fields).
static void framegen_motion_bind_sampler( CVulkanCmdBuffer *pCmdBuffer, uint32_t nSlot, gamescope::Rc<CVulkanTexture> pTexture, bool bNearest )
{
	pCmdBuffer->bindTexture( nSlot, std::move( pTexture ) );
	pCmdBuffer->setTextureSrgb( nSlot, false );
	pCmdBuffer->setSamplerUnnormalized( nSlot, false );
	pCmdBuffer->setSamplerNearest( nSlot, bNearest );
}

// Self-supervised stats probe (B4), recorded after the motion field is final:
// one workgroup zeroes the counter image, a field-size dispatch measures how
// well the checked field predicts the real frame and accumulates the verdicts
// (see cs_framegen_motion_stats.comp), and an apply dispatch folds the
// resulting global field trust into the field's confidence channel — so the
// full-res warps inherit the confidence verdict through the field fetch they
// already do, at zero added per-pixel cost. The copy at the end lands the raw
// counters in the host-mapped readback for the CPU-side threshold
// calibration. The command buffer's barrier tracking orders all of it
// (accumulate -> apply has a stats WAW plus an execution-only WAR on the
// field). Field-res work plus a 384-byte
// copy: still sub-kilobyte and intended to remain in the microsecond range.
static void framegen_record_adapt_probe( CVulkanCmdBuffer *pCmdBuffer, uint32_t lowW, uint32_t lowH )
{
	if ( g_framegenMotion.statsAccum == nullptr || g_framegenMotion.statsReadback == nullptr )
		return;

	const uint32_t pg = 8;
	pCmdBuffer->bindPipeline( g_device.pipeline( SHADER_TYPE_FRAMEGEN_MOTION_STATS ) );
	pCmdBuffer->bindTarget( g_framegenMotion.statsAccum );
	framegen_motion_bind_sampler( pCmdBuffer, 0, g_framegenMotion.lumaPrev, false );
	framegen_motion_bind_sampler( pCmdBuffer, 1, g_framegenMotion.lumaCur, false );
	// With the net refiner active this grades (and trust-scales) the REFINED
	// field — the one the warps consume — so a misbehaving checkpoint is
	// clamped in the same batch and shows up in the adapt log lines.
	framegen_motion_bind_sampler( pCmdBuffer, 2, framegen_motion_field(), true );
	pCmdBuffer->pushConstants<FramegenMotionStatsPush_t>( true );
	pCmdBuffer->dispatch( 1, 1 );
	pCmdBuffer->pushConstants<FramegenMotionStatsPush_t>( false );
	pCmdBuffer->dispatch( div_roundup( lowW, pg ), div_roundup( lowH, pg ) );
	// Scale the field confidence by the measured trust; in bidir the reverse
	// field carries the same scene-level verdict.
	pCmdBuffer->bindPipeline( g_device.pipeline( SHADER_TYPE_FRAMEGEN_MOTION_STATS_APPLY ) );
	pCmdBuffer->bindTarget( framegen_motion_field() );
	pCmdBuffer->bindTarget2( g_framegenMotion.statsAccum );
	// Finalize the sectioned histogram once. This dispatch writes stats[88],
	// then the field-sized pass broadcasts that verdict one read per workgroup.
	pCmdBuffer->pushConstants<FramegenMotionStatsApplyPush_t>(
		gamescope::framegen::k_flAdaptationTrustLo,
		gamescope::framegen::k_flAdaptationTrustHi, true );
	pCmdBuffer->dispatch( 1, 1 );
	pCmdBuffer->pushConstants<FramegenMotionStatsApplyPush_t>(
		gamescope::framegen::k_flAdaptationTrustLo,
		gamescope::framegen::k_flAdaptationTrustHi, false );
	pCmdBuffer->dispatch( div_roundup( lowW, pg ), div_roundup( lowH, pg ) );
	if ( vulkan_framegen_bidir_active() && framegen_motion_field_rev() != nullptr )
	{
		pCmdBuffer->bindTarget( framegen_motion_field_rev() );
		pCmdBuffer->bindTarget2( g_framegenMotion.statsAccum );
		pCmdBuffer->pushConstants<FramegenMotionStatsApplyPush_t>(
			gamescope::framegen::k_flAdaptationTrustLo,
			gamescope::framegen::k_flAdaptationTrustHi, false );
		pCmdBuffer->dispatch( div_roundup( lowW, pg ), div_roundup( lowH, pg ) );
	}
	// Copy after scene finalization so the next-batch debug/CPU consumer sees
	// the same verdict that this batch's warps consumed.
	pCmdBuffer->copyImage( g_framegenMotion.statsAccum, g_framegenMotion.statsReadback );
}

// Learned field refinement (Stage C), recorded with the checked fields final.
// The causal path refines only the forward field that advects the newest real
// frame into the future. Bidir additionally processes the reverse field through
// the same binding-symmetric network (confidence-veto-only by default). This
// keeps forward prediction at one inference dispatch per real frame instead of
// paying for an unused reverse result. The one-time staging->GPU weight copy
// rides the first batch after (re)allocation.
static void framegen_record_net( CVulkanCmdBuffer *pCmdBuffer, uint32_t lowW, uint32_t lowH, bool bRefineReverse )
{
	if ( g_framegenMotion.mvFieldNet == nullptr || g_framegenMotion.netWeightsGpu == nullptr
		|| g_framegenMotion.mvFieldRevChk == nullptr || g_framegenMotion.netShadingFocus == nullptr
		|| ( bRefineReverse && g_framegenMotion.mvFieldRevNet == nullptr ) )
		return;

	if ( !g_framegenMotion.bNetWeightsUploaded )
	{
		pCmdBuffer->copyImage( g_framegenMotion.netWeightsUpload, g_framegenMotion.netWeightsGpu );
		if ( g_framegenMotion.netWeightsPrior != nullptr )
			pCmdBuffer->copyImage( g_framegenMotion.netWeightsUpload, g_framegenMotion.netWeightsPrior );
		g_framegenMotion.bNetWeightsUploaded = true;
	}

	const uint32_t pg = 8; // = the shader's output tile
	const bool bConservativeBidir = vulkan_framegen_bidir_active();
	pCmdBuffer->bindPipeline( g_device.pipeline( SHADER_TYPE_FRAMEGEN_MOTION_NET ) );
	pCmdBuffer->bindTarget( g_framegenMotion.mvFieldNet );
	pCmdBuffer->bindTarget2( g_framegenMotion.netShadingFocus );
	framegen_motion_bind_sampler( pCmdBuffer, 0, g_framegenMotion.lumaPrev, false );
	framegen_motion_bind_sampler( pCmdBuffer, 1, g_framegenMotion.lumaCur, false );
	framegen_motion_bind_sampler( pCmdBuffer, 2, g_framegenMotion.mvField, true );
	framegen_motion_bind_sampler( pCmdBuffer, 3, g_framegenMotion.mvFieldRevChk, true );
	framegen_motion_bind_sampler( pCmdBuffer, 4, g_framegenMotion.netWeightsGpu, true );
	pCmdBuffer->pushConstants<FramegenMotionNetPush_t>( 0.0f, bConservativeBidir );
	pCmdBuffer->dispatch( div_roundup( lowW, pg ), div_roundup( lowH, pg ) );

	if ( bRefineReverse )
	{
		pCmdBuffer->bindTarget( g_framegenMotion.mvFieldRevNet );
		pCmdBuffer->bindTarget2( g_framegenMotion.netShadingFocus );
		framegen_motion_bind_sampler( pCmdBuffer, 0, g_framegenMotion.lumaCur, false );
		framegen_motion_bind_sampler( pCmdBuffer, 1, g_framegenMotion.lumaPrev, false );
		framegen_motion_bind_sampler( pCmdBuffer, 2, g_framegenMotion.mvFieldRevChk, true );
		framegen_motion_bind_sampler( pCmdBuffer, 3, g_framegenMotion.mvField, true );
		framegen_motion_bind_sampler( pCmdBuffer, 4, g_framegenMotion.netWeightsGpu, true );
		pCmdBuffer->pushConstants<FramegenMotionNetPush_t>( 0.0f, bConservativeBidir );
		pCmdBuffer->dispatch( div_roundup( lowW, pg ), div_roundup( lowH, pg ) );
	}

	g_framegenMotion.bNetActive = true;
}

static bool framegen_motion_history_valid( GamescopeFramegenPipeline ePipeline )
{
	const uint64_t ulCurrentIntervalNs = gamescope::framegen::present_interval_ns(
		g_framegenHistory.currentPresentTimeNs, g_framegenHistory.previousPresentTimeNs );
	const uint64_t ulHistoryIntervalNs = g_framegenMotion.uMotionHistoryIntervalNs;
	// A violent cadence transition is a poor second-derivative sample even after
	// time normalization. Fall back to constant velocity for that one interval;
	// the newly stored field immediately re-primes the next batch.
	const bool bComparableIntervals = gamescope::framegen::motion_intervals_comparable(
		ulCurrentIntervalNs, ulHistoryIntervalNs );
	return ePipeline >= GamescopeFramegenPipeline::Predict
		&& g_framegenMotion.mvFieldHistory != nullptr
		&& g_framegenMotion.uMotionHistoryFrameId != 0
		&& g_framegenMotion.uMotionHistoryFrameId + 1u == g_framegenHistory.currentFrameId
		&& bComparableIntervals;
}

static float framegen_motion_history_time_scale()
{
	if ( !framegen_motion_history_valid( GamescopeFramegenPipeline::Predict ) )
		return 1.0f;
	const uint64_t ulCurrentIntervalNs = g_framegenHistory.currentPresentTimeNs - g_framegenHistory.previousPresentTimeNs;
	return gamescope::framegen::motion_history_time_scale(
		ulCurrentIntervalNs, g_framegenMotion.uMotionHistoryIntervalNs );
}

static float framegen_motion_accel_time_factor()
{
	if ( !framegen_motion_history_valid( GamescopeFramegenPipeline::Predict ) )
		return 0.5f;
	const uint64_t ulCurrentIntervalNs = g_framegenHistory.currentPresentTimeNs - g_framegenHistory.previousPresentTimeNs;
	return gamescope::framegen::motion_acceleration_time_factor(
		ulCurrentIntervalNs, g_framegenMotion.uMotionHistoryIntervalNs );
}

static int framegen_luma_reservoir_read_index()
{
	for ( uint32_t i = 0; i < 2; i++ )
	{
		if ( g_framegenMotion.uLumaReservoirFrameId[i] != 0
			&& g_framegenMotion.uLumaReservoirFrameId[i] + 2u == g_framegenHistory.currentFrameId
			&& g_framegenMotion.lumaReservoir[i] != nullptr )
			return (int)i;
	}
	return -1;
}

// In-situ learning step (C2), recorded after the inference dispatches: two
// gradient passes (one per field direction, each workgroup a hashed training
// tile writing its slice row) and one optimizer pass folding the slices into
// Adam and publishing the EMA weights the NEXT batch's inference serves. The
// same-batch inference read the pre-step weights, so the write-after-read on
// the served texture is ordered by the command buffer's barrier tracking —
// and the one-batch-in-flight rule serializes everything across batches.
// Training reads the RAW fields: the net never trains on its own output.
// Returns true when this batch also carries a profile-dump copy.
static bool framegen_record_net_train( CVulkanCmdBuffer *pCmdBuffer, GamescopeFramegenPipeline ePipeline, bool bSceneCutGuard )
{
	FramegenMotionResources_t &m = g_framegenMotion;
	if ( m.netState == nullptr || m.netGradSlices == nullptr || m.netWeightsPrior == nullptr
		|| m.statsAccum == nullptr || m.mvFieldRevChk == nullptr || m.width < 16 || m.height < 16 )
		return false;

	const uint32_t uOptGroups = div_roundup( k_uFramegenNetFloats, 64u );
	const auto bindOpt = [&]()
	{
		pCmdBuffer->bindPipeline( g_device.pipeline( SHADER_TYPE_FRAMEGEN_MOTION_NET_OPT ) );
		pCmdBuffer->bindTarget( m.netState );
		pCmdBuffer->bindTarget2( m.netWeightsGpu );
		framegen_motion_bind_sampler( pCmdBuffer, 0, m.netGradSlices, true );
		framegen_motion_bind_sampler( pCmdBuffer, 1, m.netWeightsPrior, true );
	};

	// A fresh state texture holds garbage: one init dispatch seeds fast/EMA
	// from the prior and zeroes the Adam moments.
	if ( m.bNetStatePending )
	{
		bindOpt();
		pCmdBuffer->pushConstants<FramegenMotionNetOptPush_t>( 0.0f, 0.0f, 0.0f, 0u );
		pCmdBuffer->dispatch( uOptGroups, 1 );
		m.bNetStatePending = false;
		m.uNetTrainStep = 0;
	}

	// GAMESCOPE_FRAMEGEN_NET_EVERY=N trains on every Nth real frame — the
	// pressure valve for weak present GPUs (learning just converges slower).
	static uint32_t s_uFrameCounter = 0;
	if ( ( s_uFrameCounter++ % framegen_net_online_every() ) != 0 )
		return false;

	m.uNetTrainStep++;
	g_ulFramegenNetProgress++;
	const uint32_t uHalf = k_uFramegenNetTrainTiles / 2;
	const uint32_t uSeed = m.uNetTrainStep * 0x9E3779B9u + 0x61C88647u;
	const bool bConservativeBidir = vulkan_framegen_bidir_active();
	const int nReservoirRead = framegen_luma_reservoir_read_index();
	const bool bShadingHistoryValid = ePipeline == GamescopeFramegenPipeline::Guided
		&& framegen_shading_enabled( ePipeline )
		&& bSceneCutGuard
		&& framegen_motion_history_valid( ePipeline ) && nReservoirRead >= 0;
	const gamescope::Rc<CVulkanTexture> pOlderLuma = bShadingHistoryValid
		? gamescope::Rc<CVulkanTexture>( m.lumaReservoir[nReservoirRead] )
		: gamescope::Rc<CVulkanTexture>( m.lumaPrev );
	const gamescope::Rc<CVulkanTexture> pOlderField = bShadingHistoryValid
		? gamescope::Rc<CVulkanTexture>( m.mvFieldHistory )
		: gamescope::Rc<CVulkanTexture>( m.mvField );
	const float flHistoryTimeScale = bShadingHistoryValid
		? framegen_motion_history_time_scale() : 1.0f;

	pCmdBuffer->bindPipeline( g_device.pipeline( SHADER_TYPE_FRAMEGEN_MOTION_NET_TRAIN ) );
	pCmdBuffer->bindTarget( m.netGradSlices );
	pCmdBuffer->bindTarget2( m.statsAccum );
	framegen_motion_bind_sampler( pCmdBuffer, 0, m.lumaPrev, false );
	framegen_motion_bind_sampler( pCmdBuffer, 1, m.lumaCur, false );
	framegen_motion_bind_sampler( pCmdBuffer, 2, m.mvField, true );
	framegen_motion_bind_sampler( pCmdBuffer, 3, m.mvFieldRevChk, true );
	framegen_motion_bind_sampler( pCmdBuffer, 4, m.netState, true );
	framegen_motion_bind_sampler( pCmdBuffer, 5, pOlderLuma, false );
	framegen_motion_bind_sampler( pCmdBuffer, 6, pOlderField, false );
	pCmdBuffer->pushConstants<FramegenMotionNetTrainPush_t>( uSeed, 0u,
		bSceneCutGuard, bShadingHistoryValid, bConservativeBidir, flHistoryTimeScale );
	pCmdBuffer->dispatch( uHalf, 1 );

	pCmdBuffer->bindTarget( m.netGradSlices );
	pCmdBuffer->bindTarget2( m.statsAccum );
	framegen_motion_bind_sampler( pCmdBuffer, 0, m.lumaCur, false );
	framegen_motion_bind_sampler( pCmdBuffer, 1, m.lumaPrev, false );
	framegen_motion_bind_sampler( pCmdBuffer, 2, m.mvFieldRevChk, true );
	framegen_motion_bind_sampler( pCmdBuffer, 3, m.mvField, true );
	framegen_motion_bind_sampler( pCmdBuffer, 4, m.netState, true );
	// Reverse-field tiles cannot supervise a causal future-color trend. In the
	// conservative bidir policy, both directions train only the confidence
	// output row; the geometry heads and shared trunk remain fixed at the prior.
	framegen_motion_bind_sampler( pCmdBuffer, 5, m.lumaPrev, false );
	framegen_motion_bind_sampler( pCmdBuffer, 6, m.mvField, false );
	pCmdBuffer->pushConstants<FramegenMotionNetTrainPush_t>( uSeed ^ 0x55555555u, uHalf,
		bSceneCutGuard, false, bConservativeBidir, 1.0f );
	pCmdBuffer->dispatch( uHalf, 1 );

	bindOpt();
	pCmdBuffer->pushConstants<FramegenMotionNetOptPush_t>( framegen_net_online_lr(), k_flFramegenNetEmaAlpha, k_flFramegenNetDecay, m.uNetTrainStep );
	pCmdBuffer->dispatch( uOptGroups, 1 );

		// Snapshot the served weights every trained step (an 18.6 kB copy): the
	// mapped readback feeds the CPU-side health check and keeps the profile
	// flush at most one batch stale — the old every-1024-steps copy meant a
	// short session (or one that reset before the boundary) persisted nothing.
	if ( m.netProfileReadback != nullptr )
	{
		pCmdBuffer->copyImage( m.netWeightsGpu, m.netProfileReadback );
		return true;
	}
	return false;
}

// The actual file write uses a unique same-directory staging file, so a process
// crash, a kill, a full disk, or another Gamescope process saving the same
// profile can never expose a partial snapshot. AtomicOutputFile checks buffered
// close before publishing with rename.
// Runs on an owned worker for periodic checkpoints — file I/O on the render
// thread is a frametime spike — and synchronously for the exit/reset flush.
static void framegen_net_profile_write_file( std::vector<float> weights, uint64_t ulProgress, bool bFromWorker )
{
	const char *pszPath = framegen_net_profile_path();
	static std::atomic<bool> s_bLoggedFail = { false };
	const auto fail = [&]( const char *pszWhat, int error )
	{
		if ( !s_bLoggedFail.exchange( true ) )
			vk_log.errorf( "framegen: net profile %s '%s' failed (%s); keeping the previous file", pszWhat, pszPath, strerror( error ) );
	};
	if ( !gamescope::framegen::validate_and_migrate_net_profile_weights(
		k_uFramegenNetVersion, weights ) )
	{
		fail( "validation for", EINVAL );
		if ( bFromWorker )
			g_bFramegenNetWriteInFlight = false;
		return;
	}

	gamescope::framegen::AtomicOutputFile output( pszPath );
	if ( !output.is_open() )
	{
		fail( "open of", output.error() );
		if ( bFromWorker )
			g_bFramegenNetWriteInFlight = false;
		return;
	}
	constexpr auto metadata = gamescope::framegen::net_profile_metadata();
	bool bOk = output.write( metadata.data(), metadata.size() * sizeof( uint32_t ) )
		&& output.write( weights.data(), weights.size() * sizeof( float ) );
	if ( bOk )
		bOk = output.commit();
	if ( !bOk )
	{
		fail( "write to", output.error() != 0 ? output.error() : EIO );
	}
	else
	{
		g_ulFramegenNetSavedProgress = ulProgress;
		static std::atomic<bool> s_bLoggedFirstSave = { false };
		if ( !s_bLoggedFirstSave.exchange( true ) || g_bFramegenDebug )
			vk_log.infof( "framegen: net profile saved to '%s' (%" PRIu64 " trained steps)", pszPath, ulProgress );
	}
	if ( bFromWorker )
		g_bFramegenNetWriteInFlight = false;
}

// The checkpoint writer's only lifetime proof. It logs through the file-scope
// vk_log, so it must never outlive teardown (or, at exit, the LogScope itself).
// Joining has no polling timeout, so this is unconditionally safe to call, and
// vulkan_framegen_shutdown calls it before touching anything else.
static void framegen_net_profile_join_writer()
{
	if ( g_framegenNetWriteThread.joinable() )
		g_framegenNetWriteThread.join();
}

// Flush any unsaved learning. Called from vulkan_framegen_reset — every mode,
// resolution and teardown change funnels through it — and via atexit, so runs
// shorter than the checkpoint interval persist too (the old cadence-only write
// silently dropped them). Pure CPU on the cached copy: safe at any point,
// including after Vulkan teardown; never touches the GPU.
static void framegen_net_profile_flush()
{
	// The atomic is the cheap scheduling gate; the thread object is the lifetime
	// proof. Joining guarantees an older checkpoint cannot rename over the newer
	// synchronous flush below.
	framegen_net_profile_join_writer();
	if ( framegen_net_profile_path() == nullptr || g_framegenNetLiveWeights.size() != k_uFramegenNetFloats )
		return;
	if ( g_ulFramegenNetLiveProgress == g_ulFramegenNetSavedProgress.load() )
		return;
	framegen_net_profile_write_file( g_framegenNetLiveWeights, g_ulFramegenNetLiveProgress, false );
}

// Weights readback, CPU half: same completion gate as every other readback.
// Every trained batch carries a served-weights copy (18.6 kB), so this cache is
// never more than one batch stale. It doubles as the training health check:
// a non-finite weight anywhere means the optimizer diverged, and neither the
// decay (NaN - prior = NaN) nor Adam can recover it — so re-seed the state
// from the prior next batch instead of serving garbage until process exit.
// The bad snapshot is never cached, so it can never reach the profile file.
static void framegen_net_profile_consume()
{
	FramegenHistory_t &h = g_framegenHistory;
	if ( h.ulNetProfileSeqNo == 0 || h.ulNetProfileSeqNo == h.ulNetProfileConsumedSeqNo
		|| g_framegenMotion.netProfileReadback == nullptr
		|| g_framegenMotion.netProfileReadback->mappedData() == nullptr
		|| !g_device.hasCompletedFramegen( h.ulNetProfileSeqNo ) )
		return;
	h.ulNetProfileConsumedSeqNo = h.ulNetProfileSeqNo;

	const uint8_t *pData = g_framegenMotion.netProfileReadback->mappedData();
	g_framegenNetReadbackWeights.resize( k_uFramegenNetFloats );
	for ( uint32_t y = 0; y < k_uFramegenNetTexH; y++ )
	{
		const uint32_t uRowFloats = std::min( k_uFramegenNetTexW, k_uFramegenNetFloats - y * k_uFramegenNetTexW );
		memcpy( g_framegenNetReadbackWeights.data() + (size_t)y * k_uFramegenNetTexW,
			pData + (size_t)y * g_framegenMotion.netProfileReadback->rowPitch(), uRowFloats * sizeof( float ) );
	}
	for ( uint32_t i = 0; i < k_uFramegenNetFloats; i++ )
	{
		if ( !gamescope::framegen::is_finite_binary32( g_framegenNetReadbackWeights[ i ] ) )
		{
			vk_log.errorf( "framegen: net weights went non-finite at step %u — reinitializing from the prior (consider a lower GAMESCOPE_FRAMEGEN_NET_LR)", g_framegenMotion.uNetTrainStep );
			// framegen_record_net runs before the optimizer-init dispatch in the
			// next batch. Force its upload path too, otherwise that one inference
			// would still sample the bad served texture before state re-init.
			g_framegenMotion.bNetWeightsUploaded = false;
			g_framegenMotion.bNetStatePending = true;
			return;
		}
	}
	// Publish only after the full snapshot passes the health check. Swapping keeps
	// the preceding live allocation as next frame's scratch storage, so a bad
	// candidate cannot poison persistence and steady-state readback allocates none.
	g_framegenNetLiveWeights.swap( g_framegenNetReadbackWeights );
	g_ulFramegenNetLiveProgress = g_ulFramegenNetProgress;

	// First healthy readback ever: arm the exit flush (only online-learning
	// sessions reach this point, so plain runs register nothing).
	static const bool s_bAtExitArmed = []() { atexit( framegen_net_profile_flush ); return true; }();
	(void)s_bAtExitArmed;

	// Periodic checkpoint, off-thread — an fwrite into a possibly-contended
	// page cache has no business on the render thread. At most one writer in
	// flight; a skipped checkpoint just happens a step later.
	if ( framegen_net_profile_path() != nullptr
		&& g_ulFramegenNetLiveProgress - g_ulFramegenNetSavedProgress.load() >= (uint64_t)k_uFramegenNetProfileInterval
		&& !g_bFramegenNetWriteInFlight.exchange( true ) )
	{
		// A completed worker remains joinable until reaped. The atomic false
		// guarantees this join cannot wait on active I/O in the normal cadence.
		if ( g_framegenNetWriteThread.joinable() )
			g_framegenNetWriteThread.join();
		g_framegenNetWriteThread = std::thread( framegen_net_profile_write_file,
			g_framegenNetLiveWeights, g_ulFramegenNetLiveProgress, true );
	}
}

// Motion-compensated generation, part 1 (once per batch): build a three-level
// low-res luma pyramid for both real frames and block-match them coarse-to-
// fine into a motion field. This depends only on the two real frames —
// constant across every generated slot in the interval — so it is computed a
// single time and the field reused by each warp. Recorded into the batch
// command buffer; the dispatch path inserts the read-after-write barriers
// between passes. Returns false (fall back to extrapolation for the whole
// batch) if the intermediates can't be allocated.
//
// Why a pyramid: the full (2R+1)^2 search runs ONLY at the /4 level, where
// radius 4 spans 4x the motion range of the old single-level search (~±128
// full-res px at the default downscale) and each SAD tap integrates 4x the
// content — enough context to disambiguate self-similar detail (particle
// fields, tiled textures) that the fine level alone confidently mismatches.
// The finer levels then re-localize with a 9-candidate seeded search, so the
// finest (largest) level does ~9x less matching work than before.
static bool framegen_prepare_motion( CVulkanCmdBuffer *pCmdBuffer, uint32_t width, uint32_t height,
	const FramegenDispatch_t &dispatch, GamescopeFramegenPipeline ePipeline )
{
	// Every call below overwrites the working field. Callers which can reuse the
	// finalized field must decide that before entering; clearing the identity
	// here also makes benchmark/capture/setup calls unable to leave a stale
	// production cache behind.
	g_framegenMotion.uMotionFieldFrameId = 0;
	g_framegenMotion.uMotionFieldIntervalNs = 0;

	const uint32_t lowW = std::max( 1u, div_roundup( width, k_uFramegenMotionDownscale ) );
	const uint32_t lowH = std::max( 1u, div_roundup( height, k_uFramegenMotionDownscale ) );
	const uint32_t uFieldFormat = DRM_FORMAT_ABGR16161616F;
	const uint32_t uCoarseW[2] = { std::max( 1u, div_roundup( lowW, 2u ) ), std::max( 1u, div_roundup( lowW, 4u ) ) };
	const uint32_t uCoarseH[2] = { std::max( 1u, div_roundup( lowH, 2u ) ), std::max( 1u, div_roundup( lowH, 4u ) ) };
	// Bidir gathers the previous frame along the reverse field. The learned
	// refiner also consumes the checked reverse field as evidence even in the
	// zero-latency forward-prediction path. Both therefore require the reverse
	// chain and symmetric consistency check, overriding GAMESCOPE_FRAMEGEN_FB=0.
	const bool bBidir = vulkan_framegen_bidir_active();
	const bool bNeedMotionHistory = ePipeline >= GamescopeFramegenPipeline::Predict && !bBidir;
	const bool bNeedLumaReservoir = framegen_reservoir_enabled( ePipeline ) && !bBidir;
	const bool bCaptureNeedsReverse = framegen_record_dir() != nullptr
		&& g_uFramegenRecordCount < framegen_record_max();
	const bool bNeedCheckedReverse = bBidir || framegen_net_requested( ePipeline ) || bCaptureNeedsReverse;
	const bool bFBCheck = framegen_fbcheck_enabled( ePipeline ) || bNeedCheckedReverse;

	// Whether the net refiner ran is re-decided every batch (see
	// framegen_record_net); consumers must never inherit a stale verdict.
	g_framegenMotion.bNetActive = false;

	if ( g_framegenMotion.width != lowW || g_framegenMotion.height != lowH
		|| g_framegenMotion.lumaFormat != dispatch.motionLumaFormat
		|| g_framegenMotion.lumaPrev == nullptr || g_framegenMotion.lumaCur == nullptr || g_framegenMotion.mvField == nullptr
		|| g_framegenMotion.mvFieldCoarse[ 1 ] == nullptr
		|| ( bFBCheck && ( g_framegenMotion.mvFieldFwd == nullptr || g_framegenMotion.mvFieldRev == nullptr ) )
		|| ( bNeedCheckedReverse && g_framegenMotion.mvFieldRevChk == nullptr )
		|| ( bNeedMotionHistory && g_framegenMotion.mvFieldHistory == nullptr )
		|| ( framegen_net_requested( ePipeline ) && !g_framegenMotion.bNetAllocTried )
		|| ( bBidir && framegen_net_requested( ePipeline ) && g_framegenMotion.mvFieldRevNet == nullptr )
		|| ( bCaptureNeedsReverse && !g_framegenMotion.bRecAllocTried ) )
	{
		// Full-resolution history is keyed to the same reset as these field
		// resources. Drop it before rebuilding a different-sized pyramid.
		g_framegenMotion.lumaReservoir[0] = nullptr;
		g_framegenMotion.lumaReservoir[1] = nullptr;
		g_framegenMotion.uLumaReservoirFrameId[0] = 0;
		g_framegenMotion.uLumaReservoirFrameId[1] = 0;
		g_framegenMotion.bLumaReservoirAllocTried = false;
		bool bAllocated = framegen_create_intermediate( &g_framegenMotion.lumaPrev, lowW, lowH, dispatch.motionLumaFormat )
			&& framegen_create_intermediate( &g_framegenMotion.lumaCur, lowW, lowH, dispatch.motionLumaFormat )
			&& framegen_create_intermediate( &g_framegenMotion.mvField, lowW, lowH, uFieldFormat );
		for ( uint32_t i = 0; bAllocated && i < 2; i++ )
		{
			bAllocated = framegen_create_intermediate( &g_framegenMotion.lumaPrevCoarse[ i ], uCoarseW[ i ], uCoarseH[ i ], dispatch.motionLumaFormat )
				&& framegen_create_intermediate( &g_framegenMotion.lumaCurCoarse[ i ], uCoarseW[ i ], uCoarseH[ i ], dispatch.motionLumaFormat )
				&& framegen_create_intermediate( &g_framegenMotion.mvFieldCoarse[ i ], uCoarseW[ i ], uCoarseH[ i ], uFieldFormat );
		}
		if ( bAllocated && bFBCheck )
		{
			bAllocated = framegen_create_intermediate( &g_framegenMotion.mvFieldFwd, lowW, lowH, uFieldFormat )
				&& framegen_create_intermediate( &g_framegenMotion.mvFieldRev, lowW, lowH, uFieldFormat );
		}
		if ( bAllocated && bNeedCheckedReverse )
			bAllocated = framegen_create_intermediate( &g_framegenMotion.mvFieldRevChk, lowW, lowH, uFieldFormat );
		if ( bAllocated && bNeedMotionHistory )
			bAllocated = framegen_create_intermediate( &g_framegenMotion.mvFieldHistory, lowW, lowH, uFieldFormat );
		g_framegenMotion.uMotionHistoryFrameId = 0;
		g_framegenMotion.uMotionHistoryIntervalNs = 0;
		// B4 stats are a Learned-pipeline resource. Cheaper pipelines must not fail motion
		// setup because an adaptation-only image format/allocation failed.
		g_framegenMotion.statsAccum = nullptr;
		g_framegenMotion.statsReadback = nullptr;
		g_framegenHistory.ulAdaptStatsSeqNo = 0;
		if ( bAllocated && ( framegen_adapt_enabled( ePipeline )
			|| ( framegen_net_requested( ePipeline ) && framegen_net_online_enabled() ) ) )
		{
			CVulkanTexture::createFlags accumFlags;
			accumFlags.bStorage = true;
			accumFlags.bTransferSrc = true;
			accumFlags.bFramegenShared = true;
			g_framegenMotion.statsAccum = new CVulkanTexture();
			bAllocated = g_framegenMotion.statsAccum->BInit(
				gamescope::framegen::k_uAdaptationStatsCount,
				1, 1u, DRM_FORMAT_R32UI, accumFlags );
			if ( bAllocated )
			{
				CVulkanTexture::createFlags readbackFlags;
				readbackFlags.bMappable = true;
				readbackFlags.bTransferDst = true;
				readbackFlags.bFramegenShared = true;
				g_framegenMotion.statsReadback = new CVulkanTexture();
				bAllocated = g_framegenMotion.statsReadback->BInit(
					gamescope::framegen::k_uAdaptationStatsCount,
					1, 1u, DRM_FORMAT_R32UI, readbackFlags );
			}
		}
		// Stage C intermediates. Failures here disable the feature, not the
		// motion path: the raw checked fields keep working untouched.
		g_framegenMotion.mvFieldNet = nullptr;
		g_framegenMotion.mvFieldRevNet = nullptr;
		g_framegenMotion.netShadingFocus = nullptr;
		g_framegenMotion.bNetAllocTried = false;
		if ( bAllocated && framegen_net_requested( ePipeline ) )
		{
			g_framegenMotion.bNetAllocTried = true;
			bool bNetOk = framegen_create_intermediate( &g_framegenMotion.mvFieldNet, lowW, lowH, uFieldFormat );
			if ( bNetOk )
				bNetOk = framegen_create_intermediate( &g_framegenMotion.netShadingFocus, lowW, lowH, uFieldFormat );
			if ( bNetOk && bBidir )
				bNetOk = framegen_create_intermediate( &g_framegenMotion.mvFieldRevNet, lowW, lowH, uFieldFormat );
			// The weight texture pair is resolution-independent; created once.
			// Served weights are also a storage image (the online optimizer
			// writes them) and a transfer source (the profile dump).
			if ( bNetOk && g_framegenMotion.netWeightsGpu == nullptr )
			{
				CVulkanTexture::createFlags gpuFlags;
				gpuFlags.bSampled = true;
				gpuFlags.bStorage = true;
				gpuFlags.bTransferDst = true;
				gpuFlags.bTransferSrc = true;
				gpuFlags.bFramegenShared = true;
				g_framegenMotion.netWeightsGpu = new CVulkanTexture();
				bNetOk = g_framegenMotion.netWeightsGpu->BInit( k_uFramegenNetTexW, k_uFramegenNetTexH, 1u, DRM_FORMAT_R32F, gpuFlags );

				CVulkanTexture::createFlags stagingFlags;
				stagingFlags.bMappable = true;
				stagingFlags.bTransferSrc = true;
				stagingFlags.bFramegenShared = true;
				g_framegenMotion.netWeightsUpload = new CVulkanTexture();
				bNetOk = bNetOk && g_framegenMotion.netWeightsUpload->BInit( k_uFramegenNetTexW, k_uFramegenNetTexH, 1u, DRM_FORMAT_R32F, stagingFlags )
					&& g_framegenMotion.netWeightsUpload->mappedData() != nullptr;
				if ( bNetOk )
				{
					// Warm-start after a resize/format reset from the latest served
					// weights that passed the CPU finite check. Falling back to the
					// immutable startup prior is only needed before the first healthy
					// online readback (or when online learning is disabled).
					const bool bWarmStart = framegen_net_online_enabled()
						&& g_framegenNetLiveWeights.size() == k_uFramegenNetFloats;
					const std::vector<float> &weights = bWarmStart
						? g_framegenNetLiveWeights : framegen_net_weights();
					if ( bWarmStart && g_bFramegenDebug )
						vk_log.infof( "framegen: net warm-starting recreated GPU state from the latest healthy served weights (%" PRIu64 " trained steps)", g_ulFramegenNetLiveProgress );
					for ( uint32_t y = 0; y < k_uFramegenNetTexH; y++ )
					{
						const uint32_t uRowFloats = std::min( k_uFramegenNetTexW, (uint32_t)weights.size() - y * k_uFramegenNetTexW );
						memcpy( g_framegenMotion.netWeightsUpload->mappedData() + (size_t)y * g_framegenMotion.netWeightsUpload->rowPitch(),
							weights.data() + (size_t)y * k_uFramegenNetTexW, uRowFloats * sizeof( float ) );
					}
				}
				g_framegenMotion.bNetWeightsUploaded = false;

				// In-situ learning state (C2), also once.
				if ( bNetOk && framegen_net_online_enabled() )
				{
					CVulkanTexture::createFlags priorFlags;
					priorFlags.bSampled = true;
					priorFlags.bTransferDst = true;
					priorFlags.bFramegenShared = true;
					g_framegenMotion.netWeightsPrior = new CVulkanTexture();
					bool bOnlineOk = g_framegenMotion.netWeightsPrior->BInit( k_uFramegenNetTexW, k_uFramegenNetTexH, 1u, DRM_FORMAT_R32F, priorFlags );

					CVulkanTexture::createFlags stateFlags;
					stateFlags.bSampled = true;
					stateFlags.bStorage = true;
					stateFlags.bFramegenShared = true;
					g_framegenMotion.netState = new CVulkanTexture();
					bOnlineOk = bOnlineOk && g_framegenMotion.netState->BInit( k_uFramegenNetFloats, 4u, 1u, DRM_FORMAT_R32F, stateFlags );

					g_framegenMotion.netGradSlices = new CVulkanTexture();
					bOnlineOk = bOnlineOk && g_framegenMotion.netGradSlices->BInit( k_uFramegenNetFloats, k_uFramegenNetTrainTiles, 1u, DRM_FORMAT_R32F, stateFlags );

					// The served-weights readback backs the per-step health
					// check (non-finite detection + re-init) as well as the
					// profile persistence, so every online run gets one —
					// training without it would serve a diverged net forever.
					if ( bOnlineOk )
					{
						CVulkanTexture::createFlags readbackFlags;
						readbackFlags.bMappable = true;
						readbackFlags.bTransferDst = true;
						readbackFlags.bFramegenShared = true;
						g_framegenMotion.netProfileReadback = new CVulkanTexture();
						bOnlineOk = g_framegenMotion.netProfileReadback->BInit( k_uFramegenNetTexW, k_uFramegenNetTexH, 1u, DRM_FORMAT_R32F, readbackFlags )
							&& g_framegenMotion.netProfileReadback->mappedData() != nullptr;
					}
					if ( !bOnlineOk )
					{
						g_framegenMotion.netWeightsPrior = nullptr;
						g_framegenMotion.netState = nullptr;
						g_framegenMotion.netGradSlices = nullptr;
						g_framegenMotion.netProfileReadback = nullptr;
						vk_log.errorf( "framegen: online-learning state allocation failed; serving the prior without training" );
					}
					g_framegenMotion.bNetStatePending = true;
					g_framegenHistory.ulNetProfileSeqNo = 0;
				}
			}
			if ( !bNetOk )
			{
				g_framegenMotion.mvFieldNet = nullptr;
				g_framegenMotion.mvFieldRevNet = nullptr;
				g_framegenMotion.netShadingFocus = nullptr;
				g_framegenMotion.netWeightsGpu = nullptr;
				g_framegenMotion.netWeightsUpload = nullptr;
				vk_log.errorf( "framegen: net intermediate allocation failed, learned refinement disabled" );
			}
		}
		// Dataset-capture readbacks (recreated garbage is never parsed: the
		// pending record seqNo is dropped below).
		g_framegenMotion.recLumaPrev = nullptr;
		g_framegenMotion.recLumaCur = nullptr;
		g_framegenMotion.recField = nullptr;
		g_framegenMotion.recFieldRev = nullptr;
		g_framegenMotion.bRecAllocTried = false;
		g_framegenHistory.ulNetRecordSeqNo = 0;
		if ( bAllocated && framegen_record_dir() != nullptr
			&& g_uFramegenRecordCount < framegen_record_max() )
		{
			g_framegenMotion.bRecAllocTried = true;
			const auto makeReadback = []( gamescope::OwningRc<CVulkanTexture> *ppTex, uint32_t w, uint32_t h, uint32_t fmt )
			{
				CVulkanTexture::createFlags flags;
				flags.bMappable = true;
				flags.bTransferDst = true;
				flags.bFramegenShared = true;
				*ppTex = new CVulkanTexture();
				return ( *ppTex )->BInit( w, h, 1u, fmt, flags ) && ( *ppTex )->mappedData() != nullptr;
			};
			const bool bRecOk = makeReadback( &g_framegenMotion.recLumaPrev, lowW, lowH, dispatch.motionLumaFormat )
				&& makeReadback( &g_framegenMotion.recLumaCur, lowW, lowH, dispatch.motionLumaFormat )
				&& makeReadback( &g_framegenMotion.recField, lowW, lowH, uFieldFormat )
				&& makeReadback( &g_framegenMotion.recFieldRev, lowW, lowH, uFieldFormat );
			if ( !bRecOk )
			{
				g_framegenMotion.recLumaPrev = nullptr;
				g_framegenMotion.recLumaCur = nullptr;
				g_framegenMotion.recField = nullptr;
				g_framegenMotion.recFieldRev = nullptr;
				vk_log.errorf( "framegen: dataset-capture readback allocation failed, capture disabled" );
			}
		}
		if ( !bAllocated )
		{
			if ( g_bFramegenDebug )
				vk_log.infof( "framegen: motion intermediate allocation failed, falling back to extrapolation" );
			g_framegenMotion = {};
			return false;
		}
		g_framegenMotion.width = lowW;
		g_framegenMotion.height = lowH;
		g_framegenMotion.lumaFormat = dispatch.motionLumaFormat;
	}

	// The luma reservoir is a Guided-only enhancement, not a prerequisite
	// for motion generation. Allocation failure leaves the established guided
	// warp intact and is not retried every frame.
	if ( bNeedLumaReservoir && !g_framegenMotion.bLumaReservoirAllocTried )
	{
		g_framegenMotion.bLumaReservoirAllocTried = true;
		const bool bReservoirOk = framegen_create_luma_reservoir( &g_framegenMotion.lumaReservoir[0],
			lowW, lowH, dispatch.motionLumaFormat )
			&& framegen_create_luma_reservoir( &g_framegenMotion.lumaReservoir[1],
				lowW, lowH, dispatch.motionLumaFormat );
		if ( !bReservoirOk )
		{
			g_framegenMotion.lumaReservoir[0] = nullptr;
			g_framegenMotion.lumaReservoir[1] = nullptr;
			vk_log.errorf( "framegen: luma-reservoir allocation failed; keeping the Guided warp without third-frame disocclusion evidence" );
		}
	}

	const uint32_t pg = 8;

	// Pass 1: downscale both real frames to the base low-res luma pair.
	pCmdBuffer->bindPipeline( g_device.pipeline( dispatch.motionLumaPair ) );
	pCmdBuffer->bindTarget( g_framegenMotion.lumaPrev );
	pCmdBuffer->bindTarget2( g_framegenMotion.lumaCur );
	pCmdBuffer->bindTexture( 0, g_framegenHistory.previousReal );
	pCmdBuffer->setTextureSrgb( 0, true );
	pCmdBuffer->setSamplerUnnormalized( 0, false );
	pCmdBuffer->setSamplerNearest( 0, false );
	pCmdBuffer->bindTexture( 1, g_framegenHistory.currentReal );
	pCmdBuffer->setTextureSrgb( 1, true );
	pCmdBuffer->setSamplerUnnormalized( 1, false );
	pCmdBuffer->setSamplerNearest( 1, false );
	pCmdBuffer->dispatch( div_roundup( lowW, pg ), div_roundup( lowH, pg ) );

	// Pass 2: build the coarser pyramid levels, one 2x step per dispatch, both
	// frames per dispatch (bilinear tap = exact 2x2 box).
	for ( uint32_t i = 0; i < 2; i++ )
	{
		pCmdBuffer->bindPipeline( g_device.pipeline( dispatch.motionPyramidPair ) );
		pCmdBuffer->bindTarget( g_framegenMotion.lumaPrevCoarse[ i ] );
		pCmdBuffer->bindTarget2( g_framegenMotion.lumaCurCoarse[ i ] );
		framegen_motion_bind_sampler( pCmdBuffer, 0, i == 0 ? g_framegenMotion.lumaPrev : g_framegenMotion.lumaPrevCoarse[ 0 ], false );
		framegen_motion_bind_sampler( pCmdBuffer, 1, i == 0 ? g_framegenMotion.lumaCur : g_framegenMotion.lumaCurCoarse[ 0 ], false );
		pCmdBuffer->dispatch( div_roundup( uCoarseW[ i ], pg ), div_roundup( uCoarseH[ i ], pg ) );
	}

	// Pass 3: full block match at the coarsest level only.
	pCmdBuffer->bindPipeline( g_device.pipeline( SHADER_TYPE_FRAMEGEN_MOTION_MATCH ) );
	pCmdBuffer->bindTarget( g_framegenMotion.mvFieldCoarse[ 1 ] );
	framegen_motion_bind_sampler( pCmdBuffer, 0, g_framegenMotion.lumaPrevCoarse[ 1 ], true );
	framegen_motion_bind_sampler( pCmdBuffer, 1, g_framegenMotion.lumaCurCoarse[ 1 ], true );
	pCmdBuffer->pushConstants<FramegenMotionMatchPush_t>( 4 );
	pCmdBuffer->dispatch( div_roundup( uCoarseW[ 1 ], pg ), div_roundup( uCoarseH[ 1 ], pg ) );

	// Passes 4-5: seeded +/-1 refinement down the pyramid. The finest pass also
	// runs the sub-texel parabola and the confidence estimate, and writes the
	// field the warps consume (.rg = mv in base low-res texels, .b = conf) —
	// or, with the FB check on, the unchecked forward field it filters.
	for ( uint32_t i = 0; i < 2; i++ )
	{
		const bool bFinal = ( i == 1 );
		pCmdBuffer->bindPipeline( g_device.pipeline( SHADER_TYPE_FRAMEGEN_MOTION_MATCH_REFINE ) );
		pCmdBuffer->bindTarget( bFinal ? ( bFBCheck ? g_framegenMotion.mvFieldFwd : g_framegenMotion.mvField ) : g_framegenMotion.mvFieldCoarse[ 0 ] );
		framegen_motion_bind_sampler( pCmdBuffer, 0, bFinal ? g_framegenMotion.lumaPrev : g_framegenMotion.lumaPrevCoarse[ 0 ], true );
		framegen_motion_bind_sampler( pCmdBuffer, 1, bFinal ? g_framegenMotion.lumaCur : g_framegenMotion.lumaCurCoarse[ 0 ], true );
		framegen_motion_bind_sampler( pCmdBuffer, 2, bFinal ? g_framegenMotion.mvFieldCoarse[ 0 ] : g_framegenMotion.mvFieldCoarse[ 1 ], true );
		pCmdBuffer->pushConstants<FramegenMotionRefinePush_t>( bFinal );
		pCmdBuffer->dispatch( div_roundup( bFinal ? lowW : uCoarseW[ 0 ], pg ), div_roundup( bFinal ? lowH : uCoarseH[ 0 ], pg ) );
	}

	if ( !bFBCheck )
		return true;

	// Passes 6-8: the same coarse-to-fine match with the two luma bindings
	// swapped estimates the REVERSE flow, anchored at the previous frame
	// (prev(q) came from cur(q - R(q))). The forward chain is done with the
	// coarse fields by now, so the reverse chain reuses them as scratch — the
	// per-dispatch barriers order the reuse, and the only extra allocations
	// are the two full-res fields.
	pCmdBuffer->bindPipeline( g_device.pipeline( SHADER_TYPE_FRAMEGEN_MOTION_MATCH ) );
	pCmdBuffer->bindTarget( g_framegenMotion.mvFieldCoarse[ 1 ] );
	framegen_motion_bind_sampler( pCmdBuffer, 0, g_framegenMotion.lumaCurCoarse[ 1 ], true );
	framegen_motion_bind_sampler( pCmdBuffer, 1, g_framegenMotion.lumaPrevCoarse[ 1 ], true );
	pCmdBuffer->pushConstants<FramegenMotionMatchPush_t>( 4 );
	pCmdBuffer->dispatch( div_roundup( uCoarseW[ 1 ], pg ), div_roundup( uCoarseH[ 1 ], pg ) );

	for ( uint32_t i = 0; i < 2; i++ )
	{
		const bool bFinal = ( i == 1 );
		pCmdBuffer->bindPipeline( g_device.pipeline( SHADER_TYPE_FRAMEGEN_MOTION_MATCH_REFINE ) );
		pCmdBuffer->bindTarget( bFinal ? g_framegenMotion.mvFieldRev : g_framegenMotion.mvFieldCoarse[ 0 ] );
		framegen_motion_bind_sampler( pCmdBuffer, 0, bFinal ? g_framegenMotion.lumaCur : g_framegenMotion.lumaCurCoarse[ 0 ], true );
		framegen_motion_bind_sampler( pCmdBuffer, 1, bFinal ? g_framegenMotion.lumaPrev : g_framegenMotion.lumaPrevCoarse[ 0 ], true );
		framegen_motion_bind_sampler( pCmdBuffer, 2, bFinal ? g_framegenMotion.mvFieldCoarse[ 0 ] : g_framegenMotion.mvFieldCoarse[ 1 ], true );
		pCmdBuffer->pushConstants<FramegenMotionRefinePush_t>( bFinal );
		pCmdBuffer->dispatch( div_roundup( bFinal ? lowW : uCoarseW[ 0 ], pg ), div_roundup( bFinal ? lowH : uCoarseH[ 0 ], pg ) );
	}

	// Pass 9: forward-backward consistency. A correct forward vector round-
	// trips through the reverse field (R(p - F(p)) ~= -F(p)); lookalike
	// mislocks and disocclusions don't, and get their confidence killed so the
	// warp falls back to bounded extrapolation there instead of a confidently
	// wrong gather. Writes the checked field the warps consume.
	pCmdBuffer->bindPipeline( g_device.pipeline( SHADER_TYPE_FRAMEGEN_MOTION_FBCHECK ) );
	pCmdBuffer->bindTarget( g_framegenMotion.mvField );
	framegen_motion_bind_sampler( pCmdBuffer, 0, g_framegenMotion.mvFieldFwd, true );
	framegen_motion_bind_sampler( pCmdBuffer, 1, g_framegenMotion.mvFieldRev, true );
	pCmdBuffer->pushConstants<FramegenMotionFBCheckPush_t>( framegen_effective_fbcheck_tol( ePipeline ), k_flFramegenFBTolSlope );
	pCmdBuffer->dispatch( div_roundup( lowW, pg ), div_roundup( lowH, pg ) );

	if ( !bNeedCheckedReverse )
		return true;

	// Pass 10 (bidir/net/capture): the symmetric check for the reverse field — the
	// same pass with the fields swapped (the shader is direction-agnostic: it
	// tests field[0]'s round trip through field[1]). Bidir consumes it directly;
	// forward learned prediction uses it as consistency evidence.
	pCmdBuffer->bindPipeline( g_device.pipeline( SHADER_TYPE_FRAMEGEN_MOTION_FBCHECK ) );
	pCmdBuffer->bindTarget( g_framegenMotion.mvFieldRevChk );
	framegen_motion_bind_sampler( pCmdBuffer, 0, g_framegenMotion.mvFieldRev, true );
	framegen_motion_bind_sampler( pCmdBuffer, 1, g_framegenMotion.mvFieldFwd, true );
	pCmdBuffer->pushConstants<FramegenMotionFBCheckPush_t>( framegen_effective_fbcheck_tol( ePipeline ), k_flFramegenFBTolSlope );
	pCmdBuffer->dispatch( div_roundup( lowW, pg ), div_roundup( lowH, pg ) );

	return true;
}

// Motion-compensated generation, part 2 (per slot): warp the current frame
// forward along the shared motion field, blending back to extrapolation where
// the match is unconfident. Each slot writes a distinct target, so warps in the
// same batch don't hazard against each other.
static void framegen_warp_slot( CVulkanCmdBuffer *pCmdBuffer, const gamescope::Rc<CVulkanTexture> &pTarget,
	float flStrength, GamescopeFramegenPipeline ePipeline )
{
	const uint32_t pg = 8;
	const bool bHistoryValid = framegen_motion_history_valid( ePipeline );
	const bool bGuided = ePipeline == GamescopeFramegenPipeline::Guided;
	const int nReservoirRead = framegen_luma_reservoir_read_index();
	const bool bReservoirValid = bGuided && bHistoryValid
		&& nReservoirRead >= 0;
	const bool bShadingValid = bGuided && bHistoryValid
		&& framegen_shading_enabled( ePipeline )
		&& g_framegenMotion.bNetActive && g_framegenMotion.netShadingFocus != nullptr;
	const float flHistoryFlowScale = bHistoryValid ? framegen_motion_history_time_scale() : 1.0f;
	const float flAccelTimeFactor = bHistoryValid ? framegen_motion_accel_time_factor() : 0.5f;
	// Guided uses the accelerated shader even during the first interval: its
	// full-resolution field reconstruction does not need temporal history.
	const bool bAccelPipeline = bHistoryValid || bGuided;

	pCmdBuffer->bindPipeline( g_device.pipeline( bAccelPipeline
		? SHADER_TYPE_FRAMEGEN_MOTION_WARP_ACCEL : SHADER_TYPE_FRAMEGEN_MOTION_WARP ) );
	pCmdBuffer->bindTarget( pTarget );
	pCmdBuffer->bindTexture( 0, g_framegenHistory.previousReal );
	pCmdBuffer->setTextureSrgb( 0, true );
	pCmdBuffer->setSamplerUnnormalized( 0, false );
	pCmdBuffer->setSamplerNearest( 0, false );
	pCmdBuffer->bindTexture( 1, g_framegenHistory.currentReal );
	pCmdBuffer->setTextureSrgb( 1, true );
	pCmdBuffer->setSamplerUnnormalized( 1, false );
	pCmdBuffer->setSamplerNearest( 1, false );
	pCmdBuffer->bindTexture( 2, framegen_motion_field() );
	pCmdBuffer->setTextureSrgb( 2, false );
	pCmdBuffer->setSamplerUnnormalized( 2, false );
	pCmdBuffer->setSamplerNearest( 2, false );
	const bool bAgree = framegen_agreement_enabled( ePipeline );
	const float flAgreeLo = bAgree ? framegen_effective_agree_lo( ePipeline ) : 1e5f;
	const float flAgreeHi = bAgree ? framegen_effective_agree_hi( ePipeline ) : 1e6f;
	if ( bAccelPipeline )
	{
		framegen_motion_bind_sampler( pCmdBuffer, 3, g_framegenMotion.mvFieldHistory, false );
		// Bind a valid image even while the reservoir is warming. The push
		// constant prevents the shader from reading slot 4 until its frame-ID
		// chain and the preceding motion field are both consecutive.
		pCmdBuffer->bindTexture( 4, bReservoirValid
			? gamescope::Rc<CVulkanTexture>( g_framegenMotion.lumaReservoir[nReservoirRead] )
			: gamescope::Rc<CVulkanTexture>( g_framegenMotion.lumaPrev ) );
		pCmdBuffer->setTextureSrgb( 4, false );
		pCmdBuffer->setSamplerUnnormalized( 4, false );
		pCmdBuffer->setSamplerNearest( 4, false );
		// The focus texture is always valid when the net dispatch ran. During a
		// warm-up or non-net Guided batch, bind a harmless sampled fallback and
		// keep the shader branch disabled through the push constant.
		framegen_motion_bind_sampler( pCmdBuffer, 5, bShadingValid
			? gamescope::Rc<CVulkanTexture>( g_framegenMotion.netShadingFocus )
			: gamescope::Rc<CVulkanTexture>( g_framegenMotion.lumaCur ), false );
		pCmdBuffer->pushConstants<FramegenMotionAccelPush_t>( flStrength,
			k_flFramegenSuppressLo, k_flFramegenSuppressHi,
			(float)k_uFramegenMotionDownscale, flAgreeLo, flAgreeHi,
			1.0f, bHistoryValid, bGuided, bReservoirValid, bShadingValid,
			flHistoryFlowScale, flAccelTimeFactor );
	}
	else
	{
		pCmdBuffer->pushConstants<FramegenMotionWarpPush_t>( flStrength,
			k_flFramegenSuppressLo, k_flFramegenSuppressHi,
			(float)k_uFramegenMotionDownscale, flAgreeLo, flAgreeHi );
	}
	pCmdBuffer->dispatch( div_roundup( pTarget->width(), pg ), div_roundup( pTarget->height(), pg ) );
}

// Bidir (B3) per slot: warp BOTH real frames toward the slot's temporal phase
// (current along the checked forward field, previous along the checked reverse
// field) and blend by confidence x phase proximity; pixels neither direction
// can vouch for degrade to a phase-correct crossfade inside the shader.
static void framegen_bidir_warp_slot( CVulkanCmdBuffer *pCmdBuffer, const gamescope::Rc<CVulkanTexture> &pTarget,
	float flPhase, GamescopeFramegenPipeline ePipeline, float flOneSidedOverride = -1.0f,
	float flEndpointTraceOverride = -1.0f )
{
	const uint32_t pg = 8;
	const float flEndpointTraceStrength = ePipeline == GamescopeFramegenPipeline::Guided
		? ( flEndpointTraceOverride >= 0.0f
			? std::clamp( flEndpointTraceOverride, 0.0f, 1.0f )
			: framegen_bidir_endpoint_trace_strength( ePipeline ) )
		: 0.0f;

	pCmdBuffer->bindPipeline( g_device.pipeline( flEndpointTraceStrength > 0.0f
		? SHADER_TYPE_FRAMEGEN_MOTION_BIDIR_TRACE
		: SHADER_TYPE_FRAMEGEN_MOTION_BIDIR ) );
	pCmdBuffer->bindTarget( pTarget );
	pCmdBuffer->bindTexture( 0, g_framegenHistory.previousReal );
	pCmdBuffer->setTextureSrgb( 0, true );
	pCmdBuffer->setSamplerUnnormalized( 0, false );
	pCmdBuffer->setSamplerNearest( 0, false );
	pCmdBuffer->bindTexture( 1, g_framegenHistory.currentReal );
	pCmdBuffer->setTextureSrgb( 1, true );
	pCmdBuffer->setSamplerUnnormalized( 1, false );
	pCmdBuffer->setSamplerNearest( 1, false );
	pCmdBuffer->bindTexture( 2, framegen_motion_field() );
	pCmdBuffer->setTextureSrgb( 2, false );
	pCmdBuffer->setSamplerUnnormalized( 2, false );
	pCmdBuffer->setSamplerNearest( 2, false );
	pCmdBuffer->bindTexture( 3, framegen_motion_field_rev() );
	pCmdBuffer->setTextureSrgb( 3, false );
	pCmdBuffer->setSamplerUnnormalized( 3, false );
	pCmdBuffer->setSamplerNearest( 3, false );
	const bool bAgree = framegen_agreement_enabled( ePipeline );
	const float flOneSidedStrength = flOneSidedOverride >= 0.0f
		? std::clamp( flOneSidedOverride, 0.0f, 1.0f )
		: framegen_bidir_one_sided_strength();
	pCmdBuffer->pushConstants<FramegenMotionBidirPush_t>( flPhase, (float)k_uFramegenMotionDownscale,
		bAgree ? framegen_effective_agree_lo( ePipeline ) : 1e5f,
		bAgree ? framegen_effective_agree_hi( ePipeline ) : 1e6f,
		flOneSidedStrength, flEndpointTraceStrength );
	pCmdBuffer->dispatch( div_roundup( pTarget->width(), pg ), div_roundup( pTarget->height(), pg ) );
}

// One planned generated frame: its temporal phase (fraction of a real-frame
// interval past the current real frame), the shader forward coefficient
// derived from it, and the interval-relative slot index for refill bookkeeping.
using FramegenSlotRequest_t = gamescope::framegen::SlotRequest;

struct FramegenColorProbeRequest_t
{
	gamescope::Rc<CVulkanTexture> reference;
	EOTF eotf;
	FramegenColorProbeSweep sweep;
	uint64_t anchorId;
	uint64_t referenceId;
	uint64_t endpointId;
	uint64_t anchorTimeNs;
	uint64_t referenceTimeNs;
	uint64_t endpointTimeNs;
};

static uint64_t framegen_display_interval_ns()
{
	const int nFramegenRefreshMhz = g_nNestedRefresh ? g_nNestedRefresh : g_nOutputRefresh;
	return nFramegenRefreshMhz > 0
		? 1'000'000'000'000ull / (uint64_t)nFramegenRefreshMhz : 8'333'333ull;
}

static void framegen_shadow_plan_real( uint64_t ulRealFrameId,
	uint64_t ulSourceReadyNs, uint64_t ulProvisionalTargetNs,
	const gamescope::framegen::CadencePredictorState &cadence,
	uint64_t ulNowNs, uint64_t ulVblankIntervalNs,
	bool bSourceTimestampsReliable, bool bDedicatedQueue,
	bool bSharedQueueProvenEmpty, uint32_t nClassicGap )
{
	FramegenDeadlineShadowState_t &shadow = g_framegenDeadlineShadow;
	const uint64_t ulDisplayChainGeneration =
		g_framegenPresentState.displayTiming.generation;
	const bool bDisplayChainChanged =
		shadow.ulDisplayChainGeneration != ulDisplayChainGeneration;
	const bool bGridChanged = shadow.ulGridEpoch == 0u
		|| shadow.ulGridIntervalNs != ulVblankIntervalNs
		|| bDisplayChainChanged;
	const bool bProvenanceChanged = shadow.bSourceProvenanceInitialized
		&& shadow.bSourceTimestampsReliable != bSourceTimestampsReliable;
	if ( bGridChanged || bProvenanceChanged )
	{
		shadow.ulGridEpoch++;
		if ( bDisplayChainChanged )
			shadow.presentBias = {};
		shadow.nNextDecision = 0;
		shadow.nDecisionCount = 0;
		shadow.decisions = {};
	}
	shadow.ulGridIntervalNs = ulVblankIntervalNs;
	shadow.ulDisplayChainGeneration = ulDisplayChainGeneration;
	shadow.bSourceProvenanceInitialized = true;
	shadow.bSourceTimestampsReliable = bSourceTimestampsReliable;

	const gamescope::VBlankScheduleTime schedule =
		GetVBlankTimer().CalcNextWakeupTime( true );
	const gamescope::framegen::DisplayGrid_t rawGrid = {
		.D0 = schedule.ulTargetVBlank,
		.W0 = schedule.ulScheduledWakeupPoint,
		.T = ulVblankIntervalNs,
	};
	const gamescope::framegen::DisplayGrid_t grid =
		gamescope::framegen::apply_present_bias(
			rawGrid, shadow.presentBias.emaNs );
	shadow.anchor = {
		.realFrameId = ulRealFrameId,
		.sourceReadyNs = ulSourceReadyNs,
		.provisionalTargetNs = gamescope::framegen::apply_present_bias_ns(
			ulProvisionalTargetNs, shadow.presentBias.emaNs ),
		.provisionalBiasNs = shadow.presentBias.emaNs,
		.correctedFlipNs = std::nullopt,
		.epoch = shadow.ulGridEpoch,
	};
	const gamescope::framegen::CausalSlotPlan_t plan =
		gamescope::framegen::plan_next_causal_slot( grid, shadow.anchor, cadence, {
			.nowNs = ulNowNs,
			.gridEpoch = shadow.ulGridEpoch,
			.presentBiasNs = shadow.presentBias.emaNs,
			.configuredStrength = g_flFramegenStrength,
			.forwardStrengthCap = k_flFramegenMaxForwardStrength,
			.sourceTimestampsReliable = bSourceTimestampsReliable,
			.dedicatedQueue = bDedicatedQueue,
			.sharedQueueProvenEmpty = bSharedQueueProvenEmpty,
		} );

	shadow.decisions[ shadow.nNextDecision ] = {
		.plan = plan,
		.ulRealFrameId = ulRealFrameId,
		.nClassicGap = nClassicGap,
		.nClassicSlot = nClassicGap > 1u ? 1u : 0u,
	};
	shadow.nNextDecision = ( shadow.nNextDecision + 1u )
		% k_nFramegenShadowDecisionCapacity;
	shadow.nDecisionCount = std::min(
		shadow.nDecisionCount + 1u, k_nFramegenShadowDecisionCapacity );

	static uint64_t s_uShadowDebugLogCounter = 0;
	if ( FramegenDebugShouldLog( s_uShadowDebugLogCounter ) )
	{
		const double flTargetFromNowMs = static_cast<double>(
			static_cast<long double>( plan.targetNs )
			- static_cast<long double>( ulNowNs ) ) / 1.0e6;
		vk_log.infof( "framegen: shadow slot target=%+.2fms phase=%.2f admit=%d vs classic gap=%u k=%u",
			flTargetFromNowMs, plan.phase, plan.admit ? 1 : 0,
			nClassicGap, nClassicGap > 1u ? 1u : 0u );
	}
}

static uint64_t framegen_fixed_slot_target_ns( uint32_t nSlotIndex )
{
	if ( g_framegenHistory.ulCurrentRealVblankNs == 0 )
		return 0;
	return g_framegenHistory.ulCurrentRealVblankNs
		+ uint64_t( nSlotIndex ) * framegen_display_interval_ns();
}

// Deadline order is authoritative. Real endpoints own a colliding target, and
// multiple endpoints quantized to that target coalesce to the newest frame.
static bool framegen_insert_pending_entry(
	FramegenHistory_t::PendingGenerated_t entry )
{
	if ( entry.ulTargetFlipNs == 0u )
	{
		g_framegenHistory.pending.push_back( std::move( entry ) );
		return true;
	}

	auto it = std::lower_bound(
		g_framegenHistory.pending.begin(), g_framegenHistory.pending.end(),
		entry.ulTargetFlipNs,
		[]( const FramegenHistory_t::PendingGenerated_t &pending,
			uint64_t targetNs ) {
			return pending.ulTargetFlipNs < targetNs;
		} );
	if ( it == g_framegenHistory.pending.end()
		|| it->ulTargetFlipNs != entry.ulTargetFlipNs )
	{
		g_framegenHistory.pending.insert( it, std::move( entry ) );
		return true;
	}

	if ( entry.bReal )
	{
		if ( !it->bReal
			|| entry.ulPresentRealFrameId >= it->ulPresentRealFrameId )
		{
			*it = std::move( entry );
			return true;
		}
		return false;
	}

	// A real endpoint already owns this opportunity. For two generated
	// candidates, retain the newer interval's pixels.
	if ( it->bReal )
		return false;
	if ( entry.frameId >= it->frameId )
	{
		*it = std::move( entry );
		return true;
	}
	return false;
}

struct FramegenBidirShedResult_t
{
	size_t generated = 0;
	size_t endpoints = 0;

	[[nodiscard]] size_t total() const { return generated + endpoints; }
};

static void framegen_bidir_resync_to_newest_endpoint()
{
	const auto retained = std::ranges::max_element(
		g_framegenHistory.pending, {},
		[]( const FramegenHistory_t::PendingGenerated_t &entry ) {
			return entry.bReal ? entry.ulPresentRealFrameId : 0u;
		} );
	std::optional<FramegenHistory_t::BidirFeedbackEndpoint_t> retainedFeedback;
	uint64_t ulRetainedTargetNs = 0u;
	if ( retained != g_framegenHistory.pending.end() && retained->bReal )
	{
		ulRetainedTargetNs = retained->ulTargetFlipNs;
		const auto feedback = std::ranges::find_if(
			g_framegenHistory.bidirFeedbackEndpoints,
			[&]( const FramegenHistory_t::BidirFeedbackEndpoint_t &record ) {
				return record.endpoint.realFrameId
					== retained->ulPresentRealFrameId;
			} );
		if ( feedback != g_framegenHistory.bidirFeedbackEndpoints.end() )
			retainedFeedback = *feedback;
	}

	g_framegenHistory.ulBidirGridEpoch++;
	if ( g_framegenHistory.ulBidirGridEpoch == 0u )
		g_framegenHistory.ulBidirGridEpoch = 1u;
	g_framegenHistory.bidirFeedbackEndpoints.clear();
	if ( retainedFeedback && ulRetainedTargetNs != 0u )
	{
		g_framegenHistory.bidirEpoch =
			gamescope::framegen::establish_bidir_epoch(
				retainedFeedback->endpoint.sourceReadyNs,
				ulRetainedTargetNs,
				g_framegenHistory.ulBidirGridEpoch );
		retainedFeedback->ulEpoch = g_framegenHistory.bidirEpoch.epoch;
		g_framegenHistory.bidirFeedbackEndpoints.push_back(
			*retainedFeedback );
	}
	else
	{
		g_framegenHistory.bidirEpoch = {};
	}
}

static FramegenBidirShedResult_t framegen_bidir_shed_to_capacity(
	size_t capacity )
{
	if ( g_framegenHistory.pending.size() <= capacity )
		return {};

	std::vector<gamescope::framegen::BidirQueueEntry_t> entries;
	entries.reserve( g_framegenHistory.pending.size() );
	for ( const FramegenHistory_t::PendingGenerated_t &entry : g_framegenHistory.pending )
	{
		entries.push_back( {
			.kind = entry.bReal
				? gamescope::framegen::BidirSlotKind_t::RealEndpoint
				: gamescope::framegen::BidirSlotKind_t::Generated,
			.targetNs = entry.ulTargetFlipNs,
			.realFrameId = entry.ulPresentRealFrameId,
		} );
	}
	const gamescope::framegen::BidirQueueShedPlan_t plan =
		gamescope::framegen::plan_bidir_queue_shed( entries, capacity );
	if ( plan.indices.empty() )
		return {};

	for ( auto it = plan.indices.rbegin(); it != plan.indices.rend(); ++it )
		g_framegenHistory.pending.erase( g_framegenHistory.pending.begin() + *it );

	if ( plan.endpoints != 0u )
	{
		// Dropping a delayed real invalidates feedback ordering from the old
		// timeline. Anchor a new epoch on the newest endpoint that survived, or
		// leave it invalid so the next compatible pair re-primes from live time.
		framegen_bidir_resync_to_newest_endpoint();
	}

	framegen_metrics_note_discard( plan.indices.size() );
	return {
		.generated = plan.generated,
		.endpoints = plan.endpoints,
	};
}

static bool framegen_submit_planned( const FramegenSlotRequest_t *pRequests, uint32_t nRequestCount, uint32_t nGapVblanks, const FramegenEffective_t &eff, uint64_t ulCompositeSeqNo, uint32_t nMaxDegradeSteps, bool bClearPending, const FramegenColorProbeRequest_t *pColorProbe = nullptr, uint64_t ulSingleTargetFlipNs = 0, bool bDeadlineCostKeying = false, uint64_t ulAnchorRealFrameId = 0, bool bExplicitProvisional = false )
{
	if ( nRequestCount == 0 || nGapVblanks == 0 || ulCompositeSeqNo == 0 )
		return false;

	// B4: fold the previous batch's adaptation readback into the adaptation state
	// before recording this one, so its thresholds ride these push constants.
	framegen_adapt_consume( eff.pipeline );
	// Stage C dataset capture: flush the previous batch's training tensors
	// under the same completion gate.
	framegen_record_consume();
	framegen_color_probe_consume();
	// C2 online learning: persist the last profile snapshot, if one completed.
	framegen_net_profile_consume();

	if ( bClearPending )
		g_framegenHistory.pending.clear();

	// Reserve this interval's output slots up front so an empty batch never
	// records/submits a command buffer.
	struct SlotPlan_t { gamescope::Rc<CVulkanTexture> tex; float phase; float strength; uint32_t slotIndex; uint64_t ulTargetFlipNs; uint64_t ulWakeDeadlineNs; };
	std::vector<SlotPlan_t> slots;
	slots.reserve( nRequestCount );
	for ( uint32_t i = 0; i < nRequestCount; i++ )
	{
		gamescope::Rc<CVulkanTexture> pGenerated;
		for ( size_t nProbe = 0; nProbe < g_output.framegenOutputImages.size(); nProbe++ )
		{
			const uint32_t idx = g_framegenHistory.nNextOutputIndex % g_output.framegenOutputImages.size();
			g_framegenHistory.nNextOutputIndex++;
			CVulkanTexture *pCandidate = g_output.framegenOutputImages[ idx ].get();
			// Public texture refs cover pending/current command-buffer use;
			// backend-fb refs cover KMS requests and Wayland compositor acquire
			// lifetime. IsInUse() observes both and has no guessed commit depth.
			if ( pCandidate != nullptr && !pCandidate->IsInUse() )
			{
				pGenerated = pCandidate;
				break;
			}
		}
		if ( !pGenerated )
			continue;

		const uint64_t ulTargetFlipNs = pRequests[ i ].targetFlipNs != 0u
			? pRequests[ i ].targetFlipNs
			: ( nRequestCount == 1 && ulSingleTargetFlipNs != 0
				? ulSingleTargetFlipNs
				: framegen_fixed_slot_target_ns( pRequests[ i ].slotIndex ) );
		slots.push_back( { std::move( pGenerated ), pRequests[ i ].phase,
			pRequests[ i ].strength, pRequests[ i ].slotIndex, ulTargetFlipNs,
			pRequests[ i ].wakeDeadlineNs } );
	}
	if ( slots.size() != nRequestCount )
	{
		static uint64_t s_uOutputPressureDebugLogCounter = 0;
		if ( FramegenDebugShouldLog( s_uOutputPressureDebugLogCounter ) )
		{
			const FramegenImagePoolPressure_t pressure = framegen_image_pool_pressure(
				g_output.framegenOutputImages );

			vk_log.infof( "framegen: output-pool pressure admitted=%zu/%u pool=%zu pending=%zu texture-busy=%zu refs=%" PRIu64 " backend-busy=%zu external-refs=%" PRIu64,
				slots.size(), nRequestCount, g_output.framegenOutputImages.size(), g_framegenHistory.pending.size(),
				pressure.nTextureBusy, pressure.ulTextureRefs,
				pressure.nBackendBusy, pressure.ulBackendExternalRefs );
		}
	}

	if ( slots.empty() )
		return false;
	if ( pColorProbe != nullptr
		&& ( slots.size() != k_uFramegenColorProbeCandidates
			|| pColorProbe->reference == nullptr
			|| !framegen_color_probe_prepare( slots[ 0 ].tex->width(), slots[ 0 ].tex->height(),
				slots[ 0 ].tex->drmFormat(), pColorProbe->eotf ) ) )
		return false;

	// Record the whole interval's generation into ONE command buffer submitted
	// once: the shared motion intermediates are serialized by the per-command-
	// buffer barrier tracking, all slots share a single framegen seqNo, and the
	// batch draws from the isolated framegen descriptor ring (see markFramegen).
	std::unique_ptr<CVulkanCmdBuffer> pCmdBuffer = g_device.commandBuffer( true );
	pCmdBuffer->markFramegen();
	const FramegenDispatch_t &dispatch = framegen_dispatch_for_format( g_framegenHistory.drmFormat );
	const bool bBidir = vulkan_framegen_bidir_active();
	const bool bMotionRequested = eff.mode == GamescopeFramegenMode::Motion
		&& dispatch.motionSupported;
	// The classic/deadline/idle planners can submit several batches for one real
	// interval. Re-estimating and re-training on that identical pair both wastes
	// the deadline and statistically overweights slow intervals. The finalized
	// field remains resident until any new preparation explicitly invalidates it.
	const bool bReuseMotion = pColorProbe == nullptr
		&& bMotionRequested
		&& g_framegenMotion.uMotionFieldFrameId != 0
		&& g_framegenMotion.uMotionFieldFrameId == g_framegenHistory.currentFrameId
		&& g_framegenMotion.eMotionFieldPipeline == eff.pipeline
		&& g_framegenMotion.bMotionFieldBidir == bBidir
		&& framegen_motion_field() != nullptr
		&& ( !bBidir || framegen_motion_field_rev() != nullptr );
	// Classic batches keep their historical generated-count key and exclude
	// cached refills. Deadline pacing measures cached warps too, but under their
	// own work-class key so they cannot make full preparation look cheaper.
	const gamescope::framegen::DeadlineWorkClass_t eDeadlineWorkClass = bReuseMotion
		? gamescope::framegen::DeadlineWorkClass_t::CachedWarp
		: gamescope::framegen::DeadlineWorkClass_t::FullPreparationAndWarp;
	const int nQuerySlot = !bReuseMotion || bDeadlineCostKeying
		? g_device.framegenTimestampBegin( pCmdBuffer.get() ) : -1;

	// Preserve the preceding pair's finalized displacement before preparation
	// overwrites the working fields. The exact pipeline/mode and consecutive-ID
	// gates keep estimator changes or a skipped interval from looking like
	// physical acceleration. The copy and later history reads are ordered in
	// this same command buffer.
	if ( !bReuseMotion && pColorProbe == nullptr && bMotionRequested && !bBidir
		&& eff.pipeline >= GamescopeFramegenPipeline::Predict )
	{
		const bool bHistorySourceValid = g_framegenMotion.mvFieldHistory != nullptr
			&& g_framegenMotion.uMotionFieldFrameId != 0
			&& g_framegenMotion.uMotionFieldFrameId + 1u == g_framegenHistory.currentFrameId
			&& g_framegenMotion.eMotionFieldPipeline == eff.pipeline
			&& !g_framegenMotion.bMotionFieldBidir
			&& framegen_motion_field() != nullptr;
		if ( bHistorySourceValid )
		{
			pCmdBuffer->copyImage( framegen_motion_field(), g_framegenMotion.mvFieldHistory );
			g_framegenMotion.uMotionHistoryFrameId = g_framegenMotion.uMotionFieldFrameId;
			g_framegenMotion.uMotionHistoryIntervalNs = g_framegenMotion.uMotionFieldIntervalNs;
		}
		else
		{
			g_framegenMotion.uMotionHistoryFrameId = 0;
			g_framegenMotion.uMotionHistoryIntervalNs = 0;
		}
	}

	// Motion estimation depends only on the two real frames, so it is computed
	// once for the real pair; later batches reuse its finalized field. Falls
	// back to extrapolation for every slot if the
	// intermediates can't be allocated. The ladder's effective mode (not the base
	// global) selects the pass, so motion can be shed under GPU pressure without
	// disturbing vulkan_framegen_is_enabled() or the forced-composite tax.
	const bool bMotion = bReuseMotion || ( bMotionRequested
		&& framegen_prepare_motion( pCmdBuffer.get(), g_framegenHistory.currentReal->width(),
			g_framegenHistory.currentReal->height(), dispatch, eff.pipeline ) );
	// Held-out E2 samples grade the actual motion path, not its allocation or
	// capability fallback. A failed motion setup drops the sample and leaves all
	// real-frame presentation untouched.
	if ( pColorProbe != nullptr && !bMotion )
		return false;

	// Stage C: with the checked fields final, capture the RAW training tensors
	// (pre-refinement, pre-trust — what the net must learn to improve on),
	// then refine both fields through the net. Everything downstream — the B4
	// probe and the warps — binds the refined copies via
	// framegen_motion_field() once the net has run.
	const bool bNetRecord = bMotion && !bReuseMotion && g_framegenMotion.recField != nullptr
		&& g_uFramegenRecordCount < framegen_record_max();
	if ( bNetRecord )
	{
		pCmdBuffer->copyImage( g_framegenMotion.lumaPrev, g_framegenMotion.recLumaPrev );
		pCmdBuffer->copyImage( g_framegenMotion.lumaCur, g_framegenMotion.recLumaCur );
		pCmdBuffer->copyImage( g_framegenMotion.mvField, g_framegenMotion.recField );
		pCmdBuffer->copyImage( g_framegenMotion.mvFieldRevChk, g_framegenMotion.recFieldRev );
	}
	if ( bMotion && !bReuseMotion && g_framegenMotion.mvFieldNet != nullptr )
		framegen_record_net( pCmdBuffer.get(), g_framegenMotion.width, g_framegenMotion.height, bBidir );
	// B4: with the field final, record the adaptation probe — the warps below
	// read its verdict in this same batch, the CPU next batch.
	const bool bAdaptProbe = bMotion && !bReuseMotion && framegen_adapt_enabled( eff.pipeline );
	if ( bAdaptProbe )
		framegen_record_adapt_probe( pCmdBuffer.get(), g_framegenMotion.width, g_framegenMotion.height );

	// C2: train after the probe so a content cut can zero every gradient slice
	// before Adam sees it. Same-batch inference still used the pre-step weights;
	// the optimizer publishes only for the next batch. When adaptation is
	// explicitly disabled, training retains its previous unguarded behavior.
	const bool bNetStateReadback = pColorProbe == nullptr
		&& !bReuseMotion && g_framegenMotion.bNetActive && framegen_net_online_enabled()
		&& framegen_record_net_train( pCmdBuffer.get(), eff.pipeline, bAdaptProbe );
	if ( bMotion && !bReuseMotion && pColorProbe == nullptr )
	{
		g_framegenMotion.uMotionFieldFrameId = g_framegenHistory.currentFrameId;
		g_framegenMotion.uMotionFieldIntervalNs =
			g_framegenHistory.currentPresentTimeNs > g_framegenHistory.previousPresentTimeNs
			? g_framegenHistory.currentPresentTimeNs - g_framegenHistory.previousPresentTimeNs : 0;
		g_framegenMotion.eMotionFieldPipeline = eff.pipeline;
		g_framegenMotion.bMotionFieldBidir = bBidir;
	}
	if ( bMotion )
	{
		// Each warp reads the shared motion field(s) and history; keep per-slot.
		// Bidir slots interpolate at their exact phase (no user-strength scaling
		// — the phase IS the temporal placement between the two real frames).
		for ( size_t i = 0; i < slots.size(); i++ )
		{
			const SlotPlan_t &slot = slots[ i ];
			if ( bBidir )
			{
				const bool bProbeCandidate = pColorProbe != nullptr
					&& i < k_uFramegenColorProbeCandidates;
				float flOneSidedOverride = -1.0f;
				float flEndpointTraceOverride = -1.0f;
				if ( bProbeCandidate )
				{
					float &flOverride = pColorProbe->sweep == FramegenColorProbeSweep::EndpointTrace
						? flEndpointTraceOverride : flOneSidedOverride;
					flOverride = k_flFramegenColorProbeStrengths[ i ];
				}
				framegen_bidir_warp_slot( pCmdBuffer.get(), slot.tex,
					std::clamp( slot.phase, 0.0f, 1.0f ), eff.pipeline,
					flOneSidedOverride, flEndpointTraceOverride );
			}
			else
				framegen_warp_slot( pCmdBuffer.get(), slot.tex, slot.strength, eff.pipeline );
		}
	}
	else if ( eff.mode == GamescopeFramegenMode::Blend || bBidir )
	{
		// Bidir's degraded rung is the plain crossfade, NOT extrapolation: the
		// delayed timeline places every slot BETWEEN the two real frames, and a
		// crossfade is the cheapest phase-correct content for that position.
		// (An extrapolated slot would predict PAST the newest frame — content
		// from a different timeline that would visibly jump.)
		for ( const SlotPlan_t &slot : slots )
			framegen_bind_blend( pCmdBuffer.get(), slot.tex, std::clamp( slot.phase, 0.0f, 1.0f ) );
	}
	else
	{
		// Extrapolate: fuse slots in pairs so the two full-resolution history
		// images are read once per pair instead of once per slot (the dominant
		// cost at x3/x4). An odd final slot falls back to the single-slot shader.
		size_t i = 0;
		for ( ; i + 1 < slots.size(); i += 2 )
		{
			framegen_bind_extrapolate_pair( pCmdBuffer.get(), dispatch.extrapolatePair,
				slots[ i ].tex, slots[ i + 1 ].tex, slots[ i ].strength, slots[ i + 1 ].strength );
		}
		if ( i < slots.size() )
			framegen_bind_extrapolate( pCmdBuffer.get(), dispatch.extrapolate, slots[ i ].tex, slots[ i ].strength );
	}

	// Preserve two-interval-old real-frame luma without extending the
	// output-ring lifetime. Every warp above reads the old reservoir first;
	// this same-command-buffer copy is ordered after those reads and publishes
	// lumaPrev for the next consecutive Guided batch. At 1/8 resolution this
	// is 1/64 the texel traffic of copying the full color frame. Two tiny images
	// retain both current-2 (for deadline/refill slots of this same interval) and
	// current-1 (for the next interval); a refill finds the latter already
	// published and records no redundant copy.
	if ( bMotion && !bBidir && eff.pipeline == GamescopeFramegenPipeline::Guided
		&& g_framegenMotion.lumaReservoir[0] != nullptr
		&& g_framegenMotion.lumaReservoir[1] != nullptr )
	{
		int nExisting = -1;
		int nRead = -1;
		for ( uint32_t i = 0; i < 2; i++ )
		{
			if ( g_framegenMotion.uLumaReservoirFrameId[i] == g_framegenHistory.previousFrameId )
				nExisting = (int)i;
			if ( g_framegenMotion.uLumaReservoirFrameId[i] != 0
				&& g_framegenMotion.uLumaReservoirFrameId[i] + 2u == g_framegenHistory.currentFrameId )
				nRead = (int)i;
		}
		if ( nExisting < 0 )
		{
			const int nTarget = nRead == 0 ? 1 : 0;
			pCmdBuffer->copyImage( g_framegenMotion.lumaPrev, g_framegenMotion.lumaReservoir[nTarget] );
			g_framegenMotion.uLumaReservoirFrameId[nTarget] = g_framegenHistory.previousFrameId;
		}
	}

	g_device.framegenTimestampEnd( pCmdBuffer.get(), nQuerySlot );
	if ( pColorProbe != nullptr )
	{
		// Readback copies deliberately sit after the end timestamp: E2 measures
		// algorithm cost, not capture-tool traffic. The submission completion still
		// covers the copies, so mapped memory is never consumed early.
		for ( uint32_t i = 0; i < k_uFramegenColorProbeCandidates; i++ )
			pCmdBuffer->copyImage( slots[ i ].tex, g_framegenColorProbe.generatedReadback[ i ] );
		pCmdBuffer->copyImage( pColorProbe->reference, g_framegenColorProbe.referenceReadback );
	}
	// Step 3 reuses the existing small timestamp table, but causal one-slot work
	// is keyed by preparation shape instead of the now-constant output count.
	const uint32_t nCostKey = bDeadlineCostKeying
		? gamescope::framegen::deadline_work_class_cost_key( eDeadlineWorkClass )
		: (uint32_t)slots.size();
	// Split-family base mode ingests the new history on the COMPOSITE queue, at a
	// scratch-timeline point strictly later than the composite this batch was
	// planned against. Waiting on the composite alone would let the framegen
	// queue sample previousReal/currentReal while that copy is still writing it.
	// lastBaseIngestSeqNo is zero on every other path, so this is a no-op there.
	const uint64_t ulWaitCompositeSeqNo =
		std::max( ulCompositeSeqNo, g_framegenHistory.lastBaseIngestSeqNo );
	const uint64_t ulSeqNo = g_device.submitFramegen( std::move( pCmdBuffer ),
		ulWaitCompositeSeqNo, nQuerySlot, g_framegenHistory.nDegradeSteps, nCostKey );
	g_framegenHistory.lastFramegenWorkSeqNo = ulSeqNo;

	for ( const SlotPlan_t &slot : slots )
	{
		if ( pColorProbe != nullptr )
			break;
		FramegenHistory_t::PendingGenerated_t entry;
		entry.tex = slot.tex;
		entry.seqNo = ulSeqNo;
		entry.frameId = g_framegenHistory.currentFrameId;
		entry.ulPresentRealFrameId = g_framegenPresentState.ulCurrentRealFrameId;
		entry.ulSlotId = framegen_next_present_slot_id();
		entry.ulCompositeSeqNo = ulCompositeSeqNo;
		entry.ulTargetFlipNs = slot.ulTargetFlipNs;
		entry.ulWakeDeadlineNs = slot.ulWakeDeadlineNs;
		entry.ulAnchorRealFrameId = ulAnchorRealFrameId != 0u
			? ulAnchorRealFrameId
			: ( bDeadlineCostKeying
				? g_framegenHistory.causalAnchor.realFrameId : 0u );
		entry.phase = slot.phase;
		// Only when BOTH endpoints this batch interpolated/extrapolated from
		// were cursor-free may the present side draw the live cursor: if either
		// still had one baked in, the warped copy is already in the pixels and
		// a second one would double it.
		entry.bCursorFree = g_framegenHistory.bCursorFreePrevious
			&& g_framegenHistory.bCursorFreeCurrent;
		entry.bProvisional = bExplicitProvisional
			|| ( bDeadlineCostKeying && ulAnchorRealFrameId == 0u
				&& !g_framegenHistory.causalAnchor.correctedFlipNs.has_value() );
		framegen_insert_pending_entry( std::move( entry ) );
		g_framegenHistory.nLastGeneratedSlot = std::max( g_framegenHistory.nLastGeneratedSlot, slot.slotIndex );
	}

	g_framegenHistory.lastGeneratedSeqNo = ulSeqNo;
	g_framegenHistory.nLastGenerationGapVblanks = nGapVblanks;
	if ( bAdaptProbe )
		g_framegenHistory.ulAdaptStatsSeqNo = ulSeqNo;
	if ( bNetRecord )
		g_framegenHistory.ulNetRecordSeqNo = ulSeqNo;
	if ( bNetStateReadback )
		g_framegenHistory.ulNetProfileSeqNo = ulSeqNo;
	if ( pColorProbe != nullptr )
	{
		FramegenColorProbeResources_t &c = g_framegenColorProbe;
		c.pendingSeqNo = ulSeqNo;
		c.pendingAnchorId = pColorProbe->anchorId;
		c.pendingReferenceId = pColorProbe->referenceId;
		c.pendingEndpointId = pColorProbe->endpointId;
		c.pendingAnchorTimeNs = pColorProbe->anchorTimeNs;
		c.pendingReferenceTimeNs = pColorProbe->referenceTimeNs;
		c.pendingEndpointTimeNs = pColorProbe->endpointTimeNs;
		c.pendingPhase = slots[ 0 ].phase;
		c.pendingSweep = pColorProbe->sweep;
	}

	// Pin this batch's input slots until it finishes reading them, so a later
	// composite can't reuse a slot the framegen queue is still sampling even
	// after history is invalidated. The oversubscription guard admits only one
	// batch at a time, so these always match the current (previousReal,
	// currentReal) while incomplete. E2 additionally pins its exact held-out
	// reference, for a maximum of three slots only in capture mode.
	g_framegenHistory.genReadA = g_framegenHistory.previousReal;
	g_framegenHistory.genReadB = g_framegenHistory.currentReal;
	g_framegenHistory.genReadReference = pColorProbe != nullptr ? pColorProbe->reference : nullptr;
	g_framegenHistory.genReadSeqNo = ulSeqNo;

	const uint32_t nGeneratedCount = (uint32_t)slots.size();

	static uint64_t s_uGeneratedDebugLogCounter = 0;
	if ( FramegenDebugShouldLog( s_uGeneratedDebugLogCounter ) )
	{
		vk_log.infof( "framegen: %s %u frame(s) for real id=%" PRIu64 " gapVblanks=%u mode=%s/%s(x%u) degrade=%u/%u gpu=%.2fms%s queue family %u",
			pColorProbe != nullptr ? "captured held-out" : "generated",
			nGeneratedCount,
			g_framegenHistory.currentFrameId,
			nGapVblanks,
			gamescope::framegen::mode_name( bMotion ? GamescopeFramegenMode::Motion : eff.mode ),
			gamescope::framegen::pipeline_name( eff.pipeline ),
			eff.multiplier,
			g_framegenHistory.nDegradeSteps,
			nMaxDegradeSteps,
			g_device.framegenLastGpuTimeNs() / 1.0e6,
			bReuseMotion
				? ( bDeadlineCostKeying ? " (cached-field warp)"
					: " (prior full batch; cached-field warp unmeasured)" )
				: "",
			g_device.queueFamily() );
	}

	// Under active VRR hybrid the pending slot is shown by the mid-interval
	// timer, not by the next paint: VRR wakes always "can vblank", so forcing
	// a repaint here would present the prediction on the very next wake —
	// immediately after the real frame — collapsing the midpoint spacing to
	// ~zero. The timer wake is the (only) present trigger in that mode.
	if ( pColorProbe == nullptr && !vulkan_framegen_vrr_hybrid_active() )
		force_repaint();
	return true;
}

static bool framegen_submit_batch( uint32_t nFirstSlot, uint32_t nGapVblanks, uint32_t nGenerate, const FramegenEffective_t &eff, uint64_t ulCompositeSeqNo, uint32_t nMaxDegradeSteps, bool bClearPending )
{
	if ( nGenerate == 0 || nGapVblanks == 0 )
		return false;

	// Classic gap-count planning: slot k of an N-vblank gap sits at phase k/N.
	// The phase here is a prediction baked from a measured-gap guess; contrast
	// with framegen_causal_submit below, where the deadline planner supplies it.
	std::vector<FramegenSlotRequest_t> requests;
	requests.reserve( nGenerate );
	for ( uint32_t i = 0; i < nGenerate; i++ )
	{
		const uint32_t k = nFirstSlot + i;
		// Effective forward coefficient: temporal placement scaled by the user
		// strength (0.5 is neutral, reproducing the classic x2 half-way step).
		// Idle refill can move past the originally expected next-real slot when
		// the game stalls, but never lets prediction run away unbounded.
		requests.push_back( gamescope::framegen::classic_slot_request(
			k, nGapVblanks, g_flFramegenStrength,
			k_flFramegenMaxForwardStrength ) );
	}

	return framegen_submit_planned( requests.data(), (uint32_t)requests.size(), nGapVblanks, eff, ulCompositeSeqNo, nMaxDegradeSteps, bClearPending );
}

static gamescope::framegen::DeadlineWorkClass_t framegen_causal_work_class(
	const FramegenEffective_t &eff )
{
	const FramegenDispatch_t &dispatch = framegen_dispatch_for_format(
		g_framegenHistory.drmFormat );
	const bool bCachedMotion = eff.mode == GamescopeFramegenMode::Motion
		&& dispatch.motionSupported
		&& g_framegenMotion.uMotionFieldFrameId != 0
		&& g_framegenMotion.uMotionFieldFrameId == g_framegenHistory.currentFrameId
		&& g_framegenMotion.eMotionFieldPipeline == eff.pipeline
		&& !g_framegenMotion.bMotionFieldBidir
		&& framegen_motion_field() != nullptr;
	return bCachedMotion
		? gamescope::framegen::DeadlineWorkClass_t::CachedWarp
		: gamescope::framegen::DeadlineWorkClass_t::FullPreparationAndWarp;
}

struct FramegenCausalRungSelection_t
{
	FramegenEffective_t eff;
	gamescope::framegen::DeadlineWorkClass_t workClass =
		gamescope::framegen::DeadlineWorkClass_t::FullPreparationAndWarp;
	uint64_t ulCostNs = 0;
	uint32_t uSamples = 0;
	bool bAdmit = false;
	bool bCommittedRung = false;
	bool bCommitReservationShortfall = false;
	bool bDeadlineMissSkip = false;
};

static FramegenCausalRungSelection_t framegen_select_causal_rung(
	const gamescope::framegen::CausalSlotPlan_t &plan, uint64_t ulNowNs,
	uint64_t ulPlainWakeNs, bool bNativeCommitDeadline,
	uint64_t ulNativeSlotBudgetNs )
{
	FramegenCausalRungSelection_t result;
	// A multiplier notch cannot reduce a one-slot submission. Stop this ladder
	// after the motion-pipeline/mode rungs; the configured multiplier remains a
	// resource and density ceiling for the untouched paths.
	const uint32_t nMaxCausalDegradeSteps = gamescope::framegen::max_degrade_steps(
		g_eFramegenMode, g_eFramegenPipeline, 2 );
	g_framegenHistory.nDegradeSteps = std::min(
		g_framegenHistory.nDegradeSteps, nMaxCausalDegradeSteps );
	const uint64_t ulStartEstimateNs = plan.provisional
		? g_framegenHistory.causalAnchor.provisional_start_estimate() : 0u;

	const auto sampleRung = [&]( uint32_t nRung )
	{
		FramegenCausalRungSelection_t sample;
		sample.eff = framegen_effective_config( nRung );
		sample.workClass = framegen_causal_work_class( sample.eff );
		const uint32_t nCostKey = gamescope::framegen::deadline_work_class_cost_key(
			sample.workClass );
		sample.ulCostNs = g_device.framegenRungCostNs( nRung, nCostKey );
		sample.uSamples = g_device.framegenRungSampleCount( nRung, nCostKey );
		return sample;
	};
	const auto sampleFitsAt = [&]( const FramegenCausalRungSelection_t &sample,
		uint64_t ulWakeNs )
	{
		const bool bMature = sample.ulCostNs != 0u
			&& sample.uSamples >= gamescope::framegen::k_uDeadlineMinSamples;
		return !bMature || gamescope::framegen::deadline_cost_fits(
			sample.ulCostNs, ulWakeNs, ulNowNs, ulStartEstimateNs );
	};
	const auto sampleFits = [&]( const FramegenCausalRungSelection_t &sample )
	{
		return sampleFitsAt( sample, plan.wakeNs );
	};

	const uint32_t nCurrentRung = g_framegenHistory.nDegradeSteps;
	const bool bPriorDeadlineMiss = g_framegenHistory.bCausalDeadlineMissed;
	g_framegenHistory.bCausalDeadlineMissed = false;
	result = sampleRung( nCurrentRung );
	bool bDegradeForPriorMiss = bPriorDeadlineMiss;
	if ( ulNativeSlotBudgetNs != 0u )
	{
		const gamescope::framegen::DeadlineMissEvaluation_t miss =
			gamescope::framegen::evaluate_deadline_miss_hysteresis(
				g_framegenHistory.causalDeadlineMisses,
				bPriorDeadlineMiss,
				g_framegenPresentState.bPresentBiasHitchEpisode,
				result.ulCostNs, result.uSamples,
				ulNativeSlotBudgetNs );
		g_framegenHistory.causalDeadlineMisses = miss.state;
		bDegradeForPriorMiss = miss.degrade;
		// A hitch or one isolated miss consumes this opportunity honestly. It
		// does not move the rung or erase recovery evidence here; repeated misses
		// (or a measured slot-capacity failure) take the normal degradation path.
		if ( miss.skip )
		{
			result.bDeadlineMissSkip = true;
			return result;
		}
	}
	// When the KMS commit reservation is earlier than the ordinary compositor
	// wake, it may make this one slot unavailable even though the current rung
	// still fits the plain display-grid opportunity. Skip that slot without
	// teaching the pipeline ladder that the rung itself is too expensive.
	const bool bCurrentCostMature = result.ulCostNs != 0u
		&& result.uSamples >= gamescope::framegen::k_uDeadlineMinSamples;
	if ( !bDegradeForPriorMiss && bNativeCommitDeadline && bCurrentCostMature
		&& gamescope::framegen::commit_reservation_only_shortfall(
			result.ulCostNs, plan.wakeNs, ulPlainWakeNs,
			ulNowNs, ulStartEstimateNs ) )
	{
		result.bCommitReservationShortfall = true;
		return result;
	}
	if ( !bDegradeForPriorMiss && sampleFits( result ) )
	{
		result.bAdmit = true;
		if ( framegen_recovery_active_for_path() )
		{
			const uint64_t ulDecisionStartNs = std::max(
				ulNowNs, ulStartEstimateNs );
			const uint64_t ulRecoveryBudgetNs = ulNativeSlotBudgetNs != 0u
				? ulNativeSlotBudgetNs
				: ( plan.wakeNs > ulDecisionStartNs
					? gamescope::framegen::deadline_budget_ns(
						plan.wakeNs - ulDecisionStartNs )
					: 0u );
			std::array<gamescope::framegen::LadderRungCost_t,
				CVulkanDevice::kFramegenLadderSlots> rungCosts = {};
			for ( uint32_t nRung = 0u; nRung < nCurrentRung; nRung++ )
			{
				const FramegenCausalRungSelection_t sample = sampleRung( nRung );
				rungCosts[ nRung ] = { sample.ulCostNs, sample.uSamples };
			}
			const gamescope::framegen::LadderRecoveryTarget_t recoveryTarget =
				gamescope::framegen::select_ladder_recovery_target(
					rungCosts, nCurrentRung, ulRecoveryBudgetNs );
			const gamescope::framegen::LadderRecoveryEvaluation_t recovery =
				gamescope::framegen::evaluate_ladder_recovery(
					g_framegenHistory.recovery, nCurrentRung,
					g_framegenHistory.nDegradeHold,
					result.ulCostNs, result.uSamples,
					recoveryTarget.evidence.costNs,
					recoveryTarget.evidence.samples,
					ulRecoveryBudgetNs );
			g_framegenHistory.recovery = recovery.state;
			if ( recovery.reportBlockedThreshold )
				framegen_log_ladder_recovery_blocked(
					nMaxCausalDegradeSteps,
					g_framegenHistory.nDegradeHold );
			if ( recovery.tryRecover && recoveryTarget.rung < nCurrentRung )
			{
				const uint32_t uEvidenceDecisions = recovery.state.streak;
				g_framegenHistory.nDegradeSteps = recoveryTarget.rung;
				g_framegenHistory.recovery =
					gamescope::framegen::commit_ladder_recovery(
						g_framegenHistory.recovery );
				framegen_log_ladder_recovery(
					nMaxCausalDegradeSteps, nCurrentRung,
					uEvidenceDecisions );
				FramegenCausalRungSelection_t richer = sampleRung(
					recoveryTarget.rung );
				richer.bAdmit = true;
				richer.bCommittedRung = true;
				return richer;
			}
		}
		return result;
	}

	// The admission gate above remains tied to this decision's actual remaining
	// time. If the mature rung fits a normally planned native slot, a late-phase
	// decision is not capacity evidence: skip it without degrading or resetting
	// the recovery streak, and try again at the normal phase.
	const uint64_t ulDecisionStartNs = std::max(
		ulNowNs, ulStartEstimateNs );
	const uint64_t ulRemainingBudgetNs = plan.wakeNs > ulDecisionStartNs
		? gamescope::framegen::deadline_budget_ns(
			plan.wakeNs - ulDecisionStartNs )
		: 0u;
	if ( !bDegradeForPriorMiss && ulNativeSlotBudgetNs != 0u
		&& gamescope::framegen::deadline_shortfall_is_phase_only(
			result.ulCostNs, result.uSamples,
			ulRemainingBudgetNs, ulNativeSlotBudgetNs ) )
		return result;

	// Retain the existing post-degradation hold. Known work that misses this
	// slot's actual remaining budget is skipped rather than submitted late; each
	// such decision still advances the hold just as the old per-frame evaluator
	// did, so the ladder cannot deadlock at an over-budget rung.
	if ( !bDegradeForPriorMiss && g_framegenHistory.nDegradeHold > 0u )
	{
		g_framegenHistory.nDegradeHold--;
		framegen_recovery_note_capacity_failure();
		return result;
	}

	// Select one new monotonic rung for this decision. Mature non-fitting rungs
	// can be passed over; the first cold rung gets one non-blocking probe. If the
	// cheapest measured rung still cannot fit, settle there and skip honestly.
	for ( uint32_t nRung = nCurrentRung + 1u;
		nRung <= nMaxCausalDegradeSteps; nRung++ )
	{
		FramegenCausalRungSelection_t candidate = sampleRung( nRung );
		const bool bMature = candidate.ulCostNs != 0u
			&& candidate.uSamples >= gamescope::framegen::k_uDeadlineMinSamples;
		if ( sampleFits( candidate ) )
		{
			candidate.bAdmit = true;
			candidate.bCommittedRung = true;
			g_framegenHistory.nDegradeSteps = nRung;
			g_framegenHistory.nDegradeHold = gamescope::framegen::k_uDeadlineHoldFrames;
			framegen_recovery_note_degradation();
			return candidate;
		}
		if ( !bMature )
			break;
		result = candidate;
	}

	const bool bDegraded = nCurrentRung < nMaxCausalDegradeSteps;
	g_framegenHistory.nDegradeSteps = nMaxCausalDegradeSteps;
	g_framegenHistory.nDegradeHold = 0u;
	if ( bDegraded )
		framegen_recovery_note_degradation();
	else
		framegen_recovery_note_capacity_failure();
	return result;
}

static void framegen_record_causal_anchor( uint64_t ulRealFrameId,
	uint64_t ulSourceReadyNs, uint64_t ulProvisionalTargetNs,
	uint64_t ulVblankIntervalNs, bool bSourceTimestampsReliable )
{
	const uint64_t ulDisplayChainGeneration =
		g_framegenPresentState.displayTiming.generation;
	const bool bGridChanged = g_framegenHistory.ulDeadlineGridEpoch == 0u
		|| g_framegenHistory.ulDeadlineGridIntervalNs != ulVblankIntervalNs
		|| g_framegenHistory.ulDeadlineDisplayChainGeneration
			!= ulDisplayChainGeneration;
	const bool bProvenanceChanged =
		g_framegenHistory.bDeadlineProvenanceInitialized
		&& g_framegenHistory.bDeadlineSourceTimestampsReliable
			!= bSourceTimestampsReliable;
	if ( bGridChanged || bProvenanceChanged )
	{
		g_framegenHistory.ulDeadlineGridEpoch++;
		if ( g_framegenHistory.ulDeadlineGridEpoch == 0u )
			g_framegenHistory.ulDeadlineGridEpoch = 1u;
	}
	g_framegenHistory.ulDeadlineGridIntervalNs = ulVblankIntervalNs;
	g_framegenHistory.ulDeadlineDisplayChainGeneration =
		ulDisplayChainGeneration;
	g_framegenHistory.bDeadlineProvenanceInitialized = true;
	g_framegenHistory.bDeadlineSourceTimestampsReliable =
		bSourceTimestampsReliable;
	g_framegenHistory.ulLastPlannedTargetNs = 0u;
	g_framegenHistory.causalAnchor = {
		.realFrameId = ulRealFrameId,
		.sourceReadyNs = ulSourceReadyNs,
		.provisionalTargetNs = gamescope::framegen::apply_present_bias_ns(
			ulProvisionalTargetNs,
			g_framegenPresentState.displayTiming.presentBias.emaNs ),
		.provisionalBiasNs =
			g_framegenPresentState.displayTiming.presentBias.emaNs,
		.correctedFlipNs = std::nullopt,
		.epoch = g_framegenHistory.ulDeadlineGridEpoch,
	};
}

// Default fixed-refresh causal path: ask the pure planner for exactly one
// unused display-grid slot, apply its per-slot budget to the measured work
// class, then pass its exact phase and target through the common submit path.
static bool framegen_causal_submit( uint64_t ulCompositeSeqNo )
{
	if ( !vulkan_framegen_is_enabled() || !framegen_causal_deadline_enabled()
		|| vulkan_framegen_vrr_hybrid_active() || vulkan_framegen_bidir_active()
		|| !g_framegenHistory.valid || !g_framegenHistory.pending.empty()
		|| g_framegenHistory.previousReal == nullptr || g_framegenHistory.currentReal == nullptr
		|| framegen_predicted_interval_ns() == 0u
		|| g_framegenHistory.causalAnchor.realFrameId == 0u
		|| ulCompositeSeqNo == 0u )
		return false;

	// One batch in flight, always: the lockless descriptor/timestamp rings and
	// the cross-queue read pins all depend on it.
	if ( !g_device.hasCompletedFramegen( g_framegenHistory.lastGeneratedSeqNo ) )
		return false;

	const uint64_t now = get_time_in_nanos();
	if ( now > g_framegenHistory.currentPresentTimeNs
		&& now - g_framegenHistory.currentPresentTimeNs > k_ulFramegenMaxRealFrameGapNs )
	{
		vulkan_framegen_invalidate_history( "idle_frame_gap" );
		return false;
	}

	const uint64_t ulVblankIntervalNs = g_framegenHistory.ulDeadlineGridIntervalNs;
	const gamescope::VBlankScheduleTime schedule =
		GetVBlankTimer().CalcNextWakeupTime( true );
	const gamescope::framegen::DisplayGrid_t rawGrid = {
		.D0 = schedule.ulTargetVBlank,
		.W0 = schedule.ulScheduledWakeupPoint,
		.T = ulVblankIntervalNs,
	};
	const gamescope::framegen::DisplayGrid_t plainGrid =
		gamescope::framegen::apply_present_bias(
			rawGrid,
			g_framegenPresentState.displayTiming.presentBias.emaNs );
	// The ordinary grid wake reserves enough time to render a full compositor
	// frame. A native generated-only present instead has to finish by its actual
	// KMS commit deadline. Use that deadline for both slot selection and cost
	// admission; nested backends retain the byte-for-byte plain-grid path.
	gamescope::framegen::DisplayGrid_t grid = plainGrid;
	const bool bNativeKmsTiming = GetBackend() != nullptr
		&& GetBackend()->OwnsKMSPresentTiming();
	const uint64_t ulPresentMarginNs = ulVblankIntervalNs / 10u;
	const gamescope::framegen::FixedRefreshCommitPlan_t gridCommitPlan =
		bNativeKmsTiming
			? framegen_plan_fixed_refresh_commit(
				plainGrid.D0,
				ulPresentMarginNs, ulVblankIntervalNs )
			: gamescope::framegen::FixedRefreshCommitPlan_t{};
	if ( gridCommitPlan.earlyCommit )
		grid.W0 = gridCommitPlan.commitDeadlineNs;
	const gamescope::framegen::CausalSlotPlan_t plan =
		gamescope::framegen::plan_next_causal_slot(
			grid, g_framegenHistory.causalAnchor, g_framegenHistory.cadence, {
				.nowNs = now,
				.afterTargetNs = g_framegenHistory.ulLastPlannedTargetNs,
				.gridEpoch = g_framegenHistory.ulDeadlineGridEpoch,
				.presentBiasNs =
					g_framegenPresentState.displayTiming.presentBias.emaNs,
				.configuredStrength = g_flFramegenStrength,
				.forwardStrengthCap = k_flFramegenMaxForwardStrength,
				.sourceTimestampsReliable = g_framegenHistory.bCadenceUsesSourceTime,
				.dedicatedQueue = true,
			} );
	if ( !plan.admit )
	{
		if ( plan.skipReason
			== gamescope::framegen::DeadlineSkipReason_t::NextRealSafelyDue )
			framegen_metrics_note_admission_skip();
		return false;
	}
	const uint64_t ulStartEstimateNs = plan.provisional
		? g_framegenHistory.causalAnchor.provisional_start_estimate() : 0u;
	if ( std::max( now, ulStartEstimateNs ) >= plan.wakeNs )
		return false;

	const uint64_t ulGridIndex = gamescope::framegen::grid_index_at_or_after(
		plainGrid, plan.targetNs );
	const uint64_t ulPlainWakeNs = plainGrid.wake( ulGridIndex );
	const gamescope::framegen::FixedRefreshCommitPlan_t commitPlan =
		gridCommitPlan.earlyCommit
			? framegen_plan_fixed_refresh_commit(
				plan.targetNs,
				ulPresentMarginNs, ulVblankIntervalNs )
			: gamescope::framegen::FixedRefreshCommitPlan_t{};
	const uint64_t ulNativeSlotBudgetNs = !bNativeKmsTiming
		? 0u
		: commitPlan.earlyCommit
			? gamescope::framegen::fixed_refresh_slot_budget_ns(
				ulVblankIntervalNs, commitPlan.presentLeadNs,
				ulPresentMarginNs )
			// Before commit-lead warm-up there is no viable reservation to
			// subtract. Keep native miss hysteresis active against the maximum
			// refresh-slot budget; warm-up is far shorter than recovery evidence.
			: gamescope::framegen::deadline_budget_ns( ulVblankIntervalNs );
	FramegenCausalRungSelection_t rung = framegen_select_causal_rung(
		plan, now, ulPlainWakeNs, commitPlan.earlyCommit,
		ulNativeSlotBudgetNs );
	if ( !rung.bAdmit )
	{
		const uint64_t ulDecisionStartNs = std::max( now, ulStartEstimateNs );
		const uint64_t ulRemainingNs = plan.wakeNs > ulDecisionStartNs
			? plan.wakeNs - ulDecisionStartNs : 0u;
		const uint64_t ulPlainRemainingNs = ulPlainWakeNs > ulDecisionStartNs
			? ulPlainWakeNs - ulDecisionStartNs : 0u;
		static uint64_t s_uCausalBudgetSkipDebugLogCounter = 0;
		if ( commitPlan.earlyCommit && g_bFramegenDebug
			&& FramegenDebugShouldLog( s_uCausalBudgetSkipDebugLogCounter ) )
		{
			const double flCommitDeltaMs = commitPlan.earlyCommit
				? static_cast<double>( static_cast<long double>(
					commitPlan.commitDeadlineNs ) - static_cast<long double>( now ) )
					/ 1.0e6
				: 0.0;
			vk_log.infof( "framegen: causal skip remaining=%.2fms budget=%.2fms plain=%.2fms cost=%.2fms lead=%.2fms commit=%+.2fms reservation_only=%d",
				ulRemainingNs / 1.0e6,
				gamescope::framegen::deadline_budget_ns( ulRemainingNs ) / 1.0e6,
				ulPlainRemainingNs / 1.0e6,
				rung.ulCostNs / 1.0e6,
				commitPlan.presentLeadNs / 1.0e6,
				flCommitDeltaMs,
				rung.bCommitReservationShortfall ? 1 : 0 );
		}
		if ( rung.bCommitReservationShortfall || rung.bDeadlineMissSkip )
			g_framegenHistory.ulLastPlannedTargetNs = plan.targetNs;
		return false;
	}

	const uint64_t ulAnchorNs = g_framegenHistory.causalAnchor.display_time();
	const uint64_t ulTargetDeltaNs = plan.targetNs - ulAnchorNs;
	const uint64_t ulSlotIndex64 = std::max<uint64_t>( 1u,
		( ulTargetDeltaNs + ulVblankIntervalNs - 1u ) / ulVblankIntervalNs );
	const uint32_t nSlotIndex = static_cast<uint32_t>( std::min<uint64_t>(
		ulSlotIndex64, UINT32_MAX ) );
	const uint32_t nGapVblanks = gamescope::framegen::interval_gap_vblanks(
		framegen_predicted_interval_ns(), ulVblankIntervalNs );
	const FramegenSlotRequest_t request = {
		.phase = static_cast<float>( plan.phase ),
		.strength = gamescope::framegen::clamp_forward_strength(
			plan.rawStrength, k_flFramegenMaxForwardStrength ),
		.slotIndex = nSlotIndex,
		.targetFlipNs = plan.targetNs,
		.wakeDeadlineNs = plan.wakeNs,
	};
	const uint32_t nMaxCausalDegradeSteps = gamescope::framegen::max_degrade_steps(
		g_eFramegenMode, g_eFramegenPipeline, 2 );
	const bool bSubmitted = framegen_submit_planned( &request, 1, nGapVblanks,
		rung.eff, ulCompositeSeqNo, nMaxCausalDegradeSteps, false, nullptr,
		plan.targetNs, true );
	if ( !bSubmitted )
		return false;

	g_framegenHistory.ulLastPlannedTargetNs = plan.targetNs;
	if ( !rung.bCommittedRung && g_framegenHistory.nDegradeHold > 0u )
		g_framegenHistory.nDegradeHold--;
	static uint64_t s_uCausalDebugLogCounter = 0;
	if ( FramegenDebugShouldLog( s_uCausalDebugLogCounter ) )
	{
		vk_log.infof( "framegen: causal slot phase=%.3f strength=%.3f target=+%.2fms wake=+%.2fms anchor=%s work=%s cost=%.2fms samples=%u",
			plan.phase, request.strength,
			( plan.targetNs > now ? plan.targetNs - now : 0u ) / 1.0e6,
			( plan.wakeNs > now ? plan.wakeNs - now : 0u ) / 1.0e6,
			plan.provisional ? "provisional" : "corrected",
			rung.workClass == gamescope::framegen::DeadlineWorkClass_t::CachedWarp
				? "cached-warp" : "full-prep+warp",
			rung.ulCostNs / 1.0e6, rung.uSamples );
	}
	return true;
}

static void framegen_apply_live_flip_feedback( const DisplayFeedback_t &feedback )
{
	if ( !framegen_causal_deadline_enabled()
		|| vulkan_framegen_vrr_hybrid_active() || vulkan_framegen_bidir_active()
		|| g_framegenHistory.ulDeadlineGridIntervalNs == 0u )
		return;

	const uint64_t ulArrivalGuardNs = std::max(
		gamescope::framegen::k_ulCadenceArrivalGuardMinNs,
		g_framegenHistory.ulDeadlineGridIntervalNs
			/ gamescope::framegen::k_uCadenceArrivalGuardDivisor );
	const gamescope::framegen::AnchorCorrection_t correction =
		gamescope::framegen::apply_flip_feedback(
			g_framegenHistory.causalAnchor,
			g_framegenPresentState.displayTiming.presentBias,
			feedback.tag.ulRealFrameId,
			feedback.ulActualFlipNs,
			ulArrivalGuardNs,
			g_framegenHistory.ulDeadlineGridIntervalNs,
			g_framegenPresentState.bPresentBiasHitchEpisode );
	if ( !correction.matched )
		return;

	g_framegenHistory.causalAnchor = correction.anchor;
	g_framegenPresentState.displayTiming.presentBias = correction.presentBias;
	g_framegenPresentState.bPresentBiasHitchEpisode = false;
	g_framegenHistory.ulCurrentRealVblankNs = feedback.ulActualFlipNs;
	if ( correction.discardProvisional )
	{
		const size_t nBefore = g_framegenHistory.pending.size();
		std::erase_if( g_framegenHistory.pending,
			[&]( const FramegenHistory_t::PendingGenerated_t &entry ) {
				return !entry.bReal
					&& gamescope::framegen::discard_pending_provisional_slot(
						correction, entry.ulAnchorRealFrameId,
						entry.bProvisional );
			} );
		// The discarded target was computed from the wrong grid phase and is not
		// authoritative. Re-enter from the corrected anchor so an exact one-vblank
		// correction can move the next generated phase by that full interval.
		g_framegenHistory.ulLastPlannedTargetNs = 0u;
		const size_t nDiscarded = nBefore - g_framegenHistory.pending.size();
		framegen_metrics_note_discard( nDiscarded );
		static uint64_t s_uFeedbackDiscardDebugLogCounter = 0;
		if ( nBefore != g_framegenHistory.pending.size()
			&& FramegenDebugShouldLog( s_uFeedbackDiscardDebugLogCounter ) )
		{
			vk_log.infof( "framegen: discarded %zu provisional slot(s) after real=%" PRIu64 " anchor correction",
				nBefore - g_framegenHistory.pending.size(),
				feedback.tag.ulRealFrameId );
		}
	}

	if ( g_framegenHistory.pending.empty() )
		framegen_causal_submit( g_framegenHistory.lastCompositeSeqNo );
}

// VRR hybrid slot (#01). Plan exactly ONE generated frame at the content
// midpoint of the measured real-frame interval. This inverts #06: under a
// fixed refresh, the deadline planner asks "given the next vblank, what phase is that?"; under
// active adaptive sync there is no grid — the real frame scanned out on
// arrival — so we PICK the phase (0.5, the content-correct midpoint, exact by
// construction) and manufacture the display event for it: steamcompmgr arms an
// absolute CLOCK_MONOTONIC timer (the clock KMS flip timestamps use) for
// t_realflip + 0.5*predicted cadence and flips the frame then. Always a single slot,
// whatever the configured multiplier: each extra mid flip would multiply the
// timer/cancel bookkeeping and shrink the spacing toward the panel's minimum
// flip interval; one mid flip is the sane ceiling under VRR.
static bool framegen_vrr_hybrid_submit( uint64_t ulCompositeSeqNo, uint32_t nMaxDegradeSteps )
{
	const uint64_t ulPredictedIntervalNs = framegen_predicted_interval_ns();
	if ( !vulkan_framegen_is_enabled() || !vulkan_framegen_vrr_hybrid_active()
		|| !g_framegenHistory.valid || !g_framegenHistory.pending.empty()
		|| g_framegenHistory.previousReal == nullptr || g_framegenHistory.currentReal == nullptr
		|| ulPredictedIntervalNs == 0 || ulCompositeSeqNo == 0 )
		return false;

	// One batch in flight, always (same invariant as every other submit path).
	if ( !g_device.hasCompletedFramegen( g_framegenHistory.lastGeneratedSeqNo ) )
		return false;

	const int nFramegenRefreshMhz = g_nNestedRefresh ? g_nNestedRefresh : g_nOutputRefresh;
	const uint64_t ulVblankIntervalNs = nFramegenRefreshMhz > 0 ? 1'000'000'000'000ull / (uint64_t)nFramegenRefreshMhz : 8'333'333ull;

	const float flPhase = 0.5f;
	const float flStrength = gamescope::framegen::clamp_forward_strength(
		gamescope::framegen::forward_strength_raw( flPhase, g_flFramegenStrength ),
		k_flFramegenMaxForwardStrength );
	// Interval-relative gap equivalent, for logs and rung keying only — no
	// phase is ever derived from it in this mode.
	const uint32_t nGapVblanks = gamescope::framegen::interval_gap_vblanks(
		ulPredictedIntervalNs, ulVblankIntervalNs );

	static uint64_t s_uHybridDebugLogCounter = 0;
	if ( FramegenDebugShouldLog( s_uHybridDebugLogCounter ) )
		vk_log.infof( "framegen: vrr-hybrid slot strength=%.3f mid=+%.2fms cadence=%.2fms",
			flStrength,
			ulPredictedIntervalNs / 2.0e6,
			ulPredictedIntervalNs / 1.0e6 );

	const FramegenEffective_t eff = framegen_effective_config( g_framegenHistory.nDegradeSteps );
	const FramegenSlotRequest_t request = { flPhase, flStrength, 1u };
	const uint64_t ulTargetFlipNs = g_framegenHistory.ulCurrentRealVblankNs
		+ ulPredictedIntervalNs / 2u;
	return framegen_submit_planned( &request, 1, nGapVblanks, eff, ulCompositeSeqNo,
		nMaxDegradeSteps, false, nullptr, ulTargetFlipNs );
}

static void framegen_clear_vrr_midpoint_state( bool bClearPending )
{
	if ( bClearPending && g_framegenHistory.ulVrrMidTargetNs != 0u )
		g_framegenHistory.pending.clear();
	g_framegenHistory.ulVrrAwaitingRealFrameId = 0;
	g_framegenHistory.ulVrrAwaitingCompositeSeqNo = 0;
	g_framegenHistory.ulVrrAwaitingCadenceNs = 0;
	g_framegenHistory.ulVrrPanelIntervalNs = 0;
	g_framegenHistory.ulVrrMidTargetNs = 0;
	g_framegenHistory.ulVrrMidWakeNs = 0;
}

void vulkan_framegen_cancel_vrr_hybrid_slot( const char *reason )
{
	const bool bHadSlot = g_framegenHistory.ulVrrAwaitingRealFrameId != 0u
		|| g_framegenHistory.ulVrrMidTargetNs != 0u;
	framegen_clear_vrr_midpoint_state( true );
	static uint64_t s_uVrrCancelDebugLogCounter = 0;
	if ( bHadSlot && FramegenDebugShouldLog( s_uVrrCancelDebugLogCounter ) )
		vk_log.infof( "framegen: cancelled VRR midpoint reason=%s",
			reason ? reason : "unknown" );
}

uint64_t vulkan_framegen_vrr_hybrid_wake_deadline_ns()
{
	if ( !vulkan_framegen_vrr_hybrid_active()
		|| g_framegenHistory.ulVrrMidWakeNs == 0u
		|| g_framegenHistory.pending.empty() )
		return 0u;
	return g_framegenHistory.ulVrrMidWakeNs;
}

static void framegen_record_vrr_real_for_feedback( uint64_t ulRealFrameId,
	uint64_t ulCompositeSeqNo, uint64_t ulPredictedIntervalNs,
	uint64_t ulPanelIntervalNs, bool bCanGenerate )
{
	// A real arrival supersedes the previous interval's midpoint, whether its
	// timer fired, its GPU work remained in flight, or its feedback was lost.
	framegen_clear_vrr_midpoint_state( true );
	if ( !bCanGenerate || ulRealFrameId == 0u || ulCompositeSeqNo == 0u
		|| !gamescope::framegen::vrr_hybrid_interval_eligible(
			ulPredictedIntervalNs, ulPanelIntervalNs ) )
		return;

	g_framegenHistory.ulVrrAwaitingRealFrameId = ulRealFrameId;
	g_framegenHistory.ulVrrAwaitingCompositeSeqNo = ulCompositeSeqNo;
	g_framegenHistory.ulVrrAwaitingCadenceNs = ulPredictedIntervalNs;
	g_framegenHistory.ulVrrPanelIntervalNs = ulPanelIntervalNs;
}

static void framegen_apply_vrr_flip_feedback( const DisplayFeedback_t &feedback )
{
	if ( feedback.tag.eKind != gamescope::FramegenPresentKind_t::Real
		|| feedback.tag.ulRealFrameId == 0u
		|| feedback.tag.ulRealFrameId
			!= g_framegenHistory.ulVrrAwaitingRealFrameId )
		return;

	if ( !feedback.bPresented || !feedback.bTimestampValid
		|| !vulkan_framegen_vrr_hybrid_active() )
	{
		vulkan_framegen_cancel_vrr_hybrid_slot(
			feedback.bPresented ? "invalid_real_feedback" : "real_feedback_discarded" );
		return;
	}

	if ( !g_device.hasCompletedFramegen( g_framegenHistory.lastGeneratedSeqNo ) )
	{
		vulkan_framegen_cancel_vrr_hybrid_slot( "generation_in_flight" );
		return;
	}

	const uint64_t ulNowNs = get_time_in_nanos();
	const uint64_t ulMarginNs = g_framegenHistory.ulVrrPanelIntervalNs / 10u;
	const gamescope::framegen::VrrMidpointPlan_t plan =
		gamescope::framegen::plan_vrr_midpoint(
			feedback.ulActualFlipNs,
			g_framegenHistory.ulVrrAwaitingCadenceNs,
			static_cast<uint64_t>( std::max<int64_t>(
				0, g_framegenPresentState.displayTiming.presentLead.emaNs ) ),
			ulMarginNs, ulNowNs );
	if ( !plan.valid )
	{
		vulkan_framegen_cancel_vrr_hybrid_slot( "late_target" );
		return;
	}

	// One-slot deadline work has no useful multiplier rung. Walk only the
	// pipeline/mode portion and admit the first cold or measured-fitting class.
	const uint32_t nMaxDegradeSteps = gamescope::framegen::max_degrade_steps(
		g_eFramegenMode, g_eFramegenPipeline, 2u );
	g_framegenHistory.nDegradeSteps = std::min(
		g_framegenHistory.nDegradeSteps, nMaxDegradeSteps );
	FramegenEffective_t eff = framegen_effective_config(
		g_framegenHistory.nDegradeSteps );
	bool bFits = false;
	for ( uint32_t nRung = g_framegenHistory.nDegradeSteps;
		nRung <= nMaxDegradeSteps; nRung++ )
	{
		const FramegenEffective_t candidate = framegen_effective_config( nRung );
		const gamescope::framegen::DeadlineWorkClass_t workClass =
			framegen_causal_work_class( candidate );
		const uint32_t nCostKey =
			gamescope::framegen::deadline_work_class_cost_key( workClass );
		const uint64_t ulCostNs = g_device.framegenRungCostNs( nRung, nCostKey );
		const uint32_t uSamples = g_device.framegenRungSampleCount(
			nRung, nCostKey );
		const bool bMature = ulCostNs != 0u
			&& uSamples >= gamescope::framegen::k_uDeadlineMinSamples;
		if ( !bMature || gamescope::framegen::deadline_cost_fits(
			ulCostNs, plan.wakeDeadlineNs, ulNowNs, 0u ) )
		{
			eff = candidate;
			if ( nRung != g_framegenHistory.nDegradeSteps )
			{
				g_framegenHistory.nDegradeSteps = nRung;
				g_framegenHistory.nDegradeHold =
					gamescope::framegen::k_uDeadlineHoldFrames;
			}
			bFits = true;
			break;
		}
	}
	if ( !bFits )
	{
		vulkan_framegen_cancel_vrr_hybrid_slot( "unfittable_generation" );
		return;
	}

	const float flPhase = 0.5f;
	const FramegenSlotRequest_t request = {
		.phase = flPhase,
		.strength = gamescope::framegen::clamp_forward_strength(
			gamescope::framegen::forward_strength_raw(
				flPhase, g_flFramegenStrength ),
			k_flFramegenMaxForwardStrength ),
		.slotIndex = 1u,
		.targetFlipNs = plan.targetFlipNs,
		.wakeDeadlineNs = plan.wakeDeadlineNs,
	};
	const uint32_t nGapVblanks = gamescope::framegen::interval_gap_vblanks(
		g_framegenHistory.ulVrrAwaitingCadenceNs,
		g_framegenHistory.ulVrrPanelIntervalNs );
	const uint64_t ulCompositeSeqNo =
		g_framegenHistory.ulVrrAwaitingCompositeSeqNo;
	const bool bSubmitted = framegen_submit_planned(
		&request, 1u, nGapVblanks, eff, ulCompositeSeqNo,
		nMaxDegradeSteps, false, nullptr, 0u, true,
		feedback.tag.ulRealFrameId, false );
	if ( !bSubmitted )
	{
		vulkan_framegen_cancel_vrr_hybrid_slot( "submission_unavailable" );
		return;
	}

	g_framegenHistory.ulVrrAwaitingRealFrameId = 0u;
	g_framegenHistory.ulVrrAwaitingCompositeSeqNo = 0u;
	g_framegenHistory.ulVrrAwaitingCadenceNs = 0u;
	g_framegenHistory.ulVrrMidTargetNs = plan.targetFlipNs;
	g_framegenHistory.ulVrrMidWakeNs = plan.wakeDeadlineNs;
	static uint64_t s_uVrrDeadlineDebugLogCounter = 0;
	if ( FramegenDebugShouldLog( s_uVrrDeadlineDebugLogCounter ) )
	{
		vk_log.infof( "framegen: VRR midpoint real=%" PRIu64
			" target=+%.2fms wake=+%.2fms lead=%.2fms margin=%.2fms",
			feedback.tag.ulRealFrameId,
			( plan.targetFlipNs - ulNowNs ) / 1.0e6,
			( plan.wakeDeadlineNs - ulNowNs ) / 1.0e6,
			g_framegenPresentState.displayTiming.presentLead.emaNs / 1.0e6,
			ulMarginNs / 1.0e6 );
	}
}

static void framegen_apply_bidir_flip_feedback( const DisplayFeedback_t &feedback )
{
	if ( !vulkan_framegen_bidir_active() || !feedback.bPresented
		|| !feedback.bTimestampValid
		|| feedback.tag.eKind != gamescope::FramegenPresentKind_t::DelayedReal )
		return;

	auto it = std::ranges::find_if(
		g_framegenHistory.bidirFeedbackEndpoints,
		[&]( const FramegenHistory_t::BidirFeedbackEndpoint_t &record ) {
			return record.endpoint.realFrameId == feedback.tag.ulRealFrameId;
		} );
	if ( it == g_framegenHistory.bidirFeedbackEndpoints.end() )
		return;

	if ( it->ulEpoch == g_framegenHistory.bidirEpoch.epoch )
	{
		const gamescope::framegen::BidirEpochCorrection_t correction =
			gamescope::framegen::apply_bidir_endpoint_feedback(
				g_framegenHistory.bidirEpoch,
				it->endpoint.sourceReadyNs,
				feedback.ulActualFlipNs );
		if ( correction.applied )
		{
			g_framegenHistory.bidirEpoch = correction.epoch;
			static uint64_t s_uBidirFeedbackDebugLogCounter = 0;
			if ( FramegenDebugShouldLog( s_uBidirFeedbackDebugLogCounter ) )
				vk_log.infof( "framegen: bidir epoch corrected by delayed real=%" PRIu64,
					feedback.tag.ulRealFrameId );
		}
	}

	std::erase_if( g_framegenHistory.bidirFeedbackEndpoints,
		[&]( const FramegenHistory_t::BidirFeedbackEndpoint_t &record ) {
			return record.endpoint.realFrameId <= feedback.tag.ulRealFrameId;
		} );
}

static void framegen_bidir_record_feedback_endpoint(
	const gamescope::framegen::BidirEndpoint_t &endpoint )
{
	std::erase_if( g_framegenHistory.bidirFeedbackEndpoints,
		[&]( const FramegenHistory_t::BidirFeedbackEndpoint_t &record ) {
			return record.endpoint.realFrameId == endpoint.realFrameId;
		} );
	g_framegenHistory.bidirFeedbackEndpoints.push_back( {
		.endpoint = endpoint,
		.ulEpoch = g_framegenHistory.bidirEpoch.epoch,
	} );
}

static bool framegen_bidir_plan_pair( uint64_t ulPreviousSourceNs,
	uint64_t ulCurrentSourceNs, uint64_t ulLiveTargetNs,
	uint64_t ulVblankIntervalNs, const FramegenEffective_t &eff,
	uint64_t ulCompositeSeqNo, uint32_t nMaxDegradeSteps,
	bool bAllowGeneration, bool bSourceTimestampsReliable )
{
	if ( ulPreviousSourceNs == 0u || ulCurrentSourceNs <= ulPreviousSourceNs
		|| ulLiveTargetNs == 0u || ulVblankIntervalNs == 0u
		|| g_framegenHistory.previousReal == nullptr
		|| g_framegenHistory.currentReal == nullptr )
		return false;

	const uint64_t ulDisplayChainGeneration =
		g_framegenPresentState.displayTiming.generation;
	const bool bGridChanged = g_framegenHistory.ulBidirGridEpoch == 0u
		|| g_framegenHistory.ulBidirGridIntervalNs != ulVblankIntervalNs
		|| g_framegenHistory.ulBidirDisplayChainGeneration
			!= ulDisplayChainGeneration;
	const bool bProvenanceChanged =
		g_framegenHistory.bBidirProvenanceInitialized
		&& g_framegenHistory.bBidirSourceTimestampsReliable
			!= bSourceTimestampsReliable;
	if ( bGridChanged || bProvenanceChanged )
	{
		g_framegenHistory.bidirEpoch = {};
		g_framegenHistory.bidirFeedbackEndpoints.clear();
		g_framegenHistory.ulBidirGridEpoch++;
		if ( g_framegenHistory.ulBidirGridEpoch == 0u )
			g_framegenHistory.ulBidirGridEpoch = 1u;
	}
	g_framegenHistory.ulBidirGridIntervalNs = ulVblankIntervalNs;
	g_framegenHistory.ulBidirDisplayChainGeneration =
		ulDisplayChainGeneration;
	g_framegenHistory.bBidirProvenanceInitialized = true;
	g_framegenHistory.bBidirSourceTimestampsReliable =
		bSourceTimestampsReliable;

	bool bEstablished = false;
	if ( !g_framegenHistory.bidirEpoch.valid )
	{
		if ( g_framegenHistory.ulBidirGridEpoch == 0u )
			g_framegenHistory.ulBidirGridEpoch = 1u;
		g_framegenHistory.bidirEpoch =
			gamescope::framegen::establish_bidir_epoch(
				ulPreviousSourceNs, ulLiveTargetNs,
				g_framegenHistory.ulBidirGridEpoch );
		if ( !g_framegenHistory.bidirEpoch.valid )
			return false;

		const gamescope::VBlankScheduleTime schedule =
			GetVBlankTimer().CalcNextWakeupTime( true );
		g_framegenHistory.ulBidirGridTargetNs = ulLiveTargetNs;
		g_framegenHistory.ulBidirGridWakeNs =
			schedule.ulScheduledWakeupPoint;
		if ( schedule.ulTargetVBlank > ulLiveTargetNs )
		{
			const uint64_t deltaNs = schedule.ulTargetVBlank - ulLiveTargetNs;
			if ( deltaNs < g_framegenHistory.ulBidirGridWakeNs )
				g_framegenHistory.ulBidirGridWakeNs -= deltaNs;
		}
		else if ( ulLiveTargetNs > schedule.ulTargetVBlank )
		{
			g_framegenHistory.ulBidirGridWakeNs =
				gamescope::framegen::saturating_add_ns(
					schedule.ulScheduledWakeupPoint,
					ulLiveTargetNs - schedule.ulTargetVBlank );
		}
		bEstablished = true;
	}

	const gamescope::framegen::DisplayGrid_t grid = {
		.D0 = g_framegenHistory.ulBidirGridTargetNs,
		.W0 = g_framegenHistory.ulBidirGridWakeNs,
		.T = ulVblankIntervalNs,
	};
	const std::array endpoints = {
		gamescope::framegen::BidirEndpoint_t{
			.realFrameId = g_framegenPresentState.ulPreviousRealFrameId,
			.sourceReadyNs = ulPreviousSourceNs,
		},
		gamescope::framegen::BidirEndpoint_t{
			.realFrameId = g_framegenPresentState.ulCurrentRealFrameId,
			.sourceReadyNs = ulCurrentSourceNs,
		},
	};
	const uint32_t nPlannerMultiplier = g_device.hasFramegenQueue()
		? eff.multiplier : 2u;
	const gamescope::framegen::BidirPlan_t plan =
		gamescope::framegen::plan_bidir_slots(
			grid, g_framegenHistory.bidirEpoch, endpoints,
			nPlannerMultiplier, g_framegenHistory.ulBidirGridEpoch );
	if ( !plan.validEpoch )
		return false;

	// The number of display opportunities this real interval actually offers.
	// It bounds both the submitted batch's cost key and — via
	// bidir_pending_target — how deep the queue is allowed to be planned.
	const uint64_t sourceSpanNs = ulCurrentSourceNs - ulPreviousSourceNs;
	const uint32_t nGapVblanks = std::max<uint32_t>( 1u,
		static_cast<uint32_t>( std::min<uint64_t>( UINT32_MAX,
			( sourceSpanNs + ulVblankIntervalNs - 1u )
				/ ulVblankIntervalNs ) ) );
	const size_t uIncomingEndpoints = std::ranges::count_if(
		plan.slots,
		[&]( const gamescope::framegen::BidirSlot_t &slot ) {
			return slot.kind == gamescope::framegen::BidirSlotKind_t::RealEndpoint
				&& ( slot.endpointFrameId == endpoints[ 1 ].realFrameId
					|| ( bEstablished
						&& slot.endpointFrameId == endpoints[ 0 ].realFrameId ) );
		} );
	// One real interval of timeline, never more than the ring/latency ceiling.
	// Refilling to the flat ceiling every real frame is what pinned the queue
	// at capacity: it kept the drain valve permanently engaged (which disables
	// the per-slot display-target gate) and made the capacity shed the
	// steady-state regulator instead of an exception.
	const size_t uMaxPending = gamescope::framegen::bidir_pending_target(
		framegen_bidir_pending_hard_capacity( eff.multiplier ), nGapVblanks );
	g_framegenHistory.uBidirPendingTarget = uMaxPending;
	// Admission: what the queue can still take once this interval's own real
	// endpoints are in. Work that would not fit is never built or submitted —
	// GPU work never planned beats GPU work planned and then shed.
	const size_t uCommitted = g_framegenHistory.pending.size() + uIncomingEndpoints;
	const size_t uRoom = uMaxPending > uCommitted ? uMaxPending - uCommitted : 0u;

	std::vector<FramegenSlotRequest_t> requests;
	const uint64_t ulNowNs = get_time_in_nanos();
	for ( const gamescope::framegen::BidirSlot_t &slot : plan.slots )
	{
		// plan.slots is ordered by display target, so the admitted prefix is the
		// run that keeps the timeline dense from the next opportunity onward.
		if ( requests.size() >= uRoom )
			break;
		if ( slot.kind != gamescope::framegen::BidirSlotKind_t::Generated
			|| slot.endpointFrameId != endpoints[ 1 ].realFrameId
			|| !bAllowGeneration || ulNowNs >= slot.wakeNs )
			continue;
		if ( std::ranges::any_of( g_framegenHistory.pending,
			[&]( const FramegenHistory_t::PendingGenerated_t &entry ) {
				return entry.ulTargetFlipNs == slot.targetNs;
			} ) )
			continue;
		requests.push_back( {
			.phase = static_cast<float>( slot.phase ),
			.strength = static_cast<float>( slot.phase ),
			.slotIndex = static_cast<uint32_t>( std::min<uint64_t>(
				gamescope::framegen::grid_index_at_or_after(
					grid, slot.targetNs ), UINT32_MAX ) ),
			.targetFlipNs = slot.targetNs,
			.wakeDeadlineNs = slot.wakeNs,
		} );
	}

	const size_t uExistingGenerated = std::ranges::count_if(
		g_framegenHistory.pending,
		[]( const FramegenHistory_t::PendingGenerated_t &entry ) {
			return !entry.bReal;
		} );
	size_t uOverflow = g_framegenHistory.pending.size() + requests.size()
		+ uIncomingEndpoints > uMaxPending
		? g_framegenHistory.pending.size() + requests.size()
			+ uIncomingEndpoints - uMaxPending
		: 0u;
	FramegenBidirShedResult_t shed;
	if ( uOverflow != 0u && uExistingGenerated != 0u )
	{
		const size_t uDrop = std::min( uOverflow, uExistingGenerated );
		const FramegenBidirShedResult_t dropped =
			framegen_bidir_shed_to_capacity(
				g_framegenHistory.pending.size() - uDrop );
		shed.generated += dropped.generated;
		shed.endpoints += dropped.endpoints;
		uOverflow -= dropped.total();
	}
	if ( uOverflow != 0u && !requests.empty() )
	{
		const size_t uDrop = std::min( uOverflow, requests.size() );
		requests.erase( requests.begin(), requests.begin() + uDrop );
		shed.generated += uDrop;
		uOverflow -= uDrop;
		framegen_metrics_note_discard( uDrop );
	}
	if ( uOverflow != 0u )
	{
		const FramegenBidirShedResult_t dropped =
			framegen_bidir_shed_to_capacity(
				g_framegenHistory.pending.size() -
					std::min( uOverflow, g_framegenHistory.pending.size() ) );
		shed.generated += dropped.generated;
		shed.endpoints += dropped.endpoints;
		uOverflow -= std::min( uOverflow, dropped.total() );
	}

	// A mature batch estimate must fit before the earliest surviving wake.
	if ( !requests.empty() )
	{
		const uint32_t nCostKey = static_cast<uint32_t>( requests.size() );
		const uint64_t ulCostNs = g_device.framegenRungCostNs(
			g_framegenHistory.nDegradeSteps, nCostKey );
		const uint32_t uSamples = g_device.framegenRungSampleCount(
			g_framegenHistory.nDegradeSteps, nCostKey );
		if ( ulCostNs != 0u
			&& uSamples >= gamescope::framegen::k_uDeadlineMinSamples
			&& !gamescope::framegen::deadline_cost_fits(
				ulCostNs, requests.front().wakeDeadlineNs,
				ulNowNs, 0u ) )
			requests.clear();
	}

	if ( !requests.empty() )
	{
		framegen_submit_planned( requests.data(),
			static_cast<uint32_t>( requests.size() ), nGapVblanks,
			eff, ulCompositeSeqNo, nMaxDegradeSteps, false );
	}

	const auto queueEndpoint = [&]( const gamescope::framegen::BidirEndpoint_t &endpoint,
		const gamescope::Rc<CVulkanTexture> &texture, uint64_t frameId,
		uint64_t compositeSeqNo ) {
		const auto slot = std::ranges::find_if( plan.slots,
			[&]( const gamescope::framegen::BidirSlot_t &candidate ) {
				return candidate.kind == gamescope::framegen::BidirSlotKind_t::RealEndpoint
					&& candidate.endpointFrameId == endpoint.realFrameId;
			} );
		if ( slot == plan.slots.end() )
			return;
		FramegenHistory_t::PendingGenerated_t entry;
		entry.tex = texture;
		entry.frameId = frameId;
		entry.ulPresentRealFrameId = endpoint.realFrameId;
		entry.ulSlotId = framegen_next_present_slot_id();
		entry.ulCompositeSeqNo = compositeSeqNo;
		entry.ulTargetFlipNs = slot->targetNs;
		entry.ulWakeDeadlineNs = slot->wakeNs;
		entry.phase = 1.0f;
		entry.bReal = true;
		if ( framegen_insert_pending_entry( std::move( entry ) ) )
			framegen_bidir_record_feedback_endpoint( endpoint );
	};
	if ( bEstablished )
	{
		queueEndpoint( endpoints[ 0 ], g_framegenHistory.previousReal,
			g_framegenHistory.previousFrameId,
			g_framegenPresentState.ulPreviousRealCompositeSeqNo );
	}
	queueEndpoint( endpoints[ 1 ], g_framegenHistory.currentReal,
		g_framegenHistory.currentFrameId, ulCompositeSeqNo );
	const FramegenBidirShedResult_t finalShed =
		framegen_bidir_shed_to_capacity( uMaxPending );
	shed.generated += finalShed.generated;
	shed.endpoints += finalShed.endpoints;
	if ( shed.endpoints != 0u && finalShed.endpoints == 0u )
		framegen_bidir_resync_to_newest_endpoint();
	g_framegenHistory.bBidirQueuedReal = true;

	static uint64_t s_uBidirDeadlineDebugLogCounter = 0;
	if ( FramegenDebugShouldLog( s_uBidirDeadlineDebugLogCounter ) )
	{
		vk_log.infof( "framegen: bidir epoch=%" PRIu64
			" candidates=%zu pending=%zu/%zu gap=%u room=%zu forced=%d"
			" shed_gen=%zu shed_real=%zu%s",
			g_framegenHistory.bidirEpoch.epoch, requests.size(),
			g_framegenHistory.pending.size(), uMaxPending,
			nGapVblanks, uRoom, (int)framegen_bidir_queue_forces_drain(),
			shed.generated, shed.endpoints,
			bEstablished ? " established" : "" );
	}
	return true;
}

// Reactive causal catch-all, called by the present decision when a vblank goes to
// a hardware repeat while framegen is active (a stall, a too-slow discard, or
// a mispredicted keep-up), and by the consume path when the pending slot
// drains. It requests the earliest slot that can still be prepared. One repeat
// is unavoidable after a keep-up misprediction; an in-flight GPU overrun or the
// forward-prediction cap can require additional honest repeats.
void vulkan_framegen_causal_tick()
{
	vulkan_framegen_metrics_note_repeat();
	// VRR and bidir retain their own Step 3 policies. The classic A/B switch and
	// shared-queue fallback are intentionally no-ops here.
	if ( !framegen_causal_deadline_enabled()
		|| vulkan_framegen_vrr_hybrid_active() || vulkan_framegen_bidir_active() )
		return;
	framegen_causal_submit( g_framegenHistory.lastCompositeSeqNo );
}

// Keep the old exported symbol during the Step 3 review interval; production
// callers use the generalized name and the legacy environment variable no
// longer gates either entry point.
void vulkan_framegen_jit_tick()
{
	vulkan_framegen_causal_tick();
}

static bool framegen_refill_idle()
{
	if ( !vulkan_framegen_is_enabled() || !g_device.hasFramegenQueue()
		|| !g_framegenHistory.valid || !g_framegenHistory.pending.empty()
		|| g_framegenHistory.previousReal == nullptr || g_framegenHistory.currentReal == nullptr
		|| g_framegenHistory.lastCompositeSeqNo == 0 || g_framegenHistory.nLastGenerationGapVblanks == 0 )
		return false;

	if ( !g_device.hasCompletedFramegen( g_framegenHistory.lastGeneratedSeqNo ) )
		return false;

	const int nFramegenRefreshMhz = g_nNestedRefresh ? g_nNestedRefresh : g_nOutputRefresh;
	const uint64_t ulVblankIntervalNs = nFramegenRefreshMhz > 0 ? 1'000'000'000'000ull / (uint64_t)nFramegenRefreshMhz : 8'333'333ull;
	const uint64_t now = get_time_in_nanos();
	if ( now <= g_framegenHistory.currentPresentTimeNs )
		return false;

	const uint64_t ulAgeNs = now - g_framegenHistory.currentPresentTimeNs;
	if ( ulAgeNs > k_ulFramegenMaxRealFrameGapNs )
	{
		vulkan_framegen_invalidate_history( "idle_frame_gap" );
		return false;
	}

	const uint32_t nElapsedSlots = std::max( 1u, (uint32_t)( ( ulAgeNs + ulVblankIntervalNs / 2 ) / ulVblankIntervalNs ) );
	const uint32_t nNextSlot = std::max( g_framegenHistory.nLastGeneratedSlot + 1, nElapsedSlots + 1 );
	const uint32_t nMaxSlots = std::max( g_framegenHistory.nLastGenerationGapVblanks + 1,
		(uint32_t)( k_ulFramegenMaxRealFrameGapNs / ulVblankIntervalNs ) );
	if ( nNextSlot > nMaxSlots )
		return false;

	const FramegenEffective_t eff = framegen_effective_config( g_framegenHistory.nDegradeSteps );
	return framegen_submit_batch( nNextSlot, g_framegenHistory.nLastGenerationGapVblanks, 1,
		eff, g_framegenHistory.lastCompositeSeqNo, framegen_max_degrade_steps(), false );
}

// ---------------------------------------------------------------------------
// Frame-generation GPU microbenchmark.
//
// Times the exact production dispatch helpers above (framegen_bind_extrapolate,
// framegen_prepare_motion + framegen_warp_slot, framegen_bind_blend) in
// isolation, using GPU timestamp queries so the reported cost is pure shader
// execution — no submit, present, or pacing overhead. Runs against synthetic
// history textures on a headless device; driven by gamescope_framegen_microbench.
// ---------------------------------------------------------------------------

namespace {

struct FramegenBenchRes_t { const char *pszName; uint32_t nWidth; uint32_t nHeight; };

static gamescope::OwningRc<CVulkanTexture> framegen_bench_make_image( uint32_t nWidth, uint32_t nHeight, uint32_t uDrmFormat )
{
	CVulkanTexture::createFlags flags;
	flags.bSampled = true;
	flags.bStorage = true;
	gamescope::OwningRc<CVulkanTexture> pTex = new CVulkanTexture();
	if ( !pTex->BInit( nWidth, nHeight, 1u, uDrmFormat, flags ) )
		return nullptr;
	return pTex;
}

} // namespace

void vulkan_framegen_benchmark()
{
	VkPhysicalDeviceProperties props = {};
	g_device.vk.GetPhysicalDeviceProperties( g_device.physDev(), &props );
	const double flTimestampPeriodNs = props.limits.timestampPeriod;
	uint32_t uQueueFamilyCount = 0;
	g_device.vk.GetPhysicalDeviceQueueFamilyProperties( g_device.physDev(), &uQueueFamilyCount, nullptr );
	std::vector<VkQueueFamilyProperties> queueFamilyProps( uQueueFamilyCount );
	g_device.vk.GetPhysicalDeviceQueueFamilyProperties( g_device.physDev(), &uQueueFamilyCount, queueFamilyProps.data() );
	const uint32_t uTimestampValidBits = g_device.queueFamily() < uQueueFamilyCount
		? queueFamilyProps[ g_device.queueFamily() ].timestampValidBits : 0;
	if ( flTimestampPeriodNs == 0.0 || uTimestampValidBits == 0 )
	{
		fprintf( stderr, "framegen-bench: device does not support timestamp queries\n" );
		return;
	}
	const uint64_t ulTimestampMask = uTimestampValidBits >= 64
		? UINT64_MAX : ( ( 1ull << uTimestampValidBits ) - 1ull );

	const VkQueryPoolCreateInfo queryPoolInfo = {
		.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
		.queryType = VK_QUERY_TYPE_TIMESTAMP,
		.queryCount = 2,
	};
	VkQueryPool queryPool = VK_NULL_HANDLE;
	if ( g_device.vk.CreateQueryPool( g_device.device(), &queryPoolInfo, nullptr, &queryPool ) != VK_SUCCESS )
	{
		fprintf( stderr, "framegen-bench: CreateQueryPool failed\n" );
		return;
	}

	const FramegenBenchRes_t resolutions[] = {
		{ "1080p", 1920, 1080 },
		{ "1440p", 2560, 1440 },
		{ "2160p", 3840, 2160 },
	};

	// Two representative targets: a 10-bit integer format (HDR10/SDR path, where
	// production runs the fp16 extrapolate shader) and an fp16-float format (scRGB
	// path, forced to fp32 for precision). Both let us A/B the fp16 vs fp32 shader.
	struct FramegenBenchFormat_t { const char *pszName; uint32_t uDrmFormat; };
	const FramegenBenchFormat_t formats[] = {
		{ "ABGR2101010 (int, HDR10/SDR path)", DRM_FORMAT_ABGR2101010 },
		{ "ABGR16161616F (float, scRGB path)", DRM_FORMAT_ABGR16161616F },
	};
	const bool bFp16 = g_device.supportsShaderFloat16();

	printf( "\ngamescope frame-generation GPU microbenchmark\n" );
	printf( "device : %s\n", props.deviceName );
	printf( "fp16   : shader float16 %s\n", bFp16 ? "supported" : "unsupported" );
	printf( "timing : GPU timestamps, mean of 200 dispatches, strength=%.2f\n", g_flFramegenStrength );
	printf( "note   : (*) = variant production uses for that format\n" );

	for ( const FramegenBenchFormat_t &fmt : formats )
	{
	printf( "\n== target %s ==\n", fmt.pszName );
	printf( "%-8s  %-26s  %10s\n", "res", "pass", "GPU ms" );
	printf( "-------------------------------------------------------\n" );
	const uint32_t uDrmFormat = fmt.uDrmFormat;

	// Time a single recorded workload: warm up once (first-use allocations,
	// pipeline residency), then average the GPU timestamp delta over nIters
	// submits. Each submit is drained before the next, so nothing overlaps and
	// the descriptor ring never wraps under an in-flight dispatch.
	auto timePass = [&]( auto &&recordFn ) -> double
	{
		const uint32_t nIters = 200;
		{
			auto cmd = g_device.commandBuffer();
			recordFn( cmd.get() );
			g_device.submit( std::move( cmd ) );
			g_device.waitIdle();
		}
		double dTotalNs = 0.0;
		for ( uint32_t i = 0; i < nIters; i++ )
		{
			auto cmd = g_device.commandBuffer();
			VkCommandBuffer raw = cmd->rawBuffer();
			g_device.vk.CmdResetQueryPool( raw, queryPool, 0, 2 );
			g_device.vk.CmdWriteTimestamp( raw, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, queryPool, 0 );
			recordFn( cmd.get() );
			g_device.vk.CmdWriteTimestamp( raw, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, queryPool, 1 );
			g_device.submit( std::move( cmd ) );
			g_device.waitIdle();

			uint64_t ts[2] = { 0, 0 };
			g_device.vk.GetQueryPoolResults( g_device.device(), queryPool, 0, 2,
				sizeof( ts ), ts, sizeof( uint64_t ), VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT );
			const uint64_t ulGpuTicks = ( ts[1] - ts[0] ) & ulTimestampMask;
			dTotalNs += double( ulGpuTicks ) * flTimestampPeriodNs;
		}
		return dTotalNs / double( nIters ) / 1.0e6;
	};

	for ( const FramegenBenchRes_t &res : resolutions )
	{
		gamescope::OwningRc<CVulkanTexture> pPrev = framegen_bench_make_image( res.nWidth, res.nHeight, uDrmFormat );
		gamescope::OwningRc<CVulkanTexture> pCur  = framegen_bench_make_image( res.nWidth, res.nHeight, uDrmFormat );
		gamescope::OwningRc<CVulkanTexture> pOut  = framegen_bench_make_image( res.nWidth, res.nHeight, uDrmFormat );
		if ( !pPrev || !pCur || !pOut )
		{
			fprintf( stderr, "framegen-bench: %s image allocation failed\n", res.pszName );
			continue;
		}

		// Point framegen history at the synthetic frames (what the dispatch
		// helpers sample as previous/current real frames).
		g_framegenHistory.previousReal = pPrev;
		g_framegenHistory.currentReal = pCur;

		const FramegenDispatch_t &dispatch = framegen_dispatch_for_format( uDrmFormat );

		// Extrapolate variants on identical images — a direct A/B. (*) marks the
		// shader the vendor dispatcher actually selects for this GPU + format.
		double msExtrap32 = timePass( [&]( CVulkanCmdBuffer *cmd ) {
			framegen_bind_extrapolate( cmd, SHADER_TYPE_FRAMEGEN_EXTRAPOLATE, pOut, g_flFramegenStrength );
		} );
		printf( "%-8s  %-26s  %10.3f\n", res.pszName,
			dispatch.extrapolate == SHADER_TYPE_FRAMEGEN_EXTRAPOLATE ? "extrapolate fp32-lds (*)" : "extrapolate fp32-lds", msExtrap32 );

		if ( bFp16 )
		{
			double msExtrap16 = timePass( [&]( CVulkanCmdBuffer *cmd ) {
				framegen_bind_extrapolate( cmd, SHADER_TYPE_FRAMEGEN_EXTRAPOLATE_FP16, pOut, g_flFramegenStrength );
			} );
			printf( "%-8s  %-26s  %10.3f\n", res.pszName,
				dispatch.extrapolate == SHADER_TYPE_FRAMEGEN_EXTRAPOLATE_FP16 ? "extrapolate fp16-lds (*)" : "extrapolate fp16-lds", msExtrap16 );
		}

		// Direct (no-LDS) fp32 variant — the vendor dispatcher picks this on
		// large-cache GPUs. Mark it (*) when it is the selected production shader.
		{
			const bool bDirectSelected = ( dispatch.extrapolate == SHADER_TYPE_FRAMEGEN_EXTRAPOLATE_DIRECT );
			double msExtrapDirect = timePass( [&]( CVulkanCmdBuffer *cmd ) {
				framegen_bind_extrapolate( cmd, SHADER_TYPE_FRAMEGEN_EXTRAPOLATE_DIRECT, pOut, g_flFramegenStrength );
			} );
			printf( "%-8s  %-26s  %10.3f\n", res.pszName,
				bDirectSelected ? "extrapolate fp32-direct (*)" : "extrapolate fp32-direct", msExtrapDirect );
		}

		// Blend — cheapest mode, a useful floor.
		double msBlend = timePass( [&]( CVulkanCmdBuffer *cmd ) {
			framegen_bind_blend( cmd, pOut, 0.5f );
		} );
		printf( "%-8s  %-26s  %10.3f\n", res.pszName, "blend", msBlend );

		// Motion: per-real-frame setup (luma downscale + block match) and the
		// per-generated-frame warp are distinct costs — report both.
		if ( dispatch.motionSupported )
		{
			// Prime intermediates so the warp timing doesn't include allocation.
			{
				auto cmd = g_device.commandBuffer();
				framegen_prepare_motion( cmd.get(), res.nWidth, res.nHeight, dispatch, g_eFramegenPipeline );
				if ( g_eFramegenPipeline >= GamescopeFramegenPipeline::Predict
					&& g_framegenMotion.mvFieldHistory != nullptr )
					cmd->copyImage( framegen_motion_field(), g_framegenMotion.mvFieldHistory );
				if ( g_eFramegenPipeline == GamescopeFramegenPipeline::Guided
					&& g_framegenMotion.lumaReservoir[0] != nullptr )
					cmd->copyImage( g_framegenMotion.lumaPrev, g_framegenMotion.lumaReservoir[0] );
				g_device.submit( std::move( cmd ) );
				g_device.waitIdle();
				if ( g_eFramegenPipeline >= GamescopeFramegenPipeline::Predict
					&& g_framegenMotion.mvFieldHistory != nullptr )
				{
					g_framegenMotion.uMotionHistoryFrameId = 2;
					g_framegenMotion.uMotionHistoryIntervalNs = 16'666'667ull;
					g_framegenHistory.currentFrameId = 3;
					g_framegenHistory.previousPresentTimeNs = 16'666'667ull;
					g_framegenHistory.currentPresentTimeNs = 33'333'334ull;
				}
				if ( g_eFramegenPipeline == GamescopeFramegenPipeline::Guided
					&& g_framegenMotion.lumaReservoir[0] != nullptr )
					g_framegenMotion.uLumaReservoirFrameId[0] = 1;
			}

			double msMotionPrep = timePass( [&]( CVulkanCmdBuffer *cmd ) {
				framegen_prepare_motion( cmd, res.nWidth, res.nHeight, dispatch, g_eFramegenPipeline );
			} );
			printf( "%-8s  %-26s  %10.3f\n", res.pszName, "motion setup (per real)", msMotionPrep );

			// B4 stats probe (clear + accumulate + 384-byte readback copy) —
			// recorded after setup in production, so its cost adds to the
			// per-real tail, not per-slot.
			if ( framegen_adapt_enabled( g_eFramegenPipeline ) )
			{
				double msAdaptProbe = timePass( [&]( CVulkanCmdBuffer *cmd ) {
					framegen_record_adapt_probe( cmd, g_framegenMotion.width, g_framegenMotion.height );
				} );
				printf( "%-8s  %-26s  %10.3f\n", res.pszName, "adapt stats probe (per real)", msAdaptProbe );
			}

			// Stage C net refinement — one direction for causal prediction,
			// both only when bidir is active. Runs first so the warp rows below
			// bind the refined field, as in production.
			if ( g_framegenMotion.mvFieldNet != nullptr )
			{
				double msNet = timePass( [&]( CVulkanCmdBuffer *cmd ) {
					framegen_record_net( cmd, g_framegenMotion.width, g_framegenMotion.height, vulkan_framegen_bidir_active() );
				} );
				printf( "%-8s  %-26s  %10.3f\n", res.pszName, "motion net refine (per real)", msNet );
				if ( framegen_net_online_enabled() )
				{
					double msNetTrain = timePass( [&]( CVulkanCmdBuffer *cmd ) {
						framegen_record_net_train( cmd, g_eFramegenPipeline, false );
					} );
					printf( "%-8s  %-26s  %10.3f\n", res.pszName, "online net train (per real)", msNetTrain );
				}
			}

			double msMotionWarp = timePass( [&]( CVulkanCmdBuffer *cmd ) {
				framegen_warp_slot( cmd, pOut, g_flFramegenStrength, g_eFramegenPipeline );
			} );
			printf( "%-8s  %-26s  %10.3f\n", res.pszName,
				g_eFramegenPipeline == GamescopeFramegenPipeline::Guided
					? "motion guided warp (per gen)"
					: g_eFramegenPipeline == GamescopeFramegenPipeline::Predict
						? "motion accel warp (per gen)" : "motion warp (per gen)",
				msMotionWarp );

			if ( g_eFramegenPipeline == GamescopeFramegenPipeline::Guided
				&& g_framegenMotion.lumaReservoir[0] != nullptr )
			{
				double msReservoirUpdate = timePass( [&]( CVulkanCmdBuffer *cmd ) {
					cmd->copyImage( g_framegenMotion.lumaPrev, g_framegenMotion.lumaReservoir[0] );
				} );
				printf( "%-8s  %-26s  %10.3f\n", res.pszName,
					"luma reservoir copy (real)", msReservoirUpdate );
			}

			// Bidir (B3): the setup number above already includes the extra
			// reverse-field check when bidir is active; the two-frame warp is
			// its own per-slot cost.
			if ( vulkan_framegen_bidir_active() )
			{
				double msBidirWarp = timePass( [&]( CVulkanCmdBuffer *cmd ) {
					framegen_bidir_warp_slot( cmd, pOut, 0.5f, g_eFramegenPipeline, -1.0f, 0.0f );
				} );
				printf( "%-8s  %-26s  %10.3f\n", res.pszName, "motion bidir warp (per gen)", msBidirWarp );
				if ( g_eFramegenPipeline == GamescopeFramegenPipeline::Guided )
				{
					double msBidirTrace = timePass( [&]( CVulkanCmdBuffer *cmd ) {
						framegen_bidir_warp_slot( cmd, pOut, 0.5f, g_eFramegenPipeline, -1.0f, 1.0f );
					} );
					printf( "%-8s  %-26s  %10.3f\n", res.pszName, "motion bidir trace (per gen)", msBidirTrace );
				}
			}
		}
		else
		{
			printf( "%-8s  %-26s  %10s\n", res.pszName, "motion", "unsupported" );
		}

		printf( "-------------------------------------------------------\n" );

		// Drop history references and motion intermediates before the images go.
		g_framegenHistory.previousReal = nullptr;
		g_framegenHistory.currentReal = nullptr;
		vulkan_framegen_reset( "benchmark cleanup" );
	}
	} // format loop

	g_device.vk.DestroyQueryPool( g_device.device(), queryPool, nullptr );
	printf( "\n" );
}

// Gap E2 held-out capture. Real frames still scan out normally. For each A/B/C
// sequence (consecutive by default, configurable offset/span), B is retained as
// ground truth but deliberately omitted from motion history; three invisible
// slots are generated at B's measured temporal phase from A/C with a paired
// parameter sweep, copied beside B, and never enter the pending queue.
static bool framegen_record_color_probe_real( gamescope::Rc<CVulkanTexture> pRealFrame,
	const struct FrameInfo_t *pFrameInfo, uint64_t ulCompositeSeqNo )
{
	if ( framegen_color_record_dir() == nullptr )
		return false;
	if ( !framegen_color_probe_requested() )
		return false;

	framegen_color_probe_consume();
	if ( !framegen_color_probe_active() )
		return true;

	if ( !framegen_ensure_resources( pRealFrame->width(), pRealFrame->height(), pRealFrame->drmFormat(), false ) )
		return true;

	FramegenColorProbeResources_t &c = g_framegenColorProbe;
	const uint64_t uFrameId = ++c.nextRealId;
	const uint64_t ulNowNs = get_time_in_nanos();
	if ( uFrameId <= framegen_color_record_skip() )
	{
		c.anchor = nullptr;
		c.reference = nullptr;
		c.lastRealTimeNs = 0;
		g_framegenHistory.previousReal = nullptr;
		g_framegenHistory.currentReal = nullptr;
		return true;
	}

	const auto reseed = [&]()
	{
		c.anchor = pRealFrame;
		c.reference = nullptr;
		c.anchorId = uFrameId;
		c.referenceId = 0;
		c.anchorTimeNs = ulNowNs;
		c.referenceTimeNs = 0;
		c.lastRealTimeNs = ulNowNs;
		c.eotf = pFrameInfo->outputEncodingEOTF;
		g_framegenHistory.previousReal = nullptr;
		g_framegenHistory.currentReal = pRealFrame;
		g_framegenHistory.previousFrameId = 0;
		g_framegenHistory.currentFrameId = uFrameId;
	};

	if ( c.anchor == nullptr )
	{
		reseed();
		return true;
	}

	const bool bSequenceMismatch = c.anchor->width() != pRealFrame->width()
		|| c.anchor->height() != pRealFrame->height()
		|| c.anchor->drmFormat() != pRealFrame->drmFormat()
		|| c.eotf != pFrameInfo->outputEncodingEOTF
		|| ulNowNs <= c.lastRealTimeNs
		|| ulNowNs - c.lastRealTimeNs > k_ulFramegenMaxRealFrameGapNs;
	if ( bSequenceMismatch )
	{
		reseed();
		return true;
	}

	// Never queue a second capture behind an unfinished one. Skipping this real
	// frame starts a fresh held-out sequence; the submitted command owns its old
	// inputs/readbacks, while genReadA/B/reference keep the output-ring slots
	// immutable until completion.
	if ( c.pendingSeqNo != 0 && !g_device.hasCompletedFramegen( c.pendingSeqNo ) )
	{
		reseed();
		return true;
	}

	const uint64_t uSequenceOffset = uFrameId - c.anchorId;
	const uint32_t uReferenceOffset = framegen_color_record_offset();
	const uint32_t uEndpointOffset = framegen_color_record_span();

	if ( c.reference == nullptr )
	{
		if ( uSequenceOffset < uReferenceOffset )
		{
			c.lastRealTimeNs = ulNowNs;
			return true;
		}
		if ( uSequenceOffset != uReferenceOffset )
		{
			reseed();
			return true;
		}
		c.reference = pRealFrame;
		c.referenceId = uFrameId;
		c.referenceTimeNs = ulNowNs;
		c.lastRealTimeNs = ulNowNs;
		return true;
	}

	if ( uSequenceOffset < uEndpointOffset )
	{
		c.lastRealTimeNs = ulNowNs;
		return true;
	}
	if ( uSequenceOffset != uEndpointOffset )
	{
		reseed();
		return true;
	}

	const uint64_t ulSpanNs = ulNowNs - c.anchorTimeNs;
	const float flPhase = (float)( c.referenceTimeNs - c.anchorTimeNs ) / (float)ulSpanNs;
	const float flTargetPhase = (float)uReferenceOffset / (float)uEndpointOffset;
	if ( !gamescope::framegen::is_finite_binary32( flPhase ) || flPhase <= 0.05f || flPhase >= 0.95f
		|| std::abs( flPhase - flTargetPhase ) > framegen_color_record_phase_tolerance() )
	{
		reseed();
		return true;
	}

	gamescope::Rc<CVulkanTexture> pAnchor = c.anchor;
	gamescope::Rc<CVulkanTexture> pReference = c.reference;
	const uint64_t uAnchorId = c.anchorId;
	const uint64_t uReferenceId = c.referenceId;
	const uint64_t ulAnchorTimeNs = c.anchorTimeNs;
	const uint64_t ulReferenceTimeNs = c.referenceTimeNs;

	g_framegenHistory.pending.clear();
	g_framegenHistory.previousReal = pAnchor;
	g_framegenHistory.currentReal = pRealFrame;
	g_framegenHistory.previousFrameId = uAnchorId;
	g_framegenHistory.currentFrameId = uFrameId;
	g_framegenHistory.previousPresentTimeNs = ulAnchorTimeNs;
	g_framegenHistory.currentPresentTimeNs = ulNowNs;
	g_framegenHistory.lastCompositeSeqNo = ulCompositeSeqNo;
	g_framegenHistory.nLastGeneratedSlot = 0;
	g_framegenHistory.nLastGenerationGapVblanks = 0;

	const FramegenEffective_t eff = {
		.mode = GamescopeFramegenMode::Motion,
		.multiplier = 4u,
		.pipeline = g_eFramegenPipeline,
	};
	FramegenSlotRequest_t slots[ k_uFramegenColorProbeCandidates ];
	for ( uint32_t i = 0; i < k_uFramegenColorProbeCandidates; i++ )
		slots[ i ] = { .phase = flPhase, .strength = flPhase, .slotIndex = i + 1u };
	const FramegenColorProbeRequest_t probe = {
		.reference = pReference,
		.eotf = pFrameInfo->outputEncodingEOTF,
		.sweep = framegen_color_record_sweep(),
		.anchorId = uAnchorId,
		.referenceId = uReferenceId,
		.endpointId = uFrameId,
		.anchorTimeNs = ulAnchorTimeNs,
		.referenceTimeNs = ulReferenceTimeNs,
		.endpointTimeNs = ulNowNs,
	};
	const bool bSubmitted = framegen_submit_planned( slots, k_uFramegenColorProbeCandidates, 2u, eff,
		ulCompositeSeqNo, framegen_max_degrade_steps(), false, &probe );
	if ( !bSubmitted && g_bFramegenDebug )
		vk_log.infof( "framegen: held-out colour probe dropped (motion/capture resources unavailable)" );

	// C is a provisional next anchor. In-flight sampling is protected by
	// genReadA/B/reference; after the synchronous file write, consume() clears
	// this timestamp so storage latency cannot contaminate the next interval.
	reseed();
	return true;
}

// bCursorFreeRealFrame: output-space mode only — pRealFrame carries no cursor,
// either because the composite split it out (vulkan_composite) or because the
// stack had no cursor layer to begin with. It is the caller's verdict, not a
// property recomputable here: pFrameInfo still describes the FULL stack.
static void framegen_record_real_frame( gamescope::Rc<CVulkanTexture> pRealFrame, const struct FrameInfo_t *pFrameInfo, uint64_t ulCompositeSeqNo, bool bCursorFreeRealFrame )
{
	if ( !vulkan_framegen_is_enabled() || !pRealFrame || !pFrameInfo )
		return;

	if ( !g_bLoggedFramegenConfig )
	{
		vk_log.infof( "framegen: enabled mode=%s pipeline=%s multiplier=%d%s",
			gamescope::framegen::mode_name( g_eFramegenMode ),
			gamescope::framegen::pipeline_name( g_eFramegenPipeline ), g_nFramegenMultiplier,
			g_device.hasFramegenQueue() ? " (dedicated queue)" : "" );
		if ( GetBackend() != nullptr && GetBackend()->OwnsKMSPresentTiming()
			&& framegen_commit_lead_override_ns() != 0u )
		{
			vk_log.infof( "framegen: commit lead override %.2fms (experiment)",
				framegen_commit_lead_override_ns() / 1.0e6 );
		}
		vk_log.infof( "framegen: forcing composite path" );
		if ( vulkan_framegen_vrr_hybrid_requested() )
			vk_log.infof( "framegen: VRR hybrid requested — adaptive sync stays active, generated frames flip mid-interval; tearing remains suppressed" );
		else
			vk_log.infof( "framegen: adaptive sync (VRR) and tearing flips are suppressed while framegen is active" );
		if ( env_to_bool( getenv( "GAMESCOPE_FRAMEGEN_JIT" ) ) )
			vk_log.infof( "framegen: GAMESCOPE_FRAMEGEN_JIT is now default and has no effect" );
		if ( framegen_causal_deadline_enabled()
			&& !vulkan_framegen_vrr_hybrid_active()
			&& !vulkan_framegen_bidir_active() )
			vk_log.infof( "framegen: causal fixed-refresh deadline scheduling active by default — one exact display slot is planned at a time; multiplier is a resource ceiling" );
		else if ( g_device.hasFramegenQueue() && framegen_classic_enabled()
			&& !vulkan_framegen_vrr_hybrid_active()
			&& !vulkan_framegen_bidir_active() )
			vk_log.infof( "framegen: GAMESCOPE_FRAMEGEN_CLASSIC=1 restored the temporary classic batch path for A/B (scheduled for deletion in Step 5)" );
		if ( framegen_bidir_enabled() )
		{
			if ( g_eFramegenMode == GamescopeFramegenMode::Motion && !vulkan_framegen_vrr_hybrid_requested() )
				vk_log.infof( "framegen: bidirectional interpolation requested (B3) — generated frames interpolate between the two real frames; real-frame presentation is delayed up to one interval" );
			else
				vk_log.infof( "framegen: GAMESCOPE_FRAMEGEN_BIDIR ignored (requires motion mode and is incompatible with VRR-hybrid pacing)" );
		}
		if ( framegen_base_layer_enabled() )
			vk_log.infof( "framegen: base-layer generation + late overlay composite requested (#02) — predicting on the pre-upscale game layer, overlays/cursor composite fresh onto generated frames" );
		if ( g_eFramegenMode == GamescopeFramegenMode::Motion )
		{
			const bool bBidirActive = vulkan_framegen_bidir_active();
			if ( bBidirActive )
			{
				vk_log.infof( "framegen: bidirectional pipeline — symmetric checked forward/reverse fields%s; causal acceleration, Guided color-guided reconstruction, reservoir and shading are not scheduled",
					framegen_agreement_enabled( g_eFramegenPipeline ) ? " + full-resolution agreement" : "" );
				if ( framegen_bidir_one_sided_strength() > 0.0f )
					vk_log.infof( "framegen: experimental bidir one-sided occlusion authority %.2f — strongly asymmetric checked fields retain more of the surviving warped side without changing both-valid or both-killed fallback",
						framegen_bidir_one_sided_strength() );
				if ( framegen_bidir_endpoint_trace_strength( g_eFramegenPipeline ) > 0.0f )
					vk_log.infof( "framegen: experimental bidir endpoint trace %.2f — one symmetric closure-gated fixed-point correction; queue and flip timing are unchanged",
						framegen_bidir_endpoint_trace_strength( g_eFramegenPipeline ) );
			}
			else if ( g_eFramegenPipeline == GamescopeFramegenPipeline::Warp )
				vk_log.infof( "framegen: warp pipeline — forward matcher + constant-velocity warp only" );
			else if ( g_eFramegenPipeline == GamescopeFramegenPipeline::Checked )
				vk_log.infof( "framegen: checked pipeline — reverse consistency + full-resolution agreement enabled" );
			else if ( g_eFramegenPipeline == GamescopeFramegenPipeline::Predict )
				vk_log.infof( "framegen: predict pipeline — confidence-gated causal temporal acceleration enabled after one consecutive field warm-up" );
			else if ( g_eFramegenPipeline == GamescopeFramegenPipeline::Guided )
				vk_log.infof( "framegen: guided pipeline — color-guided reconstruction + causal acceleration%s",
					framegen_reservoir_enabled( g_eFramegenPipeline )
						? " + three-real-frame disocclusion reservoir enabled"
						: " (disocclusion reservoir disabled by GAMESCOPE_FRAMEGEN_RESERVOIR=0)" );
			if ( framegen_adapt_enabled( g_eFramegenPipeline ) )
				vk_log.infof( "framegen: self-supervised adaptation active (B4) — each real frame grades the field that predicted it; blend trust follows same-batch, thresholds auto-calibrate next batch (GAMESCOPE_FRAMEGEN_ADAPT=0 disables)" );
			else if ( g_eFramegenPipeline < GamescopeFramegenPipeline::Learned )
				vk_log.infof( "framegen: self-supervised adaptation not scheduled below the learned pipeline" );
			else
				vk_log.infof( "framegen: self-supervised adaptation disabled (GAMESCOPE_FRAMEGEN_ADAPT=0)" );
			if ( framegen_net_requested( g_eFramegenPipeline ) )
			{
				const bool bConservativeBidir = vulkan_framegen_bidir_active();
				if ( bConservativeBidir )
					vk_log.infof( "framegen: learned bidirectional confidence veto active (C) — FB-checked geometry is preserved and confidence can only decrease" );
				else
					vk_log.infof( "framegen: learned forward-field refinement active (C) — the net improves causal motion prediction once per real frame (bounded flow residual + evidence-gated confidence)" );
				if ( g_eFramegenPipeline == GamescopeFramegenPipeline::Guided && !bBidirActive )
					vk_log.infof( "framegen: causal shading-persistence head %s — three-frame in-situ supervision, bounded color-trend correction (GAMESCOPE_FRAMEGEN_SHADING=0 disables for A/B)",
						framegen_shading_enabled( g_eFramegenPipeline ) ? "enabled" : "disabled" );
				if ( framegen_net_online_enabled() )
					vk_log.infof( "framegen: in-situ learning active (C2) — %s (lr=%g, %u tiles/step, decay-to-prior; GAMESCOPE_FRAMEGEN_NET_EVERY=%u)%s",
						bConservativeBidir
							? "each real frame trains only bidir's conservative confidence output row; geometry heads and shared trunk stay frozen"
							: "the net keeps training on the framegen GPU against every real frame",
						framegen_net_online_lr(), k_uFramegenNetTrainTiles, framegen_net_online_every(),
						framegen_net_profile_path() != nullptr
							? " — persistent per-game profile (checkpointed on an owned worker, flushed at exit/reset, atomic replace)"
							: " — ephemeral model, nothing is written to disk" );
			}
			else if ( g_eFramegenPipeline < GamescopeFramegenPipeline::Learned
				&& framegen_net_lds_supported()
				&& ( framegen_net_weights_path() != nullptr || framegen_net_online_enabled() ) )
			{
				vk_log.infof( "framegen: learned refinement requested but not scheduled below the learned pipeline" );
			}
			if ( framegen_record_dir() != nullptr )
				vk_log.infof( "framegen: dataset capture requested — writing up to %u field-res training samples to '%s'", framegen_record_max(), framegen_record_dir() );
			if ( framegen_color_record_dir() != nullptr )
			{
				if ( framegen_color_probe_requested() )
				{
					vk_log.infof( "framegen: full-colour held-out validation active (E2) — A/B/C sequences offset=%u span=%u phaseTolerance=%.3f produce paired invisible B predictions (%s 0/0.5/1) on the dedicated queue; generated frames never present; synchronous GSCF writes can perturb capture cadence; up to %u samples write to '%s'",
						framegen_color_record_offset(), framegen_color_record_span(),
						framegen_color_record_phase_tolerance(),
						gamescope::framegen::color_probe_sweep_name( framegen_color_record_sweep() ),
						framegen_color_record_max(), framegen_color_record_dir() );
				}
				else
					vk_log.infof( "framegen: GAMESCOPE_FRAMEGEN_RECORD_COLOR ignored (requires motion+bidir, dedicated framegen queue, and base-layer mode off)" );
			}
		}
		g_bLoggedFramegenConfig = true;
	}

	// Only a new base-layer commit counts as a real frame. Overlay-only
	// repaints (a MangoHud tick, a notification fading) re-composite the same
	// game content: recording them would poison the pacing measurement and
	// pay for a duplicate history copy. Identity keys on steamcompmgr's commit
	// id, so a client that reacquires and recommits a buffer mapping to the
	// same texture object is still recognised as new content; pointer identity
	// only backs layers that carry no commit id.
	const CVulkanTexture *pBaseTexture = pFrameInfo->layerCount > 0 ? pFrameInfo->layers[ 0 ].tex.get() : nullptr;
	const uint64_t ulBaseCommitID = pFrameInfo->layerCount > 0
		? pFrameInfo->layers[ 0 ].ulCommitID : 0u;
	if ( pBaseTexture && !gamescope::framegen::is_new_real_frame_content(
			{ g_framegenHistory.ulLastBaseCommitID, g_framegenHistory.pLastBaseTexture },
			{ ulBaseCommitID, pBaseTexture } ) )
	{
		g_framegenHistory.bBidirSameBaseComposite = true;
		static uint64_t s_uOverlayOnlyDebugLogCounter = 0;
		if ( FramegenDebugShouldLog( s_uOverlayOnlyDebugLogCounter ) )
			vk_log.infof( "framegen: ignoring overlay-only repaint (base content unchanged)" );
		return;
	}
	g_framegenHistory.pLastBaseTexture = pBaseTexture;
	g_framegenHistory.ulLastBaseCommitID = ulBaseCommitID;
	if ( framegen_record_color_probe_real( pRealFrame, pFrameInfo, ulCompositeSeqNo ) )
		return;

	// #02 dispatcher: decide per recorded frame whether the base-layer path
	// applies. History then tracks the pre-upscale game buffer instead of the
	// composited output; the dims/mode-keyed reset inside ensure_resources
	// mediates any live switch between the two, so they never mix in a scene.
	const bool bBaseLayer = framegen_base_layer_usable( pFrameInfo );
	const gamescope::Rc<CVulkanTexture> &pHistoryFrame = bBaseLayer ? pFrameInfo->layers[ 0 ].tex : pRealFrame;

	if ( !framegen_ensure_resources( pHistoryFrame->width(), pHistoryFrame->height(), pHistoryFrame->drmFormat(), bBaseLayer ) )
		return;

	if ( bBaseLayer )
	{
		// Overlays are not part of base-mode history — they are composited
		// fresh onto each generated frame at present time — so an overlay
		// appearing or vanishing no longer invalidates prediction (toggling a
		// HUD or a notification popping used to reset generation; now it
		// doesn't). What IS a discontinuity is the game buffer's own
		// colorspace flipping on an unchanged DRM format: prediction across
		// it would blend two encodings.
		const int nBaseColorspace = (int)pFrameInfo->layers[ 0 ].colorspace;
		if ( g_framegenHistory.nLastBaseColorspace != nBaseColorspace )
		{
			if ( g_framegenHistory.nLastBaseColorspace != -1 )
				vulkan_framegen_invalidate_history( "base_colorspace_change" );
			g_framegenHistory.nLastBaseColorspace = nBaseColorspace;
		}
		g_framegenHistory.nLastLayerCount = pFrameInfo->layerCount;
		g_framegenHistory.eLastEOTF = pFrameInfo->outputEncodingEOTF;
	}
	// A layer-count change (overlay appearing/vanishing) or an output-encoding
	// change (SDR<->HDR can flip the EOTF without changing the image format)
	// abruptly replaces scene content; drop the history rather than smear the
	// old scene over the new one for a frame.
	else if ( g_framegenHistory.nLastLayerCount != pFrameInfo->layerCount || g_framegenHistory.eLastEOTF != pFrameInfo->outputEncodingEOTF )
	{
		vulkan_framegen_invalidate_history( g_framegenHistory.eLastEOTF != pFrameInfo->outputEncodingEOTF ? "output_eotf_change" : "layer_count_change" );
		g_framegenHistory.nLastLayerCount = pFrameInfo->layerCount;
		g_framegenHistory.eLastEOTF = pFrameInfo->outputEncodingEOTF;
	}

	uint64_t now = get_time_in_nanos();
	uint64_t ulPrevRealFrameTimeNs = g_framegenHistory.currentPresentTimeNs;
	const uint64_t ulSourceReadyTimeNs = pFrameInfo->layerCount > 0
		? pFrameInfo->layers[ 0 ].acquireReadyTimeNs : 0u;
	const bool bSourceTimestampValid = ulSourceReadyTimeNs != 0u
		&& ulSourceReadyTimeNs <= now;

	g_framegenHistory.currentFrameId++;
	g_framegenHistory.previousPresentTimeNs = ulPrevRealFrameTimeNs;
	g_framegenHistory.currentPresentTimeNs = now;
	// Provisional display-clock anchor: the vblank this composite is expected to
	// scan out at. The tagged backend feedback replaces it with the correlated
	// actual flip before future causal phases are planned.
	uint64_t ulRawRealVblankNs = GetVBlankTimer().GetNextVBlank( 0 );
	g_framegenHistory.ulCurrentRealVblankNs = ulRawRealVblankNs;
	g_framegenHistory.lastCompositeSeqNo = ulCompositeSeqNo;
	g_framegenHistory.nLastGeneratedSlot = 0;
	g_framegenHistory.nLastGenerationGapVblanks = 0;
	framegen_select_present_tag( gamescope::FramegenPresentKind_t::Real,
		g_framegenPresentState.ulCurrentRealFrameId, 0, ulCompositeSeqNo,
		g_framegenHistory.ulCurrentRealVblankNs );

	static uint64_t s_uRealFrameDebugLogCounter = 0;
	if ( FramegenDebugShouldLog( s_uRealFrameDebugLogCounter ) )
	{
		vk_log.infof( "framegen: real frame id=%" PRIu64 " time=%" PRIu64 " size=%ux%u format=0x%" PRIX32,
			g_framegenHistory.currentFrameId,
			now,
			pRealFrame->width(),
			pRealFrame->height(),
			pRealFrame->drmFormat() );
	}

	// A very long gap is a scene discontinuity (stall, load screen); drop history
	// so the previous scene is never smeared across the resume.
	// Pace against the SAME refresh the vblank scheduler counts in. The vblank
	// timer uses GetRefresh() == (g_nNestedRefresh ? g_nNestedRefresh : g_nOutputRefresh);
	// in nested mode g_nOutputRefresh is later overwritten with the PARENT monitor's
	// refresh while slots are still placed on g_nNestedRefresh's cadence. Deriving the
	// interval from g_nOutputRefresh there desynchronises nGapVblanks / phase / strength
	// from where the frame is actually shown — the temporal wobble. On DRM the two are
	// equal, so this is a no-op there.
	const int nFramegenRefreshMhz = g_nNestedRefresh ? g_nNestedRefresh : g_nOutputRefresh;
	const uint64_t ulVblankIntervalNs = nFramegenRefreshMhz > 0 ? 1'000'000'000'000ull / (uint64_t)nFramegenRefreshMhz : 8'333'333ull;
	if ( g_framegenHistory.valid && ulPrevRealFrameTimeNs != 0 && now - ulPrevRealFrameTimeNs > k_ulFramegenMaxRealFrameGapNs )
	{
		vulkan_framegen_invalidate_history( "frame_gap" );
		ulPrevRealFrameTimeNs = 0;
		g_framegenHistory.previousPresentTimeNs = 0;
		g_framegenHistory.currentPresentTimeNs = now;
		ulRawRealVblankNs = GetVBlankTimer().GetNextVBlank( 0 );
		g_framegenHistory.ulCurrentRealVblankNs = ulRawRealVblankNs;
		g_framegenHistory.lastCompositeSeqNo = ulCompositeSeqNo;
	}
	const bool bUseSourceCadence = framegen_observe_display_chain(
		ulVblankIntervalNs, bSourceTimestampValid );
	const uint64_t ulCadenceTimeNs = bUseSourceCadence
		? ulSourceReadyTimeNs : now;
	const uint64_t ulPreviousCadenceTimeNs =
		g_framegenHistory.ulCurrentCadenceTimeNs;

	// Live cadence predictor (#06): acquire-fence completion is the earliest
	// trustworthy observation that the renderer's frame can enter the compositor,
	// and unlike paint time it has not yet been quantized to a fixed vblank. A
	// backend without that timestamp falls back to this composite's monotonic time.
	// Never mix the two time bases in one learned state. The bounded alpha-beta
	// update tracks gradual rate changes; its one-sided residual envelope learns
	// how late this workload gets and feeds the slot-deadline admission below.
	uint64_t ulCadenceSampleNs = 0u;
	const bool bSameCadenceClock = g_framegenHistory.ulCurrentCadenceTimeNs != 0u
		&& g_framegenHistory.bCadenceUsesSourceTime == bUseSourceCadence;
	if ( bSameCadenceClock
		&& ulCadenceTimeNs > g_framegenHistory.ulCurrentCadenceTimeNs
		&& ulCadenceTimeNs - g_framegenHistory.ulCurrentCadenceTimeNs
			<= k_ulFramegenMaxRealFrameGapNs )
	{
		ulCadenceSampleNs = ulCadenceTimeNs
			- g_framegenHistory.ulCurrentCadenceTimeNs;
	}
	else if ( g_framegenHistory.ulCurrentCadenceTimeNs != 0u )
	{
		// A source/composite clock transition was already counted once by the
		// display-chain observer above. Only malformed ordering within the same
		// provenance is an additional cadence reset.
		if ( bSameCadenceClock )
			framegen_metrics_note_reset( FramegenResetReason_t::Provenance );
		g_framegenHistory.cadence = {};
	}

	if ( ulCadenceSampleNs != 0u )
	{
		const uint64_t ulPriorNs = framegen_predicted_interval_ns();
		// Hitch marker: an isolated real-frame interval well past the running
		// cadence is exactly the "smooth for a long time, then a bump" the eye
		// catches. The threshold (2x cadence AND +10ms absolute) sits above
		// nested parent-vblank doubling and a game's ordinary variance, so what
		// remains logs on every occurrence (not decimated — real hangs are rare
		// by definition) and can be correlated with the surrounding reset /
		// snap / degrade / checkpoint lines to attribute the source. Two
		// compares per real frame; debug-gated output only.
		if ( g_bFramegenDebug && ulPriorNs != 0
			&& ulCadenceSampleNs > std::max<uint64_t>(
				ulPriorNs * 2u, ulPriorNs + 10'000'000ull ) )
		{
			static uint64_t s_ulLastSpikeNs = 0;
			vk_log.infof( "framegen: frametime spike — %s gap %.2fms vs predicted %.2fms (frame id=%" PRIu64 ", %.1fs since last)",
				bUseSourceCadence ? "source-ready" : "composite",
				ulCadenceSampleNs * 1e-6, ulPriorNs * 1e-6,
				g_framegenHistory.currentFrameId,
				s_ulLastSpikeNs != 0 ? ( now - s_ulLastSpikeNs ) * 1e-9 : 0.0 );
			s_ulLastSpikeNs = now;
		}
		g_framegenHistory.cadence = gamescope::framegen::update_cadence_predictor(
			g_framegenHistory.cadence, ulCadenceSampleNs );
	}
	g_framegenHistory.ulCurrentCadenceTimeNs = ulCadenceTimeNs;
	g_framegenHistory.bCadenceUsesSourceTime = bUseSourceCadence;
	const bool bCausalDeadline = framegen_causal_deadline_enabled()
		&& !vulkan_framegen_vrr_hybrid_active()
		&& !vulkan_framegen_bidir_active();
	if ( bCausalDeadline )
	{
		// A real always supersedes causal predictions. Establish the new
		// provisional anchor now; its tagged flip feedback will replace this
		// value and may invalidate a slot before the arbiter can present it.
		g_framegenHistory.pending.clear();
		framegen_record_causal_anchor(
			g_framegenPresentState.ulCurrentRealFrameId,
			bUseSourceCadence ? ulSourceReadyTimeNs : 0u,
			ulRawRealVblankNs,
			ulVblankIntervalNs,
			bUseSourceCadence );
		g_framegenHistory.ulCurrentRealVblankNs =
			g_framegenHistory.causalAnchor.provisionalTargetNs;
		framegen_select_present_tag( gamescope::FramegenPresentKind_t::Real,
			g_framegenPresentState.ulCurrentRealFrameId, 0, ulCompositeSeqNo,
			g_framegenHistory.ulCurrentRealVblankNs );
	}

	// Two conditions decide whether the last observed interval certainly left an
	// empty vblank, both without dropping history:
	//  - the game must leave an empty vblank (gap > ~1.5 intervals); a faster
	//    interval has no slot to fill, so generating would displace real content.
	//  - the previous batch must have finished; generating past an unfinished one
	//    piles work the compositing GPU can't consume.
	// On a shared queue this remains the admission gate because generation could
	// sit in front of a later real composite. On a dedicated framegen queue it is
	// only a confidence signal: we can speculatively generate after every usable
	// real frame and discard the prediction when a real frame wins the vblank.
	// That spends the second GPU's slack to cover sudden missed slots without
	// adding real-frame latency.
	const bool bLeavesEmptyVblank = gamescope::framegen::leaves_empty_vblank(
		now, ulPrevRealFrameTimeNs, ulVblankIntervalNs );
	const bool bGpuHasHeadroom = g_device.hasCompletedFramegen( g_framegenHistory.lastGeneratedSeqNo );
	const bool bGeneratable = bLeavesEmptyVblank && bGpuHasHeadroom;
	const bool bCanSpeculate = g_device.hasFramegenQueue() && bGpuHasHeadroom && ulPrevRealFrameTimeNs != 0;
	const uint32_t nShadowConfidence = g_framegenHistory.valid
		? gamescope::framegen::update_cadence_confidence(
			g_framegenHistory.nStableFrames, bGeneratable )
		: g_framegenHistory.nStableFrames;
	const bool bShadowSharedQueueProvenEmpty = g_framegenHistory.valid
		&& gamescope::framegen::reactive_generation_ready(
			bGeneratable, nShadowConfidence );
	const uint32_t nShadowClassicGap = ulPrevRealFrameTimeNs != 0u
		&& now > ulPrevRealFrameTimeNs
		? gamescope::framegen::measured_gap_vblanks(
			now - ulPrevRealFrameTimeNs, ulVblankIntervalNs )
		: 0u;
	framegen_shadow_plan_real(
		g_framegenPresentState.ulCurrentRealFrameId,
		bUseSourceCadence ? ulSourceReadyTimeNs : 0u,
		ulRawRealVblankNs,
		g_framegenHistory.cadence,
		now,
		ulVblankIntervalNs,
		bUseSourceCadence,
		g_device.hasFramegenQueue(),
		bShadowSharedQueueProvenEmpty,
		nShadowClassicGap );

	// History shift, on every kept real frame (even non-generatable ones) so
	// the two most recent reals stay fresh and a slowdown resumes generation
	// without a re-prime. Base mode (#02) copies the client's base buffer into
	// an owned image on the framegen queue; output-space mode is the zero-copy
	// shift — the previous "current" becomes "previous" and the just-composited
	// output image becomes "current", holding references into the output ring.
	if ( bBaseLayer )
	{
		if ( !framegen_base_record_copy( pHistoryFrame, ulCompositeSeqNo ) )
			return;
		// Base mode composites the live cursor onto every generated frame in
		// framegen_base_present_composite already; the output-space split is
		// inert here and must never claim otherwise.
		g_framegenHistory.bCursorFreePrevious = false;
		g_framegenHistory.bCursorFreeCurrent = false;
	}
	else
	{
		g_framegenHistory.previousReal = g_framegenHistory.currentReal;
		g_framegenHistory.currentReal = pRealFrame;
		g_framegenHistory.bCursorFreePrevious = g_framegenHistory.bCursorFreeCurrent;
		g_framegenHistory.bCursorFreeCurrent = bCursorFreeRealFrame;
	}
	g_framegenHistory.previousFrameId = g_framegenHistory.currentFrameId - 1;

	// Prime: the first frame after a reset/invalidation only establishes history.
	if ( !g_framegenHistory.valid )
	{
		static uint64_t s_uPrimeDebugLogCounter = 0;
		if ( FramegenDebugShouldLog( s_uPrimeDebugLogCounter ) )
			vk_log.infof( "framegen: priming history with real frame id=%" PRIu64, g_framegenHistory.currentFrameId );
		g_framegenHistory.valid = true;
		return;
	}

	// Leaky-bucket hysteresis is stateless policy in scheduling.hpp. Only score
	// once we have a real gap to judge; the priming frame never reaches here.
	g_framegenHistory.nStableFrames = gamescope::framegen::update_cadence_confidence(
		g_framegenHistory.nStableFrames, bGeneratable );

	// Max degradation rungs for the startup config (0 for extrapolate/blend x2).
	// Needed here for the dormant log; the ladder itself is evaluated further down,
	// only on frames we actually generate, so it steps on real generating frames
	// and never on an idle/dormant frame.
	const uint32_t nMaxDegradeSteps = framegen_max_degrade_steps();
	const bool bVrrHybrid = vulkan_framegen_vrr_hybrid_active();
	const bool bBidirDeadline = vulkan_framegen_bidir_active();
	const uint64_t ulBidirPreviousSourceNs = ulCadenceSampleNs != 0u
		? ulPreviousCadenceTimeNs : 0u;
	if ( bBidirDeadline && bGpuHasHeadroom )
	{
		// Consume scene classification before planning the new pair. A cut keeps
		// queued endpoints but invalidates the translation, allowing this pair to
		// re-prime only when both post-cut source timestamps are compatible.
		framegen_adapt_consume(
			framegen_effective_config(
				g_framegenHistory.nDegradeSteps ).pipeline );
	}

	if ( bVrrHybrid )
	{
		framegen_record_vrr_real_for_feedback(
			g_framegenPresentState.ulCurrentRealFrameId,
			ulCompositeSeqNo, framegen_predicted_interval_ns(),
			ulVblankIntervalNs,
			bCanSpeculate && ulCadenceSampleNs != 0u );
		return;
	}

	const bool bReactiveReady = gamescope::framegen::reactive_generation_ready(
		bGeneratable, g_framegenHistory.nStableFrames );
	const bool bShouldGenerate = bCanSpeculate || bReactiveReady;

	if ( !bShouldGenerate )
	{
		if ( bBidirDeadline )
		{
			framegen_bidir_plan_pair(
				ulBidirPreviousSourceNs, ulCadenceTimeNs,
				g_framegenPresentState.pendingTag.ulTargetFlipNs,
				ulVblankIntervalNs,
				framegen_effective_config(
					g_framegenHistory.nDegradeSteps ),
				ulCompositeSeqNo, nMaxDegradeSteps, false,
				bUseSourceCadence );
			return;
		}
		static uint64_t s_uDormantDebugLogCounter = 0;
		if ( FramegenDebugShouldLog( s_uDormantDebugLogCounter ) )
		{
			const char *pszState = !bGpuHasHeadroom ? "busy" : ( bGeneratable ? "stabilizing" : "dormant" );
			vk_log.infof( "framegen: %s %u/%u degrade=%u/%u", pszState,
				g_framegenHistory.nStableFrames, gamescope::framegen::k_uCadenceConfidenceRequired,
				g_framegenHistory.nDegradeSteps, nMaxDegradeSteps );
		}
		return;
	}

	if ( g_framegenHistory.previousReal == nullptr || g_framegenHistory.currentReal == nullptr )
		return;

	// Whole-vblank gap inferred from the last real interval. On a dedicated
	// framegen queue, plan at least the configured multiplier's worth of future
	// slots even when the last interval was fast. If a real frame arrives, the
	// pending prediction is discarded before it can add latency; if it misses,
	// AMD already has work queued for the empty vblank.
	const uint32_t nMeasuredGapVblanks = gamescope::framegen::measured_gap_vblanks(
		now - ulPrevRealFrameTimeNs, ulVblankIntervalNs );

	if ( bCausalDeadline )
	{
		framegen_causal_submit( ulCompositeSeqNo );
		return;
	}

	// Deadline-driven degradation (#04): shed one pipeline rung whenever the CURRENT
	// config's measured GPU cost (see framegenGarbageCollect) overruns the vblank
	// budget, so a too-slow config never causes a missed generated frame. Evaluated
	// only here, on frames we actually generate, so the rung it settles on is a rung
	// that was really measured and an idle/dormant stretch never moves it.
	//
	// Recovery remains at this same decision site and may only select the adjacent
	// richer rung after a long run of measured headroom. Degradation and its hold
	// retain precedence; a failed recovery probe increases the scene-local back-off.
	// When no measurement is available, neither direction moves and the existing
	// reactive discard remains the safety net.
	const FramegenEffective_t curEffForLadder = framegen_effective_config( g_framegenHistory.nDegradeSteps );
	const uint32_t nLadderGapVblanks = gamescope::framegen::expanded_gap_vblanks(
		nMeasuredGapVblanks, curEffForLadder.multiplier, bCanSpeculate );
	const uint32_t nGapSlots = nLadderGapVblanks > 1 ? nLadderGapVblanks - 1 : 0;
	// VRR hybrid always submits one-slot batches, so its rung costs are keyed by
	// count 1 and only the mode rung
	// (motion pipeline or motion->extrapolate) can shed work — a multiplier notch cannot reduce a
	// count that is already minimal, and the "does the step actually help"
	// check below correctly never takes it.
	const bool bSingleSlotPacing = bVrrHybrid;
	const uint32_t nCurGenForLadder = gamescope::framegen::ladder_generated_count(
		nGapSlots, curEffForLadder.multiplier, bSingleSlotPacing );
	const uint64_t ulCurRungCostNs = g_device.framegenRungCostNs( g_framegenHistory.nDegradeSteps, nCurGenForLadder );
	const uint32_t uCurRungSamples = g_device.framegenRungSampleCount( g_framegenHistory.nDegradeSteps, nCurGenForLadder );
	const uint32_t nRecoveryHold = g_framegenHistory.nDegradeHold;
	gamescope::framegen::DeadlineLadderEvaluation ladderEvaluation =
		gamescope::framegen::evaluate_deadline_ladder(
			{ g_framegenHistory.nDegradeSteps, g_framegenHistory.nDegradeHold },
			nMaxDegradeSteps, ulCurRungCostNs, uCurRungSamples, ulVblankIntervalNs );
	if ( ladderEvaluation.state.holdFrames != g_framegenHistory.nDegradeHold )
		g_framegenHistory.nDegradeHold = ladderEvaluation.state.holdFrames;
	bool bDegradedThisDecision = false;
	if ( ladderEvaluation.tryDegrade )
	{
		// Over budget at the current rung. Only take the step if it actually
		// reduces work at THIS gap: dropping a motion pipeline or motion->extrapolate
		// always lowers per-frame cost, but a pure multiplier notch only helps when the gap
		// lets it generate fewer frames. Otherwise stepping would add cost
		// (a coarser cadence / fewer inserted frames) for zero GPU saving.
		const FramegenEffective_t nextEff = framegen_effective_config( g_framegenHistory.nDegradeSteps + 1 );
		const uint32_t nNextGen = gamescope::framegen::ladder_generated_count(
			nGapSlots, nextEff.multiplier, bSingleSlotPacing );
		if ( gamescope::framegen::degradation_reduces_work(
			curEffForLadder, nextEff, nCurGenForLadder, nNextGen ) )
		{
			ladderEvaluation.state = gamescope::framegen::commit_deadline_degradation(
				ladderEvaluation.state );
			g_framegenHistory.nDegradeSteps = ladderEvaluation.state.degradeSteps;
			g_framegenHistory.nDegradeHold = ladderEvaluation.state.holdFrames;
			bDegradedThisDecision = true;
			framegen_recovery_note_degradation();
		}
	}
	if ( !bDegradedThisDecision && framegen_recovery_active_for_path() )
	{
		const uint32_t nCurrentRung = g_framegenHistory.nDegradeSteps;
		const uint64_t ulRecoveryBudgetNs =
			gamescope::framegen::deadline_budget_ns( ulVblankIntervalNs );
		std::array<gamescope::framegen::LadderRungCost_t,
			CVulkanDevice::kFramegenLadderSlots> rungCosts = {};
		for ( uint32_t nRicherRung = 0u;
			nRicherRung < nCurrentRung; nRicherRung++ )
		{
			const FramegenEffective_t richerEff =
				framegen_effective_config( nRicherRung );
			const uint32_t nRicherGapVblanks =
				gamescope::framegen::expanded_gap_vblanks(
					nMeasuredGapVblanks, richerEff.multiplier,
					bCanSpeculate );
			const uint32_t nRicherGapSlots = nRicherGapVblanks > 1u
				? nRicherGapVblanks - 1u : 0u;
			const uint32_t nRicherGen =
				gamescope::framegen::ladder_generated_count(
					nRicherGapSlots, richerEff.multiplier,
					bSingleSlotPacing );
			rungCosts[ nRicherRung ] = {
				g_device.framegenRungCostNs( nRicherRung, nRicherGen ),
				g_device.framegenRungSampleCount( nRicherRung, nRicherGen ),
			};
		}
		const gamescope::framegen::LadderRecoveryTarget_t recoveryTarget =
			gamescope::framegen::select_ladder_recovery_target(
				rungCosts, nCurrentRung, ulRecoveryBudgetNs );

		const gamescope::framegen::LadderRecoveryEvaluation_t recovery =
			gamescope::framegen::evaluate_ladder_recovery(
				g_framegenHistory.recovery,
				nCurrentRung, nRecoveryHold,
				ulCurRungCostNs, uCurRungSamples,
				recoveryTarget.evidence.costNs,
				recoveryTarget.evidence.samples,
				ulRecoveryBudgetNs );
		g_framegenHistory.recovery = recovery.state;
		if ( recovery.reportBlockedThreshold )
			framegen_log_ladder_recovery_blocked(
				nMaxDegradeSteps, nRecoveryHold );
		if ( recovery.tryRecover && recoveryTarget.rung < nCurrentRung )
		{
			const uint32_t uEvidenceDecisions = recovery.state.streak;
			g_framegenHistory.nDegradeSteps = recoveryTarget.rung;
			g_framegenHistory.recovery =
				gamescope::framegen::commit_ladder_recovery(
					g_framegenHistory.recovery );
			framegen_log_ladder_recovery(
				nMaxDegradeSteps, nCurrentRung, uEvidenceDecisions );
		}
	}

	const FramegenEffective_t eff = framegen_effective_config( g_framegenHistory.nDegradeSteps );
	const uint64_t ulPredictedIntervalNs = framegen_predicted_interval_ns();

	// VRR hybrid (#01): the real frame just presented immediately (adaptive
	// sync — no grid quantization, no added latency), and the one generated
	// frame is placed at the content midpoint of the measured interval by a
	// correlated absolute deadline. Keep-up guard: skip when the interval is too
	// short to split; scheduling.hpp owns the threshold.
	if ( bVrrHybrid )
	{
		g_framegenHistory.pending.clear();
		if ( gamescope::framegen::vrr_hybrid_interval_eligible(
			ulPredictedIntervalNs, ulVblankIntervalNs ) )
		{
			framegen_vrr_hybrid_submit( ulCompositeSeqNo, nMaxDegradeSteps );
		}
		else
		{
			static uint64_t s_uHybridKeepUpDebugLogCounter = 0;
			if ( FramegenDebugShouldLog( s_uHybridKeepUpDebugLogCounter ) )
				vk_log.infof( "framegen: vrr-hybrid keep-up skip cadence=%.2fms min-flip-interval=%.2fms",
					ulPredictedIntervalNs / 1.0e6, ulVblankIntervalNs / 1.0e6 );
		}
		return;
	}

	// The absolute-epoch planner schedules the real endpoint even when it emits
	// zero candidates.
	if ( bBidirDeadline )
	{
		framegen_bidir_plan_pair(
			ulBidirPreviousSourceNs, ulCadenceTimeNs,
			g_framegenPresentState.pendingTag.ulTargetFlipNs,
			ulVblankIntervalNs, eff, ulCompositeSeqNo,
			nMaxDegradeSteps, true, bUseSourceCadence );
		return;
	}

	// Fill as many empty vblanks as the measured gap actually offers, capped by
	// the multiplier: generate one fewer than the whole-vblank gap (the final slot
	// is the next real frame). The ladder's effective multiplier can lower this
	// ceiling below the startup one under GPU pressure; it is always
	// <= g_nFramegenMultiplier, so the pre-sized output pool holds.
	const uint32_t nGapVblanks = gamescope::framegen::expanded_gap_vblanks(
		nMeasuredGapVblanks, eff.multiplier, bCanSpeculate );
	const uint32_t nGenerate = gamescope::framegen::generated_slots_for_gap(
		nGapVblanks, eff.multiplier, g_device.hasFramegenQueue() );

	// Without a dedicated framegen queue the batch is submitted to the same
	// in-order queue as the next composite, so it sits in front of it. Cap the
	// single-queue path to one generated frame to bound that head-of-line work —
	// the proven x2-prototype behaviour — rather than amplifying it under x3/x4.
	// generated_slots_for_gap applies that cap while forming nGenerate above.
	if ( nGenerate == 0 )
		return;

	// Any leftover pending frames belong to an older prediction; drop them before
	// queuing this interval's batch (normally already done by the supersede path
	// in the present decision).
	framegen_submit_batch( 1, nGapVblanks, nGenerate, eff,
		ulCompositeSeqNo, nMaxDegradeSteps, true );
}

static bool frame_has_non_device_local_base_import( const struct FrameInfo_t *pFrameInfo )
{
	for ( int i = 0; i < pFrameInfo->layerCount; i++ )
	{
		const FrameInfo_t::Layer_t &layer = pFrameInfo->layers[ i ];
		if ( layer.zpos == g_zposBase
			&& layer.tex
			&& layer.tex->externalImage()
			&& !layer.tex->deviceLocal()
			&& layer.pCommitTexture != nullptr )
		{
			return true;
		}
	}
	return false;
}

static bool stage_non_device_local_base_imports( struct FrameInfo_t *pFrameInfo,
	CVulkanCmdBuffer *pCmdBuffer )
{
	bool bCopied = false;
	for ( int i = 0; i < pFrameInfo->layerCount; i++ )
	{
		FrameInfo_t::Layer_t &layer = pFrameInfo->layers[ i ];
		gamescope::Rc<CVulkanTexture> pSource = layer.tex;
		if ( layer.zpos != g_zposBase
			|| !pSource
			|| !pSource->externalImage()
			|| pSource->deviceLocal()
			|| layer.pCommitTexture == nullptr )
		{
			continue;
		}

		// The same commit can appear more than once during a fade. Once its owner
		// has been swapped, point every duplicate layer at the already-recorded
		// copy rather than copying the import twice.
		if ( layer.pCommitTexture->get() != pSource.get() )
		{
			gamescope::Rc<CVulkanTexture> pExisting = *layer.pCommitTexture;
			if ( pExisting
				&& pExisting->deviceLocal()
				&& pExisting->width() == pSource->width()
				&& pExisting->height() == pSource->height()
				&& pExisting->drmFormat() == pSource->drmFormat() )
			{
				layer.tex = std::move( pExisting );
			}
			continue;
		}

		gamescope::Rc<CVulkanTexture> pStaging = pSource->AcquireDeviceLocalStagingImage();
		if ( !pStaging )
			continue;

		if ( framegen_metrics_enabled() )
		{
			// s_DRMVKFormatTable's bpp column is bytes per pixel, not bits.
			const uint32_t uBytesPerPixel = DRMFormatGetBPP( pStaging->drmFormat() );
			g_framegenMetricsPendingEvents.copyBytes +=
				(uint64_t)pStaging->width() * pStaging->height() * uBytesPerPixel;
		}
		pCmdBuffer->copyImage( pSource, pStaging );
		layer.tex = pStaging;
		*layer.pCommitTexture = std::move( pStaging );
		if ( layer.pStagedCopyCount )
			( *layer.pStagedCopyCount )++;
		bCopied = true;

		if ( g_bDebugDualGpuRoute )
		{
			vk_log.infof( "dual-gpu-route: staged base import %p -> %p (%ux%u format 0x%" PRIX32 ")",
				pSource.get(), layer.tex.get(), layer.tex->width(), layer.tex->height(), layer.tex->drmFormat() );
		}
	}
	return bCopied;
}

// Record the dual-GPU staging copies into their OWN command buffer and submit it
// on the composite queue, ahead of (and separate from) the composite that
// samples the staged image. Previously the copy rode inside the composite
// command buffer, so a slow PCIe read of a host-visible client import sat
// directly in front of the composite the flip waits on; splitting it lets the
// GPU start the copy while the CPU is still recording the rest of the composite.
//
// Synchronization:
//  - Client readiness: the composite command buffer carries the caller's acquire
//    waits, and the copy now performs the first read of that client image, so the
//    waits are duplicated onto the staging submission (submitInternal already
//    includes TRANSFER in the wait stage mask for exactly this reason). Only the
//    waits are duplicated; the caller's release signals stay on the composite
//    submission, which is submitted after this one on the same queue.
//  - Copy -> composite (RAW): both submissions are on the composite queue, so
//    submission order supplies execution order, and this command buffer's end()
//    flush barrier (ALL_COMMANDS, TRANSFER_WRITE -> read|write) supplies memory
//    visibility. The composite's first use of the staged image is therefore a
//    clean first use needing no barrier - the same cross-command-buffer contract
//    emitSync2Barriers() and framegen_base_record_copy() already rely on.
//  - Reuse (WAR): AcquireDeviceLocalStagingImage only hands out a pool image with
//    a zero refcount. An image a previous composite still reads is referenced by
//    that command buffer's m_textureRefs until resetCmdBuffers() retires it on
//    timeline completion (and by the commit's pCommitTexture slot), so it is
//    skipped and the pool grows instead. Back-to-back frames cannot stomp it.
//
// No dual-GPU staged copy exists on the single-GPU route (the base import is
// already DEVICE_LOCAL, so frame_has_non_device_local_base_import is false and
// this returns before allocating anything), leaving that path byte-identical.
static void stage_base_imports_in_own_submission( struct FrameInfo_t *pFrameInfo,
	CVulkanCmdBuffer *pCompositeCmdBuffer )
{
	if ( !frame_has_non_device_local_base_import( pFrameInfo ) )
		return;

	std::unique_ptr<CVulkanCmdBuffer> pStageCommandBuffer = g_device.commandBuffer();
	if ( !pStageCommandBuffer )
	{
		// Out of command buffers: keep the frame correct by falling back to the
		// old inline recording rather than sampling an unwritten staging image.
		// With no buffer to fall back into either, skip staging entirely — the
		// layers keep pointing at the client import, exactly as before.
		if ( pCompositeCmdBuffer != nullptr )
			stage_non_device_local_base_imports( pFrameInfo, pCompositeCmdBuffer );
		return;
	}

	if ( pCompositeCmdBuffer != nullptr )
	{
		for ( const VulkanTimelinePoint_t &dependency : pCompositeCmdBuffer->GetExternalDependencies() )
			pStageCommandBuffer->AddDependency( dependency.pTimelineSemaphore, dependency.ulPoint );
	}

	// Only measure when someone reads the metrics line; the drain that consumes
	// these slots runs on the same gate.
	const int nQuerySlot = framegen_metrics_enabled()
		? g_device.stagingTimestampBegin( pStageCommandBuffer.get() ) : -1;

	const bool bCopied = stage_non_device_local_base_imports( pFrameInfo, pStageCommandBuffer.get() );
	if ( bCopied )
		g_device.stagingTimestampEnd( pStageCommandBuffer.get(), nQuerySlot );

	// Submit even when nothing was recorded (a duplicated fade layer, or a source
	// with no usable staging image): an empty submission is far cheaper than
	// destroying a command buffer that would otherwise be recycled, and it keeps
	// the seqNo association below unconditional.
	const uint64_t ulSeqNo = g_device.submit( std::move( pStageCommandBuffer ) );
	if ( bCopied )
		g_device.noteStagingSubmission( ulSeqNo, nQuerySlot );
}

std::optional<uint64_t> vulkan_composite( struct FrameInfo_t *frameInfo, gamescope::Rc<CVulkanTexture> pPipewireTexture, bool partial, gamescope::Rc<CVulkanTexture> pOutputOverride, bool increment, std::unique_ptr<CVulkanCmdBuffer> pInCommandBuffer )
{
	// Bidir (B3): each composite decides afresh whether it queued its real
	// frame; a composite that never records (overlay-only, partial, screenshot)
	// must not inherit the previous verdict at flip-substitution time.
	g_framegenHistory.bBidirQueuedReal = false;
	g_framegenHistory.bBidirSameBaseComposite = false;

	EOTF outputTF = frameInfo->outputEncodingEOTF;
	if (!frameInfo->applyOutputColorMgmt)
		outputTF = EOTF_Count; //Disable blending stuff.

	if ( g_bDebugDualGpuRoute )
	{
		const char *pszPath = "normal composite";
		if ( frameInfo->useFSRLayer0 )
			pszPath = "FSR composite";
		else if ( frameInfo->useNISLayer0 )
			pszPath = "NIS composite";
		else if ( frameInfo->blurLayer0 )
			pszPath = "blur composite";
		else if ( !g_reshade_effect.empty() )
			pszPath = "ReShade/effect composite";

		CVulkanTexture *pLayer0 = frameInfo->layerCount > 0 ? frameInfo->layers[0].tex.get() : nullptr;
		vk_log.infof( "dual-gpu-route: frame path %s layers %d partial %s output %ux%u pipewire %s override %s queue family %u",
			pszPath,
			frameInfo->layerCount,
			partial ? "yes" : "no",
			currentOutputWidth,
			currentOutputHeight,
			pPipewireTexture ? "yes" : "no",
			pOutputOverride ? "yes" : "no",
			g_device.queueFamily() );
		if ( pLayer0 )
		{
			vk_log.infof( "dual-gpu-route:   layer0 texture %ux%u drm format 0x%" PRIX32 " scale %.4f %.4f colorspace %d",
				pLayer0->width(),
				pLayer0->height(),
				pLayer0->drmFormat(),
				frameInfo->layers[0].scale.x,
				frameInfo->layers[0].scale.y,
				frameInfo->layers[0].colorspace );
		}
	}

	// ReShade records its own submission before the normal composite command
	// buffer, so it must be staged here (earlier than the non-ReShade path below)
	// for the effect to sample DEVICE_LOCAL too.
	if ( !g_reshade_effect.empty() )
		stage_base_imports_in_own_submission( frameInfo, pInCommandBuffer.get() );

	g_pLastReshadeEffect = nullptr;
	if (!g_reshade_effect.empty())
	{
		if (frameInfo->layers[0].tex)
		{
			ReshadeEffectKey key
			{
				.path             = g_reshade_effect,
				.bufferWidth      = frameInfo->layers[0].tex->width(),
				.bufferHeight     = frameInfo->layers[0].tex->height(),
				.bufferColorSpace = frameInfo->layers[0].colorspace,
				.bufferFormat     = frameInfo->layers[0].tex->format(),
				.techniqueIdx     = g_reshade_technique_idx,
			};

			ReshadeEffectPipeline* pipeline = g_reshadeManager.pipeline(key);
			g_pLastReshadeEffect = pipeline;

			if (pipeline != nullptr)
			{
				uint64_t seq = pipeline->execute(frameInfo->layers[0].tex, &frameInfo->layers[0].tex);
				g_device.wait(seq);
			}
		}
	}
	else
	{
		g_reshadeManager.clear();
	}

	gamescope::Rc<CVulkanTexture> compositeImage;
	if ( pOutputOverride )
		compositeImage = pOutputOverride;
	else
	{
		if ( vulkan_framegen_is_enabled() && !GetBackend()->UsesVulkanSwapchain() )
		{
			framegen_release_completed_read_pins();
			if ( vulkan_framegen_bidir_active() )
			{
				// Enforce the ring-derived ceiling before trying to acquire the
				// next real target. This is a hard ownership bound, not merely a
				// generated-slot admission preference.
				framegen_bidir_shed_to_capacity(
					framegen_bidir_pending_hard_capacity() );
			}
			std::optional<uint32_t> nAvailable = framegen_find_available_output_image( g_output.nOutImage );
			if ( !nAvailable )
			{
				// Backend ownership is released by asynchronous pageflip / wl_buffer
				// events. Make one non-blocking progress pass before sacrificing
				// history or a real frame. This is exceptional-path only; the normal
				// composite path pays no extra poll or syscall.
				GetBackend()->PollState();
				framegen_release_completed_read_pins();
				nAvailable = framegen_find_available_output_image( g_output.nOutImage );
			}
			if ( !nAvailable && vulkan_framegen_bidir_active() )
			{
				// The queue is now the only compositor-owned backlog we can shed
				// without waiting. Release exactly as much as necessary, generated
				// first and then stale endpoints, checking ownership after each
				// release so the normal one-interval timeline loses the minimum.
				while ( !nAvailable && g_framegenHistory.pending.size() > 1u )
				{
					const size_t nBefore = g_framegenHistory.pending.size();
					framegen_bidir_shed_to_capacity( nBefore - 1u );
					if ( g_framegenHistory.pending.size() == nBefore )
						break;
					nAvailable = framegen_find_available_output_image(
						g_output.nOutImage );
				}
			}
			if ( !nAvailable )
			{
				// Exceptional recovery is atomic for bidir: flush its entire
				// timeline, retain only the newest endpoint as the visible hold, and
				// invalidate the epoch/history so the next real composite re-primes.
				// Non-bidir keeps the same history-release behavior.
				vulkan_framegen_invalidate_history( "real_output_ring_pressure" );
				nAvailable = framegen_find_available_output_image( g_output.nOutImage );
			}
			if ( !nAvailable )
			{
				static uint64_t s_uOutputRingPressureDebugLogCounter = 0;
				if ( FramegenDebugShouldLog( s_uOutputRingPressureDebugLogCounter ) )
					framegen_log_output_ring_pressure();
				return std::nullopt;
			}
			if ( vulkan_framegen_bidir_active() )
				g_framegenHistory.nBidirRingPressureFailures = 0u;
			g_output.nOutImage = *nAvailable;
		}
		compositeImage = partial ? g_output.outputImagesPartialOverlay[ g_output.nOutImage ] : g_output.outputImages[ g_output.nOutImage ];
	}

	// Output-space cursor split, first half. Only a composite that will actually
	// be recorded as framegen history is a candidate — the same conditions the
	// framegen_record_real_frame call at the bottom of this function uses — and
	// only in output-space mode (base mode already late-composites the live
	// cursor), never under bidir (bidir queues history images as pending REAL
	// frames and flips them directly, so history MUST stay the flippable ring
	// image with the cursor in it) and never while the held-out colour probe is
	// capturing (it retains history images across frames on its own schedule).
	//
	// When it engages, everything below composites the stack WITHOUT the cursor
	// into an owned image, and the second pass at the end of the recording block
	// draws the live cursor from that image into the real scanout target.
	const bool bCursorSplitCandidate = framegen_cursor_split_enabled()
		&& vulkan_framegen_is_enabled()
		&& !GetBackend()->UsesVulkanSwapchain()
		&& !partial && pPipewireTexture == nullptr && pOutputOverride == nullptr
		&& !vulkan_framegen_bidir_active()
		&& framegen_color_record_dir() == nullptr
		&& !framegen_base_layer_usable( frameInfo )
		&& framegen_composite_records_history( frameInfo );
	const int nCursorSplitLayer = bCursorSplitCandidate
		? framegen_cursor_top_layer_index( frameInfo ) : -1;
	gamescope::Rc<CVulkanTexture> pCursorScanoutImage;
	FrameInfo_t::Layer_t cursorSplitLayer;
	if ( nCursorSplitLayer >= 0 && compositeImage != nullptr )
	{
		gamescope::Rc<CVulkanTexture> pCursorFreeImage = framegen_acquire_cursor_history_image(
			compositeImage->width(), compositeImage->height(), compositeImage->drmFormat() );
		if ( pCursorFreeImage != nullptr )
		{
			cursorSplitLayer = frameInfo->layers[ nCursorSplitLayer ];
			frameInfo->layerCount--;
			pCursorScanoutImage = compositeImage;
			compositeImage = std::move( pCursorFreeImage );
		}
	}
	const bool bCursorSplit = pCursorScanoutImage != nullptr;
	// True whenever the image handed to framegen carries no cursor: either the
	// split ran, or the pointer is hidden/grabbed and there was never one to
	// split out. A cursor that is present but not the topmost layer (mura
	// correction sits above it) keeps today's baked-in behaviour.
	const bool bCursorFreeHistory = bCursorSplitCandidate
		&& ( bCursorSplit || !framegen_frame_has_cursor_layer( frameInfo ) );

	auto cmdBuffer = pInCommandBuffer ? std::move( pInCommandBuffer ) : g_device.commandBuffer();
	// Off the real-frame critical path: the staging copy goes out as its own
	// submission now, so the composite recorded below (and the flip that waits on
	// it) no longer contains the cross-GPU copy inline. Kept here rather than
	// beside the ReShade pre-stage above so an early return from the output-ring
	// acquisition still leaves frameInfo untouched, exactly as before.
	if ( g_reshade_effect.empty() )
		stage_base_imports_in_own_submission( frameInfo, cmdBuffer.get() );

	for (uint32_t i = 0; i < EOTF_Count; i++)
		cmdBuffer->bindColorMgmtLuts(i, frameInfo->shaperLut[i], frameInfo->lut3D[i]);

	if ( frameInfo->useFSRLayer0 )
	{
		uint32_t inputX = frameInfo->layers[0].tex->width();
		uint32_t inputY = frameInfo->layers[0].tex->height();

		uint32_t tempX = frameInfo->layers[0].integerWidth();
		uint32_t tempY = frameInfo->layers[0].integerHeight();

		if ( g_bDebugDualGpuRoute )
		{
			vk_log.infof( "dual-gpu-route: FSR dispatch input %ux%u temp %ux%u output %ux%u queue family %u",
				inputX,
				inputY,
				tempX,
				tempY,
				currentOutputWidth,
				currentOutputHeight,
				g_device.queueFamily() );
		}

		update_tmp_images(tempX, tempY);

		cmdBuffer->bindPipeline(g_device.pipeline(SHADER_TYPE_EASU));
		cmdBuffer->bindTarget(g_output.tmpOutput);
		cmdBuffer->bindTexture(0, frameInfo->layers[0].tex);
		cmdBuffer->setTextureSrgb(0, true);
		cmdBuffer->setSamplerUnnormalized(0, false);
		cmdBuffer->setSamplerNearest(0, false);
		cmdBuffer->uploadConstants<EasuPushData_t>(inputX, inputY, tempX, tempY);

		int pixelsPerGroup = 16;

		cmdBuffer->dispatch(div_roundup(tempX, pixelsPerGroup), div_roundup(tempY, pixelsPerGroup));

		cmdBuffer->bindPipeline(g_device.pipeline(SHADER_TYPE_RCAS, frameInfo->layerCount, frameInfo->ycbcrMask() & ~1, 0u, frameInfo->colorspaceMask(), outputTF ));
		bind_all_layers(cmdBuffer.get(), frameInfo);
		cmdBuffer->bindTexture(0, g_output.tmpOutput);
		cmdBuffer->setTextureSrgb(0, true);
		cmdBuffer->setSamplerUnnormalized(0, false);
		cmdBuffer->setSamplerNearest(0, false);
		cmdBuffer->bindTarget(compositeImage);
		cmdBuffer->uploadConstants<RcasPushData_t>(frameInfo, g_upscaleFilterSharpness / 10.0f);

		cmdBuffer->dispatch(div_roundup(currentOutputWidth, pixelsPerGroup), div_roundup(currentOutputHeight, pixelsPerGroup));
	}
	else if ( frameInfo->useNISLayer0 )
	{
		uint32_t inputX = frameInfo->layers[0].tex->width();
		uint32_t inputY = frameInfo->layers[0].tex->height();

		uint32_t tempX = frameInfo->layers[0].integerWidth();
		uint32_t tempY = frameInfo->layers[0].integerHeight();

		if ( g_bDebugDualGpuRoute )
		{
			vk_log.infof( "dual-gpu-route: NIS dispatch input %ux%u temp %ux%u output %ux%u queue family %u",
				inputX,
				inputY,
				tempX,
				tempY,
				currentOutputWidth,
				currentOutputHeight,
				g_device.queueFamily() );
		}

		update_tmp_images(tempX, tempY);

		float nisSharpness = (20 - g_upscaleFilterSharpness) / 20.0f;

		cmdBuffer->bindPipeline(g_device.pipeline(SHADER_TYPE_NIS));
		cmdBuffer->bindTarget(g_output.tmpOutput);
		cmdBuffer->bindTexture(0, frameInfo->layers[0].tex);
		cmdBuffer->setTextureSrgb(0, true);
		cmdBuffer->setSamplerUnnormalized(0, false);
		cmdBuffer->setSamplerNearest(0, false);
		cmdBuffer->bindTexture(VKR_NIS_COEF_SCALER_SLOT, g_output.nisScalerImage);
		cmdBuffer->setSamplerUnnormalized(VKR_NIS_COEF_SCALER_SLOT, false);
		cmdBuffer->setSamplerNearest(VKR_NIS_COEF_SCALER_SLOT, false);
		cmdBuffer->bindTexture(VKR_NIS_COEF_USM_SLOT, g_output.nisUsmImage);
		cmdBuffer->setSamplerUnnormalized(VKR_NIS_COEF_USM_SLOT, false);
		cmdBuffer->setSamplerNearest(VKR_NIS_COEF_USM_SLOT, false);
		cmdBuffer->uploadConstants<NisPushData_t>(inputX, inputY, tempX, tempY, nisSharpness);

		int pixelsPerGroupX = 32;
		int pixelsPerGroupY = 24;

		cmdBuffer->dispatch(div_roundup(tempX, pixelsPerGroupX), div_roundup(tempY, pixelsPerGroupY));

		struct FrameInfo_t nisFrameInfo = *frameInfo;
		nisFrameInfo.layers[0].tex = g_output.tmpOutput;
		nisFrameInfo.layers[0].scale.x = 1.0f;
		nisFrameInfo.layers[0].scale.y = 1.0f;

		cmdBuffer->bindPipeline( g_device.pipeline(SHADER_TYPE_BLIT, nisFrameInfo.layerCount, nisFrameInfo.ycbcrMask(), 0u, nisFrameInfo.colorspaceMask(), outputTF ));
		bind_all_layers(cmdBuffer.get(), &nisFrameInfo);
		cmdBuffer->bindTarget(compositeImage);
		cmdBuffer->uploadConstants<BlitPushData_t>(&nisFrameInfo);

		int pixelsPerGroup = 8;

		cmdBuffer->dispatch(div_roundup(currentOutputWidth, pixelsPerGroup), div_roundup(currentOutputHeight, pixelsPerGroup));
	}
	else if ( frameInfo->blurLayer0 )
	{
		update_tmp_images(currentOutputWidth, currentOutputHeight);

		ShaderType type = SHADER_TYPE_BLUR_FIRST_PASS;

		uint32_t blur_layer_count = 1;
		// Also blur the override on top if we have one.
		if (frameInfo->layerCount >= 2 && frameInfo->layers[1].zpos == g_zposOverride)
			blur_layer_count++;

		cmdBuffer->bindPipeline(g_device.pipeline(type, blur_layer_count, frameInfo->ycbcrMask() & 0x3u, 0, frameInfo->colorspaceMask(), outputTF ));
		cmdBuffer->bindTarget(g_output.tmpOutput);
		for (uint32_t i = 0; i < blur_layer_count; i++)
		{
			cmdBuffer->bindTexture(i, frameInfo->layers[i].tex);
			cmdBuffer->setTextureSrgb(i, false);
			cmdBuffer->setSamplerUnnormalized(i, true);
			cmdBuffer->setSamplerNearest(i, false);
		}
		cmdBuffer->uploadConstants<BlitPushData_t>(frameInfo);

		int pixelsPerGroup = 8;

		cmdBuffer->dispatch(div_roundup(currentOutputWidth, pixelsPerGroup), div_roundup(currentOutputHeight, pixelsPerGroup));

		bool useSrgbView = frameInfo->layers[0].colorspace == GAMESCOPE_APP_TEXTURE_COLORSPACE_LINEAR;

		type = frameInfo->blurLayer0 == BLUR_MODE_COND ? SHADER_TYPE_BLUR_COND : SHADER_TYPE_BLUR;
		cmdBuffer->bindPipeline(g_device.pipeline(type, frameInfo->layerCount, frameInfo->ycbcrMask(), blur_layer_count, frameInfo->colorspaceMask(), outputTF ));
		bind_all_layers(cmdBuffer.get(), frameInfo);
		cmdBuffer->bindTarget(compositeImage);
		cmdBuffer->bindTexture(VKR_BLUR_EXTRA_SLOT, g_output.tmpOutput);
		cmdBuffer->setTextureSrgb(VKR_BLUR_EXTRA_SLOT, !useSrgbView); // Inverted because it chooses whether to view as linear (sRGB view) or sRGB (raw view). It's horrible. I need to change it.
		cmdBuffer->setSamplerUnnormalized(VKR_BLUR_EXTRA_SLOT, true);
		cmdBuffer->setSamplerNearest(VKR_BLUR_EXTRA_SLOT, false);

		cmdBuffer->dispatch(div_roundup(currentOutputWidth, pixelsPerGroup), div_roundup(currentOutputHeight, pixelsPerGroup));
	}
	else
	{
		cmdBuffer->bindPipeline( g_device.pipeline(SHADER_TYPE_BLIT, frameInfo->layerCount, frameInfo->ycbcrMask(), 0u, frameInfo->colorspaceMask(), outputTF ));
		bind_all_layers(cmdBuffer.get(), frameInfo);
		cmdBuffer->bindTarget(compositeImage);
		cmdBuffer->uploadConstants<BlitPushData_t>(frameInfo);

		const int pixelsPerGroup = 8;

		cmdBuffer->dispatch(div_roundup(currentOutputWidth, pixelsPerGroup), div_roundup(currentOutputHeight, pixelsPerGroup));
	}

	// Output-space cursor split, second half of the real frame. The pass above
	// produced the cursor-free composite in an owned image; draw the live cursor
	// from there into the actual scanout target. Recorded into the SAME command
	// buffer as the composite — one submission, no extra fence, no change to the
	// seqno the flip and the framegen scheduler wait on — and the command
	// buffer's own image-state tracking inserts the read-after-write barrier, as
	// it already does between EASU and RCAS. See
	// framegen_build_cursor_overlay_frame_info for the colour-management
	// contract; EOTF_Count is the pipeline key for "output encoding already
	// applied, do not apply it again".
	gamescope::Rc<CVulkanTexture> pFramegenHistoryImage = compositeImage;
	if ( bCursorSplit )
	{
		FrameInfo_t cursorFrameInfo;
		framegen_build_cursor_overlay_frame_info( &cursorFrameInfo, compositeImage,
			cursorSplitLayer, frameInfo );

		cmdBuffer->bindPipeline( g_device.pipeline(SHADER_TYPE_BLIT, cursorFrameInfo.layerCount, cursorFrameInfo.ycbcrMask(), 0u, cursorFrameInfo.colorspaceMask(), EOTF_Count ));
		bind_all_layers(cmdBuffer.get(), &cursorFrameInfo);
		cmdBuffer->bindTarget(pCursorScanoutImage);
		cmdBuffer->uploadConstants<BlitPushData_t>(&cursorFrameInfo);

		const int pixelsPerGroup = 8;

		cmdBuffer->dispatch(div_roundup(currentOutputWidth, pixelsPerGroup), div_roundup(currentOutputHeight, pixelsPerGroup));

		// Everything after this point — pipewire capture, the framegen HUD, the
		// returned/flipped image, the output ring advance — must see the real
		// scanout target again. Only framegen keeps the cursor-free one.
		compositeImage = std::move( pCursorScanoutImage );
		// Restore the caller's layer stack before anything else reads it;
		// framegen_record_real_frame keys scene changes on layerCount and must
		// keep seeing the true count.
		frameInfo->layerCount++;
	}

	if ( pPipewireTexture != nullptr )
	{

		if (compositeImage->format() == pPipewireTexture->format() &&
			compositeImage->width() == pPipewireTexture->width() &&
		    compositeImage->height() == pPipewireTexture->height()) {
			cmdBuffer->copyImage(compositeImage, pPipewireTexture);
		} else {
			const bool ycbcr = pPipewireTexture->isYcbcr();

			float scale = (float)compositeImage->width() / pPipewireTexture->width();
			if ( ycbcr )
			{
				CaptureConvertBlitData_t constants( scale, colorspace_to_conversion_from_srgb_matrix( pPipewireTexture->streamColorspace() ) );
				constants.halfExtent[0] = pPipewireTexture->width() / 2.0f;
				constants.halfExtent[1] = pPipewireTexture->height() / 2.0f;
				cmdBuffer->uploadConstants<CaptureConvertBlitData_t>(constants);
			}
			else
			{
				BlitPushData_t constants( scale );
				cmdBuffer->uploadConstants<BlitPushData_t>(constants);
			}

			for (uint32_t i = 0; i < EOTF_Count; i++)
				cmdBuffer->bindColorMgmtLuts(i, nullptr, nullptr);

			cmdBuffer->bindPipeline(g_device.pipeline( ycbcr ? SHADER_TYPE_RGB_TO_NV12 : SHADER_TYPE_BLIT, 1, 0, 0, GAMESCOPE_APP_TEXTURE_COLORSPACE_SRGB, EOTF_Count ));
			cmdBuffer->bindTexture(0, compositeImage);
			cmdBuffer->setTextureSrgb(0, true);
			cmdBuffer->setSamplerNearest(0, false);
			cmdBuffer->setSamplerUnnormalized(0, true);
			for (uint32_t i = 1; i < VKR_SAMPLER_SLOTS; i++)
			{
				cmdBuffer->bindTexture(i, nullptr);
			}
			cmdBuffer->bindTarget(pPipewireTexture);

			const int pixelsPerGroup = 8;

			// For ycbcr, we operate on 2 pixels at a time, so use the half-extent.
			const int dispatchSize = ycbcr ? pixelsPerGroup * 2 : pixelsPerGroup;

			cmdBuffer->dispatch(div_roundup(pPipewireTexture->width(), dispatchSize), div_roundup(pPipewireTexture->height(), dispatchSize));
		}
	}

	// The HUD is the final pass of a real composite. Capture/screenshot side
	// composites and base-layer late composites are identified by their capture
	// or override target and deliberately never enter this path; the direct
	// pipewire copy above also happens before the HUD is drawn.
	int nFramegenHudSlot = -1;
	if ( framegen_hud_level() != 0u
		&& !GetBackend()->UsesVulkanSwapchain()
		&& !partial && pPipewireTexture == nullptr && pOutputOverride == nullptr
		&& vulkan_framegen_is_enabled() )
	{
		nFramegenHudSlot = framegen_hud_record(
			cmdBuffer.get(), compositeImage, frameInfo );
	}

	uint64_t sequence = g_device.submit(std::move(cmdBuffer));
	framegen_hud_note_submit( nFramegenHudSlot, sequence );

	// Submitted separately, after the composite: the real frame's present
	// waits on `sequence` only and is never delayed by framegen work.
	// The pPipewireTexture slot is only used by screenshot-style side
	// composites (streaming captures run their own vulkan_screenshot pass),
	// which use screenshot color management and must not enter history.
	if ( !GetBackend()->UsesVulkanSwapchain() && !partial && pPipewireTexture == nullptr && pOutputOverride == nullptr )
		framegen_record_real_frame( pFramegenHistoryImage, frameInfo, sequence, bCursorFreeHistory );

	if ( !GetBackend()->UsesVulkanSwapchain() && pOutputOverride == nullptr && increment )
	{
		// Remember the slot we just composited into before advancing off it; the
		// skip below means it is not always nOutImage-1.
		g_output.nLastOutImage = g_output.nOutImage;

		const uint32_t nRing = g_output.outputImages.size();
		const std::optional<uint32_t> nNext = framegen_find_available_output_image(
			( g_output.nOutImage + 1 ) % nRing );
		// If every slot is busy, leave the current index as a hint. The acquire
		// immediately before the next composite rechecks after backend events
		// have been dispatched and will skip safely if pressure persists.
		if ( nNext )
			g_output.nOutImage = *nNext;
	}

	return sequence;
}

void vulkan_wait( uint64_t ulSeqNo, bool bReset )
{
	return g_device.wait( ulSeqNo, bReset );
}

bool vulkan_has_drm_props()
{
	for (const auto& ext : g_device.supportedExtensions()) {
		if ( strcmp(ext.extensionName, VK_EXT_PHYSICAL_DEVICE_DRM_EXTENSION_NAME) == 0 )
			return true;
	}

	return false;
}

gamescope::Rc<CVulkanTexture> vulkan_get_last_output_image( bool partial, bool defer )
{
	const uint32_t nRing = g_output.outputImages.size();

	// The just-composited image (one step back from the next write slot).
	// For the classic 3-image ring this is (nOutImage + 2) % 3.
	uint32_t nRegularImage = ( g_output.nOutImage + nRing - 1 ) % nRing;

	// The image before that (two steps back), used for deferred/partial reads.
	// For the classic 3-image ring this is (nOutImage + 1) % 3.
	uint32_t nDeferredImage = ( g_output.nOutImage + nRing - 2 ) % nRing;

	uint32_t nOutImage = defer ? nDeferredImage : nRegularImage;

	// Under framegen the ring advance skips slots pinned as history, so the last
	// composited slot is not reliably nOutImage-1; use the explicitly tracked
	// index for the non-deferred base-layer read. Partial overlays (the only
	// deferred consumer) are disabled while framegen is active, so that path is
	// untouched.
	if ( !defer && vulkan_framegen_is_enabled() )
		nOutImage = g_output.nLastOutImage;

	if ( partial )
	{

		//vk_log.infof( "Partial overlay frame: %d", nDeferredImage );
		return g_output.outputImagesPartialOverlay[ nOutImage ];
	}


	return g_output.outputImages[ nOutImage ];
}

bool vulkan_primary_dev_id(dev_t *id)
{
	*id = g_device.primaryDevId();
	return g_device.hasDrmPrimaryDevId();
}

bool vulkan_supports_modifiers(void)
{
	return g_device.supportsModifiers();
}

static void texture_destroy( struct wlr_texture *wlr_texture )
{
	VulkanWlrTexture_t *tex = (VulkanWlrTexture_t *)wlr_texture;
	wlr_buffer_unlock( tex->buf );
	delete tex;
}

static const struct wlr_texture_impl texture_impl = {
	.destroy = texture_destroy,
};

static const struct wlr_drm_format_set *renderer_get_texture_formats( struct wlr_renderer *wlr_renderer, uint32_t buffer_caps )
{
	if (buffer_caps & WLR_BUFFER_CAP_DMABUF)
	{
		return &sampledDRMFormats;
	}
	else if (buffer_caps & WLR_BUFFER_CAP_DATA_PTR)
	{
		return &sampledShmFormats;
	}
	else
	{
		return nullptr;
	}
}

static int renderer_get_drm_fd( struct wlr_renderer *wlr_renderer )
{
	return g_device.drmRenderFd();
}

static struct wlr_texture *renderer_texture_from_buffer( struct wlr_renderer *wlr_renderer, struct wlr_buffer *buf )
{
	VulkanWlrTexture_t *tex = new VulkanWlrTexture_t();
	wlr_texture_init( &tex->base, wlr_renderer, &texture_impl, buf->width, buf->height );
	tex->buf = wlr_buffer_lock( buf );
	// TODO: check format/modifier
	// TODO: if DMA-BUF, try importing it into Vulkan
	return &tex->base;
}

static struct wlr_render_pass *renderer_begin_buffer_pass( struct wlr_renderer *renderer, struct wlr_buffer *buffer, const struct wlr_buffer_pass_options *options )
{
	abort(); // unreachable
}

static const struct wlr_renderer_impl renderer_impl = {
	.get_texture_formats = renderer_get_texture_formats,
	.get_drm_fd = renderer_get_drm_fd,
	.texture_from_buffer = renderer_texture_from_buffer,
	.begin_buffer_pass = renderer_begin_buffer_pass,
};

struct wlr_renderer *vulkan_renderer_create( void )
{
	VulkanRenderer_t *renderer = new VulkanRenderer_t();
	wlr_renderer_init(&renderer->base, &renderer_impl, WLR_BUFFER_CAP_DMABUF | WLR_BUFFER_CAP_DATA_PTR);
	return &renderer->base;
}

gamescope::OwningRc<CVulkanTexture> vulkan_create_texture_from_wlr_buffer( struct wlr_buffer *buf, gamescope::OwningRc<gamescope::IBackendFb> pBackendFb )
{

	struct wlr_dmabuf_attributes dmabuf = {0};
	if ( wlr_buffer_get_dmabuf( buf, &dmabuf ) )
	{
		return vulkan_create_texture_from_dmabuf( &dmabuf, pBackendFb );
	}

	VkResult result;

	void *src;
	uint32_t drmFormat;
	size_t stride;
	if ( !wlr_buffer_begin_data_ptr_access( buf, WLR_BUFFER_DATA_PTR_ACCESS_READ, &src, &drmFormat, &stride ) )
	{
		if ( g_bDebugDualGpuRoute )
			vk_log.errorf( "dual-gpu-route: client wlr_buffer is neither dma-buf nor CPU-readable data pointer" );
		return nullptr;
	}

	uint32_t width = buf->width;
	uint32_t height = buf->height;
	if ( g_bDebugDualGpuRoute )
	{
		vk_log.infof( "dual-gpu-route: client buffer fallback CPU copy %ux%u format 0x%" PRIX32 " stride %zu",
			width,
			height,
			drmFormat,
			stride );
	}

	VkBufferCreateInfo bufferCreateInfo = {
		.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
		.size = stride * height,
		.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
	};
	VkBuffer buffer;
	result = g_device.vk.CreateBuffer( g_device.device(), &bufferCreateInfo, nullptr, &buffer );
	if ( result != VK_SUCCESS )
	{
		wlr_buffer_end_data_ptr_access( buf );
		return nullptr;
	}

	VkMemoryRequirements memRequirements;
	g_device.vk.GetBufferMemoryRequirements(g_device.device(), buffer, &memRequirements);

	uint32_t memTypeIndex =  g_device.findMemoryType(VK_MEMORY_PROPERTY_HOST_COHERENT_BIT|VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT, memRequirements.memoryTypeBits );
	if ( memTypeIndex == ~0u )
	{
		wlr_buffer_end_data_ptr_access( buf );
		return nullptr;
	}

	VkMemoryAllocateInfo allocInfo = {
		.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
		.allocationSize = memRequirements.size,
		.memoryTypeIndex = memTypeIndex,
	};

	VkDeviceMemory bufferMemory;
	result = g_device.vk.AllocateMemory( g_device.device(), &allocInfo, nullptr, &bufferMemory);
	if ( result != VK_SUCCESS )
	{
		wlr_buffer_end_data_ptr_access( buf );
		return nullptr;
	}

	result = g_device.vk.BindBufferMemory( g_device.device(), buffer, bufferMemory, 0 );
	if ( result != VK_SUCCESS )
	{
		wlr_buffer_end_data_ptr_access( buf );
		return nullptr;
	}

	void *dst;
	result = g_device.vk.MapMemory( g_device.device(), bufferMemory, 0, VK_WHOLE_SIZE, 0, &dst );
	if ( result != VK_SUCCESS )
	{
		wlr_buffer_end_data_ptr_access( buf );
		return nullptr;
	}

	memcpy( dst, src, stride * height );

	g_device.vk.UnmapMemory( g_device.device(), bufferMemory );

	wlr_buffer_end_data_ptr_access( buf );

	gamescope::OwningRc<CVulkanTexture> pTex = new CVulkanTexture();
	CVulkanTexture::createFlags texCreateFlags;
	texCreateFlags.bSampled = true;
	texCreateFlags.bTransferDst = true;
	texCreateFlags.bFlippable = true;
	if ( pTex->BInit( width, height, 1u, drmFormat, texCreateFlags, nullptr, 0, 0, nullptr, pBackendFb ) == false )
	{
		if ( g_bDebugDualGpuRoute )
			vk_log.errorf( "dual-gpu-route: client buffer fallback CPU-copy texture creation failed" );
		return nullptr;
	}

	auto cmdBuffer = g_device.commandBuffer();

	cmdBuffer->copyBufferToImage( buffer, 0, stride / DRMFormatGetBPP(drmFormat), pTex);
	// TODO: Sync this copyBufferToImage

	uint64_t sequence = g_device.submit(std::move(cmdBuffer));

	g_device.wait(sequence);

	g_device.vk.DestroyBuffer(g_device.device(), buffer, nullptr);
	g_device.vk.FreeMemory(g_device.device(), bufferMemory, nullptr);

	if ( g_bDebugDualGpuRoute )
	{
		vk_log.infof( "dual-gpu-route: client buffer fallback CPU copy completed" );
	}

	return pTex;
}
