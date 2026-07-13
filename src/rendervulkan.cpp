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
#include "framegen/dispatch_policy.hpp"
#include "framegen/net_layout.hpp"
#include "framegen/net_profile.hpp"
#include "framegen/policy.hpp"
#include "framegen/push_constants.hpp"
#include "framegen/scheduling.hpp"
#include "framegen/settings.hpp"
#include "framegen/temporal.hpp"

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
	{ DRM_FORMAT_RGB565, VK_FORMAT_R5G6B5_UNORM_PACK16, VK_FORMAT_R5G6B5_UNORM_PACK16, 1, false, false },
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
	vk_log.infof( "selecting physical device '%s': queue family %x (general queue family %x)", props.deviceName, m_queueFamily, m_generalQueueFamily );

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
	for (const auto& ext : m_supportedExts) {
		if ( strcmp(ext.extensionName, VK_EXT_IMAGE_DRM_FORMAT_MODIFIER_EXTENSION_NAME) == 0 )
			m_bSupportsModifiers = true;

		if ( strcmp(ext.extensionName, VK_EXT_QUEUE_FAMILY_FOREIGN_EXTENSION_NAME) == 0 )
			supportsForeignQueue = true;

		if ( strcmp(ext.extensionName, VK_EXT_HDR_METADATA_EXTENSION_NAME) == 0 )
			supportsHDRMetadata = true;
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
		VkPhysicalDeviceFeatures2 features2 = {
			.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
			.pNext = &vulkan12Features,
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
		&& m_queueCount >= 2 && !env_to_bool( getenv( "GAMESCOPE_FRAMEGEN_SINGLE_QUEUE" ) );
	const uint32_t nComputeQueues = bWantFramegenQueue ? 2u : 1u;

	VkDeviceQueueCreateInfo queueCreateInfos[2] =
	{
		{
			.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
			.pNext = gamescope::Process::HasCapSysNice() ? &queueCreateInfoEXT : nullptr,
			.queueFamilyIndex = m_queueFamily,
			.queueCount = nComputeQueues,
			.pQueuePriorities = queuePriorities
		},
		{
			.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
			.pNext = gamescope::Process::HasCapSysNice() ? &queueCreateInfoEXT : nullptr,
			.queueFamilyIndex = m_generalQueueFamily,
			.queueCount = 1,
			.pQueuePriorities = queuePriorities
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

	VkDeviceCreateInfo deviceCreateInfo = {
		.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
		.pNext = &features2,
		.queueCreateInfoCount = m_queueFamily == m_generalQueueFamily ? 1u : 2u,
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

	VkResult res = vk.CreateDevice(physDev(), &deviceCreateInfo, nullptr, &m_device);
	if ( res == VK_ERROR_NOT_PERMITTED_KHR && gamescope::Process::HasCapSysNice() )
	{
		fprintf(stderr, "vkCreateDevice failed with a high-priority queue (general + compute). Falling back to regular priority (general).\n");
		queueCreateInfos[1].pNext = nullptr;
		res = vk.CreateDevice(physDev(), &deviceCreateInfo, nullptr, &m_device);


		if ( res == VK_ERROR_NOT_PERMITTED_KHR && gamescope::Process::HasCapSysNice() )
		{
			fprintf(stderr, "vkCreateDevice failed with a high-priority queue (compute). Falling back to regular priority (all).\n");
			queueCreateInfos[0].pNext = nullptr;
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
		vk.GetDeviceQueue( device(), m_queueFamily, 1, &m_framegenQueue );
		m_bHasFramegenQueue = m_framegenQueue != VK_NULL_HANDLE;
		if ( m_bHasFramegenQueue )
			vk_log.infof( "frame generation: using dedicated compute queue (family %u, index 1)", m_queueFamily );
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
		.size = upload_buffer_size,
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
	// runs without measurement and the ladder stays at full quality.
	{
		uint32_t nQueueFamilyCount = 0;
		vk.GetPhysicalDeviceQueueFamilyProperties( physDev(), &nQueueFamilyCount, nullptr );
		std::vector<VkQueueFamilyProperties> queueFamilyProps( nQueueFamilyCount );
		vk.GetPhysicalDeviceQueueFamilyProperties( physDev(), &nQueueFamilyCount, queueFamilyProps.data() );

		VkPhysicalDeviceProperties physProps = {};
		vk.GetPhysicalDeviceProperties( physDev(), &physProps );

		const bool bTimestampsUsable = m_queueFamily < nQueueFamilyCount
			&& queueFamilyProps[ m_queueFamily ].timestampValidBits > 0
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
				m_uFramegenTimestampValidBits = queueFamilyProps[ m_queueFamily ].timestampValidBits;
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
		SHADER_TYPE_FRAMEGEN_MOTION_LUMA_PAIR,
		SHADER_TYPE_FRAMEGEN_MOTION_LUMA_PAIR_RGBA,
		SHADER_TYPE_FRAMEGEN_MOTION_PYRAMID,
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
		SHADER_TYPE_FRAMEGEN_MOTION_NET,
		SHADER_TYPE_FRAMEGEN_MOTION_NET_TRAIN,
		SHADER_TYPE_FRAMEGEN_MOTION_NET_OPT,
	};

	for ( ShaderType type : pipelines )
		pipeline( type );

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

std::unique_ptr<CVulkanCmdBuffer> CVulkanDevice::commandBuffer()
{
	std::unique_ptr<CVulkanCmdBuffer> cmdBuffer;
	if (m_unusedCmdBufs.empty())
	{
		VkCommandBuffer rawCmdBuffer;
		VkCommandBufferAllocateInfo commandBufferAllocateInfo = {
			.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
			.commandPool = m_commandPool,
			.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
			.commandBufferCount = 1
		};

		VkResult res = vk.AllocateCommandBuffers( device(), &commandBufferAllocateInfo, &rawCmdBuffer );
		if ( res != VK_SUCCESS )
		{
			vk_errorf( res, "vkAllocateCommandBuffers failed" );
			return nullptr;
		}

		cmdBuffer = std::make_unique<CVulkanCmdBuffer>(this, rawCmdBuffer, queue(), queueFamily());
	}
	else
	{
		cmdBuffer = std::move(m_unusedCmdBufs.back());
		m_unusedCmdBufs.pop_back();
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
		pWaitStageFlags[i] = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
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
	for (auto src : m_boundTextures)
	{
		if (src)
			prepareSrcImage(src);
	}
	assert(m_target != nullptr);
	prepareDestImage(m_target);
	if (m_target2)
		prepareDestImage(m_target2);
	insertBarrier();

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
	insertBarrier();

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
	insertBarrier();
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

		if (image->queueFamily == VK_QUEUE_FAMILY_IGNORED)
			image->queueFamily = m_queueFamily;

		VkImageMemoryBarrier memoryBarrier =
		{
			.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
			.srcAccessMask = state.dirty ? write_bits : 0u,
			.dstAccessMask = flush ? 0u : read_bits | write_bits,
			.oldLayout = state.discarded ? VK_IMAGE_LAYOUT_UNDEFINED : VK_IMAGE_LAYOUT_GENERAL,
			.newLayout = isPresent ? GetBackend()->GetPresentLayout() : VK_IMAGE_LAYOUT_GENERAL,
			.srcQueueFamilyIndex = isExport ? image->queueFamily : state.needsImport ? externalQueue : image->queueFamily,
			.dstQueueFamilyIndex = isExport ? externalQueue : state.needsImport ? m_queueFamily : m_queueFamily,
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

bool CVulkanTexture::BInit( uint32_t width, uint32_t height, uint32_t depth, uint32_t drmFormat, createFlags flags, wlr_dmabuf_attributes *pDMA /* = nullptr */,  uint32_t contentWidth /* = 0 */, uint32_t contentHeight /* =  0 */, CVulkanTexture *pExistingImageToReuseMemory, gamescope::OwningRc<gamescope::IBackendFb> pBackendFb )
{
	m_pBackendFb = std::move( pBackendFb );
	m_drmFormat = drmFormat;
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
		.sharingMode = VK_SHARING_MODE_EXCLUSIVE,
	};

	assert( imageInfo.format != VK_FORMAT_UNDEFINED );

	if ( g_bDebugDualGpuRoute && pDMA )
	{
		vk_log.infof( "dual-gpu-route: client dma-buf Vulkan import request %dx%d format 0x%" PRIX32 " modifier 0x%" PRIX64 " planes %d usage 0x%x sampled %s storage %s transfer-dst %s",
			pDMA->width,
			pDMA->height,
			pDMA->format,
			pDMA->modifier,
			pDMA->n_planes,
			usage,
			flags.bSampled ? "yes" : "no",
			flags.bStorage ? "yes" : "no",
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
	}
	else if ( g_bDebugDualGpuRoute && pDMA && pDMA->modifier != DRM_FORMAT_MOD_INVALID )
	{
		vk_log.infof( "dual-gpu-route: client dma-buf has modifier 0x%" PRIX64 " but compositor Vulkan modifier support is %s",
			pDMA->modifier,
			g_device.supportsModifiers() ? "enabled" : "disabled" );
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
		VkImportMemoryFdInfoKHR importMemoryInfo = {};
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
			assert( allDMABUFsEqual( pDMA ) );

			// Importing memory from a FD transfers ownership of the FD
			int fd = dup( pDMA->fd[0] );
			if ( fd < 0 )
			{
				vk_log.errorf_errno( "dup failed" );
				return false;
			}

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
			return false;
		}

		m_vkImageMemory = memoryHandle;
	}
	else
	{
		vk_log.infof("%d vs %d!", (int)pExistingImageToReuseMemory->m_size, (int)m_size);
		assert(pExistingImageToReuseMemory->m_size >= m_size);

		memoryHandle = pExistingImageToReuseMemory->m_vkImageMemory;
		m_vkImageMemory = VK_NULL_HANDLE;
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
// prediction history (zero-copy), so it needs a deeper ring: history(2) +
// the frame being scanned out + the next composite target, without ever
// recompositing a slot still referenced as history. Generated frames never
// use this ring (they have their own g_output.framegenOutputImages pool), so
// 5 is sufficient for any x2..x4 multiplier.
static constexpr uint32_t k_uOutputRingSizeDefault = 3;
static constexpr uint32_t k_uOutputRingSizeFramegen = 5;

static bool vulkan_make_output_images( VulkanOutput_t *pOutput )
{
	CVulkanTexture::createFlags outputImageflags;
	outputImageflags.bFlippable = true;
	outputImageflags.bStorage = true;
	outputImageflags.bTransferSrc = true; // for screenshots
	outputImageflags.bSampled = true; // for pipewire blits
	outputImageflags.bOutputImage = true;

	const uint32_t nRing = vulkan_framegen_is_enabled() ? k_uOutputRingSizeFramegen : k_uOutputRingSizeDefault;

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

	//fprintf(stderr, "pDMA->width: %d pDMA->height: %d pDMA->format: 0x%x pDMA->modifier: 0x%lx pDMA->n_planes: %d\n",
	//	pDMA->width, pDMA->height, pDMA->format, pDMA->modifier, pDMA->n_planes);
	
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
	// Ultra-quality causal acceleration: a retained copy of the preceding
	// interval's final checked forward field. It is reprojected through the
	// current field before differencing, so acceleration compares the same
	// moving content rather than the same screen coordinate. The frame ID gate
	// rejects stale history when shared-queue admission skipped an interval.
	gamescope::OwningRc<CVulkanTexture> mvFieldHistory;
	uint64_t uMotionHistoryFrameId = 0;
	uint64_t uMotionHistoryIntervalNs = 0;
	// The finalized field for the current real-frame pair remains resident in
	// mvField/mvFieldNet. JIT and idle refills reuse it instead of estimating,
	// refining, probing and training on the same pair again. On the next
	// consecutive pair it is copied to mvFieldHistory before preparation
	// overwrites the working images.
	uint64_t uMotionFieldFrameId = 0;
	uint64_t uMotionFieldIntervalNs = 0;
	GamescopeFramegenQuality eMotionFieldQuality = GamescopeFramegenQuality::Low;
	bool bMotionFieldBidir = false;
	// Extreme-quality frames-only disocclusion evidence. This low-resolution
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
	// color change. Kept at field resolution and sampled only by Extreme's
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

struct FramegenHistory_t
{
	// The last two real frames, held as references straight into the
	// g_output.outputImages ring. Zero-copy: the composite already wrote these
	// images, so framegen never copies them — it keeps them alive and samples
	// them. Safe because the ring (k_uOutputRingSizeFramegen) is deep enough
	// that neither slot is recomposited before the next generation reads it.
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
		float phase = 0.0f; // fraction of the real-frame interval, for logs
		// Bidir: this entry is a REAL frame riding the queue behind its
		// interpolations. Its composite completed at its own paint (seqNo 0 on
		// the framegen timeline = always ready) and it must never be dropped
		// by the generated-frame discard paths.
		bool bReal = false;
	};
	std::vector<PendingGenerated_t> pending;

	uint32_t width = 0;
	uint32_t height = 0;
	uint32_t drmFormat = DRM_FORMAT_INVALID;
	uint64_t previousFrameId = 0;
	uint64_t currentFrameId = 0;
	uint64_t previousPresentTimeNs = 0;
	uint64_t currentPresentTimeNs = 0;
	// Display-clock pacing (#06 JIT phase). ulFrametimeEmaNs is a slew-limited
	// EMA of the real-frame interval: composite times are quantized by the
	// vblank wakes, so a fractional-rate game (45 fps on 60 Hz) yields
	// alternating 1- and 2-vblank samples — the true frametime exists only as
	// their average, never in any single sample. ulCurrentRealVblankNs anchors
	// the current real frame on the display clock: the vblank it scans out at,
	// taken from the vblank timer (KMS pageflip feedback), so a JIT slot's
	// phase is a display-time measurement rather than a gap-count guess.
	uint64_t ulFrametimeEmaNs = 0;
	uint64_t ulCurrentRealVblankNs = 0;
	uint64_t lastCompositeSeqNo = 0;
	// Latest submission on the framegen execution path, including descriptor-free
	// base-history copies. Reset paths wait for this token before releasing pools
	// or history; lastGeneratedSeqNo remains generation-only for the headroom gate.
	uint64_t lastFramegenWorkSeqNo = 0;
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
	// each step sheds work (motion quality tiers, then extrapolate, then a
	// multiplier notch). Never
	// reaches "stop generating" — that is left to the reactive pacing gate below.
	// Monotonic within a scene (only ever increases); reset to 0 on a scene change.
	// nDegradeHold is a post-step cooldown so the new rung's cost folds into the
	// measurement before the next step decision.
	uint32_t nDegradeSteps = 0;
	uint32_t nDegradeHold = 0;
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
static bool g_bLoggedFramegenConfig = false;
static constexpr uint64_t k_ulFramegenMaxRealFrameGapNs = 250'000'000ull;
// Dedicated-queue idle refill may extrapolate beyond the originally expected
// next real frame if the game stalls. Cap the forward step so a long stall does
// not run prediction unbounded; after this point additional refills converge
// toward the capped prediction instead of accelerating away from real content.
static constexpr float k_flFramegenMaxForwardStrength = 1.5f;

static bool framegen_refill_idle();
static bool framegen_jit_submit( uint64_t ulCompositeSeqNo, uint32_t nMaxDegradeSteps );
static bool framegen_vrr_hybrid_submit( uint64_t ulCompositeSeqNo, uint32_t nMaxDegradeSteps );
static bool framegen_format_supports_sampled_storage( uint32_t drmFormat );
static gamescope::Rc<CVulkanTexture> framegen_base_present_composite( gamescope::Rc<CVulkanTexture> pGeneratedBase, const struct FrameInfo_t *pPresentFrameInfo );

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

// JIT phase (#06): plan one slot at a time against the display clock instead of
// baking a k/gap batch from a single-interval guess. Requires the dedicated
// framegen queue — the per-vblank submit cadence must never sit in front of a
// composite on the realtime queue. Opt-in prototype toggle.
static bool framegen_jit_enabled()
{
	static const bool s_bEnabled = env_to_bool( getenv( "GAMESCOPE_FRAMEGEN_JIT" ) );
	return s_bEnabled && g_device.hasFramegenQueue();
}

// VRR hybrid (#01): instead of suppressing adaptive sync, present real frames
// immediately (full VRR latency win) and show the one generated frame via a
// timer-armed mid-interval flip. Requires the dedicated framegen queue for the
// same reason JIT does. "Requested" gates the allowVRR decision in steamcompmgr
// (VRR can only BECOME active if real flips carry allowVRR=true); "active"
// additionally requires the connector to actually be in VRR right now — when it
// is not (nested, panel without VRR, adaptive sync toggled off), every decision
// point falls back to the fixed-refresh paths, live, with no restart.
bool vulkan_framegen_vrr_hybrid_requested()
{
	static const bool s_bEnabled = env_to_bool( getenv( "GAMESCOPE_FRAMEGEN_VRR_HYBRID" ) );
	return s_bEnabled && g_device.hasFramegenQueue();
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
// prototypes (#01 VRR hybrid, #06 JIT) and the base-layer path (#02) keep
// their own timelines and are mutually exclusive with it.
static bool framegen_bidir_enabled()
{
	static const bool s_bEnabled = env_to_bool( getenv( "GAMESCOPE_FRAMEGEN_BIDIR" ) );
	return s_bEnabled;
}

// Experimental low-latency cadence compromise. The accepted bidir baseline
// places generated phases on the measured-gap grid (k/gap), which keeps warp
// distance and endpoint latency low but leaves a larger final jump whenever
// the source gap exceeds the multiplier. A value in [0,1] moves those phases
// part-way toward the uniform multiplier grid without changing queue drain or
// flip timing. Zero is the exact established behavior; keep it the default
// until live A/B proves a non-zero compromise improves both motion and image.
static float framegen_bidir_phase_bias()
{
	static const float s_flBias = []()
	{
		const auto value = gamescope::framegen::parse_finite_float_setting(
			getenv( "GAMESCOPE_FRAMEGEN_BIDIR_PHASE_BIAS" ) );
		return value.has_value() ? std::clamp( *value, 0.0f, 1.0f ) : 0.0f;
	}();
	return s_flBias;
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

// Extreme-only intermediate-grid correction. Endpoint motion fields are
// defined on their respective real-frame grids, so sampling them directly at
// an intermediate output coordinate is only the first fixed-point iterate.
// This opt-in strength selects a separately specialized, symmetric,
// closure-checked warp. Lower tiers retain the established pipeline.
static float framegen_bidir_endpoint_trace_strength( GamescopeFramegenQuality eQuality )
{
	if ( eQuality != GamescopeFramegenQuality::Extreme )
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
		&& !framegen_jit_enabled()
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

// Deadline-driven degradation ladder (#04). The startup mode/quality/multiplier are the
// quality CEILING; under GPU-time pressure the ladder sheds work one rung at a
// time. It is monotonic within a scene (only ever degrades); quality is re-probed
// from this ceiling only on a scene change (invalidate_history). It NEVER mutates
// the base globals (g_eFramegenMode / g_nFramegenMultiplier): those gate the whole
// feature (vulkan_framegen_is_enabled) and size the output pool, so the ladder only
// ever degrades within them and reads its result through framegen_effective_config.
using FramegenEffective_t = gamescope::framegen::EffectiveConfig;

// Total number of rungs available from the startup config: motion walks through
// every lower quality tier and then to extrapolate, followed by one rung per
// multiplier notch down to x2. Deliberately does NOT include a "stop generating" rung: the
// ladder always keeps generating (at worst the cheapest config), so its GPU-time
// input never starves. The genuine "even the cheapest config overruns" case is
// left to the existing reactive pacing gate (bGpuHasHeadroom + nStableFrames),
// which is driven by live per-frame signals and so self-heals, unlike a
// measurement-frozen ladder rung would.
static uint32_t framegen_max_degrade_steps()
{
	return gamescope::framegen::max_degrade_steps(
		g_eFramegenMode, g_eFramegenQuality, g_nFramegenMultiplier );
}

// Apply nDegradeSteps degradations to the startup ceiling. Motion (the priciest
// pass) is shed first, then the multiplier is stepped down to x2. nDegradeSteps
// is always clamped to framegen_max_degrade_steps(), so n is fully consumed and
// the result is always a still-generating config (never dormant).
static FramegenEffective_t framegen_effective_config( uint32_t nDegradeSteps )
{
	return gamescope::framegen::effective_config(
		g_eFramegenMode, g_eFramegenQuality, g_nFramegenMultiplier, nDegradeSteps );
}

static bool framegen_output_matches( const gamescope::OwningRc<CVulkanTexture> &pTexture, uint32_t width, uint32_t height, uint32_t drmFormat )
{
	return pTexture != nullptr
		&& pTexture->width() == width
		&& pTexture->height() == height
		&& pTexture->drmFormat() == drmFormat;
}

void vulkan_framegen_invalidate_history( const char *reason )
{
	if ( !g_framegenHistory.valid && g_framegenHistory.pending.empty()
		&& g_framegenHistory.previousReal == nullptr && g_framegenHistory.currentReal == nullptr )
		return;

	if ( g_bFramegenDebug )
		vk_log.infof( "framegen: history valid=false reason=%s", reason ? reason : "unknown" );

	g_framegenHistory.valid = false;
	// Bidir (B3): queued REAL frames are actual content the user has not seen
	// yet — dropping one at a scene change would skip a painted frame (a
	// visible hitch), so only the interpolations (now-stale predictions) go.
	// Their ring slots stay protected: the composite ring advance pins slots
	// referenced by pending real entries. Everything else clears as before.
	if ( vulkan_framegen_bidir_active() )
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
	// The cadence estimator re-seeds from the first post-prime interval: a
	// scene change (focus swap, long stall) is exactly when the old cadence
	// may be stale, and re-seeding costs only the estimator's ~8-frame settle.
	g_framegenHistory.ulFrametimeEmaNs = 0;
	g_framegenHistory.ulCurrentRealVblankNs = 0;
	g_framegenHistory.lastCompositeSeqNo = 0;
	g_framegenHistory.nLastGeneratedSlot = 0;
	g_framegenHistory.nLastGenerationGapVblanks = 0;
	// Re-probe quality from the top on a fresh scene: the ladder is monotonic
	// within a scene, so a scene change is the only place quality is restored.
	g_framegenHistory.nDegradeSteps = 0;
	g_framegenHistory.nDegradeHold = 0;
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
	// Release the retained output-ring slots so a ring rebuild is never blocked
	// by history holding a reference to an old image, and so the next real frame
	// re-primes cleanly.
	g_framegenHistory.previousReal = nullptr;
	g_framegenHistory.currentReal = nullptr;
}

static __attribute__((noinline)) void framegen_net_profile_consume();
static void framegen_net_profile_flush();

void vulkan_framegen_reset( const char *reason )
{
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
	g_device.framegenGarbageCollect();

	// C2: persist unsaved learning before the state textures go away. The latest
	// health-checked served weights remain cached process-wide and seed the new
	// state, so resize/format resets do not throw away in-session adaptation.
	framegen_net_profile_consume();
	framegen_net_profile_flush();
	framegen_color_probe_consume();

	g_framegenHistory = {};
	// The per-rung costs live on g_device and survive the history reset; forget
	// them too (as invalidate_history does) so the monotonic ladder re-probes the
	// new workload from full quality instead of stepping on the old scene's stale
	// over-deadline measurements.
	g_device.framegenResetRungCosts();
	g_output.framegenOutputImages.clear();
	g_output.framegenPresentImages.clear();
	g_framegenMotion = {};
	g_framegenColorProbe = {};
}

bool vulkan_framegen_has_pending_generated_frame()
{
	return vulkan_framegen_is_enabled() && !g_framegenHistory.pending.empty();
}

// VRR hybrid (#01): how long after the real frame's scanout the pending
// generated frame should flip. steamcompmgr arms an absolute CLOCK_MONOTONIC
// timer for (KMS flip timestamp of the real frame) + this offset — the same
// clock the pageflip events are stamped in, so the spacing is measured on
// display ground truth end to end. 0 = nothing to schedule.
uint64_t vulkan_framegen_vrr_hybrid_mid_offset_ns()
{
	if ( !vulkan_framegen_vrr_hybrid_active()
		|| g_framegenHistory.pending.empty()
		|| g_framegenHistory.ulFrametimeEmaNs == 0 )
		return 0;

	const float flPhase = g_framegenHistory.pending.front().phase;
	if ( flPhase <= 0.0f )
		return 0;

	return (uint64_t)( (double)g_framegenHistory.ulFrametimeEmaNs * (double)flPhase );
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
		static uint64_t s_uTooSlowDebugLogCounter = 0;
		if ( FramegenDebugShouldLog( s_uTooSlowDebugLogCounter ) )
			vk_log.infof( "framegen: discarded generated frame id=%" PRIu64 ".%02u reason=generation_too_slow",
				front.frameId, (unsigned)( front.phase * 100.0f ) );
		return nullptr;
	}

	g_framegenHistory.pending.erase( g_framegenHistory.pending.begin() );

	// Instant (completion checked above). On the shared-queue fallback this also
	// lets the device recycle the upload arena at a known-idle point; on the
	// dedicated queue framegen uses push constants and its own timeline, so this
	// is a cheap already-signalled wait.
	g_device.waitFramegen( front.seqNo );

	// Base-layer mode (#02): the pending slot holds a pre-upscale BASE frame;
	// composite it through the real pipeline with the live layer stack (fresh
	// overlays, latest cursor) before it can be flipped. Output-space mode returns
	// the scanout-ready generated output directly.
	gamescope::Rc<CVulkanTexture> pResult = front.tex;
	if ( g_framegenHistory.bBaseLayer )
		pResult = framegen_base_present_composite( front.tex, pPresentFrameInfo );

	if ( pResult != nullptr )
	{
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
		}
		else if ( framegen_jit_enabled() )
		{
			// Top up one slot ahead. JIT re-measures the display clock per
			// slot; the classic path continues the interval's k/gap ladder.
			framegen_jit_submit( g_framegenHistory.lastCompositeSeqNo, framegen_max_degrade_steps() );
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

	const size_t nDiscarded = nBefore - g_framegenHistory.pending.size();
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
	FramegenHistory_t::PendingGenerated_t front = g_framegenHistory.pending.front();
	if ( front.bReal || g_device.hasCompletedFramegen( front.seqNo ) )
	{
		g_framegenHistory.pending.erase( g_framegenHistory.pending.begin() );
		g_device.waitFramegen( front.seqNo );
		static uint64_t s_uFlipDebugLogCounter = 0;
		if ( FramegenDebugShouldLog( s_uFlipDebugLogCounter ) )
			vk_log.infof( "framegen: presented %s frame id=%" PRIu64 ".%02u (bidir flip substitution)",
				front.bReal ? "delayed real" : "generated", front.frameId, (unsigned)( front.phase * 100.0f ) );
		return front.tex;
	}

	if ( g_framegenHistory.previousReal != nullptr )
		return g_framegenHistory.previousReal;
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
		if ( g_framegenHistory.pending.empty() )
			return pComposite;
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
	createFlags.bOutputImage = true;

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

// #02 late overlay composite: turn a generated BASE frame into a scanout-ready
// output image by running it through the same composite pipeline a real frame
// uses — FSR EASU/RCAS (or NIS/blit), shaper + 3D LUTs, and every CURRENT
// overlay layer blended on top, cursor at its latest position. Overlays are
// therefore pixel-perfect on generated frames: never extrapolated, always
// re-composited fresh. Runs on the realtime queue at present time like any
// composite; the vblank pacing already budgets for compositing
// (UpdateWasCompositing), and pOutputOverride both keeps this composite out of
// framegen_record_real_frame (no history poisoning) and off the output ring.
static gamescope::Rc<CVulkanTexture> framegen_base_present_composite( gamescope::Rc<CVulkanTexture> pGeneratedBase, const struct FrameInfo_t *pPresentFrameInfo )
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
	gamescope::Rc<CVulkanTexture> pTarget;
	for ( size_t nProbe = 0; nProbe < g_output.framegenPresentImages.size(); nProbe++ )
	{
		const uint32_t idx = g_framegenHistory.nNextPresentIndex % (uint32_t)g_output.framegenPresentImages.size();
		g_framegenHistory.nNextPresentIndex++;
		CVulkanTexture *pCandidate = g_output.framegenPresentImages[ idx ].get();
		if ( pCandidate != nullptr && !pCandidate->IsInUse() )
		{
			pTarget = pCandidate;
			break;
		}
	}
	if ( pTarget == nullptr )
	{
		static uint64_t s_uPresentPressureDebugLogCounter = 0;
		if ( FramegenDebugShouldLog( s_uPresentPressureDebugLogCounter ) )
			vk_log.infof( "framegen: late-composite pool pressure pool=%zu", g_output.framegenPresentImages.size() );
		return nullptr;
	}

	FrameInfo_t generatedFrameInfo = *pPresentFrameInfo;
	generatedFrameInfo.layers[ 0 ].tex = pGeneratedBase;

	std::optional<uint64_t> oSeqNo = vulkan_composite( &generatedFrameInfo, nullptr, false, pTarget, false );
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

	auto pCmdBuffer = g_device.commandBuffer();
	pCmdBuffer->markFramegen();
	pCmdBuffer->copyImage( std::move( pBaseFrame ), pTarget );
	// The real composite carries the client's acquire dependency. Waiting for
	// its timeline point makes that readiness chain visible to this queue before
	// it reads the same client image, and also orders the composite's image-state
	// transitions ahead of the copy.
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
	// fp32 for them (see the fp16 shader's precision note). 16-bit UNORM
	// targets need the same treatment for a different reason: fp16's 11-bit
	// mantissa cannot represent 16-bit-deep content, so the fp16 path would
	// band it. Unreachable in output-space mode (scanout formats are
	// 8/10-bit), but base-layer mode (#02) generates in the CLIENT's format,
	// and 16-bit UNORM swapchains do occur (e.g. the NVIDIA WSI path here).
	switch ( drmFormat )
	{
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
// the current behavior. The fourth head never predicts pixels; Extreme uses it
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

// Endpoint reconstruction has an aperture-problem null space: several motion
// vectors can sample the same next-frame color while tracing very different
// paths through bidirectional intermediate phases. Keep learned bidir geometry
// behind an explicit attribution switch until a true intermediate-time target
// exists; confidence-only service/training is the safe default.
static bool framegen_net_bidir_flow_enabled()
{
	static const bool s_bEnabled = env_to_bool( getenv( "GAMESCOPE_FRAMEGEN_NET_BIDIR_FLOW" ) );
	return s_bEnabled;
}

static bool framegen_net_bidir_conservative()
{
	return vulkan_framegen_bidir_active() && !framegen_net_bidir_flow_enabled();
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

static bool framegen_net_requested( GamescopeFramegenQuality eQuality )
{
	return eQuality >= GamescopeFramegenQuality::High
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

// The field the warps and the quality probe consume: the net's refined copy
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
	// Ultra and Extreme also rotate the final field into a retained transfer destination.
	createFlags.bTransferSrc = framegen_record_dir() != nullptr
		|| g_eFramegenQuality >= GamescopeFramegenQuality::Ultra;
	createFlags.bTransferDst = g_eFramegenQuality >= GamescopeFramegenQuality::Ultra;

	*ppTexture = new CVulkanTexture();
	return ( *ppTexture )->BInit( width, height, 1u, drmFormat, createFlags );
}

static bool framegen_create_luma_reservoir( gamescope::OwningRc<CVulkanTexture> *ppTexture,
	uint32_t width, uint32_t height, uint32_t drmFormat )
{
	CVulkanTexture::createFlags createFlags;
	createFlags.bSampled = true;
	createFlags.bTransferDst = true;

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
	const bool bR16FLumaSupported = framegen_format_supports_sampled_storage( DRM_FORMAT_R16F );
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
static bool framegen_fbcheck_enabled( GamescopeFramegenQuality eQuality )
{
	if ( eQuality < GamescopeFramegenQuality::Medium )
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
static bool framegen_agreement_enabled( GamescopeFramegenQuality eQuality )
{
	if ( eQuality < GamescopeFramegenQuality::Medium )
		return false;

	static const bool s_bEnabled = []()
	{
		const char *pszEnv = getenv( "GAMESCOPE_FRAMEGEN_AGREE" );
		return pszEnv == nullptr || env_to_bool( pszEnv );
	}();
	return s_bEnabled;
}

// Extreme-only three-frame disocclusion evidence. Default on as part of the
// Extreme quality contract; the environment switch exists for live A/B cost
// and artifact attribution without changing any lower quality rung.
static bool framegen_reservoir_enabled( GamescopeFramegenQuality eQuality )
{
	if ( eQuality != GamescopeFramegenQuality::Extreme )
		return false;

	static const bool s_bEnabled = []()
	{
		const char *pszEnv = getenv( "GAMESCOPE_FRAMEGEN_RESERVOIR" );
		return pszEnv == nullptr || env_to_bool( pszEnv );
	}();
	return s_bEnabled;
}

// Extreme-only learned non-geometric color trend. The switch isolates the
// final correction for live A/B while leaving the field net and every queue /
// descriptor / timing decision identical.
static bool framegen_shading_enabled( GamescopeFramegenQuality eQuality )
{
	if ( eQuality != GamescopeFramegenQuality::Extreme )
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
//   quality-driven degradation #04's monotonic ladder must not provide (a
//   discrete quality rung would either oscillate or, held monotonic, let one
//   bad interval degrade the whole scene).
//   CPU, next batch: the readback auto-calibrates the FB tolerance and the
//   agreement window to the content (see framegen_adapt_consume).
// Default on in motion mode; GAMESCOPE_FRAMEGEN_ADAPT=0 restores B3 behavior
// bit-exactly for A/B attribution.
static bool framegen_adapt_enabled( GamescopeFramegenQuality eQuality )
{
	if ( eQuality < GamescopeFramegenQuality::High )
		return false;

	static const bool s_bEnabled = []()
	{
		const char *pszEnv = getenv( "GAMESCOPE_FRAMEGEN_ADAPT" );
		return pszEnv == nullptr || env_to_bool( pszEnv );
	}();
	return s_bEnabled;
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

static float framegen_effective_fbcheck_tol( GamescopeFramegenQuality eQuality )
{
	return framegen_adapt_enabled( eQuality )
		? framegen_adapt_fbcheck_tol() : framegen_fbcheck_tol_base();
}

static float framegen_effective_agree_lo( GamescopeFramegenQuality eQuality )
{
	return framegen_adapt_enabled( eQuality )
		? framegen_adapt_agree_lo() : k_flFramegenAgreeLo;
}

static float framegen_effective_agree_hi( GamescopeFramegenQuality eQuality )
{
	return framegen_adapt_enabled( eQuality )
		? framegen_adapt_agree_hi() : k_flFramegenAgreeHi;
}

// Parse the completed batch's stats readback and fold it into the adaptation
// EMAs, then derive the threshold values the next batch records with. Called
// at batch-planning time: the same hasCompletedFramegen() gate that admits a
// new batch guarantees the mapped memory is no longer being written.
static void framegen_adapt_consume( GamescopeFramegenQuality eQuality )
{
	if ( !framegen_adapt_enabled( eQuality ) )
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

	if ( measurement->sceneCut != 0u )
	{
		vk_log.infof( "framegen: content scene cut detected (%u/9 sections, histogram distance %.2f); presenting a real endpoint",
			measurement->changedSections,
			gamescope::framegen::scene_histogram_distance( *measurement ) );
	}

	// Slow EMA (1/8): threshold moves must be calmer than the per-frame signal,
	// or the adaptation itself becomes a flicker source. FB tolerance loosens
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
		vk_log.infof( "framegen: adapt resid=%.3f bad=%.1f%% killed=%.1f%% noise=%.4f fbP75=%.2f mv=%.1f scene=%u(%u/9 hist=%.2f) -> fbTol=%.2f agree=%.2f/%.2f",
			measurement->residual, measurement->badFraction * 100.0f,
			measurement->killedFraction * 100.0f,
			h.adaptation.noiseEma, h.adaptation.fbP75Ema, measurement->motionMean,
			measurement->sceneCut, measurement->changedSections,
			gamescope::framegen::scene_histogram_distance( *measurement ),
			framegen_adapt_fbcheck_tol(), framegen_adapt_agree_lo(), framegen_adapt_agree_hi() );
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
// full-res warps inherit the quality verdict through the field fetch they
// already do, at zero added per-pixel cost. The copy at the end lands the raw
// counters in the host-mapped readback for the CPU-side threshold
// calibration. The command buffer's barrier tracking orders all of it
// (accumulate -> apply is a WAR on the field, covered by the execution
// dependency of the stats image's own barrier). Field-res work plus a 384-byte
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
	const bool bConservativeBidir = framegen_net_bidir_conservative();
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

static bool framegen_motion_history_valid( GamescopeFramegenQuality eQuality )
{
	const uint64_t ulCurrentIntervalNs = gamescope::framegen::present_interval_ns(
		g_framegenHistory.currentPresentTimeNs, g_framegenHistory.previousPresentTimeNs );
	const uint64_t ulHistoryIntervalNs = g_framegenMotion.uMotionHistoryIntervalNs;
	// A violent cadence transition is a poor second-derivative sample even after
	// time normalization. Fall back to constant velocity for that one interval;
	// the newly stored field immediately re-primes the next batch.
	const bool bComparableIntervals = gamescope::framegen::motion_intervals_comparable(
		ulCurrentIntervalNs, ulHistoryIntervalNs );
	return eQuality >= GamescopeFramegenQuality::Ultra
		&& g_framegenMotion.mvFieldHistory != nullptr
		&& g_framegenMotion.uMotionHistoryFrameId != 0
		&& g_framegenMotion.uMotionHistoryFrameId + 1u == g_framegenHistory.currentFrameId
		&& bComparableIntervals;
}

static float framegen_motion_history_time_scale()
{
	if ( !framegen_motion_history_valid( GamescopeFramegenQuality::Ultra ) )
		return 1.0f;
	const uint64_t ulCurrentIntervalNs = g_framegenHistory.currentPresentTimeNs - g_framegenHistory.previousPresentTimeNs;
	return gamescope::framegen::motion_history_time_scale(
		ulCurrentIntervalNs, g_framegenMotion.uMotionHistoryIntervalNs );
}

static float framegen_motion_accel_time_factor()
{
	if ( !framegen_motion_history_valid( GamescopeFramegenQuality::Ultra ) )
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
static bool framegen_record_net_train( CVulkanCmdBuffer *pCmdBuffer, GamescopeFramegenQuality eQuality, bool bSceneCutGuard )
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
	const bool bConservativeBidir = framegen_net_bidir_conservative();
	const int nReservoirRead = framegen_luma_reservoir_read_index();
	const bool bShadingHistoryValid = eQuality == GamescopeFramegenQuality::Extreme
		&& framegen_shading_enabled( eQuality )
		&& bSceneCutGuard
		&& framegen_motion_history_valid( eQuality ) && nReservoirRead >= 0;
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

// Flush any unsaved learning. Called from vulkan_framegen_reset — every mode,
// resolution and teardown change funnels through it — and via atexit, so runs
// shorter than the checkpoint interval persist too (the old cadence-only write
// silently dropped them). Pure CPU on the cached copy: safe at any point,
// including after Vulkan teardown; never touches the GPU.
static void framegen_net_profile_flush()
{
	// The atomic is the cheap scheduling gate; the thread object is the lifetime
	// proof. Joining has no polling timeout and guarantees an older checkpoint
	// cannot rename over the newer synchronous flush below.
	if ( g_framegenNetWriteThread.joinable() )
		g_framegenNetWriteThread.join();
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
	const FramegenDispatch_t &dispatch, GamescopeFramegenQuality eQuality )
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
	const bool bNeedMotionHistory = eQuality >= GamescopeFramegenQuality::Ultra && !bBidir;
	const bool bNeedLumaReservoir = framegen_reservoir_enabled( eQuality ) && !bBidir;
	const bool bCaptureNeedsReverse = framegen_record_dir() != nullptr
		&& g_uFramegenRecordCount < framegen_record_max();
	const bool bNeedCheckedReverse = bBidir || framegen_net_requested( eQuality ) || bCaptureNeedsReverse;
	const bool bFBCheck = framegen_fbcheck_enabled( eQuality ) || bNeedCheckedReverse;

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
		|| ( framegen_net_requested( eQuality ) && !g_framegenMotion.bNetAllocTried )
		|| ( bBidir && framegen_net_requested( eQuality ) && g_framegenMotion.mvFieldRevNet == nullptr )
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
		// B4 stats are a high-tier resource. Lower tiers must not fail motion
		// setup because an adaptation-only image format/allocation failed.
		g_framegenMotion.statsAccum = nullptr;
		g_framegenMotion.statsReadback = nullptr;
		g_framegenHistory.ulAdaptStatsSeqNo = 0;
		if ( bAllocated && ( framegen_adapt_enabled( eQuality )
			|| ( framegen_net_requested( eQuality ) && framegen_net_online_enabled() ) ) )
		{
			CVulkanTexture::createFlags accumFlags;
			accumFlags.bStorage = true;
			accumFlags.bTransferSrc = true;
			g_framegenMotion.statsAccum = new CVulkanTexture();
			bAllocated = g_framegenMotion.statsAccum->BInit(
				gamescope::framegen::k_uAdaptationStatsCount,
				1, 1u, DRM_FORMAT_R32UI, accumFlags );
			if ( bAllocated )
			{
				CVulkanTexture::createFlags readbackFlags;
				readbackFlags.bMappable = true;
				readbackFlags.bTransferDst = true;
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
		if ( bAllocated && framegen_net_requested( eQuality ) )
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
				g_framegenMotion.netWeightsGpu = new CVulkanTexture();
				bNetOk = g_framegenMotion.netWeightsGpu->BInit( k_uFramegenNetTexW, k_uFramegenNetTexH, 1u, DRM_FORMAT_R32F, gpuFlags );

				CVulkanTexture::createFlags stagingFlags;
				stagingFlags.bMappable = true;
				stagingFlags.bTransferSrc = true;
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
					g_framegenMotion.netWeightsPrior = new CVulkanTexture();
					bool bOnlineOk = g_framegenMotion.netWeightsPrior->BInit( k_uFramegenNetTexW, k_uFramegenNetTexH, 1u, DRM_FORMAT_R32F, priorFlags );

					CVulkanTexture::createFlags stateFlags;
					stateFlags.bSampled = true;
					stateFlags.bStorage = true;
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

	// The luma reservoir is an Extreme-only enhancement, not a prerequisite
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
			vk_log.errorf( "framegen: luma-reservoir allocation failed; keeping the Extreme guided warp without third-frame disocclusion evidence" );
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
	pCmdBuffer->pushConstants<FramegenMotionFBCheckPush_t>( framegen_effective_fbcheck_tol( eQuality ), k_flFramegenFBTolSlope );
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
	pCmdBuffer->pushConstants<FramegenMotionFBCheckPush_t>( framegen_effective_fbcheck_tol( eQuality ), k_flFramegenFBTolSlope );
	pCmdBuffer->dispatch( div_roundup( lowW, pg ), div_roundup( lowH, pg ) );

	return true;
}

// Motion-compensated generation, part 2 (per slot): warp the current frame
// forward along the shared motion field, blending back to extrapolation where
// the match is unconfident. Each slot writes a distinct target, so warps in the
// same batch don't hazard against each other.
static void framegen_warp_slot( CVulkanCmdBuffer *pCmdBuffer, const gamescope::Rc<CVulkanTexture> &pTarget,
	float flStrength, GamescopeFramegenQuality eQuality )
{
	const uint32_t pg = 8;
	const bool bHistoryValid = framegen_motion_history_valid( eQuality );
	const bool bGuided = eQuality == GamescopeFramegenQuality::Extreme;
	const int nReservoirRead = framegen_luma_reservoir_read_index();
	const bool bReservoirValid = bGuided && bHistoryValid
		&& nReservoirRead >= 0;
	const bool bShadingValid = bGuided && bHistoryValid
		&& framegen_shading_enabled( eQuality )
		&& g_framegenMotion.bNetActive && g_framegenMotion.netShadingFocus != nullptr;
	const float flHistoryFlowScale = bHistoryValid ? framegen_motion_history_time_scale() : 1.0f;
	const float flAccelTimeFactor = bHistoryValid ? framegen_motion_accel_time_factor() : 0.5f;
	// Extreme uses the accelerated shader even during the first interval: its
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
	const bool bAgree = framegen_agreement_enabled( eQuality );
	const float flAgreeLo = bAgree ? framegen_effective_agree_lo( eQuality ) : 1e5f;
	const float flAgreeHi = bAgree ? framegen_effective_agree_hi( eQuality ) : 1e6f;
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
		// warm-up or non-net Extreme batch, bind a harmless sampled fallback and
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
	float flPhase, GamescopeFramegenQuality eQuality, float flOneSidedOverride = -1.0f,
	float flEndpointTraceOverride = -1.0f )
{
	const uint32_t pg = 8;
	const float flEndpointTraceStrength = eQuality == GamescopeFramegenQuality::Extreme
		? ( flEndpointTraceOverride >= 0.0f
			? std::clamp( flEndpointTraceOverride, 0.0f, 1.0f )
			: framegen_bidir_endpoint_trace_strength( eQuality ) )
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
	const bool bAgree = framegen_agreement_enabled( eQuality );
	const float flOneSidedStrength = flOneSidedOverride >= 0.0f
		? std::clamp( flOneSidedOverride, 0.0f, 1.0f )
		: framegen_bidir_one_sided_strength();
	pCmdBuffer->pushConstants<FramegenMotionBidirPush_t>( flPhase, (float)k_uFramegenMotionDownscale,
		bAgree ? framegen_effective_agree_lo( eQuality ) : 1e5f,
		bAgree ? framegen_effective_agree_hi( eQuality ) : 1e6f,
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

static bool framegen_submit_planned( const FramegenSlotRequest_t *pRequests, uint32_t nRequestCount, uint32_t nGapVblanks, const FramegenEffective_t &eff, uint64_t ulCompositeSeqNo, uint32_t nMaxDegradeSteps, bool bClearPending, const FramegenColorProbeRequest_t *pColorProbe = nullptr )
{
	if ( nRequestCount == 0 || nGapVblanks == 0 || ulCompositeSeqNo == 0 )
		return false;

	// B4: fold the previous batch's quality readback into the adaptation state
	// before recording this one, so its thresholds ride these push constants.
	framegen_adapt_consume( eff.quality );
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
	struct SlotPlan_t { gamescope::Rc<CVulkanTexture> tex; float phase; float strength; uint32_t slotIndex; };
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

		slots.push_back( { std::move( pGenerated ), pRequests[ i ].phase, pRequests[ i ].strength, pRequests[ i ].slotIndex } );
	}
	if ( slots.size() != nRequestCount )
	{
		static uint64_t s_uOutputPressureDebugLogCounter = 0;
		if ( FramegenDebugShouldLog( s_uOutputPressureDebugLogCounter ) )
			vk_log.infof( "framegen: output-pool pressure admitted=%zu/%u pool=%zu pending=%zu",
				slots.size(), nRequestCount, g_output.framegenOutputImages.size(), g_framegenHistory.pending.size() );
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
	std::unique_ptr<CVulkanCmdBuffer> pCmdBuffer = g_device.commandBuffer();
	pCmdBuffer->markFramegen();
	const FramegenDispatch_t &dispatch = framegen_dispatch_for_format( g_framegenHistory.drmFormat );
	const bool bBidir = vulkan_framegen_bidir_active();
	const bool bMotionRequested = eff.mode == GamescopeFramegenMode::Motion
		&& dispatch.motionSupported;
	// The classic/JIT/idle planners can submit several batches for one real
	// interval. Re-estimating and re-training on that identical pair both wastes
	// the deadline and statistically overweights slow intervals. The finalized
	// field remains resident until any new preparation explicitly invalidates it.
	const bool bReuseMotion = pColorProbe == nullptr
		&& bMotionRequested
		&& g_framegenMotion.uMotionFieldFrameId != 0
		&& g_framegenMotion.uMotionFieldFrameId == g_framegenHistory.currentFrameId
		&& g_framegenMotion.eMotionFieldQuality == eff.quality
		&& g_framegenMotion.bMotionFieldBidir == bBidir
		&& framegen_motion_field() != nullptr
		&& ( !bBidir || framegen_motion_field_rev() != nullptr );
	// Bracket the batch's dispatches with GPU timestamps so its cost feeds the
	// degradation ladder next interval. Cached refills contain only per-slot
	// warps; mixing those cheap samples into the same (rung,count) EMA would
	// understate the cost of the next real pair's full setup batch.
	const int nQuerySlot = bReuseMotion ? -1 : g_device.framegenTimestampBegin( pCmdBuffer.get() );

	// Preserve the preceding pair's finalized displacement before preparation
	// overwrites the working fields. The exact quality/mode and consecutive-ID
	// gates keep estimator changes or a skipped interval from looking like
	// physical acceleration. The copy and later history reads are ordered in
	// this same command buffer.
	if ( !bReuseMotion && pColorProbe == nullptr && bMotionRequested && !bBidir
		&& eff.quality >= GamescopeFramegenQuality::Ultra )
	{
		const bool bHistorySourceValid = g_framegenMotion.mvFieldHistory != nullptr
			&& g_framegenMotion.uMotionFieldFrameId != 0
			&& g_framegenMotion.uMotionFieldFrameId + 1u == g_framegenHistory.currentFrameId
			&& g_framegenMotion.eMotionFieldQuality == eff.quality
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
			g_framegenHistory.currentReal->height(), dispatch, eff.quality ) );
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
	// B4: with the field final, record the quality probe — the warps below
	// read its verdict in this same batch, the CPU next batch.
	const bool bAdaptProbe = bMotion && !bReuseMotion && framegen_adapt_enabled( eff.quality );
	if ( bAdaptProbe )
		framegen_record_adapt_probe( pCmdBuffer.get(), g_framegenMotion.width, g_framegenMotion.height );

	// C2: train after the probe so a content cut can zero every gradient slice
	// before Adam sees it. Same-batch inference still used the pre-step weights;
	// the optimizer publishes only for the next batch. When adaptation is
	// explicitly disabled, training retains its previous unguarded behavior.
	const bool bNetStateReadback = pColorProbe == nullptr
		&& !bReuseMotion && g_framegenMotion.bNetActive && framegen_net_online_enabled()
		&& framegen_record_net_train( pCmdBuffer.get(), eff.quality, bAdaptProbe );
	if ( bMotion && !bReuseMotion && pColorProbe == nullptr )
	{
		g_framegenMotion.uMotionFieldFrameId = g_framegenHistory.currentFrameId;
		g_framegenMotion.uMotionFieldIntervalNs =
			g_framegenHistory.currentPresentTimeNs > g_framegenHistory.previousPresentTimeNs
			? g_framegenHistory.currentPresentTimeNs - g_framegenHistory.previousPresentTimeNs : 0;
		g_framegenMotion.eMotionFieldQuality = eff.quality;
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
					std::clamp( slot.phase, 0.0f, 1.0f ), eff.quality,
					flOneSidedOverride, flEndpointTraceOverride );
			}
			else
				framegen_warp_slot( pCmdBuffer.get(), slot.tex, slot.strength, eff.quality );
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
	// lumaPrev for the next consecutive Extreme batch. At 1/8 resolution this
	// is 1/64 the texel traffic of copying the full color frame. Two tiny images
	// retain both current-2 (for JIT/refill slots of this same interval) and
	// current-1 (for the next interval); a refill finds the latter already
	// published and records no redundant copy.
	if ( bMotion && !bBidir && eff.quality == GamescopeFramegenQuality::Extreme
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
	// Attribute this batch's measured cost to the rung and generated-count it ran
	// at; batch cost scales with slot count, especially x3/x4 extrapolate pairs.
	const uint64_t ulSeqNo = g_device.submitFramegen( std::move( pCmdBuffer ), ulCompositeSeqNo, nQuerySlot, g_framegenHistory.nDegradeSteps, (uint32_t)slots.size() );
	g_framegenHistory.lastFramegenWorkSeqNo = ulSeqNo;

	for ( const SlotPlan_t &slot : slots )
	{
		if ( pColorProbe != nullptr )
			break;
		FramegenHistory_t::PendingGenerated_t entry;
		entry.tex = slot.tex;
		entry.seqNo = ulSeqNo;
		entry.frameId = g_framegenHistory.currentFrameId;
		entry.phase = slot.phase;
		g_framegenHistory.pending.push_back( std::move( entry ) );
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
			gamescope::framegen::quality_name( eff.quality ),
			eff.multiplier,
			g_framegenHistory.nDegradeSteps,
			nMaxDegradeSteps,
			g_device.framegenLastGpuTimeNs() / 1.0e6,
			bReuseMotion ? " (prior full batch; cached-field warp unmeasured)" : "",
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
	// with framegen_jit_submit below, where it is a display-clock measurement.
	std::vector<FramegenSlotRequest_t> requests;
	requests.reserve( nGenerate );
	const float flBidirPhaseBias = vulkan_framegen_bidir_active()
		? framegen_bidir_phase_bias() : 0.0f;
	for ( uint32_t i = 0; i < nGenerate; i++ )
	{
		const uint32_t k = nFirstSlot + i;
		// Effective forward coefficient: temporal placement scaled by the user
		// strength (0.5 is neutral, reproducing the classic x2 half-way step).
		// Idle refill can move past the originally expected next-real slot when
		// the game stalls, but never lets prediction run away unbounded.
		requests.push_back( gamescope::framegen::classic_slot_request(
			k, i, nGapVblanks, nGenerate, flBidirPhaseBias,
			g_flFramegenStrength, k_flFramegenMaxForwardStrength ) );
	}

	return framegen_submit_planned( requests.data(), (uint32_t)requests.size(), nGapVblanks, eff, ulCompositeSeqNo, nMaxDegradeSteps, bClearPending );
}

// JIT display-clock slot (#06). Plan exactly ONE generated frame, for the
// vblank AFTER the one the current wake is deciding, and compute its phase at
// submit time from two measurements instead of a gap-count guess:
//   phase = (t_targetVblank - t_realFrameVblank) / frametimeEMA
// Both vblank times come from the vblank timer, whose clock is fed by the
// backend's real flip feedback (KMS pageflip timestamps on DRM) — vendor-
// agnostic ground truth for when frames actually scan out. The pixels are
// therefore stamped with the time they will be SHOWN, which removes the batch
// path's phase-vs-vblank sawtooth at fractional rates (45 fps on 60 Hz).
// Targeting one vblank ahead gives the dispatch a full interval of GPU budget,
// so the present path's completion check almost never sees an unfinished slot.
static bool framegen_jit_submit( uint64_t ulCompositeSeqNo, uint32_t nMaxDegradeSteps )
{
	if ( !vulkan_framegen_is_enabled() || !framegen_jit_enabled()
		|| !g_framegenHistory.valid || !g_framegenHistory.pending.empty()
		|| g_framegenHistory.previousReal == nullptr || g_framegenHistory.currentReal == nullptr
		|| g_framegenHistory.ulFrametimeEmaNs == 0 || g_framegenHistory.ulCurrentRealVblankNs == 0
		|| ulCompositeSeqNo == 0 )
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

	const int nFramegenRefreshMhz = g_nNestedRefresh ? g_nNestedRefresh : g_nOutputRefresh;
	const uint64_t ulVblankIntervalNs = nFramegenRefreshMhz > 0 ? 1'000'000'000'000ull / (uint64_t)nFramegenRefreshMhz : 8'333'333ull;
	// The wake this runs in is still deciding GetNextVBlank(0)'s slot; ours is
	// the one after it.
	const uint64_t ulTargetVblankNs = GetVBlankTimer().GetNextVBlank( 0 ) + ulVblankIntervalNs;
	if ( ulTargetVblankNs <= g_framegenHistory.ulCurrentRealVblankNs )
		return false;

	const uint64_t ulTargetDeltaNs = ulTargetVblankNs - g_framegenHistory.ulCurrentRealVblankNs;
	const gamescope::framegen::TimedPrediction prediction = gamescope::framegen::timed_prediction(
		ulTargetDeltaNs, g_framegenHistory.ulFrametimeEmaNs, g_flFramegenStrength );
	// Past the forward cap every further slot would be the same capped
	// prediction — a repeat we'd pay full generation bandwidth for. Stop; the
	// display repeats the last scanned-out frame until real content returns.
	if ( prediction.rawStrength > k_flFramegenMaxForwardStrength )
		return false;
	const float flPhase = prediction.phase;
	const float flStrength = gamescope::framegen::clamp_forward_strength(
		prediction.rawStrength, k_flFramegenMaxForwardStrength );

	// Interval-relative slot index / gap equivalents, for bookkeeping and logs
	// only: a JIT tick re-measures the display clock rather than continuing a
	// slot ladder, so these never feed a later phase computation.
	const gamescope::framegen::JitBookkeeping bookkeeping = gamescope::framegen::jit_bookkeeping(
		ulTargetDeltaNs, g_framegenHistory.ulFrametimeEmaNs, ulVblankIntervalNs );
	const uint32_t nSlotIndex = bookkeeping.slotIndex;
	const uint32_t nGapVblanks = bookkeeping.gapVblanks;

	static uint64_t s_uJitDebugLogCounter = 0;
	if ( FramegenDebugShouldLog( s_uJitDebugLogCounter ) )
		vk_log.infof( "framegen: jit slot phase=%.3f strength=%.3f target=+%.2fms ema=%.2fms",
			flPhase, flStrength,
			( ulTargetVblankNs > now ? ulTargetVblankNs - now : 0 ) / 1.0e6,
			g_framegenHistory.ulFrametimeEmaNs / 1.0e6 );

	const FramegenEffective_t eff = framegen_effective_config( g_framegenHistory.nDegradeSteps );
	const FramegenSlotRequest_t request = { flPhase, flStrength, nSlotIndex };
	return framegen_submit_planned( &request, 1, nGapVblanks, eff, ulCompositeSeqNo, nMaxDegradeSteps, false );
}

// VRR hybrid slot (#01). Plan exactly ONE generated frame at the content
// midpoint of the measured real-frame interval. This inverts #06: under a
// fixed refresh, JIT asks "given the next vblank, what phase is that?"; under
// active adaptive sync there is no grid — the real frame scanned out on
// arrival — so we PICK the phase (0.5, the content-correct midpoint, exact by
// construction) and manufacture the display event for it: steamcompmgr arms an
// absolute CLOCK_MONOTONIC timer (the clock KMS flip timestamps use) for
// t_realflip + 0.5*EMA and flips the frame then. Always a single slot,
// whatever the configured multiplier: each extra mid flip would multiply the
// timer/cancel bookkeeping and shrink the spacing toward the panel's minimum
// flip interval; one mid flip is the sane ceiling under VRR.
static bool framegen_vrr_hybrid_submit( uint64_t ulCompositeSeqNo, uint32_t nMaxDegradeSteps )
{
	if ( !vulkan_framegen_is_enabled() || !vulkan_framegen_vrr_hybrid_active()
		|| !g_framegenHistory.valid || !g_framegenHistory.pending.empty()
		|| g_framegenHistory.previousReal == nullptr || g_framegenHistory.currentReal == nullptr
		|| g_framegenHistory.ulFrametimeEmaNs == 0 || ulCompositeSeqNo == 0 )
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
		g_framegenHistory.ulFrametimeEmaNs, ulVblankIntervalNs );

	static uint64_t s_uHybridDebugLogCounter = 0;
	if ( FramegenDebugShouldLog( s_uHybridDebugLogCounter ) )
		vk_log.infof( "framegen: vrr-hybrid slot strength=%.3f mid=+%.2fms ema=%.2fms",
			flStrength,
			g_framegenHistory.ulFrametimeEmaNs / 2.0e6,
			g_framegenHistory.ulFrametimeEmaNs / 1.0e6 );

	const FramegenEffective_t eff = framegen_effective_config( g_framegenHistory.nDegradeSteps );
	const FramegenSlotRequest_t request = { flPhase, flStrength, 1u };
	return framegen_submit_planned( &request, 1, nGapVblanks, eff, ulCompositeSeqNo, nMaxDegradeSteps, false );
}

// Reactive JIT catch-all, called by the present decision when a vblank goes to
// a hardware repeat while framegen is active (a stall, a too-slow discard, or
// a mispredicted keep-up), and by the consume path when the pending slot
// drains. Fills from the next vblank so a hole never exceeds one vblank.
void vulkan_framegen_jit_tick()
{
	// Under active VRR hybrid the fixed-grid JIT planner has no grid to plan
	// against (and hybrid deliberately does not fill stalls — see the consume
	// drain hook); ignore the tick until VRR deactivates.
	if ( !framegen_jit_enabled() || vulkan_framegen_vrr_hybrid_active() )
		return;
	framegen_jit_submit( g_framegenHistory.lastCompositeSeqNo, framegen_max_degrade_steps() );
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
				framegen_prepare_motion( cmd.get(), res.nWidth, res.nHeight, dispatch, g_eFramegenQuality );
				if ( g_eFramegenQuality >= GamescopeFramegenQuality::Ultra
					&& g_framegenMotion.mvFieldHistory != nullptr )
					cmd->copyImage( framegen_motion_field(), g_framegenMotion.mvFieldHistory );
				if ( g_eFramegenQuality == GamescopeFramegenQuality::Extreme
					&& g_framegenMotion.lumaReservoir[0] != nullptr )
					cmd->copyImage( g_framegenMotion.lumaPrev, g_framegenMotion.lumaReservoir[0] );
				g_device.submit( std::move( cmd ) );
				g_device.waitIdle();
				if ( g_eFramegenQuality >= GamescopeFramegenQuality::Ultra
					&& g_framegenMotion.mvFieldHistory != nullptr )
				{
					g_framegenMotion.uMotionHistoryFrameId = 2;
					g_framegenMotion.uMotionHistoryIntervalNs = 16'666'667ull;
					g_framegenHistory.currentFrameId = 3;
					g_framegenHistory.previousPresentTimeNs = 16'666'667ull;
					g_framegenHistory.currentPresentTimeNs = 33'333'334ull;
				}
				if ( g_eFramegenQuality == GamescopeFramegenQuality::Extreme
					&& g_framegenMotion.lumaReservoir[0] != nullptr )
					g_framegenMotion.uLumaReservoirFrameId[0] = 1;
			}

			double msMotionPrep = timePass( [&]( CVulkanCmdBuffer *cmd ) {
				framegen_prepare_motion( cmd, res.nWidth, res.nHeight, dispatch, g_eFramegenQuality );
			} );
			printf( "%-8s  %-26s  %10.3f\n", res.pszName, "motion setup (per real)", msMotionPrep );

			// B4 stats probe (clear + accumulate + 384-byte readback copy) —
			// recorded after setup in production, so its cost adds to the
			// per-real tail, not per-slot.
			if ( framegen_adapt_enabled( g_eFramegenQuality ) )
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
						framegen_record_net_train( cmd, g_eFramegenQuality, false );
					} );
					printf( "%-8s  %-26s  %10.3f\n", res.pszName, "online net train (per real)", msNetTrain );
				}
			}

			double msMotionWarp = timePass( [&]( CVulkanCmdBuffer *cmd ) {
				framegen_warp_slot( cmd, pOut, g_flFramegenStrength, g_eFramegenQuality );
			} );
			printf( "%-8s  %-26s  %10.3f\n", res.pszName,
				g_eFramegenQuality == GamescopeFramegenQuality::Extreme
					? "motion guided warp (per gen)"
					: g_eFramegenQuality == GamescopeFramegenQuality::Ultra
						? "motion accel warp (per gen)" : "motion warp (per gen)",
				msMotionWarp );

			if ( g_eFramegenQuality == GamescopeFramegenQuality::Extreme
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
					framegen_bidir_warp_slot( cmd, pOut, 0.5f, g_eFramegenQuality, -1.0f, 0.0f );
				} );
				printf( "%-8s  %-26s  %10.3f\n", res.pszName, "motion bidir warp (per gen)", msBidirWarp );
				if ( g_eFramegenQuality == GamescopeFramegenQuality::Extreme )
				{
					double msBidirTrace = timePass( [&]( CVulkanCmdBuffer *cmd ) {
						framegen_bidir_warp_slot( cmd, pOut, 0.5f, g_eFramegenQuality, -1.0f, 1.0f );
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
		.quality = g_eFramegenQuality,
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

static void framegen_record_real_frame( gamescope::Rc<CVulkanTexture> pRealFrame, const struct FrameInfo_t *pFrameInfo, uint64_t ulCompositeSeqNo )
{
	if ( !vulkan_framegen_is_enabled() || !pRealFrame || !pFrameInfo )
		return;

	if ( !g_bLoggedFramegenConfig )
	{
		vk_log.infof( "framegen: enabled mode=%s quality=%s multiplier=%d%s",
			gamescope::framegen::mode_name( g_eFramegenMode ),
			gamescope::framegen::quality_name( g_eFramegenQuality ), g_nFramegenMultiplier,
			g_device.hasFramegenQueue() ? " (dedicated queue)" : "" );
		vk_log.infof( "framegen: forcing composite path" );
		if ( vulkan_framegen_vrr_hybrid_requested() )
			vk_log.infof( "framegen: VRR hybrid requested — adaptive sync stays active, generated frames flip mid-interval; tearing remains suppressed" );
		else
			vk_log.infof( "framegen: adaptive sync (VRR) and tearing flips are suppressed while framegen is active" );
		if ( framegen_bidir_enabled() )
		{
			if ( g_eFramegenMode == GamescopeFramegenMode::Motion && !framegen_jit_enabled() && !vulkan_framegen_vrr_hybrid_requested() )
				vk_log.infof( "framegen: bidirectional interpolation requested (B3) — generated frames interpolate between the two real frames; real-frame presentation is delayed up to one interval" );
			else
				vk_log.infof( "framegen: GAMESCOPE_FRAMEGEN_BIDIR ignored (requires motion mode, incompatible with JIT/VRR-hybrid pacing)" );
		}
		if ( framegen_base_layer_enabled() )
			vk_log.infof( "framegen: base-layer generation + late overlay composite requested (#02) — predicting on the pre-upscale game layer, overlays/cursor composite fresh onto generated frames" );
		if ( g_eFramegenMode == GamescopeFramegenMode::Motion )
		{
			const bool bBidirActive = vulkan_framegen_bidir_active();
			if ( bBidirActive )
			{
				vk_log.infof( "framegen: bidirectional quality path — symmetric checked forward/reverse fields%s; causal acceleration, Extreme color-guided reconstruction, reservoir and shading are not scheduled",
					framegen_agreement_enabled( g_eFramegenQuality ) ? " + full-resolution agreement" : "" );
				if ( framegen_bidir_phase_bias() > 0.0f )
					vk_log.infof( "framegen: experimental bidir phase bias %.2f — low-latency queue timing preserved; generated phases move partially from k/gap toward uniform multiplier spacing",
						framegen_bidir_phase_bias() );
				if ( framegen_bidir_one_sided_strength() > 0.0f )
					vk_log.infof( "framegen: experimental bidir one-sided occlusion authority %.2f — strongly asymmetric checked fields retain more of the surviving warped side without changing both-valid or both-killed fallback",
						framegen_bidir_one_sided_strength() );
				if ( framegen_bidir_endpoint_trace_strength( g_eFramegenQuality ) > 0.0f )
					vk_log.infof( "framegen: experimental bidir endpoint trace %.2f — one symmetric closure-gated fixed-point correction; queue and flip timing are unchanged",
						framegen_bidir_endpoint_trace_strength( g_eFramegenQuality ) );
			}
			else if ( g_eFramegenQuality == GamescopeFramegenQuality::Low )
				vk_log.infof( "framegen: low quality — forward matcher + constant-velocity warp only" );
			else if ( g_eFramegenQuality == GamescopeFramegenQuality::Medium )
				vk_log.infof( "framegen: medium quality — reverse consistency + full-resolution agreement enabled" );
			else if ( g_eFramegenQuality == GamescopeFramegenQuality::Ultra )
				vk_log.infof( "framegen: ultra quality — confidence-gated causal temporal acceleration enabled after one consecutive field warm-up" );
			else if ( g_eFramegenQuality == GamescopeFramegenQuality::Extreme )
				vk_log.infof( "framegen: extreme quality — color-guided reconstruction + causal acceleration%s",
					framegen_reservoir_enabled( g_eFramegenQuality )
						? " + three-real-frame disocclusion reservoir enabled"
						: " (disocclusion reservoir disabled by GAMESCOPE_FRAMEGEN_RESERVOIR=0)" );
			if ( framegen_adapt_enabled( g_eFramegenQuality ) )
				vk_log.infof( "framegen: self-supervised adaptation active (B4) — each real frame grades the field that predicted it; blend trust follows same-batch, thresholds auto-calibrate next batch (GAMESCOPE_FRAMEGEN_ADAPT=0 disables)" );
			else if ( g_eFramegenQuality < GamescopeFramegenQuality::High )
				vk_log.infof( "framegen: self-supervised adaptation not scheduled below high quality" );
			else
				vk_log.infof( "framegen: self-supervised adaptation disabled (GAMESCOPE_FRAMEGEN_ADAPT=0)" );
			if ( framegen_net_requested( g_eFramegenQuality ) )
			{
				const bool bConservativeBidir = framegen_net_bidir_conservative();
				if ( bConservativeBidir )
					vk_log.infof( "framegen: learned bidirectional confidence veto active (C) — FB-checked geometry is preserved, confidence can only decrease; GAMESCOPE_FRAMEGEN_NET_BIDIR_FLOW=1 restores experimental endpoint-trained flow correction" );
				else
					vk_log.infof( "framegen: learned forward-field refinement active (C) — the net improves causal motion prediction once per real frame (bounded flow residual + evidence-gated confidence)" );
				if ( g_eFramegenQuality == GamescopeFramegenQuality::Extreme && !bBidirActive )
					vk_log.infof( "framegen: causal shading-persistence head %s — three-frame in-situ supervision, bounded color-trend correction (GAMESCOPE_FRAMEGEN_SHADING=0 disables for A/B)",
						framegen_shading_enabled( g_eFramegenQuality ) ? "enabled" : "disabled" );
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
			else if ( g_eFramegenQuality < GamescopeFramegenQuality::High
				&& ( framegen_net_weights_path() != nullptr || framegen_net_online_enabled() ) )
			{
				vk_log.infof( "framegen: learned refinement requested but not scheduled below high quality" );
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
	// pay for a duplicate history copy. Pointer identity is sufficient here —
	// a recycled allocation would at worst skip one record and self-heals.
	const CVulkanTexture *pBaseTexture = pFrameInfo->layerCount > 0 ? pFrameInfo->layers[ 0 ].tex.get() : nullptr;
	if ( pBaseTexture && pBaseTexture == g_framegenHistory.pLastBaseTexture )
	{
		g_framegenHistory.bBidirSameBaseComposite = true;
		static uint64_t s_uOverlayOnlyDebugLogCounter = 0;
		if ( FramegenDebugShouldLog( s_uOverlayOnlyDebugLogCounter ) )
			vk_log.infof( "framegen: ignoring overlay-only repaint (base content unchanged)" );
		return;
	}
	g_framegenHistory.pLastBaseTexture = pBaseTexture;
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

	g_framegenHistory.currentFrameId++;
	g_framegenHistory.previousPresentTimeNs = ulPrevRealFrameTimeNs;
	g_framegenHistory.currentPresentTimeNs = now;
	// Display-clock anchor (#06): the vblank this composite will scan out at.
	// GetLastVBlank() is fed by the backend's real flip feedback (KMS pageflip
	// timestamps on DRM), so this pins the real frame to the display's own
	// measured clock — the reference a JIT slot's phase is computed against.
	g_framegenHistory.ulCurrentRealVblankNs = GetVBlankTimer().GetNextVBlank( 0 );
	g_framegenHistory.lastCompositeSeqNo = ulCompositeSeqNo;
	g_framegenHistory.nLastGeneratedSlot = 0;
	g_framegenHistory.nLastGenerationGapVblanks = 0;

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
		g_framegenHistory.ulCurrentRealVblankNs = GetVBlankTimer().GetNextVBlank( 0 );
		g_framegenHistory.lastCompositeSeqNo = ulCompositeSeqNo;
	}

	// Cadence estimator (#06): fold this interval into the slew-limited EMA.
	// The clamp bounds how hard a single hitch can yank the estimate (a doubled
	// sample moves it by at most 12.5%), while a genuine rate change converges
	// in ~8 frames. Placed after the frame-gap invalidation so a stall interval
	// (ulPrevRealFrameTimeNs zeroed above) never poisons the estimate.
	if ( ulPrevRealFrameTimeNs != 0 )
	{
		const uint64_t ulSampleNs = now - ulPrevRealFrameTimeNs;
		uint64_t &ulEma = g_framegenHistory.ulFrametimeEmaNs;
		// Hitch marker: an isolated real-frame interval well past the running
		// cadence is exactly the "smooth for a long time, then a bump" the eye
		// catches. The threshold (2x cadence AND +10ms absolute) sits above
		// nested parent-vblank doubling and a game's ordinary variance, so what
		// remains logs on every occurrence (not decimated — real hangs are rare
		// by definition) and can be correlated with the surrounding reset /
		// snap / degrade / checkpoint lines to attribute the source. Two
		// compares per real frame; debug-gated output only.
		if ( g_bFramegenDebug && ulEma != 0 && ulSampleNs > std::max<uint64_t>( ulEma * 2, ulEma + 10'000'000ull ) )
		{
			static uint64_t s_ulLastSpikeNs = 0;
			vk_log.infof( "framegen: frametime spike — real-frame gap %.2fms vs ema %.2fms (frame id=%" PRIu64 ", %.1fs since last)",
				ulSampleNs * 1e-6, ulEma * 1e-6, g_framegenHistory.currentFrameId,
				s_ulLastSpikeNs != 0 ? ( now - s_ulLastSpikeNs ) * 1e-9 : 0.0 );
			s_ulLastSpikeNs = now;
		}
		ulEma = ulEma == 0
			? ulSampleNs
			: ( ulEma * 7 + std::clamp( ulSampleNs, ulEma / 2, ulEma * 2 ) ) / 8;
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
	}
	else
	{
		g_framegenHistory.previousReal = g_framegenHistory.currentReal;
		g_framegenHistory.currentReal = pRealFrame;
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

	const bool bReactiveReady = gamescope::framegen::reactive_generation_ready(
		bGeneratable, g_framegenHistory.nStableFrames );
	const bool bShouldGenerate = bCanSpeculate || bReactiveReady;

	if ( !bShouldGenerate )
	{
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

	// Deadline-driven degradation (#04): shed one quality rung whenever the CURRENT
	// config's measured GPU cost (see framegenGarbageCollect) overruns the vblank
	// budget, so a too-slow config never causes a missed generated frame. Evaluated
	// only here, on frames we actually generate, so the rung it settles on is a rung
	// that was really measured and an idle/dormant stretch never moves it.
	//
	// Deliberately MONOTONIC within a scene: it only ever degrades, never restores
	// quality mid-scene. Restoring means re-probing a richer config that may not fit
	// and then dropping it again — a visible toggle of the generated-frame look, i.e.
	// exactly the micro-stutter this feature exists to avoid. Quality is re-probed
	// from full on the next scene change (framegenResetRungCosts + nDegradeSteps = 0
	// in vulkan_framegen_invalidate_history), which is the natural "the workload may
	// have changed" signal (focus change, layer/EOTF change, long stall). When no
	// measurement is available the current rung's cost is 0 and the ladder stays at
	// full quality, leaving the existing reactive discard as the only safety net.
	const FramegenEffective_t curEffForLadder = framegen_effective_config( g_framegenHistory.nDegradeSteps );
	const uint32_t nLadderGapVblanks = gamescope::framegen::expanded_gap_vblanks(
		nMeasuredGapVblanks, curEffForLadder.multiplier, bCanSpeculate );
	const uint32_t nGapSlots = nLadderGapVblanks > 1 ? nLadderGapVblanks - 1 : 0;
	// JIT pacing and the VRR hybrid always submit one-slot batches, so their
	// rung costs are keyed by count 1 and only the mode rung
	// (motion tier or motion->extrapolate) can shed work — a multiplier notch cannot reduce a
	// count that is already minimal, and the "does the step actually help"
	// check below correctly never takes it.
	const bool bVrrHybrid = vulkan_framegen_vrr_hybrid_active();
	const bool bJitPacing = !bVrrHybrid && framegen_jit_enabled();
	const bool bSingleSlotPacing = bJitPacing || bVrrHybrid;
	const uint32_t nCurGenForLadder = gamescope::framegen::ladder_generated_count(
		nGapSlots, curEffForLadder.multiplier, bSingleSlotPacing );
	const uint64_t ulCurRungCostNs = g_device.framegenRungCostNs( g_framegenHistory.nDegradeSteps, nCurGenForLadder );
	const uint32_t uCurRungSamples = g_device.framegenRungSampleCount( g_framegenHistory.nDegradeSteps, nCurGenForLadder );
	gamescope::framegen::DeadlineLadderEvaluation ladderEvaluation =
		gamescope::framegen::evaluate_deadline_ladder(
			{ g_framegenHistory.nDegradeSteps, g_framegenHistory.nDegradeHold },
			nMaxDegradeSteps, ulCurRungCostNs, uCurRungSamples, ulVblankIntervalNs );
	if ( ladderEvaluation.state.holdFrames != g_framegenHistory.nDegradeHold )
		g_framegenHistory.nDegradeHold = ladderEvaluation.state.holdFrames;
	if ( ladderEvaluation.tryDegrade )
	{
		// Over budget at the current rung. Only take the step if it actually
		// reduces work at THIS gap: dropping a motion tier or motion->extrapolate
		// always lowers per-frame cost, but a pure multiplier notch only helps when the gap
		// lets it generate fewer frames. Otherwise stepping would cost quality
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
		}
	}

	const FramegenEffective_t eff = framegen_effective_config( g_framegenHistory.nDegradeSteps );

	// VRR hybrid (#01): the real frame just presented immediately (adaptive
	// sync — no grid quantization, no added latency), and the one generated
	// frame is placed at the content midpoint of the measured interval by a
	// timer-armed flip (steamcompmgr owns the timer and reads the offset via
	// vulkan_framegen_vrr_hybrid_mid_offset_ns). Keep-up guard: skip when the
	// interval is too short to split; scheduling.hpp owns the threshold.
	if ( bVrrHybrid )
	{
		g_framegenHistory.pending.clear();
		if ( gamescope::framegen::vrr_hybrid_interval_eligible(
			g_framegenHistory.ulFrametimeEmaNs, ulVblankIntervalNs ) )
		{
			framegen_vrr_hybrid_submit( ulCompositeSeqNo, nMaxDegradeSteps );
		}
		else
		{
			static uint64_t s_uHybridKeepUpDebugLogCounter = 0;
			if ( FramegenDebugShouldLog( s_uHybridKeepUpDebugLogCounter ) )
				vk_log.infof( "framegen: vrr-hybrid keep-up skip ema=%.2fms min-flip-interval=%.2fms",
					g_framegenHistory.ulFrametimeEmaNs / 1.0e6, ulVblankIntervalNs / 1.0e6 );
		}
		return;
	}

	// JIT display-clock pacing (#06): a new real frame supersedes any stale
	// prediction, then exactly one slot is planned for the next vblank with a
	// phase measured at submit time. Keep-up guard: when the measured cadence
	// says the game holds refresh, the next vblank will carry a real frame and
	// a speculative slot would only be discarded — skip it (this is the
	// previously missing "skip when keeping up" guard for the speculative
	// path). A genuine stall is still covered by the repeat-slot tick in the
	// present decision from its second vblank onward.
	if ( bJitPacing )
	{
		g_framegenHistory.pending.clear();
		if ( gamescope::framegen::jit_interval_eligible(
			g_framegenHistory.ulFrametimeEmaNs, ulVblankIntervalNs ) )
		{
			framegen_jit_submit( ulCompositeSeqNo, nMaxDegradeSteps );
		}
		else
		{
			static uint64_t s_uJitKeepUpDebugLogCounter = 0;
			if ( FramegenDebugShouldLog( s_uJitKeepUpDebugLogCounter ) )
				vk_log.infof( "framegen: jit keep-up skip ema=%.2fms interval=%.2fms",
					g_framegenHistory.ulFrametimeEmaNs / 1.0e6, ulVblankIntervalNs / 1.0e6 );
		}
		return;
	}

	// Fill as many empty vblanks as the measured gap actually offers, capped by
	// the multiplier: generate one fewer than the whole-vblank gap (the final slot
	// is the next real frame). The ladder's effective multiplier can lower this
	// ceiling below the startup one under GPU pressure; it is always
	// <= g_nFramegenMultiplier, so the pre-sized output pool holds.
	//
	// Bidir (B3) never speculates past the measured gap: its slots are appended
	// to the presentation queue rather than superseded by the next real frame,
	// so planning more slots than the interval has vblanks would accumulate
	// latency instead of being discarded. The measured (just-completed)
	// interval is also exactly the span its phases interpolate.
	const bool bBidir = vulkan_framegen_bidir_active();
	const uint32_t nGapVblanks = gamescope::framegen::expanded_gap_vblanks(
		nMeasuredGapVblanks, eff.multiplier, bCanSpeculate && !bBidir );
	uint32_t nGenerate = gamescope::framegen::generated_slots_for_gap(
		nGapVblanks, eff.multiplier, g_device.hasFramegenQueue() );

	// Without a dedicated framegen queue the batch is submitted to the same
	// in-order queue as the next composite, so it sits in front of it. Cap the
	// single-queue path to one generated frame to bound that head-of-line work —
	// the proven x2-prototype behaviour — rather than amplifying it under x3/x4.
	// generated_slots_for_gap applies that cap while forming nGenerate above.
	if ( nGenerate == 0 )
		return;

	if ( bBidir )
	{
		// The interpolations lead INTO this real frame on the delayed timeline;
		// queued REAL entries must stay ordered, but an interpolation becomes a
		// latency liability once the game outruns the one-flip-per-vblank drain.
		// Reserve room for this batch AND its real endpoint before appending: the
		// old check ran against the pre-append size, so a nominal cap of 10 could
		// repeatedly reach 14 with an x4 batch. Shed oldest predictions first,
		// never real frames; if real backlog alone consumes the budget, admit fewer
		// new interpolations and let the timeline catch up.
		const size_t uMaxPending = (size_t)( 2u * ( eff.multiplier + 1u ) );
		const uint32_t nRequested = nGenerate;
		const size_t uDesiredOldMax = uMaxPending > (size_t)nGenerate + 1u
			? uMaxPending - (size_t)nGenerate - 1u : 0u;
		size_t uShed = 0;
		for ( auto it = g_framegenHistory.pending.begin();
			g_framegenHistory.pending.size() > uDesiredOldMax && it != g_framegenHistory.pending.end(); )
		{
			if ( !it->bReal )
			{
				it = g_framegenHistory.pending.erase( it );
				uShed++;
			}
			else
			{
				++it;
			}
		}
		const size_t uReservedForReal = g_framegenHistory.pending.size() < uMaxPending
			? 1u : 0u;
		const size_t uAvailableGenerated = uMaxPending - std::min( uMaxPending,
			g_framegenHistory.pending.size() + uReservedForReal );
		nGenerate = std::min<uint32_t>( nGenerate, (uint32_t)uAvailableGenerated );
		static uint64_t s_uOverflowDebugLogCounter = 0;
		if ( ( uShed > 0 || nGenerate != nRequested ) && FramegenDebugShouldLog( s_uOverflowDebugLogCounter ) )
		{
			vk_log.infof( "framegen: bidir queue pressure pending=%zu/%zu shed=%zu admitted=%u/%u",
				g_framegenHistory.pending.size(), uMaxPending, uShed, nGenerate, nRequested );
		}

		if ( nGenerate > 0 )
			framegen_submit_batch( 1, nGapVblanks, nGenerate, eff, ulCompositeSeqNo, nMaxDegradeSteps, false );

		// Queue the real frame itself behind its interpolations. Its composite
		// rides the realtime queue (seqNo 0 on the framegen timeline = always
		// ready); the flip substitution presents the queue front in its place
		// this paint. Queued even if the batch failed — the queue then just
		// presents it next, degrading to (near-)zero added delay.
		FramegenHistory_t::PendingGenerated_t realEntry;
		realEntry.tex = pRealFrame;
		realEntry.seqNo = 0;
		realEntry.frameId = g_framegenHistory.currentFrameId;
		realEntry.phase = 1.0f;
		realEntry.bReal = true;
		g_framegenHistory.pending.push_back( std::move( realEntry ) );
		g_framegenHistory.bBidirQueuedReal = true;
		return;
	}

	// Any leftover pending frames belong to an older prediction; drop them before
	// queuing this interval's batch (normally already done by the supersede path
	// in the present decision).
	framegen_submit_batch( 1, nGapVblanks, nGenerate, eff, ulCompositeSeqNo, nMaxDegradeSteps, true );
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
		compositeImage = partial ? g_output.outputImagesPartialOverlay[ g_output.nOutImage ] : g_output.outputImages[ g_output.nOutImage ];

	auto cmdBuffer = pInCommandBuffer ? std::move( pInCommandBuffer ) : g_device.commandBuffer();

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

	uint64_t sequence = g_device.submit(std::move(cmdBuffer));

	// Submitted separately, after the composite: the real frame's present
	// waits on `sequence` only and is never delayed by framegen work.
	// The pPipewireTexture slot is only used by screenshot-style side
	// composites (streaming captures run their own vulkan_screenshot pass),
	// which use screenshot color management and must not enter history.
	if ( !GetBackend()->UsesVulkanSwapchain() && !partial && pPipewireTexture == nullptr && pOutputOverride == nullptr )
		framegen_record_real_frame( compositeImage, frameInfo, sequence );

	if ( !GetBackend()->UsesVulkanSwapchain() && pOutputOverride == nullptr && increment )
	{
		// Remember the slot we just composited into before advancing off it; the
		// skip below means it is not always nOutImage-1.
		g_output.nLastOutImage = g_output.nOutImage;

		const uint32_t nRing = g_output.outputImages.size();
		uint32_t nNext = ( g_output.nOutImage + 1 ) % nRing;

		// While framegen holds output-ring slots as zero-copy history, never
		// advance onto them: an overlay-only repaint (which still composites and
		// increments) would otherwise recomposite a slot still referenced as
		// previousReal/currentReal — overwriting history, or on the dedicated
		// framegen queue write-after-read racing a generation still reading it.
		// A slot an in-flight batch is still sampling (genReadA/genReadB) is
		// pinned the same way until that batch signals genReadSeqNo, covering the
		// window where history has been invalidated but the read is still running.
		// Normal generation pins at most two slots. The opt-in E2 held-out probe
		// temporarily pins A/B/C, still leaving two slots in the five-image ring;
		// the bound guards against spinning if that ever changes. When framegen is
		// off every pointer is null and this is the old advance.
		const CVulkanTexture *pCur = g_framegenHistory.currentReal.get();
		const CVulkanTexture *pPrev = g_framegenHistory.previousReal.get();
		const bool bGenReadInFlight = g_framegenHistory.genReadSeqNo != 0
			&& !g_device.hasCompletedFramegen( g_framegenHistory.genReadSeqNo );
		const CVulkanTexture *pReadA = bGenReadInFlight ? g_framegenHistory.genReadA.get() : nullptr;
		const CVulkanTexture *pReadB = bGenReadInFlight ? g_framegenHistory.genReadB.get() : nullptr;
		const CVulkanTexture *pReadReference = bGenReadInFlight ? g_framegenHistory.genReadReference.get() : nullptr;
		// E2's held-out middle frame must also stay immutable while it waits for
		// the third real frame that closes the A/B/C probe interval.
		const CVulkanTexture *pProbeReference = g_framegenColorProbe.reference.get();
		// Bidir (B3): a real frame queued for a delayed flip references a ring
		// slot too; recompositing into it before it scans out would tear the
		// frame the user is about to see. In steady state it aliases
		// previousReal/currentReal (no extra pinned slot); only transiently —
		// e.g. a scene invalidation keeping queued reals alive — is it distinct.
		auto fnPinnedByQueuedReal = []( const CVulkanTexture *pSlot ) -> bool
		{
			for ( const FramegenHistory_t::PendingGenerated_t &entry : g_framegenHistory.pending )
			{
				if ( entry.bReal && entry.tex.get() == pSlot )
					return true;
			}
			return false;
		};
		for ( uint32_t i = 0; ( pCur || pPrev || pReadA || pReadB || pReadReference || pProbeReference || !g_framegenHistory.pending.empty() ) && i < nRing; i++ )
		{
			const CVulkanTexture *pSlot = g_output.outputImages[ nNext ].get();
			if ( pSlot != pCur && pSlot != pPrev && pSlot != pReadA && pSlot != pReadB
				&& pSlot != pReadReference && pSlot != pProbeReference && !fnPinnedByQueuedReal( pSlot ) )
				break;
			nNext = ( nNext + 1 ) % nRing;
		}

		g_output.nOutImage = nNext;
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
