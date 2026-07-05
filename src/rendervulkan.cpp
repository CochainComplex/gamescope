// Initialize Vulkan and composite stuff with a compute queue

#include <cassert>
#include <fcntl.h>
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
#include "log.hpp"
#include "Utils/Process.h"

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
#include "cs_framegen_motion_luma.h"
#include "cs_framegen_motion_luma_rgba.h"
#include "cs_framegen_motion_luma_pair.h"
#include "cs_framegen_motion_luma_pair_rgba.h"
#include "cs_framegen_motion_match.h"
#include "cs_framegen_motion_warp.h"
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
			vk_errorf( check_res, #x " failed!" ); \
			abort(); \
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

	float queuePriorities[2] = { 1.0f, 1.0f };

	VkDeviceQueueGlobalPriorityCreateInfoEXT queueCreateInfoEXT = {
		.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_GLOBAL_PRIORITY_CREATE_INFO_EXT,
		.pNext = nullptr,
		.globalPriority = VK_QUEUE_GLOBAL_PRIORITY_REALTIME_EXT
	};

	// Request a second queue in the compositor's (compute) family for frame
	// generation, when framegen is enabled, the family exposes one, and it isn't
	// disabled. Vulkan has no per-queue global priority within a family, so both
	// queues inherit this create-info's REALTIME priority; the win is decoupling
	// — a slow generation on queue 1 can never block the next composite on queue
	// 0 of the in-order realtime queue. Gated on framegen so a session that never
	// uses it requests exactly the single REALTIME queue it did before.
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
	SHADER(FRAMEGEN_MOTION_LUMA, cs_framegen_motion_luma);
	SHADER(FRAMEGEN_MOTION_LUMA_RGBA, cs_framegen_motion_luma_rgba);
	SHADER(FRAMEGEN_MOTION_LUMA_PAIR, cs_framegen_motion_luma_pair);
	SHADER(FRAMEGEN_MOTION_LUMA_PAIR_RGBA, cs_framegen_motion_luma_pair_rgba);
	SHADER(FRAMEGEN_MOTION_MATCH, cs_framegen_motion_match);
	SHADER(FRAMEGEN_MOTION_WARP, cs_framegen_motion_warp);
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
	// feeding the deadline-driven degradation ladder. Only meaningful on the
	// dedicated queue (where a batch's timestamps aren't serialized behind the
	// composite); if the queue family can't timestamp, framegen simply runs
	// without the measurement and the ladder stays at full quality.
	if ( m_bHasFramegenQueue )
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
				m_flFramegenTimestampPeriodNs = physProps.limits.timestampPeriod;
				vk_log.infof( "frame generation: measuring GPU time via timestamp queries (period %.2f ns)", m_flFramegenTimestampPeriodNs );
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
	const std::array<VkSpecializationMapEntry, 7> specializationEntries = {{
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
	}};

	struct {
		uint32_t layerCount;
		uint32_t ycbcrMask;
		uint32_t debug;
		uint32_t blur_layer_count;
		uint32_t colorspace_mask;
		uint32_t output_eotf;
		uint32_t itm_enable;
	} specializationData = {
		.layerCount   = layerCount,
		.ycbcrMask    = ycbcrMask,
		.debug        = composite_debug,
		.blur_layer_count = blur_layer_count,
		.colorspace_mask = colorspace_mask,
		.output_eotf = output_eotf,
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

	if ( vulkan_framegen_is_enabled() )
	{
		pipelineInfos.emplace_back( PipelineInfo_t{ SHADER_TYPE_FRAMEGEN_BLEND, 1, 1, 1 } );
		pipelineInfos.emplace_back( PipelineInfo_t{ SHADER_TYPE_FRAMEGEN_EXTRAPOLATE, 1, 1, 1 } );
		pipelineInfos.emplace_back( PipelineInfo_t{ SHADER_TYPE_FRAMEGEN_EXTRAPOLATE_DIRECT, 1, 1, 1 } );
		pipelineInfos.emplace_back( PipelineInfo_t{ SHADER_TYPE_FRAMEGEN_EXTRAPOLATE_PAIR, 1, 1, 1 } );
		pipelineInfos.emplace_back( PipelineInfo_t{ SHADER_TYPE_FRAMEGEN_MOTION_LUMA_PAIR, 1, 1, 1 } );
		pipelineInfos.emplace_back( PipelineInfo_t{ SHADER_TYPE_FRAMEGEN_MOTION_LUMA_PAIR_RGBA, 1, 1, 1 } );
		pipelineInfos.emplace_back( PipelineInfo_t{ SHADER_TYPE_FRAMEGEN_MOTION_MATCH, 1, 1, 1 } );
		pipelineInfos.emplace_back( PipelineInfo_t{ SHADER_TYPE_FRAMEGEN_MOTION_WARP, 1, 1, 1 } );

		if ( m_bSupportsShaderFloat16 )
		{
			pipelineInfos.emplace_back( PipelineInfo_t{ SHADER_TYPE_FRAMEGEN_EXTRAPOLATE_FP16, 1, 1, 1 } );
			pipelineInfos.emplace_back( PipelineInfo_t{ SHADER_TYPE_FRAMEGEN_EXTRAPOLATE_PAIR_FP16, 1, 1, 1 } );
		}
	}

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
		if ( ( ( 1 << i ) & requiredTypeBits ) == 0 )
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

	std::vector<VkSemaphore> pSignalSemaphores;
	std::vector<uint64_t> ulSignalPoints;

	std::vector<VkPipelineStageFlags> uWaitStageFlags;
	std::vector<VkSemaphore> pWaitSemaphores;
	std::vector<uint64_t> ulWaitPoints;

	pSignalSemaphores.push_back( m_scratchTimelineSemaphore );
	ulSignalPoints.push_back( nextSeqNo );

	for ( auto &dep : cmdBuffer->GetExternalSignals() )
	{
		pSignalSemaphores.push_back( dep.pTimelineSemaphore->pVkSemaphore );
		ulSignalPoints.push_back( dep.ulPoint );
	}

	for ( auto &dep : cmdBuffer->GetExternalDependencies() )
	{
		pWaitSemaphores.push_back( dep.pTimelineSemaphore->pVkSemaphore );
		ulWaitPoints.push_back( dep.ulPoint );
		uWaitStageFlags.push_back( VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT );
	}

	VkTimelineSemaphoreSubmitInfo timelineInfo = {
		.sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO,
		// no need to ensure order of cmd buffer submission, we only have one queue
		.waitSemaphoreValueCount = static_cast<uint32_t>( ulWaitPoints.size() ),
		.pWaitSemaphoreValues = ulWaitPoints.data(),
		.signalSemaphoreValueCount = static_cast<uint32_t>( ulSignalPoints.size() ),
		.pSignalSemaphoreValues = ulSignalPoints.data(),
	};

	VkCommandBuffer rawCmdBuffer = cmdBuffer->rawBuffer();

	VkSubmitInfo submitInfo = {
		.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
		.pNext = &timelineInfo,
		.waitSemaphoreCount = static_cast<uint32_t>( pWaitSemaphores.size() ),
		.pWaitSemaphores = pWaitSemaphores.data(),
		.pWaitDstStageMask = uWaitStageFlags.data(),
		.commandBufferCount = 1,
		.pCommandBuffers = &rawCmdBuffer,
		.signalSemaphoreCount = static_cast<uint32_t>( pSignalSemaphores.size() ),
		.pSignalSemaphores = pSignalSemaphores.data(),
	};

	vk_check( vk.QueueSubmit( cmdBuffer->queue(), 1, &submitInfo, VK_NULL_HANDLE ) );

	return nextSeqNo;
}

uint64_t CVulkanDevice::submit( std::unique_ptr<CVulkanCmdBuffer> cmdBuffer)
{
	uint64_t nextSeqNo = submitInternal(cmdBuffer.get());
	m_pendingCmdBufs.emplace(nextSeqNo, std::move(cmdBuffer));
	return nextSeqNo;
}

void CVulkanDevice::garbageCollect( void )
{
	uint64_t currentSeqNo;
	vk_check( vk.GetSemaphoreCounterValue(device(), m_scratchTimelineSemaphore, &currentSeqNo) );

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
	{
		waitFramegen( m_framegenSeqNo );
		framegenGarbageCollect();
	}
}

bool CVulkanDevice::hasCompleted(uint64_t sequence)
{
	uint64_t currentSeqNo = 0;
	vk_check( vk.GetSemaphoreCounterValue(device(), m_scratchTimelineSemaphore, &currentSeqNo) );
	return currentSeqNo >= sequence;
}

// Submit frame-generation work. With a dedicated framegen queue, this runs on
// the second queue: it waits on the composite's scratch-timeline value (so it
// never reads a half-written output image) and signals its own framegen
// timeline, meaning a slow generation can never sit in front of the next
// composite on the realtime queue. Without a second queue it forwards to the
// normal shared-queue submit, where in-order execution provides the same
// ordering (at the cost of the head-of-line blocking this feature removes).
uint64_t CVulkanDevice::submitFramegen( std::unique_ptr<CVulkanCmdBuffer> cmdBuffer, uint64_t ulWaitCompositeSeqNo, int nQuerySlot, uint32_t nLadderRung, uint32_t nGeneratedCount )
{
	if ( !m_bHasFramegenQueue )
		return submit( std::move( cmdBuffer ) );

	CVulkanCmdBuffer *pRaw = cmdBuffer.get();
	pRaw->end();

	const uint64_t nextSeqNo = ++m_framegenSeqNo;

	VkSemaphore waitSemaphores[1] = { m_scratchTimelineSemaphore };
	uint64_t waitValues[1] = { ulWaitCompositeSeqNo };
	VkPipelineStageFlags waitStages[1] = { VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT };

	VkSemaphore signalSemaphores[1] = { m_framegenTimeline->pVkSemaphore };
	uint64_t signalValues[1] = { nextSeqNo };

	const bool bWait = ulWaitCompositeSeqNo != 0;

	VkTimelineSemaphoreSubmitInfo timelineInfo = {
		.sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO,
		.waitSemaphoreValueCount = bWait ? 1u : 0u,
		.pWaitSemaphoreValues = bWait ? waitValues : nullptr,
		.signalSemaphoreValueCount = 1,
		.pSignalSemaphoreValues = signalValues,
	};

	VkCommandBuffer rawCmdBuffer = pRaw->rawBuffer();
	VkSubmitInfo submitInfo = {
		.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
		.pNext = &timelineInfo,
		.waitSemaphoreCount = bWait ? 1u : 0u,
		.pWaitSemaphores = bWait ? waitSemaphores : nullptr,
		.pWaitDstStageMask = bWait ? waitStages : nullptr,
		.commandBufferCount = 1,
		.pCommandBuffers = &rawCmdBuffer,
		.signalSemaphoreCount = 1,
		.pSignalSemaphores = signalSemaphores,
	};

	vk_check( vk.QueueSubmit( m_framegenQueue, 1, &submitInfo, VK_NULL_HANDLE ) );

	m_pendingFramegenCmdBufs.emplace( nextSeqNo, std::move( cmdBuffer ) );
	// Associate the timestamp ring slot (if any), ladder rung, and generated-count
	// with its seqNo, so readback can attribute measured cost to the exact batch
	// shape. A rung's cost is not stable across x2-like and x4-like gaps.
	if ( nQuerySlot >= 0 )
		m_framegenQuerySlotBySeqNo.emplace( nextSeqNo, FramegenQueryAssoc_t{ (uint32_t)nQuerySlot, nLadderRung, nGeneratedCount } );
	return nextSeqNo;
}

bool CVulkanDevice::hasCompletedFramegen( uint64_t sequence )
{
	if ( !m_bHasFramegenQueue )
		return hasCompleted( sequence );

	uint64_t currentSeqNo = 0;
	vk_check( vk.GetSemaphoreCounterValue( device(), m_framegenTimeline->pVkSemaphore, &currentSeqNo ) );
	return currentSeqNo >= sequence;
}

// Record the opening timestamp of a framegen batch. Rotates the query-pool ring,
// resets that slot's two queries, and writes a TOP_OF_PIPE timestamp. Returns the
// slot for framegenTimestampEnd / submitFramegen, or -1 when measurement is off.
int CVulkanDevice::framegenTimestampBegin( CVulkanCmdBuffer *pCmdBuffer )
{
	if ( m_framegenQueryPool == VK_NULL_HANDLE || m_uFramegenQueryRingDepth == 0 || pCmdBuffer == nullptr )
		return -1;

	const uint32_t nSlot = m_uFramegenQueryHead;
	m_uFramegenQueryHead = ( m_uFramegenQueryHead + 1 ) % m_uFramegenQueryRingDepth;

	VkCommandBuffer raw = pCmdBuffer->rawBuffer();
	vk.CmdResetQueryPool( raw, m_framegenQueryPool, nSlot * 2, 2 );
	vk.CmdWriteTimestamp( raw, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, m_framegenQueryPool, nSlot * 2 );
	return (int)nSlot;
}

// Record the closing timestamp of a framegen batch (BOTTOM_OF_PIPE). Must be
// called while the command buffer is still recording, before submitFramegen ends it.
void CVulkanDevice::framegenTimestampEnd( CVulkanCmdBuffer *pCmdBuffer, int nSlot )
{
	if ( nSlot < 0 || m_framegenQueryPool == VK_NULL_HANDLE || pCmdBuffer == nullptr )
		return;

	vk.CmdWriteTimestamp( pCmdBuffer->rawBuffer(), VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, m_framegenQueryPool, (uint32_t)nSlot * 2 + 1 );
}

void CVulkanDevice::framegenResetRungCosts()
{
	for ( uint32_t r = 0; r < kFramegenLadderSlots; r++ )
		for ( uint32_t c = 0; c < kFramegenGeneratedCountSlots; c++ )
			m_aFramegenRungCostNs[ r ][ c ] = 0;
}

void CVulkanDevice::waitFramegen( uint64_t sequence )
{
	if ( !m_bHasFramegenQueue )
	{
		wait( sequence, false );
		return;
	}

	VkSemaphoreWaitInfo waitInfo = {
		.sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO,
		.semaphoreCount = 1,
		.pSemaphores = &m_framegenTimeline->pVkSemaphore,
		.pValues = &sequence,
	};
	vk_check( vk.WaitSemaphores( device(), &waitInfo, ~0ull ) );
}

// Recycle framegen command buffers whose framegen-timeline seqNo has completed.
// Framegen keeps its own pending map because it signals a separate timeline; the
// buffers return to the shared unused pool (same command family).
void CVulkanDevice::framegenGarbageCollect()
{
	if ( !m_bHasFramegenQueue || m_pendingFramegenCmdBufs.empty() )
		return;

	uint64_t currentSeqNo = 0;
	vk_check( vk.GetSemaphoreCounterValue( device(), m_framegenTimeline->pVkSemaphore, &currentSeqNo ) );

	for ( auto it = m_pendingFramegenCmdBufs.begin(); it != m_pendingFramegenCmdBufs.end(); )
	{
		if ( it->first > currentSeqNo )
			break; // ordered by seqNo: nothing later has completed either

		// The seqNo has completed, so any timestamps this batch wrote are ready.
		// Read them back WITHOUT WAIT_BIT (never stall the compositor thread) and
		// fold into the rolling GPU-time stat that drives the degradation ladder.
		auto slotIt = m_framegenQuerySlotBySeqNo.find( it->first );
		if ( slotIt != m_framegenQuerySlotBySeqNo.end() )
		{
			const uint32_t nSlot = slotIt->second.nSlot;
			const uint32_t nRung = slotIt->second.nRung;
			const uint32_t nGeneratedCount = slotIt->second.nGeneratedCount;
			uint64_t ts[2] = { 0, 0 };
			const VkResult res = vk.GetQueryPoolResults( device(), m_framegenQueryPool, nSlot * 2, 2,
				sizeof( ts ), ts, sizeof( uint64_t ), VK_QUERY_RESULT_64_BIT );
			if ( res == VK_SUCCESS && ts[1] > ts[0] && nRung < kFramegenLadderSlots && nGeneratedCount < kFramegenGeneratedCountSlots )
			{
				const uint64_t ulGpuNs = (uint64_t)( double( ts[1] - ts[0] ) * m_flFramegenTimestampPeriodNs );
				m_ulFramegenLastRawGpuTimeNs = ulGpuNs;
				// Fold into this exact batch shape with a symmetric slow EMA (7/8).
				// A single anomalous batch must not shed quality for the whole scene,
				// and x2-like gaps must not inherit x4-like batch timings.
				uint64_t &ulRungCost = m_aFramegenRungCostNs[ nRung ][ nGeneratedCount ];
				ulRungCost = ( ulRungCost == 0 ) ? ulGpuNs : ( ulRungCost * 7 + ulGpuNs ) / 8;
			}
			m_framegenQuerySlotBySeqNo.erase( slotIt );
		}

		it->second->reset();
		m_unusedCmdBufs.push_back( std::move( it->second ) );
		it = m_pendingFramegenCmdBufs.erase( it );
	}
}

void CVulkanDevice::resetCmdBuffers(uint64_t sequence)
{
	auto last = m_pendingCmdBufs.find(sequence);
	if (last == m_pendingCmdBufs.end())
		return;

	for (auto it = m_pendingCmdBufs.begin(); ; it++)
	{
		it->second->reset();
		m_unusedCmdBufs.push_back(std::move(it->second));
		if (it == last)
			break;
	}

	m_pendingCmdBufs.erase(m_pendingCmdBufs.begin(), ++last);
}

CVulkanCmdBuffer::CVulkanCmdBuffer(CVulkanDevice *parent, VkCommandBuffer cmdBuffer, VkQueue queue, uint32_t queueFamily)
	: m_cmdBuffer(cmdBuffer), m_device(parent), m_queue(queue), m_queueFamily(queueFamily)
{
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
	m_useSrgb.reset();
}

template<class PushData, class... Args>
void CVulkanCmdBuffer::uploadConstants(Args&&... args)
{
	PushData data(std::forward<Args>(args)...);

	auto [ptr, offset] = m_device->uploadBufferData(sizeof(data));
	m_renderBufferOffset = offset;
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
	scratchDescriptor.range = VK_WHOLE_SIZE;

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
	auto result = m_textureState.emplace(image, TextureState());
	// no need to reimport if the image didn't change
	if (!result.second)
		return;
	result.first->second.needsImport = image->externalImage();
	result.first->second.needsExport = image->externalImage();
}

void CVulkanCmdBuffer::prepareDestImage(CVulkanTexture *image)
{
	auto result = m_textureState.emplace(image, TextureState());
	// no need to discard if the image is already image/in the correct layout
	if (!result.second)
		return;
	result.first->second.discarded = true;
	result.first->second.needsExport = image->externalImage();
	result.first->second.needsPresentLayout = image->outputImage();
}

void CVulkanCmdBuffer::discardImage(CVulkanTexture *image)
{
	auto result = m_textureState.emplace(image, TextureState());
	if (!result.second)
		return;
	result.first->second.discarded = true;
}

void CVulkanCmdBuffer::markDirty(CVulkanTexture *image)
{
	auto result = m_textureState.find(image);
	// image should have been prepared already
	assert(result !=  m_textureState.end());
	result->second.dirty = true;
}

void CVulkanCmdBuffer::insertBarrier(bool flush)
{
	std::vector<VkImageMemoryBarrier> barriers;

	uint32_t externalQueue = m_device->supportsModifiers() ? VK_QUEUE_FAMILY_FOREIGN_EXT : VK_QUEUE_FAMILY_EXTERNAL_KHR;

	VkImageSubresourceRange subResRange =
	{
		.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
		.levelCount = 1,
		.layerCount = 1
	};

	for (auto& pair : m_textureState)
	{
		CVulkanTexture *image = pair.first;
		TextureState& state = pair.second;
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

		barriers.push_back(memoryBarrier);

		state.discarded = false;
		state.dirty = false;
		state.needsImport = false;
	}

	// TODO replace VK_PIPELINE_STAGE_ALL_COMMANDS_BIT
	m_device->vk.CmdPipelineBarrier(m_cmdBuffer, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
									0, 0, nullptr, 0, nullptr, barriers.size(), barriers.data());
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

void vulkan_garbage_collect( void )
{
	g_device.garbageCollect();
	g_device.framegenGarbageCollect();
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

static const char *framegen_mode_name_of( GamescopeFramegenMode eMode )
{
	switch ( eMode )
	{
		case GamescopeFramegenMode::Extrapolate:
			return "extrapolate";
		case GamescopeFramegenMode::Blend:
			return "blend";
		case GamescopeFramegenMode::Motion:
			return "motion";
		default:
			return "unknown";
	}
}

static const char *framegen_mode_name()
{
	return framegen_mode_name_of( g_eFramegenMode );
}

// Push constants for the extrapolate shaders. `strength` is the effective
// per-slot forward coefficient: the CPU folds this slot's temporal placement
// (where it lands between the two real frames) and the user --framegen-strength
// into a single value, so the shader stays slot-agnostic.
struct FramegenPushData_t
{
	float strength;
	float suppressLo;
	float suppressHi;
	float pad;

	explicit FramegenPushData_t( float flStrength, float flLo, float flHi )
		: strength( flStrength )
		, suppressLo( flLo )
		, suppressHi( flHi )
		, pad( 0.0f )
	{
	}
};

// Paired extrapolation: two slot coefficients in one dispatch, so the two 4K
// history reads are shared instead of repeated per generated frame.
struct FramegenPairPushData_t
{
	float strength0;
	float strength1;
	float suppressLo;
	float suppressHi;

	FramegenPairPushData_t( float flStrength0, float flStrength1, float flLo, float flHi )
		: strength0( flStrength0 )
		, strength1( flStrength1 )
		, suppressLo( flLo )
		, suppressHi( flHi )
	{
	}
};

// Blend mode: per-slot temporal placement as the crossfade weight, so x3/x4
// produces distinct intermediate frames instead of identical 0.5 duplicates.
struct FramegenBlendPushData_t
{
	float phase;
	float pad0, pad1, pad2;

	explicit FramegenBlendPushData_t( float flPhase )
		: phase( flPhase ), pad0( 0.0f ), pad1( 0.0f ), pad2( 0.0f )
	{
	}
};

// Push constants for the motion-mode block-match and warp passes.
struct FramegenMotionMatchPush_t
{
	int32_t searchRadius;
	int32_t pad0, pad1, pad2;

	explicit FramegenMotionMatchPush_t( int32_t nSearchRadius )
		: searchRadius( nSearchRadius ), pad0( 0 ), pad1( 0 ), pad2( 0 )
	{
	}
};

struct FramegenMotionWarpPush_t
{
	float strength;
	float suppressLo;
	float suppressHi;
	float lowResScale;

	explicit FramegenMotionWarpPush_t( float flStrength, float flLo, float flHi, float flScale )
		: strength( flStrength ), suppressLo( flLo ), suppressHi( flHi ), lowResScale( flScale )
	{
	}
};

// Motion-estimation intermediates (low-resolution luma pyramids and the motion
// field). Allocated lazily by the motion dispatch, released on framegen reset.
struct FramegenMotionResources_t
{
	gamescope::OwningRc<CVulkanTexture> lumaPrev;
	gamescope::OwningRc<CVulkanTexture> lumaCur;
	gamescope::OwningRc<CVulkanTexture> mvField;
	uint32_t width = 0;
	uint32_t height = 0;
	uint32_t lumaFormat = DRM_FORMAT_INVALID;
};
static FramegenMotionResources_t g_framegenMotion;
static constexpr uint32_t k_uFramegenMotionDownscale = 8;

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
	uint64_t genReadSeqNo = 0;

	// Generated frames waiting for their empty vblanks, presented front-first,
	// one per vblank. Depth is multiplier-1 (x2 -> 1, x4 -> 3). Drained
	// wholesale the moment a real frame supersedes them.
	struct PendingGenerated_t
	{
		gamescope::Rc<CVulkanTexture> tex;
		uint64_t seqNo = 0;
		uint64_t frameId = 0;
		float phase = 0.0f; // fraction of the real-frame interval, for logs
	};
	std::vector<PendingGenerated_t> pending;

	uint32_t width = 0;
	uint32_t height = 0;
	uint32_t drmFormat = DRM_FORMAT_INVALID;
	uint64_t previousFrameId = 0;
	uint64_t currentFrameId = 0;
	uint64_t previousPresentTimeNs = 0;
	uint64_t currentPresentTimeNs = 0;
	uint64_t lastCompositeSeqNo = 0;
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
	// Scene fingerprint: prediction across a layer-count or output-encoding
	// change would smear the previous scene over the new one.
	int nLastLayerCount = -1;
	EOTF eLastEOTF = EOTF_Count;
	// Consecutive real frames slow enough to leave an empty vblank to fill.
	uint32_t nStableFrames = 0;
	// Deadline-driven degradation ladder (#04) position: 0 = full startup config,
	// each step sheds work (motion->extrapolate, then a multiplier notch). Never
	// reaches "stop generating" — that is left to the reactive pacing gate below.
	// Monotonic within a scene (only ever increases); reset to 0 on a scene change.
	// nDegradeHold is a post-step cooldown so the new rung's cost folds into the
	// measurement before the next step decision.
	uint32_t nDegradeSteps = 0;
	uint32_t nDegradeHold = 0;
	bool valid = false;
};

static FramegenHistory_t g_framegenHistory;
static bool g_bLoggedFramegenConfig = false;
static constexpr uint64_t k_ulFramegenMaxRealFrameGapNs = 250'000'000ull;
// Hysteresis for the fallback/shared-queue pacing gate. Real heavy-load frame
// pacing is often noisy: a game can leave empty vblanks overall while still
// occasionally landing a real frame just under the threshold. Treat this as a
// leaky confidence score instead of demanding a long run of consecutive slow
// frames. On a dedicated framegen queue we go further and generate
// speculatively; the output is disposable if a real frame arrives in time.
static constexpr uint32_t k_uFramegenStableFramesRequired = 4;
static constexpr uint32_t k_uFramegenStableFramesGain = 2;
// Drain per non-generatable frame. Keep this lower than the gain so one jittered
// interval does not erase several valid empty-vblank opportunities.
static constexpr uint32_t k_uFramegenStableFramesLeak = 1;

// Deadline for the degradation ladder (#04): when a rung's measured GPU cost
// exceeds this fraction of the vblank interval, shed a rung. The margin below
// 100% is proactive headroom against per-batch jitter, so a batch whose mean sits
// here rarely spikes past the true vblank and drops its generated frame.
static constexpr uint64_t k_uFramegenDeadlinePercent = 85; // > this % of a vblank interval -> degrade
// Frames to hold after a ladder step before stepping again, so the new rung's real
// cost folds into the measurement first (the readback lags by ~1 batch). Without
// it the loop would step again on the not-yet-updated rung cost and over-degrade.
static constexpr uint32_t k_uFramegenDegradeHoldFrames = 4;
// Dedicated-queue idle refill may extrapolate beyond the originally expected
// next real frame if the game stalls. Cap the forward step so a long stall does
// not run prediction unbounded; after this point additional refills converge
// toward the capped prediction instead of accelerating away from real content.
static constexpr float k_flFramegenMaxForwardStrength = 1.5f;

static bool framegen_refill_idle();

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

// Deadline-driven degradation ladder (#04). The startup mode/multiplier are the
// quality CEILING; under GPU-time pressure the ladder sheds work one rung at a
// time. It is monotonic within a scene (only ever degrades); quality is re-probed
// from this ceiling only on a scene change (invalidate_history). It NEVER mutates
// the base globals (g_eFramegenMode / g_nFramegenMultiplier): those gate the whole
// feature (vulkan_framegen_is_enabled) and size the output pool, so the ladder only
// ever degrades within them and reads its result through framegen_effective_config.
struct FramegenEffective_t
{
	GamescopeFramegenMode mode;
	uint32_t multiplier;
};

// Total number of rungs available from the startup config: one to drop
// motion->extrapolate (only if configured for motion), plus one per multiplier
// notch down to x2. Deliberately does NOT include a "stop generating" rung: the
// ladder always keeps generating (at worst the cheapest config), so its GPU-time
// input never starves. The genuine "even the cheapest config overruns" case is
// left to the existing reactive pacing gate (bGpuHasHeadroom + nStableFrames),
// which is driven by live per-frame signals and so self-heals, unlike a
// measurement-frozen ladder rung would.
static uint32_t framegen_max_degrade_steps()
{
	const uint32_t nMotionRung = ( g_eFramegenMode == GamescopeFramegenMode::Motion ) ? 1u : 0u;
	const uint32_t nMultiplierRungs = (uint32_t)std::max( 0, g_nFramegenMultiplier - 2 );
	return nMotionRung + nMultiplierRungs;
}

// Apply nDegradeSteps degradations to the startup ceiling. Motion (the priciest
// pass) is shed first, then the multiplier is stepped down to x2. nDegradeSteps
// is always clamped to framegen_max_degrade_steps(), so n is fully consumed and
// the result is always a still-generating config (never dormant).
static FramegenEffective_t framegen_effective_config( uint32_t nDegradeSteps )
{
	FramegenEffective_t eff = { g_eFramegenMode, (uint32_t)std::max( 2, g_nFramegenMultiplier ) };
	uint32_t n = nDegradeSteps;

	if ( n > 0 && eff.mode == GamescopeFramegenMode::Motion )
	{
		eff.mode = GamescopeFramegenMode::Extrapolate;
		n--;
	}
	while ( n > 0 && eff.multiplier > 2 )
	{
		eff.multiplier--;
		n--;
	}

	return eff;
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
	g_framegenHistory.pending.clear();
	g_framegenHistory.nStableFrames = 0;
	g_framegenHistory.previousPresentTimeNs = 0;
	g_framegenHistory.currentPresentTimeNs = 0;
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
	g_framegenHistory.pLastBaseTexture = nullptr;
	// Release the retained output-ring slots so a ring rebuild is never blocked
	// by history holding a reference to an old image, and so the next real frame
	// re-primes cleanly.
	g_framegenHistory.previousReal = nullptr;
	g_framegenHistory.currentReal = nullptr;
}

void vulkan_framegen_reset( const char *reason )
{
	if ( g_bFramegenDebug )
		vk_log.infof( "framegen: reset history reason=%s", reason ? reason : "unknown" );

	g_framegenHistory = {};
	// The per-rung costs live on g_device and survive the history reset; forget
	// them too (as invalidate_history does) so the monotonic ladder re-probes the
	// new workload from full quality instead of stepping on the old scene's stale
	// over-deadline measurements.
	g_device.framegenResetRungCosts();
	g_output.framegenOutputImages.clear();
	g_framegenMotion = {};
}

bool vulkan_framegen_has_pending_generated_frame()
{
	return vulkan_framegen_is_enabled() && !g_framegenHistory.pending.empty();
}

bool vulkan_framegen_generated_frame_ready()
{
	if ( !vulkan_framegen_has_pending_generated_frame() )
		return false;
	// Peek the front slot's completion without consuming it, so the present
	// decision can choose a hardware repeat over a wasted full recomposite when
	// generation hasn't finished by its vblank.
	return g_device.hasCompletedFramegen( g_framegenHistory.pending.front().seqNo );
}

gamescope::Rc<CVulkanTexture> vulkan_framegen_consume_generated_frame()
{
	if ( !vulkan_framegen_has_pending_generated_frame() )
		return nullptr;

	FramegenHistory_t::PendingGenerated_t front = g_framegenHistory.pending.front();

	// Generation was submitted in its own command buffer (and, when a dedicated
	// framegen queue exists, on its own queue) so the real frame's present never
	// waited on it. It normally completes well before the vblank it fills; if the
	// GPU is behind, presenting now would stall scanout on compute work, so drop
	// this frame instead — the display simply repeats the last scanned-out frame.
	// Never stall the present path for a generated frame.
	if ( !g_device.hasCompletedFramegen( front.seqNo ) )
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

	static uint64_t s_uPresentedDebugLogCounter = 0;
	if ( FramegenDebugShouldLog( s_uPresentedDebugLogCounter ) )
		vk_log.infof( "framegen: presented generated frame id=%" PRIu64 ".%02u", front.frameId, (unsigned)( front.phase * 100.0f ) );

	if ( g_framegenHistory.pending.empty() )
		framegen_refill_idle();

	return front.tex;
}

void vulkan_framegen_discard_generated_frame( const char *reason )
{
	if ( g_framegenHistory.pending.empty() )
		return;

	static uint64_t s_uDiscardDebugLogCounter = 0;
	if ( FramegenDebugShouldLog( s_uDiscardDebugLogCounter ) )
		vk_log.infof( "framegen: discarded %zu generated frame(s) reason=%s",
			g_framegenHistory.pending.size(), reason ? reason : "unknown" );

	// A real frame supersedes the whole batch: every queued generated frame is a
	// stale prediction now and would inject a vblank of latency if shown after the
	// real frame. Drop them all.
	g_framegenHistory.pending.clear();
}

static bool framegen_create_output_texture( gamescope::OwningRc<CVulkanTexture> *ppTexture, uint32_t width, uint32_t height, uint32_t drmFormat )
{
	CVulkanTexture::createFlags createFlags;
	createFlags.bFlippable = true;
	createFlags.bStorage = true;
	createFlags.bOutputImage = true;

	*ppTexture = new CVulkanTexture();
	return ( *ppTexture )->BInit( width, height, 1u, drmFormat, createFlags );
}

static bool framegen_ensure_resources( uint32_t width, uint32_t height, uint32_t drmFormat )
{
	if ( g_framegenHistory.width != width || g_framegenHistory.height != height || g_framegenHistory.drmFormat != drmFormat )
	{
		vulkan_framegen_reset( "resize_or_format_change" );
		g_framegenHistory.width = width;
		g_framegenHistory.height = height;
		g_framegenHistory.drmFormat = drmFormat;
	}

	// Generated-frame scanout pool: 2*multiplier distinct images so the
	// (multiplier-1) frames in flight plus any still being scanned out never
	// alias. History (previousReal/currentReal) is NOT allocated here — it is
	// retained by reference from the output ring in framegen_record_real_frame.
	const size_t nPool = (size_t)2 * (size_t)g_nFramegenMultiplier;
	if ( g_output.framegenOutputImages.size() != nPool )
		g_output.framegenOutputImages.resize( nPool );

	for ( auto &pImage : g_output.framegenOutputImages )
	{
		if ( framegen_output_matches( pImage, width, height, drmFormat ) )
			continue;

		if ( !framegen_create_output_texture( &pImage, width, height, drmFormat ) )
		{
			vulkan_framegen_reset( "generated_allocation_failed" );
			return false;
		}
	}

	return true;
}

static bool framegen_is_float_drm_format( uint32_t drmFormat )
{
	// Float (scRGB) targets carry HDR highlights above 1.0 and wide-gamut
	// negatives; fp16 arithmetic can band those, so the extrapolate shader stays
	// fp32 for them (see the fp16 shader's precision note).
	switch ( drmFormat )
	{
		case DRM_FORMAT_ABGR16161616F:
		case DRM_FORMAT_XBGR16161616F:
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

static bool framegen_create_intermediate( gamescope::OwningRc<CVulkanTexture> *ppTexture, uint32_t width, uint32_t height, uint32_t drmFormat )
{
	CVulkanTexture::createFlags createFlags;
	createFlags.bStorage = true;
	createFlags.bSampled = true;

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

	// fp16 is used on capable hardware except for float (scRGB) targets; see
	// framegen_is_float_drm_format.
	dispatch.useFp16 = g_device.supportsShaderFloat16() && !framegen_is_float_drm_format( drmFormat );
	dispatch.extrapolate = dispatch.useFp16 ? SHADER_TYPE_FRAMEGEN_EXTRAPOLATE_FP16 : SHADER_TYPE_FRAMEGEN_EXTRAPOLATE;
	dispatch.extrapolatePair = dispatch.useFp16 ? SHADER_TYPE_FRAMEGEN_EXTRAPOLATE_PAIR_FP16 : SHADER_TYPE_FRAMEGEN_EXTRAPOLATE_PAIR;

	dispatch.useR16FLuma = framegen_format_supports_sampled_storage( DRM_FORMAT_R16F );
	dispatch.motionLumaFormat = dispatch.useR16FLuma ? DRM_FORMAT_R16F : DRM_FORMAT_ABGR16161616F;
	dispatch.motionLumaPair = dispatch.useR16FLuma ? SHADER_TYPE_FRAMEGEN_MOTION_LUMA_PAIR : SHADER_TYPE_FRAMEGEN_MOTION_LUMA_PAIR_RGBA;
	dispatch.motionSupported = framegen_format_supports_sampled_storage( DRM_FORMAT_ABGR16161616F );

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
	VkPhysicalDeviceProperties physProps = {};
	g_device.vk.GetPhysicalDeviceProperties( g_device.physDev(), &physProps );
	const bool bPreferDirectExtrapolate = ( physProps.vendorID == 0x10DE ); // NVIDIA, measured
	if ( bPreferDirectExtrapolate )
	{
		dispatch.useFp16 = false;
		dispatch.extrapolate = SHADER_TYPE_FRAMEGEN_EXTRAPOLATE_DIRECT;
		// The paired extrapolation shader (x3/x4 only) is left on its LDS variant
		// until the direct-pair path is measured.
	}

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

// Motion-compensated generation, part 1 (once per batch): build low-res luma for
// both real frames and block-match them into a motion field. This depends only
// on the two real frames — constant across every generated slot in the interval
// — so it is computed a single time and the field reused by each warp. Recorded
// into the batch command buffer; the dispatch path inserts the read-after-write
// barriers between passes. Returns false (fall back to extrapolation for the
// whole batch) if the intermediates can't be allocated.
static bool framegen_prepare_motion( CVulkanCmdBuffer *pCmdBuffer, uint32_t width, uint32_t height, const FramegenDispatch_t &dispatch )
{
	const uint32_t lowW = std::max( 1u, div_roundup( width, k_uFramegenMotionDownscale ) );
	const uint32_t lowH = std::max( 1u, div_roundup( height, k_uFramegenMotionDownscale ) );
	const uint32_t uFieldFormat = DRM_FORMAT_ABGR16161616F;

	if ( g_framegenMotion.width != lowW || g_framegenMotion.height != lowH
		|| g_framegenMotion.lumaFormat != dispatch.motionLumaFormat
		|| g_framegenMotion.lumaPrev == nullptr || g_framegenMotion.lumaCur == nullptr || g_framegenMotion.mvField == nullptr )
	{
		if ( !framegen_create_intermediate( &g_framegenMotion.lumaPrev, lowW, lowH, dispatch.motionLumaFormat )
			|| !framegen_create_intermediate( &g_framegenMotion.lumaCur, lowW, lowH, dispatch.motionLumaFormat )
			|| !framegen_create_intermediate( &g_framegenMotion.mvField, lowW, lowH, uFieldFormat ) )
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

	const uint32_t pg = 8;

	// Pass 1: downscale both real frames to low-res luma.
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

	// Pass 2: block match -> motion field (.rg = mv, .b = confidence).
	pCmdBuffer->bindPipeline( g_device.pipeline( SHADER_TYPE_FRAMEGEN_MOTION_MATCH ) );
	pCmdBuffer->bindTarget( g_framegenMotion.mvField );
	pCmdBuffer->bindTexture( 0, g_framegenMotion.lumaPrev );
	pCmdBuffer->setTextureSrgb( 0, false );
	pCmdBuffer->setSamplerUnnormalized( 0, false );
	pCmdBuffer->setSamplerNearest( 0, true );
	pCmdBuffer->bindTexture( 1, g_framegenMotion.lumaCur );
	pCmdBuffer->setTextureSrgb( 1, false );
	pCmdBuffer->setSamplerUnnormalized( 1, false );
	pCmdBuffer->setSamplerNearest( 1, true );
	pCmdBuffer->pushConstants<FramegenMotionMatchPush_t>( 4 );
	pCmdBuffer->dispatch( div_roundup( lowW, pg ), div_roundup( lowH, pg ) );

	return true;
}

// Motion-compensated generation, part 2 (per slot): warp the current frame
// forward along the shared motion field, blending back to extrapolation where
// the match is unconfident. Each slot writes a distinct target, so warps in the
// same batch don't hazard against each other.
static void framegen_warp_slot( CVulkanCmdBuffer *pCmdBuffer, const gamescope::Rc<CVulkanTexture> &pTarget, float flStrength )
{
	const uint32_t pg = 8;

	pCmdBuffer->bindPipeline( g_device.pipeline( SHADER_TYPE_FRAMEGEN_MOTION_WARP ) );
	pCmdBuffer->bindTarget( pTarget );
	pCmdBuffer->bindTexture( 0, g_framegenHistory.previousReal );
	pCmdBuffer->setTextureSrgb( 0, true );
	pCmdBuffer->setSamplerUnnormalized( 0, false );
	pCmdBuffer->setSamplerNearest( 0, false );
	pCmdBuffer->bindTexture( 1, g_framegenHistory.currentReal );
	pCmdBuffer->setTextureSrgb( 1, true );
	pCmdBuffer->setSamplerUnnormalized( 1, false );
	pCmdBuffer->setSamplerNearest( 1, false );
	pCmdBuffer->bindTexture( 2, g_framegenMotion.mvField );
	pCmdBuffer->setTextureSrgb( 2, false );
	pCmdBuffer->setSamplerUnnormalized( 2, false );
	pCmdBuffer->setSamplerNearest( 2, false );
	pCmdBuffer->pushConstants<FramegenMotionWarpPush_t>( flStrength, k_flFramegenSuppressLo, k_flFramegenSuppressHi, (float)k_uFramegenMotionDownscale );
	pCmdBuffer->dispatch( div_roundup( pTarget->width(), pg ), div_roundup( pTarget->height(), pg ) );
}

static bool framegen_submit_batch( uint32_t nFirstSlot, uint32_t nGapVblanks, uint32_t nGenerate, const FramegenEffective_t &eff, uint64_t ulCompositeSeqNo, uint32_t nMaxDegradeSteps, bool bClearPending )
{
	if ( nGenerate == 0 || nGapVblanks == 0 || ulCompositeSeqNo == 0 )
		return false;

	if ( bClearPending )
		g_framegenHistory.pending.clear();

	// Reserve this interval's output slots up front so an empty batch never
	// records/submits a command buffer.
	struct SlotPlan_t { gamescope::Rc<CVulkanTexture> tex; float phase; float strength; uint32_t slotIndex; };
	std::vector<SlotPlan_t> slots;
	slots.reserve( nGenerate );
	for ( uint32_t k = nFirstSlot; k < nFirstSlot + nGenerate; k++ )
	{
		const uint32_t idx = g_framegenHistory.nNextOutputIndex % g_output.framegenOutputImages.size();
		g_framegenHistory.nNextOutputIndex++;
		gamescope::Rc<CVulkanTexture> pGenerated = g_output.framegenOutputImages[ idx ];
		if ( !pGenerated )
			continue;

		const float phase = (float)k / (float)nGapVblanks;
		// Effective forward coefficient: temporal placement scaled by the user
		// strength (0.5 is neutral, reproducing the classic x2 half-way step).
		// Idle refill can move past the originally expected next-real slot when
		// the game stalls, but never lets prediction run away unbounded.
		const float flStrength = std::clamp( phase * ( g_flFramegenStrength / 0.5f ), 0.0f, k_flFramegenMaxForwardStrength );
		slots.push_back( { std::move( pGenerated ), phase, flStrength, k } );
	}

	if ( slots.empty() )
		return false;

	// Record the whole interval's generation into ONE command buffer submitted
	// once: the shared motion intermediates are serialized by the per-command-
	// buffer barrier tracking, all slots share a single framegen seqNo, and the
	// batch draws from the isolated framegen descriptor ring (see markFramegen).
	std::unique_ptr<CVulkanCmdBuffer> pCmdBuffer = g_device.commandBuffer();
	pCmdBuffer->markFramegen();
	// Bracket the batch's dispatches with GPU timestamps so its cost feeds the
	// degradation ladder next interval. No-op (returns -1) without timestamp support.
	const int nQuerySlot = g_device.framegenTimestampBegin( pCmdBuffer.get() );
	const FramegenDispatch_t &dispatch = framegen_dispatch_for_format( g_framegenHistory.drmFormat );

	// Motion estimation depends only on the two real frames, so it is computed
	// once for the batch; falls back to extrapolation for every slot if the
	// intermediates can't be allocated. The ladder's effective mode (not the base
	// global) selects the pass, so motion can be shed under GPU pressure without
	// disturbing vulkan_framegen_is_enabled() or the forced-composite tax.
	const bool bMotion = eff.mode == GamescopeFramegenMode::Motion
		&& dispatch.motionSupported
		&& framegen_prepare_motion( pCmdBuffer.get(), g_framegenHistory.currentReal->width(), g_framegenHistory.currentReal->height(), dispatch );

	if ( bMotion )
	{
		// Each warp reads the shared motion field and history; keep per-slot.
		for ( const SlotPlan_t &slot : slots )
			framegen_warp_slot( pCmdBuffer.get(), slot.tex, slot.strength );
	}
	else if ( eff.mode == GamescopeFramegenMode::Blend )
	{
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

	g_device.framegenTimestampEnd( pCmdBuffer.get(), nQuerySlot );
	// Attribute this batch's measured cost to the rung and generated-count it ran
	// at; batch cost scales with slot count, especially x3/x4 extrapolate pairs.
	const uint64_t ulSeqNo = g_device.submitFramegen( std::move( pCmdBuffer ), ulCompositeSeqNo, nQuerySlot, g_framegenHistory.nDegradeSteps, (uint32_t)slots.size() );

	for ( const SlotPlan_t &slot : slots )
	{
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

	// Pin this batch's input slots until it finishes reading them, so a later
	// composite can't reuse a slot the framegen queue is still sampling even
	// after history is invalidated. The oversubscription guard admits only one
	// batch at a time, so these always match the current (previousReal,
	// currentReal) while incomplete — at most two slots are ever pinned.
	g_framegenHistory.genReadA = g_framegenHistory.previousReal;
	g_framegenHistory.genReadB = g_framegenHistory.currentReal;
	g_framegenHistory.genReadSeqNo = ulSeqNo;

	const uint32_t nGeneratedCount = (uint32_t)slots.size();

	static uint64_t s_uGeneratedDebugLogCounter = 0;
	if ( FramegenDebugShouldLog( s_uGeneratedDebugLogCounter ) )
	{
		vk_log.infof( "framegen: generated %u frame(s) for real id=%" PRIu64 " gapVblanks=%u mode=%s(x%u) degrade=%u/%u gpu=%.2fms queue family %u",
			nGeneratedCount,
			g_framegenHistory.currentFrameId,
			nGapVblanks,
			framegen_mode_name_of( bMotion ? GamescopeFramegenMode::Motion : eff.mode ),
			eff.multiplier,
			g_framegenHistory.nDegradeSteps,
			nMaxDegradeSteps,
			g_device.framegenLastGpuTimeNs() / 1.0e6,
			g_device.queueFamily() );
	}

	force_repaint();
	return true;
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
	if ( flTimestampPeriodNs == 0.0 )
	{
		fprintf( stderr, "framegen-bench: device does not support timestamp queries\n" );
		return;
	}

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
			dTotalNs += double( ts[1] - ts[0] ) * flTimestampPeriodNs;
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
				framegen_prepare_motion( cmd.get(), res.nWidth, res.nHeight, dispatch );
				g_device.submit( std::move( cmd ) );
				g_device.waitIdle();
			}

			double msMotionPrep = timePass( [&]( CVulkanCmdBuffer *cmd ) {
				framegen_prepare_motion( cmd, res.nWidth, res.nHeight, dispatch );
			} );
			printf( "%-8s  %-26s  %10.3f\n", res.pszName, "motion setup (per real)", msMotionPrep );

			double msMotionWarp = timePass( [&]( CVulkanCmdBuffer *cmd ) {
				framegen_warp_slot( cmd, pOut, g_flFramegenStrength );
			} );
			printf( "%-8s  %-26s  %10.3f\n", res.pszName, "motion warp (per gen)", msMotionWarp );
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

static void framegen_record_real_frame( gamescope::Rc<CVulkanTexture> pRealFrame, const struct FrameInfo_t *pFrameInfo, uint64_t ulCompositeSeqNo )
{
	if ( !vulkan_framegen_is_enabled() || !pRealFrame || !pFrameInfo )
		return;

	if ( !g_bLoggedFramegenConfig )
	{
		vk_log.infof( "framegen: enabled mode=%s multiplier=%d%s", framegen_mode_name(), g_nFramegenMultiplier,
			g_device.hasFramegenQueue() ? " (dedicated queue)" : "" );
		vk_log.infof( "framegen: forcing composite path" );
		vk_log.infof( "framegen: adaptive sync (VRR) and tearing flips are suppressed while framegen is active" );
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
		static uint64_t s_uOverlayOnlyDebugLogCounter = 0;
		if ( FramegenDebugShouldLog( s_uOverlayOnlyDebugLogCounter ) )
			vk_log.infof( "framegen: ignoring overlay-only repaint (base content unchanged)" );
		return;
	}
	g_framegenHistory.pLastBaseTexture = pBaseTexture;

	if ( !framegen_ensure_resources( pRealFrame->width(), pRealFrame->height(), pRealFrame->drmFormat() ) )
		return;

	// A layer-count change (overlay appearing/vanishing) or an output-encoding
	// change (SDR<->HDR can flip the EOTF without changing the image format)
	// abruptly replaces scene content; drop the history rather than smear the
	// old scene over the new one for a frame.
	if ( g_framegenHistory.nLastLayerCount != pFrameInfo->layerCount || g_framegenHistory.eLastEOTF != pFrameInfo->outputEncodingEOTF )
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
		g_framegenHistory.lastCompositeSeqNo = ulCompositeSeqNo;
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
	const bool bLeavesEmptyVblank = ulPrevRealFrameTimeNs != 0
		&& now - ulPrevRealFrameTimeNs >= ( ulVblankIntervalNs * 3 ) / 2;
	const bool bGpuHasHeadroom = g_device.hasCompletedFramegen( g_framegenHistory.lastGeneratedSeqNo );
	const bool bGeneratable = bLeavesEmptyVblank && bGpuHasHeadroom;
	const bool bCanSpeculate = g_device.hasFramegenQueue() && bGpuHasHeadroom && ulPrevRealFrameTimeNs != 0;

	// Zero-copy history shift: the previous "current" becomes "previous", and the
	// just-composited output image becomes "current". We only hold references
	// into the output ring — the composite already wrote these images. We shift on
	// every kept real frame (even non-generatable ones) so the two most recent
	// reals stay fresh and a slowdown resumes generation without a re-prime.
	g_framegenHistory.previousReal = g_framegenHistory.currentReal;
	g_framegenHistory.previousFrameId = g_framegenHistory.currentFrameId - 1;
	g_framegenHistory.currentReal = pRealFrame;

	// Prime: the first frame after a reset/invalidation only establishes history.
	if ( !g_framegenHistory.valid )
	{
		static uint64_t s_uPrimeDebugLogCounter = 0;
		if ( FramegenDebugShouldLog( s_uPrimeDebugLogCounter ) )
			vk_log.infof( "framegen: priming history with real frame id=%" PRIu64, g_framegenHistory.currentFrameId );
		g_framegenHistory.valid = true;
		return;
	}

	// Leaky-bucket hysteresis (see above). Only score once we have a real gap to
	// judge (ulPrevRealFrameTimeNs != 0); the priming frame above never reaches here.
	if ( bGeneratable )
	{
		g_framegenHistory.nStableFrames = std::min( k_uFramegenStableFramesRequired,
			g_framegenHistory.nStableFrames + k_uFramegenStableFramesGain );
	}
	else
	{
		g_framegenHistory.nStableFrames = g_framegenHistory.nStableFrames > k_uFramegenStableFramesLeak
			? g_framegenHistory.nStableFrames - k_uFramegenStableFramesLeak : 0;
	}

	// Max degradation rungs for the startup config (0 for extrapolate/blend x2).
	// Needed here for the dormant log; the ladder itself is evaluated further down,
	// only on frames we actually generate, so it steps on real generating frames
	// and never on an idle/dormant frame.
	const uint32_t nMaxDegradeSteps = framegen_max_degrade_steps();

	const bool bReactiveReady = bGeneratable && g_framegenHistory.nStableFrames >= k_uFramegenStableFramesRequired;
	const bool bShouldGenerate = bCanSpeculate || bReactiveReady;

	if ( !bShouldGenerate )
	{
		static uint64_t s_uDormantDebugLogCounter = 0;
		if ( FramegenDebugShouldLog( s_uDormantDebugLogCounter ) )
		{
			const char *pszState = !bGpuHasHeadroom ? "busy" : ( bGeneratable ? "stabilizing" : "dormant" );
			vk_log.infof( "framegen: %s %u/%u degrade=%u/%u", pszState,
				g_framegenHistory.nStableFrames, k_uFramegenStableFramesRequired,
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
	const uint32_t nMeasuredGapVblanks = std::max( 1u,
		(uint32_t)( ( now - ulPrevRealFrameTimeNs + ulVblankIntervalNs / 2 ) / ulVblankIntervalNs ) );

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
	const uint32_t nLadderGapVblanks = bCanSpeculate
		? std::max( nMeasuredGapVblanks, std::max( 2u, curEffForLadder.multiplier ) )
		: nMeasuredGapVblanks;
	const uint32_t nGapSlots = nLadderGapVblanks > 1 ? nLadderGapVblanks - 1 : 0;
	const uint32_t nCurGenForLadder = std::min( nGapSlots, std::max( 1u, curEffForLadder.multiplier - 1u ) );
	const uint64_t ulCurRungCostNs = g_device.framegenRungCostNs( g_framegenHistory.nDegradeSteps, nCurGenForLadder );
	if ( nMaxDegradeSteps > 0 && ulCurRungCostNs != 0 && g_framegenHistory.nDegradeSteps < nMaxDegradeSteps )
	{
		const uint64_t ulDeadlineNs = ( ulVblankIntervalNs * k_uFramegenDeadlinePercent ) / 100;
		if ( g_framegenHistory.nDegradeHold > 0 )
		{
			// Cooldown after a step: keep generating (so the new rung's cost gets
			// measured) but hold off on another step until that fresh sample folds
			// in, otherwise we'd act on a not-yet-updated rung cost and overshoot.
			g_framegenHistory.nDegradeHold--;
		}
		else if ( ulCurRungCostNs > ulDeadlineNs )
		{
			// Over budget at the current rung. Only take the step if it actually
			// reduces work at THIS gap: dropping motion->extrapolate always lowers
			// per-frame cost, but a pure multiplier notch only helps when the gap
			// lets it generate fewer frames. Otherwise stepping would cost quality
			// (a coarser cadence / fewer inserted frames) for zero GPU saving.
			const FramegenEffective_t nextEff = framegen_effective_config( g_framegenHistory.nDegradeSteps + 1 );
			const uint32_t nNextGen = std::min( nGapSlots, std::max( 1u, nextEff.multiplier - 1u ) );
			if ( nextEff.mode != curEffForLadder.mode || nNextGen < nCurGenForLadder )
			{
				g_framegenHistory.nDegradeSteps++;
				g_framegenHistory.nDegradeHold = k_uFramegenDegradeHoldFrames;
			}
		}
	}

	const FramegenEffective_t eff = framegen_effective_config( g_framegenHistory.nDegradeSteps );

	// Fill as many empty vblanks as the measured gap actually offers, capped by
	// the multiplier: generate one fewer than the whole-vblank gap (the final slot
	// is the next real frame). The ladder's effective multiplier can lower this
	// ceiling below the startup one under GPU pressure; it is always
	// <= g_nFramegenMultiplier, so the pre-sized output pool holds.
	const uint32_t nGapVblanks = bCanSpeculate
		? std::max( nMeasuredGapVblanks, std::max( 2u, eff.multiplier ) )
		: nMeasuredGapVblanks;
	const uint32_t nSlotCeiling = std::max( 1u, eff.multiplier - 1u );
	uint32_t nGenerate = nGapVblanks > 1 ? std::min( nGapVblanks - 1, nSlotCeiling ) : 0;

	// Without a dedicated framegen queue the batch is submitted to the same
	// in-order queue as the next composite, so it sits in front of it. Cap the
	// single-queue path to one generated frame to bound that head-of-line work —
	// the proven x2-prototype behaviour — rather than amplifying it under x3/x4.
	if ( !g_device.hasFramegenQueue() )
		nGenerate = std::min( nGenerate, 1u );

	if ( nGenerate == 0 )
		return;

	// Any leftover pending frames belong to an older prediction; drop them before
	// queuing this interval's batch (normally already done by the supersede path
	// in the present decision).
	framegen_submit_batch( 1, nGapVblanks, nGenerate, eff, ulCompositeSeqNo, nMaxDegradeSteps, true );
}

std::optional<uint64_t> vulkan_composite( struct FrameInfo_t *frameInfo, gamescope::Rc<CVulkanTexture> pPipewireTexture, bool partial, gamescope::Rc<CVulkanTexture> pOutputOverride, bool increment, std::unique_ptr<CVulkanCmdBuffer> pInCommandBuffer )
{
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
		// At most two slots are pinned in a >=5 ring, so a free slot always
		// exists; the bound guards against spinning if that ever changed. When
		// framegen is off every pointer is null and this is the old advance.
		const CVulkanTexture *pCur = g_framegenHistory.currentReal.get();
		const CVulkanTexture *pPrev = g_framegenHistory.previousReal.get();
		const bool bGenReadInFlight = g_framegenHistory.genReadSeqNo != 0
			&& !g_device.hasCompletedFramegen( g_framegenHistory.genReadSeqNo );
		const CVulkanTexture *pReadA = bGenReadInFlight ? g_framegenHistory.genReadA.get() : nullptr;
		const CVulkanTexture *pReadB = bGenReadInFlight ? g_framegenHistory.genReadB.get() : nullptr;
		for ( uint32_t i = 0; ( pCur || pPrev || pReadA || pReadB ) && i < nRing; i++ )
		{
			const CVulkanTexture *pSlot = g_output.outputImages[ nNext ].get();
			if ( pSlot != pCur && pSlot != pPrev && pSlot != pReadA && pSlot != pReadB )
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
