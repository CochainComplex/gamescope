#pragma once

#include <cstdint>

namespace gamescope::framegen
{

inline constexpr uint32_t k_uNvidiaPciVendorId = 0x10deu;

struct DispatchPolicy
{
	bool useFp16;
	bool pairUseFp16;
	bool useR16FLuma;
	bool motionSupported;
	bool preferDirectExtrapolate;
};

// Vulkan capabilities select arithmetic precision and intermediate formats.
// Direct-vs-LDS extrapolation is the only measured hardware strategy override:
// no feature bit describes texture-cache effectiveness. Unknown vendors retain
// the capability-based path until benchmark evidence justifies another entry.
[[nodiscard]] constexpr DispatchPolicy select_dispatch_policy( bool supportsShaderFloat16,
	bool floatOutput, bool supportsR16FSampledStorage,
	bool supportsRGBA16FSampledStorage, uint32_t pciVendorId )
{
	const bool bBaseFp16 = supportsShaderFloat16 && !floatOutput;
	const bool bPreferDirect = pciVendorId == k_uNvidiaPciVendorId;
	return {
		.useFp16 = bBaseFp16 && !bPreferDirect,
		// The direct-pair shader has not been benchmark-qualified. Preserve the
		// independently selected LDS pair precision when single-output is direct.
		.pairUseFp16 = bBaseFp16,
		.useR16FLuma = supportsR16FSampledStorage,
		.motionSupported = supportsRGBA16FSampledStorage,
		.preferDirectExtrapolate = bPreferDirect,
	};
}

} // namespace gamescope::framegen
