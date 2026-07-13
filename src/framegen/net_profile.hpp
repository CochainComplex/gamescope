#pragma once

#include "net_layout.hpp"
#include "numeric.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <span>

namespace gamescope::framegen
{

inline constexpr uint32_t k_uFramegenNetOldestVersion = 1u;
inline constexpr uint32_t k_uFramegenNetShadingVersion = 3u;
inline constexpr size_t k_uFramegenNetProfileMetadataWords =
	3u + static_cast<size_t>( k_uFramegenNetLayerCount ) * 3u;
using NetProfileMetadata = std::array<uint32_t, k_uFramegenNetProfileMetadataWords>;

// GSFR stores one fixed header followed by the complete layer shape table.
// Keeping construction and validation together makes the loader and writer use
// one fixed word contract instead of parallel literal loops.
[[nodiscard]] constexpr NetProfileMetadata net_profile_metadata(
	uint32_t version = k_uFramegenNetVersion )
{
	NetProfileMetadata metadata = {};
	metadata[ 0 ] = k_uFramegenNetMagic;
	metadata[ 1 ] = version;
	metadata[ 2 ] = k_uFramegenNetLayerCount;
	for ( uint32_t layer = 0; layer < k_uFramegenNetLayerCount; layer++ )
	{
		const size_t offset = 3u + static_cast<size_t>( layer ) * 3u;
		metadata[ offset + 0u ] = k_uFramegenNetLayerDims[ layer ][ 0 ];
		metadata[ offset + 1u ] = k_uFramegenNetLayerDims[ layer ][ 1 ];
		metadata[ offset + 2u ] = k_uFramegenNetKernelWidth;
	}
	return metadata;
}

// Returns zero for malformed or unsupported metadata. GSFR versions start at
// one, so zero is an unambiguous failure value for the renderer's load path.
[[nodiscard]] constexpr uint32_t net_profile_metadata_version(
	std::span<const uint32_t> metadata )
{
	if ( metadata.size() != k_uFramegenNetProfileMetadataWords
		|| metadata[ 0 ] != k_uFramegenNetMagic
		|| metadata[ 1 ] < k_uFramegenNetOldestVersion
		|| metadata[ 1 ] > k_uFramegenNetVersion
		|| metadata[ 2 ] != k_uFramegenNetLayerCount )
	{
		return 0u;
	}

	for ( uint32_t layer = 0; layer < k_uFramegenNetLayerCount; layer++ )
	{
		const size_t offset = 3u + static_cast<size_t>( layer ) * 3u;
		if ( metadata[ offset + 0u ] != k_uFramegenNetLayerDims[ layer ][ 0 ]
			|| metadata[ offset + 1u ] != k_uFramegenNetLayerDims[ layer ][ 1 ]
			|| metadata[ offset + 2u ] != k_uFramegenNetKernelWidth )
		{
			return 0u;
		}
	}
	return metadata[ 1 ];
}

// Validate all served parameters before the GPU can observe them. V1/V2 used
// the same tensor shape but left output channel four undefined, so migration
// preserves every established flow/confidence parameter and zeroes only that
// legacy shading row and bias.
[[nodiscard]] inline bool validate_and_migrate_net_profile_weights(
	uint32_t version, std::span<float> weights )
{
	if ( version < k_uFramegenNetOldestVersion || version > k_uFramegenNetVersion
		|| weights.size() != k_uFramegenNetFloats )
		return false;
	for ( const float weight : weights )
	{
		if ( !is_finite_binary32( weight ) )
			return false;
	}

	if ( version < k_uFramegenNetShadingVersion )
	{
		std::fill( weights.begin() + k_uFramegenNetShadingWeightBegin,
			weights.begin() + k_uFramegenNetLayer3BiasOffset, 0.0f );
		weights[ k_uFramegenNetShadingBias ] = 0.0f;
	}
	return true;
}

static_assert( k_uFramegenNetProfileMetadataWords == 12u,
	"GSFR metadata must contain one header and three layer descriptors" );
static_assert( k_uFramegenNetShadingVersion <= k_uFramegenNetVersion,
	"GSFR shading semantics must be supported by the current profile version" );

} // namespace gamescope::framegen
