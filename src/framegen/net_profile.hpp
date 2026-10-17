#pragma once

#include "net_layout.hpp"
#include "numeric.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <optional>
#include <span>
#include <vector>

namespace gamescope::framegen
{

// A profile keeps one weight set per input encoding: the net reads raw stored
// values, so gamma code values, PQ code values and scRGB floats are different
// inputs to it. The key is the identity of the producer path, not a guess at
// what the values mean: the history image (composited output or the client's
// base layer), the base layer's declared colorspace, and the composite's
// effective output transfer (0 = Gamma22, 1 = PQ, 2 = none — colour management
// off). The same path always yields the same key. Algorithms share a set: they
// see the same features, and the train shader already confines each mode to
// the outputs it serves.
[[nodiscard]] constexpr uint32_t net_input_key( bool bBaseLayer, uint32_t baseColorspace, uint32_t outputTf )
{
	return bBaseLayer ? ( 1u << 8 ) | ( baseColorspace << 4 ) | outputTf
		: ( 0xFu << 4 ) | outputTf;
}

inline void net_input_key_name( uint32_t key, char *buf, size_t size )
{
	constexpr const char *colorspaces[] = { "linear", "srgb", "scrgb", "hdr10_pq", "passthru" };
	constexpr const char *transfers[] = { "gamma22", "pq", "none" };
	const uint32_t cs = ( key >> 4 ) & 0xFu;
	const uint32_t tf = key & 0xFu;
	if ( key == net_input_key( false, 0u, tf ) && tf < 3u )
		std::snprintf( buf, size, "output tf=%s", transfers[ tf ] );
	else if ( cs < 5u && tf < 3u && key == net_input_key( true, cs, tf ) )
		std::snprintf( buf, size, "base cs=%s tf=%s", colorspaces[ cs ], transfers[ tf ] );
	else
		std::snprintf( buf, size, "unknown key=0x%08x", key );
}

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

// Per-game profile (GSFP v1): magic, version, set count, then per set a u32
// key and one complete GSFR blob at the current semantic version. There is no
// migration: any other content (an older GSFR profile, another version, a
// damaged file) is incompatible, so the session starts a fresh profile and the
// first save replaces the file. Bare GSFR blobs remain the offline trainer's
// format for GAMESCOPE_FRAMEGEN_NET, which is not a profile.
inline constexpr uint32_t k_uFramegenNetProfileMagic = 0x50465347u; // 'GSFP'
inline constexpr uint32_t k_uFramegenNetProfileVersion = 1u;
inline constexpr size_t k_uFramegenNetProfileMaxSets = 8u;
inline constexpr size_t k_uFramegenNetBlobBytes =
	( k_uFramegenNetProfileMetadataWords + k_uFramegenNetFloats ) * sizeof( uint32_t );
inline constexpr size_t k_uFramegenNetProfileMaxBytes =
	12u + k_uFramegenNetProfileMaxSets * ( 4u + k_uFramegenNetBlobBytes );

struct NetProfileSet
{
	uint32_t key;
	std::vector<float> weights;
};

// The file is little-endian on every host.
[[nodiscard]] inline uint32_t net_profile_read_word( std::span<const uint8_t> bytes, size_t offset )
{
	return uint32_t( bytes[ offset ] ) | ( uint32_t( bytes[ offset + 1u ] ) << 8 )
		| ( uint32_t( bytes[ offset + 2u ] ) << 16 ) | ( uint32_t( bytes[ offset + 3u ] ) << 24 );
}

inline void net_profile_append_word( std::vector<uint8_t> &bytes, uint32_t word )
{
	for ( uint32_t shift = 0u; shift < 32u; shift += 8u )
		bytes.push_back( uint8_t( word >> shift ) );
}

// Returns nullopt for anything but a well-formed current profile: wrong magic
// or version, a set count outside 1..8, a duplicate key, a wrong length, an
// embedded blob that is not the current GSFR, or a non-finite weight.
[[nodiscard]] inline std::optional<std::vector<NetProfileSet>> decode_net_profile( std::span<const uint8_t> bytes )
{
	if ( bytes.size() < 12u
		|| net_profile_read_word( bytes, 0u ) != k_uFramegenNetProfileMagic
		|| net_profile_read_word( bytes, 4u ) != k_uFramegenNetProfileVersion )
		return std::nullopt;
	const uint32_t count = net_profile_read_word( bytes, 8u );
	if ( count == 0u || count > k_uFramegenNetProfileMaxSets
		|| bytes.size() != 12u + count * ( 4u + k_uFramegenNetBlobBytes ) )
		return std::nullopt;

	std::vector<NetProfileSet> sets;
	size_t offset = 12u;
	for ( uint32_t i = 0u; i < count; i++ )
	{
		const uint32_t key = net_profile_read_word( bytes, offset );
		offset += 4u;
		if ( std::any_of( sets.begin(), sets.end(),
			[&]( const NetProfileSet &set ) { return set.key == key; } ) )
			return std::nullopt;
		NetProfileMetadata metadata;
		for ( uint32_t &word : metadata )
		{
			word = net_profile_read_word( bytes, offset );
			offset += 4u;
		}
		if ( net_profile_metadata_version( metadata ) != k_uFramegenNetVersion )
			return std::nullopt;
		std::vector<float> weights( k_uFramegenNetFloats );
		for ( float &weight : weights )
		{
			weight = std::bit_cast<float>( net_profile_read_word( bytes, offset ) );
			offset += 4u;
		}
		if ( !validate_and_migrate_net_profile_weights( k_uFramegenNetVersion, weights ) )
			return std::nullopt;
		sets.push_back( { key, std::move( weights ) } );
	}
	return sets;
}

[[nodiscard]] inline std::optional<std::vector<uint8_t>> encode_net_profile( std::span<const NetProfileSet> sets )
{
	if ( sets.empty() || sets.size() > k_uFramegenNetProfileMaxSets )
		return std::nullopt;
	std::vector<uint8_t> bytes;
	bytes.reserve( 12u + sets.size() * ( 4u + k_uFramegenNetBlobBytes ) );
	net_profile_append_word( bytes, k_uFramegenNetProfileMagic );
	net_profile_append_word( bytes, k_uFramegenNetProfileVersion );
	net_profile_append_word( bytes, uint32_t( sets.size() ) );
	for ( size_t i = 0u; i < sets.size(); i++ )
	{
		const NetProfileSet &set = sets[ i ];
		for ( size_t j = 0u; j < i; j++ )
		{
			if ( sets[ j ].key == set.key )
				return std::nullopt;
		}
		if ( set.weights.size() != k_uFramegenNetFloats
			|| !std::all_of( set.weights.begin(), set.weights.end(), is_finite_binary32 ) )
			return std::nullopt;
		net_profile_append_word( bytes, set.key );
		for ( const uint32_t word : net_profile_metadata() )
			net_profile_append_word( bytes, word );
		for ( const float weight : set.weights )
			net_profile_append_word( bytes, std::bit_cast<uint32_t>( weight ) );
	}
	return bytes;
}

// Prior for an activating key, first match wins: its own set, the explicit
// GAMESCOPE_FRAMEGEN_NET blob, or the neutral init — which the caller passes
// empty when there is nothing safe to serve (no online learning). Never another
// key's learned set.
[[nodiscard]] inline std::span<const float> net_profile_prior( uint32_t key,
	std::span<const NetProfileSet> sets, std::span<const float> blob, std::span<const float> neutral )
{
	for ( const NetProfileSet &set : sets )
	{
		if ( set.key == key )
			return set.weights;
	}
	return blob.empty() ? neutral : blob;
}

// Save when a committed set changed (a switch retired one) or training moved
// the served weights past the last write — not merely when weights are live,
// or an encoding switched away from at exit would lose its unsaved learning.
[[nodiscard]] constexpr bool net_profile_needs_save( uint64_t revision, uint64_t savedRevision,
	uint64_t progress, uint64_t savedProgress )
{
	return revision != savedRevision || progress != savedProgress;
}

static_assert( k_uFramegenNetProfileMetadataWords == 12u,
	"GSFR metadata must contain one header and three layer descriptors" );
static_assert( k_uFramegenNetShadingVersion <= k_uFramegenNetVersion,
	"GSFR shading semantics must be supported by the current profile version" );

} // namespace gamescope::framegen
