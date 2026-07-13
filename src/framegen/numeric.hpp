#pragma once

#include <bit>
#include <cstdint>
#include <limits>

namespace gamescope::framegen
{

// Gamescope is compiled with -ffast-math, under which classification functions
// may be folded as though NaN and infinity cannot exist. Framegen consumes
// serialized and GPU-produced fp32 values, so inspect their IEEE-754 exponent.
[[nodiscard]] constexpr bool is_finite_binary32( float value )
{
	return ( std::bit_cast<uint32_t>( value ) & 0x7f800000u ) != 0x7f800000u;
}

static_assert( sizeof( float ) == sizeof( uint32_t )
	&& std::numeric_limits<float>::is_iec559,
	"framegen requires IEEE-754 binary32 host floats" );

} // namespace gamescope::framegen
