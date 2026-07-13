#pragma once

#include "numeric.hpp"

#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <optional>
#include <string_view>

namespace gamescope::framegen
{

enum class FramegenColorProbeSweep : uint32_t
{
	Occlusion = 0,
	EndpointTrace = 1,
};

[[nodiscard]] inline std::optional<FramegenColorProbeSweep>
parse_color_probe_sweep_setting( const char *value )
{
	if ( value == nullptr || *value == '\0' )
		return std::nullopt;

	const std::string_view setting{ value };
	if ( setting == "occlusion" )
		return FramegenColorProbeSweep::Occlusion;
	if ( setting == "trace" || setting == "endpoint-trace" )
		return FramegenColorProbeSweep::EndpointTrace;
	return std::nullopt;
}

[[nodiscard]] inline const char *color_probe_sweep_name( FramegenColorProbeSweep sweep )
{
	switch ( sweep )
	{
		case FramegenColorProbeSweep::Occlusion: return "occlusion";
		case FramegenColorProbeSweep::EndpointTrace: return "endpoint-trace";
	}
	return "unknown";
}

[[nodiscard]] inline const char *non_empty_setting( const char *value )
{
	return value != nullptr && *value != '\0' ? value : nullptr;
}

// Parse the complete string and reject non-finite values explicitly. Keeping
// range policy at the call site makes clamp-vs-fallback behavior reviewable for
// each setting while sharing one strict lexical contract.
[[nodiscard]] inline std::optional<float> parse_finite_float_setting( const char *value )
{
	if ( value == nullptr || *value == '\0' )
		return std::nullopt;

	char *end = nullptr;
	const float parsed = std::strtof( value, &end );
	if ( end == value || *end != '\0' || !is_finite_binary32( parsed ) )
		return std::nullopt;
	return parsed;
}

[[nodiscard]] inline std::optional<uint32_t> parse_uint32_setting(
	const char *value, bool allowZero )
{
	if ( value == nullptr || *value == '\0' )
		return std::nullopt;

	errno = 0;
	char *end = nullptr;
	const unsigned long long parsed = std::strtoull( value, &end, 10 );
	if ( errno != 0 || end == value || *end != '\0'
		|| parsed > std::numeric_limits<uint32_t>::max()
		|| ( !allowZero && parsed == 0 ) )
	{
		return std::nullopt;
	}
	return static_cast<uint32_t>( parsed );
}

} // namespace gamescope::framegen
