#pragma once

#include "types.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>

namespace gamescope::framegen
{

inline constexpr uint32_t k_uFramegenHudMaxLines = 6u;
inline constexpr uint32_t k_uFramegenHudMaxColumns = 72u;
inline constexpr uint32_t k_uFramegenHudPackedTextWords =
	( k_uFramegenHudMaxLines * k_uFramegenHudMaxColumns ) / 4u;
inline constexpr uint32_t k_uFramegenHudDeviceNameColumns = 19u;
inline constexpr size_t k_nFramegenHudSparklineSamples = 12u;

[[nodiscard]] constexpr uint64_t font8x8_glyph(
	uint8_t row0, uint8_t row1, uint8_t row2, uint8_t row3,
	uint8_t row4, uint8_t row5, uint8_t row6, uint8_t row7 )
{
	return static_cast<uint64_t>( row0 )
		| static_cast<uint64_t>( row1 ) << 8u
		| static_cast<uint64_t>( row2 ) << 16u
		| static_cast<uint64_t>( row3 ) << 24u
		| static_cast<uint64_t>( row4 ) << 32u
		| static_cast<uint64_t>( row5 ) << 40u
		| static_cast<uint64_t>( row6 ) << 48u
		| static_cast<uint64_t>( row7 ) << 56u;
}

// Daniel Hepper's font8x8_basic, based on IBM's public-domain VGA font.
// Public domain; source rows are stored bottom-to-top and flipped by the HUD
// shader. https://github.com/dhepper/font8x8/blob/master/font8x8_basic.h
inline constexpr uint64_t k_uFramegenHudFont8x8[128] = {
	0,
	font8x8_glyph( 0xFF, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00 ), // spark 1/8
	font8x8_glyph( 0xFF, 0xFF, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00 ), // spark 2/8
	font8x8_glyph( 0xFF, 0xFF, 0xFF, 0x00, 0x00, 0x00, 0x00, 0x00 ), // spark 3/8
	font8x8_glyph( 0xFF, 0xFF, 0xFF, 0xFF, 0x00, 0x00, 0x00, 0x00 ), // spark 4/8
	font8x8_glyph( 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x00, 0x00, 0x00 ), // spark 5/8
	font8x8_glyph( 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x00, 0x00 ), // spark 6/8
	font8x8_glyph( 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x00 ), // spark 7/8
	font8x8_glyph( 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF ), // spark 8/8
	0, 0, 0, 0, 0, 0, 0,
	0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	0, // U+0020 space
	font8x8_glyph( 0x00, 0x18, 0x00, 0x18, 0x18, 0x3C, 0x3C, 0x18 ), // !
	font8x8_glyph( 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x36, 0x36 ), // "
	font8x8_glyph( 0x00, 0x36, 0x36, 0x7F, 0x36, 0x7F, 0x36, 0x36 ), // #
	font8x8_glyph( 0x00, 0x0C, 0x1F, 0x30, 0x1E, 0x03, 0x3E, 0x0C ), // $
	font8x8_glyph( 0x00, 0x63, 0x66, 0x0C, 0x18, 0x33, 0x63, 0x00 ), // %
	font8x8_glyph( 0x00, 0x6E, 0x33, 0x3B, 0x6E, 0x1C, 0x36, 0x1C ), // &
	font8x8_glyph( 0x00, 0x00, 0x00, 0x00, 0x00, 0x03, 0x06, 0x06 ), // '
	font8x8_glyph( 0x00, 0x18, 0x0C, 0x06, 0x06, 0x06, 0x0C, 0x18 ), // (
	font8x8_glyph( 0x00, 0x06, 0x0C, 0x18, 0x18, 0x18, 0x0C, 0x06 ), // )
	font8x8_glyph( 0x00, 0x00, 0x66, 0x3C, 0xFF, 0x3C, 0x66, 0x00 ), // *
	font8x8_glyph( 0x00, 0x00, 0x0C, 0x0C, 0x3F, 0x0C, 0x0C, 0x00 ), // +
	font8x8_glyph( 0x06, 0x0C, 0x0C, 0x00, 0x00, 0x00, 0x00, 0x00 ), // ,
	font8x8_glyph( 0x00, 0x00, 0x00, 0x00, 0x3F, 0x00, 0x00, 0x00 ), // -
	font8x8_glyph( 0x00, 0x0C, 0x0C, 0x00, 0x00, 0x00, 0x00, 0x00 ), // .
	font8x8_glyph( 0x00, 0x01, 0x03, 0x06, 0x0C, 0x18, 0x30, 0x60 ), // /
	font8x8_glyph( 0x00, 0x3E, 0x67, 0x6F, 0x7B, 0x73, 0x63, 0x3E ), // 0
	font8x8_glyph( 0x00, 0x3F, 0x0C, 0x0C, 0x0C, 0x0C, 0x0E, 0x0C ), // 1
	font8x8_glyph( 0x00, 0x3F, 0x33, 0x06, 0x1C, 0x30, 0x33, 0x1E ), // 2
	font8x8_glyph( 0x00, 0x1E, 0x33, 0x30, 0x1C, 0x30, 0x33, 0x1E ), // 3
	font8x8_glyph( 0x00, 0x78, 0x30, 0x7F, 0x33, 0x36, 0x3C, 0x38 ), // 4
	font8x8_glyph( 0x00, 0x1E, 0x33, 0x30, 0x30, 0x1F, 0x03, 0x3F ), // 5
	font8x8_glyph( 0x00, 0x1E, 0x33, 0x33, 0x1F, 0x03, 0x06, 0x1C ), // 6
	font8x8_glyph( 0x00, 0x0C, 0x0C, 0x0C, 0x18, 0x30, 0x33, 0x3F ), // 7
	font8x8_glyph( 0x00, 0x1E, 0x33, 0x33, 0x1E, 0x33, 0x33, 0x1E ), // 8
	font8x8_glyph( 0x00, 0x0E, 0x18, 0x30, 0x3E, 0x33, 0x33, 0x1E ), // 9
	font8x8_glyph( 0x00, 0x0C, 0x0C, 0x00, 0x00, 0x0C, 0x0C, 0x00 ), // :
	font8x8_glyph( 0x06, 0x0C, 0x0C, 0x00, 0x00, 0x0C, 0x0C, 0x00 ), // ;
	font8x8_glyph( 0x00, 0x18, 0x0C, 0x06, 0x03, 0x06, 0x0C, 0x18 ), // <
	font8x8_glyph( 0x00, 0x00, 0x3F, 0x00, 0x00, 0x3F, 0x00, 0x00 ), // =
	font8x8_glyph( 0x00, 0x06, 0x0C, 0x18, 0x30, 0x18, 0x0C, 0x06 ), // >
	font8x8_glyph( 0x00, 0x0C, 0x00, 0x0C, 0x18, 0x30, 0x33, 0x1E ), // ?
	font8x8_glyph( 0x00, 0x1E, 0x03, 0x7B, 0x7B, 0x7B, 0x63, 0x3E ), // @
	font8x8_glyph( 0x00, 0x33, 0x33, 0x3F, 0x33, 0x33, 0x1E, 0x0C ), // A
	font8x8_glyph( 0x00, 0x3F, 0x66, 0x66, 0x3E, 0x66, 0x66, 0x3F ), // B
	font8x8_glyph( 0x00, 0x3C, 0x66, 0x03, 0x03, 0x03, 0x66, 0x3C ), // C
	font8x8_glyph( 0x00, 0x1F, 0x36, 0x66, 0x66, 0x66, 0x36, 0x1F ), // D
	font8x8_glyph( 0x00, 0x7F, 0x46, 0x16, 0x1E, 0x16, 0x46, 0x7F ), // E
	font8x8_glyph( 0x00, 0x0F, 0x06, 0x16, 0x1E, 0x16, 0x46, 0x7F ), // F
	font8x8_glyph( 0x00, 0x7C, 0x66, 0x73, 0x03, 0x03, 0x66, 0x3C ), // G
	font8x8_glyph( 0x00, 0x33, 0x33, 0x33, 0x3F, 0x33, 0x33, 0x33 ), // H
	font8x8_glyph( 0x00, 0x1E, 0x0C, 0x0C, 0x0C, 0x0C, 0x0C, 0x1E ), // I
	font8x8_glyph( 0x00, 0x1E, 0x33, 0x33, 0x30, 0x30, 0x30, 0x78 ), // J
	font8x8_glyph( 0x00, 0x67, 0x66, 0x36, 0x1E, 0x36, 0x66, 0x67 ), // K
	font8x8_glyph( 0x00, 0x7F, 0x66, 0x46, 0x06, 0x06, 0x06, 0x0F ), // L
	font8x8_glyph( 0x00, 0x63, 0x63, 0x6B, 0x7F, 0x7F, 0x77, 0x63 ), // M
	font8x8_glyph( 0x00, 0x63, 0x63, 0x73, 0x7B, 0x6F, 0x67, 0x63 ), // N
	font8x8_glyph( 0x00, 0x1C, 0x36, 0x63, 0x63, 0x63, 0x36, 0x1C ), // O
	font8x8_glyph( 0x00, 0x0F, 0x06, 0x06, 0x3E, 0x66, 0x66, 0x3F ), // P
	font8x8_glyph( 0x00, 0x38, 0x1E, 0x3B, 0x33, 0x33, 0x33, 0x1E ), // Q
	font8x8_glyph( 0x00, 0x67, 0x66, 0x36, 0x3E, 0x66, 0x66, 0x3F ), // R
	font8x8_glyph( 0x00, 0x1E, 0x33, 0x38, 0x0E, 0x07, 0x33, 0x1E ), // S
	font8x8_glyph( 0x00, 0x1E, 0x0C, 0x0C, 0x0C, 0x0C, 0x2D, 0x3F ), // T
	font8x8_glyph( 0x00, 0x3F, 0x33, 0x33, 0x33, 0x33, 0x33, 0x33 ), // U
	font8x8_glyph( 0x00, 0x0C, 0x1E, 0x33, 0x33, 0x33, 0x33, 0x33 ), // V
	font8x8_glyph( 0x00, 0x63, 0x77, 0x7F, 0x6B, 0x63, 0x63, 0x63 ), // W
	font8x8_glyph( 0x00, 0x63, 0x36, 0x1C, 0x1C, 0x36, 0x63, 0x63 ), // X
	font8x8_glyph( 0x00, 0x1E, 0x0C, 0x0C, 0x1E, 0x33, 0x33, 0x33 ), // Y
	font8x8_glyph( 0x00, 0x7F, 0x66, 0x4C, 0x18, 0x31, 0x63, 0x7F ), // Z
	font8x8_glyph( 0x00, 0x1E, 0x06, 0x06, 0x06, 0x06, 0x06, 0x1E ), // [
	font8x8_glyph( 0x00, 0x40, 0x60, 0x30, 0x18, 0x0C, 0x06, 0x03 ), // backslash
	font8x8_glyph( 0x00, 0x1E, 0x18, 0x18, 0x18, 0x18, 0x18, 0x1E ), // ]
	font8x8_glyph( 0x00, 0x00, 0x00, 0x00, 0x63, 0x36, 0x1C, 0x08 ), // ^
	font8x8_glyph( 0xFF, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00 ), // _
	font8x8_glyph( 0x00, 0x00, 0x00, 0x00, 0x00, 0x18, 0x0C, 0x0C ), // `
	font8x8_glyph( 0x00, 0x6E, 0x33, 0x3E, 0x30, 0x1E, 0x00, 0x00 ), // a
	font8x8_glyph( 0x00, 0x3B, 0x66, 0x66, 0x3E, 0x06, 0x06, 0x07 ), // b
	font8x8_glyph( 0x00, 0x1E, 0x33, 0x03, 0x33, 0x1E, 0x00, 0x00 ), // c
	font8x8_glyph( 0x00, 0x6E, 0x33, 0x33, 0x3E, 0x30, 0x30, 0x38 ), // d
	font8x8_glyph( 0x00, 0x1E, 0x03, 0x3F, 0x33, 0x1E, 0x00, 0x00 ), // e
	font8x8_glyph( 0x00, 0x0F, 0x06, 0x06, 0x0F, 0x06, 0x36, 0x1C ), // f
	font8x8_glyph( 0x1F, 0x30, 0x3E, 0x33, 0x33, 0x6E, 0x00, 0x00 ), // g
	font8x8_glyph( 0x00, 0x67, 0x66, 0x66, 0x6E, 0x36, 0x06, 0x07 ), // h
	font8x8_glyph( 0x00, 0x1E, 0x0C, 0x0C, 0x0C, 0x0E, 0x00, 0x0C ), // i
	font8x8_glyph( 0x1E, 0x33, 0x33, 0x30, 0x30, 0x30, 0x00, 0x30 ), // j
	font8x8_glyph( 0x00, 0x67, 0x36, 0x1E, 0x36, 0x66, 0x06, 0x07 ), // k
	font8x8_glyph( 0x00, 0x1E, 0x0C, 0x0C, 0x0C, 0x0C, 0x0C, 0x0E ), // l
	font8x8_glyph( 0x00, 0x63, 0x6B, 0x7F, 0x7F, 0x33, 0x00, 0x00 ), // m
	font8x8_glyph( 0x00, 0x33, 0x33, 0x33, 0x33, 0x1F, 0x00, 0x00 ), // n
	font8x8_glyph( 0x00, 0x1E, 0x33, 0x33, 0x33, 0x1E, 0x00, 0x00 ), // o
	font8x8_glyph( 0x0F, 0x06, 0x3E, 0x66, 0x66, 0x3B, 0x00, 0x00 ), // p
	font8x8_glyph( 0x78, 0x30, 0x3E, 0x33, 0x33, 0x6E, 0x00, 0x00 ), // q
	font8x8_glyph( 0x00, 0x0F, 0x06, 0x66, 0x6E, 0x3B, 0x00, 0x00 ), // r
	font8x8_glyph( 0x00, 0x1F, 0x30, 0x1E, 0x03, 0x3E, 0x00, 0x00 ), // s
	font8x8_glyph( 0x00, 0x18, 0x2C, 0x0C, 0x0C, 0x3E, 0x0C, 0x08 ), // t
	font8x8_glyph( 0x00, 0x6E, 0x33, 0x33, 0x33, 0x33, 0x00, 0x00 ), // u
	font8x8_glyph( 0x00, 0x0C, 0x1E, 0x33, 0x33, 0x33, 0x00, 0x00 ), // v
	font8x8_glyph( 0x00, 0x36, 0x7F, 0x7F, 0x6B, 0x63, 0x00, 0x00 ), // w
	font8x8_glyph( 0x00, 0x63, 0x36, 0x1C, 0x36, 0x63, 0x00, 0x00 ), // x
	font8x8_glyph( 0x1F, 0x30, 0x3E, 0x33, 0x33, 0x33, 0x00, 0x00 ), // y
	font8x8_glyph( 0x00, 0x3F, 0x26, 0x0C, 0x19, 0x3F, 0x00, 0x00 ), // z
	font8x8_glyph( 0x00, 0x38, 0x0C, 0x0C, 0x07, 0x0C, 0x0C, 0x38 ), // {
	font8x8_glyph( 0x00, 0x18, 0x18, 0x18, 0x00, 0x18, 0x18, 0x18 ), // |
	font8x8_glyph( 0x00, 0x07, 0x0C, 0x0C, 0x38, 0x0C, 0x0C, 0x07 ), // }
	font8x8_glyph( 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x3B, 0x6E ), // ~
	0, // U+007f delete
};
static_assert( std::size( k_uFramegenHudFont8x8 ) == 128u );

using FramegenHudSparkline_t =
	std::array<char, k_nFramegenHudSparklineSamples + 1u>;

[[nodiscard]] inline FramegenHudSparkline_t framegen_hud_sparkline(
	const std::array<double, k_nFramegenHudSparklineSamples> &values,
	double scale )
{
	FramegenHudSparkline_t result = {};
	for ( size_t i = 0u; i < values.size(); i++ )
	{
		const double bounded = scale > 0.0 && values[i] > 0.0
			? std::min( values[i], scale ) : 0.0;
		const uint32_t level = scale > 0.0
			? static_cast<uint32_t>( bounded * 7.0 / scale + 0.5 ) : 0u;
		result[i] = static_cast<char>( 0x01u + std::min( level, 7u ) );
	}
	return result;
}

struct FramegenHudSnapshot_t
{
	const char *version = "unknown";
	const char *deviceName = "unknown";
	GamescopeFramegenMode mode = GamescopeFramegenMode::Extrapolate;
	GamescopeFramegenQuality quality = GamescopeFramegenQuality::Low;
	uint32_t multiplier = 2u;
	uint32_t refreshMilliHz = 0u;
	bool vrrRequested = false;
	bool vrrActive = false;
	bool bidirRequested = false;
	bool bidirActive = false;
	bool baseLayer = false;
	bool clientBuffersStaged = false;
	bool netRequested = false;
	bool netActive = false;
	bool netOnline = false;
	bool adapt = false;
	uint64_t real = 0u;
	uint64_t delayedReal = 0u;
	uint64_t generated = 0u;
	uint64_t repeats = 0u;
	int32_t biasTenthsMs = 0;
	uint32_t deadlineHitPercent = 0u;
	uint32_t pacingSdTenthsMs = 0u;
	std::array<double, k_nFramegenHudSparklineSamples> deadlineHitHistory = {};
	std::array<double, k_nFramegenHudSparklineSamples> pacingSdMsHistory = {};
	uint64_t resets = 0u;
	uint64_t ringResets = 0u;
	uint64_t netTrainedSteps = 0u;
	bool netProfileLoaded = false;
};

struct FramegenHudText_t
{
	uint32_t lineCount = 0u;
	std::array<std::array<char, k_uFramegenHudMaxColumns + 1u>, k_uFramegenHudMaxLines> lines = {};

	bool operator==( const FramegenHudText_t &other ) const = default;
};

inline void framegen_hud_append( char *dst, size_t capacity, const char *text )
{
	const size_t used = std::strlen( dst );
	if ( used + std::strlen( text ) >= capacity )
		return;
	std::snprintf( dst + used, capacity - used, "%s", text );
}

inline void framegen_hud_trim_device_name( char *dst, size_t capacity,
	const char *deviceName )
{
	if ( capacity == 0u )
		return;
	const char *begin = deviceName != nullptr ? deviceName : "";
	while ( *begin == ' ' || *begin == '\t' )
		begin++;
	const char *end = begin + std::strlen( begin );
	while ( end > begin && ( end[-1] == ' ' || end[-1] == '\t' ) )
		end--;
	size_t length = std::min<size_t>(
		static_cast<size_t>( end - begin ),
		std::min<size_t>( capacity - 1u, k_uFramegenHudDeviceNameColumns ) );
	while ( length > 0u && ( begin[length - 1u] == ' ' || begin[length - 1u] == '\t' ) )
		length--;
	if ( length != 0u )
		std::memcpy( dst, begin, length );
	dst[length] = '\0';
	if ( length == 0u )
		std::snprintf( dst, capacity, "unknown" );
}

inline void framegen_hud_add_line( FramegenHudText_t &text, const char *line )
{
	if ( line == nullptr || *line == '\0' || text.lineCount >= k_uFramegenHudMaxLines )
		return;
	std::snprintf( text.lines[text.lineCount].data(),
		text.lines[text.lineCount].size(), "%s", line );
	text.lineCount++;
}

inline void framegen_hud_format_steps( char *dst, size_t capacity, uint64_t steps )
{
	if ( steps < 1'000u )
		std::snprintf( dst, capacity, "%llu", static_cast<unsigned long long>( steps ) );
	else if ( steps < 1'000'000u )
	{
		const uint64_t tenths = ( steps + 50u ) / 100u;
		std::snprintf( dst, capacity, "%llu.%lluk",
			static_cast<unsigned long long>( tenths / 10u ),
			static_cast<unsigned long long>( tenths % 10u ) );
	}
	else
	{
		const uint64_t tenths = ( steps + 50'000u ) / 100'000u;
		std::snprintf( dst, capacity, "%llu.%llum",
			static_cast<unsigned long long>( tenths / 10u ),
			static_cast<unsigned long long>( tenths % 10u ) );
	}
}

[[nodiscard]] inline FramegenHudText_t format_framegen_hud(
	uint32_t level, const FramegenHudSnapshot_t &snapshot )
{
	FramegenHudText_t result;
	if ( level == 0u )
		return result;

	const uint32_t refreshHz = std::min( 999u,
		( snapshot.refreshMilliHz + 500u ) / 1'000u );
	const uint32_t gameRate = static_cast<uint32_t>( std::min<uint64_t>(
		999u, ( snapshot.real + snapshot.delayedReal ) / 5u ) );
	const uint32_t generatedRate = static_cast<uint32_t>( std::min<uint64_t>(
		999u, snapshot.generated / 5u ) );
	const uint32_t repeatRate = static_cast<uint32_t>( std::min<uint64_t>(
		999u, snapshot.repeats / 5u ) );
	const uint32_t fillRate = static_cast<uint32_t>( std::min<uint64_t>(
		999u, ( snapshot.real + snapshot.delayedReal + snapshot.generated ) / 5u ) );

	char line[128] = {};
	std::snprintf( line, sizeof( line ),
		"%-8s %-21.21s%s x%u . %s . %uHz %s", "gameslop",
		snapshot.version != nullptr ? snapshot.version : "unknown",
		mode_name( snapshot.mode ), std::clamp( snapshot.multiplier, 2u, 4u ),
		quality_name( snapshot.quality ), refreshHz,
		snapshot.vrrActive ? "VRR" : "fixed" );
	framegen_hud_add_line( result, line );

	char deviceName[k_uFramegenHudDeviceNameColumns + 1u] = {};
	framegen_hud_trim_device_name( deviceName, sizeof( deviceName ), snapshot.deviceName );
	std::snprintf( line, sizeof( line ),
		"%-8s %-21sbuffers %s", "present", deviceName,
		snapshot.clientBuffersStaged ? "staged(cross-GPU)" : "local" );
	framegen_hud_add_line( result, line );

	const char *bidirState = snapshot.bidirActive ? "on"
		: snapshot.bidirRequested ? "requested(OFF)" : "off";
	const char *netState = snapshot.netActive
		? ( snapshot.netOnline ? "online" : "blob" )
		: snapshot.netRequested ? "requested(OFF)" : "off";
	std::snprintf( line, sizeof( line ),
		"%-8s bidir:%s  base:%s  net:%s  adapt:%s", "modes",
		bidirState, snapshot.baseLayer ? "on" : "off", netState,
		snapshot.adapt ? "on" : "off" );
	if ( snapshot.vrrRequested && !snapshot.vrrActive )
	{
		constexpr const char *pszVrrFallback = "  vrr:requested(off)";
		if ( std::strlen( line ) + std::strlen( pszVrrFallback )
			<= k_uFramegenHudMaxColumns )
		{
			framegen_hud_append( line, sizeof( line ), pszVrrFallback );
		}
		else
		{
			// Conflicting requests are the information to preserve when all four
			// ordinary state fields and the VRR fallback cannot fit together.
			std::snprintf( line, sizeof( line ),
				"%-8s bidir:%s  net:%s  vrr:requested(off)",
				"modes", bidirState, netState );
		}
	}
	framegen_hud_add_line( result, line );

	std::snprintf( line, sizeof( line ),
		"%-8s game %ufps  gen %ufps  repeat %ufps  fill %u/%u", "rates",
		gameRate, generatedRate, repeatRate, fillRate, refreshHz );
	framegen_hud_add_line( result, line );

	if ( level < 2u )
		return result;

	const uint32_t sdTenths = std::min( snapshot.pacingSdTenthsMs, 9'999u );
	const FramegenHudSparkline_t hitSparkline = framegen_hud_sparkline(
		snapshot.deadlineHitHistory, 100.0 );
	// Flip-interval standard deviation uses a fixed 0..4 ms scale; noisier
	// windows saturate at the full-height glyph.
	const FramegenHudSparkline_t jitterSparkline = framegen_hud_sparkline(
		snapshot.pacingSdMsHistory, 4.0 );
	std::snprintf( line, sizeof( line ),
		"%-8s hit %u%%  %s   jitter +/-%u.%ums  %s", "pace",
		std::min( snapshot.deadlineHitPercent, 100u ),
		hitSparkline.data(),
		sdTenths / 10u, sdTenths % 10u,
		jitterSparkline.data() );
	framegen_hud_add_line( result, line );

	if ( snapshot.netActive )
	{
		char steps[24] = {};
		framegen_hud_format_steps( steps, sizeof( steps ), snapshot.netTrainedSteps );
		std::snprintf( line, sizeof( line ),
			"%-8s %s . %s steps . profile %s", "net",
			snapshot.netOnline ? "online" : "offline", steps,
			snapshot.netProfileLoaded ? "loaded" : "none" );
		framegen_hud_add_line( result, line );
	}

	return result;
}

// Scalar-layout UBO consumed by cs_framegen_hud.comp. Text is packed four
// ASCII bytes per uint; the uint64_t font above is split explicitly so the GPU
// ABI is independent of host endianness and uint64 shader support.
struct alignas( 16 ) FramegenHudUniform_t
{
	uint32_t lineCount = 0u;
	uint32_t widthChars = 0u;
	uint32_t hdr = 0u;
	uint32_t reserved = 0u;
	std::array<uint32_t, k_uFramegenHudMaxLines> lineLengths = {};
	std::array<uint32_t, k_uFramegenHudPackedTextWords> text = {};
	std::array<uint32_t, 256> font = {};
};

static_assert( offsetof( FramegenHudUniform_t, lineLengths ) == 16u );
static_assert( offsetof( FramegenHudUniform_t, text ) == 40u );
static_assert( offsetof( FramegenHudUniform_t, font ) == 472u );
static_assert( sizeof( FramegenHudUniform_t ) == 1'504u );

struct FramegenHudPush_t
{
	uint32_t offsetX;
	uint32_t offsetY;
	uint32_t scale;
	uint32_t pad = 0u;

	FramegenHudPush_t( uint32_t x, uint32_t y, uint32_t uScale )
		: offsetX( x ), offsetY( y ), scale( uScale )
	{
	}
};
static_assert( sizeof( FramegenHudPush_t ) == 16u );

[[nodiscard]] inline FramegenHudUniform_t make_framegen_hud_uniform(
	const FramegenHudText_t &text, bool hdr )
{
	FramegenHudUniform_t result;
	result.lineCount = std::min( text.lineCount, k_uFramegenHudMaxLines );
	result.hdr = hdr;
	for ( uint32_t line = 0u; line < result.lineCount; line++ )
	{
		const uint32_t length = static_cast<uint32_t>( std::min<size_t>(
			std::strlen( text.lines[line].data() ), k_uFramegenHudMaxColumns ) );
		result.lineLengths[line] = length;
		result.widthChars = std::max( result.widthChars, length );
		for ( uint32_t column = 0u; column < length; column++ )
		{
			const uint32_t byteIndex = line * k_uFramegenHudMaxColumns + column;
			result.text[byteIndex / 4u] |=
				static_cast<uint32_t>( static_cast<uint8_t>( text.lines[line][column] ) )
				<< ( 8u * ( byteIndex % 4u ) );
		}
	}
	for ( uint32_t glyph = 0u; glyph < 128u; glyph++ )
	{
		result.font[glyph * 2u] = static_cast<uint32_t>( k_uFramegenHudFont8x8[glyph] );
		result.font[glyph * 2u + 1u] = static_cast<uint32_t>(
			k_uFramegenHudFont8x8[glyph] >> 32u );
	}
	return result;
}

} // namespace gamescope::framegen
