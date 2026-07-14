#pragma once

#include <cstdint>
#include <optional>

namespace gamescope::framegen
{

struct QueryRingSelection
{
	uint32_t slot;
	uint32_t nextHead;
};

// Select an unowned slot while preserving round-robin ordering. A full ring is
// allowed: timing is optional, and skipping a sample is safer than resetting a
// query whose completed result has not been consumed yet.
constexpr std::optional<QueryRingSelection> select_query_ring_slot(
	uint32_t head, uint32_t depth, uint64_t occupiedMask )
{
	if ( depth == 0 || depth > 64 )
		return std::nullopt;

	head %= depth;
	for ( uint32_t probe = 0; probe < depth; probe++ )
	{
		const uint32_t slot = ( head + probe ) % depth;
		if ( ( occupiedMask & ( uint64_t{ 1 } << slot ) ) == 0 )
			return QueryRingSelection{ slot, ( slot + 1 ) % depth };
	}

	return std::nullopt;
}

} // namespace gamescope::framegen
