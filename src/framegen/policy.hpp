#pragma once

#include "types.hpp"

#include <algorithm>
#include <cstdint>

namespace gamescope::framegen
{

struct EffectiveConfig
{
	GamescopeFramegenMode mode;
	uint32_t multiplier;
	GamescopeFramegenPipeline pipeline;
};

// The compositor output ring also carries zero-copy real-frame history. A
// higher generated cadence can leave more distinct buffers queued in a nested
// compositor before its aggregate release events catch up, so reserve two
// backend-lifetime slots per requested multiplier in addition to four slots
// for history and real composite progress. Ownership checks, not this capacity
// estimate, remain the correctness boundary.
[[nodiscard]] constexpr uint32_t output_ring_size_for_multiplier( int multiplier )
{
	const uint32_t boundedMultiplier = static_cast<uint32_t>(
		std::clamp( multiplier, 2, 4 ) );
	return 4u + 2u * boundedMultiplier;
}

// Count the degradation rungs below a startup pipeline ceiling. Motion sheds
// pipeline passes first, then falls back to extrapolation; multiplier reductions are
// last. There is deliberately no disabled rung, so GPU timing never starves.
[[nodiscard]] constexpr uint32_t max_degrade_steps( GamescopeFramegenMode mode,
	GamescopeFramegenPipeline pipeline, int multiplier )
{
	const uint32_t nMotionRungs = mode == GamescopeFramegenMode::Motion
		? static_cast<uint32_t>( pipeline ) + 1u : 0u;
	const uint32_t nMultiplierRungs =
		static_cast<uint32_t>( std::max( 0, multiplier - 2 ) );
	return nMotionRungs + nMultiplierRungs;
}

// Resolve one rung without touching global state. Keeping this constexpr makes
// the renderer call site zero-cost while allowing exhaustive CPU-only tests.
[[nodiscard]] constexpr EffectiveConfig effective_config( GamescopeFramegenMode mode,
	GamescopeFramegenPipeline pipeline, int multiplier, uint32_t nDegradeSteps )
{
	EffectiveConfig config = {
		mode,
		static_cast<uint32_t>( std::max( 2, multiplier ) ),
		pipeline,
	};
	uint32_t n = nDegradeSteps;

	while ( n > 0 && config.mode == GamescopeFramegenMode::Motion
		&& config.pipeline > GamescopeFramegenPipeline::Warp )
	{
		config.pipeline = static_cast<GamescopeFramegenPipeline>(
			static_cast<uint32_t>( config.pipeline ) - 1u );
		n--;
	}
	if ( n > 0 && config.mode == GamescopeFramegenMode::Motion )
	{
		config.mode = GamescopeFramegenMode::Extrapolate;
		n--;
	}
	while ( n > 0 && config.multiplier > 2u )
	{
		config.multiplier--;
		n--;
	}

	return config;
}

} // namespace gamescope::framegen
