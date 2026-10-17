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

// The motion field's texel size in output pixels. The coarse search always
// runs at /32 (±128 px at R=4), so a finer field adds pyramid levels below it
// and never narrows the motion range. Field-texel constants (FB tolerance,
// static-motion bar) were tuned at the /8 reference and convert by
// motion_reference_texels to keep their meaning in pixels.
inline constexpr uint32_t k_uMotionReferenceScale = 8u;
inline constexpr uint32_t k_uMotionFineScale = 4u;
inline constexpr uint32_t k_uMotionCoarsestScale = 32u;

// /4 on request below the learned pipelines, whose net is trained on /8
// fields; else /8. Callers pass the configured pipeline, never the effective
// one, so a degradation step can never switch the finer, costlier field on.
[[nodiscard]] constexpr uint32_t motion_field_scale( bool fineRequested,
	GamescopeFramegenPipeline configured )
{
	return fineRequested && configured < GamescopeFramegenPipeline::Learned
		? k_uMotionFineScale : k_uMotionReferenceScale;
}

// The 2x pyramid steps from the field's scale down to the coarsest level.
[[nodiscard]] constexpr uint32_t motion_coarse_levels( uint32_t scale )
{
	uint32_t levels = 0;
	for ( uint32_t s = scale; s != 0u && s < k_uMotionCoarsestScale; s *= 2u )
		levels++;
	return levels;
}

inline constexpr uint32_t k_uMotionMaxCoarseLevels = motion_coarse_levels( k_uMotionFineScale );

[[nodiscard]] constexpr float motion_reference_texels( float atReference, uint32_t scale )
{
	return atReference * static_cast<float>( k_uMotionReferenceScale ) / static_cast<float>( scale );
}

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
