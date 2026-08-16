#pragma once

#include <cstdint>

enum class GamescopeFramegenMode : uint32_t
{
	// Forward extrapolation of the previous two real frames. Low latency and
	// temporally monotonic (the generated frame advances motion). Default.
	Extrapolate,
	// 50/50 average of the previous two real frames. Softer, but the generated
	// frame lands temporally between older frames, which can read as judder.
	Blend,
	// Motion-compensated: estimate per-block motion between the last two real
	// frames (luma pyramid + block matching) and reproject along it, falling
	// back to extrapolation where matching is unconfident. Better reconstruction on
	// panning/scrolling motion at a higher compute cost.
	Motion,
};

enum class GamescopeFramegenPipeline : uint32_t
{
	// Forward hierarchical matcher plus constant-velocity warp only.
	Warp,
	// Add reverse-field consistency and the full-resolution agreement test.
	Checked,
	// Add self-supervised adaptation and permit learned field refinement.
	Learned,
	// Add confidence-gated temporal acceleration from the preceding checked
	// field.
	Predict,
	// Add full-resolution color-guided reconstruction, the three-frame
	// disocclusion reservoir, and the shading-persistence head.
	Guided,
};

namespace gamescope::framegen
{

[[nodiscard]] constexpr const char *mode_name( GamescopeFramegenMode mode )
{
	switch ( mode )
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

[[nodiscard]] constexpr const char *pipeline_name( GamescopeFramegenPipeline pipeline )
{
	switch ( pipeline )
	{
		case GamescopeFramegenPipeline::Warp:
			return "warp";
		case GamescopeFramegenPipeline::Checked:
			return "checked";
		case GamescopeFramegenPipeline::Learned:
			return "learned";
		case GamescopeFramegenPipeline::Predict:
			return "predict";
		case GamescopeFramegenPipeline::Guided:
			return "guided";
		default:
			return "unknown";
	}
}

} // namespace gamescope::framegen
