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
	// back to extrapolation where matching is unconfident. Higher quality on
	// panning/scrolling motion at a higher compute cost.
	Motion,
};

enum class GamescopeFramegenQuality : uint32_t
{
	// Forward hierarchical matching only. No reverse consistency, adaptation,
	// learned refinement, or temporal acceleration.
	Low,
	// Add reverse-field consistency and the full-resolution agreement test.
	Medium,
	// Add self-supervised adaptation and permit learned field refinement.
	// This preserves the behavior that predates explicit quality tiers.
	High,
	// Add confidence-gated temporal acceleration from the preceding checked
	// field.
	Ultra,
	// Add full-resolution, color-guided reconstruction of the low-resolution
	// field at motion boundaries. This is the most expensive causal path.
	Extreme,
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

[[nodiscard]] constexpr const char *quality_name( GamescopeFramegenQuality quality )
{
	switch ( quality )
	{
		case GamescopeFramegenQuality::Low:
			return "low";
		case GamescopeFramegenQuality::Medium:
			return "medium";
		case GamescopeFramegenQuality::High:
			return "high";
		case GamescopeFramegenQuality::Ultra:
			return "ultra";
		case GamescopeFramegenQuality::Extreme:
			return "extreme";
		default:
			return "unknown";
	}
}

} // namespace gamescope::framegen
