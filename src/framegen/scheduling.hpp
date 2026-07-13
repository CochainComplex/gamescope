#pragma once

#include "policy.hpp"

#include <algorithm>
#include <cstdint>

namespace gamescope::framegen
{

// Cadence confidence is a leaky score: sustained empty-vblank opportunities
// arm the shared-queue path, while one jittered interval does not immediately
// erase several useful observations.
inline constexpr uint32_t k_uCadenceConfidenceRequired = 4u;
inline constexpr uint32_t k_uCadenceConfidenceGain = 2u;
inline constexpr uint32_t k_uCadenceConfidenceLeak = 1u;

// Deadline-ladder policy. The headroom margin absorbs per-batch GPU-time
// jitter; the sample and hold thresholds keep delayed timestamp feedback from
// skipping multiple rungs before a newly selected rung has been measured.
inline constexpr uint64_t k_uDeadlinePercent = 85u;
inline constexpr uint32_t k_uDeadlineMinSamples = 3u;
inline constexpr uint32_t k_uDeadlineHoldFrames = 4u;

// JIT only needs one missed refresh interval. VRR hybrid needs two panel-safe
// halves plus jitter margin before inserting its midpoint flip.
inline constexpr uint64_t k_uJitKeepUpPercent = 110u;
inline constexpr uint64_t k_uVrrHybridKeepUpPercent = 220u;

// Preconditions: currentPresentNs >= previousPresentNs and vblankIntervalNs is
// non-zero. The renderer satisfies both with a monotonic clock and a validated
// output refresh interval.
[[nodiscard]] constexpr bool leaves_empty_vblank( uint64_t currentPresentNs,
	uint64_t previousPresentNs, uint64_t vblankIntervalNs )
{
	return previousPresentNs != 0u
		&& currentPresentNs - previousPresentNs >= ( vblankIntervalNs * 3u ) / 2u;
}

[[nodiscard]] constexpr uint32_t update_cadence_confidence( uint32_t confidence,
	bool generatable )
{
	if ( generatable )
		return std::min( k_uCadenceConfidenceRequired,
			confidence + k_uCadenceConfidenceGain );

	return confidence > k_uCadenceConfidenceLeak
		? confidence - k_uCadenceConfidenceLeak : 0u;
}

[[nodiscard]] constexpr bool reactive_generation_ready( bool generatable,
	uint32_t cadenceConfidence )
{
	return generatable && cadenceConfidence >= k_uCadenceConfidenceRequired;
}

// Round a measured real-frame interval to whole display intervals. The caller
// invalidates history before the interval can exceed the uint32_t result range.
// Precondition: vblankIntervalNs is non-zero.
[[nodiscard]] constexpr uint32_t measured_gap_vblanks( uint64_t realFrameIntervalNs,
	uint64_t vblankIntervalNs )
{
	return std::max( 1u, static_cast<uint32_t>(
		( realFrameIntervalNs + vblankIntervalNs / 2u ) / vblankIntervalNs ) );
}

// Expand a measured gap to the configured multiplier when the caller's timeline
// allows disposable speculation. Actual bidirectional batches pass false
// because they remain on the presentation timeline; deadline cost identity may
// independently retain its historical expanded batch shape.
// Precondition: multiplier >= 2.
[[nodiscard]] constexpr uint32_t expanded_gap_vblanks( uint32_t measuredGapVblanks,
	uint32_t multiplier, bool expandToMultiplier )
{
	return expandToMultiplier
		? std::max( measuredGapVblanks, std::max( 2u, multiplier ) )
		: measuredGapVblanks;
}

// Precondition: multiplier >= 2.
[[nodiscard]] constexpr uint32_t generated_slots_for_gap( uint32_t gapVblanks,
	uint32_t multiplier, bool dedicatedQueue )
{
	const uint32_t slotCeiling = std::max( 1u, multiplier - 1u );
	const uint32_t generated = gapVblanks > 1u
		? std::min( gapVblanks - 1u, slotCeiling ) : 0u;
	return dedicatedQueue ? generated : std::min( generated, 1u );
}

// Timestamp costs are keyed by the exact number of outputs in the batch. JIT
// and VRR hybrid always use one; classic pacing is bounded by gap and multiplier.
// Precondition: multiplier >= 2.
[[nodiscard]] constexpr uint32_t ladder_generated_count( uint32_t gapSlots,
	uint32_t multiplier, bool singleSlotPacing )
{
	return singleSlotPacing
		? 1u
		: std::min( gapSlots, std::max( 1u, multiplier - 1u ) );
}

[[nodiscard]] constexpr bool keep_up_interval_eligible( uint64_t frametimeEmaNs,
	uint64_t vblankIntervalNs, uint64_t thresholdPercent )
{
	return frametimeEmaNs * 100u >= vblankIntervalNs * thresholdPercent;
}

[[nodiscard]] constexpr bool jit_interval_eligible( uint64_t frametimeEmaNs,
	uint64_t vblankIntervalNs )
{
	return keep_up_interval_eligible(
		frametimeEmaNs, vblankIntervalNs, k_uJitKeepUpPercent );
}

[[nodiscard]] constexpr bool vrr_hybrid_interval_eligible( uint64_t frametimeEmaNs,
	uint64_t vblankIntervalNs )
{
	return keep_up_interval_eligible(
		frametimeEmaNs, vblankIntervalNs, k_uVrrHybridKeepUpPercent );
}

struct DeadlineLadderState
{
	uint32_t degradeSteps;
	uint32_t holdFrames;
};

struct DeadlineLadderEvaluation
{
	DeadlineLadderState state;
	bool tryDegrade;
};

// Evaluate timestamp maturity, cooldown, and budget without selecting a rung.
// The renderer computes the next rung only when tryDegrade is true, preserving
// the hot path's existing work and keeping GPU capability/state out of this API.
[[nodiscard]] constexpr DeadlineLadderEvaluation evaluate_deadline_ladder(
	DeadlineLadderState state, uint32_t maxDegradeSteps,
	uint64_t currentRungCostNs, uint32_t currentRungSamples,
	uint64_t vblankIntervalNs )
{
	if ( maxDegradeSteps == 0u
		|| currentRungCostNs == 0u
		|| currentRungSamples < k_uDeadlineMinSamples
		|| state.degradeSteps >= maxDegradeSteps )
		return { state, false };

	if ( state.holdFrames > 0u )
	{
		state.holdFrames--;
		return { state, false };
	}

	const uint64_t deadlineNs =
		( vblankIntervalNs * k_uDeadlinePercent ) / 100u;
	return { state, currentRungCostNs > deadlineNs };
}

[[nodiscard]] constexpr bool degradation_reduces_work(
	const EffectiveConfig &current, const EffectiveConfig &next,
	uint32_t currentGeneratedCount, uint32_t nextGeneratedCount )
{
	return next.mode != current.mode
		|| next.quality != current.quality
		|| nextGeneratedCount < currentGeneratedCount;
}

[[nodiscard]] constexpr DeadlineLadderState commit_deadline_degradation(
	DeadlineLadderState state )
{
	state.degradeSteps++;
	state.holdFrames = k_uDeadlineHoldFrames;
	return state;
}

} // namespace gamescope::framegen
