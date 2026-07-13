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

// Fixed-cadence JIT admission learns the renderer's acquire-fence cadence with
// a bounded alpha-beta filter. Four samples are enough to stop treating every
// next slot as uncertain; the one-sided late-error envelope then protects the
// display deadline without charging early frames as risk. The absolute guard
// covers fence notification and compositor scheduling noise after the learned
// source estimate, while the refresh-relative term scales to slower outputs.
inline constexpr uint32_t k_uCadencePredictorMinSamples = 4u;
inline constexpr uint64_t k_ulCadenceArrivalGuardMinNs = 250'000u;
inline constexpr uint64_t k_uCadenceArrivalGuardDivisor = 32u;

struct CadencePredictorState
{
	uint64_t intervalNs = 0;
	int64_t trendNs = 0;
	uint64_t lateErrorNs = 0;
	uint32_t samples = 0;
};

// Predict the next source interval. Runtime samples are bounded by the
// renderer's 250 ms history limit, so the signed conversion and 2x bound cannot
// overflow. The half/double clamp is also the predictor's rate-change guard.
[[nodiscard]] constexpr uint64_t predicted_cadence_interval_ns(
	const CadencePredictorState &state )
{
	if ( state.intervalNs == 0u )
		return 0u;

	const int64_t interval = static_cast<int64_t>( state.intervalNs );
	const int64_t predicted = interval + state.trendNs;
	return static_cast<uint64_t>( std::clamp(
		predicted, std::max<int64_t>( 1, interval / 2 ), interval * 2 ) );
}

// One online alpha-beta update. intervalNs is the filtered period and trendNs
// is its bounded first derivative. A hitch can move the period by at most 1/8
// of a clamped residual and the trend by 1/64, while lateErrorNs remembers the
// recent positive prediction miss as a leaky peak. This is deliberately a
// small deterministic adaptive filter, not a neural model on the render path.
[[nodiscard]] constexpr CadencePredictorState update_cadence_predictor(
	CadencePredictorState state, uint64_t sampleNs )
{
	if ( sampleNs == 0u )
		return state;

	if ( state.intervalNs == 0u )
	{
		state.intervalNs = sampleNs;
		state.samples = 1u;
		return state;
	}

	const uint64_t boundedSample = std::clamp( sampleNs,
		std::max<uint64_t>( 1u, state.intervalNs / 2u ), state.intervalNs * 2u );
	const uint64_t prior = predicted_cadence_interval_ns( state );
	const int64_t residual = static_cast<int64_t>( boundedSample )
		- static_cast<int64_t>( prior );
	const int64_t posterior = std::max<int64_t>( 1,
		static_cast<int64_t>( prior ) + residual / 8 );
	const int64_t maxTrend = std::max<int64_t>( 1, posterior / 16 );

	state.intervalNs = static_cast<uint64_t>( posterior );
	state.trendNs = std::clamp( state.trendNs + residual / 64,
		-maxTrend, maxTrend );
	const uint64_t lateResidual = residual > 0
		? static_cast<uint64_t>( residual ) : 0u;
	state.lateErrorNs = std::max( lateResidual,
		( state.lateErrorNs * 7u ) / 8u );
	if ( state.samples != UINT32_MAX )
		state.samples++;
	return state;
}

struct FixedCadenceAdmission
{
	bool generateBackup;
	bool trained;
	bool predictionOverdue;
	uint64_t predictedReadyNs;
	uint64_t safetyMarginNs;
	uint64_t deadlineHeadroomNs;
};

[[nodiscard]] constexpr uint64_t saturating_add_ns( uint64_t a, uint64_t b )
{
	return a > UINT64_MAX - b ? UINT64_MAX : a + b;
}

// Decide whether the next fixed-refresh slot needs a generated backup. A real
// frame still wins at presentation time; this policy only decides whether it is
// safe to avoid speculative GPU work. Until the source model is trained, or if
// any timestamp is unavailable, smooth cadence wins and a backup is requested.
[[nodiscard]] constexpr FixedCadenceAdmission fixed_cadence_admission(
	uint64_t currentSourceReadyNs, const CadencePredictorState &state,
	uint64_t decisionNowNs, uint64_t targetWakeNs,
	uint64_t vblankIntervalNs )
{
	const uint64_t predictedInterval = predicted_cadence_interval_ns( state );
	const bool trained = state.samples >= k_uCadencePredictorMinSamples;
	if ( currentSourceReadyNs == 0u || predictedInterval == 0u
		|| decisionNowNs == 0u || targetWakeNs == 0u
		|| vblankIntervalNs == 0u )
		return { true, false, false, 0u, 0u, 0u };

	const uint64_t arrivalGuard = std::max( k_ulCadenceArrivalGuardMinNs,
		vblankIntervalNs / k_uCadenceArrivalGuardDivisor );
	const uint64_t safetyMargin = saturating_add_ns(
		state.lateErrorNs, arrivalGuard );
	const uint64_t predictedReady = saturating_add_ns(
		currentSourceReadyNs, predictedInterval );
	const uint64_t protectedReady = saturating_add_ns(
		predictedReady, safetyMargin );
	// Once the protected estimate is in the past, the absence of a newer
	// selected source buffer is itself causal evidence that this prediction was
	// missed. Do not let an already-stale estimate suppress the backup merely
	// because the display deadline is farther in the future.
	const bool predictionOverdue = protectedReady <= decisionNowNs;
	const uint64_t headroom = targetWakeNs > protectedReady
		? targetWakeNs - protectedReady : 0u;

	return {
		.generateBackup = !trained || predictionOverdue
			|| protectedReady >= targetWakeNs,
		.trained = trained,
		.predictionOverdue = predictionOverdue,
		.predictedReadyNs = predictedReady,
		.safetyMarginNs = safetyMargin,
		.deadlineHeadroomNs = headroom,
	};
}

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

[[nodiscard]] constexpr bool keep_up_interval_eligible( uint64_t cadenceNs,
	uint64_t vblankIntervalNs, uint64_t thresholdPercent )
{
	return cadenceNs * 100u >= vblankIntervalNs * thresholdPercent;
}

[[nodiscard]] constexpr bool jit_interval_eligible( uint64_t cadenceNs,
	uint64_t vblankIntervalNs )
{
	return keep_up_interval_eligible(
		cadenceNs, vblankIntervalNs, k_uJitKeepUpPercent );
}

[[nodiscard]] constexpr bool vrr_hybrid_interval_eligible( uint64_t cadenceNs,
	uint64_t vblankIntervalNs )
{
	return keep_up_interval_eligible(
		cadenceNs, vblankIntervalNs, k_uVrrHybridKeepUpPercent );
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
