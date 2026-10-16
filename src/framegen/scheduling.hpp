#pragma once

#include "policy.hpp"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <span>

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
inline constexpr uint32_t k_uDeadlineMissWindowDecisions =
	k_uDeadlineHoldFrames * 4u;

// Recovery is deliberately much slower than degradation. At the real-game
// reference cadence of 45 decisions/s these are approximately two seconds of
// evidence, five seconds of probation, and a one-minute maximum back-off.
inline constexpr uint32_t k_uDeadlineRecoveryHeadroomDecisions = 90u;
inline constexpr uint32_t k_uDeadlineRecoveryBaseBackoffDecisions = 90u;
inline constexpr uint32_t k_uDeadlineRecoveryProbationDecisions = 225u;
inline constexpr uint32_t k_uDeadlineRecoveryMaxBackoffDecisions = 2'700u;
inline constexpr uint64_t k_uDeadlineRecoveryKnownHeadroomPercent = 70u;
inline constexpr uint64_t k_uDeadlineRecoveryColdHeadroomPercent = 50u;

// One-slot deadline pacing measures the two materially different causal
// submission shapes independently.
enum class DeadlineWorkClass_t : uint8_t
{
	FullPreparationAndWarp,
	CachedWarp,
};

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

[[nodiscard]] constexpr uint64_t cadence_arrival_guard_ns( uint64_t vblankIntervalNs )
{
	return std::max( k_ulCadenceArrivalGuardMinNs, vblankIntervalNs / k_uCadenceArrivalGuardDivisor );
}

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

	const uint64_t safetyMargin = saturating_add_ns(
		state.lateErrorNs, cadence_arrival_guard_ns( vblankIntervalNs ) );
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

// The nearest-opportunity rule shared by every queued slot: a slot is due at
// the opportunity within half a vblank after its target, and missed once the
// opportunity being serviced is more than half a vblank past it. Half a vblank
// absorbs clock re-estimation between planning and arbitration without ever
// moving a slot by a whole opportunity.
[[nodiscard]] constexpr bool slot_due_at( uint64_t targetNs, uint64_t opportunityNs,
	uint64_t vblankIntervalNs )
{
	return targetNs <= saturating_add_ns( opportunityNs, vblankIntervalNs / 2u );
}

[[nodiscard]] constexpr bool slot_missed_opportunity( uint64_t targetNs,
	uint64_t opportunityNs, uint64_t vblankIntervalNs )
{
	return saturating_add_ns( targetNs, vblankIntervalNs / 2u ) < opportunityNs;
}

// Nested fixed-grid slot vs. the vblank opportunity a compositor iteration
// services (raw vblank domain, no present bias). Due iff the opportunity is the
// slot's own grid index or later (T/2 absorbs lattice re-estimation between plan
// and arbitration) AND the slot was submitted before the compositor woke for it.
// A slot submitted after that wake had no budget there: it is not due (never a
// miss) and becomes due at the next opportunity. framegen_causal_submit never
// plans such a slot (fixed_grid_slot_submittable), so a due, not-ready slot is
// always capacity evidence.
[[nodiscard]] constexpr bool fixed_grid_slot_due( uint64_t slotVBlankNs, uint64_t submitNs,
	uint64_t opportunityVBlankNs, uint64_t opportunityWakeNs, uint64_t vblankIntervalNs )
{
	return slot_due_at( slotVBlankNs, opportunityVBlankNs, vblankIntervalNs )
		&& submitNs < opportunityWakeNs;
}

// Planner side of the same rule. (a) outside the due window of the opportunity
// the compositor last processed; (b) the slot's estimated wake is more than
// guardNs away. guardNs covers a wake that fired but is not processed yet,
// the timer's re-estimation since arming, and plan-to-submit CPU time
// (command recording included).
[[nodiscard]] constexpr bool fixed_grid_slot_submittable( uint64_t slotVBlankNs,
	uint64_t slotWakeNs, uint64_t wokenVBlankNs, uint64_t nowNs,
	uint64_t vblankIntervalNs, uint64_t guardNs )
{
	return slotVBlankNs > saturating_add_ns( wokenVBlankNs, vblankIntervalNs / 2u )
		&& slotWakeNs > saturating_add_ns( nowNs, guardNs );
}

// Bidir queue ordering. The pending queue is sorted by display target, and real
// targets, which follow record time, increase with frame id.

// First display target at or after candidateNs that stays behind a queued
// target afterNs (0: nothing queued ahead).
[[nodiscard]] constexpr uint64_t bidir_target_after( uint64_t candidateNs, uint64_t afterNs,
	uint64_t vblankIntervalNs )
{
	return afterNs == 0u ? candidateNs
		: std::max( candidateNs, saturating_add_ns( afterNs, vblankIntervalNs ) );
}

// Rounds a duration up to whole vblanks (unchanged for a zero interval).
[[nodiscard]] constexpr uint64_t round_up_to_vblanks( uint64_t durationNs,
	uint64_t vblankIntervalNs )
{
	if ( vblankIntervalNs == 0u )
		return durationNs;
	const uint64_t vblanks = durationNs / vblankIntervalNs
		+ ( durationNs % vblankIntervalNs != 0u );
	return vblanks > UINT64_MAX / vblankIntervalNs
		? UINT64_MAX : vblanks * vblankIntervalNs;
}

// The queue front lies further behind the opportunity than the queue could ever
// catch up (horizonVblanks opportunities). Only a suspended or stalled
// presentation gets there: the queued timeline is a discontinuity, not a
// backlog (0: nothing queued).
[[nodiscard]] constexpr bool bidir_timeline_stale( uint64_t frontTargetNs,
	uint64_t opportunityNs, uint64_t horizonVblanks, uint64_t vblankIntervalNs )
{
	const uint64_t horizonNs = vblankIntervalNs != 0u
		&& horizonVblanks > UINT64_MAX / vblankIntervalNs
		? UINT64_MAX : horizonVblanks * vblankIntervalNs;
	return frontTargetNs != 0u
		&& saturating_add_ns( frontTargetNs, horizonNs ) < opportunityNs;
}

enum class BidirFrontAction_t : uint8_t
{
	Present,	// show the front at this opportunity
	Wait,		// keep it: not due, or this opportunity already showed a front
	Discard,	// an interpolation that can no longer be shown on time
	TooSlow,	// a due interpolation whose GPU work missed its wake (capacity evidence)
};

struct BidirFrontInputs_t
{
	uint64_t targetNs = 0;
	uint64_t opportunityNs = 0;
	uint64_t lastFrontOpportunityNs = 0;	// 0: no front shown yet
	uint64_t vblankIntervalNs = 0;
	bool real = false;
	bool completed = false;		// ignored for reals: their composite is complete
	// The queue is at the output-ring ceiling: a dense timeline, or reals at the
	// most the plan keeps queued.
	bool forceDrain = false;
	// Decided before the compositor woke for this opportunity (a paint between
	// vblanks): unfinished work still had budget left, so it is no capacity
	// evidence.
	bool beforeWake = false;
};

// What the bidir presenter does with its queue front at one display
// opportunity. One queue entry is shown per opportunity; a real is never
// discarded here and catches up one opportunity at a time, while an
// interpolation never slides to a later opportunity.
[[nodiscard]] constexpr BidirFrontAction_t bidir_front_action( const BidirFrontInputs_t &in )
{
	const uint64_t T = in.vblankIntervalNs;
	if ( in.lastFrontOpportunityNs != 0u
		&& in.opportunityNs <= saturating_add_ns( in.lastFrontOpportunityNs, T / 2u ) )
		return BidirFrontAction_t::Wait;
	const bool due = slot_due_at( in.targetNs, in.opportunityNs, T );
	if ( in.real )
		return due || in.forceDrain ? BidirFrontAction_t::Present : BidirFrontAction_t::Wait;
	if ( slot_missed_opportunity( in.targetNs, in.opportunityNs, T ) || ( !due && in.forceDrain ) )
		return BidirFrontAction_t::Discard;
	if ( !due )
		return BidirFrontAction_t::Wait;
	if ( in.completed )
		return BidirFrontAction_t::Present;
	return in.beforeWake ? BidirFrontAction_t::Discard : BidirFrontAction_t::TooSlow;
}

// A front the ring ceiling forced out of the queue ahead of its slot: a real
// shown early, or an interpolation discarded before it was due
// (bidir_front_action). Every other front leaves on or after its slot.
[[nodiscard]] constexpr bool bidir_front_forced( uint64_t targetNs, uint64_t opportunityNs,
	uint64_t vblankIntervalNs )
{
	return !slot_due_at( targetNs, opportunityNs, vblankIntervalNs );
}

// A delayed real shown at the opportunity it was planned for. Its flip measures
// only the display chain, so it is present-bias evidence; a real the queue
// shows early (the ring ceiling) or late (catching up) is off its slot by
// design.
[[nodiscard]] constexpr bool bidir_real_on_slot( uint64_t targetNs, uint64_t opportunityNs,
	uint64_t vblankIntervalNs )
{
	return slot_due_at( targetNs, opportunityNs, vblankIntervalNs )
		&& !slot_missed_opportunity( targetNs, opportunityNs, vblankIntervalNs );
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
// Without a dedicated framegen queue the submitter emits at most one generated
// frame per gap (see generated_slots_for_gap), so the cost key must be clamped
// the same way or the ladder reads a rung slot that is never written.
[[nodiscard]] constexpr uint32_t ladder_generated_count( uint32_t gapSlots,
	uint32_t multiplier, bool singleSlotPacing, bool dedicatedQueue = true )
{
	if ( singleSlotPacing )
		return 1u;
	const uint32_t generated =
		std::min( gapSlots, std::max( 1u, multiplier - 1u ) );
	return dedicatedQueue ? generated : std::min( generated, 1u );
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

// Scene-local recovery state. decisionsSinceClimb also measures from the last
// failed recovery (a degradation), so every new climb is separated from that
// transition by the current back-off. Default construction is the scene reset.
struct RecoveryState_t
{
	uint32_t streak = 0u;
	uint32_t backoffDecisions = k_uDeadlineRecoveryBaseBackoffDecisions;
	uint32_t probationRemaining = 0u;
	uint32_t decisionsSinceClimb = 0u;
	bool blockedThresholdReported = false;
};

struct LadderRecoveryEvaluation_t
{
	RecoveryState_t state;
	bool tryRecover = false;
	bool reportBlockedThreshold = false;
};

struct LadderRungCost_t
{
	uint64_t costNs = 0u;
	uint32_t samples = 0u;
};

struct LadderRecoveryTarget_t
{
	uint32_t rung = 0u;
	LadderRungCost_t evidence;
};

struct DeadlineMissState_t
{
	uint32_t recentMisses = 0u;
};

struct DeadlineMissEvaluation_t
{
	DeadlineMissState_t state;
	bool degrade = false;
	bool skip = false;
};

[[nodiscard]] constexpr uint32_t saturating_increment_decisions(
	uint32_t decisions )
{
	return decisions == UINT32_MAX ? decisions : decisions + 1u;
}

// Advance time for one admission/generation decision. Probation is successful
// when the recovered rung survives its whole window without degrading again.
// Measured-cost admission failures reset evidence but do not themselves
// increase back-off; the existing degradation decision does that if the
// failure forces a step.
[[nodiscard]] constexpr RecoveryState_t advance_ladder_recovery_decision(
	RecoveryState_t state, bool admittedGenerated )
{
	state.decisionsSinceClimb = saturating_increment_decisions(
		state.decisionsSinceClimb );
	if ( state.probationRemaining > 0u )
	{
		state.probationRemaining--;
		if ( state.probationRemaining == 0u && admittedGenerated )
		{
			state.backoffDecisions = std::max(
				k_uDeadlineRecoveryBaseBackoffDecisions,
				state.backoffDecisions / 2u );
		}
	}
	return state;
}

// A causal miss or measured-cost admission failure breaks consecutiveness.
// This form advances the controller's decision clocks;
// reset_ladder_recovery_streak is for a generation-too-slow outcome belonging
// to a decision that was already counted.
[[nodiscard]] constexpr RecoveryState_t note_ladder_recovery_failure(
	RecoveryState_t state )
{
	state = advance_ladder_recovery_decision( state, false );
	state.streak = 0u;
	return state;
}

[[nodiscard]] constexpr RecoveryState_t reset_ladder_recovery_streak(
	RecoveryState_t state )
{
	state.streak = 0u;
	return state;
}

// A degradation during probation is a failed recovery probe. Double the wait
// for the rest of the scene (saturating at one minute), then measure the next
// back-off from this fallback rather than from the preceding climb.
[[nodiscard]] constexpr RecoveryState_t note_ladder_recovery_degradation(
	RecoveryState_t state )
{
	if ( state.probationRemaining > 0u )
	{
		state.backoffDecisions = state.backoffDecisions
			>= k_uDeadlineRecoveryMaxBackoffDecisions / 2u
			? k_uDeadlineRecoveryMaxBackoffDecisions
			: std::min( k_uDeadlineRecoveryMaxBackoffDecisions,
				state.backoffDecisions * 2u );
	}
	state.streak = 0u;
	state.probationRemaining = 0u;
	state.decisionsSinceClimb = 0u;
	state.blockedThresholdReported = false;
	return state;
}

[[nodiscard]] constexpr RecoveryState_t commit_ladder_recovery(
	RecoveryState_t state )
{
	state.streak = 0u;
	state.probationRemaining = k_uDeadlineRecoveryProbationDecisions;
	state.decisionsSinceClimb = 0u;
	state.blockedThresholdReported = false;
	return state;
}

[[nodiscard]] constexpr uint64_t deadline_budget_ns( uint64_t availableNs )
{
	return ( availableNs / 100u ) * k_uDeadlinePercent
		+ ( ( availableNs % 100u ) * k_uDeadlinePercent ) / 100u;
}

[[nodiscard]] constexpr uint64_t fixed_refresh_slot_budget_ns(
	uint64_t vblankIntervalNs, uint64_t presentLeadNs,
	uint64_t presentMarginNs )
{
	const uint64_t reservedNs = presentLeadNs
		> std::numeric_limits<uint64_t>::max() - presentMarginNs
		? std::numeric_limits<uint64_t>::max()
		: presentLeadNs + presentMarginNs;
	return deadline_budget_ns( vblankIntervalNs > reservedNs
		? vblankIntervalNs - reservedNs : 0u );
}

// Distinguish a decision which happened late within an otherwise viable slot
// from evidence that the rung itself exceeds fixed-refresh capacity. The
// actual admission gate still uses remainingBudgetNs; this classification only
// controls what the recovery/degradation learner is allowed to remember.
[[nodiscard]] constexpr bool deadline_shortfall_is_phase_only(
	uint64_t costNs, uint32_t samples,
	uint64_t remainingBudgetNs, uint64_t slotBudgetNs )
{
	return costNs != 0u && samples >= k_uDeadlineMinSamples
		&& costNs > remainingBudgetNs && costNs <= slotBudgetNs;
}

[[nodiscard]] constexpr uint32_t deadline_miss_count( uint32_t misses )
{
	uint32_t count = 0u;
	while ( misses != 0u )
	{
		count += misses & 1u;
		misses >>= 1u;
	}
	return count;
}

// One isolated present miss is an honest skipped decision, not proof that the
// selected pipeline rung is too expensive. A mature cost which cannot fit the
// full slot is direct capacity evidence; otherwise require two misses in the
// small rolling window. Hitch-tagged misses are excluded from both verdict and
// history because the display learner has already classified their cause.
[[nodiscard]] constexpr DeadlineMissEvaluation_t
evaluate_deadline_miss_hysteresis(
	DeadlineMissState_t state, bool missed, bool hitchEpisode,
	uint64_t currentRungCostNs, uint32_t currentRungSamples,
	uint64_t slotBudgetNs )
{
	static_assert( k_uDeadlineMissWindowDecisions > 0u
		&& k_uDeadlineMissWindowDecisions < 32u );
	constexpr uint32_t uWindowMask =
		( uint32_t{ 1u } << k_uDeadlineMissWindowDecisions ) - 1u;
	const bool countMiss = missed && !hitchEpisode;
	state.recentMisses = ( ( state.recentMisses << 1u )
		| static_cast<uint32_t>( countMiss ) ) & uWindowMask;
	const bool matureCostDoesNotFit = currentRungCostNs != 0u
		&& currentRungSamples >= k_uDeadlineMinSamples
		&& currentRungCostNs > slotBudgetNs;
	const bool degrade = countMiss && ( matureCostDoesNotFit
		|| deadline_miss_count( state.recentMisses ) >= 2u );
	return { state, degrade, missed && !degrade };
}

[[nodiscard]] constexpr bool deadline_recovery_headroom(
	uint64_t currentRungCostNs, uint32_t currentRungSamples,
	uint64_t richerRungCostNs, uint32_t richerRungSamples,
	uint64_t budgetNs )
{
	if ( currentRungCostNs == 0u
		|| currentRungSamples < k_uDeadlineMinSamples
		|| budgetNs == 0u )
		return false;

	const bool richerMature = richerRungCostNs != 0u
		&& richerRungSamples >= k_uDeadlineMinSamples;
	if ( richerMature )
	{
		return richerRungCostNs <= budgetNs
			&& currentRungCostNs <=
				( budgetNs / 100u )
					* k_uDeadlineRecoveryKnownHeadroomPercent
				+ ( ( budgetNs % 100u )
					* k_uDeadlineRecoveryKnownHeadroomPercent ) / 100u;
	}

	return currentRungCostNs <=
		( budgetNs / 100u ) * k_uDeadlineRecoveryColdHeadroomPercent
		+ ( ( budgetNs % 100u )
			* k_uDeadlineRecoveryColdHeadroomPercent ) / 100u;
}

// Prefer the richest measured rung which fits the full recovery budget. If no
// richer rung is mature, preserve the adjacent one-rung cold probe. A mature
// but non-fitting rung is returned only as blocking evidence, with no target.
[[nodiscard]] constexpr LadderRecoveryTarget_t select_ladder_recovery_target(
	std::span<const LadderRungCost_t> rungs, uint32_t currentRung,
	uint64_t budgetNs )
{
	LadderRecoveryTarget_t result = { currentRung, {} };
	bool foundMature = false;
	const uint32_t end = std::min<uint32_t>(
		currentRung, static_cast<uint32_t>( rungs.size() ) );
	for ( uint32_t rung = 0u; rung < end; rung++ )
	{
		const LadderRungCost_t sample = rungs[ rung ];
		if ( sample.costNs == 0u || sample.samples < k_uDeadlineMinSamples )
			continue;

		if ( !foundMature )
		{
			foundMature = true;
			result.evidence = sample;
		}
		if ( sample.costNs <= budgetNs )
			return { rung, sample };
	}

	if ( !foundMature && currentRung > 0u )
	{
		result.rung = currentRung - 1u;
		if ( result.rung < rungs.size() )
			result.evidence = rungs[ result.rung ];
	}
	return result;
}

// Observe exactly one admitted/generated decision. The renderer supplies the
// current decision's already-margin-adjusted budget and commits the target
// chosen from the available richer-rung evidence when tryRecover is true.
[[nodiscard]] constexpr LadderRecoveryEvaluation_t evaluate_ladder_recovery(
	RecoveryState_t state, uint32_t currentDegradeSteps,
	uint32_t degradeHoldFrames, uint64_t currentRungCostNs,
	uint32_t currentRungSamples, uint64_t richerRungCostNs,
	uint32_t richerRungSamples, uint64_t budgetNs )
{
	state = advance_ladder_recovery_decision( state, true );
	if ( currentDegradeSteps == 0u
		|| !deadline_recovery_headroom(
			currentRungCostNs, currentRungSamples,
			richerRungCostNs, richerRungSamples, budgetNs ) )
	{
		state.streak = 0u;
		return { state, false };
	}

	state.streak = saturating_increment_decisions( state.streak );
	const bool backoffElapsed = state.decisionsSinceClimb
		>= state.backoffDecisions;
	const bool thresholdReached = state.streak
		>= k_uDeadlineRecoveryHeadroomDecisions;
	const bool blocked = degradeHoldFrames != 0u
		|| state.probationRemaining != 0u || !backoffElapsed;
	const bool reportBlockedThreshold = thresholdReached && blocked
		&& !state.blockedThresholdReported;
	if ( reportBlockedThreshold )
		state.blockedThresholdReported = true;
	return {
		state,
		thresholdReached && !blocked,
		reportBlockedThreshold,
	};
}

// Evaluate timestamp maturity, cooldown, and budget without selecting a rung.
// The renderer computes the next rung only when tryDegrade is true, preserving
// the hot path's existing work and keeping GPU capability/state out of this API.
[[nodiscard]] constexpr DeadlineLadderEvaluation evaluate_deadline_ladder(
	DeadlineLadderState state, uint32_t maxDegradeSteps,
	uint64_t currentRungCostNs, uint32_t currentRungSamples,
	uint64_t vblankIntervalNs )
{
	// Drain the hold before any gate: at the bottom rung (or while costs are
	// missing/immature) the gates below would otherwise return early forever and
	// the hold would never reach zero, permanently blocking recovery.
	const bool bHeld = state.holdFrames > 0u;
	if ( bHeld )
		state.holdFrames--;
	if ( bHeld
		|| maxDegradeSteps == 0u
		|| currentRungCostNs == 0u
		|| currentRungSamples < k_uDeadlineMinSamples
		|| state.degradeSteps >= maxDegradeSteps )
		return { state, false };

	const uint64_t deadlineNs =
		( vblankIntervalNs * k_uDeadlinePercent ) / 100u;
	return { state, currentRungCostNs > deadlineNs };
}

[[nodiscard]] constexpr bool degradation_reduces_work(
	const EffectiveConfig &current, const EffectiveConfig &next,
	uint32_t currentGeneratedCount, uint32_t nextGeneratedCount )
{
	return next.mode != current.mode
		|| next.pipeline != current.pipeline
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
