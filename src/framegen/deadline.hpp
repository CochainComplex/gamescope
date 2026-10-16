#pragma once

#include "scheduling.hpp"
#include "temporal.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <utility>
#include <vector>

namespace gamescope::framegen
{

inline constexpr uint32_t k_uPresentBiasWarmupSamples = 4u;
inline constexpr uint32_t k_uPresentBiasOutlierVblankIntervals = 1u;
inline constexpr uint32_t k_uPresentLeadWarmupSamples = 4u;
inline constexpr uint64_t k_uMinimumViablePresentLeadFloorNs = 500'000u;
inline constexpr uint64_t k_uMinimumViablePresentLeadRelaxDivisor = 64u;
inline constexpr uint64_t k_uMinimumViablePresentLeadBackoffDivisor = 8u;

[[nodiscard]] constexpr uint64_t signed_ns_magnitude( int64_t valueNs )
{
	return valueNs >= 0
		? static_cast<uint64_t>( valueNs )
		: static_cast<uint64_t>( -( valueNs + 1 ) ) + 1u;
}

[[nodiscard]] constexpr int64_t signed_ns_delta(
	uint64_t actualNs, uint64_t targetNs )
{
	if ( actualNs >= targetNs )
	{
		return static_cast<int64_t>( std::min<uint64_t>(
			actualNs - targetNs,
			static_cast<uint64_t>( std::numeric_limits<int64_t>::max() ) ) );
	}

	const uint64_t magnitudeNs = std::min<uint64_t>(
		targetNs - actualNs,
		static_cast<uint64_t>( std::numeric_limits<int64_t>::max() ) );
	return -static_cast<int64_t>( magnitudeNs );
}

[[nodiscard]] constexpr int64_t saturating_add_signed_ns(
	int64_t valueNs, int64_t deltaNs )
{
	if ( deltaNs > 0
		&& valueNs > std::numeric_limits<int64_t>::max() - deltaNs )
		return std::numeric_limits<int64_t>::max();
	if ( deltaNs < 0
		&& valueNs < std::numeric_limits<int64_t>::min() - deltaNs )
		return std::numeric_limits<int64_t>::min();
	return valueNs + deltaNs;
}

template <typename State>
[[nodiscard]] constexpr State update_present_timing_ema_residual(
	State state, int64_t residualNs )
{
	if ( state.samples == 0u )
		state.emaNs = residualNs;
	else
		state.emaNs = saturating_add_signed_ns(
			state.emaNs, residualNs / 8 );
	if ( state.samples != UINT32_MAX )
		state.samples++;
	return state;
}

// Presentation timestamps can sit on either side of the compositor's raw grid
// estimate. Keep the learned correction signed and saturate at the timestamp
// domain boundaries rather than wrapping an absolute deadline.
[[nodiscard]] constexpr uint64_t apply_present_bias_ns(
	uint64_t rawTargetNs, int64_t biasNs )
{
	if ( biasNs >= 0 )
	{
		const uint64_t positiveBiasNs = static_cast<uint64_t>( biasNs );
		return positiveBiasNs > std::numeric_limits<uint64_t>::max() - rawTargetNs
			? std::numeric_limits<uint64_t>::max()
			: rawTargetNs + positiveBiasNs;
	}

	const uint64_t negativeBiasNs = signed_ns_magnitude( biasNs );
	return negativeBiasNs > rawTargetNs ? 0u : rawTargetNs - negativeBiasNs;
}

[[nodiscard]] constexpr uint64_t remove_present_bias_ns(
	uint64_t biasedTargetNs, int64_t biasNs )
{
	if ( biasNs >= 0 )
	{
		const uint64_t positiveBiasNs = static_cast<uint64_t>( biasNs );
		return positiveBiasNs > biasedTargetNs
			? 0u : biasedTargetNs - positiveBiasNs;
	}

	const uint64_t negativeBiasNs = signed_ns_magnitude( biasNs );
	return negativeBiasNs > std::numeric_limits<uint64_t>::max() - biasedTargetNs
		? std::numeric_limits<uint64_t>::max()
		: biasedTargetNs + negativeBiasNs;
}

// The device timestamp table is shared with the classic batch path, so the two
// work classes use two valid single-slot keys while keeping that storage detail
// out of the shared work-class type.
[[nodiscard]] constexpr uint32_t deadline_work_class_cost_key(
	DeadlineWorkClass_t workClass )
{
	return workClass == DeadlineWorkClass_t::FullPreparationAndWarp ? 1u : 2u;
}

struct DisplayGrid_t
{
	uint64_t D0 = 0;
	uint64_t W0 = 0;
	uint64_t T = 0;

	[[nodiscard]] constexpr uint64_t target( uint64_t n ) const
	{
		if ( T != 0u && n > ( std::numeric_limits<uint64_t>::max() - D0 ) / T )
			return std::numeric_limits<uint64_t>::max();
		return D0 + n * T;
	}

	[[nodiscard]] constexpr uint64_t wake( uint64_t n ) const
	{
		if ( T != 0u && n > ( std::numeric_limits<uint64_t>::max() - W0 ) / T )
			return std::numeric_limits<uint64_t>::max();
		return W0 + n * T;
	}
};

[[nodiscard]] constexpr DisplayGrid_t apply_present_bias(
	const DisplayGrid_t &grid, int64_t biasNs )
{
	return {
		.D0 = grid.D0 != 0u ? apply_present_bias_ns( grid.D0, biasNs ) : 0u,
		.W0 = grid.W0 != 0u ? apply_present_bias_ns( grid.W0, biasNs ) : 0u,
		.T = grid.T,
	};
}

struct RealAnchorState_t
{
	uint64_t realFrameId = 0;
	uint64_t sourceReadyNs = 0;
	uint64_t provisionalTargetNs = 0;
	int64_t provisionalBiasNs = 0;
	std::optional<uint64_t> correctedFlipNs;
	uint64_t epoch = 0;

	[[nodiscard]] constexpr uint64_t display_time() const
	{
		return correctedFlipNs.value_or( provisionalTargetNs );
	}

	// The real composite can complete before a nested parent's commit-to-flip
	// lead elapses. Admission therefore retains the raw pre-feedback bound while
	// display targets and wakes carry the learned presentation bias.
	[[nodiscard]] constexpr uint64_t provisional_start_estimate() const
	{
		return remove_present_bias_ns(
			provisionalTargetNs, provisionalBiasNs );
	}
};

enum class DeadlineSkipReason_t : uint8_t
{
	Admitted,
	InvalidGrid,
	EpochMismatch,
	NoCadence,
	MissedWake,
	ForwardCap,
	NextRealSafelyDue,
	SharedSourceUncertain,
};

struct CausalPlanOptions_t
{
	uint64_t nowNs = 0;
	uint64_t afterTargetNs = 0;
	uint64_t gridEpoch = 0;
	// Display targets and generation wakes carry this learned chain bias.
	// Source-cadence admission removes it again so both compared timestamps
	// remain in the pre-display-chain domain.
	int64_t presentBiasNs = 0;
	float configuredStrength = k_flNeutralStrength;
	float forwardStrengthCap = 1.5f;
	bool sourceTimestampsReliable = true;
	bool dedicatedQueue = true;
	bool sharedQueueProvenEmpty = false;
	// Skip a slot displayed at or after the predicted next real frame
	// (phase >= 1): it would extrapolate past content that is due by then.
	bool skipAtOrPastNextReal = false;
};

struct CausalSlotPlan_t
{
	uint64_t targetNs = 0;
	uint64_t wakeNs = 0;
	double phase = 0.0;
	float rawStrength = 0.0f;
	bool admit = false;
	bool provisional = false;
	DeadlineSkipReason_t skipReason = DeadlineSkipReason_t::Admitted;
};

[[nodiscard]] constexpr uint64_t grid_index_at_or_after(
	const DisplayGrid_t &grid, uint64_t timeNs )
{
	if ( grid.T == 0u )
		return 0u;
	if ( timeNs <= grid.D0 )
		return 0u;
	const uint64_t delta = timeNs - grid.D0;
	return delta / grid.T + ( delta % grid.T != 0u );
}

[[nodiscard]] constexpr uint64_t grid_index_strictly_after(
	const DisplayGrid_t &grid, uint64_t timeNs )
{
	if ( grid.T == 0u )
		return 0u;
	return timeNs < grid.D0 ? 0u : ( timeNs - grid.D0 ) / grid.T + 1u;
}

// Plans the earliest unused fixed-grid target strictly after the real anchor.
// afterTargetNs is the most recently planned target (zero when none exists).
[[nodiscard]] inline CausalSlotPlan_t plan_next_causal_slot(
	const DisplayGrid_t &grid, const RealAnchorState_t &anchor,
	const CadencePredictorState &cadence, const CausalPlanOptions_t &options )
{
	CausalSlotPlan_t result;
	if ( grid.T == 0u || grid.D0 == 0u || grid.W0 == 0u )
	{
		result.skipReason = DeadlineSkipReason_t::InvalidGrid;
		return result;
	}
	if ( anchor.epoch == 0u || anchor.epoch != options.gridEpoch )
	{
		result.skipReason = DeadlineSkipReason_t::EpochMismatch;
		return result;
	}

	const uint64_t anchorNs = anchor.display_time();
	const uint64_t usedThroughNs = std::max( anchorNs, options.afterTargetNs );
	const uint64_t n = grid_index_strictly_after( grid, usedThroughNs );
	result.targetNs = grid.target( n );
	result.wakeNs = grid.wake( n );
	result.provisional = !anchor.correctedFlipNs.has_value();
	const uint64_t predictedIntervalNs = predicted_cadence_interval_ns( cadence );
	if ( predictedIntervalNs == 0u || result.targetNs <= anchorNs )
	{
		result.skipReason = DeadlineSkipReason_t::NoCadence;
		return result;
	}

	result.phase = static_cast<double>( result.targetNs - anchorNs )
		/ static_cast<double>( predictedIntervalNs );
	result.rawStrength = forward_strength_raw(
		static_cast<float>( result.phase ), options.configuredStrength );
	if ( options.nowNs >= result.wakeNs )
	{
		result.skipReason = DeadlineSkipReason_t::MissedWake;
		return result;
	}
	if ( result.rawStrength > options.forwardStrengthCap
		|| ( options.skipAtOrPastNextReal
			&& result.targetNs - anchorNs >= predictedIntervalNs ) )
	{
		result.skipReason = DeadlineSkipReason_t::ForwardCap;
		return result;
	}

	if ( !options.sourceTimestampsReliable )
	{
		if ( !options.dedicatedQueue && !options.sharedQueueProvenEmpty )
		{
			result.skipReason = DeadlineSkipReason_t::SharedSourceUncertain;
			return result;
		}
		result.admit = true;
		return result;
	}

	const uint64_t admissionWakeNs = remove_present_bias_ns(
		result.wakeNs, options.presentBiasNs );
	const FixedCadenceAdmission admission = fixed_cadence_admission(
		anchor.sourceReadyNs, cadence, options.nowNs, admissionWakeNs, grid.T );
	if ( !admission.generateBackup )
	{
		result.skipReason = DeadlineSkipReason_t::NextRealSafelyDue;
		return result;
	}

	result.admit = true;
	return result;
}

struct PresentBiasState_t
{
	int64_t emaNs = 0;
	uint32_t samples = 0;
	uint32_t consecutiveGuardExceeds = 0;
};

struct PresentLeadState_t
{
	int64_t emaNs = 0;
	uint32_t samples = 0;
};

// Native KMS can prove that a real commit latched in the same refresh whenever
// its submit-to-flip lead is shorter than one vblank. Remember the lowest such
// lead separately from the habitual-submit EMA: real commits are commonly sent
// much earlier than required, so their average is not a useful latest-safe
// commit point. lowWaterNs relaxes upward slowly; backoffFloorNs is raised by a
// late generated flip and earned back down by subsequent proven real samples.
struct MinimumViablePresentLeadState_t
{
	uint64_t lowWaterNs = 0;
	uint64_t backoffFloorNs = 0;
	uint32_t samples = 0;
};

// The presentation timing learners describe the display chain rather than the
// content being scanned out. D0/W0 advance every refresh, so the stable grid
// identity is its interval together with the backend/connector, VRR state, and
// source-clock provenance. Content invalidation deliberately never appears in
// this key.
struct DisplayChainKey_t
{
	uint64_t backendId = 0;
	uint64_t connectorId = 0;
	uint64_t intervalNs = 0;
	bool vrrActive = false;
	bool sourceTimestampsReliable = false;
};

[[nodiscard]] constexpr bool same_display_chain(
	const DisplayChainKey_t &a, const DisplayChainKey_t &b )
{
	return a.backendId == b.backendId
		&& a.connectorId == b.connectorId
		&& a.intervalNs == b.intervalNs
		&& a.vrrActive == b.vrrActive
		&& a.sourceTimestampsReliable == b.sourceTimestampsReliable;
}

struct DisplayChainTimingState_t
{
	PresentBiasState_t presentBias;
	PresentLeadState_t presentLead;
	MinimumViablePresentLeadState_t minimumViablePresentLead;
	DisplayChainKey_t key;
	uint64_t generation = 0;
	bool initialized = false;
};

struct DisplayChainTimingTransition_t
{
	DisplayChainTimingState_t state;
	bool displayChainChanged = false;
};

// The display time of a raw vblank-clock time on this display chain: raw plus
// the learned present bias (0 stays 0: no target). Only present tags carry it;
// planners and presenters compare raw times.
[[nodiscard]] constexpr uint64_t display_target_ns( uint64_t rawNs,
	const DisplayChainTimingState_t &timing )
{
	return rawNs != 0u ? apply_present_bias_ns( rawNs, timing.presentBias.emaNs ) : 0u;
}

// Re-observing the same key is also the content-invalidation/re-prime path: the
// learning survives untouched. Only a genuine display-chain key transition
// resets all display-chain timing learners and advances the generation used by
// deadline epochs.
[[nodiscard]] constexpr DisplayChainTimingTransition_t observe_display_chain(
	DisplayChainTimingState_t state, const DisplayChainKey_t &key )
{
	DisplayChainTimingTransition_t result = { .state = state };
	result.displayChainChanged = state.initialized
		&& !same_display_chain( state.key, key );
	if ( result.displayChainChanged )
	{
		result.state.presentBias = {};
		result.state.presentLead = {};
		result.state.minimumViablePresentLead = {};
	}
	if ( !state.initialized || result.displayChainChanged )
	{
		result.state.generation++;
		if ( result.state.generation == 0u )
			result.state.generation = 1u;
	}
	result.state.key = key;
	result.state.initialized = true;
	return result;
}

struct AnchorCorrection_t
{
	RealAnchorState_t anchor;
	PresentBiasState_t presentBias;
	int64_t residualNs = 0;
	bool matched = false;
	bool discardProvisional = false;
};

struct PresentBiasUpdate_t
{
	PresentBiasState_t state;
	int64_t residualNs = 0;
	bool discardProvisional = false;
};

// A mature display-chain bias is intentionally slow to replace. A feedback
// residual becomes discard-only evidence once it lies at least one whole
// vblank beyond the magnitude of the correction already learned. Saturate the
// bound so an extreme signed bias cannot wrap it back into the ordinary range.
[[nodiscard]] constexpr uint64_t present_bias_outlier_bound_ns(
	const PresentBiasState_t &state, uint64_t vblankIntervalNs )
{
	const uint64_t biasMagnitudeNs = signed_ns_magnitude( state.emaNs );
	const uint64_t intervalMarginNs = vblankIntervalNs
		> std::numeric_limits<uint64_t>::max()
			/ k_uPresentBiasOutlierVblankIntervals
		? std::numeric_limits<uint64_t>::max()
		: vblankIntervalNs * k_uPresentBiasOutlierVblankIntervals;
	return intervalMarginNs
		> std::numeric_limits<uint64_t>::max() - biasMagnitudeNs
		? std::numeric_limits<uint64_t>::max()
		: biasMagnitudeNs + intervalMarginNs;
}

// Treat the target error as the residual of the bias already applied to this
// provisional anchor. Adding 1/8 of that residual is algebraically the same as
// a 1/8 EMA of (actual flip - raw grid target), while allowing the anchor to
// carry the exact biased target that the scheduler believed.
[[nodiscard]] constexpr PresentBiasUpdate_t update_present_bias(
	PresentBiasState_t state, uint64_t provisionalTargetNs,
	uint64_t actualFlipNs, uint64_t arrivalGuardNs,
	uint64_t vblankIntervalNs, bool hitchEpisode )
{
	PresentBiasUpdate_t result = { .state = state };
	if ( provisionalTargetNs == 0u || actualFlipNs == 0u )
		return result;

	result.residualNs = signed_ns_delta( actualFlipNs, provisionalTargetNs );
	const bool mature = state.samples >= k_uPresentBiasWarmupSamples;
	const bool outsideGuard = signed_ns_magnitude( result.residualNs )
		> arrivalGuardNs;
	if ( mature && outsideGuard )
	{
		if ( result.state.consecutiveGuardExceeds != UINT32_MAX )
			result.state.consecutiveGuardExceeds++;
		result.discardProvisional =
			result.state.consecutiveGuardExceeds >= 2u;
	}
	else
	{
		result.state.consecutiveGuardExceeds = 0u;
	}

	const bool discardEmaSample = mature
		&& ( hitchEpisode
			|| ( vblankIntervalNs != 0u
				&& signed_ns_magnitude( result.residualNs )
					>= present_bias_outlier_bound_ns(
						state, vblankIntervalNs ) ) );
	if ( !discardEmaSample )
	{
		result.state = update_present_timing_ema_residual(
			result.state, result.residualNs );
	}
	return result;
}

[[nodiscard]] constexpr AnchorCorrection_t apply_flip_feedback(
	const RealAnchorState_t &anchor, PresentBiasState_t presentBias,
	uint64_t realFrameId,
	uint64_t actualFlipNs, uint64_t arrivalGuardNs,
	uint64_t vblankIntervalNs, bool hitchEpisode = false )
{
	AnchorCorrection_t result = {
		.anchor = anchor,
		.presentBias = presentBias,
	};
	if ( realFrameId == 0u || realFrameId != anchor.realFrameId || actualFlipNs == 0u )
		return result;

	result.matched = true;
	if ( !anchor.correctedFlipNs.has_value() )
	{
		const PresentBiasUpdate_t update = update_present_bias(
			presentBias, anchor.provisionalTargetNs,
			actualFlipNs, arrivalGuardNs,
			vblankIntervalNs, hitchEpisode );
		result.presentBias = update.state;
		result.residualNs = update.residualNs;
		result.discardProvisional = update.discardProvisional;
	}
	result.anchor.correctedFlipNs = actualFlipNs;
	return result;
}

// Pending entries retain the anchor identity and whether their pixels were
// produced from its provisional display time. A mature, hysteretic correction
// only invalidates those exact pixels; unrelated/newer anchors and
// already-corrected work stay untouched.
[[nodiscard]] constexpr bool discard_pending_provisional_slot(
	const AnchorCorrection_t &correction, uint64_t slotRealFrameId,
	bool slotProvisional )
{
	return correction.matched && correction.discardProvisional
		&& slotProvisional
		&& slotRealFrameId == correction.anchor.realFrameId;
}

// The bidir translation: a real recorded at r is shown at the first grid slot
// at or after r + delayNs.
struct BidirEpoch_t
{
	uint64_t delayNs = 0;	// the bidir delay D
	uint64_t epoch = 0;
	bool valid = false;

	[[nodiscard]] constexpr uint64_t endpoint_display_time( uint64_t recordNs ) const
	{
		return saturating_add_ns( recordNs, delayNs );
	}
};

struct BidirCutEpisodeState_t
{
	bool active = false;
};

struct BidirCutTransition_t
{
	BidirCutEpisodeState_t state;
	bool discardInterpolations = false;
	bool invalidateEpoch = false;
};

// GPU classification can remain asserted for several measurements around one
// cut. Every positive sample still suppresses its interpolations, but only the
// leading edge invalidates the source/display translation. One clean measured
// pair re-arms detection for the next genuine cut.
[[nodiscard]] constexpr BidirCutTransition_t observe_bidir_scene_cut(
	BidirCutEpisodeState_t state, bool sceneCut )
{
	BidirCutTransition_t result = { .state = state };
	result.discardInterpolations = sceneCut;
	result.invalidateEpoch = sceneCut && !state.active;
	result.state.active = sceneCut;
	return result;
}

// Establish the fixed-delay translation: every real is shown D after its record
// time (plan_bidir_step). Record time follows any hold before the record (a
// frame limiter), so a change of that hold moves the reals with it and needs no
// new D. The translation is fixed until the next establishment: flip feedback
// never moves it (FIFO can only report a flip late, so any correction ratchets
// the delay), and already-planned slots are values. Lateness is absorbed by the
// presenter, which shows queued reals one per opportunity and drops late
// interpolations.
[[nodiscard]] constexpr BidirEpoch_t establish_bidir_epoch( uint64_t delayNs, uint64_t gridEpoch )
{
	return {
		.delayNs = delayNs,
		.epoch = gridEpoch,
		.valid = delayNs != 0u && gridEpoch != 0u,
	};
}

struct BidirEndpoint_t
{
	uint64_t realFrameId = 0;
	uint64_t sourceReadyNs = 0;	// source cadence: the pair's validity and the ring span
	uint64_t recordNs = 0;		// when the compositor recorded it: its place on the translation
};

enum class BidirSlotKind_t : uint8_t
{
	Generated,
	RealEndpoint,
};

struct BidirSlot_t
{
	uint64_t targetNs = 0;
	uint64_t wakeNs = 0;
	double phase = 0.0;
	uint64_t referenceFrameId = 0;
	uint64_t endpointFrameId = 0;
	BidirSlotKind_t kind = BidirSlotKind_t::Generated;
};

struct BidirPlan_t
{
	std::vector<BidirSlot_t> slots;
	bool validEpoch = false;
};

struct BidirQueueEntry_t
{
	BidirSlotKind_t kind = BidirSlotKind_t::Generated;
	uint64_t targetNs = 0;
	uint64_t realFrameId = 0;
	uint64_t idealNs = 0;	// a planned real's display time on its translation (0: none)
};

struct BidirQueueShedPlan_t
{
	std::vector<size_t> indices;	// interpolations to shed, ascending
};

// Preserve the old latency bound (two intervals at the configured density),
// but make it subordinate to the physical output ring. Two ring positions stay
// outside the pending timeline for the real composite being acquired and normal
// backend/history progress.
[[nodiscard]] constexpr size_t bidir_pending_hard_capacity(
	size_t outputRingImages, uint32_t multiplier )
{
	const size_t ringCapacity = outputRingImages > 2u
		? outputRingImages - 2u : 0u;
	const size_t latencyCapacity = 2u
		* ( static_cast<size_t>( std::max( 2u, multiplier ) ) + 1u );
	return std::min( ringCapacity, latencyCapacity );
}

// Pending entries pin output-ring slots, so a queue at the ring-derived ceiling
// can starve the composite that would have advanced the timeline. Only then is
// the front forced out ahead of its display target. Below the ceiling the
// planner's grid geometry bounds the depth and every entry flips on its own
// target.
[[nodiscard]] constexpr bool bidir_queue_forces_drain( size_t depth, size_t hardCapacity )
{
	return depth != 0u && depth >= hardCapacity;
}

// Select queued interpolations to remove, oldest first, until capacity is met.
// Reals never leave here: the plan keeps them to what the display can show in
// time (plan_bidir_step), and the presenter drains a queue at the ring ceiling
// early.
[[nodiscard]] inline BidirQueueShedPlan_t plan_bidir_queue_shed(
	std::span<const BidirQueueEntry_t> entries, size_t capacity )
{
	BidirQueueShedPlan_t result;
	size_t remaining = entries.size();
	for ( size_t i = 0u; i < entries.size() && remaining > capacity; i++ )
	{
		if ( entries[ i ].kind == BidirSlotKind_t::Generated )
		{
			result.indices.push_back( i );
			remaining--;
		}
	}
	return result;
}

// Builds a fixed-delay timeline directly from absolute endpoint display times.
// Each interval is independently density-capped; selected candidates retain
// their actual grid targets and timestamp-derived phases.
[[nodiscard]] inline BidirPlan_t plan_bidir_slots(
	const DisplayGrid_t &grid, const BidirEpoch_t &epoch,
	std::span<const BidirEndpoint_t> endpoints, uint32_t multiplier,
	uint64_t gridEpoch )
{
	BidirPlan_t result;
	if ( grid.T == 0u || grid.D0 == 0u || grid.W0 == 0u
		|| !epoch.valid || epoch.epoch == 0u || epoch.epoch != gridEpoch
		|| endpoints.size() < 2u )
		return result;
	result.validEpoch = true;
	const auto insertSlot = [&result]( BidirSlot_t slot ) {
		const auto it = std::lower_bound(
			result.slots.begin(), result.slots.end(), slot.targetNs,
			[]( const BidirSlot_t &entry, uint64_t targetNs ) {
				return entry.targetNs < targetNs;
			} );
		if ( it == result.slots.end() || it->targetNs != slot.targetNs )
		{
			result.slots.insert( it, slot );
			return;
		}

		// Real endpoints own their deadline. Multiple endpoints quantized to the
		// same target coalesce to the newest source endpoint due there.
		if ( slot.kind == BidirSlotKind_t::RealEndpoint
			&& ( it->kind != BidirSlotKind_t::RealEndpoint
				|| slot.endpointFrameId >= it->endpointFrameId ) )
			*it = slot;
	};

	const uint64_t firstIdealNs = epoch.endpoint_display_time( endpoints.front().recordNs );
	const uint64_t firstN = grid_index_at_or_after( grid, firstIdealNs );
	insertSlot( {
		.targetNs = grid.target( firstN ),
		.wakeNs = grid.wake( firstN ),
		.phase = 1.0,
		.endpointFrameId = endpoints.front().realFrameId,
		.kind = BidirSlotKind_t::RealEndpoint,
	} );

	for ( size_t j = 1u; j < endpoints.size(); j++ )
	{
		const uint64_t previousIdealNs = epoch.endpoint_display_time( endpoints[ j - 1u ].recordNs );
		const uint64_t endpointIdealNs = epoch.endpoint_display_time( endpoints[ j ].recordNs );
		if ( endpointIdealNs <= previousIdealNs )
			continue;

		const uint64_t previousEndpointN = grid_index_at_or_after( grid, previousIdealNs );
		const uint64_t endpointN = grid_index_at_or_after( grid, endpointIdealNs );
		std::vector<uint64_t> candidates;
		for ( uint64_t n = previousEndpointN + 1u; n < endpointN; n++ )
		{
			const uint64_t targetNs = grid.target( n );
			if ( targetNs > previousIdealNs && targetNs < endpointIdealNs )
				candidates.push_back( n );
		}

		const size_t ceiling = multiplier > 1u ? multiplier - 1u : 0u;
		const size_t selected = std::min( candidates.size(), ceiling );
		for ( size_t i = 0u; i < selected; i++ )
		{
			const size_t candidateIndex = candidates.size() <= ceiling
				? i : ( ( 2u * i + 1u ) * candidates.size() ) / ( 2u * selected );
			const uint64_t n = candidates[ candidateIndex ];
			const uint64_t targetNs = grid.target( n );
			insertSlot( {
				.targetNs = targetNs,
				.wakeNs = grid.wake( n ),
				.phase = static_cast<double>( targetNs - previousIdealNs )
					/ static_cast<double>( endpointIdealNs - previousIdealNs ),
				.referenceFrameId = endpoints[ j - 1u ].realFrameId,
				.endpointFrameId = endpoints[ j ].realFrameId,
				.kind = BidirSlotKind_t::Generated,
			} );
		}

		insertSlot( {
			.targetNs = grid.target( endpointN ),
			.wakeNs = grid.wake( endpointN ),
			.phase = 1.0,
			.endpointFrameId = endpoints[ j ].realFrameId,
			.kind = BidirSlotKind_t::RealEndpoint,
		} );
	}
	return result;
}

// The bidir delay D. A real recorded at r is shown at the first grid slot at or
// after r + D. Interpolations between the previous real p and the current real
// c can only be planned while their slots lie ahead of the opportunity o that
// plans c, so the pair is fully interpolated iff its pair latency o - r_p is at
// most D. One envelope E of that sample sets D (bidir_target_delay), bounded by
// the longest delay the output ring holds (BidirRingBound_t); every establish,
// admission and drop decision derives from D.
//
// D for envelope value envelopeNs: E rounded up to the grid plus one vblank (0:
// no envelope yet).
[[nodiscard]] constexpr uint64_t bidir_target_delay( uint64_t envelopeNs, uint64_t vblankIntervalNs )
{
	return envelopeNs == 0u ? 0u
		: saturating_add_ns( round_up_to_vblanks( envelopeNs, vblankIntervalNs ), vblankIntervalNs );
}

// Every bidir estimate moves on k (k_uBidirEnvelopeSamples) independent pieces
// of evidence within the last k_ulBidirWindowNs, so fewer than k outliers
// within a window never move one. The protecting direction of an estimate acts
// at once (E rises, the ring bound falls); the relaxing one waits
// k_ulBidirReleaseHoldNs.
inline constexpr uint32_t k_uBidirEnvelopeSamples = 3u;
inline constexpr uint64_t k_ulBidirWindowNs = 1'000'000'000u;
inline constexpr uint64_t k_ulBidirReleaseHoldNs = 2'000'000'000u;
// One sample per plan: the whole window up to 256 plans a second, its newest
// 256 samples above that.
inline constexpr size_t k_nBidirWindowSamples = 256u;
// Real intervals kept for the ring span; the deepest queue (multiplier 4) spans
// 7 of them.
inline constexpr size_t k_nBidirSpacings = 16u;

struct BidirSample_t
{
	uint64_t atNs = 0;		// the opportunity of the plan that took it
	uint64_t latencyNs = 0;	// pair latency: opportunity - previous real's record time (0: none)
	uint64_t sourceSpanNs = 0;	// source-ready time the ring's reals span (0: none)
	uint64_t recordSpanNs = 0;	// record time the ring's reals span (0: none)
};

// The samples of the last k_ulBidirWindowNs, and at least the newest k.
struct BidirWindow_t
{
	std::array<BidirSample_t, k_nBidirWindowSamples> samples = {};	// ring buffer, oldest at first
	size_t first = 0;
	size_t count = 0;
	uint64_t sinceNs = 0;	// the first sample's time (0: none yet)

	constexpr void push( const BidirSample_t &sample )
	{
		if ( sinceNs == 0u )
			sinceNs = sample.atNs;
		if ( count == samples.size() )
			first = ( first + 1u ) % samples.size();
		else
			count++;
		samples[ ( first + count - 1u ) % samples.size() ] = sample;
		while ( count > k_uBidirEnvelopeSamples
			&& saturating_add_ns( samples[ first ].atNs, k_ulBidirWindowNs ) <= sample.atNs )
		{
			first = ( first + 1u ) % samples.size();
			count--;
		}
	}

	// The samples span a whole window at atNs.
	[[nodiscard]] constexpr bool full( uint64_t atNs ) const
	{
		return sinceNs != 0u && atNs >= saturating_add_ns( sinceNs, k_ulBidirWindowNs );
	}

	// The k-th highest (or lowest) non-zero value of field (0: fewer than k).
	template <bool Highest>
	[[nodiscard]] constexpr uint64_t kth( uint64_t BidirSample_t::*field ) const
	{
		std::array<uint64_t, k_uBidirEnvelopeSamples> best = {};
		best.fill( Highest ? 0u : UINT64_MAX );
		size_t values = 0;
		for ( size_t i = 0u; i < count; i++ )
		{
			uint64_t value = samples[ ( first + i ) % samples.size() ].*field;
			if ( value == 0u )
				continue;
			values++;
			for ( uint64_t &kept : best )
			{
				if ( Highest ? value > kept : value < kept )
					std::swap( value, kept );
			}
		}
		return values < best.size() ? 0u : best.back();
	}
	[[nodiscard]] constexpr uint64_t kth_highest_latency() const { return kth<true>( &BidirSample_t::latencyNs ); }
	// The longer of the two clocks' k-th lowest spans (bidir_ring_delay).
	[[nodiscard]] constexpr uint64_t kth_lowest_span() const
	{
		return std::max( kth<false>( &BidirSample_t::sourceSpanNs ), kth<false>( &BidirSample_t::recordSpanNs ) );
	}
};

// The newest real intervals on one clock, each counted at least a vblank: the
// display shows at most one real per vblank.
struct BidirSpacings_t
{
	std::array<uint64_t, k_nBidirSpacings> intervalsNs = {};	// ring buffer
	size_t count = 0;	// intervals pushed so far

	constexpr void push( uint64_t intervalNs )
	{
		intervalsNs[ count++ % intervalsNs.size() ] = intervalNs;
	}

	// The time the newest n intervals span (0: fewer than n so far, or none).
	[[nodiscard]] constexpr uint64_t span( size_t n ) const
	{
		if ( n == 0u || n > intervalsNs.size() || count < n )
			return 0u;
		uint64_t spanNs = 0;
		for ( size_t i = 0u; i < n; i++ )
			spanNs = saturating_add_ns( spanNs, intervalsNs[ ( count - 1u - i ) % intervalsNs.size() ] );
		return spanNs;
	}
};

// Reals a queue of capacity entries holds on the ring bound: k entries stay
// spare for reals recorded closer together than a vblank.
[[nodiscard]] constexpr size_t bidir_ring_reals( size_t capacity )
{
	return capacity > k_uBidirEnvelopeSamples ? capacity - k_uBidirEnvelopeSamples : 0u;
}

// The predicted ring bound: the longest D whose reals fit the queue. A real
// stays queued for about D after its record, so a D no longer than the time n
// consecutive intervals span (n = bidir_ring_reals) queues at most n + 1 reals.
// kthSpanNs is the k-th lowest of the window's disjoint such spans, by
// source-ready time and by record time, whichever is longer: spans short on
// one clock only come from a change of the hold between the two (a frame
// limiter settling, catch-up composites), which the spare entries and the
// forced drain absorb.
// Whole vblanks, rounded down (0: no k spans measured yet, or a ring too small
// for any delayed timeline).
[[nodiscard]] constexpr uint64_t bidir_ring_delay( size_t capacity, uint64_t kthSpanNs,
	uint64_t vblankIntervalNs )
{
	if ( bidir_ring_reals( capacity ) == 0u || vblankIntervalNs == 0u )
		return 0u;
	return kthSpanNs - kthSpanNs % vblankIntervalNs;
}

// The ring bound: the longest D the output ring holds. The predicted bound
// (bidir_ring_delay) sets it until the first translation; after that only the
// ring itself moves it. An overflow is a front the ring ceiling forced out
// ahead of its slot (bidir_front_forced), or a real a plan dropped because the
// ring, not the display, had no room. While an overflow lies within
// k_ulBidirWindowNs the bound falls at once to a predicted bound below the D
// in force; k overflows within that window lower it a vblank when the
// prediction does not. It rises a vblank once k_ulBidirReleaseHoldNs passed
// without an overflow or a move and every plan since left two entries free: a
// vblank more of delay queues at most one more entry, and the ceiling forces a
// front out only at a full queue. It bounds nothing again once it reaches the
// envelope's D. A prediction alone never moves an established bound, so a
// cadence the prediction misjudges cannot make D flap. A queue that came
// within an entry of the ceiling blocks the rise until the next overflow or
// move: a vblank more would fill it. That latch never holds a harmful bound,
// since a D the ring cannot hold overflows, which releases it.
struct BidirRingBound_t
{
	uint64_t delayNs = 0;	// 0: none, D follows the envelope
	std::array<uint64_t, k_uBidirEnvelopeSamples> overflowNs = {};	// the newest k overflows, oldest first (0: none)
	uint64_t quietSinceNs = 0;	// the newest overflow or move of the bound
	size_t deepestQueue = 0;	// the deepest queue a plan left since then
};

// An overflow at atNs under the translation's delayNs (0: none, which bounds
// nothing). The overflows within D of the newest one belong to it: a real
// stays queued for about D, so one excess keeps the queue at the ceiling that
// long.
[[nodiscard]] constexpr BidirRingBound_t observe_bidir_ring_overflow( BidirRingBound_t bound,
	uint64_t atNs, uint64_t delayNs )
{
	const uint64_t newestNs = bound.overflowNs.back();
	if ( delayNs == 0u || ( newestNs != 0u && atNs < saturating_add_ns( newestNs, delayNs ) ) )
		return bound;
	std::shift_left( bound.overflowNs.begin(), bound.overflowNs.end(), 1 );
	bound.overflowNs.back() = atNs;
	bound.quietSinceNs = atNs;
	bound.deepestQueue = 0u;
	return bound;
}

// The ring bound at a plan at atNs: inForceNs is the D in force (the
// translation's, else the envelope's), envelopeNs the envelope's D (0: none
// yet) and predictedNs the predicted bound. A fall spends its overflows.
[[nodiscard]] constexpr BidirRingBound_t update_bidir_ring_bound( BidirRingBound_t bound,
	uint64_t inForceNs, uint64_t envelopeNs, uint64_t predictedNs, size_t capacity, uint64_t atNs,
	uint64_t vblankIntervalNs )
{
	if ( envelopeNs == 0u )
		return { .delayNs = predictedNs, .quietSinceNs = atNs };
	const auto overflowedSince = [&]( uint64_t overflowNs ) {
		return overflowNs != 0u && atNs < saturating_add_ns( overflowNs, k_ulBidirWindowNs );
	};
	if ( overflowedSince( bound.overflowNs.back() ) && predictedNs != 0u && predictedNs < inForceNs )
		return { .delayNs = predictedNs, .quietSinceNs = atNs };
	if ( overflowedSince( bound.overflowNs.front() ) && inForceNs > vblankIntervalNs )
		return { .delayNs = inForceNs - vblankIntervalNs, .quietSinceNs = atNs };
	if ( bound.delayNs != 0u && bound.deepestQueue + 2u <= capacity
		&& atNs >= saturating_add_ns( bound.quietSinceNs, k_ulBidirReleaseHoldNs ) )
	{
		const uint64_t raisedNs = saturating_add_ns( bound.delayNs, vblankIntervalNs );
		return { .delayNs = raisedNs < envelopeNs ? raisedNs : 0u, .quietSinceNs = atNs };
	}
	return bound;
}

// E is the window's k-th highest pair latency K, moved by one rule: it rises to
// K as soon as K lies above D (k samples above D within the window, whatever
// lies between them), from the first sample on, and falls to K once K stayed at
// least a vblank below E for k_ulBidirReleaseHoldNs. D follows E once the
// window first spans k_ulBidirWindowNs (bidir_envelope_delay).
struct BidirEnvelope_t
{
	uint64_t valueNs = 0;		// E (0: not set yet)
	uint64_t belowSinceNs = 0;	// since when K lies at least a vblank below E (0: it does not)
};

[[nodiscard]] constexpr BidirEnvelope_t update_bidir_envelope( BidirEnvelope_t env,
	uint64_t kthNs, uint64_t atNs, uint64_t vblankIntervalNs )
{
	const bool below = kthNs != 0u && saturating_add_ns( kthNs, vblankIntervalNs ) <= env.valueNs;
	const bool released = below && env.belowSinceNs != 0u
		&& atNs >= saturating_add_ns( env.belowSinceNs, k_ulBidirReleaseHoldNs );
	if ( released || ( kthNs != 0u && kthNs > bidir_target_delay( env.valueNs, vblankIntervalNs ) ) )
		return { .valueNs = kthNs };
	return {
		.valueNs = env.valueNs,
		.belowSinceNs = !below ? 0u : env.belowSinceNs != 0u ? env.belowSinceNs : atNs,
	};
}

[[nodiscard]] constexpr uint64_t bidir_envelope_delay( const BidirEnvelope_t &env,
	const BidirWindow_t &window, uint64_t atNs, uint64_t vblankIntervalNs )
{
	return window.full( atNs ) ? bidir_target_delay( env.valueNs, vblankIntervalNs ) : 0u;
}

// How much later than its display time on the translation a queued real is
// shown at the earliest from opportunityNs (0: on time, or not a planned real).
[[nodiscard]] constexpr uint64_t bidir_real_excess( const BidirQueueEntry_t &entry, uint64_t opportunityNs )
{
	const uint64_t shownNs = std::max( entry.targetNs, opportunityNs );
	return entry.kind != BidirSlotKind_t::RealEndpoint || entry.idealNs == 0u || shownNs <= entry.idealNs
		? 0u : shownNs - entry.idealNs;
}

// A backlog drains when its excess over D leaves the range it kept: each new
// lowest or highest excess restarts its window. One whose excess stayed above a
// vblank and within its range for windowNs did not drain.
struct BidirBacklog_t
{
	uint64_t lowestExcessNs = 0;
	uint64_t highestExcessNs = 0;
	uint64_t sinceNs = 0;	// opportunity the range last grew at (0: no backlog)
};

struct BidirBacklogStep_t
{
	BidirBacklog_t backlog;
	bool undrained = false;
};

[[nodiscard]] constexpr BidirBacklogStep_t observe_bidir_backlog( BidirBacklog_t backlog,
	uint64_t excessNs, uint64_t opportunityNs, uint64_t vblankIntervalNs, uint64_t windowNs )
{
	if ( excessNs <= vblankIntervalNs )
		return {};
	if ( backlog.sinceNs == 0u || excessNs < backlog.lowestExcessNs || excessNs > backlog.highestExcessNs )
	{
		return { .backlog = {
			.lowestExcessNs = backlog.sinceNs == 0u ? excessNs : std::min( excessNs, backlog.lowestExcessNs ),
			.highestExcessNs = std::max( excessNs, backlog.highestExcessNs ),
			.sinceNs = opportunityNs,
		} };
	}
	if ( opportunityNs >= saturating_add_ns( backlog.sinceNs, windowNs ) )
		return { .undrained = true };
	return { .backlog = backlog };
}

// The bidir timeline between plans. A discontinuity that keeps the display
// chain (a stale timeline, a scene cut) clears only the translation: the
// samples and the estimates describe the source and the compositor, not the
// dropped timeline.
struct BidirTimelineState_t
{
	BidirEpoch_t epoch;				// record/display translation (valid: established)
	DisplayGrid_t grid;				// grid the translation was established on
	uint64_t gridEpoch = 0;			// tags epoch; 0: never established (treated as invalidated)
	BidirWindow_t window;			// pair latency and ring span samples
	BidirSpacings_t sourceSpacings;	// the newest real intervals by source-ready time
	BidirSpacings_t recordSpacings;	// the newest real intervals by record time
	BidirEnvelope_t pairLatency;	// envelope of opportunity - previous real's record time
	BidirRingBound_t ringBound;		// the longest D whose reals the ring holds
	uint64_t plannedAtNs = 0;		// opportunity of the last plan
	BidirBacklog_t backlog;
};

// What the presenter knows at one display opportunity.
struct BidirPresenterView_t
{
	uint64_t opportunityNs = 0;
	uint64_t lastFrontOpportunityNs = 0;	// 0: no front shown yet
	uint64_t vblankIntervalNs = 0;
	size_t capacity = 0;					// ring-derived hard ceiling
	bool beforeWake = false;
};

// The one place the presenter's facts become bidir_front_action inputs, for the
// presenter (every front it examines) and the planner (the front leaving with
// this paint) alike. depth is the queue's depth with the front still in it.
[[nodiscard]] constexpr BidirFrontInputs_t bidir_front_inputs( const BidirQueueEntry_t &front,
	bool completed, size_t depth, const BidirPresenterView_t &view )
{
	return {
		.targetNs = front.targetNs,
		.opportunityNs = view.opportunityNs,
		.lastFrontOpportunityNs = view.lastFrontOpportunityNs,
		.vblankIntervalNs = view.vblankIntervalNs,
		.real = front.kind == BidirSlotKind_t::RealEndpoint,
		.completed = completed,
		.forceDrain = bidir_queue_forces_drain( depth, view.capacity ),
		.beforeWake = view.beforeWake,
	};
}

// The newest target of a queued real older than realFrameId: nothing of that
// frame is shown before it (0: none).
[[nodiscard]] constexpr uint64_t bidir_real_floor( std::span<const BidirQueueEntry_t> queue,
	uint64_t realFrameId )
{
	uint64_t floorNs = 0;
	for ( const BidirQueueEntry_t &entry : queue )
	{
		if ( entry.kind == BidirSlotKind_t::RealEndpoint && entry.realFrameId < realFrameId )
			floorNs = std::max( floorNs, entry.targetNs );
	}
	return floorNs;
}

// The earliest opportunity the presenter can show a front at: this one, unless
// a front was already shown here.
[[nodiscard]] constexpr uint64_t bidir_earliest_front( const BidirPresenterView_t &view )
{
	return view.lastFrontOpportunityNs == view.opportunityNs
		? saturating_add_ns( view.opportunityNs, view.vblankIntervalNs ) : view.opportunityNs;
}

// The excess of the oldest queued real while it waits behind a real: it was
// displaced behind an older one, or another queued real is due as well (0: it
// does not). A real that is merely late is not queueing: dropping reals could
// not show it any earlier.
[[nodiscard]] constexpr uint64_t bidir_backlog_excess( std::span<const BidirQueueEntry_t> queue,
	const BidirPresenterView_t &view )
{
	const uint64_t T = view.vblankIntervalNs;
	const auto isReal = []( const BidirQueueEntry_t &entry ) { return entry.kind == BidirSlotKind_t::RealEndpoint; };
	const auto oldest = std::ranges::find_if( queue, isReal );
	if ( oldest == queue.end() )
		return 0u;
	const bool displaced = oldest->targetNs > saturating_add_ns( oldest->idealNs, T );
	const bool dueBehind = std::ranges::count_if( queue, [&]( const BidirQueueEntry_t &entry ) {
		return isReal( entry ) && entry.targetNs <= view.opportunityNs; } ) >= 2;
	return displaced || dueBehind ? bidir_real_excess( *oldest, bidir_earliest_front( view ) ) : 0u;
}

// A shorter delay moves every queued real pullNs earlier on the grid: no earlier
// than earliestNs, one per vblank in order, and never later than it was. A real
// queued outside a plan has no display time and keeps its place in that order.
// The interpolations planned for the longer delay go.
inline void pull_bidir_queue( std::vector<BidirQueueEntry_t> &queue, uint64_t pullNs,
	const DisplayGrid_t &grid, uint64_t earliestNs )
{
	std::erase_if( queue, []( const BidirQueueEntry_t &entry ) { return entry.kind == BidirSlotKind_t::Generated; } );
	uint64_t floorNs = 0;
	for ( BidirQueueEntry_t &entry : queue )
	{
		uint64_t targetNs = entry.targetNs;
		if ( entry.idealNs > pullNs )
		{
			entry.idealNs -= pullNs;
			targetNs = std::min( targetNs,
				std::max( grid.target( grid_index_at_or_after( grid, entry.idealNs ) ), earliestNs ) );
		}
		entry.targetNs = bidir_target_after( targetNs, floorNs, grid.T );
		floorNs = entry.targetNs;
	}
}

enum class BidirReestablish_t : uint8_t
{
	Kept,			// the translation stands
	Established,	// a first translation, or one after a discontinuity
	Lengthened,		// D rose; queued reals keep their targets and later reals follow them
	Shortened,		// D fell; queued reals move earlier by the difference (pull_bidir_queue)
};

struct BidirPlanInputs_t
{
	std::span<const BidirQueueEntry_t> queue;	// pending, display order
	BidirEndpoint_t previous, current;
	uint64_t wakeNs = 0;					// compositor wake for presenter.opportunityNs
	uint64_t nowNs = 0;
	uint64_t consumedFrameId = 0;			// newest real that entered the timeline (queued or flipped live)
	BidirPresenterView_t presenter;
	uint32_t multiplier = 2;
	bool frontCompleted = true;			// the queue front's GPU work finished (reals: ignored)
	bool invalidate = false;			// grid interval, display chain or timestamp provenance changed
	bool allowGeneration = false;
};

struct BidirPlannedEndpoint_t
{
	bool queue = false;
	uint64_t targetNs = 0;
	uint64_t idealNs = 0;
	uint64_t wakeNs = 0;
};

// A step's outputs, applied in order: retarget the queued reals (Shortened:
// every queued interpolation goes as well), drop the reals listed, then queue
// the admitted interpolations and the endpoints, and shed to shedCapacity.
struct BidirPlanStep_t
{
	BidirTimelineState_t state;			// the caller commits it whether planned or not
	bool planned = false;				// false: the current real primes, outside the plan
	BidirReestablish_t reestablish = BidirReestablish_t::Kept;
	std::vector<BidirQueueEntry_t> retargeted;	// Shortened: every queued real, in queue order
	std::vector<uint64_t> droppedRealIds;	// queued reals the display cannot show in time
	bool backlogDropped = false;			// a backlog that did not drain dropped some of them
	std::vector<BidirSlot_t> admitted;		// interpolations to generate, display order
	BidirPlannedEndpoint_t previous, current;
	size_t shedCapacity = 0;			// the queue's depth bound once this plan is queued
};

// One bidir plan for the real pair previous -> current, as a pure function of
// the queue, the timeline state and the presenter's view of this opportunity.
//
// Every plan samples the window. The envelope, bounded by the ring, sets D, and
// a D that differs from the translation's re-establishes it: the current real
// is shown D after its record time. A longer D leaves the queued reals on their
// targets; a shorter one moves them earlier by the difference.
//
// Reals never leave for room alone. The display shows at most D/T + k reals
// within D + kT, so the newest queued reals beyond that bound go, one per real
// over it and never the front leaving with this paint; a backlog that did not
// drain drops the reals of its excess the same way. Neither changes D.
//
// Admission keeps the queue within the ring-derived capacity, counting the
// front as leaving only when the presenter's own rule shows it at this
// opportunity. Endpoints never overtake the queued reals older than them and
// are computed from the translation directly, so no real is lost to coalescing.
[[nodiscard]] inline BidirPlanStep_t plan_bidir_step( const BidirTimelineState_t &state,
	const BidirPlanInputs_t &in )
{
	BidirPlanStep_t step = { .state = state };
	BidirTimelineState_t &s = step.state;
	const BidirPresenterView_t &view = in.presenter;
	const uint64_t T = view.vblankIntervalNs;
	const uint64_t opportunityNs = view.opportunityNs;
	if ( T == 0u || in.previous.sourceReadyNs == 0u || in.current.sourceReadyNs <= in.previous.sourceReadyNs
		|| in.previous.recordNs == 0u || in.current.recordNs < in.previous.recordNs )
		return step;

	if ( in.invalidate || s.gridEpoch == 0u )
		s = { .gridEpoch = s.gridEpoch + 1u == 0u ? 1u : s.gridEpoch + 1u };

	// The pair latency is sampled only at the first plan of an opportunity
	// (later ones are catch-up composites) and while no real waits behind a real
	// (that wait is queueing).
	const uint64_t backlogExcessNs = bidir_backlog_excess( in.queue, view );
	const bool latencySample = opportunityNs != s.plannedAtNs && backlogExcessNs == 0u
		&& opportunityNs > in.previous.recordNs;
	s.sourceSpacings.push( std::max( T, in.current.sourceReadyNs - in.previous.sourceReadyNs ) );
	s.recordSpacings.push( std::max( T, in.current.recordNs - in.previous.recordNs ) );
	// A span sample every ringReals reals, so no interval lies in two samples
	// and one burst is one piece of evidence.
	const size_t ringReals = bidir_ring_reals( view.capacity );
	const bool spanSample = ringReals != 0u && s.sourceSpacings.count % ringReals == 0u;
	s.window.push( {
		.atNs = opportunityNs,
		.latencyNs = latencySample ? opportunityNs - in.previous.recordNs : 0u,
		.sourceSpanNs = spanSample ? s.sourceSpacings.span( ringReals ) : 0u,
		.recordSpanNs = spanSample ? s.recordSpacings.span( ringReals ) : 0u,
	} );
	s.plannedAtNs = opportunityNs;
	s.pairLatency = update_bidir_envelope( s.pairLatency, s.window.kth_highest_latency(), opportunityNs, T );
	const uint64_t envelopeDelayNs = bidir_envelope_delay( s.pairLatency, s.window, opportunityNs, T );
	const uint64_t heldNs = s.epoch.valid ? s.epoch.delayNs : 0u;
	s.ringBound = update_bidir_ring_bound( s.ringBound, heldNs != 0u ? heldNs : envelopeDelayNs, envelopeDelayNs,
		bidir_ring_delay( view.capacity, s.window.kth_lowest_span(), T ), view.capacity, opportunityNs, T );
	const uint64_t delayNs = envelopeDelayNs == 0u ? 0u
		: s.ringBound.delayNs != 0u ? std::min( envelopeDelayNs, s.ringBound.delayNs ) : envelopeDelayNs;
	if ( delayNs == 0u )
		return step;

	if ( delayNs != heldNs )
	{
		step.reestablish = heldNs == 0u ? BidirReestablish_t::Established
			: delayNs > heldNs ? BidirReestablish_t::Lengthened : BidirReestablish_t::Shortened;
		s.epoch = establish_bidir_epoch( delayNs, s.gridEpoch );
		s.grid = { .D0 = opportunityNs, .W0 = in.wakeNs, .T = T };
	}

	const std::array endpoints = { in.previous, in.current };
	const BidirPlan_t plan = plan_bidir_slots( s.grid, s.epoch, endpoints, in.multiplier, s.gridEpoch );
	if ( !plan.validEpoch )
		return step;
	step.planned = true;

	std::vector<BidirQueueEntry_t> queue( in.queue.begin(), in.queue.end() );
	if ( step.reestablish == BidirReestablish_t::Shortened )
	{
		pull_bidir_queue( queue, heldNs - delayNs, s.grid, bidir_earliest_front( view ) );
		step.retargeted = queue;
	}
	const BidirBacklogStep_t backlog = observe_bidir_backlog( s.backlog,
		backlogExcessNs, opportunityNs, T, k_ulBidirWindowNs );
	s.backlog = backlog.backlog;
	step.backlogDropped = backlog.undrained;

	// The front leaves with this paint only if the presenter shows it here. A
	// front it would retire counts as staying, which only admits less.
	const size_t leaving = !queue.empty() && bidir_front_action( bidir_front_inputs(
		queue.front(), in.frontCompleted, queue.size(), view ) ) == BidirFrontAction_t::Present ? 1u : 0u;
	// A new translation also queues the previous endpoint, unless it already
	// entered the timeline.
	const bool queuePrevious = step.reestablish != BidirReestablish_t::Kept
		&& in.previous.realFrameId > in.consumedFrameId;

	// The newest staying reals go while the reals exceed what the display shows
	// in time, and once per vblank of an undrained backlog's excess. Any drop is
	// progress for a backlog: its window restarts.
	const auto isReal = []( const BidirQueueEntry_t &entry ) { return entry.kind == BidirSlotKind_t::RealEndpoint; };
	const size_t realBound = std::min<size_t>( view.capacity,
		s.epoch.delayNs / T + k_uBidirEnvelopeSamples );
	const size_t incomingReals = queuePrevious ? 2u : 1u;
	size_t stayingReals = static_cast<size_t>( std::ranges::count_if( queue, isReal ) )
		- ( leaving != 0u && isReal( queue.front() ) ? 1u : 0u );
	uint64_t backlogDrops = backlog.undrained ? ( backlogExcessNs + T - 1u ) / T : 0u;
	// A real dropped while the ring, not the display, bounds the reals is an
	// overflow of the ring bound.
	const bool ringBinds = view.capacity < s.epoch.delayNs / T + k_uBidirEnvelopeSamples;
	bool ringOverflow = false;
	while ( stayingReals != 0u && ( backlogDrops != 0u || stayingReals + incomingReals > realBound ) )
	{
		ringOverflow = ringOverflow || ( backlogDrops == 0u && ringBinds );
		const auto newest = std::ranges::find_if( queue.rbegin(), queue.rend(), isReal ).base() - 1;
		step.droppedRealIds.push_back( newest->realFrameId );
		queue.erase( newest );
		stayingReals--;
		backlogDrops -= backlogDrops != 0u ? 1u : 0u;
		s.backlog = {};
	}
	if ( ringOverflow )
		s.ringBound = observe_bidir_ring_overflow( s.ringBound, opportunityNs, s.epoch.delayNs );

	const size_t committed = queue.size() - leaving + incomingReals;
	const size_t room = view.capacity > committed ? view.capacity - committed : 0u;

	const auto queueEndpoint = [&]( const BidirEndpoint_t &endpoint ) {
		const uint64_t idealNs = s.epoch.endpoint_display_time( endpoint.recordNs );
		const uint64_t n = grid_index_at_or_after( s.grid, idealNs );
		const BidirPlannedEndpoint_t planned = {
			.queue = true,
			.targetNs = bidir_target_after( s.grid.target( n ),
				bidir_real_floor( queue, endpoint.realFrameId ), T ),
			.idealNs = idealNs,
			.wakeNs = s.grid.wake( n ),
		};
		queue.push_back( {
			.kind = BidirSlotKind_t::RealEndpoint,
			.targetNs = planned.targetNs,
			.realFrameId = endpoint.realFrameId,
			.idealNs = idealNs,
		} );
		return planned;
	};
	if ( queuePrevious )
		step.previous = queueEndpoint( in.previous );
	step.current = queueEndpoint( in.current );

	// plan.slots is ordered by display target, so the admitted prefix keeps the
	// timeline dense from the next opportunity onward. Nothing is admitted at or
	// before an older real, or at an occupied target.
	const uint64_t generatedFloorNs = bidir_real_floor( queue, in.current.realFrameId );
	for ( const BidirSlot_t &slot : plan.slots )
	{
		if ( step.admitted.size() >= room )
			break;
		if ( slot.kind != BidirSlotKind_t::Generated
			|| slot.endpointFrameId != in.current.realFrameId
			|| !in.allowGeneration || in.nowNs >= slot.wakeNs
			|| slot.targetNs <= generatedFloorNs
			|| std::ranges::any_of( queue, [&]( const BidirQueueEntry_t &entry ) {
				return entry.targetNs == slot.targetNs; } ) )
			continue;
		step.admitted.push_back( slot );
	}
	step.shedCapacity = view.capacity + leaving;
	s.ringBound.deepestQueue = std::max( s.ringBound.deepestQueue, queue.size() + step.admitted.size() - leaving );
	return step;
}

[[nodiscard]] constexpr bool deadline_cost_fits( uint64_t costNs,
	uint64_t wakeNs, uint64_t nowNs, uint64_t startEstimateNs )
{
	const uint64_t startNs = std::max( nowNs, startEstimateNs );
	if ( startNs >= wakeNs )
		return false;
	const uint64_t budgetNs = wakeNs - startNs;
	const uint64_t jitterBudgetNs = ( budgetNs / 100u ) * k_uDeadlinePercent
		+ ( ( budgetNs % 100u ) * k_uDeadlinePercent ) / 100u;
	return costNs <= jitterBudgetNs;
}

[[nodiscard]] constexpr bool commit_reservation_only_shortfall(
	uint64_t costNs, uint64_t commitDeadlineNs, uint64_t plainWakeNs,
	uint64_t nowNs, uint64_t startEstimateNs )
{
	return commitDeadlineNs < plainWakeNs
		&& !deadline_cost_fits( costNs, commitDeadlineNs,
			nowNs, startEstimateNs )
		&& deadline_cost_fits( costNs, plainWakeNs,
			nowNs, startEstimateNs );
}

struct DeadlineFeedbackSample_t
{
	int64_t signedErrorNs = 0;
	uint64_t absoluteErrorNs = 0;
	bool hit = false;
	bool valid = false;
};

// targetNs is deliberately the final biased target stored in the present tag,
// so this arithmetic measures the deadline the scheduler actually selected.
[[nodiscard]] constexpr DeadlineFeedbackSample_t deadline_feedback_sample(
	uint64_t actualFlipNs, uint64_t targetNs, uint64_t vblankIntervalNs )
{
	DeadlineFeedbackSample_t result;
	if ( actualFlipNs == 0u || targetNs == 0u || vblankIntervalNs == 0u )
		return result;
	result.signedErrorNs = signed_ns_delta( actualFlipNs, targetNs );
	result.absoluteErrorNs = signed_ns_magnitude( result.signedErrorNs );
	result.hit = result.absoluteErrorNs <= vblankIntervalNs / 2u;
	result.valid = true;
	return result;
}

struct PresentLeadSample_t
{
	uint64_t leadNs = 0;
	bool valid = false;
};

// Apply the native KMS learner's exact admission rules separately so metrics
// can describe precisely the observations which are allowed to update it.
[[nodiscard]] constexpr PresentLeadSample_t present_lead_sample(
	PresentLeadState_t state, uint64_t commitSubmitNs, uint64_t actualFlipNs,
	uint64_t vblankIntervalNs, bool hitchEpisode )
{
	PresentLeadSample_t result;
	if ( commitSubmitNs == 0u || actualFlipNs <= commitSubmitNs
		|| vblankIntervalNs == 0u || hitchEpisode )
		return result;

	result.leadNs = actualFlipNs - commitSubmitNs;
	const bool mature = state.samples >= k_uPresentLeadWarmupSamples;
	const uint64_t outlierBoundNs = mature
		? present_bias_outlier_bound_ns(
			PresentBiasState_t{ .emaNs = state.emaNs },
			vblankIntervalNs )
		: vblankIntervalNs;
	result.valid = mature ? result.leadNs < outlierBoundNs
		: result.leadNs <= outlierBoundNs;
	return result;
}

// A deliberately slow 1/8 EMA. Invalid/reordered timestamp pairs do not
// contaminate the backend lead estimate. This unguarded overload is retained
// for backends which do not own KMS timing, where changing the established
// presentation learner would also change nested pacing.
[[nodiscard]] constexpr PresentLeadState_t update_present_lead(
	PresentLeadState_t state, uint64_t commitSubmitNs, uint64_t actualFlipNs )
{
	if ( commitSubmitNs == 0u || actualFlipNs <= commitSubmitNs )
		return state;

	const int64_t sampleNs = signed_ns_delta( actualFlipNs, commitSubmitNs );
	// Both values are non-negative by construction, so their difference is
	// representable in the signed timestamp range.
	const int64_t residualNs = sampleNs - state.emaNs;
	return update_present_timing_ema_residual( state, residualNs );
}

// Native KMS commit-to-flip samples share the display-chain lifetime of the
// present-bias learner. Do not seed a fresh EMA from a loading hitch, and once
// mature reject the same "current estimate plus one vblank" class of outlier
// used by the present-bias guard. During warmup, one vblank is the only safe
// bound because there is no trustworthy estimate yet.
[[nodiscard]] constexpr PresentLeadState_t update_present_lead(
	PresentLeadState_t state, uint64_t commitSubmitNs, uint64_t actualFlipNs,
	uint64_t vblankIntervalNs, bool hitchEpisode )
{
	if ( !present_lead_sample( state, commitSubmitNs, actualFlipNs,
		vblankIntervalNs, hitchEpisode ).valid )
		return state;

	return update_present_lead( state, commitSubmitNs, actualFlipNs );
}

[[nodiscard]] constexpr uint64_t minimum_viable_present_lead_ns(
	const MinimumViablePresentLeadState_t &state )
{
	return std::max( state.lowWaterNs, state.backoffFloorNs );
}

// Consume only the same valid, non-hitch real-commit observations as the
// native habitual learner, and additionally require the strict sub-vblank
// proof of a same-vblank latch. The minimum rises by at most 1/64 refresh per
// proof, so an isolated unusually-low sample expires without a stall or an
// early habitual commit destroying the useful evidence in one update.
[[nodiscard]] constexpr MinimumViablePresentLeadState_t
update_minimum_viable_present_lead(
	MinimumViablePresentLeadState_t state, PresentLeadState_t habitualLead,
	uint64_t commitSubmitNs, uint64_t actualFlipNs,
	uint64_t vblankIntervalNs, bool hitchEpisode )
{
	const PresentLeadSample_t sample = present_lead_sample(
		habitualLead, commitSubmitNs, actualFlipNs,
		vblankIntervalNs, hitchEpisode );
	if ( !sample.valid || sample.leadNs >= vblankIntervalNs )
		return state;

	const uint64_t relaxNs = std::max<uint64_t>(
		1u, vblankIntervalNs / k_uMinimumViablePresentLeadRelaxDivisor );
	if ( state.samples == 0u )
	{
		state.lowWaterNs = sample.leadNs;
	}
	else
	{
		const uint64_t relaxedLowWaterNs = state.lowWaterNs
			> std::numeric_limits<uint64_t>::max() - relaxNs
			? std::numeric_limits<uint64_t>::max()
			: state.lowWaterNs + relaxNs;
		state.lowWaterNs = std::min( sample.leadNs, relaxedLowWaterNs );
	}

	if ( state.backoffFloorNs != 0u )
	{
		const uint64_t relaxedBackoffNs = state.backoffFloorNs > relaxNs
			? state.backoffFloorNs - relaxNs : 0u;
		state.backoffFloorNs = relaxedBackoffNs > state.lowWaterNs
			? relaxedBackoffNs : 0u;
	}
	if ( state.samples != UINT32_MAX )
		state.samples++;
	return state;
}

// A missed generated latch means the current probe was too late. Retreat by
// 1/8 refresh, never earlier than the habitual real-commit EMA. Successful
// real commits subsequently relax this floor by the slower 1/64 step above.
[[nodiscard]] constexpr MinimumViablePresentLeadState_t
back_off_minimum_viable_present_lead(
	MinimumViablePresentLeadState_t state, PresentLeadState_t habitualLead,
	uint64_t vblankIntervalNs )
{
	if ( state.samples < k_uPresentLeadWarmupSamples
		|| habitualLead.samples < k_uPresentLeadWarmupSamples
		|| vblankIntervalNs == 0u )
		return state;

	const uint64_t habitualNs = static_cast<uint64_t>(
		std::max<int64_t>( 0, habitualLead.emaNs ) );
	const uint64_t currentNs = minimum_viable_present_lead_ns( state );
	if ( currentNs >= habitualNs )
		return state;

	const uint64_t stepNs = std::max<uint64_t>(
		1u, vblankIntervalNs / k_uMinimumViablePresentLeadBackoffDivisor );
	const uint64_t backedOffNs = stepNs > habitualNs - currentNs
		? habitualNs : std::min( habitualNs, currentNs + stepNs );
	state.backoffFloorNs = backedOffNs > state.lowWaterNs
		? backedOffNs : 0u;
	return state;
}

// Preserve the learner's warm-up gate while allowing a diagnostic lead to
// replace only its mature value. Zero is the no-override sentinel.
[[nodiscard]] constexpr PresentLeadState_t apply_present_lead_override(
	PresentLeadState_t presentLead, uint64_t overrideLeadNs )
{
	if ( overrideLeadNs != 0u
		&& presentLead.samples >= k_uPresentLeadWarmupSamples )
	{
		presentLead.emaNs = static_cast<int64_t>( std::min<uint64_t>(
			overrideLeadNs,
			static_cast<uint64_t>( std::numeric_limits<int64_t>::max() ) ) );
	}
	return presentLead;
}

struct FixedRefreshCommitPlan_t
{
	uint64_t commitDeadlineNs = 0;
	uint64_t presentLeadNs = 0;
	uint64_t advanceNs = 0;
	bool earlyCommit = false;
	bool learnedLead = false;
	bool overriddenLead = false;
};

struct FixedRefreshPresentLead_t
{
	uint64_t leadNs = 0;
	bool ready = false;
	bool learned = false;
	bool overridden = false;
};

[[nodiscard]] constexpr FixedRefreshPresentLead_t
select_fixed_refresh_present_lead(
	PresentLeadState_t habitualLead,
	MinimumViablePresentLeadState_t minimumViableLead,
	uint64_t presentMarginNs, uint64_t vblankIntervalNs,
	uint64_t overrideLeadNs = 0u,
	uint64_t floorNs = k_uMinimumViablePresentLeadFloorNs )
{
	FixedRefreshPresentLead_t result;
	if ( vblankIntervalNs == 0u
		|| habitualLead.samples < k_uPresentLeadWarmupSamples )
		return result;

	const uint64_t latestSafeLeadNs = vblankIntervalNs > presentMarginNs
		? vblankIntervalNs - presentMarginNs : 0u;
	if ( overrideLeadNs != 0u )
	{
		result.leadNs = std::min( overrideLeadNs, latestSafeLeadNs );
		result.ready = true;
		result.overridden = true;
		return result;
	}

	if ( minimumViableLead.samples >= k_uPresentLeadWarmupSamples )
	{
		const uint64_t habitualNs = static_cast<uint64_t>(
			std::max<int64_t>( 0, habitualLead.emaNs ) );
		result.leadNs = std::min(
			std::max( minimum_viable_present_lead_ns(
				minimumViableLead ), floorNs ),
			habitualNs );
		result.learned = true;
	}
	else
	{
		result.leadNs = static_cast<uint64_t>(
			std::max<int64_t>( 0, habitualLead.emaNs ) );
	}
	result.leadNs = std::min( result.leadNs, latestSafeLeadNs );
	result.ready = true;
	return result;
}

[[nodiscard]] constexpr bool minimum_viable_present_lead_needs_backoff(
	const FixedRefreshPresentLead_t &selectedLead,
	uint64_t actualFlipNs, uint64_t targetFlipNs,
	uint64_t vblankIntervalNs )
{
	if ( !selectedLead.learned || selectedLead.overridden )
		return false;
	const DeadlineFeedbackSample_t sample = deadline_feedback_sample(
		actualFlipNs, targetFlipNs, vblankIntervalNs );
	return sample.valid && sample.signedErrorNs
		> static_cast<int64_t>( vblankIntervalNs / 2u );
}

// Fixed-refresh content remains attached to its display-grid target D. Once the
// display-chain learner is warm, its native generation/commit deadline is
// D - commit-to-flip lead - margin. During warmup the arbiter retains its normal
// vblank opportunity, so a zero deadline explicitly means "no early commit".
[[nodiscard]] constexpr FixedRefreshCommitPlan_t plan_fixed_refresh_commit(
	uint64_t targetFlipNs, PresentLeadState_t habitualLead,
	MinimumViablePresentLeadState_t minimumViableLead,
	uint64_t presentMarginNs, uint64_t vblankIntervalNs,
	uint64_t overrideLeadNs = 0u )
{
	FixedRefreshCommitPlan_t result;
	if ( targetFlipNs == 0u )
		return result;

	const FixedRefreshPresentLead_t selected =
		select_fixed_refresh_present_lead(
			habitualLead, minimumViableLead,
			presentMarginNs, vblankIntervalNs, overrideLeadNs );
	if ( !selected.ready )
		return result;
	result.presentLeadNs = selected.leadNs;
	result.learnedLead = selected.learned;
	result.overriddenLead = selected.overridden;
	result.advanceNs = saturating_add_ns(
		result.presentLeadNs, presentMarginNs );
	if ( result.advanceNs == 0u || result.advanceNs >= targetFlipNs )
		return result;

	result.commitDeadlineNs = targetFlipNs - result.advanceNs;
	result.earlyCommit = true;
	return result;
}

// Retain the established EMA-only form byte-for-byte for tests and policy
// helpers outside native KMS. Native KMS uses the overload above with its
// minimum-viable state and the stricter one-vblank-minus-margin cap.
[[nodiscard]] constexpr FixedRefreshCommitPlan_t plan_fixed_refresh_commit(
	uint64_t targetFlipNs, PresentLeadState_t presentLead,
	uint64_t presentMarginNs, uint64_t vblankIntervalNs )
{
	FixedRefreshCommitPlan_t result;
	if ( targetFlipNs == 0u
		|| vblankIntervalNs == 0u
		|| presentLead.samples < k_uPresentLeadWarmupSamples )
		return result;

	result.presentLeadNs = std::min<uint64_t>(
		static_cast<uint64_t>( std::max<int64_t>( 0, presentLead.emaNs ) ),
		vblankIntervalNs );
	result.advanceNs = saturating_add_ns(
		result.presentLeadNs, presentMarginNs );
	if ( result.advanceNs == 0u || result.advanceNs >= targetFlipNs )
		return result;

	result.commitDeadlineNs = targetFlipNs - result.advanceNs;
	result.earlyCommit = true;
	return result;
}

// A generated early commit may block until scanout on native KMS. Only start it
// after its deadline when the compositor has no real frame ready to paint and
// no earlier present is still in flight; the next real composite then keeps
// absolute priority at the arbiter.
[[nodiscard]] constexpr bool can_start_early_generated_commit(
	bool commitDeadlineReached, bool realCompositePending,
	bool presentInFlight )
{
	return commitDeadlineReached
		&& !realCompositePending
		&& !presentInFlight;
}

// Composite requests are the overlay-plane repaint and forced repaints.
// steamcompmgr calls step_repaint_requests() once per iteration: it folds the
// previous iteration's paint log, then plans this one.
// - A request is serviced once a composite of the live layer stack made
//   after it was raised has been presented. Without bidir that is this
//   iteration's Presented paint. Under bidir a held composite becomes the
//   carrier of the requests it consumed until a paint presents it or anything
//   newer (PaintReport_t).
// - Failed drops the request as upstream does, whatever the paint was for;
//   Deferred is transient and changes nothing.
// - A pending request yields to generated output in at most
//   k_uMaxRepaintYields vblank-timer periods, counted once per period however
//   many wakes (deadline, vblank, VRR nudge) it contains.
enum class PaintOutcome_t : uint8_t { Presented, Deferred, Failed };
enum class PaintOutput_t : uint8_t { Live, Generated, Held };
enum class FlipVerdict_t : uint8_t { NotFlipped, Live, Older };
// What one paint's bidir flip substitution reports (composite seqs in device
// order, 0 = none): whether it flipped the composite it was handed, that
// composite of the live layer stack, and the newest queued real composite the
// paint put on screen (flip substitution or consume path). Framegen off and
// non-bidir paints report nothing.
struct PaintReport_t
{
	FlipVerdict_t verdict = FlipVerdict_t::NotFlipped;
	uint64_t composited = 0;
	uint64_t presented = 0;
};
// What a paint the arbiter did not select does while a prediction waits.
enum class UnselectedPaint_t : uint8_t { Paint, Yield, DiscardPrediction };

inline constexpr uint32_t k_uMaxRepaintYields = 4u;
// Over a drained bidir queue a request-only paint is flipped live
// (select_bidir_flip). If it is held instead, the presenter contract broke:
// retry once, then drop the held requests like a failed paint.
inline constexpr uint32_t k_uMaxBidirRefreshMisses = 2u;

[[nodiscard]] constexpr PaintOutput_t classify_paint_output(
	bool generatedPaint, bool baseLayer, bool bidirActive, FlipVerdict_t flip )
{
	if ( flip != FlipVerdict_t::NotFlipped )
		return flip == FlipVerdict_t::Live ? PaintOutput_t::Live : PaintOutput_t::Held;
	if ( bidirActive )	// the consume path presented a queued front
		return PaintOutput_t::Held;
	if ( generatedPaint )
		return baseLayer ? PaintOutput_t::Live : PaintOutput_t::Generated;
	return PaintOutput_t::Live;
}

// Advanced only by step_repaint_requests().
struct RepaintRequestState_t
{
	uint32_t yields = 0;
	uint64_t yieldVBlankNs = 0;
	uint64_t heldForcedRequests = 0;	// forced snapshot whose one commit was held
	uint32_t refreshMisses = 0;
	uint64_t overlayCarrierSeq = 0;		// newest held composite that consumed the overlay flag
	uint64_t forcedCarrierSeq = 0;		// first held composite carrying the held force
};

// Sampled once per iteration, after the nIgnoredOverlayRepaints resets.
struct RepaintInputs_t
{
	bool overlayPending = false;		// hasRepaintNonBasePlane
	uint64_t forcedRequests = 0;		// g_ulForceRepaintRequests
	uint64_t targetVBlankNs = 0;		// vblank timer target: one key per refresh period
	bool vblank = false;
	bool realRepaint = false;		// hasRepaint
	uint32_t ignoredOverlayRepaints = 0;
	bool bidirActive = false;
	bool bidirQueueDrained = false;
	bool baseLayerActive = false;
	bool backendPaused = false;		// VT switched away: nothing can be presented
};

struct RepaintPlan_t
{
	bool overlayDrivesPaint = false;
	bool forcedDrivesPaint = false;
	bool exhausted = false;			// overrides the legacy VRR overlay throttle
	bool generatedMayTakeSlot = true;
	UnselectedPaint_t unselectedPaint = UnselectedPaint_t::Paint;
	bool dropHeld = false;			// the held requests were dropped
};

// One iteration: what it was planned from (in, state; never written here) and
// the log of its paints, which the next step_repaint_requests() folds.
struct RepaintIteration_t
{
	RepaintInputs_t in;
	RepaintRequestState_t state;
	bool yielded = false;		// a pending request was withheld for generated output
	bool ended = false;		// some paint was not Deferred: upstream end of frame
	bool retiredOverlay = false;	// the overlay flag is consumed: presented, carried, failed or dropped
	bool retiredForced = false;	// the force was serviced by the vblank paint that took it, or dropped
	bool held = false;
	uint64_t newOverlayCarrierSeq = 0;	// the held composite that consumed the overlay flag here
	uint64_t newForcedCarrierSeq = 0;	// the held composite that took the held force here
	// Carriers at or below this composite seq are retired: the newest queued
	// real composite put on screen (seqs grow in composite order), or anything
	// a live or failed paint ended.
	uint64_t retiredThroughSeq = 0;

	// The carriers outstanding after the paints so far (0 = none).
	[[nodiscard]] constexpr uint64_t overlay_carrier() const
	{
		const uint64_t seq = newOverlayCarrierSeq != 0u ? newOverlayCarrierSeq : state.overlayCarrierSeq;
		return retiredThroughSeq >= seq ? 0u : seq;
	}
	[[nodiscard]] constexpr uint64_t forced_carrier() const
	{
		if ( retiredForced )
			return 0u;
		return newForcedCarrierSeq != 0u ? newForcedCarrierSeq : state.forcedCarrierSeq;
	}
	// The force snapshot already had its one commit held.
	[[nodiscard]] constexpr bool forced_held() const
	{
		return in.forcedRequests == state.heldForcedRequests;
	}

	// A request outlived this iteration's retirements; only then has it an age.
	[[nodiscard]] constexpr bool request_remains() const
	{
		return ( in.overlayPending && !retiredOverlay ) || overlay_carrier() != 0u
			|| ( in.forcedRequests != 0u && !retiredForced );
	}

	constexpr void add_paint( PaintOutput_t output, PaintOutcome_t outcome, bool forcedPaint,
		const PaintReport_t &report = {} )
	{
		if ( outcome == PaintOutcome_t::Deferred )
			return;
		ended = true;
		if ( outcome == PaintOutcome_t::Failed )
			output = PaintOutput_t::Live;	// dropped like upstream: never held or yielded
		// Presented real seqs never decrease, so presenting a composite services
		// what it or any older one carried.
		retiredThroughSeq = std::max( retiredThroughSeq, report.presented );
		if ( forced_carrier() != 0u && retiredThroughSeq >= forced_carrier() )
			retiredForced = true;
		switch ( output )
		{
			case PaintOutput_t::Live:
				// Its own composite is newer than any carrier; a failed paint
				// drops them like upstream.
				retiredOverlay = true;
				retiredThroughSeq = std::max( retiredThroughSeq, overlay_carrier() );
				// Upstream consumes a force only on the vblank paint that took it.
				retiredForced = retiredForced || forcedPaint || forced_carrier() != 0u;
				break;
			case PaintOutput_t::Generated:
				yielded = true;
				break;
			case PaintOutput_t::Held:
				held = true;
				// Its composite of the live layer stack carries the pending requests.
				if ( report.composited == 0u )
					break;
				if ( in.overlayPending && !retiredOverlay )
				{
					retiredOverlay = true;
					newOverlayCarrierSeq = report.composited;
				}
				// Only once a vblank paint took the force, as upstream consumes it.
				if ( in.forcedRequests != 0u && !retiredForced && forced_carrier() == 0u
					&& ( forcedPaint || forced_held() ) )
					newForcedCarrierSeq = report.composited;
				break;
		}
	}

	// The repaint guard: a paint the arbiter did not select while a prediction waits.
	[[nodiscard]] constexpr UnselectedPaint_t take_unselected_paint( const RepaintPlan_t &plan )
	{
		yielded = yielded || plan.unselectedPaint == UnselectedPaint_t::Yield;
		return plan.unselectedPaint;
	}
};

// Clears only the snapshot painted for; a request raised since survives.
inline void acknowledge_forced_requests( std::atomic<uint64_t> &requests, const RepaintIteration_t &iteration )
{
	uint64_t ulExpected = iteration.in.forcedRequests;
	if ( iteration.retiredForced )
		requests.compare_exchange_strong( ulExpected, 0u );
}

struct RepaintStep_t
{
	RepaintRequestState_t state;
	RepaintPlan_t plan;
	RepaintIteration_t iteration;
};

// Only retirement clears a request, so age is carried exactly while some
// request outlives the previous iteration (an off-vblank paint retires the
// overlay but not the force), and a request raised during a retiring paint
// starts fresh. The outstanding carriers are folded from prev alone (its
// accessors), overwriting those of the state argument, and the returned
// iteration is planned from the returned state (step.iteration.state ==
// step.state).
[[nodiscard]] constexpr RepaintStep_t step_repaint_requests(
	RepaintRequestState_t state, const RepaintIteration_t &prev, const RepaintInputs_t &now )
{
	if ( !prev.request_remains() )
		state = {};
	else
	{
		// A retired force was acknowledged: the counter may return to the snapshot.
		if ( prev.retiredForced )
			state.heldForcedRequests = 0u;
		else if ( prev.held )
			state.heldForcedRequests = prev.in.forcedRequests;
		// A legitimate new-base hold is not a miss; only a request-only paint is.
		if ( prev.held && prev.in.bidirQueueDrained && !prev.in.realRepaint )
			state.refreshMisses = std::min( state.refreshMisses + 1u, k_uMaxBidirRefreshMisses );
		if ( prev.yielded && prev.in.targetVBlankNs != state.yieldVBlankNs )
		{
			state.yields = std::min( state.yields + 1u, k_uMaxRepaintYields );
			state.yieldVBlankNs = prev.in.targetVBlankNs;
		}
		state.overlayCarrierSeq = prev.overlay_carrier();
		state.forcedCarrierSeq = prev.forced_carrier();
	}

	// Behind a non-empty bidir queue no request-driven composite can be live, so
	// none is made: the next real composite carries the requests. A newly raised
	// force still gets one commit (screen sleep and HDR metadata ride any commit).
	const bool bidirWaits = now.bidirActive && !now.bidirQueueDrained;
	// The cap ends the requests whose refreshes were held. An overlay or a
	// force raised since is a new request and is spared.
	const bool dropHeld = now.bidirActive && state.refreshMisses >= k_uMaxBidirRefreshMisses;
	const bool forcedHeld = now.forcedRequests == state.heldForcedRequests;
	// A new force snapshot is acknowledged as a whole: its CAS covers the old one.
	if ( !forcedHeld )
		state.forcedCarrierSeq = 0u;
	// A carried overlay still unpresented once the queue drains (its composite
	// was never queued, or was shed) drives one refresh.
	const bool overlayPending = now.overlayPending || state.overlayCarrierSeq != 0u;
	// Inert until generation made the request yield, so framegen OFF keeps the
	// legacy VRR throttle; afterwards ignored VRR cycles count toward its age.
	const bool exhausted = state.yields != 0u
		&& now.ignoredOverlayRepaints >= k_uMaxRepaintYields - state.yields;
	const bool pending = overlayPending || now.forcedRequests != 0u;
	const RepaintPlan_t plan = {
		.overlayDrivesPaint = overlayPending && !bidirWaits && !dropHeld,
		.forcedDrivesPaint = now.vblank && now.forcedRequests != 0u
			&& !( ( bidirWaits || dropHeld ) && forcedHeld ),
		.exhausted = exhausted,
		// A ready prediction takes the slot unless a pending request exhausted its
		// yields. Bidir cannot be pre-empted (its timeline is ordered); a base-layer
		// frame composites the live overlays itself. Nothing takes a slot while
		// the backend is paused.
		.generatedMayTakeSlot = !now.backendPaused
			&& ( !pending || now.bidirActive || now.baseLayerActive || !exhausted ),
		.unselectedPaint = now.bidirActive ? UnselectedPaint_t::Paint
			: exhausted ? UnselectedPaint_t::DiscardPrediction : UnselectedPaint_t::Yield,
		.dropHeld = dropHeld,
	};
	if ( dropHeld )
		state = {};
	const RepaintIteration_t iteration = {
		.in = now,
		.state = state,
		.retiredOverlay = dropHeld && prev.in.overlayPending && !prev.retiredOverlay,
		.retiredForced = dropHeld && forcedHeld,
	};
	return { state, plan, iteration };
}

// Bidir flip selection. Content order is kept and no composite of a new base is
// left unqueued unless a newer one supersedes it before it was shown. Behind
// queued content the queue front is presented: a composite that queued its real
// frame is already behind it; a new base that queued nothing (the prime after a
// cut or focus change, an unplanned pair) is enqueued behind the retained reals
// (QueueBehind), or replaces the queued-behind tail it supersedes, or the
// newest real of a queue whose reals fill the ring (ReplaceBehind), so
// unplanned composites hold at most one queued-behind entry, never take the
// queue past the ring, and the queue drains at one entry per refresh. A
// re-render of the newest base shows the front while that base is still
// queued. With nothing queued a composite cannot reorder content and is flipped
// live: RefreshLive for the base on screen, SnapLive for a new one.
enum class BidirFlip_t : uint8_t { QueueFront, QueueBehind, ReplaceBehind, RefreshLive, SnapLive };
struct BidirFlipInputs_t
{
	bool queuedReal;	// this composite queued its real frame
	bool sameBase;		// this composite re-rendered the newest real frame
	bool queueEmpty;
	bool timelineStarted;	// a delayed timeline exists (last output or epoch)
	bool tailQueuedBehind = false;	// the queue tail is a queued-behind composite
	bool realsFill = false;			// the queued reals alone fill the ring-derived capacity
};
[[nodiscard]] constexpr BidirFlip_t select_bidir_flip( const BidirFlipInputs_t &in )
{
	if ( in.queuedReal )
		return BidirFlip_t::QueueFront;
	if ( in.sameBase && !in.queueEmpty )
		return BidirFlip_t::QueueFront;
	if ( in.timelineStarted && in.sameBase )
		return BidirFlip_t::RefreshLive;	// the queue is empty here
	if ( in.timelineStarted && !in.queueEmpty )	// cut/hitch behind queued content
		return in.tailQueuedBehind || in.realsFill ? BidirFlip_t::ReplaceBehind : BidirFlip_t::QueueBehind;
	return BidirFlip_t::SnapLive;
}

// An early generated commit reserves KMS state and then blocks until its page
// flip. A real client frame that becomes ready inside that reservation cannot
// replace the reserved state, so it lands one vblank later than it would have
// without the generated commit. Skip the early commit when the learned source
// cadence predicts the next real frame inside that commit-to-flip window.
//
// A prediction already in the past is deliberately NOT a reason to skip: an
// overdue real frame is exactly the empty vblank generation exists to fill.
// Untrained cadence, or any missing input, also never suppresses generation.
[[nodiscard]] constexpr bool real_arrival_blocks_early_generated_commit(
	uint64_t predictedRealReadyNs, uint64_t nowNs, uint64_t commitLeadNs,
	bool cadenceTrained )
{
	if ( !cadenceTrained || predictedRealReadyNs == 0u || nowNs == 0u
		|| commitLeadNs == 0u )
		return false;
	if ( predictedRealReadyNs <= nowNs )
		return false;
	return predictedRealReadyNs - nowNs <= commitLeadNs;
}

// Real-frame content identity. A client can reacquire and recommit a buffer
// that maps to the same CVulkanTexture object, so pointer identity alone misses
// the new content. steamcompmgr's per-commit id is a monotonic sequence and is
// authoritative whenever both sides carry one; pointer identity remains as the
// fallback for compositor-owned layers that have no commit id. An overlay-only
// repaint re-presents the same base commit and is therefore not new content.
struct RealFrameIdentity_t
{
	uint64_t commitId = 0;
	const void *pTexture = nullptr;
};

[[nodiscard]] constexpr bool is_new_real_frame_content(
	const RealFrameIdentity_t &last, const RealFrameIdentity_t &current )
{
	if ( current.pTexture == nullptr )
		return false;
	if ( last.commitId != 0u && current.commitId != 0u )
		return last.commitId != current.commitId;
	return last.pTexture != current.pTexture;
}

struct VrrMidpointPlan_t
{
	uint64_t targetFlipNs = 0;
	uint64_t wakeDeadlineNs = 0;
	bool valid = false;
};

// VRR has no fixed display grid. Correlated real feedback supplies its anchor;
// presentation lead and a small compositor margin move the wake/commit early
// enough for the resulting scanout to land at the content midpoint.
[[nodiscard]] constexpr VrrMidpointPlan_t plan_vrr_midpoint(
	uint64_t realFlipNs, uint64_t predictedCadenceNs,
	uint64_t presentLeadNs, uint64_t presentMarginNs, uint64_t nowNs )
{
	VrrMidpointPlan_t result;
	if ( realFlipNs == 0u || predictedCadenceNs == 0u )
		return result;

	result.targetFlipNs = saturating_add_ns(
		realFlipNs, predictedCadenceNs / 2u );
	const uint64_t advanceNs = saturating_add_ns(
		presentLeadNs, presentMarginNs );
	if ( advanceNs >= result.targetFlipNs )
		return result;

	result.wakeDeadlineNs = result.targetFlipNs - advanceNs;
	result.valid = result.wakeDeadlineNs != 0u
		&& nowNs < result.wakeDeadlineNs;
	return result;
}

} // namespace gamescope::framegen
