#pragma once

#include "scheduling.hpp"
#include "temporal.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <vector>

namespace gamescope::framegen
{

inline constexpr uint32_t k_uPresentBiasWarmupSamples = 4u;

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

// The Step 3 device timestamp table is shared with the classic batch path for
// A/B, so use two valid single-slot keys while keeping that storage detail out
// of the shared work-class type.
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
	float configuredStrength = k_flNeutralStrength;
	float forwardStrengthCap = 1.5f;
	bool sourceTimestampsReliable = true;
	bool dedicatedQueue = true;
	bool sharedQueueProvenEmpty = false;
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
	if ( result.rawStrength > options.forwardStrengthCap )
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

	const FixedCadenceAdmission admission = fixed_cadence_admission(
		anchor.sourceReadyNs, cadence, options.nowNs, result.wakeNs, grid.T );
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
	DisplayChainKey_t key;
	uint64_t generation = 0;
	bool initialized = false;
};

struct DisplayChainTimingTransition_t
{
	DisplayChainTimingState_t state;
	bool displayChainChanged = false;
};

// Re-observing the same key is also the content-invalidation/re-prime path: the
// learning survives untouched. Only a genuine display-chain key transition
// starts both EMAs over and advances the generation used by deadline epochs.
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

// Treat the target error as the residual of the bias already applied to this
// provisional anchor. Adding 1/8 of that residual is algebraically the same as
// a 1/8 EMA of (actual flip - raw grid target), while allowing the anchor to
// carry the exact biased target that the scheduler believed.
[[nodiscard]] constexpr PresentBiasUpdate_t update_present_bias(
	PresentBiasState_t state, uint64_t provisionalTargetNs,
	uint64_t actualFlipNs, uint64_t arrivalGuardNs )
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

	result.state = update_present_timing_ema_residual(
		result.state, result.residualNs );
	return result;
}

[[nodiscard]] constexpr AnchorCorrection_t apply_flip_feedback(
	const RealAnchorState_t &anchor, PresentBiasState_t presentBias,
	uint64_t realFrameId,
	uint64_t actualFlipNs, uint64_t arrivalGuardNs )
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
			actualFlipNs, arrivalGuardNs );
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

struct BidirEpoch_t
{
	uint64_t sourceEpochNs = 0;
	uint64_t displayEpochNs = 0;
	uint64_t epoch = 0;
	bool valid = false;

	[[nodiscard]] constexpr uint64_t endpoint_display_time( uint64_t sourceReadyNs ) const
	{
		if ( sourceReadyNs >= sourceEpochNs )
			return saturating_add_ns( displayEpochNs, sourceReadyNs - sourceEpochNs );
		const uint64_t delta = sourceEpochNs - sourceReadyNs;
		return delta > displayEpochNs ? 0u : displayEpochNs - delta;
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

// Establish the fixed-delay translation at the first compatible real pair.
// The previous source endpoint is due where the current frame's live path
// would have displayed, introducing the invariant one-real-interval delay.
[[nodiscard]] constexpr BidirEpoch_t establish_bidir_epoch(
	uint64_t previousSourceReadyNs, uint64_t currentLiveTargetNs,
	uint64_t gridEpoch )
{
	return {
		.sourceEpochNs = previousSourceReadyNs,
		.displayEpochNs = currentLiveTargetNs,
		.epoch = gridEpoch,
		.valid = previousSourceReadyNs != 0u
			&& currentLiveTargetNs != 0u && gridEpoch != 0u,
	};
}

struct BidirEpochCorrection_t
{
	BidirEpoch_t epoch;
	bool applied = false;
};

// A delayed endpoint proves where that source timestamp actually scanned out.
// Only the epoch value is returned: already-created slots are values and remain
// untouched, while later plans observe the corrected source/display mapping.
[[nodiscard]] constexpr BidirEpochCorrection_t apply_bidir_endpoint_feedback(
	const BidirEpoch_t &epoch, uint64_t endpointSourceReadyNs,
	uint64_t actualFlipNs )
{
	BidirEpochCorrection_t result = { .epoch = epoch };
	if ( !epoch.valid || endpointSourceReadyNs < epoch.sourceEpochNs
		|| actualFlipNs == 0u )
		return result;

	const uint64_t sourceDeltaNs = endpointSourceReadyNs - epoch.sourceEpochNs;
	if ( sourceDeltaNs > actualFlipNs )
		return result;

	result.epoch.displayEpochNs = actualFlipNs - sourceDeltaNs;
	result.applied = true;
	return result;
}

struct BidirEndpoint_t
{
	uint64_t realFrameId = 0;
	uint64_t sourceReadyNs = 0;
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

// Return oldest-first generated entries to shed until the requested capacity
// is met. Real endpoints are never returned; if endpoints alone exceed the
// soft capacity, the caller preserves that endpoint backlog.
[[nodiscard]] inline std::vector<size_t> bidir_generated_shed_indices(
	std::span<const BidirSlotKind_t> kinds, size_t capacity )
{
	std::vector<size_t> result;
	if ( kinds.size() <= capacity )
		return result;

	size_t remaining = kinds.size();
	for ( size_t i = 0u; i < kinds.size() && remaining > capacity; i++ )
	{
		if ( kinds[ i ] != BidirSlotKind_t::Generated )
			continue;
		result.push_back( i );
		remaining--;
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

	const uint64_t firstIdealNs = epoch.endpoint_display_time( endpoints.front().sourceReadyNs );
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
		const uint64_t previousIdealNs = epoch.endpoint_display_time( endpoints[ j - 1u ].sourceReadyNs );
		const uint64_t endpointIdealNs = epoch.endpoint_display_time( endpoints[ j ].sourceReadyNs );
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

// A deliberately slow 1/8 EMA. Invalid/reordered timestamp pairs do not
// contaminate the backend lead estimate.
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
