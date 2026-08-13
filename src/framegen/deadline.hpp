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

struct RealAnchorState_t
{
	uint64_t realFrameId = 0;
	uint64_t sourceReadyNs = 0;
	uint64_t provisionalTargetNs = 0;
	std::optional<uint64_t> correctedFlipNs;
	uint64_t epoch = 0;

	[[nodiscard]] constexpr uint64_t display_time() const
	{
		return correctedFlipNs.value_or( provisionalTargetNs );
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

struct AnchorCorrection_t
{
	RealAnchorState_t anchor;
	bool matched = false;
	bool discardProvisional = false;
};

[[nodiscard]] constexpr AnchorCorrection_t apply_flip_feedback(
	const RealAnchorState_t &anchor, uint64_t realFrameId,
	uint64_t actualFlipNs, uint64_t arrivalGuardNs )
{
	AnchorCorrection_t result = { .anchor = anchor };
	if ( realFrameId == 0u || realFrameId != anchor.realFrameId || actualFlipNs == 0u )
		return result;

	result.matched = true;
	const uint64_t errorNs = actualFlipNs >= anchor.provisionalTargetNs
		? actualFlipNs - anchor.provisionalTargetNs
		: anchor.provisionalTargetNs - actualFlipNs;
	result.discardProvisional = !anchor.correctedFlipNs.has_value()
		&& errorNs > arrivalGuardNs;
	result.anchor.correctedFlipNs = actualFlipNs;
	return result;
}

// Pending entries retain the anchor identity and whether their pixels were
// produced from its provisional display time. A large first correction only
// invalidates those exact pixels; unrelated/newer anchors and already-corrected
// work stay untouched.
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

struct PresentLeadState_t
{
	uint64_t emaNs = 0;
	uint32_t samples = 0;
};

// A deliberately slow 1/8 EMA. Invalid/reordered timestamp pairs do not
// contaminate the backend lead estimate.
[[nodiscard]] constexpr PresentLeadState_t update_present_lead(
	PresentLeadState_t state, uint64_t commitSubmitNs, uint64_t actualFlipNs )
{
	if ( commitSubmitNs == 0u || actualFlipNs <= commitSubmitNs )
		return state;

	const uint64_t sampleNs = actualFlipNs - commitSubmitNs;
	if ( state.samples == 0u )
		state.emaNs = sampleNs;
	else if ( sampleNs >= state.emaNs )
		state.emaNs += ( sampleNs - state.emaNs ) / 8u;
	else
		state.emaNs -= ( state.emaNs - sampleNs ) / 8u;
	if ( state.samples != UINT32_MAX )
		state.samples++;
	return state;
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
