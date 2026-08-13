#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "framegen/deadline.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <vector>

using Catch::Approx;
using namespace gamescope::framegen;

namespace
{

CadencePredictorState trained_cadence( uint64_t intervalNs, int64_t trendNs = 0 )
{
	return {
		.intervalNs = intervalNs,
		.trendNs = trendNs,
		.samples = k_uCadencePredictorMinSamples,
	};
}

CausalPlanOptions_t dedicated_backup_options( uint64_t nowNs, uint64_t epoch )
{
	return {
		.nowNs = nowNs,
		.gridEpoch = epoch,
		.sourceTimestampsReliable = false,
		.dedicatedQueue = true,
	};
}

} // namespace

TEST_CASE( "30 fps on 120 Hz follows stable actual-grid phases", "[framegen][deadline]" )
{
	constexpr uint64_t epoch = 7u;
	const DisplayGrid_t grid = { .D0 = 1'000u, .W0 = 999u, .T = 10u };
	const RealAnchorState_t anchor = {
		.realFrameId = 1u,
		.sourceReadyNs = 1'000u,
		.provisionalTargetNs = 1'000u,
		.correctedFlipNs = 1'000u,
		.epoch = epoch,
	};
	const CadencePredictorState cadence = trained_cadence( 40u );
	uint64_t afterTargetNs = 0u;
	for ( uint32_t k = 1u; k <= 3u; k++ )
	{
		auto options = dedicated_backup_options( 1'000u, epoch );
		options.afterTargetNs = afterTargetNs;
		const CausalSlotPlan_t plan = plan_next_causal_slot( grid, anchor, cadence, options );
		REQUIRE( plan.admit );
		CHECK( plan.targetNs == grid.target( k ) );
		CHECK( plan.wakeNs == grid.wake( k ) );
		CHECK( plan.phase == Approx( static_cast<double>( k ) / 4.0 ) );
		afterTargetNs = plan.targetNs;
	}
}

TEST_CASE( "45 fps on 60 Hz stays on-grid without fractional drift", "[framegen][deadline]" )
{
	constexpr uint64_t epoch = 8u;
	const DisplayGrid_t grid = { .D0 = 1'000u, .W0 = 999u, .T = 3u };
	const CadencePredictorState cadence = trained_cadence( 4u );
	std::array<uint32_t, 3> generatedCounts = {};

	for ( uint64_t frame = 0u; frame < 1'000u; frame++ )
	{
		const uint64_t idealAnchorNs = grid.D0 + frame * 4u;
		const uint64_t idealNextNs = idealAnchorNs + 4u;
		const uint64_t anchorN = grid_index_at_or_after( grid, idealAnchorNs );
		const uint64_t nextN = grid_index_at_or_after( grid, idealNextNs );
		const uint64_t anchorTargetNs = grid.target( anchorN );
		const uint64_t nextTargetNs = grid.target( nextN );
		const RealAnchorState_t anchor = {
			.realFrameId = frame + 1u,
			.sourceReadyNs = idealAnchorNs,
			.provisionalTargetNs = anchorTargetNs,
			.correctedFlipNs = anchorTargetNs,
			.epoch = epoch,
		};
		auto options = dedicated_backup_options( anchorTargetNs, epoch );
		const CausalSlotPlan_t plan = plan_next_causal_slot( grid, anchor, cadence, options );
		uint32_t count = 0u;
		if ( plan.targetNs < nextTargetNs )
		{
			REQUIRE( plan.admit );
			CHECK( ( plan.targetNs - grid.D0 ) % grid.T == 0u );
			CHECK( plan.phase == Approx(
				static_cast<double>( plan.targetNs - anchorTargetNs ) / 4.0 ) );
			count = 1u;
		}
		generatedCounts[ frame % generatedCounts.size() ] += count;
		CHECK( nextTargetNs >= idealNextNs );
		CHECK( nextTargetNs - idealNextNs < grid.T );
	}

	CHECK( generatedCounts[ 0 ] > 0u );
	CHECK( generatedCounts[ 1 ] == 0u );
	CHECK( generatedCounts[ 2 ] == 0u );
}

TEST_CASE( "causal phase follows cadence trend", "[framegen][deadline]" )
{
	const DisplayGrid_t grid = { .D0 = 1'000u, .W0 = 999u, .T = 10u };
	const RealAnchorState_t anchor = {
		.realFrameId = 2u,
		.sourceReadyNs = 1'000u,
		.provisionalTargetNs = 1'000u,
		.correctedFlipNs = 1'000u,
		.epoch = 1u,
	};
	const CadencePredictorState cadence = trained_cadence( 40u, 4 );
	const CausalSlotPlan_t plan = plan_next_causal_slot(
		grid, anchor, cadence, dedicated_backup_options( 1'000u, 1u ) );
	REQUIRE( plan.admit );
	CHECK( predicted_cadence_interval_ns( cadence ) == 44u );
	CHECK( plan.phase == Approx( 10.0 / 44.0 ) );
}

TEST_CASE( "flip feedback corrects future causal anchors", "[framegen][deadline]" )
{
	const RealAnchorState_t anchor = {
		.realFrameId = 9u,
		.sourceReadyNs = 1'000u,
		.provisionalTargetNs = 2'000u,
		.epoch = 3u,
	};
	const AnchorCorrection_t large = apply_flip_feedback( anchor, 9u, 2'301u, 300u );
	REQUIRE( large.matched );
	CHECK( large.discardProvisional );
	REQUIRE( large.anchor.correctedFlipNs );
	CHECK( *large.anchor.correctedFlipNs == 2'301u );

	const AnchorCorrection_t small = apply_flip_feedback( anchor, 9u, 2'299u, 300u );
	REQUIRE( small.matched );
	CHECK_FALSE( small.discardProvisional );
	const DisplayGrid_t grid = { .D0 = 2'000u, .W0 = 1'999u, .T = 500u };
	const CausalSlotPlan_t future = plan_next_causal_slot( grid, small.anchor,
		trained_cadence( 2'000u ), dedicated_backup_options( 2'300u, 3u ) );
	CHECK( future.targetNs == 2'500u );
	CHECK( future.phase == Approx( 201.0 / 2'000.0 ) );
}

TEST_CASE( "large anchor correction discards only matching provisional slots", "[framegen][deadline]" )
{
	const RealAnchorState_t anchor = {
		.realFrameId = 42u,
		.provisionalTargetNs = 10'000u,
		.epoch = 1u,
	};
	const AnchorCorrection_t correction = apply_flip_feedback(
		anchor, 42u, 18'334u, 300u );
	REQUIRE( correction.discardProvisional );
	CHECK( discard_pending_provisional_slot( correction, 42u, true ) );
	CHECK_FALSE( discard_pending_provisional_slot( correction, 42u, false ) );
	CHECK_FALSE( discard_pending_provisional_slot( correction, 41u, true ) );

	const AnchorCorrection_t small = apply_flip_feedback(
		anchor, 42u, 10'300u, 300u );
	CHECK_FALSE( discard_pending_provisional_slot( small, 42u, true ) );
}

TEST_CASE( "one-vblank late provisional anchor changes the generated phase", "[framegen][deadline]" )
{
	constexpr uint64_t interval = 8'334'000u;
	const RealAnchorState_t provisional = {
		.realFrameId = 7u,
		.sourceReadyNs = 100'000'000u,
		.provisionalTargetNs = 108'334'000u,
		.epoch = 5u,
	};
	const DisplayGrid_t grid = {
		.D0 = 116'668'000u,
		.W0 = 116'000'000u,
		.T = interval,
	};
	const CadencePredictorState cadence = trained_cadence( 4u * interval );
	const CausalSlotPlan_t before = plan_next_causal_slot(
		grid, provisional, cadence,
		dedicated_backup_options( 109'000'000u, 5u ) );
	REQUIRE( before.admit );
	CHECK( before.phase == Approx( 0.25 ) );

	const AnchorCorrection_t correction = apply_flip_feedback(
		provisional, 7u, 100'000'000u, interval / 32u );
	REQUIRE( correction.discardProvisional );
	const CausalSlotPlan_t after = plan_next_causal_slot(
		grid, correction.anchor, cadence,
		dedicated_backup_options( 109'000'000u, 5u ) );
	REQUIRE( after.admit );
	CHECK( after.targetNs == before.targetNs );
	CHECK( after.phase == Approx( 0.5 ) );
}

TEST_CASE( "causal deadline costs are keyed by work class", "[framegen][deadline]" )
{
	const uint32_t full = deadline_work_class_cost_key(
		DeadlineWorkClass_t::FullPreparationAndWarp );
	const uint32_t cached = deadline_work_class_cost_key(
		DeadlineWorkClass_t::CachedWarp );
	CHECK( full == 1u );
	CHECK( cached == 2u );
	CHECK( full != cached );
}

TEST_CASE( "missed wake and refresh epochs reject causal slots", "[framegen][deadline]" )
{
	const DisplayGrid_t grid = { .D0 = 1'000u, .W0 = 900u, .T = 100u };
	const RealAnchorState_t anchor = {
		.realFrameId = 1u,
		.sourceReadyNs = 1'000u,
		.provisionalTargetNs = 1'000u,
		.epoch = 4u,
	};
	auto lateOptions = dedicated_backup_options( 1'000u, 4u );
	const CausalSlotPlan_t late = plan_next_causal_slot(
		grid, anchor, trained_cadence( 400u ), lateOptions );
	CHECK_FALSE( late.admit );
	CHECK( late.skipReason == DeadlineSkipReason_t::MissedWake );

	auto changedRefresh = dedicated_backup_options( 900u, 5u );
	const CausalSlotPlan_t stale = plan_next_causal_slot(
		grid, anchor, trained_cadence( 400u ), changedRefresh );
	CHECK_FALSE( stale.admit );
	CHECK( stale.targetNs == 0u );
	CHECK( stale.skipReason == DeadlineSkipReason_t::EpochMismatch );
}

TEST_CASE( "bidir gap-one schedules the delayed real without a live snap", "[framegen][deadline]" )
{
	const DisplayGrid_t grid = { .D0 = 1'000u, .W0 = 990u, .T = 100u };
	const BidirEpoch_t epoch = {
		.sourceEpochNs = 500u,
		.displayEpochNs = 1'000u,
		.epoch = 2u,
		.valid = true,
	};
	const std::array endpoints = {
		BidirEndpoint_t{ .realFrameId = 10u, .sourceReadyNs = 500u },
		BidirEndpoint_t{ .realFrameId = 11u, .sourceReadyNs = 590u },
	};
	const BidirPlan_t plan = plan_bidir_slots( grid, epoch, endpoints, 4u, 2u );
	REQUIRE( plan.validEpoch );
	REQUIRE( plan.slots.size() == 2u );
	CHECK( std::ranges::none_of( plan.slots,
		[]( const BidirSlot_t &slot ) { return slot.kind == BidirSlotKind_t::Generated; } ) );
	CHECK( plan.slots.back().kind == BidirSlotKind_t::RealEndpoint );
	CHECK( plan.slots.back().endpointFrameId == 11u );
	CHECK( plan.slots.back().targetNs == 1'100u );
}

TEST_CASE( "bidir epoch starts at the current live-path display target", "[framegen][deadline]" )
{
	const BidirEpoch_t invalid = establish_bidir_epoch( 0u, 5'000u, 3u );
	CHECK_FALSE( invalid.valid );

	const BidirEpoch_t epoch = establish_bidir_epoch( 1'000u, 5'000u, 3u );
	REQUIRE( epoch.valid );
	CHECK( epoch.sourceEpochNs == 1'000u );
	CHECK( epoch.displayEpochNs == 5'000u );
	CHECK( epoch.endpoint_display_time( 1'000u ) == 5'000u );
	CHECK( epoch.endpoint_display_time( 1'250u ) == 5'250u );
}

TEST_CASE( "delayed-real feedback corrects only future bidir targets", "[framegen][deadline]" )
{
	const DisplayGrid_t grid = { .D0 = 5'000u, .W0 = 4'990u, .T = 100u };
	const BidirEpoch_t epoch = establish_bidir_epoch( 1'000u, 5'000u, 4u );
	const std::array firstPair = {
		BidirEndpoint_t{ .realFrameId = 1u, .sourceReadyNs = 1'000u },
		BidirEndpoint_t{ .realFrameId = 2u, .sourceReadyNs = 1'200u },
	};
	const BidirPlan_t alreadyQueued = plan_bidir_slots(
		grid, epoch, firstPair, 2u, 4u );
	REQUIRE( alreadyQueued.slots.back().targetNs == 5'200u );

	const BidirEpochCorrection_t correction = apply_bidir_endpoint_feedback(
		epoch, 1'200u, 5'250u );
	REQUIRE( correction.applied );
	CHECK( correction.epoch.displayEpochNs == 5'050u );
	// The queued plan is a value and is never retimed by feedback.
	CHECK( alreadyQueued.slots.back().targetNs == 5'200u );

	const std::array futurePair = {
		BidirEndpoint_t{ .realFrameId = 2u, .sourceReadyNs = 1'200u },
		BidirEndpoint_t{ .realFrameId = 3u, .sourceReadyNs = 1'400u },
	};
	const BidirPlan_t future = plan_bidir_slots(
		grid, correction.epoch, futurePair, 2u, 4u );
	REQUIRE( future.slots.back().kind == BidirSlotKind_t::RealEndpoint );
	CHECK( future.slots.back().targetNs == 5'500u );
}

TEST_CASE( "bidir 24 fps on 60 Hz uses a drift-free absolute epoch", "[framegen][deadline]" )
{
	const DisplayGrid_t grid = { .D0 = 1'000u, .W0 = 999u, .T = 2u };
	const BidirEpoch_t epoch = {
		.sourceEpochNs = 10'000u,
		.displayEpochNs = 1'000u,
		.epoch = 6u,
		.valid = true,
	};
	std::vector<BidirEndpoint_t> endpoints;
	for ( uint64_t i = 0u; i <= 500u; i++ )
		endpoints.push_back( { .realFrameId = i + 1u, .sourceReadyNs = 10'000u + i * 5u } );

	const BidirPlan_t plan = plan_bidir_slots( grid, epoch, endpoints, 4u, 6u );
	REQUIRE( plan.validEpoch );
	for ( uint64_t i = 1u; i <= 500u; i++ )
	{
		const uint64_t idealNs = epoch.displayEpochNs + i * 5u;
		const uint64_t expectedTargetNs = grid.target( grid_index_at_or_after( grid, idealNs ) );
		const auto endpoint = std::ranges::find_if( plan.slots,
			[i]( const BidirSlot_t &slot ) {
				return slot.kind == BidirSlotKind_t::RealEndpoint
					&& slot.endpointFrameId == i + 1u;
			} );
		REQUIRE( endpoint != plan.slots.end() );
		CHECK( endpoint->targetNs == expectedTargetNs );

		const uint64_t previousIdealNs = idealNs - 5u;
		uint32_t intervalSlots = 0u;
		for ( const BidirSlot_t &slot : plan.slots )
		{
			if ( slot.endpointFrameId != i + 1u )
				continue;
			intervalSlots++;
			if ( slot.kind == BidirSlotKind_t::Generated )
			{
				CHECK( slot.phase > 0.0 );
				CHECK( slot.phase < 1.0 );
				CHECK( slot.phase == Approx(
					static_cast<double>( slot.targetNs - previousIdealNs ) / 5.0 ) );
			}
		}
		CHECK( intervalSlots == ( i % 2u != 0u ? 3u : 2u ) );
	}
}

TEST_CASE( "bidir coalesces stale endpoints and invalidates old grids", "[framegen][deadline]" )
{
	const DisplayGrid_t grid = { .D0 = 1'000u, .W0 = 990u, .T = 100u };
	const BidirEpoch_t epoch = {
		.sourceEpochNs = 500u,
		.displayEpochNs = 1'000u,
		.epoch = 9u,
		.valid = true,
	};
	const std::array endpoints = {
		BidirEndpoint_t{ .realFrameId = 1u, .sourceReadyNs = 500u },
		BidirEndpoint_t{ .realFrameId = 2u, .sourceReadyNs = 530u },
		BidirEndpoint_t{ .realFrameId = 3u, .sourceReadyNs = 560u },
	};
	const BidirPlan_t plan = plan_bidir_slots( grid, epoch, endpoints, 4u, 9u );
	REQUIRE( plan.slots.size() == 2u );
	CHECK( plan.slots.back().targetNs == 1'100u );
	CHECK( plan.slots.back().endpointFrameId == 3u );

	const BidirPlan_t stale = plan_bidir_slots( grid, epoch, endpoints, 4u, 10u );
	CHECK_FALSE( stale.validEpoch );
	CHECK( stale.slots.empty() );
}

TEST_CASE( "source admission respects dedicated and shared queue provenance", "[framegen][deadline]" )
{
	const DisplayGrid_t grid = { .D0 = 1'000u, .W0 = 990u, .T = 100u };
	const RealAnchorState_t anchor = {
		.realFrameId = 1u,
		.provisionalTargetNs = 1'000u,
		.epoch = 1u,
	};
	const CadencePredictorState cadence = trained_cadence( 400u );

	auto shared = dedicated_backup_options( 1'000u, 1u );
	shared.dedicatedQueue = false;
	const CausalSlotPlan_t uncertain = plan_next_causal_slot( grid, anchor, cadence, shared );
	CHECK_FALSE( uncertain.admit );
	CHECK( uncertain.skipReason == DeadlineSkipReason_t::SharedSourceUncertain );

	shared.sharedQueueProvenEmpty = true;
	CHECK( plan_next_causal_slot( grid, anchor, cadence, shared ).admit );
	CHECK( plan_next_causal_slot( grid, anchor, cadence,
		dedicated_backup_options( 1'000u, 1u ) ).admit );
}

TEST_CASE( "source admission distinguishes safe, overdue, and warmup predictions", "[framegen][deadline]" )
{
	const DisplayGrid_t grid = {
		.D0 = 100'000'000u,
		.W0 = 98'000'000u,
		.T = 10'000'000u,
	};
	const RealAnchorState_t anchor = {
		.realFrameId = 1u,
		.sourceReadyNs = 100'000'000u,
		.provisionalTargetNs = 100'000'000u,
		.correctedFlipNs = 100'000'000u,
		.epoch = 1u,
	};
	const CadencePredictorState cadence = trained_cadence( 5'000'000u );
	CausalPlanOptions_t options = {
		.nowNs = 100'100'000u,
		.gridEpoch = 1u,
		.forwardStrengthCap = 3.0f,
		.sourceTimestampsReliable = true,
		.dedicatedQueue = true,
	};
	const CausalSlotPlan_t safe = plan_next_causal_slot( grid, anchor, cadence, options );
	CHECK_FALSE( safe.admit );
	CHECK( safe.skipReason == DeadlineSkipReason_t::NextRealSafelyDue );

	options.nowNs = 105'400'000u;
	const CausalSlotPlan_t overdue = plan_next_causal_slot( grid, anchor, cadence, options );
	CHECK( overdue.admit );

	CadencePredictorState warmup = cadence;
	warmup.samples = 1u;
	options.nowNs = 100'100'000u;
	CHECK( plan_next_causal_slot( grid, anchor, warmup, options ).admit );
}

TEST_CASE( "bidir multiplier ceiling selects evenly distributed grid targets", "[framegen][deadline]" )
{
	const DisplayGrid_t grid = { .D0 = 1'000u, .W0 = 990u, .T = 100u };
	const BidirEpoch_t epoch = {
		.sourceEpochNs = 500u,
		.displayEpochNs = 1'000u,
		.epoch = 1u,
		.valid = true,
	};
	const std::array endpoints = {
		BidirEndpoint_t{ .realFrameId = 1u, .sourceReadyNs = 500u },
		BidirEndpoint_t{ .realFrameId = 2u, .sourceReadyNs = 1'000u },
	};
	const BidirPlan_t plan = plan_bidir_slots( grid, epoch, endpoints, 3u, 1u );
	std::vector<uint64_t> generatedTargets;
	for ( const BidirSlot_t &slot : plan.slots )
	{
		if ( slot.kind == BidirSlotKind_t::Generated )
			generatedTargets.push_back( slot.targetNs );
	}
	REQUIRE( generatedTargets.size() == 2u );
	CHECK( generatedTargets[ 0 ] == 1'200u );
	CHECK( generatedTargets[ 1 ] == 1'400u );
}

TEST_CASE( "bidir pressure sheds oldest generated slots before endpoints", "[framegen][deadline]" )
{
	const std::array kinds = {
		BidirSlotKind_t::RealEndpoint,
		BidirSlotKind_t::Generated,
		BidirSlotKind_t::Generated,
		BidirSlotKind_t::RealEndpoint,
		BidirSlotKind_t::Generated,
	};
	const std::vector<size_t> shed = bidir_generated_shed_indices( kinds, 3u );
	REQUIRE( shed.size() == 2u );
	CHECK( shed[ 0 ] == 1u );
	CHECK( shed[ 1 ] == 2u );

	const std::array endpointBacklog = {
		BidirSlotKind_t::RealEndpoint,
		BidirSlotKind_t::Generated,
		BidirSlotKind_t::RealEndpoint,
		BidirSlotKind_t::RealEndpoint,
	};
	const std::vector<size_t> endpointSafe = bidir_generated_shed_indices(
		endpointBacklog, 2u );
	REQUIRE( endpointSafe.size() == 1u );
	CHECK( endpointSafe.front() == 1u );
}

TEST_CASE( "VRR midpoint wake compensates tagged backend present lead", "[framegen][deadline]" )
{
	PresentLeadState_t lead;
	lead = update_present_lead( lead, 10'000u, 11'000u );
	CHECK( lead.emaNs == 1'000u );
	CHECK( lead.samples == 1u );
	lead = update_present_lead( lead, 20'000u, 23'000u );
	CHECK( lead.emaNs == 1'250u );
	CHECK( lead.samples == 2u );
	CHECK( update_present_lead( lead, 30'000u, 29'000u ).emaNs == 1'250u );

	const VrrMidpointPlan_t plan = plan_vrr_midpoint(
		1'000'000u, 40'000u, 3'000u, 1'000u, 1'010'000u );
	REQUIRE( plan.valid );
	CHECK( plan.targetFlipNs == 1'020'000u );
	CHECK( plan.wakeDeadlineNs == 1'016'000u );

	const VrrMidpointPlan_t late = plan_vrr_midpoint(
		1'000'000u, 40'000u, 3'000u, 1'000u, 1'016'000u );
	CHECK_FALSE( late.valid );
	CHECK( late.targetFlipNs == plan.targetFlipNs );
	CHECK( late.wakeDeadlineNs == plan.wakeDeadlineNs );
}

TEST_CASE( "slot budget applies 0.85 to actual remaining time", "[framegen][deadline]" )
{
	CHECK( deadline_cost_fits( 8'500u, 20'000u, 10'000u, 0u ) );
	CHECK_FALSE( deadline_cost_fits( 8'501u, 20'000u, 10'000u, 0u ) );
	CHECK( deadline_cost_fits( 4'250u, 20'000u, 10'000u, 15'000u ) );
	CHECK_FALSE( deadline_cost_fits( 4'251u, 20'000u, 10'000u, 15'000u ) );
	CHECK_FALSE( deadline_cost_fits( 1u, 20'000u, 10'000u, 20'000u ) );
}
