#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "framegen/deadline.hpp"
#include "framegen/hud.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <string_view>
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

TEST_CASE( "framegen HUD level one formats the closed five-second window", "[framegen][hud]" )
{
	const FramegenHudSnapshot_t snapshot = {
		.mode = GamescopeFramegenMode::Extrapolate,
		.quality = GamescopeFramegenQuality::Ultra,
		.multiplier = 2u,
		.refreshMilliHz = 119'600u,
		.real = 150u,
		.delayedReal = 50u,
		.generated = 390u,
		.repeats = 10u,
	};
	const FramegenHudText_t text = format_framegen_hud( 1u, snapshot );

	REQUIRE( text.lineCount == 1u );
	CHECK( text.lines[0].data()
		== std::string_view{ "E2 U 120Hz  real 40/s  gen 78/s  rep 2/s" } );
	CHECK( std::strlen( text.lines[0].data() ) <= k_uFramegenHudMaxColumns );
}

TEST_CASE( "framegen HUD level two remains structured at the 48-column limit", "[framegen][hud]" )
{
	const FramegenHudSnapshot_t snapshot = {
		.mode = GamescopeFramegenMode::Motion,
		.quality = GamescopeFramegenQuality::High,
		.multiplier = 4u,
		.refreshMilliHz = 120'000u,
		.bidir = true,
		.baseLayer = true,
		.netActive = true,
		.adapt = true,
		.real = 200u,
		.generated = 390u,
		.repeats = 10u,
		.biasTenthsMs = 1,
		.deadlineHitPercent = 99u,
		.pacingSdTenthsMs = 7u,
		.netOnline = true,
		.netTrainedSteps = 1'200u,
		.netProfilePresent = true,
	};
	const FramegenHudText_t text = format_framegen_hud( 2u, snapshot );

	REQUIRE( text.lineCount == 3u );
	CHECK( text.lines[0].data()
		== std::string_view{ "M4 BIDIR BASE NET ADAPT H 120Hz R40/s G78/s P2/s" } );
	CHECK( text.lines[1].data()
		== std::string_view{ "pace: bias +0.1ms  hit 99%  sd 0.7ms" } );
	CHECK( text.lines[2].data()
		== std::string_view{ "net: online, 1.2k steps, profile loaded" } );
	for ( uint32_t line = 0u; line < text.lineCount; line++ )
		CHECK( std::strlen( text.lines[line].data() ) <= k_uFramegenHudMaxColumns );

	const FramegenHudUniform_t uniform = make_framegen_hud_uniform( text, true );
	CHECK( uniform.lineCount == 3u );
	CHECK( uniform.widthChars == 48u );
	CHECK( uniform.hdr == 1u );
	CHECK( ( uniform.text[0] & 0xffu ) == static_cast<uint32_t>( 'M' ) );
	CHECK( uniform.font[static_cast<uint32_t>( 'A' ) * 2u]
		== static_cast<uint32_t>( k_uFramegenHudFont8x8[static_cast<uint32_t>( 'A' )] ) );
}

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
	const PresentBiasState_t matureBias = {
		.samples = k_uPresentBiasWarmupSamples,
		.consecutiveGuardExceeds = 1u,
	};
	const RealAnchorState_t anchor = {
		.realFrameId = 9u,
		.sourceReadyNs = 1'000u,
		.provisionalTargetNs = 2'000u,
		.epoch = 3u,
	};
	const AnchorCorrection_t large = apply_flip_feedback(
		anchor, matureBias, 9u, 2'301u, 300u );
	REQUIRE( large.matched );
	CHECK( large.discardProvisional );
	REQUIRE( large.anchor.correctedFlipNs );
	CHECK( *large.anchor.correctedFlipNs == 2'301u );

	const AnchorCorrection_t small = apply_flip_feedback(
		anchor, matureBias, 9u, 2'299u, 300u );
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
	const PresentBiasState_t matureBias = {
		.samples = k_uPresentBiasWarmupSamples,
		.consecutiveGuardExceeds = 1u,
	};
	const RealAnchorState_t anchor = {
		.realFrameId = 42u,
		.provisionalTargetNs = 10'000u,
		.epoch = 1u,
	};
	const AnchorCorrection_t correction = apply_flip_feedback(
		anchor, matureBias, 42u, 18'334u, 300u );
	REQUIRE( correction.discardProvisional );
	CHECK( discard_pending_provisional_slot( correction, 42u, true ) );
	CHECK_FALSE( discard_pending_provisional_slot( correction, 42u, false ) );
	CHECK_FALSE( discard_pending_provisional_slot( correction, 41u, true ) );

	const AnchorCorrection_t small = apply_flip_feedback(
		anchor, matureBias, 42u, 10'300u, 300u );
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
		provisional,
		PresentBiasState_t{
			.samples = k_uPresentBiasWarmupSamples,
			.consecutiveGuardExceeds = 1u,
		},
		7u, 100'000'000u, interval / 32u );
	REQUIRE( correction.discardProvisional );
	const CausalSlotPlan_t after = plan_next_causal_slot(
		grid, correction.anchor, cadence,
		dedicated_backup_options( 109'000'000u, 5u ) );
	REQUIRE( after.admit );
	CHECK( after.targetNs == before.targetNs );
	CHECK( after.phase == Approx( 0.5 ) );
}

TEST_CASE( "present bias converges during warmup without discarding pixels", "[framegen][deadline]" )
{
	constexpr uint64_t intervalNs = 8'333'333u;
	constexpr uint64_t guardNs = intervalNs / 32u;
	PresentBiasState_t bias;

	for ( uint64_t frame = 1u; frame <= 12u; frame++ )
	{
		const uint64_t rawTargetNs = 100'000'000u + frame * intervalNs;
		const RealAnchorState_t anchor = {
			.realFrameId = frame,
			.provisionalTargetNs = apply_present_bias_ns(
				rawTargetNs, bias.emaNs ),
			.provisionalBiasNs = bias.emaNs,
			.epoch = 1u,
		};
		const AnchorCorrection_t correction = apply_flip_feedback(
			anchor, bias, frame, rawTargetNs + intervalNs, guardNs );
		REQUIRE( correction.matched );
		CHECK_FALSE( correction.discardProvisional );
		bias = correction.presentBias;
	}

	CHECK( bias.samples == 12u );
	CHECK( bias.emaNs == static_cast<int64_t>( intervalNs ) );
	const DisplayGrid_t rawGrid = {
		.D0 = 200'000'000u,
		.W0 = 199'000'000u,
		.T = intervalNs,
	};
	const DisplayGrid_t biasedGrid = apply_present_bias( rawGrid, bias.emaNs );
	CHECK( biasedGrid.D0 == rawGrid.D0 + intervalNs );
	CHECK( biasedGrid.W0 == rawGrid.W0 + intervalNs );
}

TEST_CASE( "present-bias outliers require two consecutive guard failures", "[framegen][deadline]" )
{
	constexpr uint64_t baselineLeadNs = 1'000u;
	constexpr uint64_t guardNs = 300u;
	constexpr uint64_t spikeNs = 2'000u;
	PresentBiasState_t bias = {
		.emaNs = baselineLeadNs,
		.samples = k_uPresentBiasWarmupSamples,
	};
	uint64_t frame = 1u;
	uint64_t rawTargetNs = 100'000u;
	const auto feedback = [&]( uint64_t extraLeadNs ) {
		const RealAnchorState_t anchor = {
			.realFrameId = frame,
			.provisionalTargetNs = apply_present_bias_ns(
				rawTargetNs, bias.emaNs ),
			.provisionalBiasNs = bias.emaNs,
			.epoch = 1u,
		};
		const AnchorCorrection_t correction = apply_flip_feedback(
			anchor, bias, frame, rawTargetNs + baselineLeadNs + extraLeadNs,
			guardNs );
		bias = correction.presentBias;
		frame++;
		rawTargetNs += 10'000u;
		return correction;
	};

	CHECK_FALSE( feedback( spikeNs ).discardProvisional );
	CHECK_FALSE( feedback( 0u ).discardProvisional );
	CHECK( bias.consecutiveGuardExceeds == 0u );
	CHECK_FALSE( feedback( spikeNs ).discardProvisional );
	CHECK( feedback( spikeNs ).discardProvisional );
}

TEST_CASE( "deadline hit arithmetic uses the biased target", "[framegen][deadline]" )
{
	constexpr uint64_t rawTargetNs = 10'000u;
	constexpr int64_t biasNs = 1'000;
	constexpr uint64_t actualFlipNs = 11'050u;
	constexpr uint64_t intervalNs = 1'000u;

	const DeadlineFeedbackSample_t raw = deadline_feedback_sample(
		actualFlipNs, rawTargetNs, intervalNs );
	const DeadlineFeedbackSample_t biased = deadline_feedback_sample(
		actualFlipNs, apply_present_bias_ns( rawTargetNs, biasNs ), intervalNs );
	REQUIRE( raw.valid );
	REQUIRE( biased.valid );
	CHECK_FALSE( raw.hit );
	CHECK( raw.signedErrorNs == 1'050 );
	CHECK( biased.hit );
	CHECK( biased.signedErrorNs == 50 );
}

TEST_CASE( "biased wake preserves the raw provisional admission bound", "[framegen][deadline]" )
{
	const RealAnchorState_t anchor = {
		.provisionalTargetNs = 11'000u,
		.provisionalBiasNs = 1'000,
	};
	CHECK( anchor.provisional_start_estimate() == 10'000u );
	CHECK( deadline_cost_fits(
		8'500u, 20'000u, 10'000u,
		anchor.provisional_start_estimate() ) );
	CHECK_FALSE( deadline_cost_fits(
		8'500u, 20'000u, 10'000u,
		anchor.provisionalTargetNs ) );
}

TEST_CASE( "display timing bias survives content invalidation and resets on grid change", "[framegen][deadline]" )
{
	const DisplayChainKey_t chain = {
		.backendId = 11u,
		.connectorId = 22u,
		.intervalNs = 8'333'333u,
		.vrrActive = false,
		.sourceTimestampsReliable = true,
	};
	DisplayChainTimingState_t timing =
		observe_display_chain( {}, chain ).state;
	timing.presentBias = {
		.emaNs = 8'000'000,
		.samples = 19u,
		.consecutiveGuardExceeds = 1u,
	};
	timing.presentLead = {
		.emaNs = 1'250'000,
		.samples = 23u,
	};
	const uint64_t generation = timing.generation;

	// Content history can disappear and re-prime while the observed display
	// chain remains identical. Re-observation must retain both learned EMAs.
	const DisplayChainTimingTransition_t contentReprime =
		observe_display_chain( timing, chain );
	CHECK_FALSE( contentReprime.displayChainChanged );
	CHECK( contentReprime.state.generation == generation );
	CHECK( contentReprime.state.presentBias.emaNs == 8'000'000 );
	CHECK( contentReprime.state.presentBias.samples == 19u );
	CHECK( contentReprime.state.presentBias.consecutiveGuardExceeds == 1u );
	CHECK( contentReprime.state.presentLead.emaNs == 1'250'000 );
	CHECK( contentReprime.state.presentLead.samples == 23u );

	DisplayChainKey_t changedGrid = chain;
	changedGrid.intervalNs = 16'666'667u;
	const DisplayChainTimingTransition_t gridChange =
		observe_display_chain( contentReprime.state, changedGrid );
	REQUIRE( gridChange.displayChainChanged );
	CHECK( gridChange.state.generation == generation + 1u );
	CHECK( gridChange.state.presentBias.samples == 0u );
	CHECK( gridChange.state.presentBias.emaNs == 0 );
	CHECK( gridChange.state.presentLead.samples == 0u );
	CHECK( gridChange.state.presentLead.emaNs == 0 );

	const auto checkChainReset = [&]( const DisplayChainKey_t &changedChain ) {
		const DisplayChainTimingTransition_t changed =
			observe_display_chain( timing, changedChain );
		REQUIRE( changed.displayChainChanged );
		CHECK( changed.state.presentBias.samples == 0u );
		CHECK( changed.state.presentLead.samples == 0u );
	};
	DisplayChainKey_t changedBackend = chain;
	changedBackend.backendId++;
	checkChainReset( changedBackend );
	DisplayChainKey_t changedConnector = chain;
	changedConnector.connectorId++;
	checkChainReset( changedConnector );
	DisplayChainKey_t changedVrr = chain;
	changedVrr.vrrActive = true;
	checkChainReset( changedVrr );
	DisplayChainKey_t changedClock = chain;
	changedClock.sourceTimestampsReliable = false;
	checkChainReset( changedClock );
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

TEST_CASE( "bidir epoch re-establishment does not thrash across a cut sequence", "[framegen][deadline]" )
{
	constexpr uint64_t gridEpoch = 9u;
	BidirEpoch_t epoch = establish_bidir_epoch( 1'000u, 5'000u, gridEpoch );
	BidirCutEpisodeState_t cutState;
	uint32_t invalidations = 0u;
	uint32_t establishments = 1u;
	const std::array cutSequence = {
		false, true, true, true, false, true, true,
	};

	for ( size_t i = 0u; i < cutSequence.size(); i++ )
	{
		const BidirCutTransition_t cut = observe_bidir_scene_cut(
			cutState, cutSequence[ i ] );
		cutState = cut.state;
		CHECK( cut.discardInterpolations == cutSequence[ i ] );
		if ( cut.invalidateEpoch )
		{
			epoch = {};
			invalidations++;
		}
		if ( !epoch.valid )
		{
			epoch = establish_bidir_epoch(
				1'100u + i * 100u, 5'100u + i * 100u, gridEpoch );
			REQUIRE( epoch.valid );
			CHECK( epoch.epoch == gridEpoch );
			establishments++;
		}
	}

	// One invalidation/re-establishment per cut episode, not per asserted
	// detector sample. The clean pair between episodes re-arms a genuine cut.
	CHECK( invalidations == 2u );
	CHECK( establishments == 3u );
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

TEST_CASE( "biased admission preserves both empty vblanks at 46 fps on 120 Hz", "[framegen][deadline]" )
{
	constexpr uint64_t intervalNs = 8'333'333u;
	constexpr uint64_t biasNs = 7'300'000u;
	constexpr uint64_t sourceReadyNs = 100'000'000u;
	constexpr uint64_t cadenceNs = 21'700'000u;
	constexpr DisplayGrid_t rawGrid = {
		.D0 = 108'333'333u,
		.W0 = 107'333'333u,
		.T = intervalNs,
	};
	const DisplayGrid_t grid = apply_present_bias( rawGrid, biasNs );
	const RealAnchorState_t anchor = {
		.realFrameId = 1u,
		.sourceReadyNs = sourceReadyNs,
		.provisionalTargetNs = apply_present_bias_ns( sourceReadyNs, biasNs ),
		.provisionalBiasNs = biasNs,
		.epoch = 1u,
	};
	const CadencePredictorState cadence = trained_cadence( cadenceNs );
	CausalPlanOptions_t options = {
		.nowNs = 107'000'000u,
		.gridEpoch = 1u,
		.presentBiasNs = biasNs,
		.sourceTimestampsReliable = true,
		.dedicatedQueue = true,
	};

	const CausalSlotPlan_t first = plan_next_causal_slot(
		grid, anchor, cadence, options );
	REQUIRE( first.admit );
	options.afterTargetNs = first.targetNs;
	const CausalSlotPlan_t second = plan_next_causal_slot(
		grid, anchor, cadence, options );

	const FixedCadenceAdmission oldAsymmetricAdmission =
		fixed_cadence_admission( sourceReadyNs, cadence, options.nowNs,
			second.wakeNs, intervalNs );
	REQUIRE_FALSE( oldAsymmetricAdmission.generateBackup );
	CHECK( oldAsymmetricAdmission.predictedReadyNs == 121'700'000u );
	CHECK( oldAsymmetricAdmission.safetyMarginNs == 260'416u );
	CHECK( second.wakeNs == 122'966'666u );
	CHECK( remove_present_bias_ns( second.wakeNs, biasNs ) == 115'666'666u );
	CHECK( apply_present_bias_ns(
		oldAsymmetricAdmission.predictedReadyNs
			+ oldAsymmetricAdmission.safetyMarginNs,
		biasNs ) > second.targetNs );
	CHECK( second.admit );
}

TEST_CASE( "zero present bias retains source-wake admission behavior", "[framegen][deadline]" )
{
	constexpr DisplayGrid_t grid = {
		.D0 = 108'333'333u,
		.W0 = 107'333'333u,
		.T = 8'333'333u,
	};
	const RealAnchorState_t anchor = {
		.realFrameId = 1u,
		.sourceReadyNs = 100'000'000u,
		.provisionalTargetNs = 100'000'000u,
		.epoch = 1u,
	};
	const CausalPlanOptions_t options = {
		.nowNs = 100'100'000u,
		.gridEpoch = 1u,
		.presentBiasNs = 0,
		.forwardStrengthCap = 3.0f,
		.sourceTimestampsReliable = true,
		.dedicatedQueue = true,
	};

	for ( const uint64_t cadenceNs : { 7'000'000u, 8'000'000u } )
	{
		const CadencePredictorState cadence = trained_cadence( cadenceNs );
		const FixedCadenceAdmission unchanged = fixed_cadence_admission(
			anchor.sourceReadyNs, cadence, options.nowNs, grid.W0, grid.T );
		CHECK( plan_next_causal_slot( grid, anchor, cadence, options ).admit
			== unchanged.generateBackup );
	}
}

TEST_CASE( "admission skips a real displayable before the slot vblank", "[framegen][deadline]" )
{
	constexpr uint64_t intervalNs = 8'333'333u;
	constexpr int64_t biasNs = 7'300'000;
	constexpr uint64_t sourceReadyNs = 100'000'000u;
	const DisplayGrid_t grid = apply_present_bias( {
		.D0 = 116'666'666u,
		.W0 = 115'666'666u,
		.T = intervalNs,
	}, biasNs );
	const RealAnchorState_t anchor = {
		.realFrameId = 1u,
		.sourceReadyNs = sourceReadyNs,
		.provisionalTargetNs = apply_present_bias_ns( sourceReadyNs, biasNs ),
		.provisionalBiasNs = biasNs,
		.epoch = 1u,
	};
	const CadencePredictorState cadence = trained_cadence( 13'000'000u );
	const CausalPlanOptions_t options = {
		.nowNs = 107'000'000u,
		.gridEpoch = 1u,
		.presentBiasNs = biasNs,
		.sourceTimestampsReliable = true,
		.dedicatedQueue = true,
	};

	const FixedCadenceAdmission admission = fixed_cadence_admission(
		sourceReadyNs, cadence, options.nowNs,
		remove_present_bias_ns( grid.W0, biasNs ), intervalNs );
	REQUIRE( apply_present_bias_ns(
		admission.predictedReadyNs + admission.safetyMarginNs,
		biasNs ) < grid.D0 );
	const CausalSlotPlan_t plan = plan_next_causal_slot(
		grid, anchor, cadence, options );
	CHECK_FALSE( plan.admit );
	CHECK( plan.skipReason == DeadlineSkipReason_t::NextRealSafelyDue );
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

TEST_CASE( "bidir pressure has a ring-derived hard ceiling and ordered shedding", "[framegen][deadline]" )
{
	CHECK( bidir_pending_hard_capacity( 8u, 2u ) == 6u );
	CHECK( bidir_pending_hard_capacity( 10u, 3u ) == 8u );
	CHECK( bidir_pending_hard_capacity( 12u, 4u ) == 10u );
	CHECK( bidir_pending_hard_capacity( 2u, 4u ) == 0u );

	const std::array entries = {
		BidirQueueEntry_t{ BidirSlotKind_t::RealEndpoint, 100u, 1u },
		BidirQueueEntry_t{ BidirSlotKind_t::Generated, 200u, 0u },
		BidirQueueEntry_t{ BidirSlotKind_t::Generated, 300u, 0u },
		BidirQueueEntry_t{ BidirSlotKind_t::RealEndpoint, 400u, 2u },
		BidirQueueEntry_t{ BidirSlotKind_t::Generated, 500u, 0u },
	};
	const BidirQueueShedPlan_t generatedFirst =
		plan_bidir_queue_shed( entries, 3u );
	REQUIRE( generatedFirst.indices.size() == 2u );
	CHECK( generatedFirst.indices[ 0 ] == 1u );
	CHECK( generatedFirst.indices[ 1 ] == 2u );
	CHECK( generatedFirst.generated == 2u );
	CHECK( generatedFirst.endpoints == 0u );

	const std::array endpointBacklog = {
		BidirQueueEntry_t{ BidirSlotKind_t::RealEndpoint, 100u, 1u },
		BidirQueueEntry_t{ BidirSlotKind_t::Generated, 150u, 0u },
		BidirQueueEntry_t{ BidirSlotKind_t::RealEndpoint, 200u, 2u },
		BidirQueueEntry_t{ BidirSlotKind_t::RealEndpoint, 200u, 3u },
		BidirQueueEntry_t{ BidirSlotKind_t::RealEndpoint, 300u, 4u },
	};
	const BidirQueueShedPlan_t endpointShed =
		plan_bidir_queue_shed( endpointBacklog, 2u );
	REQUIRE( endpointShed.indices.size() == 3u );
	CHECK( endpointShed.indices[ 0 ] == 0u );
	CHECK( endpointShed.indices[ 1 ] == 1u );
	CHECK( endpointShed.indices[ 2 ] == 2u );
	CHECK( endpointShed.generated == 1u );
	CHECK( endpointShed.endpoints == 2u );
	REQUIRE( endpointShed.newestRetainedEndpoint.has_value() );
	CHECK( *endpointShed.newestRetainedEndpoint == 4u );
	CHECK_FALSE( bidir_queue_forces_drain( 5u, 6u ) );
	CHECK( bidir_queue_forces_drain( 6u, 6u ) );
}

TEST_CASE( "bidir 46 fps into 120 Hz remains bounded for 1000 frames", "[framegen][deadline]" )
{
	constexpr uint64_t displayIntervalNs = 8'333'333u;
	constexpr uint64_t sourceIntervalNs = 1'000'000'000u / 46u;
	constexpr size_t ringImages = 8u;
	constexpr uint32_t multiplier = 2u;
	constexpr size_t hardCapacity =
		bidir_pending_hard_capacity( ringImages, multiplier );
	std::vector<BidirQueueEntry_t> queue;
	uint64_t nextVblankNs = displayIntervalNs;
	size_t maxDepth = 0u;
	size_t compositeBlockingStates = 0u;

	for ( uint64_t frame = 1u; frame <= 1'000u; frame++ )
	{
		const uint64_t sourceReadyNs = frame * sourceIntervalNs;
		while ( nextVblankNs <= sourceReadyNs )
		{
			if ( !queue.empty()
				&& ( queue.front().targetNs <= nextVblankNs
					|| bidir_queue_forces_drain(
						queue.size(), hardCapacity ) ) )
				queue.erase( queue.begin() );
			nextVblankNs += displayIntervalNs;
		}

		// Reserve the real endpoint plus every empty display slot in this source
		// interval before the composite records them.
		const uint64_t endpointTargetNs = nextVblankNs + sourceIntervalNs;
		std::vector<BidirQueueEntry_t> incoming;
		for ( uint64_t targetNs = nextVblankNs + displayIntervalNs;
			targetNs < endpointTargetNs; targetNs += displayIntervalNs )
		{
			incoming.push_back( {
				BidirSlotKind_t::Generated, targetNs, 0u } );
		}
		incoming.push_back( {
			BidirSlotKind_t::RealEndpoint, endpointTargetNs, frame } );

		queue.insert( queue.end(), incoming.begin(), incoming.end() );
		std::ranges::sort( queue, {}, &BidirQueueEntry_t::targetNs );
		const BidirQueueShedPlan_t shed =
			plan_bidir_queue_shed( queue, hardCapacity );
		for ( auto it = shed.indices.rbegin(); it != shed.indices.rend(); ++it )
			queue.erase( queue.begin() + *it );

		maxDepth = std::max( maxDepth, queue.size() );
		// One non-queue ring owner models ordinary backend/history ownership;
		// the second reserved image must remain acquirable by this composite.
		compositeBlockingStates += queue.size() + 1u >= ringImages;
		REQUIRE( queue.size() <= hardCapacity );
		REQUIRE( std::ranges::any_of( queue,
			[&]( const BidirQueueEntry_t &entry ) {
				return entry.kind == BidirSlotKind_t::RealEndpoint
					&& entry.realFrameId == frame;
			} ) );
	}

	CHECK( maxDepth <= hardCapacity );
	CHECK( compositeBlockingStates == 0u );
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
