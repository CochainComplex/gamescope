#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "framegen/deadline.hpp"
#include "framegen/hud.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
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

TEST_CASE( "framegen HUD describes causal net cross-GPU state", "[framegen][hud]" )
{
	const FramegenHudSnapshot_t snapshot = {
		.version = "0.1.0+76e6d5",
		.deviceName = "  AMD Radeon Graphics  ",
		.otherDeviceName = "NVIDIA GeForce RTX 3070",
		.renderOrigin = "linear",
		.mode = GamescopeFramegenMode::Motion,
		.pipeline = GamescopeFramegenPipeline::Learned,
		.multiplier = 2u,
		.refreshMilliHz = 120'000u,
		.clientBuffersStaged = true,
		.netRequested = true,
		.netActive = true,
		.netOnline = true,
		.adapt = true,
		.real = 200u,
		.generated = 390u,
		.repeats = 10u,
		.biasTenthsMs = 1,
		.deadlineHitPercent = 99u,
		.pacingSdTenthsMs = 7u,
		.deadlineHitHistory = { 99.0, 99.0, 99.0, 99.0, 99.0, 99.0,
			99.0, 99.0, 99.0, 99.0, 99.0, 99.0 },
		.pacingSdMsHistory = { 0.7, 0.7, 0.7, 0.7, 0.7, 0.7,
			0.7, 0.7, 0.7, 0.7, 0.7, 0.7 },
		.netTrainedSteps = 1'200u,
		.netProfileLoaded = true,
	};
	const FramegenHudText_t text = format_framegen_hud( 2u, snapshot );

	REQUIRE( text.lineCount == 7u );
	CHECK( text.lines[0].data() == std::string_view{
		"gameslop 0.1.0+76e6d5         motion x2 . learned . 120Hz fixed" } );
	CHECK( text.lines[1].data() == std::string_view{
		"present  AMD Radeon Graphics render NVIDIA GeForce RTX  buffers xGPU" } );
	CHECK( text.lines[2].data() == std::string_view{
		"modes    bidir:off  base:off  net:online  adapt:on" } );
	CHECK( text.lines[3].data() == std::string_view{
		"rates    source 40fps[200] gen 78fps[390] repeat 2fps fill 98%" } );
	CHECK( text.lines[4].data() == std::string_view{
		"ladder   learned/motion rung 0/0  full" } );
	std::string paceLine = "pace     hit 99%  ";
	paceLine.append( k_nFramegenHudSparklineSamples, '\x08' );
	paceLine += "   jitter +/-0.7ms  ";
	paceLine.append( k_nFramegenHudSparklineSamples, '\x02' );
	CHECK( text.lines[5].data() == paceLine );
	CHECK( text.lines[6].data() == std::string_view{
		"net      online . 1.2k steps . profile loaded" } );
	for ( uint32_t line = 0u; line < text.lineCount; line++ )
		CHECK( std::strlen( text.lines[line].data() ) <= k_uFramegenHudMaxColumns );

	const FramegenHudUniform_t uniform = make_framegen_hud_uniform( text, true );
	CHECK( uniform.lineCount == 7u );
	CHECK( uniform.widthChars == 68u );
	CHECK( uniform.hdr == 1u );
	CHECK( ( uniform.text[0] & 0xffu ) == static_cast<uint32_t>( 'g' ) );
	CHECK( uniform.font[static_cast<uint32_t>( 'A' ) * 2u]
		== static_cast<uint32_t>( k_uFramegenHudFont8x8[static_cast<uint32_t>( 'A' )] ) );

	auto tiledSnapshot = snapshot;
	tiledSnapshot.renderOrigin = "NVIDIA";
	const FramegenHudText_t tiledText = format_framegen_hud( 2u, tiledSnapshot );
	CHECK( tiledText.lines[1].data() == std::string_view{
		"present  AMD Radeon Graphics render NVIDIA              buffers xGPU" } );

	auto ambiguousLinearSnapshot = snapshot;
	ambiguousLinearSnapshot.otherDeviceName = nullptr;
	const FramegenHudText_t ambiguousLinearText =
		format_framegen_hud( 2u, ambiguousLinearSnapshot );
	CHECK( ambiguousLinearText.lines[1].data() == std::string_view{
		"present  AMD Radeon Graphics render other-GPU           buffers xGPU" } );
}

TEST_CASE( "framegen HUD exposes requested modes that fell back", "[framegen][hud]" )
{
	const FramegenHudSnapshot_t snapshot = {
		.version = "0.1.0+76e6d5",
		.deviceName = "AMD Radeon RX 5700 XT",
		.renderOrigin = nullptr,
		.mode = GamescopeFramegenMode::Motion,
		.pipeline = GamescopeFramegenPipeline::Learned,
		.multiplier = 2u,
		.refreshMilliHz = 120'000u,
		.bidirRequested = true,
		.real = 200u,
		.generated = 390u,
		.repeats = 10u,
	};
	const FramegenHudText_t text = format_framegen_hud( 1u, snapshot );

	REQUIRE( text.lineCount == 5u );
	CHECK( text.lines[1].data() == std::string_view{
		"present  AMD Radeon RX 5700  render n/a                 buffers local" } );
	CHECK( text.lines[2].data() == std::string_view{
		"modes    bidir:requested(OFF)  base:off  net:off  adapt:off" } );
	auto netOnly = snapshot;
	netOnly.bidirRequested = false;
	netOnly.netRequested = true;
	const FramegenHudText_t netText = format_framegen_hud( 1u, netOnly );
	CHECK( netText.lines[2].data() == std::string_view{
		"modes    bidir:off  base:off  net:requested(OFF)  adapt:off" } );
	auto bothRequested = snapshot;
	bothRequested.netRequested = true;
	const FramegenHudText_t bothText = format_framegen_hud( 1u, bothRequested );
	CHECK( bothText.lines[2].data() == std::string_view{
		"modes    bidir:requested(OFF)  base:off  net:requested(OFF)  adapt:off" } );
	auto vrrOnly = snapshot;
	vrrOnly.bidirRequested = false;
	vrrOnly.vrrRequested = true;
	const FramegenHudText_t vrrText = format_framegen_hud( 1u, vrrOnly );
	CHECK( vrrText.lines[2].data() == std::string_view{
		"modes    bidir:off  base:off  net:off  adapt:off  vrr:requested(off)" } );
	for ( uint32_t line = 0u; line < text.lineCount; line++ )
		CHECK( std::strlen( text.lines[line].data() ) <= k_uFramegenHudMaxColumns );
}

TEST_CASE( "framegen HUD keeps single-GPU no-net level one lean", "[framegen][hud]" )
{
	const FramegenHudSnapshot_t snapshot = {
		.version = "0.1.0+76e6d5",
		.deviceName = "AMD Radeon 890M",
		.mode = GamescopeFramegenMode::Motion,
		.pipeline = GamescopeFramegenPipeline::Learned,
		.multiplier = 2u,
		.refreshMilliHz = 120'000u,
		.real = 200u,
		.generated = 390u,
		.repeats = 10u,
	};
	const FramegenHudText_t text = format_framegen_hud( 1u, snapshot );

	REQUIRE( text.lineCount == 5u );
	CHECK( text.lines[1].data() == std::string_view{
		"present  AMD Radeon 890M     render n/a                 buffers local" } );
	CHECK( text.lines[2].data() == std::string_view{
		"modes    bidir:off  base:off  net:off  adapt:off" } );
	CHECK( text.lines[3].data() == std::string_view{
		"rates    source 40fps[200] gen 78fps[390] repeat 2fps fill 98%" } );
	CHECK( text.lines[4].data() == std::string_view{
		"ladder   learned/motion rung 0/0  full" } );
}

TEST_CASE( "framegen HUD describes ladder recovery states", "[framegen][hud]" )
{
	FramegenHudSnapshot_t snapshot = {
		.mode = GamescopeFramegenMode::Extrapolate,
		.pipeline = GamescopeFramegenPipeline::Checked,
		.refreshMilliHz = 120'000u,
		.real = 1'234u,
		.generated = 56'789u,
		.ladderSteps = 2u,
		.ladderMaxSteps = 4u,
		.ladderHold = 3u,
		.ladderRecoveryBackoffDecisions = 90u,
	};
	FramegenHudText_t text = format_framegen_hud( 1u, snapshot );
	CHECK( text.lines[3].data() == std::string_view{
		"rates    source 246fps[1.2k] gen 999fps[56.8k] repeat 0fps fill 100%" } );
	CHECK( text.lines[4].data() == std::string_view{
		"ladder   checked/extrapolate rung 2/4  hold" } );

	snapshot.ladderHold = 0u;
	snapshot.ladderRecoveryProbationRemaining = 46u;
	text = format_framegen_hud( 1u, snapshot );
	CHECK( text.lines[4].data() == std::string_view{
		"ladder   checked/extrapolate rung 2/4  probation 2s" } );

	snapshot.ladderRecoveryProbationRemaining = 0u;
	snapshot.ladderRecoveryDecisionsSinceClimb = 44u;
	text = format_framegen_hud( 1u, snapshot );
	CHECK( text.lines[4].data() == std::string_view{
		"ladder   checked/extrapolate rung 2/4  recover in 2s" } );

	snapshot.ladderRecoveryDecisionsSinceClimb = 90u;
	text = format_framegen_hud( 1u, snapshot );
	CHECK( text.lines[4].data() == std::string_view{
		"ladder   checked/extrapolate rung 2/4  watching" } );
}

TEST_CASE( "framegen HUD sparkline mapping is monotone and saturated",
	"[framegen][hud]" )
{
	const std::array<double, k_nFramegenHudSparklineSamples> values = {
		0.0, 10.0, 20.0, 30.0, 40.0, 50.0,
		60.0, 70.0, 80.0, 90.0, 100.0, 150.0,
	};
	const FramegenHudSparkline_t sparkline = framegen_hud_sparkline( values, 100.0 );
	for ( size_t i = 1u; i < values.size(); i++ )
	{
		CHECK( static_cast<uint8_t>( sparkline[i - 1u] )
			<= static_cast<uint8_t>( sparkline[i] ) );
	}
	CHECK( static_cast<uint8_t>( sparkline.front() ) == 0x01u );
	CHECK( static_cast<uint8_t>( sparkline[10] ) == 0x08u );
	CHECK( static_cast<uint8_t>( sparkline[11] ) == 0x08u );
}

TEST_CASE( "framegen HUD step formatting rounds within its own unit",
	"[framegen][hud]" )
{
	auto formatted = []( uint64_t steps ) {
		char buffer[32] = {};
		framegen_hud_format_steps( buffer, sizeof( buffer ), steps );
		return std::string( buffer );
	};

	CHECK( formatted( 0u ) == "0" );
	CHECK( formatted( 999u ) == "999" );
	CHECK( formatted( 1'000u ) == "1.0k" );
	CHECK( formatted( 999'949u ) == "999.9k" );
	// Rounding must not push the "k" branch past its own unit.
	CHECK( formatted( 999'950u ) == "1.0m" );
	CHECK( formatted( 999'999u ) == "1.0m" );
	CHECK( formatted( 1'000'000u ) == "1.0m" );
}

TEST_CASE( "framegen HUD sparkline maps empty windows to the shortest bar",
	"[framegen][hud]" )
{
	const std::array<double, k_nFramegenHudSparklineSamples> empty = {};
	const FramegenHudSparkline_t sparkline = framegen_hud_sparkline( empty, 4.0 );
	for ( char glyph : std::string_view{ sparkline.data(), empty.size() } )
		CHECK( static_cast<uint8_t>( glyph ) == 0x01u );
}

TEST_CASE( "framegen HUD embeds eight full-width spark bars", "[framegen][hud]" )
{
	for ( uint32_t level = 1u; level <= 8u; level++ )
	{
		const uint64_t expected = level == 8u
			? UINT64_MAX : ( uint64_t{ 1u } << ( level * 8u ) ) - 1u;
		CHECK( k_uFramegenHudFont8x8[level] == expected );
	}
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

TEST_CASE( "causal slots at or past the next real frame are skipped on request", "[framegen][deadline]" )
{
	// 30 fps on 60 Hz. After anchor+T the next unused target is
	// anchor+2T, phase exactly 1: the next real frame is due there, so a
	// correction replan must not extrapolate onto it.
	constexpr uint64_t epoch = 9u;
	constexpr uint64_t T = 16'666'667u;
	const DisplayGrid_t grid = { .D0 = 1'000'000u, .W0 = 999'000u, .T = T };
	const RealAnchorState_t anchor = {
		.realFrameId = 1u,
		.sourceReadyNs = grid.D0,
		.provisionalTargetNs = grid.D0,
		.correctedFlipNs = grid.D0,
		.epoch = epoch,
	};
	const auto plan = [&]( uint64_t intervalNs, bool skip, float strength ) {
		auto options = dedicated_backup_options( grid.D0, epoch );
		options.afterTargetNs = grid.target( 1u );
		options.configuredStrength = strength;
		options.skipAtOrPastNextReal = skip;
		return plan_next_causal_slot( grid, anchor, trained_cadence( intervalNs ), options );
	};
	for ( const float strength : { k_flNeutralStrength, 0.0f } )
	{
		CAPTURE( strength );
		const CausalSlotPlan_t below = plan( 2u * T + 1u, true, strength );
		CHECK( below.targetNs == grid.target( 2u ) );
		CHECK( below.phase < 1.0 );
		CHECK( below.admit );
		for ( const uint64_t intervalNs : { 2u * T, 2u * T - 1u, 3u * T / 2u } )
		{
			CAPTURE( intervalNs );
			const CausalSlotPlan_t skipped = plan( intervalNs, true, strength );
			CHECK( skipped.targetNs == grid.target( 2u ) );
			CHECK_FALSE( skipped.admit );
			CHECK( skipped.skipReason == DeadlineSkipReason_t::ForwardCap );
			// Unset, the strength cap alone decides, as before.
			const CausalSlotPlan_t unset = plan( intervalNs, false, strength );
			CHECK( unset.admit == ( unset.rawStrength <= dedicated_backup_options( 0u, 0u ).forwardStrengthCap ) );
			CHECK( unset.targetNs == skipped.targetNs );
			CHECK( unset.wakeNs == skipped.wakeNs );
			CHECK( unset.phase == skipped.phase );
		}
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
		anchor, matureBias, 9u, 2'301u, 300u, 1'000u );
	REQUIRE( large.matched );
	CHECK( large.discardProvisional );
	REQUIRE( large.anchor.correctedFlipNs );
	CHECK( *large.anchor.correctedFlipNs == 2'301u );

	const AnchorCorrection_t small = apply_flip_feedback(
		anchor, matureBias, 9u, 2'299u, 300u, 1'000u );
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
		anchor, matureBias, 42u, 18'334u, 300u, 8'334u );
	REQUIRE( correction.discardProvisional );
	CHECK( discard_pending_provisional_slot( correction, 42u, true ) );
	CHECK_FALSE( discard_pending_provisional_slot( correction, 42u, false ) );
	CHECK_FALSE( discard_pending_provisional_slot( correction, 41u, true ) );

	const AnchorCorrection_t small = apply_flip_feedback(
		anchor, matureBias, 42u, 10'300u, 300u, 8'334u );
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
		7u, 100'000'000u, interval / 32u, interval );
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
			anchor, bias, frame, rawTargetNs + intervalNs, guardNs,
			intervalNs );
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
			guardNs, 10'000u );
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

TEST_CASE( "mature present-bias outlier is not folded", "[framegen][deadline]" )
{
	constexpr uint64_t intervalNs = 8'333u;
	constexpr PresentBiasState_t mature = {
		.emaNs = 8'000,
		.samples = k_uPresentBiasWarmupSamples,
	};
	constexpr uint64_t outlierNs = 8'000u + intervalNs;
	constexpr PresentBiasUpdate_t update = update_present_bias(
		mature, 100'000u, 100'000u + outlierNs, 300u,
		intervalNs, false );
	constexpr PresentBiasUpdate_t hitchUpdate = update_present_bias(
		mature, 100'000u, 100'100u, 300u, intervalNs, true );

	STATIC_REQUIRE( update.residualNs == static_cast<int64_t>( outlierNs ) );
	STATIC_REQUIRE( update.state.emaNs == mature.emaNs );
	STATIC_REQUIRE( update.state.samples == mature.samples );
	STATIC_REQUIRE( update.state.consecutiveGuardExceeds == 1u );
	STATIC_REQUIRE_FALSE( update.discardProvisional );
	STATIC_REQUIRE( hitchUpdate.state.emaNs == mature.emaNs );
	STATIC_REQUIRE( hitchUpdate.state.samples == mature.samples );
}

TEST_CASE( "normal present-bias residual still folds one eighth", "[framegen][deadline]" )
{
	constexpr PresentBiasState_t mature = {
		.emaNs = 8'000,
		.samples = k_uPresentBiasWarmupSamples,
	};
	constexpr PresentBiasUpdate_t update = update_present_bias(
		mature, 100'000u, 100'800u, 1'000u, 8'333u, false );

	STATIC_REQUIRE( update.state.emaNs == 8'100 );
	STATIC_REQUIRE( update.state.samples == mature.samples + 1u );
	STATIC_REQUIRE( update.state.consecutiveGuardExceeds == 0u );
}

TEST_CASE( "present-bias warmup still folds outlier and hitch samples", "[framegen][deadline]" )
{
	constexpr uint64_t intervalNs = 8'333u;
	constexpr PresentBiasState_t warming = {
		.emaNs = 8'000,
		.samples = k_uPresentBiasWarmupSamples - 1u,
	};
	constexpr int64_t residualNs = 8'000 + static_cast<int64_t>( intervalNs );
	constexpr PresentBiasUpdate_t update = update_present_bias(
		warming, 100'000u,
		100'000u + static_cast<uint64_t>( residualNs ),
		300u, intervalNs, true );

	STATIC_REQUIRE( update.state.emaNs == warming.emaNs + residualNs / 8 );
	STATIC_REQUIRE( update.state.samples == k_uPresentBiasWarmupSamples );
	STATIC_REQUIRE( update.state.consecutiveGuardExceeds == 0u );
}

TEST_CASE( "discard-only present-bias outliers retain the two-strike guard",
	"[framegen][deadline]" )
{
	constexpr uint64_t intervalNs = 8'333u;
	constexpr PresentBiasState_t mature = {
		.emaNs = 8'000,
		.samples = k_uPresentBiasWarmupSamples,
	};
	constexpr uint64_t outlierNs = 8'000u + intervalNs;
	constexpr PresentBiasUpdate_t first = update_present_bias(
		mature, 100'000u, 100'000u + outlierNs, 300u,
		intervalNs, false );
	constexpr PresentBiasUpdate_t second = update_present_bias(
		first.state, 200'000u, 200'000u + outlierNs, 300u,
		intervalNs, false );

	STATIC_REQUIRE_FALSE( first.discardProvisional );
	STATIC_REQUIRE( second.discardProvisional );
	STATIC_REQUIRE( second.state.emaNs == mature.emaNs );
	STATIC_REQUIRE( second.state.samples == mature.samples );
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
	timing.minimumViablePresentLead = {
		.lowWaterNs = 750'000u,
		.backoffFloorNs = 1'000'000u,
		.samples = 17u,
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
	CHECK( contentReprime.state.minimumViablePresentLead.lowWaterNs == 750'000u );
	CHECK( contentReprime.state.minimumViablePresentLead.backoffFloorNs == 1'000'000u );
	CHECK( contentReprime.state.minimumViablePresentLead.samples == 17u );

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
	CHECK( gridChange.state.minimumViablePresentLead.samples == 0u );

	const auto checkChainReset = [&]( const DisplayChainKey_t &changedChain ) {
		const DisplayChainTimingTransition_t changed =
			observe_display_chain( timing, changedChain );
		REQUIRE( changed.displayChainChanged );
		CHECK( changed.state.presentBias.samples == 0u );
		CHECK( changed.state.presentLead.samples == 0u );
		CHECK( changed.state.minimumViablePresentLead.samples == 0u );
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
	const BidirEpoch_t epoch = establish_bidir_epoch( 500u, 2u );
	const std::array endpoints = {
		BidirEndpoint_t{ .realFrameId = 10u, .sourceReadyNs = 500u, .recordNs = 500u },
		BidirEndpoint_t{ .realFrameId = 11u, .sourceReadyNs = 590u, .recordNs = 590u },
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

TEST_CASE( "bidir epoch shows every real D after its record", "[framegen][deadline]" )
{
	STATIC_REQUIRE_FALSE( establish_bidir_epoch( 0u, 3u ).valid );
	STATIC_REQUIRE_FALSE( establish_bidir_epoch( 4'000u, 0u ).valid );

	constexpr BidirEpoch_t epoch = establish_bidir_epoch( 4'000u, 3u );
	STATIC_REQUIRE( epoch.valid );
	STATIC_REQUIRE( epoch.delayNs == 4'000u );
	STATIC_REQUIRE( epoch.endpoint_display_time( 1'000u ) == 5'000u );
	STATIC_REQUIRE( epoch.endpoint_display_time( 1'250u ) == 5'250u );
	STATIC_REQUIRE( epoch.endpoint_display_time( UINT64_MAX - 1u ) == UINT64_MAX );
}

TEST_CASE( "bidir slots follow record time, not source time", "[framegen][deadline]" )
{
	// The two reals were ready 400 apart at the source, but a hold before the
	// record (a frame limiter) recorded them 300 apart: ideals, targets and
	// phases follow the records.
	const DisplayGrid_t grid = { .D0 = 1'000u, .W0 = 990u, .T = 100u };
	const BidirEpoch_t epoch = establish_bidir_epoch( 500u, 4u );
	const std::array endpoints = {
		BidirEndpoint_t{ .realFrameId = 1u, .sourceReadyNs = 100u, .recordNs = 500u },
		BidirEndpoint_t{ .realFrameId = 2u, .sourceReadyNs = 500u, .recordNs = 800u },
	};
	const BidirPlan_t plan = plan_bidir_slots( grid, epoch, endpoints, 4u, 4u );
	REQUIRE( plan.validEpoch );
	std::vector<uint64_t> targets;
	for ( const BidirSlot_t &slot : plan.slots )
	{
		targets.push_back( slot.targetNs );
		if ( slot.kind == BidirSlotKind_t::Generated )
			CHECK( slot.phase == Approx( static_cast<double>( slot.targetNs - 1'000u ) / 300.0 ) );
	}
	CHECK( targets == std::vector<uint64_t>{ 1'000u, 1'100u, 1'200u, 1'300u } );
}

TEST_CASE( "bidir epoch re-establishment does not thrash across a cut sequence", "[framegen][deadline]" )
{
	constexpr uint64_t gridEpoch = 9u;
	BidirEpoch_t epoch = establish_bidir_epoch( 4'000u, gridEpoch );
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
			epoch = establish_bidir_epoch( 4'000u + i * 100u, gridEpoch );
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

TEST_CASE( "bidir 24 fps on 60 Hz uses a drift-free absolute epoch", "[framegen][deadline]" )
{
	const DisplayGrid_t grid = { .D0 = 11'000u, .W0 = 10'999u, .T = 2u };
	const BidirEpoch_t epoch = establish_bidir_epoch( 1'000u, 6u );
	std::vector<BidirEndpoint_t> endpoints;
	for ( uint64_t i = 0u; i <= 500u; i++ )
	{
		endpoints.push_back( { .realFrameId = i + 1u, .sourceReadyNs = 10'000u + i * 5u,
			.recordNs = 10'000u + i * 5u } );
	}

	const BidirPlan_t plan = plan_bidir_slots( grid, epoch, endpoints, 4u, 6u );
	REQUIRE( plan.validEpoch );
	for ( uint64_t i = 1u; i <= 500u; i++ )
	{
		const uint64_t idealNs = 11'000u + i * 5u;
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
	const BidirEpoch_t epoch = establish_bidir_epoch( 500u, 9u );
	const std::array endpoints = {
		BidirEndpoint_t{ .realFrameId = 1u, .sourceReadyNs = 500u, .recordNs = 500u },
		BidirEndpoint_t{ .realFrameId = 2u, .sourceReadyNs = 530u, .recordNs = 530u },
		BidirEndpoint_t{ .realFrameId = 3u, .sourceReadyNs = 560u, .recordNs = 560u },
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
	const BidirEpoch_t epoch = establish_bidir_epoch( 500u, 1u );
	const std::array endpoints = {
		BidirEndpoint_t{ .realFrameId = 1u, .sourceReadyNs = 500u, .recordNs = 500u },
		BidirEndpoint_t{ .realFrameId = 2u, .sourceReadyNs = 1'000u, .recordNs = 1'000u },
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
	// Interpolations go oldest-first, only as many as needed.
	const BidirQueueShedPlan_t generatedFirst =
		plan_bidir_queue_shed( entries, 3u );
	REQUIRE( generatedFirst.indices.size() == 2u );
	CHECK( generatedFirst.indices[ 0 ] == 1u );
	CHECK( generatedFirst.indices[ 1 ] == 2u );
	CHECK( plan_bidir_queue_shed( entries, 5u ).indices.empty() );

	// Reals are never shed: beyond the capacity only the interpolations go.
	const std::array endpointBacklog = {
		BidirQueueEntry_t{ BidirSlotKind_t::RealEndpoint, 100u, 1u },
		BidirQueueEntry_t{ BidirSlotKind_t::Generated, 150u, 0u },
		BidirQueueEntry_t{ BidirSlotKind_t::RealEndpoint, 200u, 2u },
		BidirQueueEntry_t{ BidirSlotKind_t::RealEndpoint, 300u, 3u },
	};
	const BidirQueueShedPlan_t endpointShed =
		plan_bidir_queue_shed( endpointBacklog, 2u );
	REQUIRE( endpointShed.indices.size() == 1u );
	CHECK( endpointShed.indices[ 0 ] == 1u );
	const std::array realsOnly = { endpointBacklog[ 0 ], endpointBacklog[ 2 ], endpointBacklog[ 3 ] };
	CHECK( plan_bidir_queue_shed( realsOnly, 0u ).indices.empty() );
}

TEST_CASE( "bidir forced drain engages only at the ring ceiling", "[framegen][deadline]" )
{
	// { depth, hard capacity } -> forced
	STATIC_REQUIRE_FALSE( bidir_queue_forces_drain( 0u, 6u ) );
	STATIC_REQUIRE_FALSE( bidir_queue_forces_drain( 1u, 6u ) );
	STATIC_REQUIRE_FALSE( bidir_queue_forces_drain( 5u, 6u ) );
	STATIC_REQUIRE( bidir_queue_forces_drain( 6u, 6u ) );
	STATIC_REQUIRE( bidir_queue_forces_drain( 7u, 6u ) );
	// A ring too small for any pending timeline forces every non-empty queue.
	STATIC_REQUIRE_FALSE( bidir_queue_forces_drain( 0u, 0u ) );
	STATIC_REQUIRE( bidir_queue_forces_drain( 1u, 0u ) );
}

namespace
{

// A closed bidir loop on the production planner and presenter: real frames
// arrive on a source cadence, and each composite plans its pair with
// plan_bidir_step and applies the outputs as framegen_bidir_plan_pair does.
// Every vblank, and every composite's paint, presents the queue front by
// bidir_front_action (bidir_front_inputs) and abandons a stale timeline
// (bidir_timeline_stale, clearing the translation). Each composite first sheds
// the queue's interpolations to the ring-derived ceiling, as the composite start
// does. A pair the step does not plan primes like the flip selection: flipped
// live over an empty queue, else queued behind it, replacing a queued-behind
// tail or the newest real of a queue whose reals fill the ring
// (select_bidir_flip). A real is consumed when it enters the timeline, queued
// or flipped live. Composites run at the compositor's wake.
//
// Injections: a per-real latency from source time to commit (a real is never
// committed before an older one), a compositor stall (no iteration, then a
// burst of composites or only the newest one latched), a suspension (neither
// composites nor presents), lost vblank presents, presents only at composites,
// an off-grid jump of the opportunities, a scene cut that clears the
// translation, and a queued real whose present image cannot be made.
//
// Source time and record time are separate: a real's latency is the hold from
// its source-ready time to its commit, and it is recorded at the first wake
// after that (recordNs).
constexpr uint64_t k_ulLoopVblankNs = 16'666'667u;
constexpr uint64_t k_ulLoopWakeLeadNs = 3'000'000u;
constexpr uint64_t k_ulLoopStartNs = 1'000'000'000u;
constexpr uint64_t k_ulLoopSecondNs = 1'000'000'000u;
constexpr uint64_t k_ulLoopInjectAtNs = k_ulLoopStartNs + 4'000'000'000u;

struct BidirLoopInjection_t
{
	std::function<uint64_t( size_t id, uint64_t sourceNs )> latency;	// source-ready -> commit hold
	uint64_t stallAtNs = 0;		// no compositor iteration for vblanks in [stallAtNs, stallAtNs + stallForNs)
	uint64_t stallForNs = 0;
	size_t compositesPerVblank = 0;	// at most this many composites per vblank (0: all)
	bool latchNewest = false;	// the first vblank after the stall latches only the newest real
	uint64_t suspendAtNs = 0;	// neither composites nor presents in [suspendAtNs, suspendAtNs + suspendForNs)
	uint64_t suspendForNs = 0;
	std::function<bool( uint64_t opportunityNs )> lostPresent;	// the vblank's own present is lost
	bool presentsOnlyAtComposites = false;
	uint64_t jumpAtNs = UINT64_MAX;	// opportunities from jumpAtNs on lie jumpNs later
	uint64_t jumpNs = 0;
	uint64_t cutAtNs = 0;	// a scene cut clears the translation at the first plan from here (0: none)
	uint64_t failedImageRealId = 0;	// this queued real's present image cannot be made (0: none)
	size_t ringImages = 8u;
	uint32_t multiplier = 2u;
};

struct BidirLoopReal_t
{
	uint64_t id = 0;
	uint64_t sourceNs = 0;
	uint64_t recordNs = 0;
	uint64_t flipNs = 0;
	uint64_t delayNs = 0;		// D it was queued with (0: flipped live, or queued outside a plan)
	uint64_t pushNs = 0;		// target - its own slot: displaced behind an older real
	uint64_t pastIdealNs = 0;	// target - its display time on the translation
	bool kept = false;			// queued by a plan that kept the translation
};

struct BidirLoopEvent_t
{
	uint64_t atNs = 0;
	BidirReestablish_t kind = BidirReestablish_t::Kept;
	uint64_t delayNs = 0;
};

struct BidirLoopResult_t
{
	std::vector<BidirLoopReal_t> shown;		// in presentation order
	std::vector<BidirLoopEvent_t> events;	// every planned step that did not keep the translation
	std::vector<uint64_t> servicedNs;		// opportunities that could present
	std::vector<uint64_t> holeNs;			// serviced opportunities that showed nothing new
	std::vector<uint64_t> lateSlotNs;		// a kept plan's interpolation slot planned after its wake
	size_t capacity = 0;
	size_t recorded = 0;
	size_t generated = 0;
	size_t doublePresents = 0;			// after the first establishment's opportunity
	size_t maxDepth = 0;
	size_t queuedAtEnd = 0;				// reals still queued when the run ends
	std::vector<uint64_t> backlogDropNs;	// plans that dropped reals of a backlog that did not drain
	// Reals never shown, by the path that dropped them.
	size_t staleDropped = 0;			// an abandoned stale timeline
	size_t planDropped = 0;				// a plan: beyond the display's bound, or an undrained backlog
	size_t replacedDropped = 0;			// an unplanned composite replaced a queued one
	size_t failedDropped = 0;			// no present image
	uint32_t staleAbandons = 0;
	uint64_t firstEstablishNs = 0;
	uint64_t finalDelayNs = 0;
	uint64_t finalEnvelopeNs = 0;

	size_t dropped() const
	{
		return staleDropped + planDropped + replacedDropped + failedDropped;
	}
	size_t count( BidirReestablish_t kind, uint64_t fromNs = 0u, uint64_t toNs = UINT64_MAX ) const
	{
		return static_cast<size_t>( std::ranges::count_if( events, [&]( const BidirLoopEvent_t &event ) {
			return event.kind == kind && event.atNs >= fromNs && event.atNs < toNs; } ) );
	}
	size_t delay_changes( uint64_t fromNs = 0u, uint64_t toNs = UINT64_MAX ) const
	{
		return count( BidirReestablish_t::Lengthened, fromNs, toNs ) + count( BidirReestablish_t::Shortened, fromNs, toNs );
	}
	uint64_t last_event_ns() const { return events.empty() ? 0u : events.back().atNs; }
	uint64_t last_ns( BidirReestablish_t kind ) const
	{
		const auto it = std::ranges::find( events.rbegin(), events.rend(), kind, &BidirLoopEvent_t::kind );
		return it == events.rend() ? 0u : it->atNs;
	}
	static size_t count_in( const std::vector<uint64_t> &ns, uint64_t fromNs, uint64_t toNs )
	{
		return static_cast<size_t>( std::ranges::count_if( ns,
			[&]( uint64_t at ) { return at >= fromNs && at < toNs; } ) );
	}
	size_t late_slots_from( uint64_t fromNs ) const { return count_in( lateSlotNs, fromNs, UINT64_MAX ); }
	// Opportunities that showed nothing new from fromNs to 300 ms before the
	// last real.
	size_t holes_from( uint64_t fromNs ) const
	{
		return shown.empty() ? 0u : count_in( holeNs, fromNs, shown.back().flipNs - 300'000'000u );
	}
};

struct BidirLoopEntry_t
{
	uint64_t targetNs = 0;
	bool real = false;
	uint64_t id = 0;
	bool behind = false;	// queued behind outside a plan
	uint64_t idealNs = 0;
};

BidirLoopResult_t run_bidir_loop( const std::vector<uint64_t> &intervalsNs, const BidirLoopInjection_t &inject = {} )
{
	constexpr uint64_t T = k_ulLoopVblankNs;
	BidirLoopResult_t r = { .capacity = bidir_pending_hard_capacity( inject.ringImages, inject.multiplier ) };
	std::vector<uint64_t> source = { k_ulLoopStartNs + T - 5'000'000u };
	for ( const uint64_t intervalNs : intervalsNs )
		source.push_back( source.back() + intervalNs );
	std::vector<uint64_t> ready( source.size() );
	for ( size_t id = 0; id < source.size(); id++ )
	{
		ready[ id ] = std::max( id != 0u ? ready[ id - 1u ] : 0u,
			source[ id ] + ( inject.latency ? inject.latency( id, source[ id ] ) : 0u ) );
	}
	std::vector<BidirLoopReal_t> reals( source.size() );

	std::deque<BidirLoopEntry_t> queue;
	BidirTimelineState_t state;
	uint64_t lastFrontNs = 0;
	uint64_t lastPresentNs = 0;
	uint64_t consumedId = 0;
	bool cut = false;
	std::optional<size_t> previous;

	const auto within = []( uint64_t ns, uint64_t atNs, uint64_t forNs ) {
		return forNs != 0u && ns >= atNs && ns < atNs + forNs;
	};
	const auto suspended = [&]( uint64_t ns ) { return within( ns, inject.suspendAtNs, inject.suspendForNs ); };
	const auto entries_of_queue = [&]() {
		std::vector<BidirQueueEntry_t> entries;
		for ( const BidirLoopEntry_t &entry : queue )
		{
			entries.push_back( { entry.real ? BidirSlotKind_t::RealEndpoint : BidirSlotKind_t::Generated,
				entry.targetNs, entry.id, entry.idealNs } );
		}
		return entries;
	};
	const auto view_at = [&]( uint64_t opportunityNs ) {
		return BidirPresenterView_t{
			.opportunityNs = opportunityNs,
			.lastFrontOpportunityNs = lastFrontNs,
			.vblankIntervalNs = T,
			.capacity = r.capacity,
		};
	};
	const auto drop_queue = [&]( size_t &droppedReals ) {
		droppedReals += static_cast<size_t>( std::ranges::count_if( queue, &BidirLoopEntry_t::real ) );
		queue.clear();
		lastFrontNs = 0;
	};
	// Sheds interpolations to capacity, as framegen_bidir_shed_to_capacity does.
	const auto shed = [&]( size_t capacity ) {
		const BidirQueueShedPlan_t plan = plan_bidir_queue_shed( entries_of_queue(), capacity );
		for ( auto it = plan.indices.rbegin(); it != plan.indices.rend(); ++it )
			queue.erase( queue.begin() + static_cast<std::ptrdiff_t>( *it ) );
	};
	const auto find_real = [&]( uint64_t id ) {
		return std::ranges::find_if( queue, [&]( const BidirLoopEntry_t &entry ) { return entry.real && entry.id == id; } );
	};
	const auto show = [&]( uint64_t opportunityNs ) {
		r.doublePresents += r.firstEstablishNs != 0u && opportunityNs > r.firstEstablishNs
			&& lastPresentNs == opportunityNs;
		lastPresentNs = opportunityNs;
	};
	const auto show_real = [&]( uint64_t id, uint64_t opportunityNs ) {
		show( opportunityNs );
		r.shown.push_back( reals[ id ] );
		r.shown.back().flipNs = opportunityNs;
	};
	const auto present = [&]( uint64_t opportunityNs ) {
		const BidirPresenterView_t view = view_at( opportunityNs );
		if ( !queue.empty() && bidir_timeline_stale( queue.front().targetNs, opportunityNs, view.capacity, T ) )
		{
			drop_queue( r.staleDropped );
			state.epoch = {};
			r.staleAbandons++;
			return;
		}
		if ( suspended( opportunityNs ) )
			return;
		while ( !queue.empty() )
		{
			const BidirLoopEntry_t front = queue.front();
			const BidirFrontAction_t action = bidir_front_action( bidir_front_inputs(
				{ front.real ? BidirSlotKind_t::RealEndpoint : BidirSlotKind_t::Generated, front.targetNs, front.id },
				true, queue.size(), view ) );
			if ( action == BidirFrontAction_t::Wait )
				return;
			queue.pop_front();
			if ( action != BidirFrontAction_t::Present )
				continue;
			if ( front.real && front.id == inject.failedImageRealId )
			{
				r.failedDropped++;
				return;
			}
			lastFrontNs = opportunityNs;
			if ( front.real )
				show_real( front.id, opportunityNs );
			else
			{
				show( opportunityNs );
				r.generated++;
			}
			return;
		}
	};
	const auto insert = [&]( BidirLoopEntry_t entry ) {
		const auto it = std::ranges::lower_bound( queue, entry.targetNs, {}, &BidirLoopEntry_t::targetNs );
		if ( it == queue.end() || it->targetNs != entry.targetNs )
			queue.insert( it, entry );
		else if ( entry.real )
		{
			r.replacedDropped += it->real;
			*it = entry;
		}
	};
	// Plans the pair ending at real frame `next` like framegen_bidir_plan_pair;
	// returns false when the step did not plan it.
	const auto plan_pair = [&]( size_t prev, size_t next, uint64_t opportunityNs, uint64_t nowNs ) {
		if ( inject.cutAtNs != 0u && !cut && opportunityNs >= inject.cutAtNs )
		{
			state.epoch = {};
			cut = true;
		}
		const std::vector<BidirQueueEntry_t> entries = entries_of_queue();
		const BidirPlanStep_t step = plan_bidir_step( state, {
			.queue = entries,
			.previous = { prev, source[ prev ], reals[ prev ].recordNs },
			.current = { next, source[ next ], reals[ next ].recordNs },
			.wakeNs = opportunityNs - k_ulLoopWakeLeadNs,
			.nowNs = nowNs,
			.consumedFrameId = consumedId,
			.presenter = view_at( opportunityNs ),
			.multiplier = inject.multiplier,
			.allowGeneration = true,
		} );
		state = step.state;
		if ( !step.planned )
			return false;
		const bool kept = step.reestablish == BidirReestablish_t::Kept;
		if ( kept )
		{
			// Interpolations of this pair that no GPU could deliver: their wake
			// had passed when the pair was planned.
			const std::array endpoints = { BidirEndpoint_t{ prev, source[ prev ], reals[ prev ].recordNs },
				BidirEndpoint_t{ next, source[ next ], reals[ next ].recordNs } };
			const BidirPlan_t plan = plan_bidir_slots( state.grid, state.epoch, endpoints, inject.multiplier, state.gridEpoch );
			for ( const BidirSlot_t &slot : plan.slots )
			{
				if ( slot.kind == BidirSlotKind_t::Generated && slot.endpointFrameId == next && nowNs >= slot.wakeNs )
					r.lateSlotNs.push_back( opportunityNs );
			}
		}
		else
		{
			r.events.push_back( { opportunityNs, step.reestablish, state.epoch.delayNs } );
			if ( step.reestablish == BidirReestablish_t::Established && r.firstEstablishNs == 0u )
				r.firstEstablishNs = opportunityNs;
		}
		if ( step.reestablish == BidirReestablish_t::Shortened )
		{
			std::deque<BidirLoopEntry_t> retargeted;
			for ( const BidirQueueEntry_t &entry : step.retargeted )
			{
				const auto old = find_real( entry.realFrameId );
				retargeted.push_back( { entry.targetNs, true, entry.realFrameId, old != queue.end() && old->behind, entry.idealNs } );
			}
			queue = std::move( retargeted );
		}
		for ( const uint64_t id : step.droppedRealIds )
		{
			queue.erase( find_real( id ) );
			r.planDropped++;
		}
		if ( step.backlogDropped )
			r.backlogDropNs.push_back( opportunityNs );
		for ( const BidirSlot_t &slot : step.admitted )
			insert( { slot.targetNs, false, 0u } );
		const auto queue_endpoint = [&]( const BidirPlannedEndpoint_t &planned, size_t id ) {
			BidirLoopReal_t &real = reals[ id ];
			real.delayNs = state.epoch.delayNs;
			real.pushNs = planned.targetNs - state.grid.target( grid_index_at_or_after( state.grid, planned.idealNs ) );
			real.pastIdealNs = planned.targetNs - planned.idealNs;
			real.kept = kept;
			consumedId = std::max<uint64_t>( consumedId, id );
			insert( { planned.targetNs, true, id, false, planned.idealNs } );
		};
		if ( step.previous.queue )
			queue_endpoint( step.previous, prev );
		queue_endpoint( step.current, next );
		shed( step.shedCapacity );
		return true;
	};

	size_t next = 0;
	for ( uint64_t n = 1; k_ulLoopStartNs + n * T < source.back() + 300'000'000u; n++ )
	{
		uint64_t vblankNs = k_ulLoopStartNs + n * T;
		if ( vblankNs >= inject.jumpAtNs )
			vblankNs += inject.jumpNs;
		if ( within( vblankNs, inject.stallAtNs, inject.stallForNs ) )
			continue;
		const uint64_t wakeNs = vblankNs - k_ulLoopWakeLeadNs;
		const bool afterStall = inject.stallForNs != 0u && within( vblankNs, inject.stallAtNs + inject.stallForNs, T );
		// Composites of the reals committed before this vblank's wake; each
		// paint flips at this opportunity.
		size_t composites = 0;
		for ( ; next < source.size() && ready[ next ] < wakeNs; next++ )
		{
			if ( suspended( ready[ next ] ) )
			{
				previous.reset();
				continue;
			}
			// Replaced by a newer commit before the compositor latched it.
			if ( inject.latchNewest && afterStall && next + 1u < source.size() && ready[ next + 1u ] < wakeNs )
				continue;
			if ( inject.compositesPerVblank != 0u && composites >= inject.compositesPerVblank )
				break;
			composites++;
			r.recorded++;
			reals[ next ] = { .id = next, .sourceNs = source[ next ], .recordNs = std::max( ready[ next ], wakeNs ) };
			shed( r.capacity );
			const std::optional<size_t> prev = std::exchange( previous, next );
			if ( !prev || !plan_pair( *prev, next, vblankNs, std::max( ready[ next ], wakeNs ) ) )
			{
				consumedId = std::max<uint64_t>( consumedId, next );
				if ( queue.empty() )
					show_real( next, vblankNs );	// flipped live
				else if ( queue.back().behind
					|| static_cast<size_t>( std::ranges::count_if( queue, &BidirLoopEntry_t::real ) ) >= r.capacity )
				{
					r.replacedDropped++;
					queue.back() = { queue.back().targetNs, true, next, true };
				}
				else
				{
					shed( r.capacity - 1u );
					queue.push_back( { bidir_target_after( vblankNs, queue.back().targetNs, T ), true, next, true } );
				}
			}
			present( vblankNs );
			r.maxDepth = std::max( r.maxDepth, queue.size() );
		}
		if ( !inject.presentsOnlyAtComposites && !( inject.lostPresent && inject.lostPresent( vblankNs ) ) )
			present( vblankNs );
		r.maxDepth = std::max( r.maxDepth, queue.size() );
		if ( !suspended( vblankNs ) )
		{
			r.servicedNs.push_back( vblankNs );
			if ( lastPresentNs != vblankNs )
				r.holeNs.push_back( vblankNs );
		}
	}
	r.queuedAtEnd = static_cast<size_t>( std::ranges::count_if( queue, &BidirLoopEntry_t::real ) );
	r.finalDelayNs = state.epoch.delayNs;
	r.finalEnvelopeNs = state.pairLatency.valueNs;
	return r;
}

// Deterministic integer LCG for cadences, latencies and lost presents.
struct LoopRng_t
{
	uint32_t state = 0x2545f491u;
	uint32_t next() { state = state * 1664525u + 1013904223u; return state >> 8; }
	uint64_t uniform( uint64_t lo, uint64_t hi ) { return lo + next() % ( hi - lo + 1u ); }
};

// A repeating pattern of source intervals for durationNs.
std::vector<uint64_t> pattern_cadence( std::vector<uint64_t> pattern, uint64_t durationNs )
{
	std::vector<uint64_t> intervals;
	for ( uint64_t totalNs = 0; totalNs < durationNs; totalNs += intervals.back() )
		intervals.push_back( pattern[ intervals.size() % pattern.size() ] );
	return intervals;
}

// 42-50 fps with +-3 ms of jitter.
std::vector<uint64_t> jittered_cadence( uint64_t durationNs )
{
	LoopRng_t rng;
	std::vector<uint64_t> intervals;
	for ( uint64_t totalNs = 0; totalNs < durationNs; totalNs += intervals.back() )
		intervals.push_back( rng.uniform( 20'000'000u, 23'809'524u ) + rng.uniform( 0u, 6'000'000u ) - 3'000'000u );
	return intervals;
}

struct LoopCadence_t
{
	const char *name;
	std::vector<uint64_t> intervals;
	bool everyIntervalAtLeastAVblank;
};

std::vector<LoopCadence_t> loop_cadences( uint64_t durationNs )
{
	return {
		{ "30 fps", pattern_cadence( { 33'333'333u }, durationNs ), true },
		{ "42 fps", pattern_cadence( { 23'809'524u }, durationNs ), true },
		{ "42-50 fps +-3 ms", jittered_cadence( durationNs ), true },
		{ "13/13/30 ms", pattern_cadence( { 13'000'000u, 13'000'000u, 30'000'000u }, durationNs ), false },
		{ "incommensurate", pattern_cadence( { 21'400'000u, 15'200'000u, 31'100'000u, 13'000'000u, 29'300'000u }, durationNs ), false },
		{ "18 ms", pattern_cadence( { 18'000'000u }, durationNs ), true },
	};
}

std::function<uint64_t( size_t, uint64_t )> constant_latency( uint64_t extraNs )
{
	return [=]( size_t, uint64_t ) { return extraNs; };
}

// Every recorded real is shown once, in order, or dropped by a counted path;
// the queue never exceeds the ring-derived ceiling.
void check_loop( const BidirLoopResult_t &r )
{
	CHECK( std::ranges::adjacent_find( r.shown, std::greater_equal<>(), &BidirLoopReal_t::id ) == r.shown.end() );
	CHECK( r.shown.size() + r.dropped() + r.queuedAtEnd == r.recorded );
	CHECK( r.maxDepth <= r.capacity );
	CHECK( r.doublePresents == 0u );
}

// Reals recorded from fromNs on, queued by a plan, and shown later than D + T
// after their record, plus slackNs, plus their displacement behind an older
// real when withPush.
size_t reals_beyond_bound( const BidirLoopResult_t &r, uint64_t fromNs, uint64_t slackNs = 0u, bool withPush = true )
{
	return static_cast<size_t>( std::ranges::count_if( r.shown, [&]( const BidirLoopReal_t &real ) {
		return real.delayNs != 0u && real.recordNs >= fromNs
			&& real.flipNs - real.recordNs > real.delayNs + k_ulLoopVblankNs + slackNs + ( withPush ? real.pushNs : 0u );
	} ) );
}

size_t displaced_reals( const BidirLoopResult_t &r, uint64_t fromNs )
{
	return static_cast<size_t>( std::ranges::count_if( r.shown, [&]( const BidirLoopReal_t &real ) {
		return real.kept && real.recordNs >= fromNs && real.pushNs != 0u;
	} ) );
}

} // namespace

namespace
{

// Holes between consecutive shown reals recorded 2..multiplier vblanks apart,
// from fromNs on to 300 ms before the last real: pairs with a slot the
// multiplier can fill.
size_t fillable_holes_from( const BidirLoopResult_t &r, uint64_t fromNs, uint32_t multiplier = 2u )
{
	constexpr uint64_t T = k_ulLoopVblankNs;
	size_t holes = 0;
	const uint64_t endNs = r.shown.empty() ? 0u : r.shown.back().flipNs - 300'000'000u;
	for ( size_t i = 1u; i < r.shown.size(); i++ )
	{
		const BidirLoopReal_t &a = r.shown[ i - 1u ];
		const BidirLoopReal_t &b = r.shown[ i ];
		const uint64_t gapNs = b.recordNs - a.recordNs;
		if ( gapNs + 1'000u < 2u * T || gapNs > multiplier * T + 1'000u || a.flipNs < fromNs || b.flipNs > endNs )
			continue;
		holes += BidirLoopResult_t::count_in( r.holeNs, a.flipNs + 1u, b.flipNs );
	}
	return holes;
}

// Consecutive reals shown from fromNs on to 300 ms before the last real whose
// present spacing departs from their source spacing by more than a vblank.
// A pair across a dropped real is no pair.
size_t spacing_errors( const BidirLoopResult_t &r, uint64_t fromNs )
{
	constexpr int64_t T = k_ulLoopVblankNs;
	size_t errors = 0;
	const uint64_t endNs = r.shown.empty() ? 0u : r.shown.back().flipNs - 300'000'000u;
	for ( size_t i = 1u; i < r.shown.size(); i++ )
	{
		const BidirLoopReal_t &a = r.shown[ i - 1u ];
		const BidirLoopReal_t &b = r.shown[ i ];
		if ( a.flipNs < fromNs || b.flipNs >= endNs || b.id != a.id + 1u )
			continue;
		const int64_t errorNs = static_cast<int64_t>( b.flipNs - a.flipNs ) - static_cast<int64_t>( b.sourceNs - a.sourceNs );
		errors += errorNs > T || errorNs < -T;
	}
	return errors;
}

// A latency alternating between a low and a high band every 0.1-0.9 s.
std::vector<uint64_t> bimodal_latency( const std::vector<uint64_t> &intervals, uint64_t lowNs, uint64_t lowSpreadNs,
	uint64_t highNs, uint64_t highSpreadNs )
{
	LoopRng_t rng;
	std::vector<uint64_t> latency;
	bool high = false;
	uint64_t untilNs = 0;
	uint64_t sourceNs = k_ulLoopStartNs;
	for ( size_t i = 0; i <= intervals.size(); i++ )
	{
		if ( sourceNs >= untilNs )
		{
			high = !high;
			untilNs = sourceNs + rng.uniform( 100'000'000u, 900'000'000u );
		}
		latency.push_back( high ? rng.uniform( highNs, highNs + highSpreadNs ) : rng.uniform( lowNs, lowNs + lowSpreadNs ) );
		if ( i < intervals.size() )
			sourceNs += intervals[ i ];
	}
	return latency;
}

} // namespace

TEST_CASE( "bidir delay settles and holds for 10k frames", "[framegen][deadline]" )
{
	// Every cadence establishes once, after the first window, on a D that covers
	// its pairs, periodic long intervals included. D never changes, every
	// interpolation slot is planned before its wake and shown, no real is
	// dropped, and every real is shown within D + T of its record (plus its
	// displacement behind an older real, which only sources faster than the
	// display have).
	//
	// A pre-record hold jittered by 10 ms moves the records by whole vblanks, so
	// the pair latency is a random mix of one to three vblank gaps. The k-th
	// highest of it can settle once more within the first seconds (a density
	// rise, or the release of a first value whose D covered more than it
	// needed); after that D holds for the rest of the run, every pair with a
	// slot to fill shows one, and the delay of the reals does not drift: its
	// worst case in the first and the second half agree within a vblank.
	constexpr uint64_t T = k_ulLoopVblankNs;
	for ( const LoopCadence_t &cadence : loop_cadences( 10'000u * 25'000'000u ) )
	{
		for ( const bool jittered : { false, true } )
		{
			CAPTURE( cadence.name, jittered );
			std::vector<uint64_t> intervals = cadence.intervals;
			intervals.resize( 10'000u, intervals.back() );
			auto rng = std::make_shared<LoopRng_t>();
			const BidirLoopResult_t r = run_bidir_loop( intervals, { .latency = [=]( size_t, uint64_t ) {
				return jittered ? rng->uniform( 10'000'000u, 20'000'000u ) : 0u; } } );
			check_loop( r );
			CHECK( r.dropped() == 0u );
			CHECK( r.count( BidirReestablish_t::Established ) == 1u );
			CHECK( r.delay_changes() <= ( jittered ? 2u : 0u ) );
			CHECK( r.delay_changes( k_ulLoopStartNs + 20u * k_ulLoopSecondNs ) == 0u );
			CHECK( r.firstEstablishNs >= k_ulLoopStartNs + k_ulBidirWindowNs );
			const uint64_t settledNs = r.last_event_ns() + k_ulLoopSecondNs;
			CHECK( r.late_slots_from( settledNs ) == 0u );
			CHECK( fillable_holes_from( r, settledNs ) == 0u );
			CHECK( reals_beyond_bound( r, r.firstEstablishNs ) == 0u );
			if ( !jittered )
			{
				CHECK( r.holes_from( settledNs ) == 0u );
				if ( cadence.everyIntervalAtLeastAVblank )
					CHECK( displaced_reals( r, settledNs ) == 0u );
			}
			CHECK( r.finalDelayNs == bidir_target_delay( r.finalEnvelopeNs, T ) );

			const uint64_t middleNs = settledNs + ( r.shown.back().flipNs - settledNs ) / 2u;
			uint64_t worstNs[ 2 ] = {};
			for ( const BidirLoopReal_t &real : r.shown )
			{
				if ( real.flipNs >= settledNs )
					worstNs[ real.flipNs >= middleNs ] = std::max( worstNs[ real.flipNs >= middleNs ], real.flipNs - real.recordNs );
			}
			CHECK( std::max( worstNs[ 0 ], worstNs[ 1 ] ) - std::min( worstNs[ 0 ], worstNs[ 1 ] ) < T );
		}
	}
}

TEST_CASE( "a periodic long bidir interval is covered within a second", "[framegen][deadline]" )
{
	// At x4 a long interval recurring one pair in three (or five) from the start
	// is part of the first window, so the first D covers it: no
	// re-establishment, and every interpolation slot of the long pair is planned
	// before its wake.
	//
	// One that only starts recurring later puts k samples above D within a
	// window: D lengthens once, to within a vblank of the from-start D, and a
	// second after the onset every slot is planned before its wake. A source
	// slower than the display loses no real, a faster one only the reals the
	// display cannot show in time.
	constexpr uint64_t T = k_ulLoopVblankNs;
	struct Pattern_t { const char *name; std::vector<uint64_t> intervals; bool everyIntervalAtLeastAVblank; };
	const std::array patterns = {
		Pattern_t{ "18/18/70 ms", { 18'000'000u, 18'000'000u, 70'000'000u }, true },
		Pattern_t{ "12/12/45 ms", { 12'000'000u, 12'000'000u, 45'000'000u }, false },
		Pattern_t{ "20x4/75 ms", { 20'000'000u, 20'000'000u, 20'000'000u, 20'000'000u, 75'000'000u }, true },
	};
	for ( const Pattern_t &pattern : patterns )
	{
		for ( const uint64_t latencyNs : { 0ull, 3'000'000ull, 7'000'000ull, 11'000'000ull, 15'000'000ull } )
		{
			CAPTURE( pattern.name, latencyNs );
			const BidirLoopInjection_t inject = { .latency = constant_latency( latencyNs ), .ringImages = 12u, .multiplier = 4u };
			const BidirLoopResult_t r = run_bidir_loop( pattern_cadence( pattern.intervals, 10u * k_ulLoopSecondNs ), inject );
			check_loop( r );
			CHECK( r.dropped() == 0u );
			CHECK( r.count( BidirReestablish_t::Established ) == 1u );
			CHECK( r.delay_changes() == 0u );
			CHECK( r.late_slots_from( r.firstEstablishNs + k_ulLoopSecondNs ) == 0u );

			std::vector<uint64_t> onset = pattern_cadence( { pattern.intervals.front() }, k_ulLoopInjectAtNs - k_ulLoopStartNs );
			const std::vector<uint64_t> recurring = pattern_cadence( pattern.intervals, 10u * k_ulLoopSecondNs );
			onset.insert( onset.end(), recurring.begin(), recurring.end() );
			const BidirLoopResult_t late = run_bidir_loop( onset, inject );
			check_loop( late );
			CHECK( late.count( BidirReestablish_t::Established ) == 1u );
			CHECK( late.count( BidirReestablish_t::Lengthened ) <= 1u );
			CHECK( late.count( BidirReestablish_t::Shortened ) == 0u );
			CHECK( std::max( late.finalDelayNs, r.finalDelayNs ) - std::min( late.finalDelayNs, r.finalDelayNs ) <= T );
			CHECK( late.late_slots_from( k_ulLoopInjectAtNs + k_ulLoopSecondNs ) == 0u );
			CHECK( late.backlogDropNs.empty() );
			CHECK( late.dropped() == late.planDropped );
			if ( pattern.everyIntervalAtLeastAVblank )
				CHECK( late.dropped() == 0u );
		}
	}
}

TEST_CASE( "a bidir pre-record hold step is no delay transition", "[framegen][deadline]" )
{
	// The loop's latency is a hold between a real's source time and its record
	// (a frame limiter, a slower submission). Record time follows it, so a step
	// of it makes one long or one short pair, never a D transition: steps of
	// 0..3T in 2 ms steps on every cadence, and of 0..100 ms at 30 and 42 fps,
	// up and down, keep the one D. A source no faster than the display drops no
	// real; a faster one only what the display cannot show in time. A second
	// after the step every slot is planned in time and every opportunity filled.
	constexpr uint64_t T = k_ulLoopVblankNs;
	constexpr uint64_t at = k_ulLoopInjectAtNs;
	const uint64_t durationNs = 6u * k_ulLoopSecondNs + k_ulBidirReleaseHoldNs;
	struct Sweep_t { LoopCadence_t cadence; uint64_t maxStepNs; };
	std::vector<Sweep_t> sweeps;
	for ( const LoopCadence_t &cadence : loop_cadences( durationNs ) )
		sweeps.push_back( { cadence, 3u * T } );
	sweeps.push_back( { { "30 fps", pattern_cadence( { 33'333'333u }, durationNs ), true }, 100'000'000u } );
	sweeps.push_back( { { "42 fps", pattern_cadence( { 23'809'524u }, durationNs ), true }, 100'000'000u } );
	for ( const Sweep_t &sweep : sweeps )
	{
		for ( uint64_t stepNs = 0u; stepNs <= sweep.maxStepNs; stepNs += 2'000'000u )
		{
			for ( const bool up : { true, false } )
			{
				CAPTURE( sweep.cadence.name, stepNs, up );
				const BidirLoopResult_t r = run_bidir_loop( sweep.cadence.intervals, { .latency =
					[=]( size_t, uint64_t sourceNs ) { return ( sourceNs >= at ) == up ? stepNs : 0u; } } );
				check_loop( r );
				CHECK( r.events.size() == 1u );
				CHECK( r.dropped() == r.planDropped );
				if ( sweep.cadence.everyIntervalAtLeastAVblank )
					CHECK( r.dropped() == 0u );
				CHECK( r.late_slots_from( at + k_ulLoopSecondNs ) == 0u );
				CHECK( r.holes_from( at + k_ulLoopSecondNs ) == 0u );
				CHECK( reals_beyond_bound( r, r.firstEstablishNs ) == 0u );
			}
		}
	}
}

TEST_CASE( "bidir latency pulses never cost a real or a slot", "[framegen][deadline]" )
{
	// Pre-record holds of +20..+35 ms for 0.2-0.6 s every 1-3 s, and a fixed
	// +35 ms pulse of 0.2 s every 3 s, on every cadence: no real is dropped, and
	// a second after the first pulse (or a lengthening) every slot is planned in
	// time and every pair with a slot to fill shows one.
	constexpr uint64_t at = k_ulLoopInjectAtNs;
	constexpr uint64_t trainEndNs = k_ulLoopStartNs + 20u * k_ulLoopSecondNs;
	struct Pulse_t { uint64_t fromNs, toNs, extraNs; };
	std::vector<Pulse_t> seeded;
	LoopRng_t rng;
	for ( uint64_t t = at; t < trainEndNs; t += rng.uniform( k_ulLoopSecondNs, 3u * k_ulLoopSecondNs ) )
	{
		const uint64_t forNs = rng.uniform( 200'000'000u, 600'000'000u );
		seeded.push_back( { t, t + forNs, rng.uniform( 20'000'000u, 35'000'000u ) } );
	}
	std::vector<Pulse_t> fixed;
	for ( uint64_t t = at; t < trainEndNs; t += 3u * k_ulLoopSecondNs )
		fixed.push_back( { t, t + 200'000'000u, 35'000'000u } );

	for ( const LoopCadence_t &cadence : loop_cadences( 30u * k_ulLoopSecondNs ) )
	{
		for ( const std::vector<Pulse_t> *pulses : { &seeded, &fixed } )
		{
			CAPTURE( cadence.name, pulses == &fixed );
			const BidirLoopResult_t r = run_bidir_loop( cadence.intervals, { .latency =
				[pulses]( size_t, uint64_t sourceNs ) {
					for ( const Pulse_t &pulse : *pulses )
					{
						if ( sourceNs >= pulse.fromNs && sourceNs < pulse.toNs )
							return pulse.extraNs;
					}
					return uint64_t{ 0 };
				} } );
			check_loop( r );
			CHECK( r.dropped() == 0u );
			CHECK( r.delay_changes() <= 1u );
			const uint64_t settledNs = std::max( at, r.last_ns( BidirReestablish_t::Lengthened ) ) + k_ulLoopSecondNs;
			CHECK( r.late_slots_from( settledNs ) == 0u );
			CHECK( fillable_holes_from( r, settledNs ) == 0u );
		}
	}
}

TEST_CASE( "a lone high or low bidir latency never moves the delay", "[framegen][deadline]" )
{
	// One real 60 ms later, or 25 ms earlier, than the rest (25 ms), on the third
	// real or on the first real after a scene cut: D never changes, and a cut
	// keeps it, since the envelope survives the translation.
	for ( const LoopCadence_t &cadence : loop_cadences( 20u * k_ulLoopSecondNs ) )
	{
		size_t firstAfterCut = 0;
		for ( uint64_t sourceNs = k_ulLoopStartNs + k_ulLoopVblankNs - 5'000'000u; sourceNs < k_ulLoopInjectAtNs; )
			sourceNs += cadence.intervals[ firstAfterCut++ ];
		for ( const bool atCut : { false, true } )
		{
			for ( const uint64_t loneNs : { 85'000'000ull, 0ull } )
			{
				CAPTURE( cadence.name, atCut, loneNs );
				const size_t loneId = atCut ? firstAfterCut : 2u;
				const BidirLoopResult_t r = run_bidir_loop( cadence.intervals, {
					.latency = [=]( size_t id, uint64_t ) { return id == loneId ? loneNs : 25'000'000u; },
					.cutAtNs = atCut ? k_ulLoopInjectAtNs : 0u } );
				check_loop( r );
				CHECK( r.dropped() == 0u );
				CHECK( r.delay_changes() == 0u );
				CHECK( r.count( BidirReestablish_t::Established ) == ( atCut ? 2u : 1u ) );
				CHECK( std::ranges::all_of( r.events, [&]( const BidirLoopEvent_t &event ) {
					return event.delayNs == r.events.front().delayNs; } ) );
			}
		}
	}
	// One 85 ms hitch of an 18 ms source.
	{
		std::vector<uint64_t> intervals = pattern_cadence( { 18'000'000u }, 20u * k_ulLoopSecondNs );
		intervals[ 400u ] = 85'000'000u;
		const BidirLoopResult_t r = run_bidir_loop( intervals );
		check_loop( r );
		CHECK( r.dropped() == 0u );
		CHECK( r.delay_changes() == 0u );
		CHECK( r.events.size() == 1u );
	}
}

TEST_CASE( "a noisy bimodal bidir latency keeps one delay", "[framegen][deadline]" )
{
	// A pre-record hold alternating between 10-25 ms and 45-65 ms every
	// 0.1-0.9 s, on every cadence at the default ring: one D, established once,
	// no real dropped. At x4 on a ring of 12 no real is dropped either. A 10-25 /
	// 30-40 ms hold at 30 fps also fills every pair with a slot to fill.
	for ( const LoopCadence_t &cadence : loop_cadences( 25u * k_ulLoopSecondNs ) )
	{
		CAPTURE( cadence.name );
		const std::vector<uint64_t> latency = bimodal_latency( cadence.intervals, 10'000'000u, 15'000'000u, 45'000'000u, 20'000'000u );
		const auto latencyOf = [&]( size_t id, uint64_t ) { return latency[ id ]; };
		const BidirLoopResult_t r = run_bidir_loop( cadence.intervals, { .latency = latencyOf } );
		check_loop( r );
		CHECK( r.dropped() == 0u );
		CHECK( r.backlogDropNs.empty() );
		CHECK( r.count( BidirReestablish_t::Established ) == 1u );
		CHECK( r.delay_changes() == 0u );
		const BidirLoopResult_t x4 = run_bidir_loop( cadence.intervals, { .latency = latencyOf, .ringImages = 12u, .multiplier = 4u } );
		check_loop( x4 );
		CHECK( x4.dropped() == 0u );
		CHECK( x4.backlogDropNs.empty() );
	}
	const std::vector<uint64_t> intervals = pattern_cadence( { 33'333'333u }, 25u * k_ulLoopSecondNs );
	const std::vector<uint64_t> latency = bimodal_latency( intervals, 10'000'000u, 15'000'000u, 30'000'000u, 10'000'000u );
	const BidirLoopResult_t r = run_bidir_loop( intervals, { .latency = [&]( size_t id, uint64_t ) { return latency[ id ]; } } );
	check_loop( r );
	CHECK( r.dropped() == 0u );
	CHECK( r.delay_changes() == 0u );
	CHECK( fillable_holes_from( r, r.firstEstablishNs + k_ulLoopSecondNs ) == 0u );
}

TEST_CASE( "a compositor stall abandons the bidir timeline once and loses only it", "[framegen][deadline]" )
{
	// A 200 ms stall: the queued timeline goes stale and is dropped once,
	// counted, and never shown afterwards. Latching only the newest commit
	// loses nothing else; two composites per vblank burst the queue, whose reals
	// beyond what the display can show in time go one by one. D never changes:
	// catch-up composites are no latency samples.
	constexpr uint64_t at = k_ulLoopInjectAtNs;
	for ( const LoopCadence_t &cadence : loop_cadences( 20u * k_ulLoopSecondNs ) )
	{
		for ( const bool latchNewest : { true, false } )
		{
			CAPTURE( cadence.name, latchNewest );
			const BidirLoopResult_t r = run_bidir_loop( cadence.intervals, { .stallAtNs = at + 5'000'000u,
				.stallForNs = 200'000'000u, .compositesPerVblank = latchNewest ? 0u : 2u, .latchNewest = latchNewest } );
			check_loop( r );
			CHECK( r.staleAbandons == 1u );
			CHECK( r.staleDropped != 0u );
			CHECK( r.dropped() == r.staleDropped + r.planDropped );
			if ( latchNewest )
				CHECK( r.planDropped == 0u );
			CHECK( r.backlogDropNs.empty() );
			CHECK( r.count( BidirReestablish_t::Established ) == 2u );
			CHECK( r.delay_changes() == 0u );
		}
	}
}

TEST_CASE( "a bidir real dropped anywhere is never shown again", "[framegen][deadline]" )
{
	// A real enters the timeline once: once queued it is consumed, so a
	// re-establishment never queues it again, whichever path dropped it.
	constexpr uint64_t T = k_ulLoopVblankNs;
	// A real whose present image fails just before a source gap, then a scene
	// cut: the next plan re-establishes on the pair the dropped real begins.
	{
		std::vector<uint64_t> intervals = pattern_cadence( { 33'333'333u }, 10u * k_ulLoopSecondNs );
		constexpr size_t failedId = 150u;
		intervals[ failedId ] = 200'000'000u;
		uint64_t cutAtNs = k_ulLoopStartNs + T - 5'000'000u;
		for ( size_t i = 0; i <= failedId; i++ )
			cutAtNs += intervals[ i ];
		const BidirLoopResult_t r = run_bidir_loop( intervals, { .cutAtNs = cutAtNs, .failedImageRealId = failedId } );
		check_loop( r );
		CHECK( r.failedDropped == 1u );
		CHECK( r.dropped() == 1u );
		CHECK( r.count( BidirReestablish_t::Established ) == 2u );
		CHECK( std::ranges::none_of( r.shown, [&]( const BidirLoopReal_t &real ) { return real.id == failedId; } ) );
	}
	// A suspended presentation: the stale timeline is dropped, the next pair
	// re-establishes, and its previous real, dropped with the timeline, stays
	// dropped.
	{
		constexpr uint64_t suspendAtNs = k_ulLoopInjectAtNs;
		const BidirLoopResult_t r = run_bidir_loop( pattern_cadence( { 23'809'524u }, 10u * k_ulLoopSecondNs ),
			{ .suspendAtNs = suspendAtNs, .suspendForNs = 200'000'000u } );
		check_loop( r );
		CHECK( r.staleAbandons == 1u );
		CHECK( r.staleDropped != 0u );
		CHECK( r.dropped() == r.staleDropped );
		CHECK( r.count( BidirReestablish_t::Established ) == 2u );
		CHECK( r.delay_changes() == 0u );
		for ( const BidirLoopReal_t &real : r.shown )
			CHECK( ( real.flipNs < suspendAtNs || real.flipNs >= suspendAtNs + 200'000'000u ) );
	}
	// A source faster than the display: the plan drops the reals the display
	// cannot show in time.
	{
		const BidirLoopResult_t r = run_bidir_loop( pattern_cadence( { 16'000'000u }, 10u * k_ulLoopSecondNs ) );
		check_loop( r );
		CHECK( r.planDropped != 0u );
		CHECK( r.dropped() == r.planDropped );
	}
}

TEST_CASE( "bidir presents only at composites keep every real of a steady source", "[framegen][deadline]" )
{
	// Without vblank presents a real can be shown only at a composite, up to a
	// composite interval after its slot: that is lateness, not a backlog, and
	// never drops a real of a steady source, with or without lost presents.
	constexpr uint64_t T = k_ulLoopVblankNs;
	for ( const uint64_t intervalNs : { 33'333'333ull, 23'809'524ull, 18'000'000ull } )
	{
		for ( const uint32_t lostPercent : { 0u, 10u } )
		{
			CAPTURE( intervalNs, lostPercent );
			auto rng = std::make_shared<LoopRng_t>();
			const BidirLoopResult_t r = run_bidir_loop( pattern_cadence( { intervalNs }, 20u * k_ulLoopSecondNs ), {
				.latency = constant_latency( 20'000'000u ),
				.lostPresent = [=]( uint64_t ) { return rng->next() % 100u < lostPercent; },
				.presentsOnlyAtComposites = true } );
			check_loop( r );
			CHECK( r.dropped() == 0u );
			CHECK( r.count( BidirReestablish_t::Established ) == 1u );
			CHECK( r.events.size() == 1u );
			CHECK( reals_beyond_bound( r, r.firstEstablishNs, round_up_to_vblanks( intervalNs, T ) ) == 0u );
		}
	}
	// Reals arriving faster than the composites that could show them are a
	// genuine backlog: the plan drops them one by one, never the whole queue.
	for ( const std::vector<uint64_t> &pattern : { std::vector<uint64_t>{ 13'000'000u, 13'000'000u, 30'000'000u },
		std::vector<uint64_t>{ 21'400'000u, 15'200'000u, 31'100'000u, 13'000'000u, 29'300'000u } } )
	{
		const BidirLoopResult_t r = run_bidir_loop( pattern_cadence( pattern, 20u * k_ulLoopSecondNs ),
			{ .latency = constant_latency( 20'000'000u ), .presentsOnlyAtComposites = true } );
		check_loop( r );
		CHECK( r.planDropped != 0u );
		CHECK( r.dropped() == r.planDropped );
		CHECK( r.delay_changes() == 0u );
	}
}

TEST_CASE( "a bidir backlog that does not drain drops only its excess", "[framegen][deadline]" )
{
	// Only queueing drops a backlog's reals: reals that wait behind reals with an
	// excess over D that keeps its range for a window. Each such drop takes the
	// reals of the excess, once per window at most; every other drop is a real
	// the display cannot show in time. D never changes for it.
	//
	// A 50 ms hold that ends records three reals at one wake. At the display's
	// rate, or barely slower than it, the excess this leaves (53 ms with the
	// wake lead) drains too slowly: one backlog drop of its four vblanks' reals,
	// never the whole queue. Slower sources drain it.
	constexpr uint64_t T = k_ulLoopVblankNs;
	struct Case_t { const char *name; std::vector<uint64_t> intervals; BidirLoopInjection_t inject; size_t maxDropped; };
	const auto dropAt5s = []( uint64_t extraNs ) {
		return [=]( size_t, uint64_t sourceNs ) { return sourceNs < k_ulLoopStartNs + 5u * k_ulLoopSecondNs ? extraNs : 0u; };
	};
	std::vector<uint64_t> fastThenSlow( 300u, 13'300'000u );
	fastThenSlow.resize( 600u, 31'250'000u );
	const std::array cases = {
		Case_t{ "16.0 ms source", pattern_cadence( { 16'000'000u }, 20u * k_ulLoopSecondNs ), {}, SIZE_MAX },
		Case_t{ "16.67 ms source, 50 ms hold drop", pattern_cadence( { 16'666'667u }, 20u * k_ulLoopSecondNs ),
			{ .latency = dropAt5s( 50'000'000u ) }, 4u },
		Case_t{ "17 ms source, 50 ms hold drop", pattern_cadence( { 17'000'000u }, 20u * k_ulLoopSecondNs ),
			{ .latency = dropAt5s( 50'000'000u ) }, 4u },
		Case_t{ "18 ms source, 50 ms hold drop", pattern_cadence( { 18'000'000u }, 20u * k_ulLoopSecondNs ),
			{ .latency = dropAt5s( 50'000'000u ) }, 0u },
		Case_t{ "75 fps, then 32 fps", fastThenSlow, {}, SIZE_MAX },
	};
	for ( const Case_t &c : cases )
	{
		CAPTURE( c.name );
		const BidirLoopResult_t r = run_bidir_loop( c.intervals, c.inject );
		check_loop( r );
		CHECK( r.dropped() == r.planDropped );
		CHECK( r.dropped() <= c.maxDropped );
		if ( c.maxDropped != SIZE_MAX )
			CHECK( r.backlogDropNs.size() <= 1u );
		CHECK( r.delay_changes() == 0u );
		for ( size_t i = 1u; i < r.backlogDropNs.size(); i++ )
			CHECK( r.backlogDropNs[ i ] - r.backlogDropNs[ i - 1u ] >= k_ulBidirWindowNs );
		CHECK( reals_beyond_bound( r, r.firstEstablishNs, k_ulBidirWindowNs + T, false ) == 0u );
	}

	// Lost presents make reals late, not queued behind each other: no drop on a
	// source no faster than the display.
	for ( const LoopCadence_t &cadence : loop_cadences( 20u * k_ulLoopSecondNs ) )
	{
		if ( !cadence.everyIntervalAtLeastAVblank )
			continue;
		CAPTURE( cadence.name );
		auto rng = std::make_shared<LoopRng_t>();
		const BidirLoopResult_t r = run_bidir_loop( cadence.intervals, { .lostPresent =
			[=]( uint64_t opportunityNs ) { return opportunityNs > k_ulLoopInjectAtNs && rng->next() % 100u < 25u; } } );
		check_loop( r );
		CHECK( r.dropped() == 0u );
		CHECK( r.backlogDropNs.empty() );
		CHECK( reals_beyond_bound( r, r.firstEstablishNs, r.capacity * T, false ) == 0u );
	}
}

TEST_CASE( "on-target bidir flips never ratchet the delay", "[framegen][deadline]" )
{
	// Every endpoint rounds up to the grid. The translation is never corrected
	// by a flip, so the rounding never accumulates: on a source no faster than
	// the display each real is queued on its own slot, within one vblank of its
	// display time on the translation.
	for ( const LoopCadence_t &cadence : loop_cadences( 20u * k_ulLoopSecondNs ) )
	{
		if ( !cadence.everyIntervalAtLeastAVblank )
			continue;
		CAPTURE( cadence.name );
		const BidirLoopResult_t r = run_bidir_loop( cadence.intervals );
		CHECK( std::ranges::all_of( r.shown, []( const BidirLoopReal_t &real ) {
			return real.delayNs == 0u || ( real.pushNs == 0u && real.pastIdealNs < k_ulLoopVblankNs ); } ) );
	}
}

TEST_CASE( "two composites per vblank never take the bidir queue past the ring ceiling", "[framegen][deadline]" )
{
	// After a compositor stall two composites land on one opportunity, and a
	// jump of the opportunities does the same. The second composite's plan
	// counts the front as leaving only if the presenter shows it, so the queue
	// never exceeds the ring ceiling. A draining burst is no backlog, D never
	// changes, and a stall of up to 100 ms keeps the timeline: latching only the
	// newest commit drops no real, a burst of every commit at most the two the
	// display cannot show in time.
	constexpr uint64_t T = k_ulLoopVblankNs;
	constexpr uint64_t at = k_ulLoopInjectAtNs;
	struct Injection_t { std::string name; BidirLoopInjection_t inject; size_t maxDropped; };
	std::vector<Injection_t> injections = {
		{ "opportunities one vblank later", { .jumpAtNs = at, .jumpNs = T }, 0u },
		{ "opportunities two vblanks later", { .jumpAtNs = at, .jumpNs = 2u * T }, 0u },
	};
	for ( const uint64_t stallMs : { 34ull, 50ull, 67ull, 84ull, 100ull } )
	{
		for ( const bool latchNewest : { false, true } )
		{
			injections.push_back( { std::to_string( stallMs ) + " ms stall" + ( latchNewest ? ", newest latched" : "" ), {
				.stallAtNs = at + 5'000'000u, .stallForNs = stallMs * 1'000'000u, .compositesPerVblank = 2u,
				.latchNewest = latchNewest }, stallMs <= 50u || latchNewest ? 0u : 2u } );
		}
	}
	for ( const LoopCadence_t &cadence : loop_cadences( 12u * k_ulLoopSecondNs ) )
	{
		for ( const Injection_t &injection : injections )
		{
			CAPTURE( cadence.name, injection.name );
			const BidirLoopResult_t r = run_bidir_loop( cadence.intervals, injection.inject );
			check_loop( r );
			CHECK( r.dropped() == r.planDropped );
			CHECK( r.dropped() <= injection.maxDropped );
			CHECK( r.backlogDropNs.empty() );
			CHECK( r.count( BidirReestablish_t::Established ) == 1u );
			CHECK( r.delay_changes() == 0u );
		}
		// A 117 ms stall outlasts the catch-up horizon: the timeline goes stale.
		const BidirLoopResult_t r = run_bidir_loop( cadence.intervals, { .stallAtNs = at + 5'000'000u,
			.stallForNs = 117'000'000u, .compositesPerVblank = 2u } );
		check_loop( r );
		CHECK( r.dropped() == r.planDropped + r.staleDropped );
		CHECK( r.delay_changes() == 0u );
	}
}

TEST_CASE( "a pre-record hold never costs bidir interpolations", "[framegen][deadline]" )
{
	// A frame limiter holds every real alike between its source-ready time and
	// its record. Record time follows the hold, so D covers the pairs however
	// long it is, and a change of it, either way, is no D transition: one
	// establishment only, every opportunity filled, every interpolation slot
	// planned before its wake, no real dropped and every real shown within D + T
	// of its record.
	constexpr uint64_t at = k_ulLoopInjectAtNs;
	for ( const uint64_t intervalNs : { 33'333'333ull, 23'809'524ull } )
	{
		for ( const uint64_t holdMs : { 0ull, 16ull, 33ull, 50ull, 55ull, 70ull, 100ull } )
		{
			CAPTURE( intervalNs, holdMs );
			const BidirLoopResult_t r = run_bidir_loop( pattern_cadence( { intervalNs }, 12u * k_ulLoopSecondNs ),
				{ .latency = constant_latency( holdMs * 1'000'000u ) } );
			check_loop( r );
			CHECK( r.dropped() == 0u );
			CHECK( r.events.size() == 1u );
			const uint64_t settledNs = r.firstEstablishNs + k_ulLoopSecondNs;
			CHECK( r.holes_from( settledNs ) == 0u );
			CHECK( r.late_slots_from( settledNs ) == 0u );
			CHECK( reals_beyond_bound( r, r.firstEstablishNs ) == 0u );
		}
		for ( const auto &[fromMs, toMs] : std::vector<std::pair<uint64_t, uint64_t>>{
			{ 55u, 0u }, { 0u, 55u }, { 20u, 0u }, { 0u, 20u }, { 70u, 0u }, { 0u, 70u } } )
		{
			CAPTURE( intervalNs, fromMs, toMs );
			const BidirLoopResult_t r = run_bidir_loop( pattern_cadence( { intervalNs }, 14u * k_ulLoopSecondNs ),
				{ .latency = [=]( size_t, uint64_t sourceNs ) { return ( sourceNs < at ? fromMs : toMs ) * 1'000'000u; } } );
			check_loop( r );
			CHECK( r.events.size() == 1u );
			CHECK( r.dropped() == 0u );
			CHECK( r.holes_from( at + k_ulLoopSecondNs ) == 0u );
			CHECK( r.late_slots_from( at + k_ulLoopSecondNs ) == 0u );
			CHECK( reals_beyond_bound( r, r.firstEstablishNs ) == 0u );
		}
	}
}

TEST_CASE( "isolated bidir outliers never re-establish", "[framegen][deadline]" )
{
	// Two isolated long pairs within a second, fewer than the k an estimate
	// takes, from long source intervals or from pre-record hold spikes, at
	// 42 fps: they lose some of their own interpolations, D stays, no real is
	// dropped, and every opportunity is filled again a second after the last of
	// them.
	constexpr std::array outlierIds = { 200u, 210u };
	const auto isOutlier = [&]( size_t id ) { return std::ranges::find( outlierIds, id ) != outlierIds.end(); };
	std::vector<uint64_t> longIntervals = pattern_cadence( { 23'809'524u }, 14u * k_ulLoopSecondNs );
	for ( const size_t id : outlierIds )
		longIntervals[ id - 1u ] = 60'000'000u;
	const std::array runs = {
		run_bidir_loop( longIntervals ),
		run_bidir_loop( pattern_cadence( { 23'809'524u }, 14u * k_ulLoopSecondNs ), { .latency =
			[&]( size_t id, uint64_t ) { return isOutlier( id ) ? 70'000'000ull : 10'000'000ull; } } ),
	};
	for ( const BidirLoopResult_t &r : runs )
	{
		check_loop( r );
		CHECK( r.dropped() == 0u );
		CHECK( r.events.size() == 1u );
		const auto last = std::ranges::find( r.shown, uint64_t{ outlierIds.back() }, &BidirLoopReal_t::id );
		REQUIRE( last != r.shown.end() );
		CHECK( r.holes_from( last->flipNs + k_ulLoopSecondNs ) == 0u );
	}
}

TEST_CASE( "a steady bidir regime yields one delay however it is reached", "[framegen][deadline]" )
{
	// The first envelope and a release use one estimator: a cadence gives the
	// same D from the start as after a release from a slower one, within the
	// vblank by which a release to K can cover more than K needs. The ring
	// bound may shorten D at once before the envelope's release does.
	constexpr uint64_t T = k_ulLoopVblankNs;
	for ( const LoopCadence_t &cadence : loop_cadences( 16u * k_ulLoopSecondNs ) )
	{
		CAPTURE( cadence.name );
		const BidirLoopResult_t seeded = run_bidir_loop( cadence.intervals );
		std::vector<uint64_t> slowFirst = pattern_cadence( { 60'000'000u }, k_ulLoopInjectAtNs - k_ulLoopStartNs );
		slowFirst.insert( slowFirst.end(), cadence.intervals.begin(), cadence.intervals.end() );
		const BidirLoopResult_t released = run_bidir_loop( slowFirst );
		check_loop( released );
		CHECK( released.dropped() == 0u );
		CHECK( released.count( BidirReestablish_t::Shortened ) >= 1u );
		CHECK( released.count( BidirReestablish_t::Shortened ) <= 2u );
		CHECK( released.count( BidirReestablish_t::Lengthened ) == 0u );
		CHECK( std::max( seeded.finalDelayNs, released.finalDelayNs )
			- std::min( seeded.finalDelayNs, released.finalDelayNs ) <= T );
	}
}

TEST_CASE( "a slow bidir cadence turning fast keeps its reals and their spacing", "[framegen][deadline]" )
{
	// A loading-screen cadence of 60-240 ms, then 16.7-33.3 ms. D never
	// lengthens. It shortens at once where the ring bound falls below it, and
	// after the release hold where the envelope falls further; the queued reals
	// move with it, and a second after the switch consecutive reals are shown at
	// their source spacing within a vblank. No real is dropped, except at the
	// display's own rate (16.7 ms), where the excess left over from the long D
	// cannot drain and one backlog drop takes at most two reals.
	constexpr uint64_t T = k_ulLoopVblankNs;
	for ( const uint64_t slowNs : { 60'000'000ull, 100'000'000ull, 150'000'000ull, 240'000'000ull } )
	{
		for ( const uint64_t fastNs : { 16'700'000ull, 20'000'000ull, 33'333'333ull } )
		{
			CAPTURE( slowNs, fastNs );
			std::vector<uint64_t> intervals = pattern_cadence( { slowNs }, 6u * k_ulLoopSecondNs );
			const size_t switchId = intervals.size();
			const std::vector<uint64_t> fast = pattern_cadence( { fastNs }, 10u * k_ulLoopSecondNs );
			intervals.insert( intervals.end(), fast.begin(), fast.end() );
			const BidirLoopResult_t r = run_bidir_loop( intervals );
			check_loop( r );
			CHECK( r.count( BidirReestablish_t::Lengthened ) == 0u );
			CHECK( r.count( BidirReestablish_t::Shortened ) >= 1u );
			CHECK( r.count( BidirReestablish_t::Shortened ) <= 2u );
			const auto switched = std::ranges::find( r.shown, uint64_t{ switchId }, &BidirLoopReal_t::id );
			REQUIRE( switched != r.shown.end() );
			CHECK( r.dropped() == r.planDropped );
			if ( fastNs > T + 1'000'000u )
				CHECK( r.dropped() == 0u );
			else
			{
				CHECK( r.backlogDropNs.size() <= 1u );
				CHECK( r.dropped() <= 2u );
			}
			CHECK( spacing_errors( r, switched->flipNs + k_ulLoopSecondNs ) == 0u );
		}
	}
}

TEST_CASE( "a bursty bidir source never flaps the ring bound", "[framegen][deadline]" )
{
	// 20 and 30 fps with three 13 ms frames every 1.2, 1.5 or 2.5 s: a burst is
	// counted once in the span of the reals the ring holds, and the bound only
	// rises after a hold, so D changes at most once and no real is dropped.
	for ( const uint64_t baseNs : { 50'000'000ull, 33'333'333ull } )
	{
		for ( const uint64_t everyNs : { 1'200'000'000ull, 1'500'000'000ull, 2'500'000'000ull } )
		{
			CAPTURE( baseNs, everyNs );
			std::vector<uint64_t> intervals;
			uint64_t totalNs = 0;
			uint64_t nextBurstNs = everyNs;
			while ( totalNs < 20u * k_ulLoopSecondNs )
			{
				if ( totalNs >= nextBurstNs )
				{
					intervals.insert( intervals.end(), { 13'000'000u, 13'000'000u, 13'000'000u } );
					totalNs += 39'000'000u;
					nextBurstNs += everyNs;
				}
				intervals.push_back( baseNs );
				totalNs += baseNs;
			}
			const BidirLoopResult_t r = run_bidir_loop( intervals );
			check_loop( r );
			CHECK( r.delay_changes() <= 1u );
			CHECK( r.dropped() == 0u );
		}
	}
}

TEST_CASE( "a growing pre-record hold keeps the bidir delay", "[framegen][deadline]" )
{
	// A frame limiter keeps committing at 30 fps while its hold grows from 0 to
	// 80 ms over six frames (a new client settling in), so those frames turn
	// source-ready 20 ms apart. The reals the ring holds are spaced by their
	// records, which stay 33.3 ms apart: D never changes and no real is dropped.
	constexpr uint64_t periodNs = 33'333'333u;
	constexpr uint64_t stepNs = 13'333'333u;
	constexpr size_t rampId = 200u;
	const auto holdOf = [=]( size_t id ) {
		return id <= rampId ? 0u : std::min<uint64_t>( id - rampId, 6u ) * stepNs;
	};
	std::vector<uint64_t> intervals = pattern_cadence( { periodNs }, 14u * k_ulLoopSecondNs );
	for ( size_t id = 1u; id <= intervals.size(); id++ )
		intervals[ id - 1u ] -= holdOf( id ) - holdOf( id - 1u );
	const BidirLoopResult_t r = run_bidir_loop( intervals, { .latency = [=]( size_t id, uint64_t ) { return holdOf( id ); } } );
	check_loop( r );
	CHECK( r.events.size() == 1u );
	CHECK( r.delay_changes() == 0u );
	CHECK( r.dropped() == 0u );
}

TEST_CASE( "a bidir source faster than the display is shown within its bound", "[framegen][deadline]" )
{
	// The display shows at most D/T + k reals within D + kT: a source faster
	// than it drops the reals beyond that, one by one and never by a backlog,
	// and every real shown is shown within D + kT of its record (taken at the
	// compositor's wake, a wake lead before the vblank).
	constexpr uint64_t T = k_ulLoopVblankNs;
	struct Case_t { uint64_t intervalNs; size_t ringImages; uint32_t multiplier; };
	for ( const Case_t &c : { Case_t{ 16'000'000u, 8u, 2u }, Case_t{ 13'300'000u, 8u, 2u }, Case_t{ 12'000'000u, 12u, 4u } } )
	{
		CAPTURE( c.intervalNs, c.ringImages, c.multiplier );
		const BidirLoopResult_t r = run_bidir_loop( pattern_cadence( { c.intervalNs }, 10u * k_ulLoopSecondNs ),
			{ .ringImages = c.ringImages, .multiplier = c.multiplier } );
		check_loop( r );
		CHECK( r.planDropped != 0u );
		CHECK( r.dropped() == r.planDropped );
		CHECK( r.backlogDropNs.empty() );
		CHECK( reals_beyond_bound( r, r.firstEstablishNs, ( k_uBidirEnvelopeSamples - 1u ) * T + k_ulLoopWakeLeadNs, false ) == 0u );
	}
}

TEST_CASE( "the bidir envelope moves only on confirmed evidence", "[framegen][deadline]" )
{
	constexpr uint64_t T = 100u;
	constexpr uint64_t W = k_ulBidirWindowNs;
	constexpr uint64_t H = k_ulBidirReleaseHoldNs;
	struct Feed_t
	{
		BidirWindow_t window;
		BidirEnvelope_t env;
		Feed_t &operator()( uint64_t latencyNs, uint64_t atNs )
		{
			window.push( { .atNs = atNs, .latencyNs = latencyNs } );
			env = update_bidir_envelope( env, window.kth_highest_latency(), atNs, T );
			return *this;
		}
		uint64_t delay( uint64_t atNs ) const { return bidir_envelope_delay( env, window, atNs, T ); }
	};

	// The first value is the third highest sample, from the third sample on (a
	// replay of the rise rule); D follows it once the samples span a window.
	{
		Feed_t f;
		f( 500u, 1u )( 9'000u, 2u );
		CHECK( f.env.valueNs == 0u );
		f( 600u, 3u );
		CHECK( f.env.valueNs == 500u );
		CHECK( f.delay( W ) == 0u );
		CHECK( f.delay( 1u + W ) == bidir_target_delay( 500u, T ) );
		// Later samples raise it by the same rule, once K lies above D.
		f( 900u, 4u );
		CHECK( f.env.valueNs == 500u );
		f( 950u, 5u );
		CHECK( f.env.valueNs == 900u );
	}

	// E = 450: D = 600, and K in [350, 600] never moves E.
	Feed_t settled;
	settled( 450u, 1u )( 450u, 2u )( 450u, 3u );
	REQUIRE( settled.env.valueNs == 450u );
	REQUIRE( bidir_target_delay( settled.env.valueNs, T ) == 600u );
	for ( uint64_t valueNs = 350u; valueNs <= 600u; valueNs += 10u )
	{
		Feed_t f = settled;
		for ( uint64_t i = 0; i < 100u; i++ )
			f( valueNs, 10u + i * H / 10u );
		CHECK( f.env.valueNs == ( valueNs == 350u ? 350u : 450u ) );
	}

	// Fewer than k samples above D within a window never move E, however high.
	{
		Feed_t f = settled;
		for ( uint64_t i = 0; i < 60u; i++ )
			f( i % 30u == 7u || i % 30u == 9u ? 100'000u : 450u, 10u + i * W / 20u );
		CHECK( f.env.valueNs == 450u );
	}
	// k of them do, whatever lies between them: E takes the k-th highest.
	{
		Feed_t f = settled;
		f( 700u, 10u )( 450u, 20u )( 100'000u, 30u )( 450u, 40u );
		CHECK( f.env.valueNs == 450u );
		f( 650u, 50u );
		CHECK( f.env.valueNs == 650u );
	}
	// Samples older than the window no longer count.
	{
		Feed_t f = settled;
		f( 700u, 10u )( 700u, 20u )( 450u, 10u + W )( 450u, 20u + W )( 450u, 30u + W )( 700u, 40u + W );
		CHECK( f.env.valueNs == 450u );
	}

	// K at least a vblank below E releases E to K exactly at the hold: the margin
	// is inclusive, and K rising back within it restarts the hold.
	{
		Feed_t f = settled;
		f( 350u, 10u * W )( 350u, 10u * W + 1u )( 350u, 10u * W + 2u );
		f( 350u, 10u * W + H - 1u );
		CHECK( f.env.valueNs == 450u );
		f( 350u, 10u * W + H );
		CHECK( f.env.valueNs == 350u );
	}
	{
		Feed_t f = settled;
		f( 300u, 10u * W );
		for ( uint64_t i = 1u; i <= 4u; i++ )
			f( 500u, 10u * W + i );
		f( 300u, 11u * W + 10u )( 300u, 11u * W + 20u )( 300u, 11u * W + 30u );
		f( 300u, 10u * W + H );
		CHECK( f.env.valueNs == 450u );
		f( 300u, 11u * W + 10u + H );
		CHECK( f.env.valueNs == 300u );
	}

	// Zero samples (gated plans) never count: K needs k non-zero latencies.
	{
		Feed_t f;
		f( 500u, 1u )( 0u, 2u )( 0u, 3u )( 500u, 4u )( 0u, 5u );
		CHECK( f.env.valueNs == 0u );
		f( 500u, 6u );
		CHECK( f.env.valueNs == 500u );
	}

	// Saturating near the top of the range.
	{
		Feed_t f;
		f( UINT64_MAX, 1u )( UINT64_MAX, 2u )( UINT64_MAX, 3u );
		CHECK( f.env.valueNs == UINT64_MAX );
		CHECK( f.delay( UINT64_MAX ) == UINT64_MAX );
	}
}

TEST_CASE( "the bidir window keeps a window of samples, and at least k", "[framegen][deadline]" )
{
	constexpr uint64_t W = k_ulBidirWindowNs;
	BidirWindow_t window;
	CHECK_FALSE( window.full( W ) );
	for ( uint64_t i = 1u; i <= 3u; i++ )
		window.push( { .atNs = i, .latencyNs = i * 10u, .sourceSpanNs = 100u - i, .recordSpanNs = 100u - i } );
	CHECK( window.count == 3u );
	CHECK( window.kth_highest_latency() == 10u );
	CHECK( window.kth_lowest_span() == 99u );
	// Spans short on one clock only leave the longer clock's.
	BidirWindow_t clocks = window;
	for ( uint64_t i = 4u; i <= 6u; i++ )
		clocks.push( { .atNs = i, .sourceSpanNs = 10u, .recordSpanNs = 200u } );
	CHECK( clocks.kth_lowest_span() == 99u );
	CHECK_FALSE( window.full( W ) );
	CHECK( window.full( 1u + W ) );
	// A sample a window later evicts the older ones but keeps the newest k.
	window.push( { .atNs = 10u * W, .latencyNs = 5u } );
	CHECK( window.count == 3u );
	CHECK( window.kth_highest_latency() == 5u );
	CHECK( window.kth_lowest_span() == 0u );
	// More samples than the buffer within a window keep the newest of them.
	BidirWindow_t dense;
	for ( uint64_t i = 1u; i <= k_nBidirWindowSamples + 44u; i++ )
		dense.push( { .atNs = i, .latencyNs = i } );
	CHECK( dense.count == k_nBidirWindowSamples );
	CHECK( dense.kth_highest_latency() == k_nBidirWindowSamples + 42u );
	CHECK( dense.sinceNs == 1u );
}

TEST_CASE( "the bidir ring bound holds the reals the ring can hold", "[framegen][deadline]" )
{
	constexpr uint64_t T = 100u;
	constexpr uint64_t H = k_ulBidirReleaseHoldNs;
	// k entries stay spare; a ring of no more than k holds no delayed timeline.
	STATIC_REQUIRE( bidir_ring_reals( 3u ) == 0u );
	STATIC_REQUIRE( bidir_ring_reals( 6u ) == 3u );
	STATIC_REQUIRE( bidir_ring_delay( 3u, 1'000u, T ) == 0u );
	STATIC_REQUIRE( bidir_ring_delay( 6u, 1'000u, 0u ) == 0u );
	// One real per vblank until a span is measured; whole vblanks, rounded down.
	STATIC_REQUIRE( bidir_ring_delay( 6u, 0u, T ) == 3u * T );
	STATIC_REQUIRE( bidir_ring_delay( 10u, 0u, T ) == 7u * T );
	STATIC_REQUIRE( bidir_ring_delay( 6u, 3u * T + 99u, T ) == 3u * T );
	STATIC_REQUIRE( bidir_ring_delay( 6u, 9u * T, T ) == 9u * T );

	// The span sums the newest n intervals.
	BidirSpacings_t spacings;
	CHECK( spacings.span( 1u ) == 0u );
	for ( uint64_t i = 1u; i <= 3u; i++ )
		spacings.push( i * T );
	CHECK( spacings.span( 0u ) == 0u );
	CHECK( spacings.span( 4u ) == 0u );
	CHECK( spacings.span( 3u ) == 6u * T );
	CHECK( spacings.span( 2u ) == 5u * T );
	for ( uint64_t i = 4u; i <= 20u; i++ )
		spacings.push( i * T );
	CHECK( spacings.span( 3u ) == 57u * T );
	CHECK( spacings.span( k_nBidirSpacings ) != 0u );
	CHECK( spacings.span( k_nBidirSpacings + 1u ) == 0u );

	// The bound in force falls at once and rises only after the measured bound
	// stayed above it for the hold; a dip restarts the hold.
	constexpr BidirRingBound_t bound = { .delayNs = 5u * T };
	STATIC_REQUIRE( hold_bidir_ring_bound( bound, 3u * T, 10u ).delayNs == 3u * T );
	STATIC_REQUIRE( hold_bidir_ring_bound( bound, 5u * T, 10u ).aboveSinceNs == 0u );
	constexpr BidirRingBound_t above = hold_bidir_ring_bound( bound, 9u * T, 10u );
	STATIC_REQUIRE( above.delayNs == 5u * T );
	STATIC_REQUIRE( above.aboveSinceNs == 10u );
	STATIC_REQUIRE( hold_bidir_ring_bound( above, 8u * T, 9u + H ).delayNs == 5u * T );
	STATIC_REQUIRE( hold_bidir_ring_bound( above, 8u * T, 10u + H ).delayNs == 8u * T );
	STATIC_REQUIRE( hold_bidir_ring_bound( above, 8u * T, 10u + H ).aboveSinceNs == 0u );
	constexpr BidirRingBound_t dipped = hold_bidir_ring_bound( above, 5u * T, 20u );
	STATIC_REQUIRE( dipped.aboveSinceNs == 0u );
	STATIC_REQUIRE( hold_bidir_ring_bound( hold_bidir_ring_bound( dipped, 9u * T, 30u ), 9u * T, 10u + H ).delayNs == 5u * T );
}

TEST_CASE( "the bidir delay derives from the envelope", "[framegen][deadline]" )
{
	constexpr uint64_t T = 100u;
	STATIC_REQUIRE( bidir_target_delay( 0u, T ) == 0u );
	STATIC_REQUIRE( bidir_target_delay( 1u, T ) == 2u * T );
	STATIC_REQUIRE( bidir_target_delay( 200u, T ) == 3u * T );
	STATIC_REQUIRE( bidir_target_delay( 201u, T ) == 4u * T );
	STATIC_REQUIRE( bidir_target_delay( 10'000u, T ) == 101u * T );
	STATIC_REQUIRE( bidir_target_delay( UINT64_MAX - T, T ) == UINT64_MAX );
	STATIC_REQUIRE( bidir_target_delay( UINT64_MAX, T ) == UINT64_MAX );

	STATIC_REQUIRE( round_up_to_vblanks( 20'000'000u, 16'666'667u ) == 33'333'334u );
	STATIC_REQUIRE( round_up_to_vblanks( 16'666'667u, 16'666'667u ) == 16'666'667u );
	STATIC_REQUIRE( round_up_to_vblanks( 5u, 0u ) == 5u );
	STATIC_REQUIRE( round_up_to_vblanks( UINT64_MAX, 3u ) == UINT64_MAX );
}

TEST_CASE( "a bidir backlog is a queued real waiting behind a real", "[framegen][deadline]" )
{
	constexpr uint64_t T = 100u;
	constexpr uint64_t opp = 10'000u;
	constexpr BidirSlotKind_t real = BidirSlotKind_t::RealEndpoint;
	constexpr BidirSlotKind_t generated = BidirSlotKind_t::Generated;

	// A queued real's excess: how much later than its display time it is shown.
	STATIC_REQUIRE( bidir_real_excess( { real, opp + T, 1u, opp + 50u }, opp ) == 50u );
	STATIC_REQUIRE( bidir_real_excess( { real, opp - T, 1u, opp - 4u * T }, opp ) == 4u * T );
	STATIC_REQUIRE( bidir_real_excess( { real, opp + T, 1u, 0u }, opp ) == 0u );
	STATIC_REQUIRE( bidir_real_excess( { generated, opp + T, 0u, opp }, opp ) == 0u );

	// One late real on its own slot is not a backlog.
	constexpr BidirPresenterView_t view = { .opportunityNs = opp, .vblankIntervalNs = T, .capacity = 6u };
	const std::array late = { BidirQueueEntry_t{ real, opp - 3u * T, 1u, opp - 3u * T - 10u } };
	CHECK( bidir_backlog_excess( late, view ) == 0u );
	// The oldest real displaced behind an older one is.
	const std::array displaced = {
		BidirQueueEntry_t{ generated, opp, 0u },
		BidirQueueEntry_t{ real, opp + T, 1u, opp - T + 10u },
	};
	CHECK( bidir_backlog_excess( displaced, view ) == 2u * T - 10u );
	// So are two due reals; once a front was shown at this opportunity the
	// oldest can only be shown at the next one.
	const std::array twoDue = {
		BidirQueueEntry_t{ real, opp - T, 1u, opp - T - 10u },
		BidirQueueEntry_t{ real, opp, 2u, opp - 10u },
	};
	CHECK( bidir_backlog_excess( twoDue, view ) == T + 10u );
	BidirPresenterView_t shown = view;
	shown.lastFrontOpportunityNs = opp;
	STATIC_REQUIRE( bidir_earliest_front( view ) == opp );
	CHECK( bidir_earliest_front( shown ) == opp + T );
	CHECK( bidir_backlog_excess( twoDue, shown ) == 2u * T + 10u );
	CHECK( bidir_backlog_excess( std::span<const BidirQueueEntry_t>(), view ) == 0u );

	// An excess of at most a vblank clears the backlog; a new lowest or highest
	// excess restarts its window; one that keeps its range for the window did
	// not drain.
	constexpr uint64_t window = 6u * T;
	constexpr BidirBacklog_t backlog = { .lowestExcessNs = 300u, .highestExcessNs = 400u, .sinceNs = opp };
	STATIC_REQUIRE_FALSE( observe_bidir_backlog( backlog, T, opp + T, T, window ).undrained );
	STATIC_REQUIRE( observe_bidir_backlog( backlog, T, opp + T, T, window ).backlog.sinceNs == 0u );
	STATIC_REQUIRE( observe_bidir_backlog( {}, 300u, opp, T, window ).backlog.sinceNs == opp );
	STATIC_REQUIRE( observe_bidir_backlog( {}, 300u, opp, T, window ).backlog.highestExcessNs == 300u );
	STATIC_REQUIRE( observe_bidir_backlog( backlog, 250u, opp + 5u * T, T, window ).backlog.sinceNs == opp + 5u * T );
	STATIC_REQUIRE( observe_bidir_backlog( backlog, 250u, opp + 5u * T, T, window ).backlog.lowestExcessNs == 250u );
	STATIC_REQUIRE( observe_bidir_backlog( backlog, 450u, opp + 6u * T, T, window ).backlog.sinceNs == opp + 6u * T );
	STATIC_REQUIRE( observe_bidir_backlog( backlog, 450u, opp + 6u * T, T, window ).backlog.highestExcessNs == 450u );
	STATIC_REQUIRE_FALSE( observe_bidir_backlog( backlog, 350u, opp + 5u * T, T, window ).undrained );
	STATIC_REQUIRE( observe_bidir_backlog( backlog, 350u, opp + 6u * T, T, window ).undrained );
	STATIC_REQUIRE( observe_bidir_backlog( backlog, 400u, opp + 6u * T, T, window ).undrained );
	STATIC_REQUIRE( observe_bidir_backlog( backlog, 300u, opp + 6u * T, T, window ).backlog.sinceNs == 0u );
}

TEST_CASE( "bidir planning helpers share the presenter's rules", "[framegen][deadline]" )
{
	constexpr uint64_t T = 100u;
	constexpr uint64_t opp = 10'000u;

	// The presenter's facts, field by field; the completion is the caller's.
	constexpr BidirPresenterView_t view = {
		.opportunityNs = opp,
		.lastFrontOpportunityNs = opp - T,
		.vblankIntervalNs = T,
		.capacity = 6u,
		.beforeWake = true,
	};
	constexpr BidirQueueEntry_t real = { BidirSlotKind_t::RealEndpoint, opp + T, 7u };
	constexpr BidirFrontInputs_t inputs = bidir_front_inputs( real, false, 5u, view );
	STATIC_REQUIRE( inputs.targetNs == opp + T );
	STATIC_REQUIRE( inputs.opportunityNs == opp );
	STATIC_REQUIRE( inputs.lastFrontOpportunityNs == opp - T );
	STATIC_REQUIRE( inputs.vblankIntervalNs == T );
	STATIC_REQUIRE( inputs.real );
	STATIC_REQUIRE_FALSE( inputs.completed );
	STATIC_REQUIRE_FALSE( inputs.forceDrain );
	STATIC_REQUIRE( inputs.beforeWake );
	STATIC_REQUIRE( bidir_front_inputs( real, true, 6u, view ).forceDrain );
	STATIC_REQUIRE( bidir_front_inputs( real, true, 6u, view ).completed );
	STATIC_REQUIRE_FALSE( bidir_front_inputs( { BidirSlotKind_t::Generated, opp, 0u }, true, 1u, view ).real );

	// The floor of a real is the newest target of an older queued real.
	const std::array queue = {
		BidirQueueEntry_t{ BidirSlotKind_t::RealEndpoint, opp, 4u },
		BidirQueueEntry_t{ BidirSlotKind_t::Generated, opp + T, 5u },
		BidirQueueEntry_t{ BidirSlotKind_t::RealEndpoint, opp + 2u * T, 5u },
	};
	CHECK( bidir_real_floor( queue, 4u ) == 0u );
	CHECK( bidir_real_floor( queue, 5u ) == opp );
	CHECK( bidir_real_floor( queue, 6u ) == opp + 2u * T );
}

TEST_CASE( "a shorter bidir delay moves the queued reals with it", "[framegen][deadline]" )
{
	constexpr uint64_t T = 100u;
	constexpr uint64_t opp = 10'000u;
	const DisplayGrid_t grid = { .D0 = opp, .W0 = opp - 30u, .T = T };
	const std::vector<BidirQueueEntry_t> queue = {
		{ BidirSlotKind_t::RealEndpoint, opp + 2u * T, 1u, opp + 2u * T },
		{ BidirSlotKind_t::Generated, opp + 3u * T, 2u },
		{ BidirSlotKind_t::RealEndpoint, opp + 4u * T, 2u, opp + 4u * T },
		{ BidirSlotKind_t::RealEndpoint, opp + 5u * T, 3u, opp + 5u * T - 50u },
		{ BidirSlotKind_t::RealEndpoint, opp + 6u * T, 4u, 0u },	// queued outside a plan
	};
	// Each real moves the difference earlier on the grid, never before the
	// earliest opportunity, one per vblank and never later than it was; the
	// interpolation goes.
	std::vector<BidirQueueEntry_t> pulled = queue;
	pull_bidir_queue( pulled, 3u * T, grid, opp );
	REQUIRE( pulled.size() == 4u );
	CHECK( pulled[ 0 ].targetNs == opp );
	CHECK( pulled[ 0 ].idealNs == opp - T );
	CHECK( pulled[ 1 ].targetNs == opp + T );
	CHECK( pulled[ 1 ].idealNs == opp + T );
	CHECK( pulled[ 2 ].targetNs == opp + 2u * T );
	CHECK( pulled[ 2 ].idealNs == opp + 2u * T - 50u );
	CHECK( pulled[ 3 ].targetNs == opp + 6u * T );
	CHECK( pulled[ 3 ].idealNs == 0u );
	// A front already shown at this opportunity: the next one is the earliest.
	pulled = queue;
	pull_bidir_queue( pulled, 3u * T, grid, opp + T );
	CHECK( pulled[ 0 ].targetNs == opp + T );
	CHECK( pulled[ 1 ].targetNs == opp + 2u * T );
	CHECK( pulled[ 2 ].targetNs == opp + 3u * T );
	// A small difference keeps the reals on their own vblanks.
	pulled = queue;
	pull_bidir_queue( pulled, 50u, grid, opp );
	CHECK( pulled[ 0 ].targetNs == opp + 2u * T );
	CHECK( pulled[ 1 ].targetNs == opp + 4u * T );
	CHECK( pulled[ 2 ].targetNs == opp + 4u * T + T );
}

namespace
{

constexpr uint64_t k_ulStepT = 100u;
constexpr uint64_t k_ulStepOpportunityNs = k_ulBidirReleaseHoldNs + k_ulBidirWindowNs + 10'000u;

BidirPlanInputs_t step_inputs( std::span<const BidirQueueEntry_t> queue, BidirEndpoint_t previous,
	BidirEndpoint_t current, uint64_t opportunityNs = k_ulStepOpportunityNs, size_t capacity = 6u )
{
	return {
		.queue = queue,
		.previous = previous,
		.current = current,
		.wakeNs = opportunityNs - 30u,
		.nowNs = opportunityNs - 25u,
		.presenter = { .opportunityNs = opportunityNs, .vblankIntervalNs = k_ulStepT, .capacity = capacity },
		.allowGeneration = true,
	};
}

// A real recorded (and ready at the source) at recordNs.
constexpr BidirEndpoint_t step_real( uint64_t id, uint64_t recordNs )
{
	return { .realFrameId = id, .sourceReadyNs = recordNs, .recordNs = recordNs };
}

// A translation already established with D = delayNs on E = delayNs - 150,
// samples spanning a window and a ring bound of ringNs: a real recorded at r is
// shown at r + D.
BidirTimelineState_t established_state( uint64_t delayNs = 300u, uint64_t ringNs = 300u )
{
	constexpr uint64_t opp = k_ulStepOpportunityNs;
	BidirTimelineState_t state = {
		.epoch = establish_bidir_epoch( delayNs, 1u ),
		.grid = { .D0 = opp, .W0 = opp - 30u, .T = k_ulStepT },
		.gridEpoch = 1u,
		.pairLatency = { .valueNs = delayNs - 150u },
		.ringBound = { .delayNs = ringNs },
	};
	state.window.sinceNs = opp - k_ulBidirWindowNs;
	return state;
}

std::vector<BidirQueueEntry_t> reals_queue( uint64_t firstTargetNs, size_t count )
{
	std::vector<BidirQueueEntry_t> queue;
	for ( uint64_t i = 0u; i < count; i++ )
		queue.push_back( { BidirSlotKind_t::RealEndpoint, firstTargetNs + i * k_ulStepT, i + 1u, firstTargetNs + i * k_ulStepT } );
	return queue;
}

} // namespace

TEST_CASE( "the bidir plan step establishes, re-establishes and admits by one rule", "[framegen][deadline]" )
{
	constexpr uint64_t T = k_ulStepT;
	constexpr uint64_t opp = k_ulStepOpportunityNs;
	constexpr uint64_t W = k_ulBidirWindowNs;
	const std::span<const BidirQueueEntry_t> empty;

	// The pairs of the first window prime: D needs samples spanning it.
	BidirTimelineState_t state;
	for ( const uint64_t atNs : { opp - W, opp - W / 2u } )
	{
		const BidirPlanStep_t step = plan_bidir_step( state, step_inputs( empty, step_real( 1u, atNs - 200u ),
			step_real( 2u, atNs - 100u ), atNs ) );
		CHECK_FALSE( step.planned );
		CHECK( step.state.gridEpoch == 1u );
		state = step.state;
	}
	// The third establishes on the pair latency (200): D = 200 + T. The current
	// real is shown D after its record; the previous real, not yet consumed, is
	// queued too, with the interpolation between them.
	const BidirPlanStep_t first = plan_bidir_step( state, step_inputs( empty, step_real( 3u, opp - 200u ), step_real( 4u, opp - 100u ) ) );
	REQUIRE( first.planned );
	CHECK( first.reestablish == BidirReestablish_t::Established );
	CHECK( first.state.pairLatency.valueNs == 200u );
	CHECK( first.state.epoch.delayNs == 3u * T );
	CHECK( first.state.ringBound.delayNs == 3u * T );
	CHECK( first.state.epoch.endpoint_display_time( opp - 100u ) == opp + 200u );
	CHECK( first.state.grid.D0 == opp );
	CHECK( first.state.grid.W0 == opp - 30u );
	CHECK( first.previous.queue );
	CHECK( first.previous.targetNs == opp + T );
	CHECK( first.previous.idealNs == opp + T );
	CHECK( first.current.targetNs == opp + 2u * T );
	CHECK( first.current.idealNs == opp + 2u * T );
	CHECK( first.current.wakeNs == opp + 2u * T - 30u );
	CHECK( first.admitted.empty() );
	CHECK( first.retargeted.empty() );
	CHECK( first.droppedRealIds.empty() );
	CHECK( first.shedCapacity == 6u );
	// Record time places the reals, whatever their source times.
	{
		BidirPlanInputs_t in = step_inputs( empty, step_real( 3u, opp - 200u ), step_real( 4u, opp - 100u ) );
		in.previous.sourceReadyNs = opp - 900u;
		in.current.sourceReadyNs = opp - 850u;
		const BidirPlanStep_t step = plan_bidir_step( state, in );
		CHECK( step.current.idealNs == opp + 2u * T );
		// A pair whose records are out of order, or unknown, is not planned.
		in.current.recordNs = opp - 201u;
		CHECK_FALSE( plan_bidir_step( state, in ).planned );
		in.current.recordNs = opp - 100u;
		in.previous.recordNs = 0u;
		CHECK_FALSE( plan_bidir_step( state, in ).planned );
	}
	// A previous real that already entered the timeline is not queued again.
	{
		BidirPlanInputs_t in = step_inputs( empty, step_real( 3u, opp - 200u ), step_real( 4u, opp - 100u ) );
		in.consumedFrameId = 3u;
		CHECK_FALSE( plan_bidir_step( state, in ).previous.queue );
	}

	// No vblank: nothing at all changes.
	{
		BidirPlanInputs_t in = step_inputs( empty, step_real( 4u, opp - 100u ), step_real( 5u, opp ) );
		in.presenter.vblankIntervalNs = 0u;
		const BidirPlanStep_t step = plan_bidir_step( first.state, in );
		CHECK_FALSE( step.planned );
		CHECK( step.state.gridEpoch == first.state.gridEpoch );
		CHECK( step.state.window.count == first.state.window.count );
	}

	// An invalidation resets the samples and the estimates with the translation,
	// on a new grid epoch that never wraps to 0; a cleared translation keeps the
	// envelope and D.
	{
		BidirPlanInputs_t in = step_inputs( empty, step_real( 4u, opp - 100u ), step_real( 5u, opp ), opp + T );
		in.invalidate = true;
		const BidirPlanStep_t step = plan_bidir_step( first.state, in );
		CHECK_FALSE( step.planned );
		CHECK( step.state.gridEpoch == 2u );
		CHECK_FALSE( step.state.epoch.valid );
		CHECK( step.state.pairLatency.valueNs == 0u );
		CHECK( step.state.window.count == 1u );
		CHECK( step.state.epoch.delayNs == 0u );

		BidirTimelineState_t wrapped = first.state;
		wrapped.gridEpoch = UINT64_MAX;
		CHECK( plan_bidir_step( wrapped, in ).state.gridEpoch == 1u );

		BidirTimelineState_t cut = first.state;
		cut.epoch = {};
		const BidirPlanStep_t again = plan_bidir_step( cut, step_inputs( empty, step_real( 4u, opp - 100u ), step_real( 5u, opp ), opp + T ) );
		REQUIRE( again.planned );
		CHECK( again.reestablish == BidirReestablish_t::Established );
		CHECK( again.state.epoch.delayNs == first.state.epoch.delayNs );
		CHECK( again.state.epoch.endpoint_display_time( opp ) == opp + 3u * T );
	}

	// The pair latency is sampled only at the first plan of an opportunity and
	// while no real waits behind a real; the ring span at every plan.
	{
		const auto sampled = []( uint64_t plannedAtNs, std::span<const BidirQueueEntry_t> queue ) {
			BidirTimelineState_t state = established_state();
			state.plannedAtNs = plannedAtNs;
			const BidirPlanStep_t step = plan_bidir_step( state, step_inputs( queue, step_real( 2u, opp - 1'000u ), step_real( 3u, opp - 900u ) ) );
			CHECK( step.state.plannedAtNs == opp );
			CHECK( step.state.window.count == 1u );
			CHECK( step.state.sourceSpacings.count == 1u );
			CHECK( step.state.recordSpacings.count == 1u );
			return step.state.window.samples[ step.state.window.first ].latencyNs;
		};
		CHECK( sampled( 0u, empty ) == 1'000u );
		CHECK( sampled( opp, empty ) == 0u );
		CHECK( sampled( opp + T, empty ) == 1'000u );
		const std::array backlog = {
			BidirQueueEntry_t{ BidirSlotKind_t::RealEndpoint, opp, 1u, opp - 2u * T },
		};
		CHECK( sampled( 0u, backlog ) == 0u );
	}
}

TEST_CASE( "the bidir plan step moves D at once to protect and after a hold to relax", "[framegen][deadline]" )
{
	constexpr uint64_t T = k_ulStepT;
	constexpr uint64_t opp = k_ulStepOpportunityNs;
	constexpr uint64_t H = k_ulBidirReleaseHoldNs;
	const std::span<const BidirQueueEntry_t> empty;

	// k pair latencies above D lengthen it at once; queued reals keep their
	// targets and the current real follows them.
	{
		BidirTimelineState_t rising = established_state( 300u, 700u );
		for ( const uint64_t atNs : { opp - 200u, opp - 100u } )
			rising.window.push( { .atNs = atNs, .latencyNs = 400u } );
		const std::array queue = { BidirQueueEntry_t{ BidirSlotKind_t::RealEndpoint, opp + 3u * T, 2u, opp + 3u * T } };
		BidirPlanInputs_t in = step_inputs( queue, step_real( 2u, opp - 400u ), step_real( 3u, opp - 350u ), opp, 10u );
		in.consumedFrameId = 2u;
		const BidirPlanStep_t step = plan_bidir_step( rising, in );
		REQUIRE( step.planned );
		CHECK( step.reestablish == BidirReestablish_t::Lengthened );
		CHECK( step.state.pairLatency.valueNs == 400u );
		CHECK( step.state.epoch.delayNs == 5u * T );
		CHECK( step.retargeted.empty() );
		CHECK_FALSE( step.previous.queue );
		CHECK( step.current.idealNs == opp - 350u + 5u * T );
		CHECK( step.current.targetNs == opp + 4u * T );
		// The ring bound in force caps it: a bound of 4T lengthens to 4T only.
		BidirTimelineState_t capped = rising;
		capped.ringBound = { .delayNs = 4u * T };
		CHECK( plan_bidir_step( capped, in ).state.epoch.delayNs == 4u * T );
	}

	// A held D beyond the ring bound shortens at once to it: the ring of a
	// smaller capacity holds 3T. A larger ring raises the bound back only after
	// it measured above it for the hold.
	{
		const BidirTimelineState_t held = established_state( 500u, 700u );
		const BidirPlanStep_t shortened = plan_bidir_step( held,
			step_inputs( empty, step_real( 2u, opp - 400u ), step_real( 3u, opp - 350u ), opp, 6u ) );
		REQUIRE( shortened.planned );
		CHECK( shortened.reestablish == BidirReestablish_t::Shortened );
		CHECK( shortened.state.epoch.delayNs == 3u * T );
		CHECK( shortened.state.pairLatency.valueNs == 350u );

		BidirTimelineState_t state = shortened.state;
		uint64_t atNs = opp;
		for ( uint32_t i = 1u; i <= 3u; i++ )
		{
			atNs = opp + i * H / 2u;
			const BidirPlanStep_t step = plan_bidir_step( state,
				step_inputs( empty, step_real( 2u + i, atNs - 400u ), step_real( 3u + i, atNs - 350u ), atNs, 10u ) );
			REQUIRE( step.planned );
			CHECK( step.state.epoch.delayNs == ( i < 3u ? 3u * T : 5u * T ) );
			state = step.state;
		}
	}

	// K at least a vblank below E for the hold releases it: D shortens, the
	// queued reals move earlier by the difference and the interpolations
	// planned for the longer D go. The newest staying real beyond what the
	// display shows within the new D goes in the same step, and both are
	// reported.
	{
		BidirTimelineState_t falling = established_state( 500u, 700u );
		falling.pairLatency.belowSinceNs = opp - H;
		for ( const uint64_t atNs : { opp - 200u, opp - 100u } )
			falling.window.push( { .atNs = atNs, .latencyNs = 100u } );
		std::vector<BidirQueueEntry_t> queue = reals_queue( opp + 2u * T, 6u );
		queue.insert( queue.begin() + 1, { BidirSlotKind_t::Generated, opp + 2u * T + T / 2u, 2u } );
		BidirPlanInputs_t in = step_inputs( queue, step_real( 6u, opp - 200u ), step_real( 7u, opp - 100u ), opp, 10u );
		in.consumedFrameId = 6u;
		const BidirPlanStep_t step = plan_bidir_step( falling, in );
		REQUIRE( step.planned );
		CHECK( step.reestablish == BidirReestablish_t::Shortened );
		CHECK( step.state.pairLatency.valueNs == 100u );
		CHECK( step.state.epoch.delayNs == 2u * T );
		REQUIRE( step.retargeted.size() == 6u );
		for ( size_t i = 0u; i < step.retargeted.size(); i++ )
		{
			CHECK( step.retargeted[ i ].kind == BidirSlotKind_t::RealEndpoint );
			CHECK( step.retargeted[ i ].realFrameId == i + 1u );
			CHECK( step.retargeted[ i ].targetNs == opp + i * T );
			CHECK( step.retargeted[ i ].targetNs <= queue[ i == 0u ? 0u : i + 1u ].targetNs );
		}
		// The front leaves here; D/T + k = 5 reals stay at most with the current.
		CHECK( step.droppedRealIds == std::vector<uint64_t>{ 6u } );
		CHECK_FALSE( step.backlogDropped );
		CHECK( step.current.targetNs == opp + 5u * T );
	}
}

TEST_CASE( "the bidir plan step drops reals only beyond what the display shows in time", "[framegen][deadline]" )
{
	constexpr uint64_t T = k_ulStepT;
	constexpr uint64_t opp = k_ulStepOpportunityNs;

	// D = 3T: the display shows at most 6 reals within D + kT, on a ring of 10.
	// Six staying reals and the current one: the newest staying one goes, never
	// the front.
	{
		const BidirTimelineState_t state = established_state();
		const std::vector<BidirQueueEntry_t> full = reals_queue( opp + T, 6u );
		BidirPlanInputs_t in = step_inputs( full, step_real( 6u, opp - 200u ), step_real( 7u, opp - 100u ), opp, 10u );
		in.consumedFrameId = 6u;
		const BidirPlanStep_t step = plan_bidir_step( state, in );
		REQUIRE( step.planned );
		CHECK( step.reestablish == BidirReestablish_t::Kept );
		CHECK( step.droppedRealIds == std::vector<uint64_t>{ 6u } );
		CHECK_FALSE( step.backlogDropped );
		CHECK( step.current.targetNs == opp + 6u * T );
		// With the front leaving at this opportunity they fit.
		const std::vector<BidirQueueEntry_t> leaving = reals_queue( opp, 6u );
		BidirPlanInputs_t fits = step_inputs( leaving, step_real( 6u, opp - 200u ), step_real( 7u, opp - 100u ), opp, 10u );
		fits.consumedFrameId = 6u;
		CHECK( plan_bidir_step( state, fits ).droppedRealIds.empty() );
		// A front already shown here stays: then one goes.
		fits.presenter.lastFrontOpportunityNs = opp;
		CHECK( plan_bidir_step( state, fits ).droppedRealIds == std::vector<uint64_t>{ 6u } );
	}

	// A displaced oldest real whose excess keeps its range for a window did not
	// drain: the newest staying reals go, one per vblank of its excess, through
	// the same path. D and the translation stay.
	{
		BidirTimelineState_t state = established_state();
		state.backlog = { .lowestExcessNs = 2u * T, .highestExcessNs = 2u * T, .sinceNs = opp - k_ulBidirWindowNs };
		const std::array queue = {
			BidirQueueEntry_t{ BidirSlotKind_t::RealEndpoint, opp, 1u, opp - 2u * T },
			BidirQueueEntry_t{ BidirSlotKind_t::RealEndpoint, opp + T, 2u, opp - T },
			BidirQueueEntry_t{ BidirSlotKind_t::RealEndpoint, opp + 2u * T, 3u, opp },
		};
		BidirPlanInputs_t in = step_inputs( queue, step_real( 3u, opp - 400u ), step_real( 4u, opp - 250u ) );
		in.consumedFrameId = 3u;
		const BidirPlanStep_t step = plan_bidir_step( state, in );
		REQUIRE( step.planned );
		CHECK( step.reestablish == BidirReestablish_t::Kept );
		CHECK( step.backlogDropped );
		CHECK( step.droppedRealIds == std::vector<uint64_t>{ 3u, 2u } );
		CHECK( step.state.backlog.sinceNs == 0u );
		CHECK( step.state.epoch.delayNs == 3u * T );
		CHECK( step.current.targetNs == opp + T );
		// Its excess shrinking keeps every real, on a new window.
		BidirTimelineState_t draining = state;
		draining.backlog.lowestExcessNs = 3u * T;
		draining.backlog.highestExcessNs = 3u * T;
		const BidirPlanStep_t kept = plan_bidir_step( draining, in );
		CHECK( kept.droppedRealIds.empty() );
		CHECK_FALSE( kept.backlogDropped );
		CHECK( kept.state.backlog.sinceNs == opp );
	}

	// The front leaves with this paint only if the presenter shows it here: not
	// when this opportunity already showed a front, and not when it is an
	// interpolation whose GPU work missed its wake.
	{
		const std::array queue = {
			BidirQueueEntry_t{ BidirSlotKind_t::RealEndpoint, opp + T, 1u, opp + T },
			BidirQueueEntry_t{ BidirSlotKind_t::Generated, opp + 2u * T, 2u },
			BidirQueueEntry_t{ BidirSlotKind_t::RealEndpoint, opp + 3u * T, 2u, opp + 3u * T },
		};
		BidirPlanInputs_t in = step_inputs( queue, step_real( 2u, opp - 300u ), step_real( 3u, opp - 150u ), opp + T );
		const BidirPlanStep_t shown = plan_bidir_step( established_state(), in );
		REQUIRE( shown.planned );
		CHECK( shown.reestablish == BidirReestablish_t::Kept );
		CHECK( shown.shedCapacity == 7u );
		in.presenter.lastFrontOpportunityNs = opp + T;
		CHECK( plan_bidir_step( established_state(), in ).shedCapacity == 6u );

		const std::array generatedFront = {
			BidirQueueEntry_t{ BidirSlotKind_t::Generated, opp + T, 2u },
			BidirQueueEntry_t{ BidirSlotKind_t::RealEndpoint, opp + 3u * T, 2u, opp + 3u * T },
		};
		BidirPlanInputs_t slow = step_inputs( generatedFront, step_real( 2u, opp - 300u ), step_real( 3u, opp - 150u ), opp + T );
		slow.frontCompleted = false;
		CHECK( plan_bidir_step( established_state(), slow ).shedCapacity == 6u );
		slow.frontCompleted = true;
		CHECK( plan_bidir_step( established_state(), slow ).shedCapacity == 7u );
	}

	// Two endpoints whose display times share a grid slot are both queued, the
	// newer one on the next opportunity: no real is lost to coalescing.
	{
		BidirTimelineState_t state = established_state();
		state.epoch = {};
		const BidirPlanStep_t step = plan_bidir_step( state, step_inputs( {}, step_real( 1u, opp - 250u ), step_real( 2u, opp - 200u ) ) );
		REQUIRE( step.planned );
		CHECK( step.state.epoch.endpoint_display_time( opp - 250u ) == opp + 50u );
		CHECK( step.previous.queue );
		CHECK( step.previous.targetNs == opp + T );
		CHECK( step.current.targetNs == opp + 2u * T );
	}
}

TEST_CASE( "bidir front action shows one entry per opportunity and never a late interpolation", "[framegen][deadline]" )
{
	constexpr uint64_t T = 100u;
	constexpr uint64_t opp = 10'000u;
	using enum BidirFrontAction_t;
	constexpr auto act = []( uint64_t target, bool real, bool completed, bool forceDrain, uint64_t lastFront = 0u,
		bool beforeWake = false ) {
		return bidir_front_action( { .targetNs = target, .opportunityNs = opp, .lastFrontOpportunityNs = lastFront,
			.vblankIntervalNs = T, .real = real, .completed = completed, .forceDrain = forceDrain,
			.beforeWake = beforeWake } );
	};
	// { target, real, completed, forceDrain } -> action
	STATIC_REQUIRE( act( opp, true, false, false ) == Present );
	STATIC_REQUIRE( act( opp + T / 2u, true, false, false ) == Present );
	STATIC_REQUIRE( act( opp + T / 2u + 1u, true, false, false ) == Wait );
	STATIC_REQUIRE( act( opp + T / 2u + 1u, true, false, true ) == Present );
	STATIC_REQUIRE( act( opp - T / 2u - 1u, true, false, false ) == Present );	// a late real catches up
	STATIC_REQUIRE( act( opp - 5u * T, true, true, true ) == Present );
	STATIC_REQUIRE( act( opp, false, true, false ) == Present );
	STATIC_REQUIRE( act( opp + T / 2u, false, true, false ) == Present );
	STATIC_REQUIRE( act( opp - T / 2u, false, true, false ) == Present );
	STATIC_REQUIRE( act( opp, false, false, false ) == TooSlow );
	STATIC_REQUIRE( act( opp - T / 2u, false, false, true ) == TooSlow );
	// Unfinished before the compositor woke for the opportunity: still had
	// budget, so a discard rather than capacity evidence.
	STATIC_REQUIRE( act( opp, false, false, false, 0u, true ) == Discard );
	STATIC_REQUIRE( act( opp, false, true, false, 0u, true ) == Present );
	STATIC_REQUIRE( act( opp, true, false, false, 0u, true ) == Present );
	STATIC_REQUIRE( act( opp + T / 2u + 1u, false, true, false ) == Wait );
	STATIC_REQUIRE( act( opp + T / 2u + 1u, false, false, false ) == Wait );
	STATIC_REQUIRE( act( opp + T / 2u + 1u, false, true, true ) == Discard );
	STATIC_REQUIRE( act( opp - T / 2u - 1u, false, true, false ) == Discard );
	STATIC_REQUIRE( act( opp - T / 2u - 1u, false, false, false ) == Discard );
	// An opportunity that already showed a front waits, whatever the front.
	STATIC_REQUIRE( act( opp, true, true, true, opp ) == Wait );
	STATIC_REQUIRE( act( opp - 3u * T, false, true, true, opp - T / 2u ) == Wait );
	STATIC_REQUIRE( act( opp, true, true, false, opp - T / 2u - 1u ) == Present );

	// Exhaustively: a real is never retired, a missed interpolation is never
	// shown, and too slow is reported only for a due front.
	for ( const uint64_t target : { opp - 2u * T, opp - T / 2u - 1u, opp - T / 2u, opp, opp + T / 2u, opp + T / 2u + 1u, opp + 2u * T } )
	{
		for ( const bool real : { false, true } )
		{
			for ( const bool completed : { false, true } )
			{
				for ( const bool forceDrain : { false, true } )
				{
					for ( const uint64_t lastFront : { uint64_t( 0u ), opp - T, opp } )
					{
						for ( const bool beforeWake : { false, true } )
						{
							CAPTURE( target, real, completed, forceDrain, lastFront, beforeWake );
							const BidirFrontAction_t action = act( target, real, completed, forceDrain, lastFront, beforeWake );
							if ( real )
								CHECK( ( action == Present || action == Wait ) );
							if ( !real && slot_missed_opportunity( target, opp, T ) )
								CHECK( action != Present );
							if ( action == TooSlow )
								CHECK( ( slot_due_at( target, opp, T ) && !beforeWake ) );
							if ( lastFront == opp )
								CHECK( action == Wait );
							if ( forceDrain && lastFront != opp && !slot_due_at( target, opp, T ) )
								CHECK( action == ( real ? Present : Discard ) );
						}
					}
				}
			}
		}
	}
}

TEST_CASE( "the nearest-opportunity rule has half-vblank boundaries", "[framegen][deadline]" )
{
	constexpr uint64_t T = 100u;
	constexpr uint64_t opp = 10'000u;
	STATIC_REQUIRE( slot_due_at( opp + T / 2u, opp, T ) );
	STATIC_REQUIRE_FALSE( slot_due_at( opp + T / 2u + 1u, opp, T ) );
	STATIC_REQUIRE_FALSE( slot_missed_opportunity( opp - T / 2u, opp, T ) );
	STATIC_REQUIRE( slot_missed_opportunity( opp - T / 2u - 1u, opp, T ) );
	STATIC_REQUIRE( slot_due_at( UINT64_MAX, UINT64_MAX - 1u, T ) );
	STATIC_REQUIRE_FALSE( slot_missed_opportunity( UINT64_MAX - 1u, UINT64_MAX, T ) );
	// A slot is never both missed and not due.
	for ( uint64_t target = opp - 2u * T; target <= opp + 2u * T; target++ )
		CHECK( !( slot_missed_opportunity( target, opp, T ) && !slot_due_at( target, opp, T ) ) );
}

TEST_CASE( "a bidir timeline is stale only beyond its catch-up horizon", "[framegen][deadline]" )
{
	constexpr uint64_t T = 100u;
	constexpr uint64_t front = 10'000u;
	constexpr uint64_t horizon = 6u;
	STATIC_REQUIRE_FALSE( bidir_timeline_stale( front, front + horizon * T, horizon, T ) );
	STATIC_REQUIRE( bidir_timeline_stale( front, front + horizon * T + 1u, horizon, T ) );
	STATIC_REQUIRE_FALSE( bidir_timeline_stale( 0u, UINT64_MAX, horizon, T ) );
	STATIC_REQUIRE_FALSE( bidir_timeline_stale( UINT64_MAX - 1u, UINT64_MAX, horizon, T ) );
	STATIC_REQUIRE_FALSE( bidir_timeline_stale( front, UINT64_MAX, UINT64_MAX, T ) );
	STATIC_REQUIRE_FALSE( bidir_timeline_stale( front + T, front, horizon, T ) );
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

TEST_CASE( "native present-lead warmup rejects hitches and multi-vblank samples",
	"[framegen][deadline]" )
{
	constexpr uint64_t intervalNs = 8'333u;
	constexpr PresentLeadState_t empty;
	constexpr PresentLeadState_t hitch = update_present_lead(
		empty, 10'000u, 14'000u, intervalNs, true );
	constexpr PresentLeadState_t loadingOutlier = update_present_lead(
		empty, 10'000u, 10'000u + intervalNs + 1u,
		intervalNs, false );
	constexpr PresentLeadState_t missingInterval = update_present_lead(
		empty, 10'000u, 14'000u, 0u, false );
	constexpr PresentLeadState_t oneVblank = update_present_lead(
		empty, 10'000u, 10'000u + intervalNs, intervalNs, false );

	STATIC_REQUIRE( hitch.samples == 0u );
	STATIC_REQUIRE( hitch.emaNs == 0 );
	STATIC_REQUIRE( loadingOutlier.samples == 0u );
	STATIC_REQUIRE( loadingOutlier.emaNs == 0 );
	STATIC_REQUIRE( missingInterval.samples == 0u );
	STATIC_REQUIRE( missingInterval.emaNs == 0 );
	STATIC_REQUIRE( oneVblank.samples == 1u );
	STATIC_REQUIRE( oneVblank.emaNs == static_cast<int64_t>( intervalNs ) );
}

TEST_CASE( "mature native present-lead learner rejects guarded outliers",
	"[framegen][deadline]" )
{
	constexpr uint64_t intervalNs = 8'333u;
	constexpr PresentLeadState_t mature = {
		.emaNs = 2'000,
		.samples = k_uPresentLeadWarmupSamples,
	};
	constexpr PresentLeadState_t ordinary = update_present_lead(
		mature, 10'000u, 13'000u, intervalNs, false );
	constexpr PresentLeadState_t outlier = update_present_lead(
		mature, 20'000u, 20'000u + 2'000u + intervalNs,
		intervalNs, false );

	STATIC_REQUIRE( ordinary.emaNs == 2'125 );
	STATIC_REQUIRE( ordinary.samples == mature.samples + 1u );
	STATIC_REQUIRE( outlier.emaNs == mature.emaNs );
	STATIC_REQUIRE( outlier.samples == mature.samples );
}

TEST_CASE( "native minimum viable lead learns only proven latches and decays upward",
	"[framegen][deadline]" )
{
	constexpr uint64_t intervalNs = 8'000'000u;
	PresentLeadState_t habitual;
	MinimumViablePresentLeadState_t viable;
	const auto observe = [&]( uint64_t submitNs, uint64_t flipNs,
		bool hitch = false )
	{
		viable = update_minimum_viable_present_lead(
			viable, habitual, submitNs, flipNs, intervalNs, hitch );
		habitual = update_present_lead(
			habitual, submitNs, flipNs, intervalNs, hitch );
	};

	observe( 1'000'000u, 1'000'000u + intervalNs, false );
	observe( 20'000'000u, 24'000'000u, true );
	observe( 30'000'000u, 30'000'000u + intervalNs + 1u, false );
	CHECK( viable.samples == 0u );

	observe( 40'000'000u, 44'000'000u );
	observe( 50'000'000u, 52'000'000u );
	observe( 60'000'000u, 61'000'000u );
	observe( 70'000'000u, 75'000'000u );
	REQUIRE( viable.samples == k_uPresentLeadWarmupSamples );
	CHECK( viable.lowWaterNs == 1'125'000u );

	const FixedRefreshPresentLead_t learned =
		select_fixed_refresh_present_lead(
			habitual, viable, intervalNs / 10u, intervalNs );
	REQUIRE( learned.ready );
	CHECK( learned.learned );
	CHECK_FALSE( learned.overridden );
	CHECK( learned.leadNs == 1'125'000u );

	observe( 80'000'000u, 85'000'000u );
	CHECK( viable.lowWaterNs == 1'250'000u );
}

TEST_CASE( "native minimum viable lead backs off on late generation and re-earns",
	"[framegen][deadline]" )
{
	constexpr uint64_t intervalNs = 8'000'000u;
	PresentLeadState_t habitual = {
		.emaNs = 4'000'000,
		.samples = k_uPresentLeadWarmupSamples,
	};
	MinimumViablePresentLeadState_t viable = {
		.lowWaterNs = 1'000'000u,
		.samples = k_uPresentLeadWarmupSamples,
	};

	viable = back_off_minimum_viable_present_lead(
		viable, habitual, intervalNs );
	CHECK( viable.lowWaterNs == 1'000'000u );
	CHECK( viable.backoffFloorNs == 2'000'000u );
	CHECK( minimum_viable_present_lead_ns( viable ) == 2'000'000u );

	viable = update_minimum_viable_present_lead(
		viable, habitual, 10'000'000u, 15'000'000u,
		intervalNs, false );
	CHECK( viable.lowWaterNs == 1'125'000u );
	CHECK( viable.backoffFloorNs == 1'875'000u );

	for ( uint32_t miss = 0u; miss < 8u; miss++ )
		viable = back_off_minimum_viable_present_lead(
			viable, habitual, intervalNs );
	CHECK( minimum_viable_present_lead_ns( viable )
		== static_cast<uint64_t>( habitual.emaNs ) );
}

TEST_CASE( "native minimum viable lead backs off only for late learned presents",
	"[framegen][deadline]" )
{
	constexpr uint64_t intervalNs = 8'000'000u;
	constexpr uint64_t targetNs = 20'000'000u;
	constexpr FixedRefreshPresentLead_t learned = {
		.leadNs = 1'000'000u,
		.ready = true,
		.learned = true,
	};
	constexpr FixedRefreshPresentLead_t habitual = {
		.leadNs = 4'000'000u,
		.ready = true,
	};
	constexpr FixedRefreshPresentLead_t overridden = {
		.leadNs = 3'000'000u,
		.ready = true,
		.overridden = true,
	};

	STATIC_REQUIRE_FALSE( minimum_viable_present_lead_needs_backoff(
		learned, targetNs + intervalNs / 2u, targetNs, intervalNs ) );
	STATIC_REQUIRE( minimum_viable_present_lead_needs_backoff(
		learned, targetNs + intervalNs / 2u + 1u, targetNs, intervalNs ) );
	STATIC_REQUIRE_FALSE( minimum_viable_present_lead_needs_backoff(
		habitual, targetNs + intervalNs, targetNs, intervalNs ) );
	STATIC_REQUIRE_FALSE( minimum_viable_present_lead_needs_backoff(
		overridden, targetNs + intervalNs, targetNs, intervalNs ) );
}

TEST_CASE( "native commit lead keeps EMA warmup and lets override win",
	"[framegen][deadline]" )
{
	constexpr uint64_t intervalNs = 8'000'000u;
	constexpr uint64_t marginNs = intervalNs / 10u;
	constexpr PresentLeadState_t habitual = {
		.emaNs = 5'000'000,
		.samples = k_uPresentLeadWarmupSamples,
	};
	constexpr MinimumViablePresentLeadState_t warming = {
		.lowWaterNs = 1'000'000u,
		.samples = k_uPresentLeadWarmupSamples - 1u,
	};
	constexpr FixedRefreshPresentLead_t warmup =
		select_fixed_refresh_present_lead(
			habitual, warming, marginNs, intervalNs );
	STATIC_REQUIRE( warmup.ready );
	STATIC_REQUIRE_FALSE( warmup.learned );
	STATIC_REQUIRE( warmup.leadNs == 5'000'000u );

	constexpr MinimumViablePresentLeadState_t learned = {
		.lowWaterNs = 1'000'000u,
		.samples = k_uPresentLeadWarmupSamples,
	};
	constexpr FixedRefreshCommitPlan_t overridden =
		plan_fixed_refresh_commit(
			20'000'000u, habitual, learned,
			marginNs, intervalNs, 3'000'000u );
	STATIC_REQUIRE( overridden.earlyCommit );
	STATIC_REQUIRE( overridden.overriddenLead );
	STATIC_REQUIRE_FALSE( overridden.learnedLead );
	STATIC_REQUIRE( overridden.presentLeadNs == 3'000'000u );
	STATIC_REQUIRE( overridden.advanceNs == 3'800'000u );

	constexpr MinimumViablePresentLeadState_t belowFloor = {
		.lowWaterNs = 200'000u,
		.samples = k_uPresentLeadWarmupSamples,
	};
	constexpr FixedRefreshPresentLead_t floored =
		select_fixed_refresh_present_lead(
			habitual, belowFloor, marginNs, intervalNs );
	STATIC_REQUIRE( floored.leadNs
		== k_uMinimumViablePresentLeadFloorNs );

	constexpr MinimumViablePresentLeadState_t aboveHabitual = {
		.lowWaterNs = 7'000'000u,
		.samples = k_uPresentLeadWarmupSamples,
	};
	constexpr FixedRefreshPresentLead_t capped =
		select_fixed_refresh_present_lead(
			habitual, aboveHabitual, marginNs, intervalNs );
	STATIC_REQUIRE( capped.leadNs
		== static_cast<uint64_t>( habitual.emaNs ) );
}

TEST_CASE( "fixed-refresh commit wake uses mature present lead and margin", "[framegen][deadline]" )
{
	constexpr uint64_t intervalNs = 8'000u;
	const PresentLeadState_t warming = {
		.emaNs = 3'000u,
		.samples = k_uPresentLeadWarmupSamples - 1u,
	};
	const FixedRefreshCommitPlan_t warmup =
		plan_fixed_refresh_commit( 20'000u, warming, 1'000u, intervalNs );
	CHECK_FALSE( warmup.earlyCommit );
	CHECK( warmup.commitDeadlineNs == 0u );

	PresentLeadState_t mature = warming;
	mature.samples = k_uPresentLeadWarmupSamples;
	const FixedRefreshCommitPlan_t plan =
		plan_fixed_refresh_commit( 20'000u, mature, 1'000u, intervalNs );
	REQUIRE( plan.earlyCommit );
	CHECK( plan.commitDeadlineNs == 16'000u );
	CHECK( plan.presentLeadNs == 3'000u );
	CHECK( plan.advanceNs == 4'000u );

	// The margin remains part of the advance even at a zero learned lead.
	mature.emaNs = 0;
	const FixedRefreshCommitPlan_t marginOnly =
		plan_fixed_refresh_commit( 20'000u, mature, 1'000u, intervalNs );
	REQUIRE( marginOnly.earlyCommit );
	CHECK( marginOnly.commitDeadlineNs == 19'000u );
	CHECK( marginOnly.presentLeadNs == 0u );

	mature.emaNs = 3'000u;
	const FixedRefreshCommitPlan_t underflow =
		plan_fixed_refresh_commit( 3'000u, mature, 1'000u, intervalNs );
	CHECK_FALSE( underflow.earlyCommit );
	CHECK( underflow.commitDeadlineNs == 0u );

	const FixedRefreshCommitPlan_t noInterval =
		plan_fixed_refresh_commit( 20'000u, mature, 1'000u, 0u );
	CHECK_FALSE( noInterval.earlyCommit );
}

TEST_CASE( "fixed-refresh commit lead is clamped to one vblank",
	"[framegen][deadline]" )
{
	constexpr PresentLeadState_t poisoned = {
		.emaNs = 30'000,
		.samples = k_uPresentLeadWarmupSamples,
	};
	constexpr FixedRefreshCommitPlan_t plan =
		plan_fixed_refresh_commit( 20'000u, poisoned, 1'000u, 8'000u );

	STATIC_REQUIRE( plan.earlyCommit );
	STATIC_REQUIRE( plan.presentLeadNs == 8'000u );
	STATIC_REQUIRE( plan.advanceNs == 9'000u );
	STATIC_REQUIRE( plan.commitDeadlineNs == 11'000u );
}

TEST_CASE( "fixed-refresh commit lead override waits for learner warmup",
	"[framegen][deadline]" )
{
	constexpr uint64_t intervalNs = 8'000u;
	constexpr uint64_t overrideLeadNs = 2'000u;
	constexpr PresentLeadState_t warming = {
		.emaNs = 6'000,
		.samples = k_uPresentLeadWarmupSamples - 1u,
	};
	constexpr FixedRefreshCommitPlan_t warmup = plan_fixed_refresh_commit(
		20'000u,
		apply_present_lead_override( warming, overrideLeadNs ),
		1'000u, intervalNs );
	STATIC_REQUIRE_FALSE( warmup.earlyCommit );

	constexpr PresentLeadState_t mature = {
		.emaNs = 6'000,
		.samples = k_uPresentLeadWarmupSamples,
	};
	constexpr FixedRefreshCommitPlan_t overridden = plan_fixed_refresh_commit(
		20'000u,
		apply_present_lead_override( mature, overrideLeadNs ),
		1'000u, intervalNs );
	STATIC_REQUIRE( overridden.earlyCommit );
	STATIC_REQUIRE( overridden.presentLeadNs == overrideLeadNs );
	STATIC_REQUIRE( overridden.commitDeadlineNs == 17'000u );

	constexpr FixedRefreshCommitPlan_t unchanged = plan_fixed_refresh_commit(
		20'000u, apply_present_lead_override( mature, 0u ),
		1'000u, intervalNs );
	STATIC_REQUIRE( unchanged.presentLeadNs == 6'000u );
}

TEST_CASE( "fixed-refresh admission budgets work to the KMS commit",
	"[framegen][deadline]" )
{
	constexpr PresentLeadState_t lead = {
		.emaNs = 3'000,
		.samples = k_uPresentLeadWarmupSamples,
	};
	constexpr FixedRefreshCommitPlan_t plan =
		plan_fixed_refresh_commit( 20'000u, lead, 1'000u, 8'000u );

	// The full-compositor wake is needlessly early for a generated-only flip.
	STATIC_REQUIRE_FALSE( deadline_cost_fits(
		3'000u, 12'000u, 10'000u, 0u ) );
	STATIC_REQUIRE( deadline_cost_fits(
		3'000u, plan.commitDeadlineNs, 10'000u, 0u ) );

	STATIC_REQUIRE( commit_reservation_only_shortfall(
		2'000u, 12'000u, 14'000u, 10'000u, 0u ) );
	STATIC_REQUIRE_FALSE( commit_reservation_only_shortfall(
		1'000u, 12'000u, 14'000u, 10'000u, 0u ) );
	STATIC_REQUIRE_FALSE( commit_reservation_only_shortfall(
		4'000u, 12'000u, 14'000u, 10'000u, 0u ) );
}

TEST_CASE( "generated commit never races a real present", "[framegen][deadline]" )
{
	CHECK( can_start_early_generated_commit( true, false, false ) );
	CHECK_FALSE( can_start_early_generated_commit( false, false, false ) );
	CHECK_FALSE( can_start_early_generated_commit( true, true, false ) );
	CHECK_FALSE( can_start_early_generated_commit( true, false, true ) );
	CHECK_FALSE( can_start_early_generated_commit( true, true, true ) );
}

TEST_CASE( "early generated commit yields to an imminent real frame", "[framegen][deadline]" )
{
	// now = 100'000; commit lead = 2'000.
	constexpr uint64_t kNow = 100'000u;
	constexpr uint64_t kLead = 2'000u;

	// Real predicted inside the lead window: skip.
	STATIC_REQUIRE( real_arrival_blocks_early_generated_commit(
		kNow + 1u, kNow, kLead, true ) );
	STATIC_REQUIRE( real_arrival_blocks_early_generated_commit(
		kNow + kLead, kNow, kLead, true ) );
	// Just outside it: generate.
	STATIC_REQUIRE_FALSE( real_arrival_blocks_early_generated_commit(
		kNow + kLead + 1u, kNow, kLead, true ) );

	// An overdue prediction is the empty vblank generation exists to fill and
	// must never suppress it.
	STATIC_REQUIRE_FALSE( real_arrival_blocks_early_generated_commit(
		kNow, kNow, kLead, true ) );
	STATIC_REQUIRE_FALSE( real_arrival_blocks_early_generated_commit(
		kNow - 50'000u, kNow, kLead, true ) );

	// Conservative: untrained cadence or any missing input never skips.
	STATIC_REQUIRE_FALSE( real_arrival_blocks_early_generated_commit(
		kNow + 1u, kNow, kLead, false ) );
	STATIC_REQUIRE_FALSE( real_arrival_blocks_early_generated_commit(
		0u, kNow, kLead, true ) );
	STATIC_REQUIRE_FALSE( real_arrival_blocks_early_generated_commit(
		kNow + 1u, 0u, kLead, true ) );
	STATIC_REQUIRE_FALSE( real_arrival_blocks_early_generated_commit(
		kNow + 1u, kNow, 0u, true ) );
}

TEST_CASE( "real-frame identity follows the commit id", "[framegen][deadline]" )
{
	int texA = 0, texB = 0;

	// Same texture object, new commit: this IS new content. Pointer identity
	// alone would have missed it (the reacquired-buffer case).
	STATIC_REQUIRE( is_new_real_frame_content( { 7u, &texA }, { 8u, &texA } ) );
	// Overlay-only repaint: same base commit re-presented, not a real frame.
	STATIC_REQUIRE_FALSE( is_new_real_frame_content( { 7u, &texA }, { 7u, &texA } ) );
	// Commit id wins over the pointer in both directions.
	STATIC_REQUIRE_FALSE( is_new_real_frame_content( { 7u, &texA }, { 7u, &texB } ) );
	STATIC_REQUIRE( is_new_real_frame_content( { 7u, &texA }, { 8u, &texB } ) );

	// Fallback to pointer identity when either side carries no commit id
	// (compositor-owned layers), including the first-ever record.
	STATIC_REQUIRE( is_new_real_frame_content( { 0u, nullptr }, { 0u, &texA } ) );
	STATIC_REQUIRE_FALSE( is_new_real_frame_content( { 0u, &texA }, { 0u, &texA } ) );
	STATIC_REQUIRE( is_new_real_frame_content( { 0u, &texA }, { 0u, &texB } ) );
	STATIC_REQUIRE( is_new_real_frame_content( { 7u, &texA }, { 0u, &texB } ) );

	// No base texture at all is never new content.
	STATIC_REQUIRE_FALSE( is_new_real_frame_content( { 7u, &texA }, { 9u, nullptr } ) );
}

TEST_CASE( "slot budget applies 0.85 to actual remaining time", "[framegen][deadline]" )
{
	CHECK( deadline_cost_fits( 8'500u, 20'000u, 10'000u, 0u ) );
	CHECK_FALSE( deadline_cost_fits( 8'501u, 20'000u, 10'000u, 0u ) );
	CHECK( deadline_cost_fits( 4'250u, 20'000u, 10'000u, 15'000u ) );
	CHECK_FALSE( deadline_cost_fits( 4'251u, 20'000u, 10'000u, 15'000u ) );
	CHECK_FALSE( deadline_cost_fits( 1u, 20'000u, 10'000u, 20'000u ) );
}

TEST_CASE( "native recovery preserves evidence across phase-short decisions",
	"[framegen][deadline][recovery]" )
{
	constexpr uint64_t intervalNs = 8'000'000u;
	constexpr uint64_t leadNs = 1'000'000u;
	constexpr uint64_t marginNs = 800'000u;
	constexpr uint64_t slotBudgetNs = fixed_refresh_slot_budget_ns(
		intervalNs, leadNs, marginNs );
	constexpr uint64_t shortRemainingBudgetNs = deadline_budget_ns( 2'000'000u );
	constexpr uint64_t currentCostNs = 2'500'000u;
	constexpr uint64_t richerCostNs = 4'500'000u;

	STATIC_REQUIRE( currentCostNs > shortRemainingBudgetNs );
	STATIC_REQUIRE( deadline_shortfall_is_phase_only(
		currentCostNs, k_uDeadlineMinSamples,
		shortRemainingBudgetNs, slotBudgetNs ) );
	STATIC_REQUIRE( deadline_recovery_headroom(
		currentCostNs, k_uDeadlineMinSamples,
		richerCostNs, k_uDeadlineMinSamples, slotBudgetNs ) );

	RecoveryState_t state;
	for ( uint32_t decision = 1u;
		decision < k_uDeadlineRecoveryHeadroomDecisions; decision++ )
	{
		const LadderRecoveryEvaluation_t evidence = evaluate_ladder_recovery(
			state, 1u, 0u,
			currentCostNs, k_uDeadlineMinSamples,
			richerCostNs, k_uDeadlineMinSamples, slotBudgetNs );
		state = evidence.state;
		CHECK_FALSE( evidence.tryRecover );
	}
	REQUIRE( state.streak == k_uDeadlineRecoveryHeadroomDecisions - 1u );

	// The renderer skips a phase-only shortfall without feeding it to the
	// recovery controller. Had it been fed, the short remaining budget would show
	// no headroom and wipe the accumulated streak — that is exactly what skipping
	// avoids. The next normal-phase decision then completes the streak.
	const LadderRecoveryEvaluation_t ifFedShortBudget = evaluate_ladder_recovery(
		state, 1u, 0u,
		currentCostNs, k_uDeadlineMinSamples,
		richerCostNs, k_uDeadlineMinSamples, shortRemainingBudgetNs );
	CHECK_FALSE( ifFedShortBudget.tryRecover );
	CHECK( ifFedShortBudget.state.streak == 0u );

	const RecoveryState_t afterShortDecision = state;
	const LadderRecoveryEvaluation_t recovered = evaluate_ladder_recovery(
		afterShortDecision, 1u, 0u,
		currentCostNs, k_uDeadlineMinSamples,
		richerCostNs, k_uDeadlineMinSamples, slotBudgetNs );
	CHECK( recovered.tryRecover );
}

TEST_CASE( "native deadline misses use two-miss hysteresis",
	"[framegen][deadline][recovery]" )
{
	constexpr uint64_t fittingCostNs = 3'000u;
	constexpr uint64_t slotBudgetNs = 5'000u;
	DeadlineMissState_t state;

	const DeadlineMissEvaluation_t first = evaluate_deadline_miss_hysteresis(
		state, true, false,
		fittingCostNs, k_uDeadlineMinSamples, slotBudgetNs );
	state = first.state;
	CHECK_FALSE( first.degrade );
	CHECK( first.skip );
	CHECK( deadline_miss_count( state.recentMisses ) == 1u );

	const DeadlineMissEvaluation_t clear = evaluate_deadline_miss_hysteresis(
		state, false, false,
		fittingCostNs, k_uDeadlineMinSamples, slotBudgetNs );
	state = clear.state;
	CHECK_FALSE( clear.degrade );
	CHECK_FALSE( clear.skip );

	const DeadlineMissEvaluation_t second = evaluate_deadline_miss_hysteresis(
		state, true, false,
		fittingCostNs, k_uDeadlineMinSamples, slotBudgetNs );
	CHECK( second.degrade );
	CHECK_FALSE( second.skip );

	const DeadlineMissEvaluation_t capacityFailure =
		evaluate_deadline_miss_hysteresis(
			{}, true, false,
			slotBudgetNs + 1u, k_uDeadlineMinSamples, slotBudgetNs );
	CHECK( capacityFailure.degrade );
	CHECK_FALSE( capacityFailure.skip );

	const DeadlineMissEvaluation_t hitch = evaluate_deadline_miss_hysteresis(
		{}, true, true,
		slotBudgetNs + 1u, k_uDeadlineMinSamples, slotBudgetNs );
	CHECK_FALSE( hitch.degrade );
	CHECK( hitch.skip );
	CHECK( hitch.state.recentMisses == 0u );
}

TEST_CASE( "deadline ladder recovers after sustained measured headroom",
	"[framegen][deadline][recovery]" )
{
	RecoveryState_t state;
	LadderRecoveryEvaluation_t evaluation;
	for ( uint32_t decision = 1u;
		decision <= k_uDeadlineRecoveryHeadroomDecisions; decision++ )
	{
		evaluation = evaluate_ladder_recovery(
			state, 3u, 0u,
			5'000u, k_uDeadlineMinSamples,
			9'000u, k_uDeadlineMinSamples,
			10'000u );
		state = evaluation.state;
		CHECK( evaluation.tryRecover
			== ( decision == k_uDeadlineRecoveryHeadroomDecisions ) );
	}
	CHECK( state.streak == k_uDeadlineRecoveryHeadroomDecisions );

	state = commit_ladder_recovery( state );
	CHECK( state.streak == 0u );
	CHECK( state.decisionsSinceClimb == 0u );
	CHECK( state.probationRemaining
		== k_uDeadlineRecoveryProbationDecisions );
	for ( uint32_t decision = 0u;
		decision < k_uDeadlineRecoveryHeadroomDecisions; decision++ )
	{
		const LadderRecoveryEvaluation_t probation =
			evaluate_ladder_recovery(
				state, 2u, 0u,
				5'000u, k_uDeadlineMinSamples,
				9'000u, k_uDeadlineMinSamples,
				10'000u );
		state = probation.state;
		CHECK_FALSE( probation.tryRecover );
	}
}

TEST_CASE( "deadline recovery target uses the richest fitting evidence",
	"[framegen][deadline][recovery]" )
{
	constexpr std::array<LadderRungCost_t, 5> costs = {{
		{ 7'000u, k_uDeadlineMinSamples },
		{ 6'000u, k_uDeadlineMinSamples },
		{ 0u, 0u },
		{ 4'000u, k_uDeadlineMinSamples },
		{ 3'000u, k_uDeadlineMinSamples },
	}};
	constexpr LadderRecoveryTarget_t target =
		select_ladder_recovery_target( costs, 4u, 10'000u );
	STATIC_REQUIRE( target.rung == 0u );
	STATIC_REQUIRE( target.evidence.costNs == 7'000u );
}

TEST_CASE( "deadline recovery target skips immature richer rungs",
	"[framegen][deadline][recovery]" )
{
	constexpr std::array<LadderRungCost_t, 4> costs = {{
		{ 2'000u, k_uDeadlineMinSamples - 1u },
		{ 4'000u, k_uDeadlineMinSamples },
		{ 3'000u, k_uDeadlineMinSamples },
		{ 2'000u, k_uDeadlineMinSamples },
	}};
	constexpr LadderRecoveryTarget_t target =
		select_ladder_recovery_target( costs, 3u, 5'000u );
	STATIC_REQUIRE( target.rung == 1u );
	STATIC_REQUIRE( target.evidence.samples == k_uDeadlineMinSamples );
}

TEST_CASE( "deadline recovery target falls back to one cold rung",
	"[framegen][deadline][recovery]" )
{
	constexpr std::array<LadderRungCost_t, 4> costs = {{
		{ 0u, 0u },
		{ 2'000u, k_uDeadlineMinSamples - 1u },
		{ 1'000u, 1u },
		{ 900u, k_uDeadlineMinSamples },
	}};
	constexpr LadderRecoveryTarget_t target =
		select_ladder_recovery_target( costs, 3u, 5'000u );
	STATIC_REQUIRE( target.rung == 2u );
	STATIC_REQUIRE( target.evidence.samples == 1u );
	STATIC_REQUIRE( deadline_recovery_headroom(
		2'500u, k_uDeadlineMinSamples,
		target.evidence.costNs, target.evidence.samples, 5'000u ) );
}

TEST_CASE( "deadline recovery uses mature non-adjacent native evidence",
	"[framegen][deadline][recovery]" )
{
	constexpr uint64_t currentCostNs = 1'780'000u;
	constexpr uint64_t budgetNs = 4'900'000u;
	constexpr std::array<LadderRungCost_t, 5> costs = {{
		{ 2'500'000u, k_uDeadlineMinSamples },
		{ 0u, 0u },
		{ 0u, 0u },
		{ 0u, 0u },
		{ currentCostNs, k_uDeadlineMinSamples },
	}};
	constexpr LadderRecoveryTarget_t target =
		select_ladder_recovery_target( costs, 4u, budgetNs );
	STATIC_REQUIRE( target.rung == 0u );

	RecoveryState_t state;
	LadderRecoveryEvaluation_t recovery;
	for ( uint32_t decision = 0u;
		decision < k_uDeadlineRecoveryHeadroomDecisions; decision++ )
	{
		recovery = evaluate_ladder_recovery(
			state, 4u, 0u, currentCostNs, k_uDeadlineMinSamples,
			target.evidence.costNs, target.evidence.samples, budgetNs );
		state = recovery.state;
	}
	CHECK( recovery.tryRecover );
	CHECK( ( recovery.tryRecover ? target.rung : 4u ) == 0u );
}

TEST_CASE( "deadline ladder recovery requires large current-rung headroom",
	"[framegen][deadline][recovery]" )
{
	RecoveryState_t knownState;
	RecoveryState_t coldState;
	for ( uint32_t decision = 0u;
		decision < k_uDeadlineRecoveryHeadroomDecisions * 2u; decision++ )
	{
		const LadderRecoveryEvaluation_t known = evaluate_ladder_recovery(
			knownState, 2u, 0u,
			7'001u, k_uDeadlineMinSamples,
			9'000u, k_uDeadlineMinSamples,
			10'000u );
		knownState = known.state;
		CHECK_FALSE( known.tryRecover );

		const LadderRecoveryEvaluation_t cold = evaluate_ladder_recovery(
			coldState, 2u, 0u,
			5'001u, k_uDeadlineMinSamples,
			0u, 0u, 10'000u );
		coldState = cold.state;
		CHECK_FALSE( cold.tryRecover );
	}

	CHECK( deadline_recovery_headroom(
		5'000u, k_uDeadlineMinSamples, 0u, 0u, 10'000u ) );
	CHECK_FALSE( deadline_recovery_headroom(
		5'000u, k_uDeadlineMinSamples,
		10'001u, k_uDeadlineMinSamples, 10'000u ) );
}

TEST_CASE( "deadline ladder miss resets the recovery streak",
	"[framegen][deadline][recovery]" )
{
	RecoveryState_t state;
	for ( uint32_t decision = 1u;
		decision < k_uDeadlineRecoveryHeadroomDecisions; decision++ )
	{
		const LadderRecoveryEvaluation_t evaluation =
			evaluate_ladder_recovery(
				state, 1u, 0u,
				5'000u, k_uDeadlineMinSamples,
				9'000u, k_uDeadlineMinSamples,
				10'000u );
		state = evaluation.state;
		CHECK_FALSE( evaluation.tryRecover );
	}
	CHECK( state.streak == k_uDeadlineRecoveryHeadroomDecisions - 1u );

	state = note_ladder_recovery_failure( state );
	CHECK( state.streak == 0u );
	for ( uint32_t decision = 1u;
		decision < k_uDeadlineRecoveryHeadroomDecisions; decision++ )
	{
		const LadderRecoveryEvaluation_t evaluation =
			evaluate_ladder_recovery(
				state, 1u, 0u,
				5'000u, k_uDeadlineMinSamples,
				9'000u, k_uDeadlineMinSamples,
				10'000u );
		state = evaluation.state;
		CHECK_FALSE( evaluation.tryRecover );
	}
	const LadderRecoveryEvaluation_t recovered = evaluate_ladder_recovery(
		state, 1u, 0u,
		5'000u, k_uDeadlineMinSamples,
		9'000u, k_uDeadlineMinSamples,
		10'000u );
	CHECK( recovered.tryRecover );
}

TEST_CASE( "deadline ladder recovery doubles backoff after a flap",
	"[framegen][deadline][recovery]" )
{
	RecoveryState_t state = commit_ladder_recovery( RecoveryState_t{} );
	state = note_ladder_recovery_degradation( state );
	CHECK( state.backoffDecisions
		== k_uDeadlineRecoveryBaseBackoffDecisions * 2u );
	CHECK( state.decisionsSinceClimb == 0u );

	for ( uint32_t decision = 1u;
		decision < state.backoffDecisions; decision++ )
	{
		const LadderRecoveryEvaluation_t evaluation =
			evaluate_ladder_recovery(
				state, 1u, 0u,
				5'000u, k_uDeadlineMinSamples,
				9'000u, k_uDeadlineMinSamples,
				10'000u );
		state = evaluation.state;
		CHECK_FALSE( evaluation.tryRecover );
	}
	const LadderRecoveryEvaluation_t recovered = evaluate_ladder_recovery(
		state, 1u, 0u,
		5'000u, k_uDeadlineMinSamples,
		9'000u, k_uDeadlineMinSamples,
		10'000u );
	CHECK( recovered.tryRecover );
}

TEST_CASE( "deadline ladder reports a blocked recovery threshold once per backoff",
	"[framegen][deadline][recovery]" )
{
	RecoveryState_t state = {
		.backoffDecisions = k_uDeadlineRecoveryHeadroomDecisions * 2u,
	};
	for ( uint32_t decision = 1u;
		decision < k_uDeadlineRecoveryHeadroomDecisions; decision++ )
	{
		const LadderRecoveryEvaluation_t evaluation =
			evaluate_ladder_recovery(
				state, 1u, 0u,
				5'000u, k_uDeadlineMinSamples,
				9'000u, k_uDeadlineMinSamples,
				10'000u );
		state = evaluation.state;
		CHECK_FALSE( evaluation.reportBlockedThreshold );
	}

	LadderRecoveryEvaluation_t blocked = evaluate_ladder_recovery(
		state, 1u, 0u,
		5'000u, k_uDeadlineMinSamples,
		9'000u, k_uDeadlineMinSamples,
		10'000u );
	state = blocked.state;
	CHECK_FALSE( blocked.tryRecover );
	CHECK( blocked.reportBlockedThreshold );

	state = reset_ladder_recovery_streak( state );
	for ( uint32_t decision = 0u;
		decision < k_uDeadlineRecoveryHeadroomDecisions; decision++ )
	{
		blocked = evaluate_ladder_recovery(
			state, 1u, 1u,
			5'000u, k_uDeadlineMinSamples,
			9'000u, k_uDeadlineMinSamples,
			10'000u );
		state = blocked.state;
		CHECK_FALSE( blocked.reportBlockedThreshold );
	}

	state = note_ladder_recovery_degradation( state );
	CHECK_FALSE( state.blockedThresholdReported );
}

TEST_CASE( "deadline ladder recovery backoff caps and relaxes on probation",
	"[framegen][deadline][recovery]" )
{
	RecoveryState_t state = {
		.backoffDecisions = 1'440u,
		.probationRemaining = 1u,
	};
	state = note_ladder_recovery_degradation( state );
	CHECK( state.backoffDecisions
		== k_uDeadlineRecoveryMaxBackoffDecisions );
	state = commit_ladder_recovery( state );
	state = note_ladder_recovery_degradation( state );
	CHECK( state.backoffDecisions
		== k_uDeadlineRecoveryMaxBackoffDecisions );

	state.backoffDecisions = k_uDeadlineRecoveryBaseBackoffDecisions * 2u;
	state.probationRemaining = 1u;
	state = advance_ladder_recovery_decision( state, true );
	CHECK( state.backoffDecisions
		== k_uDeadlineRecoveryBaseBackoffDecisions );
	CHECK( state.probationRemaining == 0u );
}

TEST_CASE( "deadline ladder scene reset restores base recovery backoff",
	"[framegen][deadline][recovery]" )
{
	RecoveryState_t state = {
		.streak = 47u,
		.backoffDecisions = k_uDeadlineRecoveryMaxBackoffDecisions,
		.probationRemaining = 12u,
		.decisionsSinceClimb = 400u,
	};
	state = {};
	CHECK( state.streak == 0u );
	CHECK( state.backoffDecisions
		== k_uDeadlineRecoveryBaseBackoffDecisions );
	CHECK( state.probationRemaining == 0u );
	CHECK( state.decisionsSinceClimb == 0u );
}

namespace
{

constexpr uint64_t k_ulRepaintPeriodNs = 16'666'667u;

// One vblank-timer wake's sampled inputs; framegen off unless a test says so.
RepaintInputs_t repaint_inputs( bool overlayPending, uint64_t forcedRequests, uint64_t period )
{
	return {
		.overlayPending = overlayPending,
		.forcedRequests = forcedRequests,
		.targetVBlankNs = period * k_ulRepaintPeriodNs,
		.vblank = true,
	};
}

bool repaint_state_fresh( const RepaintRequestState_t &state )
{
	return state.yields == 0u && state.yieldVBlankNs == 0u
		&& state.heldForcedRequests == 0u && state.refreshMisses == 0u;
}

// Drives the production step exactly as steamcompmgr does: the previous
// iteration's log is exchanged out, folded once, and the new log is stored at
// the end of the iteration.
struct RepaintLoop_t
{
	RepaintRequestState_t state;
	RepaintIteration_t last;

	RepaintStep_t begin( const RepaintInputs_t &in )
	{
		const RepaintStep_t step = step_repaint_requests( state, std::exchange( last, {} ), in );
		state = step.state;
		return step;
	}
	void end( const RepaintIteration_t &iteration ) { last = iteration; }
};

// What the presenter reports for the production flip selection: the queue
// front (or the hold) is an older image; refresh and snap flip the composite.
constexpr FlipVerdict_t flip_verdict( BidirFlip_t flip )
{
	return flip == BidirFlip_t::QueueFront || flip == BidirFlip_t::QueueBehind
		|| flip == BidirFlip_t::ReplaceBehind ? FlipVerdict_t::Older : FlipVerdict_t::Live;
}

// The bidir presenter at step level, driven by the production select_bidir_flip:
// what the flip substitution does for each BidirFlip_t, and what the consume
// path does on a vblank without a composite. No rendervulkan code is linked into
// these tests, so this model is the integration test of the repaint step and
// the flip selection. Entries hold a composite seq (0: generated) and a target
// vblank; bidir_front_action presents the front once due and drops late
// interpolations.
struct BidirSim_t
{
	struct Entry_t
	{
		uint64_t seq;
		int target;
		bool behind = false;	// queued behind retained content (QueueBehind)
	};
	std::deque<Entry_t> queue;
	uint64_t lastSeq = 0;
	bool timelineStarted = false;
	size_t snapped = 0;	// queued entries a live snap dropped

	struct Paint_t
	{
		uint64_t seq;		// this paint's composite, 0 for the consume path
		uint64_t shown;		// the composite put on screen, 0 for a generated frame or a hold
		PaintReport_t report;	// what the presenter reports to steamcompmgr
		uint64_t superseded = 0;	// the queued-behind composite this one replaced
	};

	uint64_t lastFrontNs = 0;	// the opportunity the last front was shown at

	static constexpr uint64_t opportunity_ns( int v ) { return uint64_t( v + 1 ) * k_ulRepaintPeriodNs; }

	uint64_t present_front( int v )
	{
		while ( !queue.empty() )
		{
			const Entry_t &front = queue.front();
			const BidirFrontAction_t action = bidir_front_action( {
				.targetNs = opportunity_ns( front.target ),
				.opportunityNs = opportunity_ns( v ),
				.lastFrontOpportunityNs = lastFrontNs,
				.vblankIntervalNs = k_ulRepaintPeriodNs,
				.real = front.seq != 0u,
				.completed = true,
			} );
			if ( action == BidirFrontAction_t::Wait )
				return 0u;
			const uint64_t seq = front.seq;
			queue.pop_front();
			if ( action == BidirFrontAction_t::Present )
			{
				lastFrontNs = opportunity_ns( v );
				return seq;
			}
		}
		return 0u;
	}

	// A composite of the live layer stack at vblank v: a new base (queued gap
	// vblanks out behind its interpolations when the pair was planned) or a
	// re-render of the newest base.
	Paint_t composite( int v, bool newBase, bool planned, int gap )
	{
		const uint64_t seq = ++lastSeq;
		const bool queuedReal = newBase && planned;
		if ( queuedReal )
		{
			for ( int i = 1; i < gap; i++ )
				queue.push_back( { 0u, v + i } );
			queue.push_back( { seq, v + gap } );
		}
		const BidirFlip_t flip = select_bidir_flip( { .queuedReal = queuedReal, .sameBase = !newBase,
			.queueEmpty = queue.empty(), .timelineStarted = timelineStarted,
			.tailQueuedBehind = !queue.empty() && queue.back().behind } );
		timelineStarted = true;
		// Held: the presenter shows the queue front, which is this composite
		// once its replaced or queued entry is the due front.
		const auto held = [&]( uint64_t superseded ) {
			const uint64_t shown = present_front( v );
			return Paint_t{ seq, shown,
				{ shown == seq ? FlipVerdict_t::Live : FlipVerdict_t::Older, seq, shown }, superseded };
		};
		switch ( flip )
		{
			case BidirFlip_t::ReplaceBehind:
				return held( std::exchange( queue.back().seq, seq ) );
			case BidirFlip_t::QueueBehind:
				queue.push_back( { seq, std::max( v, queue.back().target ) + 1, true } );
				[[fallthrough]];
			case BidirFlip_t::QueueFront:
				return held( 0u );
			case BidirFlip_t::SnapLive:
				snapped += queue.size();
				queue.clear();
				[[fallthrough]];
			case BidirFlip_t::RefreshLive:
				break;
		}
		// Flipped live: its own composite is newer than any carrier, nothing to report.
		return { seq, seq, { FlipVerdict_t::Live, seq } };
	}

	Paint_t consume( int v )
	{
		const uint64_t shown = present_front( v );
		return { 0u, shown, { .presented = shown } };
	}
};

// Feeds one sim paint to the iteration the way steamcompmgr does.
void add_sim_paint( RepaintIteration_t &iteration, const BidirSim_t::Paint_t &paint, bool forcedPaint )
{
	iteration.add_paint( classify_paint_output( paint.seq == 0u, false, true, paint.report.verdict ),
		PaintOutcome_t::Presented, forcedPaint, paint.report );
}

RepaintInputs_t bidir_inputs( bool overlayPending, uint64_t forcedRequests, int v, const BidirSim_t &sim )
{
	RepaintInputs_t in = repaint_inputs( overlayPending, forcedRequests, uint64_t( v + 1 ) );
	in.bidirActive = true;
	in.bidirQueueDrained = sim.queue.empty();
	return in;
}

// A bidir composite that latched no new base (idle game, overlay-only change).
constexpr FlipVerdict_t idle_base_flip( bool queueDrained )
{
	return flip_verdict( select_bidir_flip( {
		.queuedReal = false, .sameBase = true, .queueEmpty = queueDrained, .timelineStarted = true } ) );
}

} // namespace

TEST_CASE( "bidir flip selection keeps the ordered timeline and refreshes only a drained queue", "[framegen][deadline][repaint]" )
{
	struct Row_t { BidirFlipInputs_t in; BidirFlip_t flip; };
	using enum BidirFlip_t;
	// { queuedReal, sameBase, queueEmpty, timelineStarted }
	constexpr std::array<Row_t, 16> k_Rows = { {
		{ { false, false, false, false }, SnapLive },
		{ { false, false, false, true }, QueueBehind },	// cut/focus change behind queued reals
		{ { false, false, true, false }, SnapLive },
		{ { false, false, true, true }, SnapLive },	// prime after an invalidation
		{ { false, true, false, false }, QueueFront },
		{ { false, true, false, true }, QueueFront },
		{ { false, true, true, false }, SnapLive },
		{ { false, true, true, true }, RefreshLive },
		{ { true, false, false, false }, QueueFront },
		{ { true, false, false, true }, QueueFront },
		{ { true, false, true, false }, QueueFront },
		{ { true, false, true, true }, QueueFront },
		{ { true, true, false, false }, QueueFront },
		{ { true, true, false, true }, QueueFront },
		{ { true, true, true, false }, QueueFront },
		{ { true, true, true, true }, QueueFront },
	} };
	STATIC_REQUIRE( std::ranges::all_of( k_Rows, []( const Row_t &row ) {
		return select_bidir_flip( row.in ) == row.flip;
	} ) );
	// Over a drained queue a composite that queued nothing is never held,
	// whether it re-rendered the base on screen or latched a new one.
	STATIC_REQUIRE( std::ranges::all_of( k_Rows, []( const Row_t &row ) {
		return row.in.queuedReal || !row.in.queueEmpty
			|| ( select_bidir_flip( row.in ) != QueueFront && select_bidir_flip( row.in ) != QueueBehind );
	} ) );
	// No composite of a new base is left unqueued behind the front.
	STATIC_REQUIRE( std::ranges::all_of( k_Rows, []( const Row_t &row ) {
		return row.in.sameBase || row.in.queuedReal || select_bidir_flip( row.in ) != QueueFront;
	} ) );
}

TEST_CASE( "bidir flip selection replaces a queued-behind tail", "[framegen][deadline][repaint]" )
{
	// A new base that queued nothing replaces the queued-behind tail it
	// supersedes instead of queueing another entry; every other row is as above.
	using enum BidirFlip_t;
	constexpr auto replaces_only_queue_behind = []() {
		for ( uint32_t bits = 0u; bits < 16u; bits++ )
		{
			BidirFlipInputs_t in = { bool( bits & 8u ), bool( bits & 4u ), bool( bits & 2u ), bool( bits & 1u ) };
			const BidirFlip_t untagged = select_bidir_flip( in );
			in.tailQueuedBehind = true;
			const BidirFlip_t tagged = select_bidir_flip( in );
			if ( tagged != ( untagged == QueueBehind ? ReplaceBehind : untagged ) )
				return false;
		}
		return true;
	};
	STATIC_REQUIRE( replaces_only_queue_behind() );
	STATIC_REQUIRE( select_bidir_flip( { .queuedReal = false, .sameBase = false, .queueEmpty = false,
		.timelineStarted = true, .tailQueuedBehind = true } ) == ReplaceBehind );
	STATIC_REQUIRE( flip_verdict( ReplaceBehind ) == FlipVerdict_t::Older );

	// So does one behind a queue whose reals fill the ring: the queue never
	// grows past the ring, and the replaced real is the one dropped.
	constexpr auto replaces_when_reals_fill = []() {
		for ( uint32_t bits = 0u; bits < 16u; bits++ )
		{
			BidirFlipInputs_t in = { bool( bits & 8u ), bool( bits & 4u ), bool( bits & 2u ), bool( bits & 1u ) };
			const BidirFlip_t untagged = select_bidir_flip( in );
			in.realsFill = true;
			if ( select_bidir_flip( in ) != ( untagged == QueueBehind ? ReplaceBehind : untagged ) )
				return false;
		}
		return true;
	};
	STATIC_REQUIRE( replaces_when_reals_fill() );
}

TEST_CASE( "paint output classification trusts the presenter first", "[framegen][deadline][repaint]" )
{
	// ( generatedPaint, baseLayer, bidirActive, flip )
	constexpr auto all_args = []( FlipVerdict_t flip, PaintOutput_t expected ) {
		for ( int i = 0; i < 8; i++ )
		{
			if ( classify_paint_output( i & 1, i & 2, i & 4, flip ) != expected )
				return false;
		}
		return true;
	};
	STATIC_REQUIRE( all_args( FlipVerdict_t::Live, PaintOutput_t::Live ) );
	STATIC_REQUIRE( all_args( FlipVerdict_t::Older, PaintOutput_t::Held ) );
	// No flip under bidir: the consume path presented a queued front.
	STATIC_REQUIRE( classify_paint_output( false, false, true, FlipVerdict_t::NotFlipped ) == PaintOutput_t::Held );
	STATIC_REQUIRE( classify_paint_output( true, false, true, FlipVerdict_t::NotFlipped ) == PaintOutput_t::Held );
	STATIC_REQUIRE( classify_paint_output( true, true, false, FlipVerdict_t::NotFlipped ) == PaintOutput_t::Live );
	STATIC_REQUIRE( classify_paint_output( true, false, false, FlipVerdict_t::NotFlipped ) == PaintOutput_t::Generated );
	STATIC_REQUIRE( classify_paint_output( false, false, false, FlipVerdict_t::NotFlipped ) == PaintOutput_t::Live );
	STATIC_REQUIRE( classify_paint_output( false, true, false, FlipVerdict_t::NotFlipped ) == PaintOutput_t::Live );
}

TEST_CASE( "bidir idle base services a held request with one live composite", "[framegen][deadline][repaint]" )
{
	struct Scenario_t
	{
		uint64_t forced;		// forced snapshot raised with the overlay request
		int drainVBlank;		// first vblank whose iteration starts drained
		bool raiseDuringService;	// a force is raised while the service paint runs
		int requestComposites;		// composites the requests drive in total
	};
	constexpr int k_nVBlanks = 131;
	const Scenario_t k_Scenarios[] = {
		{ 7u, 9, false, 2 },	// one forced commit at V0, one refresh at V9
		{ 0u, 9, false, 1 },	// overlay only
		{ 7u, 0, false, 1 },	// already drained at the raise
		{ 7u, 9, true, 3 },	// the force raised mid-service gets one more refresh
	};
	for ( const Scenario_t &scenario : k_Scenarios )
	{
		CAPTURE( scenario.forced, scenario.drainVBlank, scenario.raiseDuringService );
		std::atomic<uint64_t> forced{ scenario.forced };
		bool overlay = true;	// hasRepaintNonBasePlane
		RepaintLoop_t loop;
		int nRequestComposites = 0;
		int nOverlayComposites = 0;
		int nServiceVBlank = -1;
		int nLastComposite = -1;
		for ( int v = 0; v < k_nVBlanks; v++ )
		{
			const bool bDrained = v >= scenario.drainVBlank;
			RepaintInputs_t in = repaint_inputs( overlay, forced.load(), uint64_t( v + 1 ) );
			in.bidirActive = true;
			in.bidirQueueDrained = bDrained;
			const RepaintStep_t step = loop.begin( in );
			RepaintIteration_t iteration = step.iteration;
			const bool bForcedPaint = step.plan.forcedDrivesPaint;
			if ( step.plan.overlayDrivesPaint || bForcedPaint )
			{
				nRequestComposites++;
				nOverlayComposites += step.plan.overlayDrivesPaint;
				nLastComposite = v;
				iteration.add_paint( classify_paint_output( false, false, true, idle_base_flip( bDrained ) ),
					PaintOutcome_t::Presented, bForcedPaint );
				if ( scenario.raiseDuringService && iteration.retiredOverlay && nServiceVBlank < 0 )
					forced.fetch_add( 1u );
			}
			else if ( !bDrained )
			{
				// The queue drains through the consume path.
				iteration.add_paint( classify_paint_output( true, false, true, FlipVerdict_t::NotFlipped ),
					PaintOutcome_t::Presented, false );
			}
			if ( iteration.ended )
			{
				if ( iteration.retiredOverlay )
				{
					overlay = false;
					if ( nServiceVBlank < 0 )
						nServiceVBlank = v;
				}
				acknowledge_forced_requests( forced, iteration );
			}
			loop.end( iteration );
		}
		CHECK( nRequestComposites == scenario.requestComposites );
		CHECK( nOverlayComposites == 1 );
		// Bound: serviced on the first vblank whose iteration starts drained.
		CHECK( nServiceVBlank == scenario.drainVBlank );
		CHECK( nLastComposite <= scenario.drainVBlank + 1 );
		CHECK_FALSE( overlay );
		CHECK( forced.load() == 0u );
		CHECK( repaint_state_fresh( loop.state ) );
	}
}

TEST_CASE( "bidir held forced request drives one commit per new snapshot", "[framegen][deadline][repaint]" )
{
	RepaintLoop_t loop;
	uint64_t ulForced = 7u;
	int nCommits = 0;
	for ( int v = 0; v < 120; v++ )
	{
		if ( v == 60 )
			ulForced++;	// screen sleep / HDR metadata raised during the hold
		RepaintInputs_t in = repaint_inputs( true, ulForced, uint64_t( v + 1 ) );
		in.bidirActive = true;
		const RepaintStep_t step = loop.begin( in );
		CHECK_FALSE( step.plan.overlayDrivesPaint );
		RepaintIteration_t iteration = step.iteration;
		if ( step.plan.forcedDrivesPaint )
		{
			nCommits++;
			iteration.add_paint( classify_paint_output( false, false, true, idle_base_flip( false ) ),
				PaintOutcome_t::Presented, true );
			CHECK( iteration.held );
			CHECK_FALSE( iteration.retiredForced );
		}
		loop.end( iteration );
	}
	CHECK( nCommits == 2 );
	CHECK( loop.state.heldForcedRequests == ulForced );
	CHECK( loop.state.refreshMisses == 0u );
}

TEST_CASE( "bidir refresh misses drop the held request within the cap", "[framegen][deadline][repaint]" )
{
	// A broken presenter holds every request-only composite over a drained
	// queue. The request drives k_uMaxBidirRefreshMisses composites, is dropped
	// on the next iteration, and nothing composites per vblank afterwards.
	{
		std::atomic<uint64_t> forced{ 3u };
		bool overlay = true;
		RepaintLoop_t loop;
		int nComposites = 0;
		int nDrops = 0;
		int nDropVBlank = -1;
		for ( int v = 0; v < 240; v++ )
		{
			RepaintInputs_t in = repaint_inputs( overlay, forced.load(), uint64_t( v + 1 ) );
			in.bidirActive = true;
			in.bidirQueueDrained = true;
			const RepaintStep_t step = loop.begin( in );
			RepaintIteration_t iteration = step.iteration;
			if ( step.plan.dropHeld )
			{
				nDrops++;
				nDropVBlank = v;
			}
			if ( step.plan.overlayDrivesPaint || step.plan.forcedDrivesPaint )
			{
				nComposites++;
				iteration.add_paint( classify_paint_output( false, false, true, FlipVerdict_t::Older ),
					PaintOutcome_t::Presented, step.plan.forcedDrivesPaint );
			}
			if ( iteration.retiredOverlay )
				overlay = false;
			acknowledge_forced_requests( forced, iteration );
			loop.end( iteration );
		}
		CHECK( nComposites == int( k_uMaxBidirRefreshMisses ) );
		CHECK( nDrops == 1 );
		CHECK( nDropVBlank == int( k_uMaxBidirRefreshMisses ) );
		CHECK_FALSE( overlay );
		CHECK( forced.load() == 0u );
		CHECK( repaint_state_fresh( loop.state ) );

		// A request raised after the drop is fresh: it drives again.
		RepaintInputs_t in = repaint_inputs( true, 0u, 241u );
		in.bidirActive = true;
		in.bidirQueueDrained = true;
		const RepaintStep_t step = loop.begin( in );
		CHECK( step.plan.overlayDrivesPaint );
		CHECK_FALSE( step.plan.dropHeld );
	}

	// A force raised during the hold is a new request: it keeps its one commit
	// and the drop leaves it pending.
	{
		std::atomic<uint64_t> forced{ 3u };
		RepaintLoop_t loop;
		int nCommits = 0;
		for ( int v = 0; v < 16; v++ )
		{
			if ( v == int( k_uMaxBidirRefreshMisses ) - 1 )
				forced.fetch_add( 1u );
			RepaintInputs_t in = repaint_inputs( false, forced.load(), uint64_t( v + 1 ) );
			in.bidirActive = true;
			in.bidirQueueDrained = true;
			const RepaintStep_t step = loop.begin( in );
			RepaintIteration_t iteration = step.iteration;
			if ( step.plan.forcedDrivesPaint )
			{
				nCommits++;
				iteration.add_paint( classify_paint_output( false, false, true, FlipVerdict_t::Older ),
					PaintOutcome_t::Presented, true );
			}
			acknowledge_forced_requests( forced, iteration );
			loop.end( iteration );
		}
		CHECK( forced.load() == 0u );
		CHECK( nCommits <= int( 2u * k_uMaxBidirRefreshMisses ) );
	}

	// Only bidir drops: once it is dormant the request drives and a live paint
	// services it, whatever misses it carries.
	{
		const RepaintRequestState_t capped = { .heldForcedRequests = 3u, .refreshMisses = k_uMaxBidirRefreshMisses };
		const RepaintIteration_t prev = { .in = repaint_inputs( true, 3u, 1u ) };
		const RepaintStep_t step = step_repaint_requests( capped, prev, repaint_inputs( true, 3u, 2u ) );
		REQUIRE( step.state.refreshMisses == k_uMaxBidirRefreshMisses );
		CHECK_FALSE( step.plan.dropHeld );
		REQUIRE( step.plan.overlayDrivesPaint );
		REQUIRE( step.plan.forcedDrivesPaint );
		RepaintIteration_t iteration = step.iteration;
		CHECK_FALSE( iteration.retiredOverlay );
		iteration.add_paint( classify_paint_output( false, false, false, FlipVerdict_t::NotFlipped ),
			PaintOutcome_t::Presented, true );
		CHECK( iteration.retiredOverlay );
		CHECK( iteration.retiredForced );
		CHECK_FALSE( iteration.request_remains() );
	}

	// A new base latched in a hitch is a legitimate hold, not a miss.
	{
		RepaintLoop_t loop;
		for ( int v = 0; v < 10; v++ )
		{
			RepaintInputs_t in = repaint_inputs( true, 0u, uint64_t( v + 1 ) );
			in.bidirActive = true;
			in.bidirQueueDrained = true;
			in.realRepaint = true;
			const RepaintStep_t step = loop.begin( in );
			CHECK( loop.state.refreshMisses == 0u );
			REQUIRE( step.plan.overlayDrivesPaint );
			RepaintIteration_t iteration = step.iteration;
			const FlipVerdict_t eFlip = flip_verdict( select_bidir_flip( {
				.queuedReal = true, .sameBase = false, .queueEmpty = true, .timelineStarted = true } ) );
			iteration.add_paint( classify_paint_output( false, false, true, eFlip ), PaintOutcome_t::Presented, false );
			CHECK( iteration.held );
			loop.end( iteration );
		}
		// A later idle drained iteration still services the request.
		RepaintInputs_t in = repaint_inputs( true, 0u, 11u );
		in.bidirActive = true;
		in.bidirQueueDrained = true;
		const RepaintStep_t step = loop.begin( in );
		CHECK( loop.state.refreshMisses == 0u );
		REQUIRE( step.plan.overlayDrivesPaint );
		RepaintIteration_t iteration = step.iteration;
		iteration.add_paint( classify_paint_output( false, false, true, idle_base_flip( true ) ),
			PaintOutcome_t::Presented, false );
		CHECK( iteration.retiredOverlay );
	}
}

TEST_CASE( "bidir retires a request only when the live composite was flipped", "[framegen][deadline][repaint]" )
{
	// GR6-1: a real paint whose flip presented an older image, and a consumed
	// queue front, both hold; neither retires.
	for ( const FlipVerdict_t eFlip : { FlipVerdict_t::Older, FlipVerdict_t::NotFlipped } )
	{
		RepaintLoop_t loop;
		RepaintInputs_t in = repaint_inputs( true, 4u, 1u );
		in.bidirActive = true;
		in.realRepaint = true;
		RepaintIteration_t iteration = loop.begin( in ).iteration;
		iteration.add_paint( classify_paint_output( eFlip == FlipVerdict_t::NotFlipped, false, true, eFlip ),
			PaintOutcome_t::Presented, true );
		CHECK( iteration.held );
		CHECK_FALSE( iteration.retiredOverlay );
		CHECK_FALSE( iteration.retiredForced );
		loop.end( iteration );

		in = repaint_inputs( true, 4u, 2u );
		in.bidirActive = true;
		const RepaintStep_t step = loop.begin( in );
		CHECK_FALSE( step.plan.overlayDrivesPaint );
		CHECK_FALSE( step.plan.forcedDrivesPaint );
	}

	// The same real paint flipped live retires.
	{
		RepaintIteration_t iteration;
		iteration.add_paint( classify_paint_output( false, false, true, FlipVerdict_t::Live ),
			PaintOutcome_t::Presented, true );
		CHECK( iteration.retiredOverlay );
		CHECK( iteration.retiredForced );
	}

	// r1 raised, held real paint, r2 merges into the same bool, the consume
	// path presents the front, the queue drains, a refresh retires both.
	RepaintLoop_t loop;
	bool overlay = true;	// hasRepaintNonBasePlane: r1 and r2 merge into it
	int nRetired = 0;
	int nRetireVBlank = -1;
	const auto run = [&]( int v, bool drained, bool realRepaint, bool generatedPaint ) {
		RepaintInputs_t in = repaint_inputs( overlay, 0u, uint64_t( v ) );
		in.bidirActive = true;
		in.bidirQueueDrained = drained;
		in.realRepaint = realRepaint;
		const RepaintStep_t step = loop.begin( in );
		RepaintIteration_t iteration = step.iteration;
		if ( realRepaint || generatedPaint || step.plan.overlayDrivesPaint )
		{
			const FlipVerdict_t eFlip = generatedPaint ? FlipVerdict_t::NotFlipped
				: flip_verdict( select_bidir_flip( { .queuedReal = realRepaint, .sameBase = !realRepaint,
					.queueEmpty = drained, .timelineStarted = true } ) );
			iteration.add_paint( classify_paint_output( generatedPaint, false, true, eFlip ),
				PaintOutcome_t::Presented, false );
		}
		if ( iteration.retiredOverlay )
		{
			overlay = false;
			nRetired++;
			nRetireVBlank = v;
		}
		loop.end( iteration );
	};
	run( 1, false, true, false );	// r1 + new base: queued, front flipped
	run( 2, false, false, false );	// r2 raised: no composite behind the queue
	run( 3, false, false, true );	// consume presents the queued front
	run( 4, true, false, false );	// drained: one refresh
	run( 5, true, false, false );
	CHECK( nRetired == 1 );
	CHECK( nRetireVBlank == 4 );
}

TEST_CASE( "bidir toggling inside an iteration is classified by the presenter", "[framegen][deadline][repaint]" )
{
	// Sampled inactive, but the flip ran with bidir active and held: no retire.
	{
		RepaintIteration_t iteration = RepaintLoop_t{}.begin( repaint_inputs( true, 2u, 1u ) ).iteration;
		iteration.add_paint( classify_paint_output( false, false, false, FlipVerdict_t::Older ),
			PaintOutcome_t::Presented, true );
		CHECK( iteration.held );
		CHECK_FALSE( iteration.retiredOverlay );
		CHECK_FALSE( iteration.retiredForced );
	}
	// Sampled active, but bidir went inactive before the flip: held, and the
	// next iteration (bidir inactive) drives the request.
	{
		RepaintLoop_t loop;
		RepaintInputs_t in = repaint_inputs( true, 2u, 1u );
		in.bidirActive = true;
		in.realRepaint = true;
		RepaintIteration_t iteration = loop.begin( in ).iteration;
		iteration.add_paint( classify_paint_output( false, false, true, FlipVerdict_t::NotFlipped ),
			PaintOutcome_t::Presented, true );
		CHECK( iteration.held );
		CHECK_FALSE( iteration.retiredOverlay );
		loop.end( iteration );
		const RepaintStep_t step = loop.begin( repaint_inputs( true, 2u, 2u ) );
		CHECK( step.plan.overlayDrivesPaint );
		CHECK( step.plan.forcedDrivesPaint );
	}
}

TEST_CASE( "framegen off keeps a forced request until the vblank paint that takes it", "[framegen][deadline][repaint]" )
{
	// Upstream consumes a force only on a vblank. An off-vblank VRR paint
	// presents real content but leaves the force for the next vblank.
	std::atomic<uint64_t> forced{ 3u };
	RepaintLoop_t loop;
	RepaintInputs_t in = repaint_inputs( true, forced.load(), 1u );
	in.vblank = false;
	in.realRepaint = true;
	RepaintStep_t step = loop.begin( in );
	CHECK_FALSE( step.plan.forcedDrivesPaint );
	RepaintIteration_t iteration = step.iteration;
	iteration.add_paint( classify_paint_output( false, false, false, FlipVerdict_t::NotFlipped ),
		PaintOutcome_t::Presented, step.plan.forcedDrivesPaint );
	CHECK( iteration.retiredOverlay );
	CHECK_FALSE( iteration.retiredForced );
	acknowledge_forced_requests( forced, iteration );
	CHECK( forced.load() == 3u );
	loop.end( iteration );

	step = loop.begin( repaint_inputs( false, forced.load(), 1u ) );
	REQUIRE( step.plan.forcedDrivesPaint );
	iteration = step.iteration;
	iteration.add_paint( classify_paint_output( false, false, false, FlipVerdict_t::NotFlipped ),
		PaintOutcome_t::Presented, step.plan.forcedDrivesPaint );
	CHECK( iteration.retiredForced );
	acknowledge_forced_requests( forced, iteration );
	CHECK( forced.load() == 0u );
}

TEST_CASE( "framegen off plans every request exactly as upstream", "[framegen][deadline][repaint]" )
{
	// With bidir and base layer off and nothing generated, the plan is the
	// upstream constant for every input, whatever paints come out of it. The
	// slot gate is never consulted with framegen off; only a pause closes it.
	RepaintLoop_t loop;
	uint64_t ulPeriod = 1u;
	for ( uint32_t uIgnored = 0; uIgnored < 64u; uIgnored++ )
	{
		for ( int i = 0; i < 96; i++ )
		{
			const bool bVBlank = i & 1, bOverlay = i & 2, bReal = i & 4;
			const uint64_t ulForced = ( i & 8 ) ? 5u : 0u;
			constexpr PaintOutcome_t k_Outcomes[] = { PaintOutcome_t::Presented, PaintOutcome_t::Deferred, PaintOutcome_t::Failed };
			const PaintOutcome_t eOutcome = k_Outcomes[ ( i >> 4 ) % 3 ];
			const bool bPaused = i >= 48;
			RepaintInputs_t in = repaint_inputs( bOverlay, ulForced, ulPeriod++ );
			in.vblank = bVBlank;
			in.realRepaint = bReal;
			in.ignoredOverlayRepaints = uIgnored;
			in.backendPaused = bPaused;
			const RepaintStep_t step = loop.begin( in );
			CHECK( step.plan.overlayDrivesPaint == bOverlay );
			CHECK( step.plan.forcedDrivesPaint == ( bVBlank && ulForced != 0u ) );
			CHECK_FALSE( step.plan.exhausted );
			CHECK( step.plan.generatedMayTakeSlot == !bPaused );
			CHECK_FALSE( step.plan.dropHeld );
			CHECK( repaint_state_fresh( loop.state ) );
			RepaintIteration_t iteration = step.iteration;
			if ( bReal || step.plan.overlayDrivesPaint || step.plan.forcedDrivesPaint )
				iteration.add_paint( classify_paint_output( false, false, false, FlipVerdict_t::NotFlipped ),
					eOutcome, step.plan.forcedDrivesPaint );
			loop.end( iteration );
		}
	}
}

TEST_CASE( "repaint plan gates the generated slot and the unselected paint", "[framegen][deadline][repaint]" )
{
	// The slot gate and the guard decision live in the step. A paused
	// backend never lets generated output take a slot.
	for ( uint32_t uYields = 0; uYields <= k_uMaxRepaintYields; uYields++ )
	for ( uint32_t uIgnored = 0; uIgnored <= 4u; uIgnored++ )
	for ( int i = 0; i < 16; i++ )
	{
		const bool bPending = i & 1, bBidir = i & 2, bBaseLayer = i & 4, bPaused = i & 8;
		if ( bBidir && bBaseLayer )
			continue;	// bidir excludes base-layer mode
		CAPTURE( uYields, uIgnored, bPending, bBidir, bBaseLayer, bPaused );
		const RepaintRequestState_t aged = { .yields = uYields, .yieldVBlankNs = k_ulRepaintPeriodNs };
		const RepaintIteration_t prev = { .in = repaint_inputs( true, 0u, 1u ) };
		RepaintInputs_t in = repaint_inputs( bPending, 0u, 2u );
		in.ignoredOverlayRepaints = uIgnored;
		in.bidirActive = bBidir;
		in.baseLayerActive = bBaseLayer;
		in.backendPaused = bPaused;
		const RepaintStep_t step = step_repaint_requests( aged, prev, in );
		REQUIRE( step.state.yields == uYields );
		// The shared VRR throttle bound: yields plus ignored cycles reach the cap.
		const bool bExhausted = uYields != 0u && uYields + uIgnored >= k_uMaxRepaintYields;
		CHECK( step.plan.exhausted == bExhausted );
		CHECK( step.plan.generatedMayTakeSlot
			== ( !bPaused && ( !bPending || bBidir || bBaseLayer || !bExhausted ) ) );
		const UnselectedPaint_t eExpected = bBidir ? UnselectedPaint_t::Paint
			: bExhausted ? UnselectedPaint_t::DiscardPrediction : UnselectedPaint_t::Yield;
		CHECK( step.plan.unselectedPaint == eExpected );

		RepaintIteration_t iteration = step.iteration;
		CHECK( iteration.take_unselected_paint( step.plan ) == eExpected );
		CHECK( iteration.yielded == ( eExpected == UnselectedPaint_t::Yield ) );
	}
}

TEST_CASE( "a repaint log is folded exactly once", "[framegen][deadline][repaint]" )
{
	// steamcompmgr exchanges the log out, so a skipped store can never fold
	// the same yield or miss twice.
	RepaintLoop_t loop;
	RepaintInputs_t in = repaint_inputs( true, 0u, 1u );
	in.bidirActive = true;
	in.bidirQueueDrained = true;
	RepaintIteration_t iteration = loop.begin( in ).iteration;
	iteration.add_paint( PaintOutput_t::Held, PaintOutcome_t::Presented, false );
	loop.end( iteration );
	CHECK( loop.begin( in ).state.refreshMisses == 1u );
	CHECK( loop.begin( in ).state.refreshMisses == 0u );

	RepaintLoop_t yields;
	iteration = yields.begin( repaint_inputs( true, 0u, 1u ) ).iteration;
	iteration.add_paint( PaintOutput_t::Generated, PaintOutcome_t::Presented, false );
	yields.end( iteration );
	CHECK( yields.begin( repaint_inputs( true, 0u, 2u ) ).state.yields == 1u );
	CHECK( yields.begin( repaint_inputs( true, 0u, 3u ) ).state.yields == 0u );
}

TEST_CASE( "deferred paints change nothing while failed paints end the frame", "[framegen][deadline][repaint]" )
{
	constexpr uint64_t ulForced = 5u;
	const RepaintRequestState_t aged = {
		.yields = 2u,
		.yieldVBlankNs = 2u * k_ulRepaintPeriodNs,
		.heldForcedRequests = ulForced,
		.refreshMisses = 1u,
	};
	const auto fold = [&]( const RepaintIteration_t &iteration ) {
		return step_repaint_requests( aged, iteration, repaint_inputs( true, ulForced, 4u ) ).state;
	};

	for ( const PaintOutput_t eOutput : { PaintOutput_t::Live, PaintOutput_t::Generated, PaintOutput_t::Held } )
	{
		RepaintIteration_t iteration = { .in = repaint_inputs( true, ulForced, 3u ) };
		iteration.add_paint( eOutput, PaintOutcome_t::Deferred, true );
		CHECK_FALSE( iteration.ended );
		CHECK_FALSE( iteration.retiredOverlay );
		CHECK_FALSE( iteration.retiredForced );
		CHECK_FALSE( iteration.yielded );
		CHECK_FALSE( iteration.held );
		const RepaintRequestState_t next = fold( iteration );
		CHECK( next.yields == aged.yields );
		CHECK( next.yieldVBlankNs == aged.yieldVBlankNs );
		CHECK( next.heldForcedRequests == aged.heldForcedRequests );
		CHECK( next.refreshMisses == aged.refreshMisses );
	}

	RepaintIteration_t iteration = { .in = repaint_inputs( true, ulForced, 3u ) };
	iteration.add_paint( PaintOutput_t::Live, PaintOutcome_t::Failed, true );
	CHECK( iteration.ended );
	CHECK( iteration.retiredOverlay );
	CHECK( iteration.retiredForced );
	CHECK( repaint_state_fresh( fold( iteration ) ) );

	// A failed paint drops its requests as upstream does, whatever it
	// was for; it is never held, never a yield and never a refresh miss.
	for ( const PaintOutput_t eOutput : { PaintOutput_t::Live, PaintOutput_t::Generated, PaintOutput_t::Held } )
	{
		iteration = { .in = repaint_inputs( true, ulForced, 3u ) };
		iteration.in.bidirActive = eOutput == PaintOutput_t::Held;
		iteration.in.bidirQueueDrained = eOutput == PaintOutput_t::Held;
		iteration.add_paint( eOutput, PaintOutcome_t::Failed, true );
		CHECK( iteration.ended );
		CHECK( iteration.retiredOverlay );
		CHECK( iteration.retiredForced );
		CHECK_FALSE( iteration.yielded );
		CHECK_FALSE( iteration.held );
		CHECK( repaint_state_fresh( fold( iteration ) ) );

		// Not the vblank paint that took the force: the force keeps its age.
		iteration = { .in = repaint_inputs( true, ulForced, 3u ) };
		iteration.in.bidirActive = eOutput == PaintOutput_t::Held;
		iteration.in.bidirQueueDrained = eOutput == PaintOutput_t::Held;
		iteration.add_paint( eOutput, PaintOutcome_t::Failed, false );
		CHECK( iteration.retiredOverlay );
		CHECK_FALSE( iteration.retiredForced );
		const RepaintRequestState_t next = fold( iteration );
		CHECK( next.yields == aged.yields );
		CHECK( next.heldForcedRequests == aged.heldForcedRequests );
		CHECK( next.refreshMisses == aged.refreshMisses );
	}

	// Multi-focus: a deferred focus does not stop another from servicing.
	iteration = { .in = repaint_inputs( true, ulForced, 3u ) };
	iteration.add_paint( PaintOutput_t::Live, PaintOutcome_t::Deferred, true );
	iteration.add_paint( PaintOutput_t::Live, PaintOutcome_t::Presented, true );
	CHECK( iteration.ended );
	CHECK( iteration.retiredOverlay );
	CHECK( iteration.retiredForced );
}

TEST_CASE( "repaint yields count once per vblank-timer period at x4 early commit", "[framegen][deadline][repaint]" )
{
	// The request is raised after R0 painted on timer key K0; G1..G3 follow.
	// An early generated commit runs on a deadline wake that carries the key of
	// the preceding timer wake, and a timer wake between commits withholds the
	// request for the next prediction. Either wake may come first in a period.
	struct Wake_t
	{
		uint64_t period;
		bool generatedCommit;
	};
	constexpr uint64_t K0 = 10u;
	const std::vector<Wake_t> deadlineFirst = {
		{ K0, true }, { K0 + 1u, false }, { K0 + 1u, true }, { K0 + 2u, false }, { K0 + 2u, true },
	};
	const std::vector<Wake_t> vblankFirst = {
		{ K0 + 1u, false }, { K0 + 1u, true }, { K0 + 2u, false }, { K0 + 2u, true }, { K0 + 3u, false }, { K0 + 3u, true },
	};
	for ( const auto &wakes : { deadlineFirst, vblankFirst } )
	{
		RepaintLoop_t loop;
		for ( const Wake_t &wake : wakes )
		{
			const RepaintStep_t step = loop.begin( repaint_inputs( true, 0u, wake.period ) );
			// Neither the arbiter nor the guard may discard a prediction.
			REQUIRE( step.plan.generatedMayTakeSlot );
			REQUIRE( step.plan.unselectedPaint == UnselectedPaint_t::Yield );
			RepaintIteration_t iteration = step.iteration;
			if ( wake.generatedCommit )
				iteration.add_paint( classify_paint_output( true, false, false, FlipVerdict_t::NotFlipped ),
					PaintOutcome_t::Presented, false );
			else
				CHECK( iteration.take_unselected_paint( step.plan ) == UnselectedPaint_t::Yield );
			loop.end( iteration );
		}

		// R1 (or the live paint once nothing is pending) on the next timer
		// period retires the request: four slots after it was raised.
		const uint64_t ulRetirePeriod = wakes.back().period + 1u;
		CHECK( ulRetirePeriod - wakes.front().period + 1u <= k_uMaxRepaintYields );
		const RepaintStep_t step = loop.begin( repaint_inputs( true, 0u, ulRetirePeriod ) );
		CHECK( step.state.yields == 3u );
		CHECK_FALSE( step.plan.exhausted );
		RepaintIteration_t real = step.iteration;
		real.add_paint( classify_paint_output( false, false, false, FlipVerdict_t::NotFlipped ), PaintOutcome_t::Presented, false );
		loop.end( real );
		CHECK( repaint_state_fresh( loop.begin( repaint_inputs( false, 0u, ulRetirePeriod + 1u ) ).state ) );
	}

	// VRR nudges inside one period count once.
	RepaintLoop_t loop;
	for ( int i = 0; i < 8; i++ )
	{
		RepaintInputs_t in = repaint_inputs( true, 0u, 1u );
		in.vblank = false;
		const RepaintStep_t step = loop.begin( in );
		RepaintIteration_t nudge = step.iteration;
		CHECK( nudge.take_unselected_paint( step.plan ) == UnselectedPaint_t::Yield );
		loop.end( nudge );
	}
	RepaintIteration_t live = loop.begin( repaint_inputs( true, 0u, 1u ) ).iteration;
	CHECK( loop.state.yields == 1u );

	// Retiring before the next period leaves no age behind.
	live.add_paint( PaintOutput_t::Live, PaintOutcome_t::Presented, false );
	loop.end( live );
	CHECK( loop.begin( repaint_inputs( true, 0u, 1u ) ).state.yields == 0u );
}

TEST_CASE( "repaint yields are bounded without a real frame", "[framegen][deadline][repaint]" )
{
	RepaintLoop_t loop;
	uint64_t ulPeriod = 0;
	for ( uint32_t uYield = 1; uYield <= k_uMaxRepaintYields; uYield++ )
	{
		const RepaintStep_t step = loop.begin( repaint_inputs( true, 1u, ++ulPeriod ) );
		REQUIRE_FALSE( step.plan.exhausted );
		REQUIRE( step.plan.generatedMayTakeSlot );
		RepaintIteration_t iteration = step.iteration;
		iteration.add_paint( classify_paint_output( true, false, false, FlipVerdict_t::NotFlipped ),
			PaintOutcome_t::Presented, false );
		CHECK( iteration.yielded );
		loop.end( iteration );
	}

	// The fifth slot composites live and retires the request.
	const RepaintStep_t step = loop.begin( repaint_inputs( true, 1u, ++ulPeriod ) );
	CHECK( step.plan.exhausted );
	CHECK_FALSE( step.plan.generatedMayTakeSlot );
	CHECK( step.plan.unselectedPaint == UnselectedPaint_t::DiscardPrediction );
	RepaintIteration_t iteration = step.iteration;
	iteration.add_paint( classify_paint_output( false, false, false, FlipVerdict_t::NotFlipped ),
		PaintOutcome_t::Presented, step.plan.forcedDrivesPaint );
	CHECK( iteration.retiredOverlay );
	CHECK( iteration.retiredForced );
	loop.end( iteration );
	CHECK( repaint_state_fresh( loop.begin( repaint_inputs( false, 0u, ++ulPeriod ) ).state ) );
}

TEST_CASE( "repaint yields share the VRR overlay throttle bound", "[framegen][deadline][repaint]" )
{
	const auto plan = []( uint32_t yields, uint32_t ignored ) {
		RepaintInputs_t in = repaint_inputs( true, 0u, 2u );
		in.ignoredOverlayRepaints = ignored;
		return step_repaint_requests( { .yields = yields, .yieldVBlankNs = k_ulRepaintPeriodNs },
			RepaintIteration_t{ .in = repaint_inputs( true, 0u, 1u ) }, in ).plan;
	};
	for ( uint32_t uIgnored = 0; uIgnored < 64u; uIgnored++ )
		CHECK_FALSE( plan( 0u, uIgnored ).exhausted );
	CHECK_FALSE( plan( 2u, 1u ).exhausted );
	CHECK( plan( 3u, 1u ).exhausted );
	CHECK( plan( 4u, 0u ).exhausted );

	RepaintIteration_t iteration = { .in = repaint_inputs( true, 0u, 2u ) };
	iteration.add_paint( PaintOutput_t::Generated, PaintOutcome_t::Presented, false );
	CHECK( step_repaint_requests( { .yields = k_uMaxRepaintYields, .yieldVBlankNs = k_ulRepaintPeriodNs },
		iteration, repaint_inputs( true, 0u, 3u ) ).state.yields == k_uMaxRepaintYields );
}

TEST_CASE( "failed acquire or present retires or retries without spinning", "[framegen][deadline][repaint]" )
{
	constexpr uint64_t ulForced = 9u;

	// Terminal: the frame fails like upstream and nothing is left to drive.
	{
		std::atomic<uint64_t> counter{ ulForced };
		RepaintLoop_t loop;
		RepaintStep_t step = loop.begin( repaint_inputs( true, counter.load(), 1u ) );
		REQUIRE( step.plan.forcedDrivesPaint );
		RepaintIteration_t iteration = step.iteration;
		iteration.add_paint( PaintOutput_t::Live, PaintOutcome_t::Failed, true );
		acknowledge_forced_requests( counter, iteration );
		CHECK( iteration.retiredOverlay );
		CHECK( counter.load() == 0u );
		loop.end( iteration );
		step = loop.begin( repaint_inputs( false, counter.load(), 2u ) );
		CHECK( repaint_state_fresh( step.state ) );
		CHECK_FALSE( step.iteration.request_remains() );
		CHECK_FALSE( step.plan.forcedDrivesPaint );
	}

	// Transient: the renderer raised a force during the paint. The stale
	// acknowledgement fails, so exactly one retry follows per raised request.
	{
		std::atomic<uint64_t> counter{ ulForced };
		RepaintLoop_t loop;
		RepaintStep_t step = loop.begin( repaint_inputs( false, counter.load(), 1u ) );
		counter.fetch_add( 1u );
		RepaintIteration_t iteration = step.iteration;
		iteration.add_paint( PaintOutput_t::Live, PaintOutcome_t::Failed, true );
		acknowledge_forced_requests( counter, iteration );
		CHECK( counter.load() == ulForced + 1u );
		loop.end( iteration );

		step = loop.begin( repaint_inputs( false, counter.load(), 2u ) );
		CHECK( repaint_state_fresh( step.state ) );
		REQUIRE( step.plan.forcedDrivesPaint );
		iteration = step.iteration;
		iteration.add_paint( PaintOutput_t::Live, PaintOutcome_t::Presented, true );
		acknowledge_forced_requests( counter, iteration );
		CHECK( counter.load() == 0u );
		loop.end( iteration );
		CHECK_FALSE( loop.begin( repaint_inputs( false, counter.load(), 3u ) ).plan.forcedDrivesPaint );
	}

	// Repeated deferrals neither end frames nor age the request.
	RepaintLoop_t loop{ .state = { .yields = 2u, .yieldVBlankNs = k_ulRepaintPeriodNs } };
	loop.last = { .in = repaint_inputs( true, ulForced, 1u ) };
	for ( uint64_t i = 2; i < 100u; i++ )
	{
		const RepaintStep_t step = loop.begin( repaint_inputs( true, ulForced, i ) );
		CHECK( step.state.yields == 2u );
		CHECK( step.state.yieldVBlankNs == k_ulRepaintPeriodNs );
		RepaintIteration_t iteration = step.iteration;
		iteration.add_paint( PaintOutput_t::Live, PaintOutcome_t::Deferred, true );
		CHECK_FALSE( iteration.ended );
		loop.end( iteration );
	}
}

TEST_CASE( "bidir prime after an invalidation is presented live", "[framegen][deadline][repaint]" )
{
	// A gap (frame_gap, idle_frame_gap), cut (colorspace, EOTF, layer
	// count), focus change or ring-pressure flush leaves an empty queue and a
	// started timeline. The first real composite queues nothing: it is flipped
	// live and retires a pending request on that composite.
	RepaintLoop_t loop;
	RepaintInputs_t in = repaint_inputs( true, 0u, 1u );
	in.bidirActive = true;
	in.bidirQueueDrained = true;
	in.realRepaint = true;
	const RepaintStep_t step = loop.begin( in );
	REQUIRE( step.plan.overlayDrivesPaint );
	const BidirFlip_t eFlip = select_bidir_flip( {
		.queuedReal = false, .sameBase = false, .queueEmpty = true, .timelineStarted = true } );
	CHECK( eFlip == BidirFlip_t::SnapLive );
	RepaintIteration_t iteration = step.iteration;
	iteration.add_paint( classify_paint_output( false, false, true, flip_verdict( eFlip ) ),
		PaintOutcome_t::Presented, false );
	CHECK( iteration.retiredOverlay );
	CHECK_FALSE( iteration.held );

	// A cut that retained queued real frames still shows them first, and the
	// prime queues behind them.
	CHECK( select_bidir_flip( { .queuedReal = false, .sameBase = false, .queueEmpty = false,
		.timelineStarted = true } ) == BidirFlip_t::QueueBehind );
}

TEST_CASE( "a paused paint fails like upstream and drives nothing per vblank", "[framegen][deadline][repaint]" )
{
	// A VT-switched-away backend makes paint_all fail. The request is
	// retired on that paint, so later vblanks plan no composite.
	for ( const bool bBidir : { false, true } )
	{
		CAPTURE( bBidir );
		std::atomic<uint64_t> forced{ 2u };
		bool overlay = true;
		RepaintLoop_t loop;
		int nPaints = 0;
		for ( int v = 0; v < 64; v++ )
		{
			RepaintInputs_t in = repaint_inputs( overlay, forced.load(), uint64_t( v + 1 ) );
			in.bidirActive = bBidir;
			in.bidirQueueDrained = bBidir;
			const RepaintStep_t step = loop.begin( in );
			RepaintIteration_t iteration = step.iteration;
			if ( step.plan.overlayDrivesPaint || step.plan.forcedDrivesPaint )
			{
				nPaints++;
				// Failed before the flip substitution ran.
				iteration.add_paint( classify_paint_output( false, false, bBidir, FlipVerdict_t::NotFlipped ),
					PaintOutcome_t::Failed, step.plan.forcedDrivesPaint );
			}
			if ( iteration.retiredOverlay )
				overlay = false;
			acknowledge_forced_requests( forced, iteration );
			loop.end( iteration );
		}
		CHECK( nPaints == 1 );
		CHECK_FALSE( overlay );
		CHECK( forced.load() == 0u );
		CHECK( repaint_state_fresh( loop.state ) );
	}
}

TEST_CASE( "an off-vblank overlay retirement keeps the pending force's age", "[framegen][deadline][repaint]" )
{
	// VRR hybrid: each period a vblank wake lets a generated frame take
	// the forced slot, then an off-vblank real paint retires the overlay but not
	// the force. The force keeps aging and is serviced within the yield bound.
	std::atomic<uint64_t> forced{ 6u };
	RepaintLoop_t loop;
	int nServicedPeriod = -1;
	for ( uint64_t p = 1; p <= 64u && nServicedPeriod < 0; p++ )
	{
		const RepaintStep_t step = loop.begin( repaint_inputs( true, forced.load(), p ) );
		CHECK( loop.state.yields == std::min<uint64_t>( p - 1u, k_uMaxRepaintYields ) );
		REQUIRE( step.plan.forcedDrivesPaint );
		RepaintIteration_t iteration = step.iteration;
		iteration.add_paint( classify_paint_output( step.plan.generatedMayTakeSlot, false, false,
			FlipVerdict_t::NotFlipped ), PaintOutcome_t::Presented, true );
		acknowledge_forced_requests( forced, iteration );
		if ( forced.load() == 0u )
			nServicedPeriod = int( p );
		loop.end( iteration );

		RepaintInputs_t in = repaint_inputs( true, forced.load(), p );
		in.vblank = false;
		in.realRepaint = true;
		iteration = loop.begin( in ).iteration;
		iteration.add_paint( classify_paint_output( false, false, false, FlipVerdict_t::NotFlipped ),
			PaintOutcome_t::Presented, false );
		CHECK( iteration.retiredOverlay );
		CHECK_FALSE( iteration.retiredForced );
		loop.end( iteration );
	}
	CHECK( nServicedPeriod == int( k_uMaxRepaintYields ) + 1 );
}

TEST_CASE( "bidir prime after a cut over queued reals is presented behind them", "[framegen][deadline][repaint]" )
{
	// A cut or focus change over [real, interp, real], then no commit.
	// The prime queues behind the retained reals, is presented within queue
	// length + 1 vblanks, and services the overlay request it was made for.
	BidirSim_t sim;
	sim.queue = { { 1u, 1 }, { 0u, 2 }, { 2u, 3 } };
	sim.lastSeq = 2u;
	sim.timelineStarted = true;
	const size_t nQueued = sim.queue.size();

	RepaintLoop_t loop;
	bool overlay = true;
	RepaintInputs_t in = bidir_inputs( overlay, 0u, 1, sim );
	in.realRepaint = true;
	RepaintIteration_t iteration = loop.begin( in ).iteration;
	const BidirSim_t::Paint_t prime = sim.composite( 1, true, false, 2 );
	add_sim_paint( iteration, prime, false );
	CHECK( prime.report.verdict == FlipVerdict_t::Older );
	CHECK( iteration.held );
	CHECK( iteration.request_remains() );
	if ( iteration.retiredOverlay )
		overlay = false;
	loop.end( iteration );

	int nPrimeVBlank = -1;
	for ( int v = 2; v <= 16 && nPrimeVBlank < 0; v++ )
	{
		const RepaintStep_t step = loop.begin( bidir_inputs( overlay, 0u, v, sim ) );
		CHECK_FALSE( step.plan.overlayDrivesPaint );	// carried, never driving
		iteration = step.iteration;
		const BidirSim_t::Paint_t paint = sim.consume( v );
		add_sim_paint( iteration, paint, false );
		if ( paint.shown == prime.seq )
		{
			nPrimeVBlank = v;
			CHECK_FALSE( iteration.request_remains() );
		}
		loop.end( iteration );
	}
	REQUIRE( nPrimeVBlank > 0 );
	CHECK( size_t( nPrimeVBlank - 1 ) <= nQueued + 1u );
	CHECK_FALSE( overlay );
	CHECK( repaint_state_fresh( loop.begin( bidir_inputs( false, 0u, 99, sim ) ).state ) );
}

TEST_CASE( "bidir unplanned composites every vblank drain the queue to live", "[framegen][deadline][repaint]" )
{
	// Depth 3 of retained content, then a new base that queued nothing
	// on every vblank (a hitching game outrunning its plan). Each one replaces
	// the queued-behind tail, so the queue drains at one entry per refresh and
	// the composite goes live within the depth instead of lagging it forever.
	// Shown content only moves forward, nothing queued is snapped away, and the
	// overlay request a superseded composite carried is serviced by its successor.
	BidirSim_t sim;
	sim.queue = { { 1u, 1 }, { 0u, 2 }, { 2u, 3 } };
	sim.lastSeq = 2u;
	sim.timelineStarted = true;
	const int nDepth = int( sim.queue.size() );
	RepaintLoop_t loop;
	bool overlay = true;
	std::atomic<uint64_t> forced{ 1u };
	uint64_t ulLastShown = 0;
	int nLiveVBlank = -1;
	int nServiceVBlank = -1;
	for ( int v = 1; v <= 64; v++ )
	{
		CAPTURE( v );
		RepaintInputs_t in = bidir_inputs( overlay, forced.load(), v, sim );
		in.realRepaint = true;
		const RepaintStep_t step = loop.begin( in );
		RepaintIteration_t iteration = step.iteration;
		const BidirSim_t::Paint_t paint = sim.composite( v, true, false, 2 );
		add_sim_paint( iteration, paint, step.plan.forcedDrivesPaint );
		REQUIRE( std::ranges::count_if( sim.queue, &BidirSim_t::Entry_t::behind ) <= 1 );
		REQUIRE( int( sim.queue.size() ) <= nDepth );
		if ( paint.shown != 0u )
		{
			REQUIRE( paint.shown > ulLastShown );
			ulLastShown = paint.shown;
		}
		if ( paint.report.verdict == FlipVerdict_t::Live && nLiveVBlank < 0 )
			nLiveVBlank = v;
		if ( nLiveVBlank > 0 )
			CHECK( paint.report.verdict == FlipVerdict_t::Live );	// stays live, like upstream
		if ( iteration.retiredOverlay )
			overlay = false;
		acknowledge_forced_requests( forced, iteration );
		if ( !iteration.request_remains() && nServiceVBlank < 0 )
			nServiceVBlank = v;
		loop.end( iteration );
	}
	CHECK( forced.load() == 0u );
	REQUIRE( nLiveVBlank > 0 );
	CHECK( nLiveVBlank <= nDepth + 1 );
	CHECK( nServiceVBlank == nLiveVBlank );
	CHECK( sim.snapped == 0u );
	CHECK_FALSE( overlay );
	CHECK( repaint_state_fresh( loop.state ) );
}

TEST_CASE( "bidir consecutive unplanned composites keep one queued-behind entry", "[framegen][deadline][repaint]" )
{
	// N unplanned new bases before the front is due (off-vblank paints in one
	// refresh): the first queues behind, the rest replace it, so the queue grows
	// by one entry however many there are, and the newest composite is queued.
	constexpr int k_nComposites = 8;
	BidirSim_t sim;
	sim.queue = { { 1u, 5 }, { 0u, 6 }, { 2u, 7 } };
	sim.lastSeq = 2u;
	sim.timelineStarted = true;
	const size_t nQueued = sim.queue.size();
	uint64_t ulPrevSeq = 0;
	for ( int i = 0; i < k_nComposites; i++ )
	{
		const BidirSim_t::Paint_t paint = sim.composite( 1, true, false, 2 );
		CHECK( paint.shown == 0u );
		CHECK( paint.report.verdict == FlipVerdict_t::Older );
		CHECK( paint.superseded == ulPrevSeq );
		REQUIRE( sim.queue.size() == nQueued + 1u );
		CHECK( sim.queue.back().seq == paint.seq );
		CHECK( sim.queue.back().target == 8 );
		ulPrevSeq = paint.seq;
	}
	CHECK( std::ranges::count_if( sim.queue, &BidirSim_t::Entry_t::behind ) == 1 );
	CHECK( sim.snapped == 0u );
	// A planned real frame queues behind it; the next unplanned one queues anew.
	const BidirSim_t::Paint_t planned = sim.composite( 2, true, true, 2 );
	CHECK( planned.superseded == 0u );
	const BidirSim_t::Paint_t unplanned = sim.composite( 2, true, false, 2 );
	CHECK( unplanned.superseded == 0u );
	CHECK( std::ranges::count_if( sim.queue, &BidirSim_t::Entry_t::behind ) == 2 );
}

TEST_CASE( "bidir presents every real composite within the hard capacity", "[framegen][deadline][repaint]" )
{
	// The production flip selection over cadences, cuts, unplanned pairs,
	// overlay-only re-renders and stalls: every real composite reaches the
	// screen in composite order within the pending hard capacity, skipped only
	// when a newer unplanned composite replaced it while it was queued behind.
	const size_t uCapacity = bidir_pending_hard_capacity( 6u, 2u );
	uint32_t uRng = 0x9e3779b9u;
	const auto next = [&]() { uRng = uRng * 1664525u + 1013904223u; return uRng >> 16; };
	for ( uint32_t gap = 1u; gap + 1u <= uCapacity; gap++ )
	{
		for ( int run = 0; run < 64; run++ )
		{
			CAPTURE( gap, run );
			BidirSim_t sim;
			std::vector<std::pair<uint64_t, int>> composited;	// seq, vblank
			uint64_t ulLastShown = 0;
			int nNextCommit = 0;
			for ( int v = 0; v < 400; v++ )
			{
				BidirSim_t::Paint_t paint;
				const uint32_t r = next() % 16u;
				if ( v == nNextCommit && v < 360 )
				{
					// r < 2: a cut or focus change primes history and plans nothing.
					const bool planned = r >= 2u && sim.timelineStarted;
					paint = sim.composite( v, true, planned, int( gap ) );
					if ( paint.superseded != 0u )
					{
						REQUIRE( paint.superseded < paint.seq );
						REQUIRE( std::erase_if( composited, [&]( const auto &entry ) {
							return entry.first == paint.superseded; } ) == 1u );
					}
					composited.emplace_back( paint.seq, v );
					nNextCommit = v + int( gap ) + ( r == 15u ? int( next() % 8u ) : 0 );
				}
				else if ( r == 3u )
					paint = sim.composite( v, false, false, int( gap ) );	// overlay-only re-render
				else
					paint = sim.consume( v );
				REQUIRE( sim.queue.size() <= uCapacity );
				if ( paint.shown != 0u )
				{
					REQUIRE( paint.shown > ulLastShown );	// request retirement relies on this order
					ulLastShown = paint.shown;
					const auto it = std::ranges::find( composited, paint.shown,
						&std::pair<uint64_t, int>::first );
					if ( it != composited.end() )
					{
						REQUIRE( it == composited.begin() );
						CHECK( v - it->second <= int( uCapacity ) );
						composited.erase( it );
					}
				}
				REQUIRE( ( composited.empty() || v - composited.front().second <= int( uCapacity ) ) );
			}
			CHECK( composited.empty() );
			CHECK( sim.snapped == 0u );
		}
	}
}

TEST_CASE( "bidir services requests behind a queue that never drains", "[framegen][deadline][repaint]" )
{
	// 30 fps on 60 Hz at x2, so every iteration samples a non-empty
	// queue. An overlay request and a force stay pending until the first real
	// composite made after them is presented, are retired on that paint within
	// the hard capacity of it, and never drive a composite. Ones raised after
	// that composite wait for the next.
	const int nCapacity = int( bidir_pending_hard_capacity( 6u, 2u ) );
	const auto run = [&]( std::initializer_list<int> raiseVBlanks, int reRenderVBlank ) {
		BidirSim_t sim;
		sim.queue = { { 1u, 0 } };
		sim.lastSeq = 1u;
		sim.timelineStarted = true;
		RepaintLoop_t loop;
		std::atomic<uint64_t> forced{ 0u };
		bool overlay = false;
		uint64_t ulCarrier = 0;		// first real composite after the last raise
		int nCarrierVBlank = -1;
		uint64_t ulShown = 0;
		for ( int v = 0; v < 48; v++ )
		{
			if ( std::ranges::find( raiseVBlanks, v ) != raiseVBlanks.end() )
			{
				overlay = true;
				forced.fetch_add( 1u );
				ulCarrier = 0u;
			}
			RepaintInputs_t in = bidir_inputs( overlay, forced.load(), v, sim );
			REQUIRE_FALSE( in.bidirQueueDrained );
			in.realRepaint = v % 2 == 0;
			const RepaintStep_t step = loop.begin( in );
			CHECK_FALSE( step.plan.overlayDrivesPaint );
			RepaintIteration_t iteration = step.iteration;
			BidirSim_t::Paint_t paint;
			if ( in.realRepaint )
			{
				paint = sim.composite( v, true, true, 2 );
				if ( ulCarrier == 0u && ( overlay || forced.load() != 0u ) )
				{
					ulCarrier = paint.seq;
					nCarrierVBlank = v;
				}
			}
			else if ( v == reRenderVBlank )
				paint = sim.composite( v, false, false, 2 );	// never queued, never presented
			else
				paint = sim.consume( v );	// a forced vblank without a commit
			add_sim_paint( iteration, paint, step.plan.forcedDrivesPaint );
			ulShown = std::max( ulShown, paint.shown );
			if ( iteration.retiredOverlay )
				overlay = false;
			acknowledge_forced_requests( forced, iteration );
			const bool bOutstanding = overlay || iteration.overlay_carrier() != 0u;
			loop.end( iteration );
			if ( ulCarrier == 0u || ulShown < ulCarrier )
			{
				// Nothing made after the last raise is on screen yet.
				CHECK( ( bOutstanding || v < *raiseVBlanks.begin() ) );
				CHECK( ( forced.load() != 0u || v < *raiseVBlanks.begin() ) );
				continue;
			}
			CHECK_FALSE( bOutstanding );
			CHECK( forced.load() == 0u );
			return v - nCarrierVBlank;
		}
		return -1;
	};
	// v11 re-renders the base between the v10 carrier and its presentation.
	for ( const int reRender : { -1, 11 } )
	{
		CAPTURE( reRender );
		const int nOne = run( { 9 }, reRender );
		CHECK( nOne > 0 );
		CHECK( nOne <= nCapacity );
	}
	// v11 is raised after the v10 composite: it rides the v12 composite, also
	// when a v11 re-render consumed it first, so presenting v10 retires nothing.
	for ( const int reRender : { -1, 11 } )
	{
		CAPTURE( reRender );
		const int nTwo = run( { 9, 11 }, reRender );
		CHECK( nTwo > 0 );
		CHECK( nTwo <= nCapacity );
	}
}

TEST_CASE( "a held composite carries only a force a vblank paint took", "[framegen][deadline][repaint]" )
{
	// Upstream consumes a force only on the vblank paint that took it, and a
	// failed paint drops it. Behind a bidir queue a carried force keeps both.
	std::atomic<uint64_t> forced{ 1u };
	RepaintLoop_t loop;
	RepaintInputs_t in = repaint_inputs( false, forced.load(), 1u );
	in.bidirActive = true;
	in.vblank = false;
	in.realRepaint = true;
	RepaintStep_t step = loop.begin( in );
	REQUIRE_FALSE( step.plan.forcedDrivesPaint );
	RepaintIteration_t iteration = step.iteration;
	iteration.add_paint( PaintOutput_t::Held, PaintOutcome_t::Presented, false, { .composited = 5u } );
	CHECK( iteration.forced_carrier() == 0u );	// an off-vblank paint did not take it
	loop.end( iteration );

	in = repaint_inputs( false, forced.load(), 2u );
	in.bidirActive = true;
	in.realRepaint = true;
	step = loop.begin( in );
	iteration = step.iteration;
	iteration.add_paint( PaintOutput_t::Held, PaintOutcome_t::Presented, step.plan.forcedDrivesPaint,
		{ .composited = 6u, .presented = 4u } );
	CHECK( iteration.forced_carrier() == 6u );	// the held snapshot rides the next composite
	acknowledge_forced_requests( forced, iteration );
	loop.end( iteration );

	in = repaint_inputs( false, forced.load(), 2u );
	in.bidirActive = true;
	in.vblank = false;
	in.realRepaint = true;
	step = loop.begin( in );
	iteration = step.iteration;
	REQUIRE( iteration.forced_carrier() == 6u );
	iteration.add_paint( PaintOutput_t::Held, PaintOutcome_t::Failed, false );
	CHECK( iteration.retiredForced );
	acknowledge_forced_requests( forced, iteration );
	CHECK( forced.load() == 0u );
	CHECK_FALSE( iteration.request_remains() );
}

TEST_CASE( "a carrier that was never queued drives one refresh after the drain", "[framegen][deadline][repaint]" )
{
	// A same-base re-render behind the queue carries the overlay but is never
	// presented. No real frame follows: once the queue drains, exactly one live
	// refresh services the request, and nothing composites before that.
	BidirSim_t sim;
	sim.queue = { { 1u, 1 }, { 0u, 2 }, { 2u, 3 } };
	sim.lastSeq = 2u;
	sim.timelineStarted = true;
	RepaintLoop_t loop;
	bool overlay = true;
	RepaintInputs_t in = bidir_inputs( overlay, 0u, 0, sim );
	in.realRepaint = true;
	RepaintIteration_t iteration = loop.begin( in ).iteration;
	add_sim_paint( iteration, sim.composite( 0, false, false, 2 ), false );
	REQUIRE( iteration.overlay_carrier() == 3u );
	if ( iteration.retiredOverlay )
		overlay = false;
	loop.end( iteration );

	int nRefreshes = 0;
	int nDrainVBlank = -1;
	for ( int v = 1; v < 32; v++ )
	{
		in = bidir_inputs( overlay, 0u, v, sim );
		if ( in.bidirQueueDrained && nDrainVBlank < 0 )
			nDrainVBlank = v;
		const RepaintStep_t step = loop.begin( in );
		iteration = step.iteration;
		CHECK( ( !step.plan.overlayDrivesPaint || in.bidirQueueDrained ) );
		if ( step.plan.overlayDrivesPaint )
		{
			nRefreshes++;
			const BidirSim_t::Paint_t paint = sim.composite( v, false, false, 2 );
			CHECK( paint.report.verdict == FlipVerdict_t::Live );
			add_sim_paint( iteration, paint, false );
			CHECK_FALSE( iteration.request_remains() );
		}
		else if ( !in.bidirQueueDrained )
			add_sim_paint( iteration, sim.consume( v ), false );
		loop.end( iteration );
	}
	CHECK( nDrainVBlank == 4 );
	CHECK( nRefreshes == 1 );
	CHECK_FALSE( overlay );
	CHECK( repaint_state_fresh( loop.state ) );
}

TEST_CASE( "bidir refresh-miss drop spares an overlay raised in the drop iteration", "[framegen][deadline][repaint]" )
{
	// Like a force, the overlay flag is dropped only if it was
	// pending, and not carried by a held composite, at the miss that hit the
	// cap. A broken presenter holds every refresh over a drained queue.
	for ( const bool bCarried : { true, false } )
	{
		for ( const bool bRaiseAtDrop : { false, true } )
		{
			CAPTURE( bCarried, bRaiseAtDrop );
			RepaintLoop_t loop;
			bool overlay = true;
			uint64_t ulSeq = 0;
			int nComposites = 0;
			int nDrops = 0;
			for ( int v = 0; v < 12; v++ )
			{
				const bool bDropVBlank = v == int( k_uMaxBidirRefreshMisses );
				if ( bDropVBlank && bRaiseAtDrop )
					overlay = true;
				RepaintInputs_t in = repaint_inputs( overlay, 0u, uint64_t( v + 1 ) );
				in.bidirActive = true;
				in.bidirQueueDrained = true;
				const RepaintStep_t step = loop.begin( in );
				if ( v <= int( k_uMaxBidirRefreshMisses ) )
					CHECK( step.plan.dropHeld == bDropVBlank );
				nDrops += step.plan.dropHeld;
				RepaintIteration_t iteration = step.iteration;
				if ( bDropVBlank )
					CHECK( iteration.retiredOverlay == !bCarried );
				if ( step.plan.overlayDrivesPaint )
				{
					nComposites++;
					iteration.add_paint( classify_paint_output( false, false, true, FlipVerdict_t::Older ),
						PaintOutcome_t::Presented, false, { .composited = bCarried ? ++ulSeq : 0u } );
				}
				if ( iteration.retiredOverlay )
					overlay = false;
				loop.end( iteration );
			}
			// Spared only when raised after the carrying composite: it is a new
			// request with its own capped refreshes.
			const bool bSpared = bCarried && bRaiseAtDrop;
			CHECK( nComposites == int( k_uMaxBidirRefreshMisses ) * ( bSpared ? 2 : 1 ) );
			CHECK( nDrops == ( bSpared ? 2 : 1 ) );
			CHECK_FALSE( overlay );
			CHECK( repaint_state_fresh( loop.state ) );
		}
	}
}

TEST_CASE( "a force re-raised to the same counter value after a carried retirement keeps its one commit", "[framegen][deadline][repaint]" )
{
	// The carried force is acknowledged to 0 while a carried overlay keeps
	// the request state alive. A new force brings the counter back to the old
	// snapshot; it is a new request and still drives its one commit.
	BidirSim_t sim;
	sim.queue = { { 1u, 0 } };
	sim.lastSeq = 1u;
	sim.timelineStarted = true;
	RepaintLoop_t loop;
	std::atomic<uint64_t> forced{ 0u };
	bool overlay = false;
	bool bForceRetiredWithOverlayInFlight = false;
	for ( int v = 0; v < 6; v++ )
	{
		if ( v == 1 || v == 5 )
			forced.fetch_add( 1u );
		if ( v == 3 )
			overlay = true;
		RepaintInputs_t in = bidir_inputs( overlay, forced.load(), v, sim );
		REQUIRE_FALSE( in.bidirQueueDrained );
		in.realRepaint = v % 2 == 0;
		const RepaintStep_t step = loop.begin( in );
		if ( v == 1 || v == 5 )
			CHECK( step.plan.forcedDrivesPaint );	// each new force gets its one commit
		RepaintIteration_t iteration = step.iteration;
		add_sim_paint( iteration, in.realRepaint ? sim.composite( v, true, true, 2 ) : sim.consume( v ),
			step.plan.forcedDrivesPaint );
		if ( iteration.retiredOverlay )
			overlay = false;
		acknowledge_forced_requests( forced, iteration );
		if ( iteration.retiredForced && iteration.overlay_carrier() != 0u )
			bForceRetiredWithOverlayInFlight = true;
		loop.end( iteration );
	}
	CHECK( bForceRetiredWithOverlayInFlight );
}

TEST_CASE( "composite seqs change nothing on live and generated paints", "[framegen][deadline][repaint]" )
{
	// Framegen off and non-bidir never hold, so whatever the presenter accessor
	// reports cannot change their bookkeeping.
	constexpr PaintReport_t k_Reports[] = { { .composited = 0u, .presented = 7u }, { .composited = 7u, .presented = 0u },
		{ .composited = 7u, .presented = 7u }, { .composited = ~0ull, .presented = ~0ull } };
	constexpr PaintOutcome_t k_Outcomes[] = { PaintOutcome_t::Presented, PaintOutcome_t::Deferred,
		PaintOutcome_t::Failed };
	for ( int i = 0; i < 48; i++ )
	{
		const bool bOverlay = i & 1, bForcedPaint = i & 2, bForcedHeld = i & 4;
		const PaintOutput_t eOutput = ( i & 8 ) ? PaintOutput_t::Generated : PaintOutput_t::Live;
		const PaintOutcome_t eOutcome = k_Outcomes[ i >> 4 ];
		CAPTURE( i );
		const RepaintIteration_t base = { .in = repaint_inputs( bOverlay, 3u, 1u ),
			.state = { .heldForcedRequests = bForcedHeld ? 3u : 0u } };
		RepaintIteration_t expected = base;
		expected.add_paint( eOutput, eOutcome, bForcedPaint );
		for ( const PaintReport_t &report : k_Reports )
		{
			RepaintIteration_t actual = base;
			actual.add_paint( eOutput, eOutcome, bForcedPaint, report );
			CHECK( actual.yielded == expected.yielded );
			CHECK( actual.ended == expected.ended );
			CHECK( actual.retiredOverlay == expected.retiredOverlay );
			CHECK( actual.retiredForced == expected.retiredForced );
			CHECK( actual.held == expected.held );
			CHECK( actual.overlay_carrier() == 0u );
			CHECK( actual.forced_carrier() == 0u );
		}
	}
}
