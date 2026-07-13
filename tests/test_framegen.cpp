#include "framegen/adaptation.hpp"
#include "framegen/dispatch_policy.hpp"
#include "framegen/net_layout.hpp"
#include "framegen/net_profile.hpp"
#include "framegen/policy.hpp"
#include "framegen/push_constants.hpp"
#include "framegen/scheduling.hpp"
#include "framegen/settings.hpp"
#include "framegen/temporal.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <string_view>

using gamescope::framegen::EffectiveConfig;
using gamescope::framegen::effective_config;
using gamescope::framegen::max_degrade_steps;

static bool g_bPassed = true;

static void check( bool bCondition, const char *pszExpression, int nLine )
{
	if ( bCondition )
		return;

	std::fprintf( stderr, "framegen contract check failed at line %d: %s\n",
		nLine, pszExpression );
	g_bPassed = false;
}

#define CHECK( expression ) check( static_cast<bool>( expression ), #expression, __LINE__ )
#define CHECK_NEAR( actual, expected, tolerance ) \
	CHECK( std::fabs( ( actual ) - ( expected ) ) <= ( tolerance ) )

static void test_degradation_policy()
{
	constexpr std::array modes = {
		GamescopeFramegenMode::Extrapolate,
		GamescopeFramegenMode::Blend,
		GamescopeFramegenMode::Motion,
	};

	for ( const GamescopeFramegenMode mode : modes )
	{
		for ( uint32_t q = 0; q <= static_cast<uint32_t>( GamescopeFramegenQuality::Extreme ); q++ )
		{
			const auto quality = static_cast<GamescopeFramegenQuality>( q );
			for ( int multiplier = 2; multiplier <= 4; multiplier++ )
			{
				const uint32_t motionRungs = mode == GamescopeFramegenMode::Motion ? q + 1u : 0u;
				const uint32_t expectedMax = motionRungs + static_cast<uint32_t>( multiplier - 2 );
				CHECK( max_degrade_steps( mode, quality, multiplier ) == expectedMax );

				for ( uint32_t step = 0; step <= expectedMax + 2u; step++ )
				{
					const EffectiveConfig config = effective_config( mode, quality, multiplier, step );
					const uint32_t qualitySteps = mode == GamescopeFramegenMode::Motion
						? std::min( step, q ) : 0u;
					const bool fellBack = mode == GamescopeFramegenMode::Motion && step > q;
					const uint32_t multiplierSteps = step > motionRungs
						? step - motionRungs : 0u;

					CHECK( config.mode == ( fellBack ? GamescopeFramegenMode::Extrapolate : mode ) );
					CHECK( config.quality == static_cast<GamescopeFramegenQuality>( q - qualitySteps ) );
					CHECK( config.multiplier == static_cast<uint32_t>(
						std::max( 2, multiplier - static_cast<int>( multiplierSteps ) ) ) );
				}
			}
		}
	}
}

static void test_learned_net_layout()
{
	CHECK( k_uFramegenNetLayerCount == 3u );
	CHECK( k_uFramegenNetKernelWidth == 3u );
	CHECK( k_uFramegenNetLayerDims[ 0 ][ 0 ] == 12u );
	CHECK( k_uFramegenNetLayerDims[ 0 ][ 1 ] == 16u );
	CHECK( k_uFramegenNetLayerDims[ 1 ][ 0 ] == 16u );
	CHECK( k_uFramegenNetLayerDims[ 1 ][ 1 ] == 16u );
	CHECK( k_uFramegenNetLayerDims[ 2 ][ 0 ] == 16u );
	CHECK( k_uFramegenNetLayerDims[ 2 ][ 1 ] == 4u );
	CHECK( k_uFramegenNetLayer2Offset == 1744u );
	CHECK( k_uFramegenNetLayer3Offset == 4064u );
	CHECK( k_uFramegenNetShadingWeightBegin == 4496u );
	CHECK( k_uFramegenNetLayer3BiasOffset == 4640u );
	CHECK( k_uFramegenNetShadingBias == 4643u );
	CHECK( k_uFramegenNetFloats == 4644u );
	CHECK( k_uFramegenNetTexW == 2048u );
	CHECK( k_uFramegenNetTexH == 3u );
}

static void test_names()
{
	using gamescope::framegen::mode_name;
	using gamescope::framegen::quality_name;

	CHECK( std::string_view( mode_name( GamescopeFramegenMode::Extrapolate ) ) == "extrapolate" );
	CHECK( std::string_view( mode_name( GamescopeFramegenMode::Blend ) ) == "blend" );
	CHECK( std::string_view( mode_name( GamescopeFramegenMode::Motion ) ) == "motion" );
	CHECK( std::string_view( mode_name( static_cast<GamescopeFramegenMode>( 99u ) ) ) == "unknown" );
	CHECK( std::string_view( quality_name( GamescopeFramegenQuality::Low ) ) == "low" );
	CHECK( std::string_view( quality_name( GamescopeFramegenQuality::Medium ) ) == "medium" );
	CHECK( std::string_view( quality_name( GamescopeFramegenQuality::High ) ) == "high" );
	CHECK( std::string_view( quality_name( GamescopeFramegenQuality::Ultra ) ) == "ultra" );
	CHECK( std::string_view( quality_name( GamescopeFramegenQuality::Extreme ) ) == "extreme" );
	CHECK( std::string_view( quality_name( static_cast<GamescopeFramegenQuality>( 99u ) ) ) == "unknown" );
}

static void test_net_profile_contract()
{
	using namespace gamescope::framegen;

	constexpr NetProfileMetadata current = net_profile_metadata();
	static_assert( current[ 0 ] == k_uFramegenNetMagic );
	static_assert( current[ 1 ] == k_uFramegenNetVersion );
	static_assert( current[ 2 ] == k_uFramegenNetLayerCount );
	static_assert( current[ 3 ] == 12u && current[ 4 ] == 16u && current[ 5 ] == 3u );
	static_assert( current[ 6 ] == 16u && current[ 7 ] == 16u && current[ 8 ] == 3u );
	static_assert( current[ 9 ] == 16u && current[ 10 ] == 4u && current[ 11 ] == 3u );
	CHECK( net_profile_metadata_version( current ) == k_uFramegenNetVersion );

	for ( uint32_t version = k_uFramegenNetOldestVersion;
		version <= k_uFramegenNetVersion; version++ )
	{
		const NetProfileMetadata metadata = net_profile_metadata( version );
		CHECK( net_profile_metadata_version( metadata ) == version );
	}

	NetProfileMetadata malformed = current;
	malformed[ 0 ] ^= 1u;
	CHECK( net_profile_metadata_version( malformed ) == 0u );
	malformed = net_profile_metadata( 0u );
	CHECK( net_profile_metadata_version( malformed ) == 0u );
	malformed = net_profile_metadata( k_uFramegenNetVersion + 1u );
	CHECK( net_profile_metadata_version( malformed ) == 0u );
	malformed = current;
	malformed[ 2 ]++;
	CHECK( net_profile_metadata_version( malformed ) == 0u );
	for ( size_t word = 3u; word < malformed.size(); word++ )
	{
		malformed = current;
		malformed[ word ]++;
		CHECK( net_profile_metadata_version( malformed ) == 0u );
	}
	CHECK( net_profile_metadata_version(
		std::span<const uint32_t>( current.data(), current.size() - 1u ) ) == 0u );

	std::array<float, k_uFramegenNetFloats> weights = {};
	for ( size_t i = 0; i < weights.size(); i++ )
		weights[ i ] = static_cast<float>( i + 1u );
	const auto original = weights;
	CHECK( validate_and_migrate_net_profile_weights( k_uFramegenNetShadingVersion, weights ) );
	CHECK( weights == original );

	for ( const uint32_t version : { 1u, 2u } )
	{
		weights = original;
		CHECK( validate_and_migrate_net_profile_weights( version, weights ) );
		for ( size_t i = 0; i < weights.size(); i++ )
		{
			const bool bLegacyShadingWeight = i >= k_uFramegenNetShadingWeightBegin
				&& i < k_uFramegenNetLayer3BiasOffset;
			const bool bLegacyShadingBias = i == k_uFramegenNetShadingBias;
			CHECK( weights[ i ] == ( bLegacyShadingWeight || bLegacyShadingBias
				? 0.0f : original[ i ] ) );
		}
	}

	weights = original;
	weights[ 17 ] = std::numeric_limits<float>::quiet_NaN();
	CHECK( !validate_and_migrate_net_profile_weights( k_uFramegenNetShadingVersion, weights ) );
	weights = original;
	weights[ 18 ] = std::numeric_limits<float>::infinity();
	CHECK( !validate_and_migrate_net_profile_weights( k_uFramegenNetShadingVersion, weights ) );
	CHECK( !validate_and_migrate_net_profile_weights( 0u, weights ) );
	CHECK( !validate_and_migrate_net_profile_weights( k_uFramegenNetVersion + 1u, weights ) );
	CHECK( !validate_and_migrate_net_profile_weights( k_uFramegenNetShadingVersion,
		std::span<float>( weights.data(), weights.size() - 1u ) ) );
}

static void test_numeric_settings_contract()
{
	using namespace gamescope::framegen;

	static_assert( is_finite_binary32( 0.0f ) );
	static_assert( is_finite_binary32( -std::numeric_limits<float>::max() ) );
	static_assert( !is_finite_binary32( std::numeric_limits<float>::infinity() ) );
	static_assert( !is_finite_binary32( std::numeric_limits<float>::quiet_NaN() ) );

	CHECK( non_empty_setting( nullptr ) == nullptr );
	CHECK( non_empty_setting( "" ) == nullptr );
	const char path[] = "/tmp/framegen-profile";
	CHECK( non_empty_setting( path ) == path );

	CHECK( !parse_finite_float_setting( nullptr ).has_value() );
	CHECK( !parse_finite_float_setting( "" ).has_value() );
	CHECK( !parse_finite_float_setting( "value" ).has_value() );
	CHECK( !parse_finite_float_setting( "0.5suffix" ).has_value() );
	CHECK( !parse_finite_float_setting( "nan" ).has_value() );
	CHECK( !parse_finite_float_setting( "inf" ).has_value() );
	CHECK( !parse_finite_float_setting( "1e999" ).has_value() );
	CHECK( parse_finite_float_setting( "0.5" ).value_or( -1.0f ) == 0.5f );
	CHECK( parse_finite_float_setting( "-2.25" ).value_or( 0.0f ) == -2.25f );

	CHECK( !parse_uint32_setting( nullptr, true ).has_value() );
	CHECK( !parse_uint32_setting( "", true ).has_value() );
	CHECK( !parse_uint32_setting( "0", false ).has_value() );
	CHECK( parse_uint32_setting( "0", true ).value_or( 1u ) == 0u );
	CHECK( parse_uint32_setting( "4294967295", false ).value_or( 0u ) == UINT32_MAX );
	CHECK( !parse_uint32_setting( "4294967296", false ).has_value() );
	CHECK( !parse_uint32_setting( "-1", false ).has_value() );
	CHECK( !parse_uint32_setting( "12suffix", false ).has_value() );
	CHECK( parse_uint32_setting( "12", false ).value_or( 0u ) == 12u );
}

static void test_adaptation_policy()
{
	using namespace gamescope::framegen;

	static_assert( k_uAdaptationStatsCount == 96u );
	static_assert( k_flAdaptationTrustLo == 0.15f );
	static_assert( k_flAdaptationTrustHi == 0.45f );
	static_assert( k_flAdaptationBadResidual == 0.10f );

	std::array<uint32_t, k_uAdaptationStatsCount> stats = {};
	CHECK( !decode_adaptation_stats( stats ).has_value() );
	stats[ 0 ] = k_uAdaptationMaxTotal + 1u;
	CHECK( !decode_adaptation_stats( stats ).has_value() );

	stats = {};
	stats[ 0 ] = 100u;
	stats[ 1 ] = 10u * 1024u;
	stats[ 2 ] = 20u;
	stats[ 3 ] = 20u;
	stats[ 4 ] = 5u * 1024u;
	stats[ 5 ] = 100u;
	stats[ 6 ] = 24u;
	stats[ 7 ] = 51u;
	stats[ 14 ] = 100u * 64u;
	stats[ 88 ] = 1u;
	stats[ 89 ] = 7u;
	stats[ 90 ] = 512u;
	const auto measurement = decode_adaptation_stats( stats );
	CHECK( measurement.has_value() );
	CHECK_NEAR( measurement->residual, 0.10f, 1e-7f );
	CHECK_NEAR( measurement->badFraction, 0.25f, 1e-7f );
	CHECK_NEAR( measurement->killedFraction, 0.20f, 1e-7f );
	CHECK_NEAR( measurement->motionMean, 1.0f, 1e-7f );
	CHECK_NEAR( measurement->noise, 0.05f, 1e-7f );
	CHECK_NEAR( measurement->fbP75, 0.50f, 1e-7f );
	CHECK( measurement->sceneCut == 1u );
	CHECK( measurement->changedSections == 7u );
	CHECK( measurement->sceneHistogramDistanceQ10 == 512u );
	CHECK_NEAR( scene_histogram_distance( *measurement ), 0.50f, 1e-7f );

	AdaptationState state;
	update_adaptation_state( state, *measurement, 0.75f, false );
	CHECK_NEAR( state.residualEma, 0.10f, 1e-7f );
	CHECK_NEAR( state.noiseEma, 0.05f, 1e-7f );
	CHECK_NEAR( state.fbP75Ema, 0.50f, 1e-7f );
	CHECK( state.fbTolerance == -1.0f );
	CHECK_NEAR( state.agreementOffset, k_flAdaptationAgreementOffsetMax, 1e-7f );
	CHECK_NEAR( active_agreement_lo( state, 0.12f ), 0.27f, 1e-7f );
	CHECK_NEAR( active_agreement_hi( state, 0.45f ), 0.75f, 1e-7f );

	float ema = -1.0f;
	fold_adaptation_ema( ema, -1.0f );
	CHECK( ema == -1.0f );
	fold_adaptation_ema( ema, 0.5f );
	CHECK( ema == 0.5f );
	fold_adaptation_ema( ema, 1.0f );
	CHECK_NEAR( ema, 0.5625f, 1e-7f );

	AdaptationMeasurement ambiguous = *measurement;
	ambiguous.residual = 0.04f;
	ambiguous.noise = -1.0f;
	ambiguous.fbP75 = 2.0f;
	AdaptationState adaptive;
	update_adaptation_state( adaptive, ambiguous, 0.75f, false );
	CHECK_NEAR( adaptive.fbTolerance, 2.5f, 1e-7f );
	CHECK( adaptive.noiseEma == -1.0f );
	CHECK( adaptive.agreementOffset == 0.0f );

	AdaptationState pinned;
	update_adaptation_state( pinned, ambiguous, 0.75f, true );
	CHECK( pinned.fbTolerance == -1.0f );

	stats = {};
	stats[ 0 ] = 100u;
	stats[ 6 ] = 74u;
	stats[ 7 ] = 1u;
	const auto percentileBoundary = decode_adaptation_stats( stats );
	CHECK( percentileBoundary.has_value() );
	CHECK_NEAR( percentileBoundary->fbP75, 0.50f, 1e-7f );
	CHECK( percentileBoundary->noise == -1.0f );

	stats[ 6 ] = 74u;
	stats[ 7 ] = 0u;
	const auto incompleteHistogram = decode_adaptation_stats( stats );
	CHECK( incompleteHistogram.has_value() );
	CHECK( incompleteHistogram->fbP75 == -1.0f );
}

static void test_temporal_policy()
{
	using namespace gamescope::framegen;

	CHECK( present_interval_ns( 20u, 8u ) == 12u );
	CHECK( present_interval_ns( 8u, 20u ) == 0u );
	CHECK( present_interval_ns( 8u, 8u ) == 0u );
	CHECK( !motion_intervals_comparable( 0u, 1u ) );
	CHECK( motion_intervals_comparable( 4u, 1u ) );
	CHECK( motion_intervals_comparable( 1u, 4u ) );
	CHECK( !motion_intervals_comparable( 5u, 1u ) );
	CHECK( !motion_intervals_comparable( 1u, 5u ) );
	CHECK( motion_history_time_scale( 20u, 10u ) == 2.0f );
	CHECK_NEAR( motion_acceleration_time_factor( 20u, 10u ), 2.0f / 3.0f, 1e-6f );

	const SlotRequest measured = classic_slot_request(
		1u, 0u, 4u, 2u, 0.0f, 0.5f, 1.5f );
	CHECK( measured.phase == 0.25f );
	CHECK( measured.strength == 0.25f );
	CHECK( measured.slotIndex == 1u );

	const SlotRequest uniform = classic_slot_request(
		1u, 0u, 4u, 2u, 1.0f, 0.5f, 1.5f );
	CHECK_NEAR( uniform.phase, 1.0f / 3.0f, 1e-6f );
	const SlotRequest halfway = classic_slot_request(
		1u, 0u, 4u, 2u, 0.5f, 0.5f, 1.5f );
	CHECK_NEAR( halfway.phase, ( 0.25f + 1.0f / 3.0f ) * 0.5f, 1e-6f );

	const SlotRequest capped = classic_slot_request(
		2u, 0u, 2u, 1u, 0.0f, 1.0f, 1.5f );
	CHECK( capped.phase == 1.0f );
	CHECK( capped.strength == 1.5f );

	constexpr std::array biases = { 0.0f, 0.5f, 1.0f };
	constexpr std::array strengths = { 0.25f, 0.5f, 1.0f };
	for ( uint32_t generatedCount = 1u; generatedCount <= 3u; generatedCount++ )
	{
		for ( uint32_t gap = generatedCount + 1u; gap <= 8u; gap++ )
		{
			for ( uint32_t i = 0u; i < generatedCount; i++ )
			{
				const uint32_t slot = i + 1u;
				const float gapPhase = static_cast<float>( slot ) / static_cast<float>( gap );
				const float uniformPhase = static_cast<float>( i + 1u )
					/ static_cast<float>( generatedCount + 1u );
				for ( const float bias : biases )
				{
					const float expectedPhase = bias > 0.0f
						? gapPhase + ( uniformPhase - gapPhase ) * bias : gapPhase;
					for ( const float strength : strengths )
					{
						const SlotRequest request = classic_slot_request(
							slot, i, gap, generatedCount, bias, strength, 1.5f );
						CHECK_NEAR( request.phase, expectedPhase, 1e-7f );
						CHECK_NEAR( request.strength,
							std::clamp( expectedPhase * ( strength / 0.5f ), 0.0f, 1.5f ), 1e-7f );
						CHECK( request.slotIndex == slot );
					}
				}
			}
		}
	}

	const TimedPrediction prediction = timed_prediction(
		10'000'000u, 20'000'000u, 0.5f );
	CHECK( prediction.phase == 0.5f );
	CHECK( prediction.rawStrength == 0.5f );
	const JitBookkeeping regular = jit_bookkeeping(
		10'000'000u, 20'000'000u, 10'000'000u );
	CHECK( regular.slotIndex == 1u );
	CHECK( regular.gapVblanks == 2u );
	const JitBookkeeping late = jit_bookkeeping(
		26'000'000u, 20'000'000u, 10'000'000u );
	CHECK( late.slotIndex == 3u );
	CHECK( late.gapVblanks == 4u );
	CHECK( interval_gap_vblanks( 25'000'000u, 10'000'000u ) == 3u );
	CHECK( forward_strength_raw( 2.0f, 0.5f ) > 1.5f );
}

static void test_scheduling_policy()
{
	using namespace gamescope::framegen;

	constexpr uint64_t interval = 10'000'000u;
	CHECK( !leaves_empty_vblank( 20'000'000u, 0u, interval ) );
	CHECK( !leaves_empty_vblank( 24'999'999u, 10'000'000u, interval ) );
	CHECK( leaves_empty_vblank( 25'000'000u, 10'000'000u, interval ) );

	for ( uint32_t confidence = 0u;
		confidence <= k_uCadenceConfidenceRequired; confidence++ )
	{
		CHECK( update_cadence_confidence( confidence, true )
			== std::min( k_uCadenceConfidenceRequired,
				confidence + k_uCadenceConfidenceGain ) );
		CHECK( update_cadence_confidence( confidence, false )
			== ( confidence > k_uCadenceConfidenceLeak
				? confidence - k_uCadenceConfidenceLeak : 0u ) );
		CHECK( reactive_generation_ready( true, confidence )
			== ( confidence >= k_uCadenceConfidenceRequired ) );
		CHECK( !reactive_generation_ready( false, confidence ) );
	}

	CHECK( measured_gap_vblanks( 4'999'999u, interval ) == 1u );
	CHECK( measured_gap_vblanks( 14'999'999u, interval ) == 1u );
	CHECK( measured_gap_vblanks( 15'000'000u, interval ) == 2u );
	CHECK( measured_gap_vblanks( 25'000'000u, interval ) == 3u );

	for ( uint32_t gap = 1u; gap <= 8u; gap++ )
	{
		for ( uint32_t multiplier = 2u; multiplier <= 4u; multiplier++ )
		{
			for ( const bool expandToMultiplier : { false, true } )
			{
				const uint32_t expectedGap = expandToMultiplier
					? std::max( gap, multiplier ) : gap;
				CHECK( expanded_gap_vblanks(
					gap, multiplier, expandToMultiplier ) == expectedGap );
			}

			const uint32_t expectedGenerated = gap > 1u
				? std::min( gap - 1u, multiplier - 1u ) : 0u;
			CHECK( generated_slots_for_gap( gap, multiplier, true )
				== expectedGenerated );
			CHECK( generated_slots_for_gap( gap, multiplier, false )
				== std::min( expectedGenerated, 1u ) );
			CHECK( ladder_generated_count( gap - 1u, multiplier, false )
				== expectedGenerated );
			CHECK( ladder_generated_count( gap - 1u, multiplier, true ) == 1u );
		}
	}

	CHECK( !jit_interval_eligible( 10'999'999u, interval ) );
	CHECK( jit_interval_eligible( 11'000'000u, interval ) );
	CHECK( !vrr_hybrid_interval_eligible( 21'999'999u, interval ) );
	CHECK( vrr_hybrid_interval_eligible( 22'000'000u, interval ) );

	const DeadlineLadderState initial = { 0u, 0u };
	DeadlineLadderEvaluation evaluation = evaluate_deadline_ladder(
		initial, 7u, 8'500'000u, k_uDeadlineMinSamples, interval );
	CHECK( !evaluation.tryDegrade );
	CHECK( evaluation.state.degradeSteps == 0u );
	CHECK( evaluation.state.holdFrames == 0u );
	evaluation = evaluate_deadline_ladder(
		initial, 7u, 8'500'001u, k_uDeadlineMinSamples, interval );
	CHECK( evaluation.tryDegrade );

	for ( uint32_t samples = 0u; samples < k_uDeadlineMinSamples; samples++ )
		CHECK( !evaluate_deadline_ladder(
			initial, 7u, interval, samples, interval ).tryDegrade );
	CHECK( !evaluate_deadline_ladder(
		initial, 0u, interval, k_uDeadlineMinSamples, interval ).tryDegrade );
	CHECK( !evaluate_deadline_ladder(
		initial, 7u, 0u, k_uDeadlineMinSamples, interval ).tryDegrade );
	CHECK( !evaluate_deadline_ladder(
		{ 7u, 0u }, 7u, interval, k_uDeadlineMinSamples, interval ).tryDegrade );

	evaluation = evaluate_deadline_ladder(
		{ 2u, 3u }, 7u, interval, k_uDeadlineMinSamples, interval );
	CHECK( !evaluation.tryDegrade );
	CHECK( evaluation.state.degradeSteps == 2u );
	CHECK( evaluation.state.holdFrames == 2u );
	DeadlineLadderState committed = commit_deadline_degradation( evaluation.state );
	CHECK( committed.degradeSteps == 3u );
	CHECK( committed.holdFrames == k_uDeadlineHoldFrames );

	const EffectiveConfig motionHigh = {
		GamescopeFramegenMode::Motion, 4u, GamescopeFramegenQuality::High };
	const EffectiveConfig motionMedium = {
		GamescopeFramegenMode::Motion, 4u, GamescopeFramegenQuality::Medium };
	const EffectiveConfig extrapolate4 = {
		GamescopeFramegenMode::Extrapolate, 4u, GamescopeFramegenQuality::Low };
	const EffectiveConfig extrapolate3 = {
		GamescopeFramegenMode::Extrapolate, 3u, GamescopeFramegenQuality::Low };
	CHECK( degradation_reduces_work( motionHigh, motionMedium, 3u, 3u ) );
	CHECK( degradation_reduces_work( motionMedium, extrapolate4, 3u, 3u ) );
	CHECK( degradation_reduces_work( extrapolate4, extrapolate3, 3u, 2u ) );
	CHECK( !degradation_reduces_work( extrapolate4, extrapolate3, 1u, 1u ) );
}

static void test_dispatch_policy()
{
	using gamescope::framegen::select_dispatch_policy;
	constexpr std::array<uint32_t, 3> vendors = { 0u, 0x1002u, 0x10deu };

	for ( uint32_t mask = 0u; mask < 16u; mask++ )
	{
		const bool supportsFp16 = ( mask & 1u ) != 0u;
		const bool floatOutput = ( mask & 2u ) != 0u;
		const bool supportsR16F = ( mask & 4u ) != 0u;
		const bool supportsRGBA16F = ( mask & 8u ) != 0u;
		for ( const uint32_t vendor : vendors )
		{
			const auto policy = select_dispatch_policy( supportsFp16, floatOutput,
				supportsR16F, supportsRGBA16F, vendor );
			const bool baseFp16 = supportsFp16 && !floatOutput;
			const bool direct = vendor == 0x10deu;
			CHECK( policy.useFp16 == ( baseFp16 && !direct ) );
			CHECK( policy.pairUseFp16 == baseFp16 );
			CHECK( policy.useR16FLuma == supportsR16F );
			CHECK( policy.motionSupported == supportsRGBA16F );
			CHECK( policy.preferDirectExtrapolate == direct );
		}
	}

	const auto genericInt = select_dispatch_policy( true, false, true, true, 0x1002u );
	CHECK( genericInt.useFp16 );
	CHECK( genericInt.pairUseFp16 );
	CHECK( genericInt.useR16FLuma );
	CHECK( genericInt.motionSupported );
	CHECK( !genericInt.preferDirectExtrapolate );

	const auto genericFloat = select_dispatch_policy( true, true, false, true, 0x8086u );
	CHECK( !genericFloat.useFp16 );
	CHECK( !genericFloat.pairUseFp16 );
	CHECK( !genericFloat.useR16FLuma );
	CHECK( genericFloat.motionSupported );
	CHECK( !genericFloat.preferDirectExtrapolate );

	const auto nvidiaInt = select_dispatch_policy( true, false, true, true, 0x10deu );
	CHECK( !nvidiaInt.useFp16 );
	CHECK( nvidiaInt.pairUseFp16 );
	CHECK( nvidiaInt.preferDirectExtrapolate );

	const auto nvidiaNoFp16 = select_dispatch_policy( false, false, true, false, 0x10deu );
	CHECK( !nvidiaNoFp16.useFp16 );
	CHECK( !nvidiaNoFp16.pairUseFp16 );
	CHECK( !nvidiaNoFp16.motionSupported );
	CHECK( nvidiaNoFp16.preferDirectExtrapolate );
}

static void test_push_constant_encoding()
{
	const FramegenMotionAccelPush_t accel(
		1.25f, 0.08f, 0.40f, 4.0f, 0.12f, 0.45f, 0.5f,
		true, false, true, false, 0.75f, 0.25f );
	CHECK( accel.strength == 1.25f );
	CHECK( accel.historyValid == 1.0f );
	CHECK( accel.guidedReconstruction == 0.0f );
	CHECK( accel.reservoirValid == 1.0f );
	CHECK( accel.shadingValid == 0.0f );
	CHECK( accel.historyFlowScale == 0.75f );
	CHECK( accel.accelTimeFactor == 0.25f );

	const FramegenMotionNetPush_t conservativeNet( 1.0f, true );
	CHECK( conservativeNet.mode == 1.0f );
	CHECK( conservativeNet.flowScale == 0.0f );
	CHECK( conservativeNet.confidenceRaiseScale == 0.0f );
	CHECK( conservativeNet.shadingScale == 0.0f );

	const FramegenMotionNetTrainPush_t train( 7u, 8u, true, true, true, 0.5f );
	CHECK( train.seed == 7u );
	CHECK( train.sliceBase == 8u );
	CHECK( train.flags == 7u );
	CHECK( train.historyTimeScale == 0.5f );
}

int main()
{
	test_degradation_policy();
	test_learned_net_layout();
	test_names();
	test_net_profile_contract();
	test_numeric_settings_contract();
	test_adaptation_policy();
	test_temporal_policy();
	test_scheduling_policy();
	test_dispatch_policy();
	test_push_constant_encoding();
	return g_bPassed ? 0 : 1;
}

#undef CHECK_NEAR
#undef CHECK
