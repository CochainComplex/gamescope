#include "framegen/dispatch_policy.hpp"
#include "framegen/net_layout.hpp"
#include "framegen/policy.hpp"
#include "framegen/push_constants.hpp"
#include "framegen/temporal.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>

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
	test_temporal_policy();
	test_dispatch_policy();
	test_push_constant_encoding();
	return g_bPassed ? 0 : 1;
}

#undef CHECK_NEAR
#undef CHECK
