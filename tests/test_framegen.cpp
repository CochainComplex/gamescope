#include "framegen/net_layout.hpp"
#include "framegen/policy.hpp"
#include "framegen/push_constants.hpp"

#include <algorithm>
#include <array>
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
	test_push_constant_encoding();
	return g_bPassed ? 0 : 1;
}

#undef CHECK
