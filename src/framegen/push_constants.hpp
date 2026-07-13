#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>

// Push constants for the extrapolate shaders. `strength` is the effective
// per-slot forward coefficient: the CPU folds this slot's temporal placement
// (where it lands between the two real frames) and --framegen-strength into a
// single value, so the shader stays slot-agnostic.
struct FramegenPushData_t
{
	float strength;
	float suppressLo;
	float suppressHi;
	float pad;

	explicit FramegenPushData_t( float flStrength, float flLo, float flHi )
		: strength( flStrength )
		, suppressLo( flLo )
		, suppressHi( flHi )
		, pad( 0.0f )
	{
	}
};

// Paired extrapolation: two slot coefficients in one dispatch, so the two 4K
// history reads are shared instead of repeated per generated frame.
struct FramegenPairPushData_t
{
	float strength0;
	float strength1;
	float suppressLo;
	float suppressHi;

	FramegenPairPushData_t( float flStrength0, float flStrength1, float flLo, float flHi )
		: strength0( flStrength0 )
		, strength1( flStrength1 )
		, suppressLo( flLo )
		, suppressHi( flHi )
	{
	}
};

// Blend mode: per-slot temporal placement as the crossfade weight, so x3/x4
// produces distinct intermediate frames instead of identical 0.5 duplicates.
struct FramegenBlendPushData_t
{
	float phase;
	float pad0, pad1, pad2;

	explicit FramegenBlendPushData_t( float flPhase )
		: phase( flPhase ), pad0( 0.0f ), pad1( 0.0f ), pad2( 0.0f )
	{
	}
};

// Push constants for the motion-mode block-match and warp passes.
struct FramegenMotionMatchPush_t
{
	int32_t searchRadius;
	int32_t pad0, pad1, pad2;

	explicit FramegenMotionMatchPush_t( int32_t nSearchRadius )
		: searchRadius( nSearchRadius ), pad0( 0 ), pad1( 0 ), pad2( 0 )
	{
	}
};

struct FramegenMotionWarpPush_t
{
	float strength;
	float suppressLo;
	float suppressHi;
	float lowResScale;
	float agreeLo;
	float agreeHi;

	explicit FramegenMotionWarpPush_t( float flStrength, float flLo, float flHi,
		float flScale, float flAgreeLo, float flAgreeHi )
		: strength( flStrength ), suppressLo( flLo ), suppressHi( flHi ), lowResScale( flScale )
		, agreeLo( flAgreeLo ), agreeHi( flAgreeHi )
	{
	}
};

struct FramegenMotionAccelPush_t
{
	float strength;
	float suppressLo;
	float suppressHi;
	float lowResScale;
	float agreeLo;
	float agreeHi;
	float accelMax;
	float historyValid;
	float guidedReconstruction;
	float reservoirValid;
	float shadingValid;
	// The retained field is displacement over the preceding real-frame
	// interval, not a velocity. Normalize it to the current interval before
	// differencing, then use the irregular-sample quadratic coefficient below.
	float historyFlowScale;
	float accelTimeFactor;

	FramegenMotionAccelPush_t( float flStrength, float flLo, float flHi, float flScale,
		float flAgreeLo, float flAgreeHi, float flAccelMax, bool bHistoryValid,
		bool bGuidedReconstruction, bool bReservoirValid, bool bShadingValid,
		float flHistoryFlowScale, float flAccelTimeFactor )
		: strength( flStrength ), suppressLo( flLo ), suppressHi( flHi ), lowResScale( flScale )
		, agreeLo( flAgreeLo ), agreeHi( flAgreeHi ), accelMax( flAccelMax )
		, historyValid( bHistoryValid ? 1.0f : 0.0f )
		, guidedReconstruction( bGuidedReconstruction ? 1.0f : 0.0f )
		, reservoirValid( bReservoirValid ? 1.0f : 0.0f )
		, shadingValid( bShadingValid ? 1.0f : 0.0f )
		, historyFlowScale( flHistoryFlowScale ), accelTimeFactor( flAccelTimeFactor )
	{
	}
};

struct FramegenMotionRefinePush_t
{
	int32_t finalLevel;
	int32_t pad0, pad1, pad2;

	explicit FramegenMotionRefinePush_t( bool bFinalLevel )
		: finalLevel( bFinalLevel ? 1 : 0 ), pad0( 0 ), pad1( 0 ), pad2( 0 )
	{
	}
};

struct FramegenMotionFBCheckPush_t
{
	float tolBase;
	float tolSlope;
	float pad0, pad1;

	explicit FramegenMotionFBCheckPush_t( float flTolBase, float flTolSlope )
		: tolBase( flTolBase ), tolSlope( flTolSlope ), pad0( 0.0f ), pad1( 0.0f )
	{
	}
};

// Bidirectional interpolation warp: temporal phase within the completed real
// interval plus the field scale and agreement window used by the forward warp.
struct FramegenMotionBidirPush_t
{
	float phase;
	float lowResScale;
	float agreeLo;
	float agreeHi;
	float oneSidedStrength;
	float endpointTraceStrength;
	float pad0, pad1;

	explicit FramegenMotionBidirPush_t( float flPhase, float flScale,
		float flAgreeLo, float flAgreeHi, float flOneSidedStrength,
		float flEndpointTraceStrength )
		: phase( flPhase ), lowResScale( flScale ), agreeLo( flAgreeLo ), agreeHi( flAgreeHi )
		, oneSidedStrength( flOneSidedStrength ), endpointTraceStrength( flEndpointTraceStrength )
		, pad0( 0.0f ), pad1( 0.0f )
	{
	}
};

// Self-supervised stats probe thresholds. The constructor remains beside the
// tuning constants in rendervulkan.cpp; only its GLSL-facing layout lives here.
struct FramegenMotionStatsPush_t
{
	uint32_t clearOnly;
	float badThresh;
	float staticMvMax;
	float minConfSurvive;

	explicit FramegenMotionStatsPush_t( bool bClearOnly );
};

// Window over the measured mispredicting fraction that scales field confidence.
struct FramegenMotionStatsApplyPush_t
{
	float trustLo;
	float trustHi;
	float sceneHistSection;
	float sceneHistGlobal;
	float sceneResid;
	float sceneUnreliable;
	uint32_t sceneMinSections;
	uint32_t detectOnly;

	FramegenMotionStatsApplyPush_t( float flTrustLo, float flTrustHi, bool bDetectOnly )
		: trustLo( flTrustLo )
		, trustHi( flTrustHi )
		, sceneHistSection( 0.35f )
		, sceneHistGlobal( 0.25f )
		, sceneResid( 0.12f )
		, sceneUnreliable( 0.55f )
		, sceneMinSections( 7u )
		, detectOnly( bDetectOnly ? 1u : 0u )
	{
	}
};

// Learned field refinement. Conservative bidir may suppress geometry,
// confidence raises, and causal shading without changing the feature encoding.
struct FramegenMotionNetPush_t
{
	float mode;
	float flowScale;
	float confidenceRaiseScale;
	float shadingScale;

	FramegenMotionNetPush_t( float flMode, bool bConservativeBidir )
		: mode( flMode )
		, flowScale( bConservativeBidir ? 0.0f : 1.0f )
		, confidenceRaiseScale( bConservativeBidir ? 0.0f : 1.0f )
		, shadingScale( bConservativeBidir ? 0.0f : 1.0f )
	{
	}
};

// In-situ gradient pass: tile seed and the slice rows owned by this direction.
struct FramegenMotionNetTrainPush_t
{
	uint32_t seed;
	uint32_t sliceBase;
	float mode;
	uint32_t flags;
	float historyTimeScale;

	FramegenMotionNetTrainPush_t( uint32_t uSeed, uint32_t uSliceBase, bool bSceneCutGuard,
		bool bShadingHistoryValid, bool bConservativeBidir, float flHistoryTimeScale )
		: seed( uSeed ), sliceBase( uSliceBase ), mode( 0.0f )
		, flags( ( bSceneCutGuard ? 1u : 0u )
			| ( bShadingHistoryValid ? 2u : 0u )
			| ( bConservativeBidir ? 4u : 0u ) )
		, historyTimeScale( flHistoryTimeScale )
	{
	}
};

// In-situ optimizer: step size, served-weight EMA, prior decay, and step count.
struct FramegenMotionNetOptPush_t
{
	float lr;
	float emaAlpha;
	float decay;
	uint32_t step;

	FramegenMotionNetOptPush_t( float flLr, float flEmaAlpha, float flDecay, uint32_t uStep )
		: lr( flLr ), emaAlpha( flEmaAlpha ), decay( flDecay ), step( uStep )
	{
	}
};

// These types are copied verbatim into GLSL push-constant blocks. A CPU-side
// member reorder must fail compilation until the corresponding shader ABI is
// deliberately updated.
#define FRAMEGEN_PUSH_SIZE( type, size ) \
	static_assert( sizeof( type ) == size, #type " size must match its GLSL push-constant block" )
#define FRAMEGEN_PUSH_MEMBER( type, member, offset ) \
	static_assert( offsetof( type, member ) == offset, #type "::" #member " must match its GLSL push-constant block" )
#define FRAMEGEN_PUSH_PROPERTIES( type ) \
	static_assert( std::is_standard_layout_v<type> && std::is_trivially_copyable_v<type>, \
		#type " must remain directly copyable into Vulkan push constants" )

FRAMEGEN_PUSH_SIZE( FramegenPushData_t, 16 );
FRAMEGEN_PUSH_PROPERTIES( FramegenPushData_t );
FRAMEGEN_PUSH_MEMBER( FramegenPushData_t, strength, 0 );
FRAMEGEN_PUSH_MEMBER( FramegenPushData_t, suppressLo, 4 );
FRAMEGEN_PUSH_MEMBER( FramegenPushData_t, suppressHi, 8 );

FRAMEGEN_PUSH_SIZE( FramegenPairPushData_t, 16 );
FRAMEGEN_PUSH_PROPERTIES( FramegenPairPushData_t );
FRAMEGEN_PUSH_MEMBER( FramegenPairPushData_t, strength0, 0 );
FRAMEGEN_PUSH_MEMBER( FramegenPairPushData_t, strength1, 4 );
FRAMEGEN_PUSH_MEMBER( FramegenPairPushData_t, suppressLo, 8 );
FRAMEGEN_PUSH_MEMBER( FramegenPairPushData_t, suppressHi, 12 );

FRAMEGEN_PUSH_SIZE( FramegenBlendPushData_t, 16 );
FRAMEGEN_PUSH_PROPERTIES( FramegenBlendPushData_t );
FRAMEGEN_PUSH_MEMBER( FramegenBlendPushData_t, phase, 0 );

FRAMEGEN_PUSH_SIZE( FramegenMotionMatchPush_t, 16 );
FRAMEGEN_PUSH_PROPERTIES( FramegenMotionMatchPush_t );
FRAMEGEN_PUSH_MEMBER( FramegenMotionMatchPush_t, searchRadius, 0 );

FRAMEGEN_PUSH_SIZE( FramegenMotionWarpPush_t, 24 );
FRAMEGEN_PUSH_PROPERTIES( FramegenMotionWarpPush_t );
FRAMEGEN_PUSH_MEMBER( FramegenMotionWarpPush_t, strength, 0 );
FRAMEGEN_PUSH_MEMBER( FramegenMotionWarpPush_t, suppressLo, 4 );
FRAMEGEN_PUSH_MEMBER( FramegenMotionWarpPush_t, suppressHi, 8 );
FRAMEGEN_PUSH_MEMBER( FramegenMotionWarpPush_t, lowResScale, 12 );
FRAMEGEN_PUSH_MEMBER( FramegenMotionWarpPush_t, agreeLo, 16 );
FRAMEGEN_PUSH_MEMBER( FramegenMotionWarpPush_t, agreeHi, 20 );

FRAMEGEN_PUSH_SIZE( FramegenMotionAccelPush_t, 52 );
FRAMEGEN_PUSH_PROPERTIES( FramegenMotionAccelPush_t );
FRAMEGEN_PUSH_MEMBER( FramegenMotionAccelPush_t, strength, 0 );
FRAMEGEN_PUSH_MEMBER( FramegenMotionAccelPush_t, suppressLo, 4 );
FRAMEGEN_PUSH_MEMBER( FramegenMotionAccelPush_t, suppressHi, 8 );
FRAMEGEN_PUSH_MEMBER( FramegenMotionAccelPush_t, lowResScale, 12 );
FRAMEGEN_PUSH_MEMBER( FramegenMotionAccelPush_t, agreeLo, 16 );
FRAMEGEN_PUSH_MEMBER( FramegenMotionAccelPush_t, agreeHi, 20 );
FRAMEGEN_PUSH_MEMBER( FramegenMotionAccelPush_t, accelMax, 24 );
FRAMEGEN_PUSH_MEMBER( FramegenMotionAccelPush_t, historyValid, 28 );
FRAMEGEN_PUSH_MEMBER( FramegenMotionAccelPush_t, guidedReconstruction, 32 );
FRAMEGEN_PUSH_MEMBER( FramegenMotionAccelPush_t, reservoirValid, 36 );
FRAMEGEN_PUSH_MEMBER( FramegenMotionAccelPush_t, shadingValid, 40 );
FRAMEGEN_PUSH_MEMBER( FramegenMotionAccelPush_t, historyFlowScale, 44 );
FRAMEGEN_PUSH_MEMBER( FramegenMotionAccelPush_t, accelTimeFactor, 48 );

FRAMEGEN_PUSH_SIZE( FramegenMotionRefinePush_t, 16 );
FRAMEGEN_PUSH_PROPERTIES( FramegenMotionRefinePush_t );
FRAMEGEN_PUSH_MEMBER( FramegenMotionRefinePush_t, finalLevel, 0 );

FRAMEGEN_PUSH_SIZE( FramegenMotionFBCheckPush_t, 16 );
FRAMEGEN_PUSH_PROPERTIES( FramegenMotionFBCheckPush_t );
FRAMEGEN_PUSH_MEMBER( FramegenMotionFBCheckPush_t, tolBase, 0 );
FRAMEGEN_PUSH_MEMBER( FramegenMotionFBCheckPush_t, tolSlope, 4 );

FRAMEGEN_PUSH_SIZE( FramegenMotionBidirPush_t, 32 );
FRAMEGEN_PUSH_PROPERTIES( FramegenMotionBidirPush_t );
FRAMEGEN_PUSH_MEMBER( FramegenMotionBidirPush_t, phase, 0 );
FRAMEGEN_PUSH_MEMBER( FramegenMotionBidirPush_t, lowResScale, 4 );
FRAMEGEN_PUSH_MEMBER( FramegenMotionBidirPush_t, agreeLo, 8 );
FRAMEGEN_PUSH_MEMBER( FramegenMotionBidirPush_t, agreeHi, 12 );
FRAMEGEN_PUSH_MEMBER( FramegenMotionBidirPush_t, oneSidedStrength, 16 );
FRAMEGEN_PUSH_MEMBER( FramegenMotionBidirPush_t, endpointTraceStrength, 20 );

FRAMEGEN_PUSH_SIZE( FramegenMotionStatsPush_t, 16 );
FRAMEGEN_PUSH_PROPERTIES( FramegenMotionStatsPush_t );
FRAMEGEN_PUSH_MEMBER( FramegenMotionStatsPush_t, clearOnly, 0 );
FRAMEGEN_PUSH_MEMBER( FramegenMotionStatsPush_t, badThresh, 4 );
FRAMEGEN_PUSH_MEMBER( FramegenMotionStatsPush_t, staticMvMax, 8 );
FRAMEGEN_PUSH_MEMBER( FramegenMotionStatsPush_t, minConfSurvive, 12 );

FRAMEGEN_PUSH_SIZE( FramegenMotionStatsApplyPush_t, 32 );
FRAMEGEN_PUSH_PROPERTIES( FramegenMotionStatsApplyPush_t );
FRAMEGEN_PUSH_MEMBER( FramegenMotionStatsApplyPush_t, trustLo, 0 );
FRAMEGEN_PUSH_MEMBER( FramegenMotionStatsApplyPush_t, trustHi, 4 );
FRAMEGEN_PUSH_MEMBER( FramegenMotionStatsApplyPush_t, sceneHistSection, 8 );
FRAMEGEN_PUSH_MEMBER( FramegenMotionStatsApplyPush_t, sceneHistGlobal, 12 );
FRAMEGEN_PUSH_MEMBER( FramegenMotionStatsApplyPush_t, sceneResid, 16 );
FRAMEGEN_PUSH_MEMBER( FramegenMotionStatsApplyPush_t, sceneUnreliable, 20 );
FRAMEGEN_PUSH_MEMBER( FramegenMotionStatsApplyPush_t, sceneMinSections, 24 );
FRAMEGEN_PUSH_MEMBER( FramegenMotionStatsApplyPush_t, detectOnly, 28 );

FRAMEGEN_PUSH_SIZE( FramegenMotionNetPush_t, 16 );
FRAMEGEN_PUSH_PROPERTIES( FramegenMotionNetPush_t );
FRAMEGEN_PUSH_MEMBER( FramegenMotionNetPush_t, mode, 0 );
FRAMEGEN_PUSH_MEMBER( FramegenMotionNetPush_t, flowScale, 4 );
FRAMEGEN_PUSH_MEMBER( FramegenMotionNetPush_t, confidenceRaiseScale, 8 );
FRAMEGEN_PUSH_MEMBER( FramegenMotionNetPush_t, shadingScale, 12 );

FRAMEGEN_PUSH_SIZE( FramegenMotionNetTrainPush_t, 20 );
FRAMEGEN_PUSH_PROPERTIES( FramegenMotionNetTrainPush_t );
FRAMEGEN_PUSH_MEMBER( FramegenMotionNetTrainPush_t, seed, 0 );
FRAMEGEN_PUSH_MEMBER( FramegenMotionNetTrainPush_t, sliceBase, 4 );
FRAMEGEN_PUSH_MEMBER( FramegenMotionNetTrainPush_t, mode, 8 );
FRAMEGEN_PUSH_MEMBER( FramegenMotionNetTrainPush_t, flags, 12 );
FRAMEGEN_PUSH_MEMBER( FramegenMotionNetTrainPush_t, historyTimeScale, 16 );

FRAMEGEN_PUSH_SIZE( FramegenMotionNetOptPush_t, 16 );
FRAMEGEN_PUSH_PROPERTIES( FramegenMotionNetOptPush_t );
FRAMEGEN_PUSH_MEMBER( FramegenMotionNetOptPush_t, lr, 0 );
FRAMEGEN_PUSH_MEMBER( FramegenMotionNetOptPush_t, emaAlpha, 4 );
FRAMEGEN_PUSH_MEMBER( FramegenMotionNetOptPush_t, decay, 8 );
FRAMEGEN_PUSH_MEMBER( FramegenMotionNetOptPush_t, step, 12 );

#undef FRAMEGEN_PUSH_MEMBER
#undef FRAMEGEN_PUSH_PROPERTIES
#undef FRAMEGEN_PUSH_SIZE
