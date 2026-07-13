#pragma once

#include <algorithm>
#include <cstdint>
#include <optional>
#include <span>

namespace gamescope::framegen
{

// Shared B4 counter and policy contract. The GPU writes exactly this many
// R32_UINT values; the renderer copies them only after the submission retires.
inline constexpr uint32_t k_uAdaptationStatsCount = 96u;
inline constexpr uint32_t k_uAdaptationMaxTotal = 4u * 1024u * 1024u;

// Same-batch field trust and next-batch CPU adaptation thresholds. Keep these
// together so resource sizing, push constants, and readback policy cannot drift
// into parallel definitions in the renderer.
inline constexpr float k_flAdaptationTrustLo = 0.15f;
inline constexpr float k_flAdaptationTrustHi = 0.45f;
inline constexpr float k_flAdaptationBadResidual = 0.10f;
inline constexpr float k_flAdaptationResidualLow = 0.045f;
inline constexpr float k_flAdaptationNoiseToAgreement = 6.0f;
inline constexpr float k_flAdaptationAgreementOffsetMax = 0.15f;

struct AdaptationMeasurement
{
	float residual;
	float badFraction;
	float killedFraction;
	float motionMean;
	// Negative means that this readback carried no usable sample for the EMA.
	float noise;
	float fbP75;
	uint32_t sceneCut;
	uint32_t changedSections;
	uint32_t sceneHistogramDistanceQ10;
};

struct AdaptationState
{
	float residualEma = -1.0f;
	float noiseEma = -1.0f;
	float fbP75Ema = -1.0f;
	// Negative means that the static/manual base tolerance remains authoritative.
	float fbTolerance = -1.0f;
	float agreementOffset = 0.0f;
};

// Decode the completed GPU counter image without acquiring any ownership of
// the mapped memory. Counter indices are the public B4 layout documented by
// cs_framegen_motion_stats.comp. Invalid total counts reject the entire sample;
// absent noise and percentile evidence use the existing negative sentinel.
[[nodiscard]] inline std::optional<AdaptationMeasurement> decode_adaptation_stats(
	std::span<const uint32_t, k_uAdaptationStatsCount> stats )
{
	const uint32_t total = stats[ 0 ];
	if ( total == 0u || total > k_uAdaptationMaxTotal )
		return std::nullopt;

	const uint32_t alive = std::max(
		total - std::min( stats[ 3 ], total ), 1u );
	const float residual = ( static_cast<float>( stats[ 1 ] ) / 1024.0f )
		/ static_cast<float>( total );
	const float badFraction = static_cast<float>( stats[ 2 ] )
		/ static_cast<float>( alive );
	const float killedFraction = static_cast<float>( stats[ 3 ] )
		/ static_cast<float>( total );
	const float motionMean = ( static_cast<float>( stats[ 14 ] ) / 64.0f )
		/ static_cast<float>( total );
	const float noise = stats[ 5 ] >= 64u
		? ( static_cast<float>( stats[ 4 ] ) / 1024.0f )
			/ static_cast<float>( stats[ 5 ] )
		: -1.0f;

	// Upper edge of the first 0.25-texel bin whose cumulative count reaches
	// the 75th percentile. The shader emits one histogram sample per texel, so
	// total is the established percentile denominator.
	float fbP75 = -1.0f;
	uint32_t cumulative = 0u;
	for ( uint32_t bin = 0u; bin < 8u; bin++ )
	{
		cumulative += stats[ 6u + bin ];
		if ( cumulative * 4u >= total * 3u )
		{
			fbP75 = 0.25f * static_cast<float>( bin + 1u );
			break;
		}
	}

	return AdaptationMeasurement{
		.residual = residual,
		.badFraction = badFraction,
		.killedFraction = killedFraction,
		.motionMean = motionMean,
		.noise = noise,
		.fbP75 = fbP75,
		.sceneCut = stats[ 88 ],
		.changedSections = stats[ 89 ],
		.sceneHistogramDistanceQ10 = stats[ 90 ],
	};
}

[[nodiscard]] constexpr float scene_histogram_distance(
	const AdaptationMeasurement &measurement )
{
	return static_cast<float>( measurement.sceneHistogramDistanceQ10 ) / 1024.0f;
}

inline void fold_adaptation_ema( float &ema, float sample )
{
	if ( sample < 0.0f )
		return;
	ema = ema < 0.0f ? sample : ema + ( sample - ema ) / 8.0f;
}

// Fold one valid measurement and derive the thresholds recorded by the next
// batch. This is pure CPU policy: completion, scene lifetime, and application
// of these values to Vulkan push constants remain renderer responsibilities.
inline void update_adaptation_state( AdaptationState &state,
	const AdaptationMeasurement &measurement, float baseTolerance,
	bool tolerancePinned )
{
	fold_adaptation_ema( state.residualEma, measurement.residual );
	fold_adaptation_ema( state.noiseEma, measurement.noise );
	fold_adaptation_ema( state.fbP75Ema, measurement.fbP75 );

	float tolerance = -1.0f;
	if ( !tolerancePinned
		&& state.residualEma >= 0.0f
		&& state.residualEma < k_flAdaptationResidualLow
		&& state.fbP75Ema > baseTolerance )
	{
		tolerance = std::min( state.fbP75Ema * 1.5f, 2.5f );
	}
	state.fbTolerance = tolerance;

	state.agreementOffset = state.noiseEma > 0.0f
		? std::clamp( k_flAdaptationNoiseToAgreement * state.noiseEma,
			0.0f, k_flAdaptationAgreementOffsetMax )
		: 0.0f;
}

[[nodiscard]] constexpr float active_agreement_lo(
	const AdaptationState &state, float baseAgreementLo )
{
	return baseAgreementLo + state.agreementOffset;
}

[[nodiscard]] constexpr float active_agreement_hi(
	const AdaptationState &state, float baseAgreementHi )
{
	return baseAgreementHi + 2.0f * state.agreementOffset;
}

static_assert( k_uAdaptationStatsCount == 96u,
	"B4 CPU readback must match the shader counter image" );

} // namespace gamescope::framegen
