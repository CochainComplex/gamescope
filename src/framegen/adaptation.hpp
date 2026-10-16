#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <optional>
#include <span>

namespace gamescope::framegen
{

// Shared adaptation counter and policy contract. The GPU writes exactly this many
// R32_UINT values; the renderer copies them only after the submission retires.
inline constexpr uint32_t k_uAdaptationStatsCount = 96u;
inline constexpr uint32_t k_uAdaptationMaxTotal = 4u * 1024u * 1024u;
// Moving-texel counters of the quality meter; the layout is documented in
// cs_framegen_motion_stats.comp.
inline constexpr uint32_t k_uAdaptationStatMoving = 15u;
inline constexpr uint32_t k_uAdaptationStatMovingKilled = 91u;
inline constexpr uint32_t k_uAdaptationStatMovingBad = 92u;
inline constexpr uint32_t k_uAdaptationStatMovingKilledMatch = 93u;
inline constexpr uint32_t k_uAdaptationStatMovingResidual = 94u;

// Same-batch field trust and next-batch CPU adaptation thresholds. Keep these
// together so resource sizing, push constants, and readback policy cannot drift
// into parallel definitions in the renderer.
inline constexpr float k_flAdaptationTrustLo = 0.15f;
inline constexpr float k_flAdaptationTrustHi = 0.45f;
inline constexpr float k_flAdaptationBadResidual = 0.10f;
// Enter and leave FB-tolerance relaxation through different residual
// thresholds. The old single threshold could alternate the tolerance between
// 0.75 and 2.5 field texels on adjacent real frames, visibly changing the
// confidence mask even though every input statistic was already smoothed.
inline constexpr float k_flAdaptationResidualRelaxEnter = 0.040f;
inline constexpr float k_flAdaptationResidualRelaxExit = 0.055f;
// Raising tolerance grants motion vectors more authority, so do it more
// cautiously than revoking that authority. These are field texels per real
// frame; at the 1/8-resolution estimator they bound the confidence-boundary
// motion to one and two output pixels respectively per observation.
inline constexpr float k_flAdaptationToleranceRisePerFrame = 0.125f;
inline constexpr float k_flAdaptationToleranceFallPerFrame = 0.250f;
inline constexpr float k_flAdaptationNoiseToAgreement = 6.0f;
inline constexpr float k_flAdaptationAgreementOffsetMax = 0.15f;

// Motion-field quality over moving texels, summed over stats-probe samples.
// A static background predicts itself whatever the field says, so only moving
// texels judge the field. A window reports ratios of sums, so every moving
// texel weighs the same whatever sample it came from. Each ratio is 0 when its
// denominator is empty; samples == 0 means nothing was measured.
struct MotionQualityCounts
{
	uint64_t samples = 0;
	uint64_t moving = 0;
	// Final field confidence below the survive threshold.
	uint64_t movingKilled = 0;
	// The killed texels the matcher had already left unconfident; the rest
	// were killed by the later consistency check or refinement.
	uint64_t movingKilledMatch = 0;
	// Survived, yet still mispredicted the real frame.
	uint64_t movingBad = 0;
	// Warped-luma residual over moving texels, x1024.
	uint64_t movingResidualQ10 = 0;

	MotionQualityCounts &operator+=( const MotionQualityCounts &other )
	{
		samples += other.samples;
		moving += other.moving;
		movingKilled += other.movingKilled;
		movingKilledMatch += other.movingKilledMatch;
		movingBad += other.movingBad;
		movingResidualQ10 += other.movingResidualQ10;
		return *this;
	}

	[[nodiscard]] double residual() const
	{
		return moving != 0 ? static_cast<double>( movingResidualQ10 ) / 1024.0
			/ static_cast<double>( moving ) : 0.0;
	}
	// Of the moving texels that survived.
	[[nodiscard]] double badFraction() const
	{
		const uint64_t survived = moving - std::min( movingKilled, moving );
		return survived != 0 ? static_cast<double>( movingBad )
			/ static_cast<double>( survived ) : 0.0;
	}
	[[nodiscard]] double killedFraction() const { return fraction( movingKilled ); }
	[[nodiscard]] double killedMatchFraction() const { return fraction( movingKilledMatch ); }
	[[nodiscard]] double killedFbFraction() const
	{
		return fraction( movingKilled - std::min( movingKilledMatch, movingKilled ) );
	}

private:
	[[nodiscard]] double fraction( uint64_t count ) const
	{
		return moving != 0 ? static_cast<double>( count )
			/ static_cast<double>( moving ) : 0.0;
	}
};

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
	// One quality-meter sample (samples == 1).
	MotionQualityCounts quality;
};

struct AdaptationState
{
	float residualEma = -1.0f;
	float noiseEma = -1.0f;
	float fbP75Ema = -1.0f;
	// Negative means that the static/manual base tolerance remains authoritative.
	float fbTolerance = -1.0f;
	bool fbRelaxationActive = false;
	float agreementOffset = 0.0f;
};

// Decode the completed GPU counter image without acquiring any ownership of
// the mapped memory. Counter indices are the public adaptation layout documented by
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

	// Each moving counter is clamped to its parent set, so a window's parts
	// always add up.
	const uint32_t moving = std::min( stats[ k_uAdaptationStatMoving ], total );
	const uint32_t movingKilled = std::min( stats[ k_uAdaptationStatMovingKilled ], moving );
	const MotionQualityCounts quality = {
		.samples = 1u,
		.moving = moving,
		.movingKilled = movingKilled,
		.movingKilledMatch = std::min( stats[ k_uAdaptationStatMovingKilledMatch ], movingKilled ),
		.movingBad = std::min( stats[ k_uAdaptationStatMovingBad ], moving - movingKilled ),
		.movingResidualQ10 = stats[ k_uAdaptationStatMovingResidual ],
	};

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
		.quality = quality,
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

	if ( tolerancePinned )
	{
		state.fbTolerance = -1.0f;
		state.fbRelaxationActive = false;
	}
	else
	{
		const bool hasRelaxationEvidence = state.residualEma >= 0.0f
			&& state.fbP75Ema > baseTolerance;
		if ( state.fbRelaxationActive )
		{
			state.fbRelaxationActive = hasRelaxationEvidence
				&& state.residualEma < k_flAdaptationResidualRelaxExit;
		}
		else
		{
			state.fbRelaxationActive = hasRelaxationEvidence
				&& state.residualEma <= k_flAdaptationResidualRelaxEnter;
		}

		const float targetTolerance = state.fbRelaxationActive
			? std::min( state.fbP75Ema * 1.5f, 2.5f )
			: baseTolerance;
		const float currentTolerance = state.fbTolerance >= 0.0f
			? state.fbTolerance : baseTolerance;
		const float delta = std::clamp( targetTolerance - currentTolerance,
			-k_flAdaptationToleranceFallPerFrame,
			k_flAdaptationToleranceRisePerFrame );
		const float nextTolerance = currentTolerance + delta;
		state.fbTolerance = !state.fbRelaxationActive
			&& std::abs( nextTolerance - baseTolerance ) <= 1e-6f
			? -1.0f : nextTolerance;
	}

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
	"adaptation CPU readback must match the shader counter image" );

} // namespace gamescope::framegen
