#pragma once

#include <algorithm>
#include <cstdint>

namespace gamescope::framegen
{

inline constexpr float k_flNeutralStrength = 0.5f;

[[nodiscard]] constexpr uint64_t present_interval_ns( uint64_t currentPresentNs, uint64_t previousPresentNs )
{
	return currentPresentNs > previousPresentNs
		? currentPresentNs - previousPresentNs : 0u;
}

// Acceleration uses a second displacement sample only across comparable real
// intervals. A larger cadence transition falls back to constant velocity.
[[nodiscard]] constexpr bool motion_intervals_comparable( uint64_t currentIntervalNs,
	uint64_t historyIntervalNs )
{
	return currentIntervalNs != 0u && historyIntervalNs != 0u
		&& currentIntervalNs <= historyIntervalNs * 4u
		&& historyIntervalNs <= currentIntervalNs * 4u;
}

// Precondition: motion_intervals_comparable(current, history).
[[nodiscard]] inline float motion_history_time_scale( uint64_t currentIntervalNs,
	uint64_t historyIntervalNs )
{
	return static_cast<float>( static_cast<double>( currentIntervalNs )
		/ static_cast<double>( historyIntervalNs ) );
}

// Irregular-sample quadratic coefficient. Precondition as above.
[[nodiscard]] inline float motion_acceleration_time_factor( uint64_t currentIntervalNs,
	uint64_t historyIntervalNs )
{
	const double flCurrent = static_cast<double>( currentIntervalNs );
	const double flHistory = static_cast<double>( historyIntervalNs );
	return static_cast<float>( flCurrent / ( flCurrent + flHistory ) );
}

struct SlotRequest
{
	float phase;
	float strength;
	uint32_t slotIndex;
};

[[nodiscard]] constexpr float forward_strength_raw( float phase, float configuredStrength )
{
	return phase * ( configuredStrength / k_flNeutralStrength );
}

[[nodiscard]] constexpr float clamp_forward_strength( float rawStrength, float maxForwardStrength )
{
	return std::clamp( rawStrength, 0.0f, maxForwardStrength );
}

// Classic fixed-grid slot k/N. Bidirectional phase bias moves the measured-gap
// phase toward uniform generated-slot spacing without changing queue timing.
// Preconditions: gapVblanks and generatedCount are non-zero.
[[nodiscard]] constexpr SlotRequest classic_slot_request( uint32_t slotIndex, uint32_t generatedIndex,
	uint32_t gapVblanks, uint32_t generatedCount, float bidirPhaseBias,
	float configuredStrength, float maxForwardStrength )
{
	const float flGapPhase = static_cast<float>( slotIndex )
		/ static_cast<float>( gapVblanks );
	const float flUniformPhase = static_cast<float>( generatedIndex + 1u )
		/ static_cast<float>( generatedCount + 1u );
	const float phase = bidirPhaseBias > 0.0f
		? flGapPhase + ( flUniformPhase - flGapPhase ) * bidirPhaseBias
		: flGapPhase;
	const float flStrength = clamp_forward_strength(
		forward_strength_raw( phase, configuredStrength ), maxForwardStrength );
	return { phase, flStrength, slotIndex };
}

struct TimedPrediction
{
	float phase;
	float rawStrength;
};

// Display-clock phase for a target vblank. The caller retains the established
// early rejection when rawStrength exceeds the forward cap. Precondition:
// frametimeEmaNs is non-zero.
[[nodiscard]] inline TimedPrediction timed_prediction( uint64_t targetDeltaNs, uint64_t frametimeEmaNs,
	float configuredStrength )
{
	const float flPhase = static_cast<float>( static_cast<double>( targetDeltaNs )
		/ static_cast<double>( frametimeEmaNs ) );
	return { flPhase, forward_strength_raw( flPhase, configuredStrength ) };
}

struct JitBookkeeping
{
	uint32_t slotIndex;
	uint32_t gapVblanks;
};

// These rounded values key logging and degradation cost only. JIT phase always
// comes from timed_prediction(), never from this synthetic gap. Precondition:
// vblankIntervalNs is non-zero.
[[nodiscard]] constexpr JitBookkeeping jit_bookkeeping( uint64_t targetDeltaNs,
	uint64_t frametimeEmaNs, uint64_t vblankIntervalNs )
{
	const uint32_t nSlotIndex = std::max( 1u,
		static_cast<uint32_t>( ( targetDeltaNs + vblankIntervalNs / 2u )
			/ vblankIntervalNs ) );
	const uint32_t nGapVblanks = std::max( nSlotIndex + 1u, std::max( 2u,
		static_cast<uint32_t>( ( frametimeEmaNs + vblankIntervalNs / 2u )
			/ vblankIntervalNs ) ) );
	return { nSlotIndex, nGapVblanks };
}

// VRR has no fixed grid; this equivalent gap is used only for logs and timing
// rung identity, while the generated frame remains at the exact 0.5 midpoint.
// Precondition: vblankIntervalNs is non-zero.
[[nodiscard]] constexpr uint32_t interval_gap_vblanks( uint64_t frametimeEmaNs,
	uint64_t vblankIntervalNs )
{
	return std::max( 2u, static_cast<uint32_t>(
		( frametimeEmaNs + vblankIntervalNs / 2u ) / vblankIntervalNs ) );
}

} // namespace gamescope::framegen
