#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace gamescope::framegen
{

// Fixed-width histogram over nanosecond samples, reported in milliseconds.
//
// StepNs is the bucket width and Buckets the bucket count, so full scale is
// StepNs * Buckets. add() clamps out-of-range samples into the last bucket and
// percentile() returns the upper edge of the bucket the rank lands in, which
// means a distribution fed samples past full scale reports exactly full scale
// for its high percentiles with no way to tell that apart from a real value.
// Size the range for what the field measures: sub-vblank timing accuracy is
// fine at 16 ms, while wall-clock presentation latency at 60 Hz is >= one
// vblank by construction and needs the wide variant below.
template < uint64_t StepNs, size_t Buckets >
struct MetricsDistribution_t
{
	static_assert( StepNs != 0u, "bucket width must be nonzero" );
	static_assert( Buckets != 0u, "bucket count must be nonzero" );

	static constexpr uint64_t k_ulStepNs = StepNs;
	static constexpr size_t k_nBuckets = Buckets;
	// Upper edge of the last bucket, i.e. what a saturating percentile prints.
	static constexpr double k_flFullScaleMs =
		(double)( StepNs * (uint64_t)Buckets ) / 1.0e6;

	uint64_t n = 0;
	double sum = 0.0;
	double sumSquares = 0.0;
	double min = std::numeric_limits<double>::max();
	double max = 0.0;
	std::array<uint64_t, Buckets> histogram = {};

	void add( uint64_t ulNs )
	{
		const double flMs = ulNs / 1.0e6;
		n++;
		sum += flMs;
		sumSquares += flMs * flMs;
		min = std::min( min, flMs );
		max = std::max( max, flMs );
		const size_t nBucket = std::min<size_t>( ulNs / StepNs, Buckets - 1 );
		histogram[ nBucket ]++;
	}

	double average() const { return n != 0 ? sum / n : 0.0; }
	double stddev() const
	{
		return n != 0 ? std::sqrt( std::max( 0.0,
			sumSquares / n - average() * average() ) ) : 0.0;
	}
	double percentile( uint32_t nPercent ) const
	{
		if ( n == 0 )
			return 0.0;
		const uint64_t nRank = ( n * nPercent + 99u ) / 100u;
		uint64_t nSeen = 0;
		for ( size_t i = 0; i < histogram.size(); i++ )
		{
			nSeen += histogram[ i ];
			if ( nSeen >= nRank )
				return ( i + 1 ) * StepNs / 1.0e6;
		}
		return k_flFullScaleMs;
	}
	double p50() const { return percentile( 50u ); }
	double p95() const { return percentile( 95u ); }
};

// Bucket width shared by every metrics distribution: 250 us resolves the
// sub-millisecond structure of GPU pass and deadline-error timings.
inline constexpr uint64_t k_ulMetricsHistogramStepNs = 250'000ull;
// 16 ms full scale. Only for quantities that cannot plausibly run past it.
inline constexpr size_t k_nMetricsHistogramBuckets = 64;
// 64 ms full scale at the same resolution, for wall-clock presentation
// latencies: a 60 Hz flip interval alone is 16.67 ms, and a source-ready ->
// flip latency adds queueing on top of that.
inline constexpr size_t k_nMetricsLatencyHistogramBuckets = 256;

using MetricsDistribution =
	MetricsDistribution_t< k_ulMetricsHistogramStepNs, k_nMetricsHistogramBuckets >;
using MetricsLatencyDistribution =
	MetricsDistribution_t< k_ulMetricsHistogramStepNs, k_nMetricsLatencyHistogramBuckets >;

} // namespace gamescope::framegen
