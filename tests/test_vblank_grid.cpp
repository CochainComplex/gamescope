#include <catch2/catch_test_macros.hpp>

#include "vblankmanager.hpp"

using gamescope::VBlankAtOrBefore;

static constexpr uint64_t kInterval = 16'666'666;
static constexpr uint64_t kAnchor = 1'000'000'000;

TEST_CASE( "A vblank time is its own grid point" )
{
	CHECK( VBlankAtOrBefore( kAnchor, kInterval, kAnchor + 3 * kInterval ) == kAnchor + 3 * kInterval );
}

TEST_CASE( "An off-grid time snaps back to the grid through the anchor" )
{
	CHECK( VBlankAtOrBefore( kAnchor, kInterval, kAnchor + 3 * kInterval + 2'500'000 ) == kAnchor + 3 * kInterval );
	CHECK( VBlankAtOrBefore( kAnchor, kInterval, kAnchor + kInterval - 1 ) == kAnchor );
}

TEST_CASE( "A time at or before the anchor never moves the clock back" )
{
	CHECK( VBlankAtOrBefore( kAnchor, kInterval, kAnchor ) == kAnchor );
	CHECK( VBlankAtOrBefore( kAnchor, kInterval, kAnchor - 5'000'000 ) == kAnchor );
	CHECK( VBlankAtOrBefore( kAnchor, kInterval, 0 ) == kAnchor );
}

TEST_CASE( "Off-grid presentation times keep the phase and advance monotonically" )
{
	// Host completion times spaced 17.8-19.6 ms with a 60 ms outlier, as a
	// headless host reports them.
	const uint64_t steps[] = { 60'200'000, 17'800'000, 19'600'000, 18'400'000, 2'500'000, 16'666'666, 33'000'000 };
	uint64_t ulAnchor = kAnchor, ulTime = kAnchor;
	for ( uint64_t step : steps )
	{
		ulTime += step;
		const uint64_t ulNext = VBlankAtOrBefore( ulAnchor, kInterval, ulTime );
		CHECK( ( ulNext - kAnchor ) % kInterval == 0 );
		CHECK( ulNext >= ulAnchor );
		CHECK( ulNext <= ulTime );
		CHECK( ulTime - ulNext < kInterval );
		ulAnchor = ulNext;
	}
}

TEST_CASE( "Hours after the anchor the arithmetic stays exact" )
{
	const uint64_t ulHours = 10ull * 3600 * 1'000'000'000;
	const uint64_t ulTime = kAnchor + ulHours + 7;
	const uint64_t ulSnap = VBlankAtOrBefore( kAnchor, kInterval, ulTime );
	CHECK( ( ulSnap - kAnchor ) % kInterval == 0 );
	CHECK( ulTime - ulSnap < kInterval );
}
