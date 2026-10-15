#include <catch2/catch_test_macros.hpp>

#include "Utils/SwapchainPolicy.h"

#include <cstdint>
#include <vector>

using namespace gamescope;

namespace
{

constexpr uint64_t k_ulMs = 1'000'000ull;
constexpr uint64_t k_ulFrameNs = 16 * k_ulMs;
// Monotonic clocks never read zero; neither do these traces.
constexpr uint64_t k_ulStartNs = 1'000 * k_ulMs;

VkSurfaceCapabilitiesKHR caps_with( VkExtent2D current, VkExtent2D min = { 1, 1 }, VkExtent2D max = { 16384, 16384 } )
{
	VkSurfaceCapabilitiesKHR caps = {};
	caps.currentExtent = current;
	caps.minImageExtent = min;
	caps.maxImageExtent = max;
	return caps;
}

constexpr VkExtent2D k_undefinedExtent = { UINT32_MAX, UINT32_MAX };

bool same_extent( std::optional<VkExtent2D> oExtent, uint32_t uWidth, uint32_t uHeight )
{
	return oExtent && oExtent->width == uWidth && oExtent->height == uHeight;
}

}

TEST_CASE( "swapchain results classify by VkResult alone", "[swapchain]" )
{
	CHECK( ClassifySwapchainResult( VK_SUCCESS ) == SwapchainStatus_t::Ok );
	CHECK( ClassifySwapchainResult( VK_SUBOPTIMAL_KHR ) == SwapchainStatus_t::Ok );

	for ( VkResult nResult : { VK_ERROR_OUT_OF_DATE_KHR, VK_ERROR_OUT_OF_HOST_MEMORY, VK_ERROR_OUT_OF_DEVICE_MEMORY,
		VK_ERROR_FULL_SCREEN_EXCLUSIVE_MODE_LOST_EXT, VK_ERROR_UNKNOWN, VK_TIMEOUT, VK_NOT_READY, VkResult( -123456 ) } )
	{
		CAPTURE( int( nResult ) );
		CHECK( ClassifySwapchainResult( nResult ) == SwapchainStatus_t::Transient );
	}

	for ( VkResult nResult : { VK_ERROR_SURFACE_LOST_KHR, VK_ERROR_DEVICE_LOST, VK_ERROR_FORMAT_NOT_SUPPORTED,
		VK_ERROR_NATIVE_WINDOW_IN_USE_KHR, VK_ERROR_INITIALIZATION_FAILED } )
	{
		CAPTURE( int( nResult ) );
		CHECK( ClassifySwapchainResult( nResult ) == SwapchainStatus_t::Terminal );
	}
}

TEST_CASE( "vkWaitForPresentKHR results use the same classifier", "[swapchain]" )
{
	// Every code the spec allows, except VK_TIMEOUT: that is the expiry of the
	// waiter's own time slice, handled before classification.
	CHECK( ClassifySwapchainResult( VK_SUCCESS ) == SwapchainStatus_t::Ok );
	CHECK( ClassifySwapchainResult( VK_SUBOPTIMAL_KHR ) == SwapchainStatus_t::Ok );
	// Remake before the next acquire; the frame already rendered still presents.
	CHECK( ClassifySwapchainResult( VK_ERROR_OUT_OF_DATE_KHR ) == SwapchainStatus_t::Transient );
	CHECK( ClassifySwapchainResult( VK_ERROR_OUT_OF_HOST_MEMORY ) == SwapchainStatus_t::Transient );
	CHECK( ClassifySwapchainResult( VK_ERROR_OUT_OF_DEVICE_MEMORY ) == SwapchainStatus_t::Transient );
	CHECK( ClassifySwapchainResult( VK_ERROR_FULL_SCREEN_EXCLUSIVE_MODE_LOST_EXT ) == SwapchainStatus_t::Transient );
	// One shutdown, through the same path as a compositor-side failure.
	CHECK( ClassifySwapchainResult( VK_ERROR_SURFACE_LOST_KHR ) == SwapchainStatus_t::Terminal );
	CHECK( ClassifySwapchainResult( VK_ERROR_DEVICE_LOST ) == SwapchainStatus_t::Terminal );
}

TEST_CASE( "swapchain extent is the output size or nothing", "[swapchain]" )
{
	// Undefined currentExtent (Wayland): the output size within the limits.
	CHECK( same_extent( SwapchainExtentForOutput( caps_with( k_undefinedExtent ), 1280, 720 ), 1280, 720 ) );
	CHECK_FALSE( SwapchainExtentForOutput( caps_with( k_undefinedExtent, { 64, 64 } ), 32, 720 ) );
	CHECK_FALSE( SwapchainExtentForOutput( caps_with( k_undefinedExtent, { 64, 64 } ), 1280, 32 ) );
	CHECK_FALSE( SwapchainExtentForOutput( caps_with( k_undefinedExtent, { 1, 1 }, { 1024, 1024 } ), 1280, 720 ) );
	CHECK_FALSE( SwapchainExtentForOutput( caps_with( k_undefinedExtent, { 1, 1 }, { 2048, 512 } ), 1280, 720 ) );
	CHECK_FALSE( SwapchainExtentForOutput( caps_with( k_undefinedExtent ), 0, 720 ) );
	CHECK_FALSE( SwapchainExtentForOutput( caps_with( k_undefinedExtent ), 1280, 0 ) );

	// Defined currentExtent (X11): exactly the output size.
	CHECK( same_extent( SwapchainExtentForOutput( caps_with( { 1280, 720 }, { 1280, 720 }, { 1280, 720 } ), 1280, 720 ), 1280, 720 ) );
	CHECK_FALSE( SwapchainExtentForOutput( caps_with( { 1281, 720 }, { 1281, 720 }, { 1281, 720 } ), 1280, 720 ) );
	CHECK_FALSE( SwapchainExtentForOutput( caps_with( { 1280, 721 } ), 1280, 720 ) );

	// Minimised: a 0x0 surface backs nothing.
	CHECK_FALSE( SwapchainExtentForOutput( caps_with( { 0, 0 }, { 0, 0 }, { 0, 0 } ), 1280, 720 ) );
	CHECK_FALSE( SwapchainExtentForOutput( caps_with( { 0, 0 }, { 0, 0 }, { 0, 0 } ), 0, 0 ) );
}

TEST_CASE( "transient swapchain failures are bounded in time", "[swapchain]" )
{
	SwapchainRecovery_t recovery;
	uint64_t ulNow = k_ulStartNs;
	for ( ; ulNow - k_ulStartNs < k_ulSwapchainRecoveryBudgetNs; ulNow += k_ulFrameNs )
		REQUIRE( recovery.NoteTransient( ulNow ) );
	CHECK_FALSE( recovery.NoteTransient( ulNow ) );
}

TEST_CASE( "recreating at an unchanged extent is not progress", "[swapchain]" )
{
	SwapchainRecovery_t recovery;
	recovery.NoteCreated( { 1280, 720 } );

	// Each failure is followed by a successful remake at the same size (and a
	// successful acquire, which never reaches the policy).
	uint64_t ulNow = k_ulStartNs;
	bool bWithinBudget = true;
	for ( ; bWithinBudget; ulNow += k_ulFrameNs )
	{
		bWithinBudget = recovery.NoteTransient( ulNow );
		CHECK_FALSE( recovery.NoteCreated( { 1280, 720 } ) );
	}
	CHECK( ulNow - k_ulFrameNs - k_ulStartNs == k_ulSwapchainRecoveryBudgetNs );
}

TEST_CASE( "a failed creation does not make the next success count as a resize", "[swapchain]" )
{
	SwapchainRecovery_t recovery;
	recovery.NoteCreated( { 1280, 720 } );

	uint64_t ulNow = k_ulStartNs;
	bool bWithinBudget = true;
	for ( ; bWithinBudget; ulNow += k_ulFrameNs )
	{
		// Present fails; the remake's creation fails; the next creation succeeds.
		bWithinBudget = recovery.NoteTransient( ulNow );
		if ( !bWithinBudget )
			break;
		bWithinBudget = recovery.NoteTransient( ulNow + k_ulMs );
		recovery.NoteCreated( { 1280, 720 } );
	}
	CHECK( ulNow - k_ulStartNs <= k_ulSwapchainRecoveryBudgetNs + k_ulFrameNs );
	CHECK( ulNow - k_ulStartNs >= k_ulSwapchainRecoveryBudgetNs - k_ulFrameNs );
}

TEST_CASE( "progress restarts the budget", "[swapchain]" )
{
	const uint64_t ulLate = k_ulStartNs + k_ulSwapchainRecoveryBudgetNs - k_ulFrameNs;

	SECTION( "creation at a changed extent" )
	{
		SwapchainRecovery_t recovery;
		recovery.NoteCreated( { 1280, 720 } );
		REQUIRE( recovery.NoteTransient( k_ulStartNs ) );
		REQUIRE( recovery.NoteTransient( ulLate ) );
		CHECK_FALSE( recovery.NoteCreated( { 1281, 720 } ) );
		CHECK( recovery.NoteTransient( ulLate + 2 * k_ulFrameNs ) );
		CHECK( recovery.NoteTransient( ulLate + k_ulSwapchainRecoveryBudgetNs - k_ulFrameNs ) );
	}

	SECTION( "a successful present" )
	{
		SwapchainRecovery_t recovery;
		REQUIRE( recovery.NoteTransient( k_ulStartNs ) );
		REQUIRE( recovery.NoteTransient( ulLate ) );
		CHECK_FALSE( recovery.NoteProgress() );
		CHECK( recovery.NoteTransient( ulLate + 2 * k_ulFrameNs ) );
		CHECK( recovery.NoteTransient( ulLate + k_ulSwapchainRecoveryBudgetNs - k_ulFrameNs ) );
	}

	SECTION( "a gap longer than the budget" )
	{
		SwapchainRecovery_t recovery;
		REQUIRE( recovery.NoteTransient( k_ulStartNs ) );
		REQUIRE( recovery.NoteTransient( ulLate ) );
		const uint64_t ulResume = ulLate + k_ulSwapchainRecoveryBudgetNs + k_ulMs;
		CHECK( recovery.NoteTransient( ulResume ) );
		CHECK( recovery.NoteTransient( ulResume + k_ulSwapchainRecoveryBudgetNs - k_ulFrameNs ) );
		CHECK_FALSE( recovery.NoteTransient( ulResume + k_ulSwapchainRecoveryBudgetNs ) );
	}
}

TEST_CASE( "not presentable is waiting, not failing", "[swapchain]" )
{
	SwapchainRecovery_t recovery;
	REQUIRE( recovery.NoteTransient( k_ulStartNs ) );

	// Minimised for 10 s, retried every frame: logged exactly once, at 2 s.
	uint64_t ulNow = k_ulStartNs + k_ulFrameNs;
	const uint64_t ulRunStart = ulNow;
	uint32_t nLogs = 0;
	for ( ; ulNow - ulRunStart < 10'000 * k_ulMs; ulNow += k_ulFrameNs )
	{
		if ( recovery.NoteNotPresentable( ulNow ) )
		{
			nLogs++;
			CHECK( ulNow - ulRunStart >= k_ulSwapchainRecoveryBudgetNs );
			CHECK( ulNow - ulRunStart < k_ulSwapchainRecoveryBudgetNs + k_ulFrameNs );
		}
	}
	CHECK( nLogs == 1 );

	// The transient episode before it ended; a new one has the full budget.
	CHECK( recovery.NoteTransient( ulNow ) );
	CHECK( recovery.NoteTransient( ulNow + k_ulSwapchainRecoveryBudgetNs - k_ulFrameNs ) );
}

TEST_CASE( "progress reports a logged not-presentable run once", "[swapchain]" )
{
	SECTION( "a successful present" )
	{
		SwapchainRecovery_t recovery;
		CHECK_FALSE( recovery.NoteNotPresentable( k_ulStartNs ) );
		CHECK( recovery.NoteNotPresentable( k_ulStartNs + k_ulSwapchainRecoveryBudgetNs ) );
		CHECK( recovery.NoteProgress() );
		CHECK_FALSE( recovery.NoteProgress() );
	}

	SECTION( "creation at a changed extent" )
	{
		SwapchainRecovery_t recovery;
		recovery.NoteCreated( { 1280, 720 } );
		CHECK_FALSE( recovery.NoteNotPresentable( k_ulStartNs ) );
		CHECK( recovery.NoteNotPresentable( k_ulStartNs + k_ulSwapchainRecoveryBudgetNs ) );
		CHECK( recovery.NoteCreated( { 640, 480 } ) );
		CHECK_FALSE( recovery.NoteCreated( { 640, 480 } ) );
		CHECK_FALSE( recovery.NoteProgress() );
	}

	SECTION( "an unlogged run reports nothing" )
	{
		SwapchainRecovery_t recovery;
		CHECK_FALSE( recovery.NoteNotPresentable( k_ulStartNs ) );
		CHECK_FALSE( recovery.NoteProgress() );
		// A new run starts its own clock.
		CHECK_FALSE( recovery.NoteNotPresentable( k_ulStartNs + k_ulSwapchainRecoveryBudgetNs ) );
	}
}

TEST_CASE( "the budget limiter allows one event per budget", "[swapchain]" )
{
	BudgetLimiter_t limiter;
	CHECK( limiter.Allow( k_ulStartNs ) );
	CHECK_FALSE( limiter.Allow( k_ulStartNs ) );
	CHECK_FALSE( limiter.Allow( k_ulStartNs + k_ulSwapchainRecoveryBudgetNs - 1 ) );
	CHECK( limiter.Allow( k_ulStartNs + k_ulSwapchainRecoveryBudgetNs ) );
	CHECK_FALSE( limiter.Allow( k_ulStartNs + k_ulSwapchainRecoveryBudgetNs + k_ulFrameNs ) );
}

TEST_CASE( "persistent SUBOPTIMAL remakes once per run and extent", "[swapchain]" )
{
	SwapchainRecovery_t recovery;
	uint64_t ulNow = k_ulStartNs;
	const auto present = [&]( VkResult nResult ) {
		ulNow += k_ulFrameNs;
		return recovery.NotePresented( nResult, ulNow );
	};
	recovery.NoteCreated( { 1280, 720 } );
	CHECK_FALSE( present( VK_SUCCESS ) );

	// The first SUBOPTIMAL remakes. A swapchain that stays SUBOPTIMAL after
	// that remake still presents correctly and is kept, however long.
	CHECK( present( VK_SUBOPTIMAL_KHR ) );
	recovery.NoteCreated( { 1280, 720 } );
	for ( int i = 0; i < 1000; i++ )
		REQUIRE_FALSE( present( VK_SUBOPTIMAL_KHR ) );
	// A new extent is a new configuration: it may remake at once, exactly once,
	// and recreating at that same extent never repeats it.
	recovery.NoteCreated( { 640, 480 } );
	CHECK( present( VK_SUBOPTIMAL_KHR ) );
	CHECK_FALSE( present( VK_SUBOPTIMAL_KHR ) );
	for ( int i = 0; i < 1000; i++ )
	{
		recovery.NoteCreated( { 640, 480 } );
		REQUIRE_FALSE( present( VK_SUBOPTIMAL_KHR ) );
	}

	// Failure bookkeeping neither ends nor restarts a run.
	REQUIRE( recovery.NoteTransient( ulNow ) );
	CHECK_FALSE( recovery.NoteNotPresentable( ulNow + k_ulFrameNs ) );
	recovery.NoteProgress();
	CHECK_FALSE( present( VK_SUBOPTIMAL_KHR ) );

	// An optimal present ends the run; the next SUBOPTIMAL remakes once again.
	CHECK_FALSE( present( VK_SUCCESS ) );
	CHECK( present( VK_SUBOPTIMAL_KHR ) );
	const uint64_t ulRemakeNs = ulNow;
	CHECK_FALSE( present( VK_SUBOPTIMAL_KHR ) );

	// Recreating at the same extent keeps the limit: a new run within the
	// budget of that remake is kept, one at the budget remakes.
	recovery.NoteCreated( { 640, 480 } );
	CHECK_FALSE( present( VK_SUCCESS ) );
	CHECK_FALSE( recovery.NotePresented( VK_SUBOPTIMAL_KHR, ulRemakeNs + k_ulSwapchainRecoveryBudgetNs - 1 ) );
	CHECK_FALSE( recovery.NotePresented( VK_SUCCESS, ulRemakeNs + k_ulSwapchainRecoveryBudgetNs - 1 ) );
	CHECK( recovery.NotePresented( VK_SUBOPTIMAL_KHR, ulRemakeNs + k_ulSwapchainRecoveryBudgetNs ) );
}

TEST_CASE( "SUBOPTIMAL remakes are bounded in time", "[swapchain]" )
{
	// A swapchain alternating k optimal presents with one SUBOPTIMAL, remade
	// whenever the policy says so, for 10 s at 60 Hz. Without a time bound
	// that is a remake every k + 1 frames (300, 200, 150).
	constexpr uint64_t k_ulFrame60Ns = 16'666'667ull;
	constexpr uint64_t k_ulRunNs = 10'000 * k_ulMs;
	for ( const uint32_t k : { 1u, 2u, 3u } )
	{
		CAPTURE( k );
		SwapchainRecovery_t recovery;
		recovery.NoteCreated( { 1280, 720 } );
		std::vector<uint64_t> remakes;
		uint32_t nPresents = 0; // on the current swapchain
		for ( uint64_t ulNow = k_ulStartNs; ulNow < k_ulStartNs + k_ulRunNs; ulNow += k_ulFrame60Ns )
		{
			const VkResult nResult = nPresents++ % ( k + 1u ) == k ? VK_SUBOPTIMAL_KHR : VK_SUCCESS;
			if ( !recovery.NotePresented( nResult, ulNow ) )
				continue;
			remakes.push_back( ulNow );
			recovery.NoteCreated( { 1280, 720 } );
			nPresents = 0;
		}
		REQUIRE_FALSE( remakes.empty() );
		// Upstream parity: the first SUBOPTIMAL remakes.
		CHECK( remakes.front() == k_ulStartNs + k * k_ulFrame60Ns );
		for ( size_t i = 1; i < remakes.size(); i++ )
			CHECK( remakes[ i ] - remakes[ i - 1 ] >= k_ulSwapchainRecoveryBudgetNs );
		CHECK( remakes.size() <= 1u + k_ulRunNs / k_ulSwapchainRecoveryBudgetNs );
	}
}

TEST_CASE( "a persistently SUBOPTIMAL swapchain is kept over time", "[swapchain]" )
{
	SwapchainRecovery_t recovery;
	recovery.NoteCreated( { 1280, 720 } );
	uint32_t nRemakes = 0;
	for ( uint64_t ulNow = k_ulStartNs; ulNow < k_ulStartNs + 10'000 * k_ulMs; ulNow += k_ulFrameNs )
	{
		if ( !recovery.NotePresented( VK_SUBOPTIMAL_KHR, ulNow ) )
			continue;
		nRemakes++;
		recovery.NoteCreated( { 1280, 720 } );
	}
	CHECK( nRemakes == 1u );
}

TEST_CASE( "present-wait results map to one action", "[swapchain]" )
{
	using enum PresentWaitAction_t;
	const uint64_t ulT0 = k_ulStartNs;

	SECTION( "a timeout retries without spending the limit" )
	{
		PresentWaitRecovery_t recovery;
		CHECK( recovery.NoteWaitResult( VK_TIMEOUT, ulT0 ) == Retry );
		CHECK( recovery.NoteWaitResult( VK_ERROR_OUT_OF_DATE_KHR, ulT0 + k_ulMs ) == DisableAndRemake );
	}

	SECTION( "presented images publish" )
	{
		PresentWaitRecovery_t recovery;
		CHECK( recovery.NoteWaitResult( VK_SUCCESS, ulT0 ) == Publish );
		CHECK( recovery.NoteWaitResult( VK_SUBOPTIMAL_KHR, ulT0 + k_ulFrameNs ) == Publish );
	}

	SECTION( "a transient requests one recreation per budget" )
	{
		PresentWaitRecovery_t recovery;
		CHECK( recovery.NoteWaitResult( VK_ERROR_OUT_OF_DATE_KHR, ulT0 ) == DisableAndRemake );
		CHECK( recovery.NoteWaitResult( VK_ERROR_OUT_OF_DATE_KHR, ulT0 + k_ulFrameNs ) == Disable );
		CHECK( recovery.NoteWaitResult( VK_ERROR_OUT_OF_DATE_KHR, ulT0 + k_ulSwapchainRecoveryBudgetNs - 1 ) == Disable );
		CHECK( recovery.NoteWaitResult( VK_ERROR_OUT_OF_DATE_KHR, ulT0 + k_ulSwapchainRecoveryBudgetNs ) == DisableAndRemake );
	}

	SECTION( "a terminal result fails" )
	{
		PresentWaitRecovery_t recovery;
		CHECK( recovery.NoteWaitResult( VK_ERROR_SURFACE_LOST_KHR, ulT0 ) == Fail );
		CHECK( recovery.NoteWaitResult( VK_ERROR_DEVICE_LOST, ulT0 + k_ulFrameNs ) == Fail );
	}

	SECTION( "an Ok wait in the middle does not re-arm early" )
	{
		// Transient, recreation, one Ok wait on the fresh swapchain, transient
		// again: waits stay disabled, no second request within the budget.
		PresentWaitRecovery_t recovery;
		CHECK( recovery.NoteWaitResult( VK_ERROR_OUT_OF_DATE_KHR, ulT0 ) == DisableAndRemake );
		CHECK( recovery.NoteWaitResult( VK_SUCCESS, ulT0 + 33 * k_ulMs ) == Publish );
		CHECK( recovery.NoteWaitResult( VK_ERROR_OUT_OF_DATE_KHR, ulT0 + 66 * k_ulMs ) == Disable );
	}

	SECTION( "disabled waits re-arm the request after the budget" )
	{
		PresentWaitRecovery_t recovery;
		CHECK( recovery.NoteWaitResult( VK_ERROR_OUT_OF_DATE_KHR, ulT0 ) == DisableAndRemake );
		CHECK_FALSE( recovery.NoteDisabledWake( ulT0 + k_ulFrameNs ) );
		CHECK_FALSE( recovery.NoteDisabledWake( ulT0 + k_ulSwapchainRecoveryBudgetNs - 1 ) );
		CHECK( recovery.NoteDisabledWake( ulT0 + k_ulSwapchainRecoveryBudgetNs ) );
		CHECK( recovery.NoteWaitResult( VK_ERROR_OUT_OF_DATE_KHR, ulT0 + k_ulSwapchainRecoveryBudgetNs + k_ulFrameNs ) == Disable );
	}
}

TEST_CASE( "present-wait remake requests are rate limited, never per frame", "[swapchain]" )
{
	// A driver whose wait fails on every fresh swapchain, woken once per frame
	// for 10 s; each request is followed by a recreation that re-enables waits.
	PresentWaitRecovery_t recovery;
	bool bDisabled = false;
	uint32_t nRequests = 0;
	for ( uint64_t ulNow = k_ulStartNs; ulNow < k_ulStartNs + 10'000 * k_ulMs; ulNow += k_ulFrameNs )
	{
		bool bRequested = false;
		if ( bDisabled )
			bRequested = recovery.NoteDisabledWake( ulNow );
		else
		{
			const PresentWaitAction_t eAction = recovery.NoteWaitResult( VK_ERROR_OUT_OF_DATE_KHR, ulNow );
			REQUIRE( ( eAction == PresentWaitAction_t::DisableAndRemake || eAction == PresentWaitAction_t::Disable ) );
			bRequested = eAction == PresentWaitAction_t::DisableAndRemake;
		}
		nRequests += bRequested ? 1u : 0u;
		bDisabled = !bRequested;
	}
	CHECK( nRequests == 5u );
}

TEST_CASE( "a not-presentable SDL paint ends the frame", "[swapchain]" )
{
	// Ready presents; Transient is retried (Deferred); NotPresentable is
	// neither, so steamcompmgr retires the frame as Failed.
	STATIC_REQUIRE( SwapchainImagePresentErrno( SwapchainImage_t::Ready ) == 0 );
	STATIC_REQUIRE( SwapchainImagePresentErrno( SwapchainImage_t::Transient ) == -EAGAIN );
	STATIC_REQUIRE( SwapchainImagePresentErrno( SwapchainImage_t::NotPresentable ) == -ENODEV );
	STATIC_REQUIRE( SwapchainImagePresentErrno( SwapchainImage_t::NotPresentable ) != 0 );
	STATIC_REQUIRE( SwapchainImagePresentErrno( SwapchainImage_t::NotPresentable ) != -EAGAIN );
}

TEST_CASE( "a remake owed while a resize is published is left to the size check", "[swapchain]" )
{
	STATIC_REQUIRE( SwapchainNextStep( true, true ) == SwapchainStep_t::AwaitResize );
	STATIC_REQUIRE( SwapchainNextStep( true, false ) == SwapchainStep_t::Remake );
	STATIC_REQUIRE( SwapchainNextStep( false, true ) == SwapchainStep_t::Keep );
	STATIC_REQUIRE( SwapchainNextStep( false, false ) == SwapchainStep_t::Keep );
}

TEST_CASE( "a sticky SUBOPTIMAL run that starts within the remake limit is remade once it expires", "[swapchain]" )
{
	SwapchainRecovery_t recovery;
	recovery.NoteCreated( { 1280, 720 } );
	// A remake followed by optimal presents: the limit runs from that remake.
	REQUIRE( recovery.NotePresented( VK_SUBOPTIMAL_KHR, k_ulStartNs ) );
	recovery.NoteCreated( { 1280, 720 } );
	REQUIRE_FALSE( recovery.NotePresented( VK_SUCCESS, k_ulStartNs + k_ulFrameNs ) );

	// Within the limit the chain turns SUBOPTIMAL for good, and so does every
	// chain recreated at the same extent (Mesa X11 keeps the status sticky).
	std::vector<uint64_t> remakes;
	for ( uint64_t ulNow = k_ulStartNs + 2 * k_ulFrameNs; ulNow < k_ulStartNs + 10'000 * k_ulMs; ulNow += k_ulFrameNs )
	{
		if ( !recovery.NotePresented( VK_SUBOPTIMAL_KHR, ulNow ) )
			continue;
		remakes.push_back( ulNow );
		recovery.NoteCreated( { 1280, 720 } );
	}
	REQUIRE( remakes.size() == 1u );
	CHECK( remakes.front() >= k_ulStartNs + k_ulSwapchainRecoveryBudgetNs );
	CHECK( remakes.front() < k_ulStartNs + k_ulSwapchainRecoveryBudgetNs + k_ulFrameNs );
}
