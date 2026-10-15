#pragma once

#include <cerrno>
#include <cstdint>
#include <optional>

#include <vulkan/vulkan_core.h>

// Pure policy of the window-swapchain output (Vulkan-swapchain backends).
// rendervulkan applies it; tests drive these same functions.

namespace gamescope
{
    enum class SwapchainStatus_t : uint8_t
    {
        Ok,
        Transient,
        Terminal,
    };

    // One policy per VkResult, whichever call (surface query, creation, acquire,
    // present) returned it.
    [[nodiscard]] constexpr SwapchainStatus_t ClassifySwapchainResult( VkResult nResult )
    {
        switch ( nResult )
        {
            case VK_SUCCESS:
            case VK_SUBOPTIMAL_KHR:
                return SwapchainStatus_t::Ok;
            case VK_ERROR_SURFACE_LOST_KHR:
            case VK_ERROR_DEVICE_LOST:
            // No surface format in the requested colour space; HDR never falls back.
            case VK_ERROR_FORMAT_NOT_SUPPORTED:
            case VK_ERROR_NATIVE_WINDOW_IN_USE_KHR:
            // Also reported for a swapchain without images.
            case VK_ERROR_INITIALIZATION_FAILED:
                return SwapchainStatus_t::Terminal;
            default:
                // OUT_OF_DATE, out of memory, unknown codes and other status codes.
                return SwapchainStatus_t::Transient;
        }
    }

    // The swapchain is exactly the published output size or nothing
    // (VUID-VkSwapchainCreateInfoKHR-imageExtent-01274/-01689): composites
    // dispatch over the output size. nullopt: the surface cannot present this
    // output now (minimised to 0x0, or a resize SDL has not published yet).
    [[nodiscard]] constexpr std::optional<VkExtent2D> SwapchainExtentForOutput(
        const VkSurfaceCapabilitiesKHR &caps, uint32_t uWidth, uint32_t uHeight )
    {
        if ( uWidth == 0 || uHeight == 0 )
            return std::nullopt;
        if ( uWidth < caps.minImageExtent.width || uWidth > caps.maxImageExtent.width
            || uHeight < caps.minImageExtent.height || uHeight > caps.maxImageExtent.height )
            return std::nullopt;
        // 0xFFFFFFFF: the swapchain decides the surface size (Wayland).
        const bool bSurfaceSized = caps.currentExtent.width != UINT32_MAX;
        if ( bSurfaceSized && ( caps.currentExtent.width != uWidth || caps.currentExtent.height != uHeight ) )
            return std::nullopt;
        return VkExtent2D{ uWidth, uHeight };
    }

    // Whether a paint has a swapchain image to render into.
    enum class SwapchainImage_t : uint8_t
    {
        Ready,
        Transient,      // retried within the recovery budget
        NotPresentable, // the window cannot present now, or the output failed
    };

    // Backend Present errno for a paint without a swapchain image. Transient is
    // retried within the recovery budget (-EAGAIN: Deferred). NotPresentable ends
    // the frame (Failed: retired as upstream does); the wake comes when the window
    // can present again (map, expose, size publish) or shutdown is underway.
    [[nodiscard]] constexpr int SwapchainImagePresentErrno( SwapchainImage_t eImage )
    {
        switch ( eImage )
        {
            case SwapchainImage_t::Ready:
                return 0;
            case SwapchainImage_t::Transient:
                return -EAGAIN;
            case SwapchainImage_t::NotPresentable:
                break;
        }
        return -ENODEV;
    }

    // What a paint does with the live swapchain: before rendering into the image
    // it holds, and after presenting it.
    enum class SwapchainStep_t : uint8_t
    {
        Keep,        // render into the held image / acquire from the live swapchain
        Remake,      // recreate the swapchain, then acquire
        AwaitResize, // hold no image and repaint; the compositor's size check remakes
    };

    // bRemake: the live swapchain cannot take the next image as it is.
    // bResizePending: the window published a size the compositor has not adopted.
    // Its size check remakes at that size on the next iteration, so remaking now
    // would recreate twice at one extent. Holding no image keeps the remake owed:
    // if the size reverts first, the next paint remakes instead.
    [[nodiscard]] constexpr SwapchainStep_t SwapchainNextStep( bool bRemake, bool bResizePending )
    {
        if ( !bRemake )
            return SwapchainStep_t::Keep;
        return bResizePending ? SwapchainStep_t::AwaitResize : SwapchainStep_t::Remake;
    }

    inline constexpr uint64_t k_ulSwapchainRecoveryBudgetNs = 2'000'000'000ull;

    // At most one event per k_ulSwapchainRecoveryBudgetNs; the first is allowed.
    struct BudgetLimiter_t
    {
        uint64_t ulLastNs = 0; // 0: none yet

        [[nodiscard]] constexpr bool Allow( uint64_t ulNow )
        {
            if ( ulLastNs != 0 && ulNow - ulLastNs < k_ulSwapchainRecoveryBudgetNs )
                return false;
            ulLastNs = ulNow;
            return true;
        }
    };

    // Bounds transient swapchain failures in wall-clock time. Compositor thread
    // only (and the main thread during init, before the compositor starts).
    struct SwapchainRecovery_t
    {
        uint64_t ulEpisodeStartNs = 0;
        uint64_t ulLastTransientNs = 0;       // 0: no episode
        uint64_t ulNotPresentableSinceNs = 0; // 0: no not-presentable run
        bool bNotPresentableLogged = false;
        VkExtent2D lastCreated = {};          // extent of the last SUCCESSFUL creation
        bool bSuboptimalRun = false;          // a remake was requested and every present since was SUBOPTIMAL
        BudgetLimiter_t suboptimalRemakes;

        // True while within the budget. A gap longer than the budget (no
        // attempts, e.g. an unmapped window) starts a new episode.
        [[nodiscard]] constexpr bool NoteTransient( uint64_t ulNow )
        {
            if ( ulLastTransientNs == 0 || ulNow - ulLastTransientNs > k_ulSwapchainRecoveryBudgetNs )
                ulEpisodeStartNs = ulNow;
            ulLastTransientNs = ulNow;
            return ulNow - ulEpisodeStartNs < k_ulSwapchainRecoveryBudgetNs;
        }

        // Ends any episode: waiting for the window is not a failure. True exactly
        // once per run, when the run reaches the budget (log then).
        [[nodiscard]] constexpr bool NoteNotPresentable( uint64_t ulNow )
        {
            ulLastTransientNs = 0;
            if ( ulNotPresentableSinceNs == 0 )
                ulNotPresentableSinceNs = ulNow;
            if ( bNotPresentableLogged || ulNow - ulNotPresentableSinceNs < k_ulSwapchainRecoveryBudgetNs )
                return false;
            bNotPresentableLogged = true;
            return true;
        }

        // An image reached the swapchain: ends the episode and the
        // not-presentable run. True iff a not-presentable line was logged.
        constexpr bool NoteProgress()
        {
            const bool bLogged = bNotPresentableLogged;
            ulLastTransientNs = 0;
            ulNotPresentableSinceNs = 0;
            bNotPresentableLogged = false;
            return bLogged;
        }

        // A creation succeeded. A changed extent is progress (a resize explains
        // the failures) and returns NoteProgress(); an unchanged one changes
        // nothing, so recreating at the same size cannot extend the budget.
        // A changed extent also ends a SUBOPTIMAL run and its remake limit: the
        // new configuration may remake at once.
        constexpr bool NoteCreated( VkExtent2D extent )
        {
            if ( extent.width == lastCreated.width && extent.height == lastCreated.height )
                return false;
            lastCreated = extent;
            bSuboptimalRun = false;
            suboptimalRemakes = {};
            return NoteProgress();
        }

        // An image was presented (SUCCESS or SUBOPTIMAL, both success codes).
        // True on the first SUBOPTIMAL the limit allows. A refused SUBOPTIMAL
        // does not start a run, so a swapchain that never turns optimal (Mesa X11
        // keeps reporting SUBOPTIMAL) is remade once the limit expires. A
        // swapchain that stays SUBOPTIMAL after its remake presents correctly and
        // is kept; one that alternates OPTIMAL/SUBOPTIMAL is remade at most once
        // per budget, never once per run.
        [[nodiscard]] constexpr bool NotePresented( VkResult nResult, uint64_t ulNow )
        {
            const bool bSuboptimal = nResult == VK_SUBOPTIMAL_KHR;
            const bool bRemake = bSuboptimal && !bSuboptimalRun && suboptimalRemakes.Allow( ulNow );
            bSuboptimalRun = bSuboptimal && ( bSuboptimalRun || bRemake );
            return bRemake;
        }
    };

    enum class PresentWaitAction_t : uint8_t
    {
        Publish,          // the image was presented: publish its timing
        Retry,            // VK_TIMEOUT, a success code: wait again
        Disable,          // disable waits until a swapchain recreation
        DisableAndRemake, // ... and request one
        Fail,             // terminal: the output has failed
    };

    // Present-wait recovery: one action per vkWaitForPresentKHR result, through
    // ClassifySwapchainResult like every other swapchain call. A transient result
    // disables waits until a swapchain recreation re-enables them; the waiter
    // requests at most one recreation per k_ulSwapchainRecoveryBudgetNs. While
    // waits stay disabled the request re-arms after the budget, so waits, and the
    // nested vblank clock they drive, are never off for good.
    // An Ok wait does not reset the limit: a driver that fails on every fresh
    // swapchain would otherwise cost a recreation per frame; waits stay off for
    // at most one budget instead.
    // Waiter thread only.
    struct PresentWaitRecovery_t
    {
        BudgetLimiter_t remakes;

        [[nodiscard]] constexpr PresentWaitAction_t NoteWaitResult( VkResult nResult, uint64_t ulNow )
        {
            if ( nResult == VK_TIMEOUT )
                return PresentWaitAction_t::Retry;
            switch ( ClassifySwapchainResult( nResult ) )
            {
                case SwapchainStatus_t::Ok:
                    return PresentWaitAction_t::Publish;
                case SwapchainStatus_t::Transient:
                    return remakes.Allow( ulNow ) ? PresentWaitAction_t::DisableAndRemake
                        : PresentWaitAction_t::Disable;
                case SwapchainStatus_t::Terminal:
                    break;
            }
            return PresentWaitAction_t::Fail;
        }

        // Waits are disabled: whether to re-arm the recreation request.
        [[nodiscard]] constexpr bool NoteDisabledWake( uint64_t ulNow )
        {
            return remakes.Allow( ulNow );
        }
    };
}
