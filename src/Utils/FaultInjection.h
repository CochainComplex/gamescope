#pragma once

#include <charconv>
#include <cstdint>
#include <optional>
#include <string_view>

// Strict grammar shared by the GAMESCOPE_DEBUG_*_INJECT knobs:
//
//   spec := kind [ ":" code ] [ ":" N ]
//
// "code" is present only for ParseFaultInjectionCode and must be a negative
// int32 (a Vulkan error); zero and positive status codes would fabricate
// completions. N is a strictly positive decimal count; without it the
// injection never ends. No signs on N, whitespace or trailing characters.

namespace gamescope
{
    struct FaultInjection_t
    {
        int32_t nCode = 0;
        uint64_t ulRemaining = 0;
        bool bUnlimited = true;

        // True when this operation must fail; consumes one injection.
        bool Consume()
        {
            if ( bUnlimited )
                return true;
            if ( ulRemaining == 0 )
                return false;
            ulRemaining--;
            return true;
        }

        bool Exhausted() const { return !bUnlimited && ulRemaining == 0; }
    };

    namespace FaultInjectionDetail
    {
        // Parses the optional ":N" suffix that ends every spec.
        inline std::optional<FaultInjection_t> ParseCount( std::string_view sRest, FaultInjection_t injection )
        {
            if ( sRest.empty() )
                return injection;
            if ( !sRest.starts_with( ':' ) )
                return std::nullopt;

            sRest.remove_prefix( 1 );
            const char *pEnd = sRest.data() + sRest.size();
            auto result = std::from_chars( sRest.data(), pEnd, injection.ulRemaining );
            if ( result.ec != std::errc{} || result.ptr != pEnd || injection.ulRemaining == 0 )
                return std::nullopt;

            injection.bUnlimited = false;
            return injection;
        }
    }

    // "kind" or "kind:N".
    inline std::optional<FaultInjection_t> ParseFaultInjection( std::string_view sSpec, std::string_view sKind )
    {
        if ( !sSpec.starts_with( sKind ) )
            return std::nullopt;

        sSpec.remove_prefix( sKind.size() );
        return FaultInjectionDetail::ParseCount( sSpec, {} );
    }

    // "kind:code" or "kind:code:N".
    inline std::optional<FaultInjection_t> ParseFaultInjectionCode( std::string_view sSpec, std::string_view sKind )
    {
        if ( !sSpec.starts_with( sKind ) )
            return std::nullopt;

        sSpec.remove_prefix( sKind.size() );
        if ( !sSpec.starts_with( ':' ) )
            return std::nullopt;

        sSpec.remove_prefix( 1 );
        FaultInjection_t injection;
        auto result = std::from_chars( sSpec.data(), sSpec.data() + sSpec.size(), injection.nCode );
        if ( result.ec != std::errc{} || injection.nCode >= 0 )
            return std::nullopt;

        sSpec.remove_prefix( result.ptr - sSpec.data() );
        return FaultInjectionDetail::ParseCount( sSpec, injection );
    }
}
