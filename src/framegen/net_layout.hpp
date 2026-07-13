#pragma once

#include <cstdint>

// Shared CPU-side contract for GSFR profiles and the inference/training GPU
// resources. Shader literals and offline tools must be updated deliberately if
// this shape ever changes; the assertions below keep accidental drift fatal.
inline constexpr uint32_t k_uFramegenNetMagic = 0x52465347u; // 'GSFR'
inline constexpr uint32_t k_uFramegenNetVersion = 3u;
inline constexpr uint32_t k_uFramegenNetLayerCount = 3u;
inline constexpr uint32_t k_uFramegenNetKernelWidth = 3u;
inline constexpr uint32_t k_uFramegenNetKernelElements =
	k_uFramegenNetKernelWidth * k_uFramegenNetKernelWidth;
inline constexpr uint32_t k_uFramegenNetLayerDims[ k_uFramegenNetLayerCount ][ 2 ] =
	{ { 12u, 16u }, { 16u, 16u }, { 16u, 4u } };
inline constexpr uint32_t k_uFramegenNetLayer1Weights =
	k_uFramegenNetLayerDims[ 0 ][ 0 ] * k_uFramegenNetLayerDims[ 0 ][ 1 ] * k_uFramegenNetKernelElements;
inline constexpr uint32_t k_uFramegenNetLayer2Offset =
	k_uFramegenNetLayer1Weights + k_uFramegenNetLayerDims[ 0 ][ 1 ];
inline constexpr uint32_t k_uFramegenNetLayer2Weights =
	k_uFramegenNetLayerDims[ 1 ][ 0 ] * k_uFramegenNetLayerDims[ 1 ][ 1 ] * k_uFramegenNetKernelElements;
inline constexpr uint32_t k_uFramegenNetLayer3Offset =
	k_uFramegenNetLayer2Offset + k_uFramegenNetLayer2Weights + k_uFramegenNetLayerDims[ 1 ][ 1 ];
inline constexpr uint32_t k_uFramegenNetLayer3WeightsPerOutput =
	k_uFramegenNetLayerDims[ 2 ][ 0 ] * k_uFramegenNetKernelElements;
inline constexpr uint32_t k_uFramegenNetLayer3BiasOffset =
	k_uFramegenNetLayer3Offset + k_uFramegenNetLayer3WeightsPerOutput * k_uFramegenNetLayerDims[ 2 ][ 1 ];
inline constexpr uint32_t k_uFramegenNetFloats =
	k_uFramegenNetLayer3BiasOffset + k_uFramegenNetLayerDims[ 2 ][ 1 ];
inline constexpr uint32_t k_uFramegenNetShadingOutput = 3u;
inline constexpr uint32_t k_uFramegenNetShadingWeightBegin =
	k_uFramegenNetLayer3Offset + k_uFramegenNetShadingOutput * k_uFramegenNetLayer3WeightsPerOutput;
inline constexpr uint32_t k_uFramegenNetShadingBias =
	k_uFramegenNetLayer3BiasOffset + k_uFramegenNetShadingOutput;

// Shader indexes are (idx & 2047, idx >> 11).
inline constexpr uint32_t k_uFramegenNetTexW = 2048u;
inline constexpr uint32_t k_uFramegenNetTexH =
	( k_uFramegenNetFloats + k_uFramegenNetTexW - 1u ) / k_uFramegenNetTexW;

static_assert( k_uFramegenNetShadingOutput < k_uFramegenNetLayerDims[ 2 ][ 1 ],
	"framegen shading output must exist in the final net layer" );
static_assert( k_uFramegenNetLayer2Offset == 1744u
	&& k_uFramegenNetLayer3Offset == 4064u
	&& k_uFramegenNetLayer3BiasOffset == 4640u,
	"framegen net offsets must match the inference and training shaders" );
static_assert( k_uFramegenNetFloats == 4644u,
	"framegen net shape must match the inference/training shaders and GSFR tools" );
