// Frame-generation queue submission and timestamp accounting.

#include "../rendervulkan.hpp"

#include <utility>

namespace
{

// These operations are fatal after device initialization, matching the
// renderer's Vulkan-call contract.
#define vk_check( expression ) \
	do \
	{ \
		const VkResult checkResult = ( expression ); \
		if ( checkResult != VK_SUCCESS ) \
			vulkan_check_fatal( checkResult, #expression ); \
	} while ( 0 )

} // namespace

// Submit frame-generation work. With a dedicated framegen queue, this runs on
// the second queue: it waits on the composite's scratch-timeline value (so it
// never reads a half-written output image) and signals its own framegen
// timeline, meaning a slow generation can never sit in front of the next
// composite on the realtime queue. Without a second queue it forwards to the
// normal shared-queue submit, where in-order execution provides the same
// ordering (at the cost of the head-of-line blocking this feature removes).
uint64_t CVulkanDevice::submitFramegen( std::unique_ptr<CVulkanCmdBuffer> cmdBuffer, uint64_t ulWaitCompositeSeqNo, int nQuerySlot, uint32_t nLadderRung, uint32_t nGeneratedCount )
{
	if ( !m_bHasFramegenQueue )
	{
		const uint64_t ulSeqNo = submit( std::move( cmdBuffer ) );
		if ( nQuerySlot >= 0 )
			m_framegenQuerySlotBySeqNo.emplace_back( ulSeqNo,
				FramegenQueryAssoc_t{ (uint32_t)nQuerySlot, nLadderRung, nGeneratedCount } );
		return ulSeqNo;
	}

	CVulkanCmdBuffer *pRaw = cmdBuffer.get();
	pRaw->end();

	const uint64_t nextSeqNo = ++m_framegenSeqNo;

	VkSemaphore waitSemaphores[1] = { m_scratchTimelineSemaphore };
	uint64_t waitValues[1] = { ulWaitCompositeSeqNo };
	// Nothing in this command buffer is independent of the just-composited real
	// frame. Wait at the earliest execution scope so the opening TOP_OF_PIPE
	// timestamp cannot run before the semaphore and accidentally charge
	// composite-wait time to the framegen degradation ladder. ALL_COMMANDS also
	// keeps future non-compute setup/copy passes inside the same dependency.
	VkPipelineStageFlags waitStages[1] = { VK_PIPELINE_STAGE_ALL_COMMANDS_BIT };

	VkSemaphore signalSemaphores[1] = { m_framegenTimeline->pVkSemaphore };
	uint64_t signalValues[1] = { nextSeqNo };

	const bool bWait = ulWaitCompositeSeqNo != 0;

	VkTimelineSemaphoreSubmitInfo timelineInfo = {
		.sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO,
		.waitSemaphoreValueCount = bWait ? 1u : 0u,
		.pWaitSemaphoreValues = bWait ? waitValues : nullptr,
		.signalSemaphoreValueCount = 1,
		.pSignalSemaphoreValues = signalValues,
	};

	VkCommandBuffer rawCmdBuffer = pRaw->rawBuffer();
	VkSubmitInfo submitInfo = {
		.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
		.pNext = &timelineInfo,
		.waitSemaphoreCount = bWait ? 1u : 0u,
		.pWaitSemaphores = bWait ? waitSemaphores : nullptr,
		.pWaitDstStageMask = bWait ? waitStages : nullptr,
		.commandBufferCount = 1,
		.pCommandBuffers = &rawCmdBuffer,
		.signalSemaphoreCount = 1,
		.pSignalSemaphores = signalSemaphores,
	};

	vk_check( vk.QueueSubmit( m_framegenQueue, 1, &submitInfo, VK_NULL_HANDLE ) );

	m_pendingFramegenCmdBufs.emplace_back( nextSeqNo, std::move( cmdBuffer ) );
	// Associate the timestamp ring slot (if any), ladder rung, and generated-count
	// with its seqNo, so readback can attribute measured cost to the exact batch
	// shape. A rung's cost is not stable across x2-like and x4-like gaps.
	if ( nQuerySlot >= 0 )
		m_framegenQuerySlotBySeqNo.emplace_back( nextSeqNo, FramegenQueryAssoc_t{ (uint32_t)nQuerySlot, nLadderRung, nGeneratedCount } );
	return nextSeqNo;
}

bool CVulkanDevice::hasCompletedFramegen( uint64_t sequence )
{
	if ( !m_bHasFramegenQueue )
		return hasCompleted( sequence );
	if ( sequence == 0
		|| m_framegenCompletedSeqNo.load( std::memory_order_relaxed ) >= sequence )
		return true;

	uint64_t currentSeqNo = 0;
	vk_check( vk.GetSemaphoreCounterValue( device(), m_framegenTimeline->pVkSemaphore, &currentSeqNo ) );
	cache_timeline_completion( m_framegenCompletedSeqNo, currentSeqNo );
	return currentSeqNo >= sequence;
}

// Record the opening timestamp of a framegen batch. Rotates the query-pool ring,
// resets that slot's two queries, and writes a TOP_OF_PIPE timestamp. Returns the
// slot for framegenTimestampEnd / submitFramegen, or -1 when measurement is off.
int CVulkanDevice::framegenTimestampBegin( CVulkanCmdBuffer *pCmdBuffer )
{
	if ( m_framegenQueryPool == VK_NULL_HANDLE || m_uFramegenQueryRingDepth == 0 || pCmdBuffer == nullptr )
		return -1;

	const uint32_t nSlot = m_uFramegenQueryHead;
	m_uFramegenQueryHead = ( m_uFramegenQueryHead + 1 ) % m_uFramegenQueryRingDepth;

	VkCommandBuffer raw = pCmdBuffer->rawBuffer();
	vk.CmdResetQueryPool( raw, m_framegenQueryPool, nSlot * 2, 2 );
	vk.CmdWriteTimestamp( raw, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, m_framegenQueryPool, nSlot * 2 );
	return (int)nSlot;
}

// Record the closing timestamp of a framegen batch (BOTTOM_OF_PIPE). Must be
// called while the command buffer is still recording, before submitFramegen ends it.
void CVulkanDevice::framegenTimestampEnd( CVulkanCmdBuffer *pCmdBuffer, int nSlot )
{
	if ( nSlot < 0 || m_framegenQueryPool == VK_NULL_HANDLE || pCmdBuffer == nullptr )
		return;

	vk.CmdWriteTimestamp( pCmdBuffer->rawBuffer(), VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, m_framegenQueryPool, (uint32_t)nSlot * 2 + 1 );
}

void CVulkanDevice::framegenResetRungCosts()
{
	// A scene reset may race logically with an already-submitted batch on the
	// dedicated queue. Its command buffer still completes and is recycled, but
	// its timing belongs to the old scene and must not re-seed the fresh ladder.
	m_framegenQuerySlotBySeqNo.clear();
	m_ulFramegenLastRawGpuTimeNs = 0;
	for ( uint32_t r = 0; r < kFramegenLadderSlots; r++ )
		for ( uint32_t c = 0; c < kFramegenGeneratedCountSlots; c++ )
		{
			m_aFramegenRungCostNs[ r ][ c ] = 0;
			m_aFramegenRungSamples[ r ][ c ] = 0;
		}
}

void CVulkanDevice::waitFramegen( uint64_t sequence )
{
	// Sequence zero denotes a queued real frame in the bidirectional timeline.
	// It has no framegen submission to retire, so avoid an otherwise pointless
	// vkWaitSemaphores(value=0) on every such flip.
	if ( sequence == 0 )
		return;

	if ( !m_bHasFramegenQueue )
	{
		wait( sequence, false );
		return;
	}

	VkSemaphoreWaitInfo waitInfo = {
		.sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO,
		.semaphoreCount = 1,
		.pSemaphores = &m_framegenTimeline->pVkSemaphore,
		.pValues = &sequence,
	};
	vk_check( vk.WaitSemaphores( device(), &waitInfo, ~0ull ) );
	cache_timeline_completion( m_framegenCompletedSeqNo, sequence );
}

// Recycle dedicated-queue command buffers and consume timestamp queries whose
// owning timeline has completed. Shared-queue command buffers are recycled by
// garbageCollect(), but their framegen query associations live here too so the
// deadline ladder behaves identically on single-queue devices.
void CVulkanDevice::framegenGarbageCollect()
{
	if ( m_pendingFramegenCmdBufs.empty() && m_framegenQuerySlotBySeqNo.empty() )
		return;

	uint64_t currentSeqNo = 0;
	const VkSemaphore completionTimeline = m_bHasFramegenQueue
		? m_framegenTimeline->pVkSemaphore : m_scratchTimelineSemaphore;
	vk_check( vk.GetSemaphoreCounterValue( device(), completionTimeline, &currentSeqNo ) );
	if ( m_bHasFramegenQueue )
		cache_timeline_completion( m_framegenCompletedSeqNo, currentSeqNo );
	else
		cache_timeline_completion( m_submissionCompletedSeqNo, currentSeqNo );

	if ( m_bHasFramegenQueue )
	{
		auto completedEnd = m_pendingFramegenCmdBufs.begin();
		while ( completedEnd != m_pendingFramegenCmdBufs.end()
			&& completedEnd->first <= currentSeqNo )
		{
			completedEnd->second->reset();
			m_unusedCmdBufs.push_back( std::move( completedEnd->second ) );
			++completedEnd;
		}
		m_pendingFramegenCmdBufs.erase( m_pendingFramegenCmdBufs.begin(), completedEnd );
	}

	// A completed timeline signal makes both timestamps available. Read without
	// WAIT_BIT so this maintenance path can never stall the compositor thread.
	auto completedQueriesEnd = m_framegenQuerySlotBySeqNo.begin();
	while ( completedQueriesEnd != m_framegenQuerySlotBySeqNo.end()
		&& completedQueriesEnd->first <= currentSeqNo )
	{
		const uint32_t nSlot = completedQueriesEnd->second.nSlot;
		const uint32_t nRung = completedQueriesEnd->second.nRung;
		const uint32_t nGeneratedCount = completedQueriesEnd->second.nGeneratedCount;
		uint64_t ts[2] = { 0, 0 };
		const VkResult res = vk.GetQueryPoolResults( device(), m_framegenQueryPool, nSlot * 2, 2,
			sizeof( ts ), ts, sizeof( uint64_t ), VK_QUERY_RESULT_64_BIT );
		const uint64_t ulTimestampMask = m_uFramegenTimestampValidBits >= 64
			? UINT64_MAX : ( ( 1ull << m_uFramegenTimestampValidBits ) - 1ull );
		const uint64_t ulGpuTicks = ( ts[1] - ts[0] ) & ulTimestampMask;
		if ( res == VK_SUCCESS && ulGpuTicks != 0
			&& nRung < kFramegenLadderSlots && nGeneratedCount < kFramegenGeneratedCountSlots )
		{
			// Timestamp values wrap at timestampValidBits, which can be less
			// than 64 even though GetQueryPoolResults returns uint64_t values.
			// Modular subtraction above keeps a wrap-crossing batch valid.
			const uint64_t ulGpuNs = (uint64_t)( double( ulGpuTicks ) * m_flFramegenTimestampPeriodNs );
			m_ulFramegenLastRawGpuTimeNs = ulGpuNs;
			// Fold into this exact batch shape with a symmetric slow EMA (7/8).
			// A single anomalous batch must not shed quality for the whole scene,
			// and x2-like gaps must not inherit x4-like batch timings.
			uint64_t &ulRungCost = m_aFramegenRungCostNs[ nRung ][ nGeneratedCount ];
			ulRungCost = ( ulRungCost == 0 ) ? ulGpuNs : ( ulRungCost * 7 + ulGpuNs ) / 8;
			if ( m_aFramegenRungSamples[ nRung ][ nGeneratedCount ] != UINT32_MAX )
				m_aFramegenRungSamples[ nRung ][ nGeneratedCount ]++;
		}
		++completedQueriesEnd;
	}
	m_framegenQuerySlotBySeqNo.erase( m_framegenQuerySlotBySeqNo.begin(), completedQueriesEnd );
}

#undef vk_check
