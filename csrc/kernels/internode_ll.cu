#include "configs.cuh"
#include "exception.cuh"
#include "launch.cuh"
#include "ibgda_device.cuh"

namespace deep_ep {

// Forward declaration of ExpertSyncInfo from config.hpp
struct ExpertSyncInfo {
    int expected_tokens_per_rank[8];
    int received_tokens_per_rank[8];
    int total_expected_tokens;
    int total_received_tokens;
    int completed_ranks;
    int expert_processing_complete;
    void* combined_x_ptr;  // Pointer to combined_x buffer in NVSHMEM symmetric heap
    int padding[1];
};

namespace internode_ll {

// Diagnostic arrays for tracking sync points and block execution
__device__ int g_sync_counter[1024];      // Track progress state for each block
__device__ int g_sync_reached[1024];      // Track which blocks reached sync point
__device__ int g_active_blocks[1024];     // Track which blocks are actually processing
__device__ int g_block_expert_idx[1024];  // Track which expert each block is processing

template <int kNumThreads> __launch_bounds__(kNumThreads, 1)
__global__ void clean_low_latency_buffer(int* clean_0, int num_clean_int_0,
                                         int* clean_1, int num_clean_int_1) {
    // Barrier before cleaning (in case of unfinished chunked EP)
    nvshmemx_barrier_all_block();

    // Clean - handle null pointers gracefully
    auto thread_id = static_cast<int>(threadIdx.x);
    if (clean_0 != nullptr) {
        #pragma unroll
        for (int i = thread_id; i < num_clean_int_0; i += kNumThreads)
            clean_0[i] = 0;
    }
    if (clean_1 != nullptr) {
        #pragma unroll
        for (int i = thread_id; i < num_clean_int_1; i += kNumThreads)
            clean_1[i] = 0;
    }

    // Barrier after cleaning (make sure the low-latency mode works fine)
    nvshmemx_barrier_all_block();
}

// Extended version that also cleans ExpertSyncInfo
template <int kNumThreads> __launch_bounds__(kNumThreads, 1)
__global__ void clean_low_latency_buffer_with_sync(int* clean_0, int num_clean_int_0,
                                                   int* clean_1, int num_clean_int_1,
                                                   ExpertSyncInfo* expert_sync_info,
                                                   int num_experts) {
    // Barrier before cleaning (in case of unfinished chunked EP)
    nvshmemx_barrier_all_block();

    // Clean - handle null pointers gracefully
    auto thread_id = static_cast<int>(threadIdx.x);
    if (clean_0 != nullptr) {
        #pragma unroll
        for (int i = thread_id; i < num_clean_int_0; i += kNumThreads)
            clean_0[i] = 0;
    }
    if (clean_1 != nullptr) {
        #pragma unroll
        for (int i = thread_id; i < num_clean_int_1; i += kNumThreads)
            clean_1[i] = 0;
    }

    // Clean ExpertSyncInfo
    if (expert_sync_info != nullptr) {
        for (int expert_idx = thread_id; expert_idx < num_experts; expert_idx += kNumThreads) {
            // Initialize all fields to zero
            for (int rank = 0; rank < 8; ++rank) {
                expert_sync_info[expert_idx].expected_tokens_per_rank[rank] = 0;
                expert_sync_info[expert_idx].received_tokens_per_rank[rank] = 0;
            }
            expert_sync_info[expert_idx].total_expected_tokens = 0;
            expert_sync_info[expert_idx].total_received_tokens = 0;
            expert_sync_info[expert_idx].completed_ranks = 0;
            expert_sync_info[expert_idx].expert_processing_complete = 0;
            for (int i = 0; i < 1; ++i) {
                expert_sync_info[expert_idx].padding[i] = 0;
            }
        }
    }

    // Barrier after cleaning (make sure the low-latency mode works fine)
    nvshmemx_barrier_all_block();
}

void clean_low_latency_buffer(int* clean_0, int num_clean_int_0,
                              int* clean_1, int num_clean_int_1,
                              cudaStream_t stream) {
    constexpr int kNumThreads = 256;

    // Skip if nothing to clean
    if ((clean_0 == nullptr || num_clean_int_0 == 0) &&
        (clean_1 == nullptr || num_clean_int_1 == 0)) {
        return;
    }

    SETUP_LAUNCH_CONFIG(1, kNumThreads, stream);
    LAUNCH_KERNEL(&cfg, clean_low_latency_buffer<kNumThreads>,
                  clean_0, num_clean_int_0, clean_1, num_clean_int_1);
}

void clean_low_latency_buffer_with_sync(int* clean_0, int num_clean_int_0,
                                       int* clean_1, int num_clean_int_1,
                                       ExpertSyncInfo* expert_sync_info,
                                       int num_experts,
                                       cudaStream_t stream) {
    constexpr int kNumThreads = 256;

    // Skip if nothing to clean (but still need to clean sync info)
    if ((clean_0 == nullptr || num_clean_int_0 == 0) &&
        (clean_1 == nullptr || num_clean_int_1 == 0) &&
        expert_sync_info == nullptr) {
        return;
    }

    SETUP_LAUNCH_CONFIG(1, kNumThreads, stream);
    LAUNCH_KERNEL(&cfg, clean_low_latency_buffer_with_sync<kNumThreads>,
                  clean_0, num_clean_int_0, clean_1, num_clean_int_1,
                  expert_sync_info, num_experts);
}

template <bool kUseFP8, bool kUseUE8M0, int kHidden>
__global__ __launch_bounds__(1024, 1) void
dispatch(void* packed_recv_x, void* packed_recv_x_scales,
         int* packed_recv_src_info, int64_t* packed_recv_layout_range,
         int* packed_recv_count,
         int* cumulative_local_expert_recv_stats,
         int64_t* dispatch_wait_recv_cost_stats,
         void* rdma_recv_x, int* rdma_recv_count, void* rdma_x,
         ExpertSyncInfo* expert_sync_info_buffer,
         void* combined_x,  // combined_x buffer for NVSHMEM GET
         const void* x, const int64_t* topk_idx,
         int* atomic_counter_per_expert, int* atomic_finish_counter_per_expert_unused,
         int* next_clean, int num_next_clean_int,
         const int* num_recv_tokens_per_rank,
         int num_tokens, int num_max_dispatch_tokens_per_rank,
         int num_topk, int num_experts, int rank, int num_ranks,
         int num_warp_groups, int num_warps_per_group,
         bool round_scale, int phases) {
    const auto sm_id = static_cast<int>(blockIdx.x);
    const auto thread_id = static_cast<int>(threadIdx.x);
    const auto warp_id = thread_id / 32, lane_id = get_lane_id();
    const auto num_warps = num_warp_groups * num_warps_per_group;
    const auto num_local_experts = num_experts / num_ranks;

    // In Pure EP mode, num_ranks equals expert_parallel_size and there's no data parallelism
    // This means each rank processes ALL tokens but only for its local experts
    const bool is_pure_ep_mode = (num_ranks == num_experts / num_local_experts);

    // Initialize ExpertSyncInfo at the beginning of dispatch
    if (thread_id == 0 && sm_id == 0 && expert_sync_info_buffer != nullptr) {
        for (int i = 0; i < num_experts; i++) {
            // Initialize all fields to 0
            memset(&expert_sync_info_buffer[i], 0, sizeof(ExpertSyncInfo));
        }
        __threadfence_system();
    }
    __syncthreads();

    const auto warp_group_id = warp_id / num_warps_per_group;
    const auto sub_warp_id = warp_id % num_warps_per_group;
    const auto responsible_expert_idx = blockIdx.x;

    // Initialize diagnostic arrays at kernel start
    if (thread_id == 0) {
        g_sync_counter[sm_id] = 1;  // Block started
        g_block_expert_idx[sm_id] = responsible_expert_idx;
        g_sync_reached[sm_id] = 0;
        g_active_blocks[sm_id] = (responsible_expert_idx < num_experts) ? 1 : 0;
    }

    // May extract UE8M0 from the scales
    using scale_t = std::conditional_t<kUseUE8M0, uint8_t, float>;
    using packed_t = std::conditional_t<kUseUE8M0, uint32_t, float>;
    EP_STATIC_ASSERT(sizeof(packed_t) % sizeof(scale_t) == 0, "Invalid vector length");

    // FP8 staffs
    constexpr int kNumPerChannels = 128;
    const int num_scales = kHidden / kNumPerChannels;
    const size_t hidden_bytes = kHidden * (kUseFP8 ? sizeof(__nv_fp8_storage_t) : sizeof(nv_bfloat16));

    // Message package: hidden data, FP8 scales, index at source
    // NOTES: currently we have 3 reserved int fields for future use
    using vec_t = typename std::conditional<kUseFP8, int2, int4>::type;
    // Separate metadata size from data size for consistency
    const size_t num_bytes_per_data = kUseFP8 ? (kHidden + num_scales * sizeof(float)) : (kHidden * sizeof(nv_bfloat16));
    const size_t num_bytes_per_msg = sizeof(int4) + num_bytes_per_data;  // Metadata + data for RDMA
    const size_t num_int4_per_msg = num_bytes_per_msg / sizeof(int4);
    const size_t num_int4_per_data = num_bytes_per_data / sizeof(int4);  // Data only size
    EP_DEVICE_ASSERT(num_bytes_per_msg % sizeof(int4) == 0);
    EP_DEVICE_ASSERT(num_bytes_per_data % sizeof(int4) == 0);

    // Expert counts
    constexpr int kNumMaxWarpGroups = 32;
    __shared__ int shared_num_tokens_sent_per_expert[kNumMaxWarpGroups];

    // Sending phase
    if ((phases & LOW_LATENCY_SEND_PHASE) == 0) {
        goto LOW_LATENCY_DISPATCH_RECV;
    }


    // NOTE: Finish counter initialization removed - not needed with simplified approach

    // There are 2 kinds of warps in this part:
    // 1. The first-kind warps for FP8 cast and sending top-k tokens
    // 2. The last warp for reading `topk_idx` and count for per-expert information
    if (warp_id < num_warps - 1) {
        constexpr int kNumElemsPerRead = sizeof(int4) / sizeof(nv_bfloat16);
        EP_DEVICE_ASSERT(kHidden % kNumElemsPerRead == 0);
        EP_STATIC_ASSERT(kNumElemsPerRead * 32 % kNumPerChannels == 0, "Invalid vectorization");
        const size_t hidden_bf16_int4 = kHidden / kNumElemsPerRead;

        // Token processing: Each block processes ALL tokens but only sends to its responsible expert
        // This ensures every expert receives its assigned tokens regardless of block assignment


        // All warps except the last one participate in token processing
        // Distribute tokens across warps to prevent duplicate processing
        const int num_processing_warps = num_warps - 1;
        for (int token_idx = warp_id; token_idx < num_tokens; token_idx += num_processing_warps) {
            const auto x_int4 = static_cast<const int4*>(x) + token_idx * num_int4_per_data;
            const auto rdma_x_src_idx = reinterpret_cast<int*>(static_cast<uint8_t*>(rdma_x) + token_idx * num_bytes_per_msg);
            const auto rdma_x_vec = reinterpret_cast<vec_t*>(reinterpret_cast<uint8_t*>(rdma_x_src_idx) + sizeof(int4));
            const auto rdma_x_scales = reinterpret_cast<float*>(reinterpret_cast<uint8_t*>(rdma_x_vec) + hidden_bytes);

            // In Pure EP mode, we need to make this token available to ALL ranks
            // Each rank will process it with their local experts
            if (is_pure_ep_mode) {
                // Store token id header for this token (per-warp leader)
                if (lane_id == 0) {
                    *rdma_x_src_idx = token_idx;
                }
                // __syncwarp();
            } else {
                // Mixed mode: normal processing
                if (lane_id == 0) {
                    *rdma_x_src_idx = token_idx;
                }
                // __syncwarp();
            }

            // FP8 cast (do this once per token, not per expert)
            // All threads in the processing warp participate in FP8 cast
            #pragma unroll
            for (int i = lane_id; i < hidden_bf16_int4; i += 32) {
                // Read
                auto int4_value = __ldg(x_int4 + i);

                if constexpr (kUseFP8) {
                    // Calculate local amax
                    auto bf16_values = reinterpret_cast<nv_bfloat16*>(&int4_value);
                    float fp32_values[kNumElemsPerRead];
                    float amax = kFP8Margin, scale, scale_inv;
                    #pragma unroll
                    for (int j = 0; j < kNumElemsPerRead; ++ j) {
                        fp32_values[j] = static_cast<float>(bf16_values[j]);
                        amax = fmaxf(amax, fabsf(fp32_values[j]));
                    }

                    // Reduce amax and scale
                    EP_STATIC_ASSERT(kNumElemsPerRead * 32 / kNumPerChannels == 2, "Invalid vectorization");
                    amax = half_warp_reduce_max(amax);
                    calculate_fp8_scales(amax, scale, scale_inv, round_scale);
                    if (lane_id == 0 or lane_id == 16)
                        rdma_x_scales[i * kNumElemsPerRead / 128] = scale_inv;

                    // Cast into send buffer
                    vec_t int2_value;
                    auto fp8x2_values = reinterpret_cast<__nv_fp8x2_storage_t*>(&int2_value);
                    #pragma unroll
                    for (int j = 0; j < kNumElemsPerRead; j += 2) {
                        float2 fp32x2 = {fp32_values[j] * scale, fp32_values[j + 1] * scale};
                        fp8x2_values[j / 2] = __nv_cvt_float2_to_fp8x2(fp32x2, __NV_SATFINITE, __NV_E4M3);
                    }
                    rdma_x_vec[i] = int2_value;
                } else {
                    // Reinterpret-cast is for C++14 compatibility
                    rdma_x_vec[i] = *reinterpret_cast<vec_t*>(&int4_value);
                }
            }
            // Sync within the processing warp
            // __syncwarp();

            // Each block only processes tokens destined for its responsible expert
            // This prevents multiple blocks from incrementing the same expert's counter
            // Token ownership masking: In Pure EP, only the owner rank dispatches this token
            int owner_rank_for_token = is_pure_ep_mode ? (token_idx % num_ranks) : -1;

            for (int topk_offset = 0; topk_offset < num_topk; topk_offset++) {
                auto dst_expert_idx = static_cast<int>(__ldg(topk_idx + token_idx * num_topk + topk_offset));

                // Skip invalid experts (e.g., padding tokens with -1)
                if (dst_expert_idx < 0) {
                    continue;
                }

                // Skip if this token is not for our responsible expert
                if (dst_expert_idx != responsible_expert_idx) {
                    continue;
                }

                // Ownership mask: only the owner rank dispatches in Pure EP mode
                if (is_pure_ep_mode && rank != owner_rank_for_token) {
                    continue;
                }

                // Issue IBGDA sends
                if (dst_expert_idx >= 0) {
                    const auto dst_rank = dst_expert_idx / num_local_experts;
                    const auto dst_expert_local_idx = dst_expert_idx % num_local_experts;


                    // Use per-rank counter to match buffer layout
                    // Counter index = dst_expert_idx * num_ranks + source_rank
                    const int counter_idx = dst_expert_idx * num_ranks + rank;
                    int slot_idx = lane_id == 0 ? atomicAdd(atomic_counter_per_expert + counter_idx, 1) : 0;
                    slot_idx = __shfl_sync(0xffffffff, slot_idx, 0);

                    // Add bounds check to prevent buffer overflow
                    if (slot_idx >= num_max_dispatch_tokens_per_rank) {
                        // All threads must hit the assert to ensure proper termination
                        // __syncwarp();
                        assert(false && "Buffer overflow: Token dispatch buffer is full!");
                    }

                    // Process valid slots only

                    // NEW: Update expected token count in ExpertSyncInfo
                    if (lane_id == 0 && expert_sync_info_buffer != nullptr) {
                        atomicAdd(&expert_sync_info_buffer[dst_expert_idx].expected_tokens_per_rank[rank], 1);
                        atomicAdd(&expert_sync_info_buffer[dst_expert_idx].total_expected_tokens, 1);
                    }

                    const auto src_ptr = reinterpret_cast<uint64_t>(rdma_x_src_idx);
                    const auto dst_ptr = reinterpret_cast<uint64_t>(rdma_recv_x) +
                                            dst_expert_local_idx * num_ranks * num_max_dispatch_tokens_per_rank * num_bytes_per_msg +
                                            rank * num_max_dispatch_tokens_per_rank * num_bytes_per_msg +
                                            slot_idx * num_bytes_per_msg;
                    const auto dst_p2p_ptr = nvshmemi_get_p2p_ptr(dst_ptr, rank, dst_rank);
                    if (dst_p2p_ptr == 0) {
                        nvshmemi_ibgda_put_nbi_warp(dst_ptr, src_ptr, num_bytes_per_msg, dst_rank, dst_expert_local_idx, lane_id, slot_idx);
                    } else {
                        // NOTES: only 2 load iterations for 7K hidden with 8 unrolls
                        const auto* src_int4_ptr = reinterpret_cast<const int4*>(src_ptr);
                        const auto* dst_int4_ptr = reinterpret_cast<int4*>(dst_p2p_ptr);
                        UNROLLED_WARP_COPY(8, lane_id, num_int4_per_msg, dst_int4_ptr, src_int4_ptr, ld_nc_global, st_na_global);
                        // Enhanced synchronization for P2P write
                        // __syncwarp();
                        // __threadfence_system();
                        asm volatile("membar.sys;");  // Additional barrier for P2P
                    }

                    // Increase counter after finishing
                    // if (dst_p2p_ptr == 0) {
                    //     // For RDMA path, sync after the operation
                    //     __syncwarp();  // First sync warp
                    //     __threadfence_system();  // Then memory fence to ensure RDMA writes are visible
                    // }
                    // NOTE: Per-token finish counter update removed - will update once after all tokens sent
                }  // End of if (dst_expert_idx >= 0)
            }  // End of expert processing loop
        }  // End of token processing loop
    }

    if (warp_id == num_warps - 1) {

        if (sm_id == 0) {

            // The first SM is also responsible for cleaning the next buffer
            #pragma unroll
            for (int i = lane_id; i < num_next_clean_int; i += 32)
                next_clean[i] = 0;
        }

        // This SM should be responsible for one destination expert
        const auto expert_begin_idx = blockIdx.x;
        const auto expert_end_idx = min(expert_begin_idx + 1, num_experts);

        // Store the actual sent count for later use
        if (expert_begin_idx < num_experts && thread_id == 0) {
            const int my_counter_idx = expert_begin_idx * num_ranks + rank;
            int actual_sent_count = atomic_counter_per_expert[my_counter_idx];

            // Store the actual sent count for count send phase
            shared_num_tokens_sent_per_expert[0] = actual_sent_count;
        }
    }
    __syncthreads();

    // Issue count sends - each block handles its own expert
    // IMPORTANT: This section must complete quickly to avoid grid sync deadlock

    if (responsible_expert_idx < num_experts) {
        // Only one thread per block performs the count send
        if (sub_warp_id == 0 and lane_id == 0) {
        // The actual count is in atomic_counter_per_expert after all sends
        // This reflects tokens actually sent (excluding dropped ones)
        // Since we now use per-rank counters, need to sum all ranks
        // This SM processed tokens from current rank only
        const int my_counter_idx = responsible_expert_idx * num_ranks + rank;
        const auto num_tokens_sent = atomic_counter_per_expert[my_counter_idx];

        // Calculate expected count from ALL tokens
        // Each warp processes only its assigned tokens
        int expected_count = 0;

        // Check all tokens for this expert, but only count tokens processed by all warps
        for (int token_idx = 0; token_idx < num_tokens; token_idx++) {
            // Check top-k experts for this token
            for (int k = 0; k < num_topk; k++) {
                auto idx = static_cast<int>(__ldg(topk_idx + token_idx * num_topk + k));
                // Skip invalid experts (padding tokens)
                if (idx < 0) {
                    continue;
                }
                if (idx == responsible_expert_idx) {
                    // Ownership masking: In Pure EP, only the owner rank sends this token
                    if (!is_pure_ep_mode) {
                        expected_count++;
                    } else {
                        const int owner_rank = token_idx % num_ranks;
                        if (owner_rank == rank) expected_count++;
                    }
                }
            }
        }

        // Check for token drops - MUST NEVER HAPPEN
        if (num_tokens_sent < expected_count) {
            printf("[FATAL ERROR] Token drop detected!\n");
            printf("  Expert %d: Sent only %d tokens (expected %d, dropped %d)\n",
                   responsible_expert_idx, num_tokens_sent, expected_count, expected_count - num_tokens_sent);
            printf("  This is a critical error that must be fixed\n");
            printf("  Block: %d, Rank: %d, num_warps: %d, num_topk: %d\n", sm_id, rank, num_warps, num_topk);
            assert(false && "Token drop detected! This must never happen.");
        }

        // Count send moved to after grid sync to prevent deadlock

        // Note: packed_recv_count is already initialized to zero in deep_ep.cpp (line 1123)
        // No need to initialize it here in the kernel
        }  // End of if (sub_warp_id == 0 and lane_id == 0)
    }  // End of if (responsible_expert_idx < num_experts) - moved brace here

    // Ensure all threads in the block complete before moving on
    // This must happen OUTSIDE the if statement to ensure ALL blocks sync
    __syncthreads();

    if (responsible_expert_idx >= num_experts) {
        if (thread_id == 0) {
            g_active_blocks[sm_id] = 0;  // Mark as inactive
        }
    }

    // Ensure ALL blocks reach this point before grid sync
    // This barrier ensures that both active and inactive blocks are at the same point
    __syncthreads();



    // Track if this block actually performed any NVSHMEM puts
    __shared__ int block_put_count;
    if (thread_id == 0) {
        if (responsible_expert_idx < num_experts) {
            const int my_counter_idx = responsible_expert_idx * num_ranks + rank;
            block_put_count = atomic_counter_per_expert[my_counter_idx];
        } else {
            block_put_count = 0;
        }
    }
    __syncthreads();

    // Don't use nvshmem_quiet() here - it causes deadlock
    // Grid sync is sufficient to ensure all blocks complete
    __syncthreads();  // Ensure all threads wait

    // Grid sync after send phase to ensure all ranks complete sending
    // This must happen before any rank checks if it should skip receive phase
    if (phases & LOW_LATENCY_SEND_PHASE) {
        // Update sync counter before sync
        if (thread_id == 0) {
            g_sync_counter[sm_id] = 2;  // Approaching sync
            g_sync_reached[sm_id] = 1;
        }

        __syncthreads();  // Block-wide sync first

        // Check for conditional sync skip
        #ifdef DEEPEP_SKIP_GRID_SYNC
        #else
            cg::this_grid().sync();  // Then grid-wide sync

            if (thread_id == 0) {
                g_sync_counter[sm_id] = 3;  // Passed sync
            }
        #endif

        __threadfence_system();  // Ensure all writes are visible
    }

    // Send counts after dispatch send phase completes
    // This MUST be outside the send phase block to handle return_recv_hook case
    // When return_recv_hook=true, dispatch only runs send phase and exits
    // Count must still be sent for later combine kernel to work correctly
    //
    // The rdma_recv_count buffer MUST NOT be cleared between dispatch
    // send and receive phases! The clean_low_latency_buffer function now excludes
    // the count buffer to preserve these values.
    //
        // Always send counts regardless of phase to prevent deadlock
        // All ranks must send counts, even if they only run RECV_PHASE
        // EACH BLOCK sends count for its responsible expert only
    if (responsible_expert_idx < num_experts && thread_id == 0 && atomic_counter_per_expert != nullptr) {
        const int dst_rank = responsible_expert_idx / num_local_experts;
        const int dst_expert_local_idx = responsible_expert_idx % num_local_experts;

        // This is the current rank's count for this expert
        const int counter_idx = responsible_expert_idx * num_ranks + rank;
        int actual_sent_count = atomic_counter_per_expert[counter_idx];

        // Ownership masking: In Pure EP, ensure non-owner ranks report zero for non-owned tokens
        if (is_pure_ep_mode) {
            // NOTE: atomic_counter_per_expert는 누적 카운터이므로, 여기서는 음수 카운트 프로토콜(-n-1)을 유지하면서
            //       비-오너 랭크가 보낸 토큰이 없도록 위에서 디스패치가 이미 차단되어야 합니다.
            //       혹시라도 누수가 있으면 카운트를 0으로 강제하여 리시브 측 프로토콜 일관성을 보장합니다.
            if (actual_sent_count < 0) actual_sent_count = 0;
        }

        // We need to write to the destination rank's buffer
        // The count buffer is arranged as [num_local_experts][num_ranks]
        // We want to write to dst_rank's buffer at position [dst_expert_local_idx][src_rank]
        // where src_rank is us (rank)

        // Calculate the offset in the destination rank's count buffer
        // dst_expert_local_idx: which local expert on the destination rank
        // rank: our rank (the source rank from dst's perspective)
        int dst_offset = dst_expert_local_idx * num_ranks + rank;

        // Get the NVSHMEM pointer for the destination rank's count buffer
        // rdma_recv_count is in the NVSHMEM symmetric heap
        auto dst_addr = reinterpret_cast<uint64_t>(rdma_recv_count + dst_offset);
        auto dst_p2p_ptr = nvshmemi_get_p2p_ptr(dst_addr, rank, dst_rank);

        if (dst_p2p_ptr == 0) {
            // Use IBGDA for remote access
            #ifdef DEEPEP_VERBOSE_DEBUG
            printf("[COUNT_SEND_IBGDA] rank=%d block=%d -> rank=%d: sending count=%d to expert=%d (offset=%d)",
                   rank, sm_id, dst_rank, -actual_sent_count - 1, dst_expert_local_idx, dst_offset);
            #endif
            nvshmemi_ibgda_amo_nonfetch_add(rdma_recv_count + dst_offset,
                -actual_sent_count - 1, dst_rank, dst_expert_local_idx);
        } else {
            // Use atomic for P2P path
            #ifdef DEEPEP_VERBOSE_DEBUG
            printf("[COUNT_SEND_P2P] rank=%d block=%d -> rank=%d: sending count=%d to expert=%d (offset=%d)",
                   rank, sm_id, dst_rank, -actual_sent_count - 1, dst_expert_local_idx, dst_offset);
            #endif
            atomicAdd_system(reinterpret_cast<int*>(dst_p2p_ptr),
                -actual_sent_count - 1);
        }

        // Ensure count send is visible before proceeding
        __threadfence_system();
    }

    // Ensure all blocks have sent their counts before any block starts receiving
    // This prevents the timeout issue where blocks wait for counts that haven't been sent yet
    __syncthreads();  // Block-wide sync first
    cg::this_grid().sync();  // Grid-wide sync to ensure all counts are sent
    __threadfence_system();  // Ensure all count writes are globally visible


    // Receiving phase
    LOW_LATENCY_DISPATCH_RECV:

    if ((phases & LOW_LATENCY_RECV_PHASE) == 0) {
        return;
    }


    // Determine if this rank should process tokens
    bool should_process = (num_recv_tokens_per_rank[rank] > 0);

    // Don't return early! Even if this rank doesn't receive tokens,
    // blocks still need to participate in the receive protocol
    // The early return was causing some blocks to never execute their receive logic
    // which could cause hangs in other ranks waiting for responses

    // Receiving and packing
    // In dispatch receive, we need to receive data sent TO our local experts FROM other ranks
    // Each block processes one (local_expert, src_rank) pair
    if (should_process && responsible_expert_idx < num_local_experts * num_ranks) {
        const int pair_idx = responsible_expert_idx;
        const auto local_expert_idx = pair_idx / num_ranks;  // Which of OUR local experts
        const auto src_rank = pair_idx % num_ranks;  // From which rank

        // Multi-node validation: ensure expert indices are within valid range
        EP_DEVICE_ASSERT(local_expert_idx < num_local_experts);
        EP_DEVICE_ASSERT(src_rank < num_ranks);

        const auto rdma_recv_x_uint8 = static_cast<uint8_t*>(rdma_recv_x) +
                local_expert_idx * num_ranks * num_max_dispatch_tokens_per_rank * num_bytes_per_msg +
                src_rank * num_max_dispatch_tokens_per_rank * num_bytes_per_msg;
        // Use data-only size for consistency with buffer layout
        const auto recv_src_info = packed_recv_src_info +
                local_expert_idx * num_ranks * num_max_dispatch_tokens_per_rank +
                src_rank * num_max_dispatch_tokens_per_rank;
        const auto recv_range = packed_recv_layout_range + local_expert_idx * num_ranks;
        const auto num_aligned_scales = align<int>(num_scales, sizeof(float) / sizeof(scale_t));
        const auto recv_x_scales = static_cast<scale_t*>(packed_recv_x_scales) + local_expert_idx * num_ranks * num_max_dispatch_tokens_per_rank * num_aligned_scales;

        // Shared between sub-warps in warp groups
        __shared__ int shared_num_recv_tokens[kNumMaxWarpGroups], shared_recv_token_begin_idx[kNumMaxWarpGroups];

        // Wait tokens to arrive
        // using sub-warp 1 to overlap with sub-warp 0
        int num_recv_tokens, recv_token_begin_idx;
        EP_DEVICE_ASSERT(num_warps_per_group > 1 and num_warp_groups < 15);
        if (sub_warp_id == 1 and lane_id == 0) {
            // Wait for count to arrive
            // Count protocol: -actual_count - 1 (so -1 means 0 tokens)
            num_recv_tokens = 0;
            int wait_iterations = 0;
            const int MAX_WAIT = 100000000;  // 100M iterations

            // In Pure EP mode, we must wait for counts from ALL ranks
            // because padding tokens might cause some experts to receive 0 tokens
            // but the count (-1) must still be sent and received
            while (true) {
                num_recv_tokens = ld_acquire_sys_global(rdma_recv_count + local_expert_idx * num_ranks + src_rank);

                // Check if count has arrived (any non-zero value)
                if (num_recv_tokens != 0) {
                    // Count received! Break out of wait loop
                    break;
                }

                // Still waiting...
                wait_iterations++;
                // Timeout check - but this should not happen in correct operation
                if (wait_iterations > MAX_WAIT) {
                    printf("[FATAL] rank=%d block=%d timeout waiting for count! local_expert=%d, src_rank=%d\n",
                           rank, blockIdx.x, local_expert_idx, src_rank);
                    printf("[FATAL] Expected count at rdma_recv_count[%d], but value is still %d\n",
                           local_expert_idx * num_ranks + src_rank, num_recv_tokens);
                    printf("[FATAL] This indicates src_rank=%d never sent count to dst_rank=%d for expert=%d\n",
                           src_rank, rank, local_expert_idx + rank * num_local_experts);
                    assert(false && "Timeout waiting for count - synchronization failure");
                }
            }

            // Process only if we have tokens
            if (num_recv_tokens != 0) {
                num_recv_tokens = -num_recv_tokens - 1;

            // Buffer layout is [local_expert][from_rank][token]
            // Each source rank has its own dedicated range in the buffer
            // We need a per-rank counter, not a global counter per expert
            // Use a temporary atomic counter in packed_recv_count for each (expert, rank) pair
            const int counter_idx = local_expert_idx * num_ranks + src_rank;
            recv_token_begin_idx = atomicAdd(packed_recv_count + counter_idx, num_recv_tokens);

            // Ensure we don't overflow the per-rank section of the buffer
            if (recv_token_begin_idx + num_recv_tokens > num_max_dispatch_tokens_per_rank) {
                // Assert instead of clamping to prevent silent data loss
                printf("[FATAL ERROR] Receive buffer overflow detected!\n");
                printf("  Rank: %d\n", rank);
                printf("  Local expert: %d (global expert %d)\n", local_expert_idx, rank * num_local_experts + local_expert_idx);
                printf("  Source rank: %d\n", src_rank);
                printf("  Receive begin index: %d\n", recv_token_begin_idx);
                printf("  Number of tokens to receive: %d\n", num_recv_tokens);
                printf("  Max tokens per rank: %d\n", num_max_dispatch_tokens_per_rank);
                printf("  Total would be: %d (exceeds buffer capacity)\n", recv_token_begin_idx + num_recv_tokens);
                printf("  packed_recv_count[%d] was: %d before this addition\n",
                       local_expert_idx, recv_token_begin_idx);

                // Force an assertion failure
                assert(false && "Receive buffer overflow: Too many tokens for expert! Increase num_max_dispatch_tokens_per_rank.");
            }

            shared_num_recv_tokens[warp_group_id] = num_recv_tokens;
            shared_recv_token_begin_idx[warp_group_id] = recv_token_begin_idx;
            recv_range[src_rank] = pack2<int, int64_t>(num_recv_tokens, recv_token_begin_idx);
            if (cumulative_local_expert_recv_stats != nullptr)
                atomicAdd(cumulative_local_expert_recv_stats + local_expert_idx, num_recv_tokens);
            } else {
                // No tokens for this (expert, rank) pair
                shared_num_recv_tokens[warp_group_id] = 0;
                shared_recv_token_begin_idx[warp_group_id] = 0;
                if (lane_id == 0) {
                    printf("[RECV SKIP] Block %d: No tokens for local_expert=%d from src_rank=%d\n",
                           blockIdx.x, local_expert_idx, src_rank);
                }
            }
        }
        asm volatile("bar.sync %0, %1;" :: "r"(warp_group_id + 2), "r"(num_warps_per_group * 32));
        num_recv_tokens = shared_num_recv_tokens[warp_group_id];
        recv_token_begin_idx = shared_recv_token_begin_idx[warp_group_id];

        // Skip token copying if no tokens
        if (num_recv_tokens == 0) {
            return;
        }

        // Copy tokens
        EP_DEVICE_ASSERT(num_scales <= 64);
        for (int i = sub_warp_id; i < num_recv_tokens; i += num_warps_per_group) {
            // Copy source info
            const auto src_src_idx = reinterpret_cast<int*>(rdma_recv_x_uint8 + i * num_bytes_per_msg);
            if (lane_id == 0)
                // Now recv_src_info already points to the correct rank's section
                recv_src_info[i] = ld_nc_global(src_src_idx);
            // __syncwarp();

            // Copy data
            // NOTES: only 2 load iterations for 7K hidden with 7 unrolls
            const auto src_data = reinterpret_cast<int4*>(reinterpret_cast<uint8_t*>(src_src_idx) + sizeof(int4));

            // Calculate destination address - packed_recv_x is just data, no metadata
            // Buffer layout: [local_expert][from_rank][token][hidden]
            // Use consistent data-only size (num_int4_per_data)

            // Verify offset calculation doesn't exceed buffer bounds
            const int token_slot_idx = recv_token_begin_idx + i;
            if (token_slot_idx >= num_max_dispatch_tokens_per_rank) {
                if (lane_id == 0) {
                    printf("[FATAL] Token slot index %d exceeds buffer capacity %d!\n",
                           token_slot_idx, num_max_dispatch_tokens_per_rank);
                    printf("  recv_token_begin_idx=%d, i=%d, local_expert=%d, src_rank=%d\n",
                           recv_token_begin_idx, i, local_expert_idx, src_rank);
                    assert(false && "Token slot index exceeds buffer bounds!");
                }
            }

            const auto dst_data = static_cast<int4*>(packed_recv_x) +
                local_expert_idx * num_ranks * num_max_dispatch_tokens_per_rank * num_int4_per_data +
                src_rank * num_max_dispatch_tokens_per_rank * num_int4_per_data +
                token_slot_idx * num_int4_per_data;

            // Use data-only size for copy
            UNROLLED_WARP_COPY(7, lane_id, num_int4_per_data, dst_data, src_data, ld_nc_global, st_na_global);
            // Copy scales
            if constexpr (kUseFP8) {
                // Equivalent CuTe layout:
                //   (num_tokens, (num_packed, num_elems_per_pack)):(num_elems_per_pack, (num_tokens * num_elems_per_pack, 1))
                const auto src_scales = reinterpret_cast<float*>(reinterpret_cast<uint8_t*>(src_data) + hidden_bytes);
                const auto num_elems_per_pack = static_cast<int>(sizeof(packed_t) / sizeof(scale_t));
                const auto token_idx = recv_token_begin_idx + i;
                const auto token_stride = num_elems_per_pack;
                const auto pack_stride = num_ranks * num_max_dispatch_tokens_per_rank * num_elems_per_pack;
                if (lane_id < num_scales) {
                    const auto pack_idx = lane_id / num_elems_per_pack;
                    const auto elem_idx = lane_id % num_elems_per_pack;
                    auto scale = extract_required_scale_format<kUseUE8M0>(ld_nc_global(src_scales + lane_id));
                    recv_x_scales[token_idx * token_stride + pack_idx * pack_stride + elem_idx] = scale;
                }
                if (lane_id + 32 < num_scales) {
                    const auto pack_idx = (lane_id + 32) / num_elems_per_pack;
                    const auto elem_idx = (lane_id + 32) % num_elems_per_pack;
                    auto scale = extract_required_scale_format<kUseUE8M0>(ld_nc_global(src_scales + lane_id + 32));
                    recv_x_scales[token_idx * token_stride + pack_idx * pack_stride + elem_idx] = scale;
                }
            }
        }

        // Add system-wide memory fence to ensure all writes are visible
        // This is critical for cross-node RDMA operations
        // All threads must execute the fence for consistency
        __threadfence_system();
    }

    // Final diagnostic: kernel exit
    if (thread_id == 0) {
        g_sync_counter[sm_id] = 4;  // Kernel completed
    }
}

void dispatch(void* packed_recv_x, void* packed_recv_x_scales,
              int* packed_recv_src_info, int64_t* packed_recv_layout_range,
              int* packed_recv_count,
              int* cumulative_local_expert_recv_stats,
              int64_t* dispatch_wait_recv_cost_stats,
              void* rdma_recv_x, int* rdma_recv_count, void* rdma_x,
              ExpertSyncInfo* expert_sync_info_buffer,
              void* combined_x,  // NEW: combined_x buffer for NVSHMEM GET
              const void* x, const int64_t* topk_idx,
              int* next_clean, int num_next_clean_int,
              const int* num_recv_tokens_per_rank,
              int num_tokens, int hidden, int num_max_dispatch_tokens_per_rank,
              int num_topk, int num_experts, int rank, int num_ranks,
              bool use_fp8, bool round_scale, bool use_ue8m0,
              void* workspace, int num_device_sms,
              cudaStream_t stream, int phases) {
    constexpr int kNumMaxTopK = 9;

    // Ensure we have enough warps to handle all top-k experts
    // We need at least num_topk + 1 warps (num_topk for processing, 1 for counting)
    const int min_warps_needed = num_topk + 1;

    // Calculate num_warp_groups based on the constraint that we need enough warps
    int num_warp_groups = 1;
    int num_warps_per_group = 32;

    // Find the best configuration
    for (int wg = 1; wg <= 32; wg++) {
        int wpg = 32 / wg;
        if (wpg > 0 && wg * wpg >= min_warps_needed && wg <= ceil_div(num_experts, num_device_sms)) {
            num_warp_groups = wg;
            num_warps_per_group = wpg;
            break;
        }
    }

    // Verify we have enough warps
    const auto num_warps = num_warp_groups * num_warps_per_group;
    if (num_warps < min_warps_needed) {
        printf("FATAL: Cannot allocate enough warps! Need %d warps but only have %d\n",
               min_warps_needed, num_warps);
        printf("  num_topk=%d, num_experts=%d, num_device_sms=%d\n",
               num_topk, num_experts, num_device_sms);
        EP_HOST_ASSERT(false && "Insufficient warps for top-k processing");
    }

    EP_HOST_ASSERT(num_warp_groups > 0 and num_warps_per_group > 0);
    EP_HOST_ASSERT(kNumMaxTopK + 1 <= num_warp_groups * num_warps_per_group);

    const auto num_sms = ceil_div(num_experts, num_warp_groups);
    EP_HOST_ASSERT(num_topk <= kNumMaxTopK);

    // Allocate workspace for per-rank counters
    // Layout: [num_experts][num_ranks] for both counter types
    const int counters_per_type = num_experts * num_ranks;
    auto atomic_counter_per_expert_rank = static_cast<int*>(workspace);
    auto atomic_finish_counter_per_expert_rank = atomic_counter_per_expert_rank + counters_per_type;
    // For backward compatibility, also create pointers to per-expert counters
    auto atomic_counter_per_expert = atomic_counter_per_expert_rank;  // Will be used with rank offset
    auto atomic_finish_counter_per_expert_unused = atomic_finish_counter_per_expert_rank;  // Not used anymore
    EP_HOST_ASSERT(counters_per_type * sizeof(int) * 2 <= NUM_WORKSPACE_BYTES);

    // FP8 checks
    if (use_ue8m0)
        EP_HOST_ASSERT(round_scale and "UE8M0 SF requires `round_scale=True`");

#define DISPATCH_LAUNCH_CASE(hidden) { \
auto dispatch_func = dispatch<false, false, hidden>; \
if (use_fp8 and not use_ue8m0) \
    dispatch_func = dispatch<true, false, hidden>; \
if (use_fp8 and use_ue8m0) \
    dispatch_func = dispatch<true, true, hidden>; \
LAUNCH_KERNEL(&cfg, dispatch_func, \
              packed_recv_x, packed_recv_x_scales, \
              packed_recv_src_info, packed_recv_layout_range, \
              packed_recv_count, \
              cumulative_local_expert_recv_stats, \
              dispatch_wait_recv_cost_stats, \
              rdma_recv_x, rdma_recv_count, rdma_x, \
              expert_sync_info_buffer, \
              combined_x, \
              x, topk_idx, \
              atomic_counter_per_expert, atomic_finish_counter_per_expert_unused, \
              next_clean, num_next_clean_int, \
              num_recv_tokens_per_rank, \
              num_tokens, num_max_dispatch_tokens_per_rank, \
              num_topk, num_experts, rank, num_ranks, \
              num_warp_groups, num_warps_per_group, \
              round_scale, phases); } break

    // Set grid size to num_experts to ensure all experts are processed
    // Each block will handle one expert
    const int dispatch_grid_size = num_experts;
    SETUP_LAUNCH_CONFIG(dispatch_grid_size, num_warps * 32, stream);
    SWITCH_HIDDEN(DISPATCH_LAUNCH_CASE);
#undef DISPATCH_LAUNCH_CASE
}

template <int kHidden, int kNumMaxTopk>
__global__ __launch_bounds__(1024, 1) void
combine(void* combined_x,
        void* fp32_workspace,  // Separate FP32 workspace for NVSHMEM reduction
        void* rdma_recv_x, int* rdma_recv_flag, void* rdma_send_x,
        const void* x, const int64_t* topk_idx, const float* topk_weights,
        const int* src_info, const int64_t* layout_range,
        int64_t* combine_wait_recv_cost_stats,
        int* next_clean, int num_next_clean_int,
        int* atomic_clean_flag,
        ExpertSyncInfo* expert_sync_info_buffer,
        int num_combined_tokens, int hidden, int num_topk,
        int num_max_dispatch_tokens_per_rank,
        int num_experts, int rank, int num_ranks,
        int num_warp_groups, int num_warps_per_group,
        int phases, bool zero_copy) {
    const auto sm_id = static_cast<int>(blockIdx.x);
    // Use gridDim.x which equals num_experts from the launch config
    const auto num_sms = static_cast<int>(gridDim.x);
    const auto thread_id = static_cast<int>(threadIdx.x);
    const auto num_threads = static_cast<int>(blockDim.x);
    const auto warp_id = thread_id / 32, lane_id = get_lane_id();
    const auto num_local_experts = num_experts / num_ranks;
    const auto warp_group_id = warp_id / num_warps_per_group;
    const auto sub_warp_id = warp_id % num_warps_per_group;

    // Store num_topk in shared memory immediately to prevent corruption
    __shared__ int shared_num_topk;
    __shared__ int shared_num_experts;
    __shared__ int shared_num_ranks;
    __shared__ int shared_num_combined_tokens;
    __shared__ int shared_hidden;
    __shared__ int shared_num_max_dispatch_tokens_per_rank;

    if (thread_id == 0) {
        shared_num_topk = num_topk;
        shared_num_experts = num_experts;
        shared_num_ranks = num_ranks;
        shared_num_combined_tokens = num_combined_tokens;
        shared_hidden = hidden;
        shared_num_max_dispatch_tokens_per_rank = num_max_dispatch_tokens_per_rank;
    }
    __syncthreads();

    // Use shared memory values from now on
    int safe_num_topk = shared_num_topk;
    int safe_num_experts = shared_num_experts;
    int safe_num_ranks = shared_num_ranks;
    int safe_num_combined_tokens = shared_num_combined_tokens;
    int safe_hidden = shared_hidden;
    int safe_num_max_dispatch_tokens_per_rank = shared_num_max_dispatch_tokens_per_rank;
    int safe_num_local_experts = safe_num_experts / safe_num_ranks;

    //  Detect Pure EP mode
    const bool is_pure_ep_mode = (safe_num_ranks == safe_num_experts / safe_num_local_experts);

    // blockIdx.x is now limited by grid_size (num_device_sms), not num_experts
    const auto responsible_expert_idx = blockIdx.x;

    // Data type staffs
    constexpr int kNumElemsPerInt4 = sizeof(int4) / sizeof(nv_bfloat16);

    // Use actual hidden size from safe parameter, not template parameter
    const size_t actual_num_bytes_per_data = safe_hidden * sizeof(nv_bfloat16);
    const size_t actual_num_int4_per_data = actual_num_bytes_per_data / sizeof(int4);

    // Verify alignment
    EP_DEVICE_ASSERT(safe_hidden % kNumElemsPerInt4 == 0);
    EP_DEVICE_ASSERT(actual_num_bytes_per_data % sizeof(int4) == 0);

    // For compile-time checks, keep template-based calculations
    constexpr size_t num_bytes_per_data = kHidden * sizeof(nv_bfloat16);
    constexpr size_t num_bytes_per_msg = sizeof(int4) + num_bytes_per_data;
    EP_STATIC_ASSERT(num_bytes_per_data % sizeof(int4) == 0, "Invalid vectorization");
    EP_STATIC_ASSERT(num_bytes_per_msg % sizeof(int4) == 0, "Invalid message size");

    // dispatch uses num_bytes_per_msg for RDMA transfers (includes metadata)
    // But packed_recv_x buffer uses data-only layout

    // Count sending must be done in dispatch kernel, not combine
    // The combine kernel doesn't have access to rdma_recv_count buffer
    // Count sending was moved to dispatch kernel to fix this issue

    // Declare expert_idx before goto to avoid initialization bypass
    int expert_idx;

    // Sending phase
    if ((phases & LOW_LATENCY_SEND_PHASE) == 0)
        goto LOW_LATENCY_COMBINE_RECV;

    // Clean up next buffer
    if (sm_id == 0 and warp_group_id == 0 and sub_warp_id == 0) {
        #pragma unroll
        for (int i = lane_id; i < num_next_clean_int; i += 32)
            next_clean[i] = 0;

        // Notify before executing `int_p`
        // __syncwarp();
        if (lane_id == 0)
            atomic_add_release_global(atomic_clean_flag, num_experts);
    }

    // Issue IBGDA sends
    // Each block processes its assigned expert
    expert_idx = responsible_expert_idx;
    if (expert_idx < num_experts) {
        const auto dst_rank = expert_idx / num_local_experts;
        const auto local_expert_idx = expert_idx % num_local_experts;
        const auto layout = __ldg(layout_range + local_expert_idx * num_ranks + dst_rank);
        const auto local_x = static_cast<const int4*>(x) +
                local_expert_idx * safe_num_ranks * safe_num_max_dispatch_tokens_per_rank * actual_num_int4_per_data;
        const auto local_src_info = src_info + local_expert_idx * safe_num_ranks * safe_num_max_dispatch_tokens_per_rank;
        const auto rdma_send_x_vec = static_cast<uint8_t*>(rdma_send_x) +
                local_expert_idx * safe_num_ranks * safe_num_max_dispatch_tokens_per_rank * num_bytes_per_data;

        // Unpack layout
        int offset = 0, num_tokens_to_send = 0;
        unpack2(layout, num_tokens_to_send, offset);

        // Issue IBGDA send
        for (int token_idx = offset + sub_warp_id; token_idx < offset + num_tokens_to_send; token_idx += num_warps_per_group) {
            // local_x already points to this expert's data, need to add rank offset
            const auto x_int4 = local_x + dst_rank * safe_num_max_dispatch_tokens_per_rank * actual_num_int4_per_data + token_idx * actual_num_int4_per_data;
            // Add rank offset for rdma_send_x_vec access
            const auto rdma_send_type_row = reinterpret_cast<int*>(rdma_send_x_vec + dst_rank * safe_num_max_dispatch_tokens_per_rank * num_bytes_per_data + token_idx * num_bytes_per_data);
            const auto rdma_send_x_vec_row = reinterpret_cast<uint8_t*>(rdma_send_type_row);

            // Copy directly to local rank, or copy to buffer and issue RDMA
            // Add rank offset for src_info access
            const auto src_idx = __ldg(local_src_info + dst_rank * safe_num_max_dispatch_tokens_per_rank + token_idx);
            const auto buf_ptr = reinterpret_cast<int64_t>(rdma_send_x_vec_row);

            // Calculate slot_idx from token_idx - offset
            // token_idx ranges from offset to offset+num_tokens_to_send-1
            // slot_idx should range from 0 to num_tokens_to_send-1
            const auto slot_idx = token_idx - offset;

            // Bounds check for slot_idx
            if (slot_idx < 0 || slot_idx >= safe_num_max_dispatch_tokens_per_rank) {
                if (lane_id == 0) {
                    printf("[ERROR] combine kernel: slot_idx out of bounds! token_idx=%d, offset=%d, slot_idx=%d, max=%d",
                           token_idx, offset, slot_idx, safe_num_max_dispatch_tokens_per_rank);
                }
                continue;
            }
            // Use local expert index and rank-based offset calculation to match dispatch buffer layout
            // Must use num_bytes_per_msg to match dispatch's buffer layout
            const auto dst_ptr = reinterpret_cast<uint64_t>(rdma_recv_x) +
                                 (local_expert_idx * safe_num_ranks * safe_num_max_dispatch_tokens_per_rank +
                                  rank * safe_num_max_dispatch_tokens_per_rank +
                                  slot_idx) * num_bytes_per_msg;
            const auto dst_p2p_ptr = nvshmemi_get_p2p_ptr(dst_ptr, rank, dst_rank);
            if (dst_p2p_ptr == 0) {
                // IBGDA path: prepare a contiguous message with [header(int4) | data]
                // 1) Copy data right after header space in the local send buffer
                const auto buf_int4_ptr = reinterpret_cast<int4*>(buf_ptr);
                if (not zero_copy)
                    UNROLLED_WARP_COPY(7, lane_id, actual_num_int4_per_data, buf_int4_ptr + 1, x_int4, ld_nc_global, st_na_global);

                // 2) Write header (source token index) into the first int of the header int4
                if (lane_id == 0) {
                    reinterpret_cast<int*>(rdma_send_x_vec_row)[0] = src_idx;
                }
                // __syncwarp();

                // 3) Send header then payload
                const auto dst_expert_local_idx = expert_idx % num_local_experts;
                // send header (int4 = 16 bytes)
                nvshmemi_ibgda_put_nbi_warp(dst_ptr, buf_ptr, sizeof(int4), dst_rank, dst_expert_local_idx, lane_id, slot_idx);
                // send data (bf16 payload)
                nvshmemi_ibgda_put_nbi_warp(dst_ptr + sizeof(int4), buf_ptr + sizeof(int4), hidden * sizeof(nv_bfloat16), dst_rank, dst_expert_local_idx, lane_id, slot_idx);
            } else {
                // P2P path: write header then data directly to remote mapped memory
                if (lane_id == 0) {
                    st_na_global(reinterpret_cast<int*>(dst_p2p_ptr), src_idx);
                }
                // __syncwarp();
                const auto dst_int4_ptr = reinterpret_cast<int4*>(reinterpret_cast<uint8_t*>(dst_p2p_ptr) + sizeof(int4));
                UNROLLED_WARP_COPY(7, lane_id, actual_num_int4_per_data, dst_int4_ptr, x_int4, ld_nc_global, st_na_global);
                // Ensure visibility before flag set
                // __syncwarp();
                // __threadfence_system();
                asm volatile("membar.sys;");
            }
        }

        // Put the finishing flag
        EP_DEVICE_ASSERT(num_warps_per_group > 1 and num_warp_groups < 16);
        asm volatile("bar.sync %0, %1;" :: "r"(warp_group_id + 1), "r"(num_warps_per_group * 32));

        // Ensure ALL P2P writes from ALL warps complete before ANY flag is set
        __syncthreads();  // Block-wide synchronization

        if (num_tokens_to_send > 0 && sub_warp_id == 1 && lane_id == 0) {
            // Wait for all local processing to complete
            while (ld_acquire_global(atomic_clean_flag) == 0);

            // Use more robust synchronization
            // First, ensure all data writes are globally visible
            __threadfence_system();

            // Memory barrier to guarantee ordering
            asm volatile("membar.sys;");

            // Now update the flag with proper synchronization
            // Use dst_expert_local_idx for the destination's flag buffer
            const auto dst_expert_local_idx = expert_idx % num_local_experts;
            auto flag_dst_ptr = reinterpret_cast<uint64_t>(rdma_recv_flag + dst_expert_local_idx);
            auto flag_dst_p2p_ptr = nvshmemi_get_p2p_ptr(flag_dst_ptr, rank, dst_rank);

            // Use correct QP ID for remote operations
            if (flag_dst_p2p_ptr == 0) {
                // For remote updates, use NVSHMEM atomic with destination's local expert index as QP ID
                nvshmemi_ibgda_amo_nonfetch_add(reinterpret_cast<int*>(flag_dst_ptr), 1, dst_rank, dst_expert_local_idx);
            } else {
                // For P2P, use atomic with memory ordering
                atomicAdd_system(reinterpret_cast<int*>(flag_dst_p2p_ptr), 1);
            }

            // Update received token count in ExpertSyncInfo
            if (expert_sync_info_buffer != nullptr) {
                // Update the destination expert's received count
                atomicAdd(&expert_sync_info_buffer[expert_idx].received_tokens_per_rank[rank], num_tokens_to_send);
                atomicAdd(&expert_sync_info_buffer[expert_idx].total_received_tokens, num_tokens_to_send);
            }

            // Final memory fence to ensure flag update is visible
            __threadfence_system();
            asm volatile("membar.sys;");

            #ifdef DEEPEP_VERBOSE_DEBUG
            printf("[FLAG SET] rank=%d -> rank=%d, expert=%d, local_expert_idx=%d, blockIdx.x=%d\n",
                   rank, dst_rank, expert_idx, local_expert_idx, blockIdx.x);
            #endif
        }
        // __syncwarp();

        // Mark expert processing as complete
        if (warp_group_id == 0 && sub_warp_id == 0 && lane_id == 0 && expert_sync_info_buffer != nullptr) {
            atomicAdd(&expert_sync_info_buffer[expert_idx].expert_processing_complete, 1);
            __threadfence_system();
        }
    } // End of if (expert_idx < num_experts)

    // Ensure all expert processing is complete before starting combine
    __syncthreads();

    // Receiving phase
    LOW_LATENCY_COMBINE_RECV:
    __syncthreads();

    // Set flag to skip processing but continue to grid sync
    bool skip_recv_phase = ((phases & LOW_LATENCY_RECV_PHASE) == 0);
    if (skip_recv_phase) {
        if (thread_id == 0 && sm_id == 0) {
            printf("[COMBINE] rank=%d will skip combine processing (no RECV_PHASE in phases=0x%x) but participate in sync\n", rank, phases);
        }
    }

    // Each block waits for its assigned expert
    expert_idx = responsible_expert_idx;
    if (expert_idx < num_experts) {
        const auto expert_rank = expert_idx / num_local_experts;

        // Only process experts that belong to this rank
        if (expert_rank == rank) {
            const auto local_expert_idx = expert_idx % num_local_experts;

            // Check if any rank will send data to this expert by checking layout_range
            bool will_receive_data = false;
            if (layout_range != nullptr) {
                for (int src_rank = 0; src_rank < num_ranks; src_rank++) {
                    auto layout = __ldg(layout_range + local_expert_idx * num_ranks + src_rank);
                    int num_tokens_from_rank, offset;
                    unpack2(layout, num_tokens_from_rank, offset);
                    if (num_tokens_from_rank > 0) {
                        will_receive_data = true;
                        break;
                    }
                }
            }

            EP_DEVICE_ASSERT(num_warps_per_group > 1);
            if (sub_warp_id == 0 && lane_id == 0) {
                if (!will_receive_data) {
                    // No data expected for this expert - skip waiting and mark as complete
                    #ifdef DEEPEP_VERBOSE_DEBUG
                    printf("[FLAG SKIP] rank=%d expert=%d has no incoming data, skipping wait\n", rank, expert_idx);
                    #endif
                    // DO NOT return here! Must participate in grid sync below
                } else {
                    // se same address calculation as flag setting
                    // Flag setting uses local_expert_idx, so flag waiting must also use local_expert_idx
                    volatile int* flag_ptr = rdma_recv_flag + local_expert_idx;

                    #ifdef DEEPEP_VERBOSE_DEBUG
                    printf("[FLAG WAIT] rank=%d waiting for expert=%d (local_idx=%d) flag (expecting data)\n",
                           rank, expert_idx, local_expert_idx);
                    #endif

                    // Use ExpertSyncInfo for more robust synchronization
                    if (expert_sync_info_buffer != nullptr) {
                        ExpertSyncInfo* sync_info = &expert_sync_info_buffer[expert_idx];

                        // Wait until all expected tokens are received
                        int timeout_counter = 0;
                        while (sync_info->total_received_tokens < sync_info->total_expected_tokens) {
                            __threadfence_system();
                            if (++timeout_counter > 100000000) {
                                printf("[GPU HANG DETECTED] Rank %d waiting for expert %d (local_idx=%d)\n",
                                       rank, expert_idx, local_expert_idx);
                                printf("  Expected: %d tokens, Received: %d tokens\n",
                                       sync_info->total_expected_tokens, sync_info->total_received_tokens);
                                printf("  Per-rank breakdown:\n");
                                for (int r = 0; r < num_ranks && r < 8; ++r) {
                                    printf("    Rank %d: Expected=%d, Received=%d\n", r,
                                           sync_info->expected_tokens_per_rank[r],
                                           sync_info->received_tokens_per_rank[r]);
                                }
                                timeout_counter = 0;
                            }
                        }
                    } else {
                        // Fallback to old flag-based synchronization
                        int timeout_counter = 0;
                        while (ld_acquire_sys_global((const int*)flag_ptr) == 0) {
                            if (++timeout_counter > 100000000) {
                                printf("[GPU HANG DETECTED] Rank %d waiting for expert %d (local_idx=%d) flag indefinitely\n",
                                       rank, expert_idx, local_expert_idx);
                                timeout_counter = 0;
                            }
                        }
                    }
                    #ifdef DEEPEP_VERBOSE_DEBUG
                    printf("[FLAG WAIT DONE] rank=%d, expert_idx=%d, local_idx=%d, flag_value=%d, blockIdx.x=%d\n",
                           rank, expert_idx, local_expert_idx, ld_acquire_sys_global((const int*)flag_ptr), blockIdx.x);
                    #endif
                }
            }
        }
    }  // End of if (expert_idx < num_experts)

    // Proper grid-wide synchronization
    // All warps must complete their work before grid sync
    __syncthreads();

    // All threads participate in grid sync (collective operation)
    cg::this_grid().sync();

    // Additional system-wide memory fence for RDMA visibility
    // All threads must execute the fence for consistency
    __threadfence_system();

    // Counter for missing remote expert results
    __shared__ int missing_remote_experts;
    if (thread_id == 0) {
        missing_remote_experts = 0;
    }
    __syncthreads();

    // Reduce tokens with FP8 cast

    EP_DEVICE_ASSERT(num_topk <= 32);
    EP_DEVICE_ASSERT(actual_num_int4_per_data <= num_threads);
    EP_STATIC_ASSERT(kHidden % (32 * kNumElemsPerInt4) == 0, "Invalid vectorization");

    // Validate safe_num_topk before use to prevent memory corruption
    bool is_corrupted = (safe_num_topk <= 0 || safe_num_topk > 32);
    if (is_corrupted) {
        if (thread_id == 0 && sm_id == 0) {
            printf("[FATAL] safe_num_topk corrupted: %d (0x%x), original param=%d\n",
                   safe_num_topk, (unsigned int)safe_num_topk, num_topk);
            printf("[FATAL] Common corruption patterns:\n");
            printf("  0x9DC79A28 = freed memory\n");
            printf("  0xF49E0AA8 = uninitialized\n");
            printf("  0xEC01D228 = stack corruption\n");
            printf("[FATAL] Other params: num_experts=%d, num_ranks=%d, num_combined_tokens=%d\n",
                   safe_num_experts, safe_num_ranks, safe_num_combined_tokens);
        }
        // Cannot return early - must participate in grid sync
        // Set safe values to prevent crash
        safe_num_topk = 1;  // Minimum valid value
    }

    // Use actual size and ensure thread is within bounds
    // Process tokens even if skip_recv_phase is true to maintain NVSHMEM collective consistency
    if (thread_id < num_threads) {
        for (int token_idx = sm_id; token_idx < safe_num_combined_tokens; token_idx += num_sms) {
            // token_idx is the global token index we're combining results for

            // Ensure token_idx is within bounds
            if (token_idx >= safe_num_combined_tokens) {
                break;
            }

            // Additional safety check
            if (token_idx < 0) {
                if (thread_id == 0) {
                    printf("[ERROR] Invalid token_idx=%d, sm_id=%d, num_sms=%d\n", token_idx, sm_id, num_sms);
                }
                break;
            }

            // Read top-k indices and weights
            int reg_topk_idx[kNumMaxTopk];
            float reg_topk_weights[kNumMaxTopk];

            // Initialize arrays to safe values
            #pragma unroll
            for (int i = 0; i < kNumMaxTopk; ++i) {
                reg_topk_idx[i] = -1;
                reg_topk_weights[i] = 0.0f;
            }

            // Bounds check before reading topk arrays
            const size_t topk_offset = token_idx * safe_num_topk;
            if (topk_idx == nullptr || topk_weights == nullptr) {
                if (thread_id == 0) {
                    printf("[ERROR] Null pointer: topk_idx=%p, topk_weights=%p\n", topk_idx, topk_weights);
                }
            } else {
                #pragma unroll
                for (int i = 0; i < safe_num_topk; ++ i) {
                    reg_topk_idx[i] = static_cast<int>(__ldg(topk_idx + topk_offset + i));
                    reg_topk_weights[i] = __ldg(topk_weights + topk_offset + i);
                }
            }

            // Stride over int4 chunks to cover full hidden size even when blockDim.x < actual_num_int4_per_data
            for (int int4_idx = thread_id; int4_idx < actual_num_int4_per_data; int4_idx += num_threads) {
                float combined_values[kNumElemsPerInt4] = {0.0f};

                // Accumulate contributions from local experts only
                for (int i = 0; i < safe_num_topk; ++i) {
                    const int src_expert_idx_local = reg_topk_idx[i];
                    if (src_expert_idx_local < 0) continue;

                    const int src_rank_local = src_expert_idx_local / safe_num_local_experts;
                    const int src_local_expert_idx = src_expert_idx_local % safe_num_local_experts;

                    // Only local experts contribute here; remote handled by reduction
                    if (src_rank_local == rank && layout_range != nullptr) {
                        // Map [expert, rank, slot] -> original token id
                        const int* local_src_info =
                            src_info + src_local_expert_idx * safe_num_ranks * safe_num_max_dispatch_tokens_per_rank;
                        for (int from_rank = 0; from_rank < num_ranks; ++from_rank) {
                            auto layout = __ldg(layout_range + src_local_expert_idx * num_ranks + from_rank);
                            int num_tokens_from_rank, offset_in_packed;
                            unpack2(layout, num_tokens_from_rank, offset_in_packed);
                            for (int slot = 0; slot < num_tokens_from_rank; ++slot) {
                                const int packed_slot_idx = offset_in_packed + slot;
                                if (packed_slot_idx >= safe_num_max_dispatch_tokens_per_rank) continue;

                                // Check ownership: does this slot belong to current token_idx?
                                const int src_token_id = __ldg(
                                    local_src_info + from_rank * safe_num_max_dispatch_tokens_per_rank + packed_slot_idx);
                                if (src_token_id != token_idx) continue;

                                // Only now load and accumulate
                                const int4* packed_x_ptr = static_cast<const int4*>(x)
                                    + src_local_expert_idx * safe_num_ranks * safe_num_max_dispatch_tokens_per_rank * actual_num_int4_per_data
                                    + (from_rank * safe_num_max_dispatch_tokens_per_rank + packed_slot_idx) * actual_num_int4_per_data;

                                const size_t buffer_offset = packed_x_ptr + int4_idx - static_cast<const int4*>(x);
                                const size_t max_buffer_size = static_cast<size_t>(safe_num_local_experts) * safe_num_ranks * safe_num_max_dispatch_tokens_per_rank * actual_num_int4_per_data;
                                int4 x_vec = (buffer_offset >= max_buffer_size) ? make_int4(0,0,0,0) : ld_nc_global(packed_x_ptr + int4_idx);
                                const nv_bfloat16* x_bf16 = reinterpret_cast<const nv_bfloat16*>(&x_vec);
                                #pragma unroll
                                for (int j = 0; j < kNumElemsPerInt4; ++j) {
                                    combined_values[j] += static_cast<float>(x_bf16[j]) * reg_topk_weights[i];
                                }
                                // Pure EP: per expert per rank at most one slot matches → stop scanning slots for this rank
                                break;
                            }
                        }
                    }
                }

                // Write out this int4 chunk
                if (!skip_recv_phase) {
                    if (token_idx < safe_num_combined_tokens) {
                        if (is_pure_ep_mode) {
                            // Pure EP: keep FP32 values intact; do not perform BF16 in-place conversion
                            const int elements_per_token = actual_num_int4_per_data * kNumElemsPerInt4;
                            float* token_fp32_ptr = reinterpret_cast<float*>(fp32_workspace) + token_idx * elements_per_token;
                            #pragma unroll
                            for (int j = 0; j < kNumElemsPerInt4; ++j) {
                                token_fp32_ptr[int4_idx * kNumElemsPerInt4 + j] = combined_values[j];
                            }
                        } else {
                            // Non-Pure EP: pack 8 bf16 into one int4 and write to output buffer
                            int4 packed_bf16;
                            nv_bfloat16* packed_ptr = reinterpret_cast<nv_bfloat16*>(&packed_bf16);
                            #pragma unroll
                            for (int j = 0; j < kNumElemsPerInt4; ++j) {
                                packed_ptr[j] = static_cast<nv_bfloat16>(combined_values[j]);
                            }
                            int4* combined_x_ptr = static_cast<int4*>(combined_x) + token_idx * actual_num_int4_per_data + int4_idx;
                            *combined_x_ptr = packed_bf16;
                        }
                    } else {
                        if (thread_id == 0) {
                            printf("[ERROR] Write out of bounds! token_idx=%d, int4_idx=%d\n", token_idx, int4_idx);
                        }
                    }
                } else if (is_pure_ep_mode) {
                    // Do not zero-out FP32 workspace on skip_recv_phase; keep previous partials for NVSHMEM reduction
                    // No-op to avoid tail sections being overwritten to zero by non-owner ranks
                }
            } // end int4 stride loop
        }
    }

    // Perform NVSHMEM reduction for Pure EP mode
    if (is_pure_ep_mode) {
        // Grid-wide sync before reduction to ensure all ranks have written their partial results
        cg::this_grid().sync();

        // If this PE skipped recv phase, its FP32 workspace may contain stale data from previous steps.
        // Zero the entire FP32 workspace so that this PE contributes pure zeros to the reduction.
        if (blockIdx.x == 0 && skip_recv_phase) {
            const int elements_per_token_zero = actual_num_int4_per_data * kNumElemsPerInt4;
            const int max_tokens_zero = safe_num_combined_tokens;
            for (int token_idx_zero = 0; token_idx_zero < max_tokens_zero; ++token_idx_zero) {
                float* p = reinterpret_cast<float*>(fp32_workspace) + token_idx_zero * elements_per_token_zero;
                for (int e = thread_id; e < elements_per_token_zero; e += blockDim.x) {
                    p[e] = 0.0f;
                }
                __syncthreads();
            }
        }

        // Ensure all blocks see a clean FP32 workspace state before entering collective reduction
        cg::this_grid().sync();

        // Use the agreed, safe number of tokens to reduce across PEs
        // Must exactly match the number of tokens combined in this pass
        EP_DEVICE_ASSERT(fp32_workspace != nullptr);
        const int max_combined_tokens = safe_num_combined_tokens;  // equals num_combined_tokens

        // Only block 0 performs NVSHMEM reduction
        // All PEs must have exactly one block 0 that calls the collective
        if (blockIdx.x == 0) {
            const int elements_per_token = actual_num_int4_per_data * kNumElemsPerInt4;
            EP_DEVICE_ASSERT(elements_per_token > 0 && max_combined_tokens > 0);

            // Zero-fill FP32 workspace for tokens beyond local num_combined_tokens
            // to avoid stale data contributing to NVSHMEM reduction.
            // Each PE may have different num_combined_tokens; the reduction loops up to
            // max_combined_tokens for collective consistency, so invalid local tokens
            // must be explicitly set to 0.
            {
                // All threads in block 0 participate to parallelize zero-fill
                for (int token_idx = num_combined_tokens; token_idx < max_combined_tokens; ++token_idx) {
                    float* token_fp32_ptr = reinterpret_cast<float*>(fp32_workspace) + token_idx * elements_per_token;
                    for (int elem = thread_id; elem < elements_per_token; elem += blockDim.x) {
                        token_fp32_ptr[elem] = 0.0f;
                    }
                    __syncthreads();
                }
                // Removed grid-wide sync here; global syncs occur at 1807, 1887, 1915
            }

            // Process ALL tokens up to max_combined_tokens to ensure consistent participation
            for (int token_idx = 0; token_idx < max_combined_tokens; token_idx++) {
                float* token_fp32_ptr = reinterpret_cast<float*>(fp32_workspace) + token_idx * elements_per_token;

                // Check if this is a valid token for this PE
                bool is_valid_token = (token_idx < num_combined_tokens);

                // Log pre-reduction values for valid tokens only
                if (thread_id == 0 && token_idx < 2 && is_valid_token) {
                    printf("[NVSHMEM_REDUCE] rank=%d, block=0, token=%d, pre-reduce: %.6f, %.6f, %.6f\n",
                           rank, token_idx, token_fp32_ptr[0], token_fp32_ptr[1], token_fp32_ptr[2]);
                }

                // Process in chunks to avoid memory issues
                const int chunk_size = 256;  // Process 256 floats at a time

                // Calculate number of chunks (same for all PEs)
                const int num_chunks = (elements_per_token + chunk_size - 1) / chunk_size;

                for (int chunk_idx = 0; chunk_idx < num_chunks; chunk_idx++) {
                    const int chunk_start = chunk_idx * chunk_size;
                    const int chunk_end = min(chunk_start + chunk_size, elements_per_token);
                    const int chunk_elems = chunk_end - chunk_start;

                    // For invalid tokens, use dummy data (zeros already in workspace)
                    // This ensures all PEs call the collective the same number of times

                    // All threads in block 0 must participate in the collective
                    // Use block-level NVSHMEM reduction
                    nvshmemx_float_sum_reduce_block(NVSHMEM_TEAM_WORLD,
                                                  token_fp32_ptr + chunk_start,  // destination
                                                  token_fp32_ptr + chunk_start,  // source
                                                  chunk_elems);                  // count

                    // Sync after each chunk to ensure completion
                    __syncthreads();
                }

                // Log post-reduction values for valid tokens only
                if (thread_id == 0 && token_idx < 2 && is_valid_token) {
                    printf("[NVSHMEM_REDUCE] rank=%d, block=0, token=%d, post-reduce: %.6f, %.6f, %.6f\n",
                           rank, token_idx, token_fp32_ptr[0], token_fp32_ptr[1], token_fp32_ptr[2]);
                }
            }
        }

        // All blocks sync here to ensure block 0 has completed reduction
        cg::this_grid().sync();

        // Now ALL blocks participate in converting FP32 back to BF16
        // Only convert valid tokens (up to num_combined_tokens)
        const int elements_per_token = actual_num_int4_per_data * kNumElemsPerInt4;
        const int tokens_per_block = (num_combined_tokens + gridDim.x - 1) / gridDim.x;
        const int start_token = blockIdx.x * tokens_per_block;
        const int end_token = min(start_token + tokens_per_block, num_combined_tokens);

        // Each block converts its assigned tokens from FP32 to BF16
        for (int token_idx = start_token; token_idx < end_token; token_idx++) {
            float* token_fp32_ptr = reinterpret_cast<float*>(fp32_workspace) + token_idx * elements_per_token;

            // Each thread handles its portion
            const int elems_per_thread = (elements_per_token + blockDim.x - 1) / blockDim.x;
            const int thread_start = thread_id * elems_per_thread;
            const int thread_end = min(thread_start + elems_per_thread, elements_per_token);

            for (int elem_idx = thread_start; elem_idx < thread_end; elem_idx++) {
                nv_bfloat16* bf16_ptr = reinterpret_cast<nv_bfloat16*>(combined_x) +
                                       token_idx * elements_per_token + elem_idx;
                *bf16_ptr = static_cast<nv_bfloat16>(token_fp32_ptr[elem_idx]);
            }

            __syncthreads();
        }

        // Final grid sync to ensure all conversions complete
        cg::this_grid().sync();
    }

    // Reset flags for next iteration using atomic decrement
    // This handles the race condition where multiple ranks may have incremented the flag
    // Each block handles its assigned expert
    if (thread_id == 0 && responsible_expert_idx < num_experts) {
        const auto expert_idx = responsible_expert_idx;
        const auto expert_rank = expert_idx / num_local_experts;
        if (expert_rank == rank) {
            // This expert belongs to this rank, handle it
            const auto local_expert_idx = expert_idx % num_local_experts;

            // Additional bounds check
            if (local_expert_idx < num_local_experts) {
                // Check if we actually waited for this expert (i.e., it received data)
                bool did_wait_for_expert = false;
                if (layout_range != nullptr) {
                    for (int src_rank = 0; src_rank < num_ranks; src_rank++) {
                        auto layout = __ldg(layout_range + local_expert_idx * num_ranks + src_rank);
                        int num_tokens_from_rank, offset;
                        unpack2(layout, num_tokens_from_rank, offset);
                        if (num_tokens_from_rank > 0) {
                            did_wait_for_expert = true;
                            break;
                        }
                    }
                }

                if (did_wait_for_expert) {
                    // More careful flag handling to avoid race conditions
                    // First ensure all reads are complete
                    __threadfence_system();
                    asm volatile("membar.sys;");

                    // Count actual tokens received from all ranks
                    int total_received = 0;
                    for (int src_rank = 0; src_rank < num_ranks; src_rank++) {
                        auto layout = __ldg(layout_range + local_expert_idx * num_ranks + src_rank);
                        int num_tokens_from_rank, offset;
                        unpack2(layout, num_tokens_from_rank, offset);
                        total_received += num_tokens_from_rank;
                    }

                    // Only decrement by the actual number of senders
                    // This prevents over-decrementing if multiple ranks sent data
                    int expected_decrements = 0;
                    for (int src_rank = 0; src_rank < num_ranks; src_rank++) {
                        if (src_rank != rank) {  // Only remote ranks set flags
                            auto layout = __ldg(layout_range + local_expert_idx * num_ranks + src_rank);
                            int num_tokens_from_rank, offset;
                            unpack2(layout, num_tokens_from_rank, offset);
                            if (num_tokens_from_rank > 0) {
                                expected_decrements++;
                            }
                        }
                    }

                    // Safe atomic update
                    if (expected_decrements > 0) {
                        atomicSub(&rdma_recv_flag[local_expert_idx], expected_decrements);
                        __threadfence_system();

                        #ifdef DEEPEP_VERBOSE_DEBUG
                        int final_value = rdma_recv_flag[local_expert_idx];
                        printf("[FLAG DECREMENT] rank=%d decremented flag for local_expert=%d by %d, new value=%d\n",
                               rank, local_expert_idx, expected_decrements, final_value);
                        #endif
                    }
                }
            } else {
                printf("[ERROR] Invalid local_expert_idx=%d, expert_idx=%d, num_local_experts=%d\n",
                       local_expert_idx, expert_idx, num_local_experts);
            }
        } // End if (expert_rank == rank)
    }

    // Report missing remote experts at the end
    __syncthreads();
    if (thread_id == 0 && missing_remote_experts > 0) {
        printf("[CRITICAL] rank=%d: %d remote expert results are missing!\n",
               rank, missing_remote_experts);
        printf("  This causes incorrect output. NVSHMEM+NCCL conflict prevents AllReduce.\n");
        printf("  Need to implement NVSHMEM-based aggregation in the kernel.\n");
    }
}

void combine(void* combined_x,
             void* fp32_workspace,  // Separate FP32 workspace for NVSHMEM reduction
             void* rdma_recv_x, int* rdma_recv_flag, void* rdma_send_x,
             ExpertSyncInfo* expert_sync_info_buffer,  // Expert synchronization info
             const void* x, const int64_t* topk_idx, const float* topk_weights,
             const int* src_info, const int64_t* layout_range,
             int64_t* combine_wait_recv_cost_stats,
             int* next_clean, int num_next_clean_int,
             int num_combined_tokens, int hidden, int num_max_dispatch_tokens_per_rank,
             int num_topk, int num_experts, int rank, int num_ranks,
             bool use_logfmt,
             void* workspace, int num_device_sms,
             cudaStream_t stream, int phases, bool zero_copy) {
    // Add detailed parameter validation
    EP_HOST_ASSERT(combined_x != nullptr);
    EP_HOST_ASSERT(x != nullptr);
    EP_HOST_ASSERT(topk_idx != nullptr);
    EP_HOST_ASSERT(topk_weights != nullptr);
    EP_HOST_ASSERT(num_combined_tokens > 0);
    EP_HOST_ASSERT(hidden > 0 && hidden % 128 == 0);
    EP_HOST_ASSERT(num_experts > 0 && num_experts % num_ranks == 0);
    EP_HOST_ASSERT(num_ranks > 0);
    EP_HOST_ASSERT(num_topk > 0 && num_topk <= 9);

    // Set grid size to num_experts to ensure all experts are processed
    // Each block will handle one expert
    const int grid_size = num_experts;

    // Calculate warps based on device SM count, not experts
    const int num_warp_groups = ceil_div(num_experts, num_device_sms);
    const int num_warps_per_group = min(32 / num_warp_groups, 32);
    EP_HOST_ASSERT(num_warp_groups > 0 and num_warps_per_group > 0);

    const auto num_warps = min(num_warp_groups * num_warps_per_group, 32);

    // Check workspace
    auto atomic_clean_flag = static_cast<int*>(workspace);
    EP_HOST_ASSERT(sizeof(int) <= NUM_WORKSPACE_BYTES);
    EP_HOST_ASSERT(num_topk <= 9);

#define COMBINE_LAUNCH_CASE(hidden) { \
auto combine_func = combine<hidden, 9>; \
LAUNCH_KERNEL(&cfg, combine_func, \
              combined_x, \
              fp32_workspace, \
              rdma_recv_x, rdma_recv_flag, rdma_send_x, \
              x, topk_idx, topk_weights, src_info, layout_range, \
              combine_wait_recv_cost_stats, \
              next_clean, num_next_clean_int, \
              atomic_clean_flag, \
              expert_sync_info_buffer, \
              num_combined_tokens, hidden, num_topk, \
              num_max_dispatch_tokens_per_rank, \
              num_experts, rank, num_ranks, \
              num_warp_groups, num_warps_per_group, \
              phases, zero_copy); } break

    // Setup launch configuration
    // Ensure threads per block cover all int4 chunks of hidden (hidden/8 for bf16)
    const int required_threads_int4 = hidden / 8;  // bf16: 8 elems per int4
    const int threads_per_block = (num_warps * 32 > required_threads_int4) ? (num_warps * 32) : required_threads_int4;
    SETUP_LAUNCH_CONFIG(grid_size, threads_per_block, stream);  // Use grid_size instead of num_experts
    SWITCH_HIDDEN(COMBINE_LAUNCH_CASE);
#undef COMBINE_LAUNCH_CASE
}

} // namespace internode_ll

} // namespace deep_ep
