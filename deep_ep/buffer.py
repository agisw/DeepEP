import os
import pickle
import uuid
import ctypes

import torch
import torch.distributed as dist
from typing import Callable, List, Tuple, Optional, Union

# noinspection PyUnresolvedReferences
import deep_ep_cpp
# noinspection PyUnresolvedReferences
from deep_ep_cpp import Config, EventHandle
from .utils import EventOverlap, check_nvlink_connections


class Buffer:
    """
    The core expert-parallel (EP) communication buffers for Mixture of Experts (MoE) model, which supports:
        - high-throughput intranode all-to-all (dispatch and combine, using NVLink)
        - high-throughput internode all-to-all (dispatch and combine, using RDMA and NVLink)
        - low-latency all-to-all (dispatch and combine, using RDMA)

    Attributes:
        num_sms: the SMs used in high-throughput kernels.
        rank: the local rank number.
        group_size: the number of ranks in the group.
        group: the communication group.
        num_nvl_bytes: the buffer size for intranode NVLink communication.
        num_rdma_bytes: the buffer size for internode (also for intranode with low-latency mode) RDMA communication.
        runtime: the C++ runtime.
    """

    num_sms: int = 20

    def __init__(self, group: dist.ProcessGroup,
                 num_nvl_bytes: int = 0, num_rdma_bytes: int = 0,
                 low_latency_mode: bool = False, num_qps_per_rank: int = 24,
                 allow_nvlink_for_low_latency_mode: bool = True,
                 allow_mnnvl: bool = False,
                 explicitly_destroy: bool = False) -> None:
        """
        Initialize the communication buffer.

        Arguments:
            group: the communication group.
            num_nvl_bytes: the buffer size for intranode NVLink communication.
            num_rdma_bytes: the buffer size for internode (also for intranode with low-latency mode) RDMA communication.
            low_latency_mode: whether to enable low-latency mode.
            num_qps_per_rank: the number of QPs for RDMA, the low-latency mode requires that this number equals
                to the number of local experts.
            allow_nvlink_for_low_latency_mode: whether allow NVLink traffic for low-latency mode, you should notice
                this is somehow incompatible with the hook-based overlapping.
                Warning: PCIe connections may lead to errors due to memory ordering issues,
                please make sure all connections are via NVLink.
            allow_mnnvl: whether to allow MNNVL
            explicitly_destroy: If this flag is set to True, you need to explicitly call `destroy()` to release resources;
                otherwise, the resources will be released by the destructor.
                Note: Releasing resources in the destructor may cause Python's exception handling process to hang.
        """
        check_nvlink_connections(group)

        # Initialize the CPP runtime
        self.rank = group.rank()
        self.group_size = group.size()
        self.group = group
        self.num_nvl_bytes = num_nvl_bytes
        self.num_rdma_bytes = num_rdma_bytes
        self.low_latency_mode = low_latency_mode
        self.runtime = deep_ep_cpp.Buffer(self.rank, self.group_size, num_nvl_bytes, num_rdma_bytes, low_latency_mode, explicitly_destroy)

        # --- Redesigned Initialization Flow ---

        # Generate local information at the Python/PyTorch level.
        local_device_id = torch.cuda.current_device()

        local_ipc_handle_bytes = b"\0" * 64
        if not self.low_latency_mode and num_nvl_bytes > 0:
            local_ipc_handle_bytes = self.runtime.get_local_ipc_handle()

        # Gather all device IDs and IPC handles from all ranks using a robust tensor-based collective.
        # This avoids using pickle-based `_object` calls which can be unstable in complex
        # multi-node environments with dynamic subgroups.
        # We pack device_id (as 1 byte) and ipc_handle (64 bytes) into one tensor.

        local_info_list = [local_device_id] + list(local_ipc_handle_bytes)
        local_info_tensor = torch.tensor(
            local_info_list,
            dtype=torch.uint8,
            device=f"cuda:{local_device_id}",
        )

        expected_size = 65  # 1 for device_id, 64 for ipc_handle
        if local_info_tensor.size(0) != expected_size:
            raise RuntimeError(f"Local info tensor size mismatch: expected {expected_size}, got {local_info_tensor.size(0)}")

        gathered_info_tensor = torch.empty(
            self.group_size,
            expected_size,
            dtype=torch.uint8,
            device=f"cuda:{local_device_id}",
        )

        try:
            dist.all_gather_into_tensor(
                gathered_info_tensor, local_info_tensor, group=self.group
            )
        except Exception as e:
            if self.rank == 0:
                print(f"[DeepEP Buffer] all_gather_into_tensor failed: {e}")
            # 재시도
            import time
            time.sleep(2)
            try:
                dist.all_gather_into_tensor(
                    gathered_info_tensor, local_info_tensor, group=self.group
                )
                if self.rank == 0:
                    print(f"[DeepEP Buffer] all_gather_into_tensor succeeded on retry")
            except Exception as retry_e:
                if self.rank == 0:
                    print(f"[DeepEP Buffer] all_gather_into_tensor retry also failed: {retry_e}")
                raise RuntimeError(f"Failed to gather device information: {retry_e}")

        if self.rank == 0:
            print(f"[DeepEP Buffer] all_gather_into_tensor completed successfully")

        gathered_info = gathered_info_tensor.cpu().numpy()
        device_ids = [int(row[0]) for row in gathered_info]

        ipc_handles = []
        for i in range(self.group_size):
            if not self.low_latency_mode and num_nvl_bytes > 0:
                handle_bytes = gathered_info[i, 1:].tobytes()
                ipc_handles.append(bytearray(handle_bytes))
                # Check IPC handle validity
                print(f"[DeepEP Buffer] Rank {self.rank} received IPC handle from rank {i}: {handle_bytes[:16].hex()}")
            else:
                ipc_handles.append(None)

        # Check IPC handle creation and access
        if not self.low_latency_mode and num_nvl_bytes > 0:
            print(f"[DeepEP Buffer] Rank {self.rank} checking IPC handle access...")
            try:
                # Try to access each IPC handle
                for i, handle in enumerate(ipc_handles):
                    if handle is not None and i != self.rank:
                        try:
                            # Convert handle to CUDA IPC handle format
                            handle_array = (ctypes.c_ubyte * 64).from_buffer(handle)
                            print(f"[DeepEP Buffer] Rank {self.rank} IPC handle {i} format: OK")
                        except Exception as ipc_e:
                            print(f"[DeepEP Buffer] Rank {self.rank} IPC handle {i} format error: {ipc_e}")
            except Exception as e:
                print(f"[DeepEP Buffer] Rank {self.rank} IPC handle check failed: {e}")
        else:
            print(f"[DeepEP Buffer] Rank {self.rank} skipping IPC handle check (low latency mode)")

        # The root rank generates and distributes the NVSHMEM unique ID.
        root_unique_id = None
        if num_rdma_bytes > 0:
            # IBGDA and NVSHMEM settings are now handled entirely by launch script
            # No need to set environment variables here as they should be pre-configured
            if self.runtime.get_num_rdma_ranks() > 1 or low_latency_mode:
                assert num_qps_per_rank > 0
                if self.rank == 0:
                    print("NVSHMEM settings inherited from launch script")

            # Set NVSHMEM bootstrap session ID for multi-node communication
            # Use the current master node IP for bootstrap coordination
            # This is the standard and robust way to ensure all ranks use the same address.
            # Set default values only if not already configured by launch script

            # Multiple barriers to ensure all ranks are synchronized
            # This fixes the collective sequence number mismatch between ranks
            import struct
            import time

            if self.rank == 0:
                print(f"[DeepEP Buffer] Starting NVSHMEM unique ID synchronization process")

            print(f"[DeepEP Buffer] Rank {self.rank} starting NVSHMEM initialization (test-proven approach)")
            time.sleep(0.1)  # 최소 지연

            # Determine the correct rank to generate unique ID
            # In low latency mode, all ranks participate in NVSHMEM, but only rdma_rank=0 can generate unique ID
            rdma_rank = self.rank // 8  # NUM_MAX_NVL_PEERS = 8
            can_generate_unique_id = (rdma_rank == 0)

            # Find the rank that can generate unique ID (rdma_rank=0)
            root_rdma_rank = 0  # This is the global rank with rdma_rank=0

            # Print rank mapping information
            print(f"[DeepEP Buffer] Rank {self.rank} mapping - RDMA rank: {rdma_rank}, Can generate: {can_generate_unique_id}")

            # Check NVSHMEM PE initialization status
            print(f"[DeepEP Buffer] Rank {self.rank} checking NVSHMEM PE status...")
            try:
                # Check if NVSHMEM is initialized for this rank
                nvshmem_rank = self.rank if low_latency_mode else rdma_rank
                num_nvshmem_ranks = self.group_size if low_latency_mode else self.runtime.get_num_rdma_ranks()
                print(f"[DeepEP Buffer] Rank {self.rank} NVSHMEM mapping: PE {nvshmem_rank}/{num_nvshmem_ranks}")

                # Check if runtime has RDMA buffer allocated
                if hasattr(self.runtime, 'get_rdma_buffer_ptr'):
                    rdma_ptr = self.runtime.get_rdma_buffer_ptr()
                    print(f"[DeepEP Buffer] Rank {self.rank} RDMA buffer pointer: {rdma_ptr}")
                else:
                    print(f"[DeepEP Buffer] Rank {self.rank} RDMA buffer pointer: not available")

                # Check current CUDA device and context
                current_device = torch.cuda.current_device()
                print(f"[DeepEP Buffer] Rank {self.rank} current CUDA device: {current_device}")

                # Check GPU memory status
                free_mem, total_mem = torch.cuda.mem_get_info()
                print(f"[DeepEP Buffer] Rank {self.rank} GPU memory: {free_mem//1024//1024}MB free / {total_mem//1024//1024}MB total")

            except Exception as e:
                print(f"[DeepEP Buffer] Rank {self.rank} NVSHMEM PE status check failed: {e}")

            # Check CUDA P2P access capabilities
            print(f"[DeepEP Buffer] Rank {self.rank} checking CUDA P2P access...")
            try:
                current_device = torch.cuda.current_device()
                device_count = torch.cuda.device_count()
                print(f"[DeepEP Buffer] Rank {self.rank} total GPU devices: {device_count}")

                # Check P2P access between all GPU pairs on this node
                for i in range(device_count):
                    for j in range(device_count):
                        if i != j:
                            try:
                                can_access = torch.cuda.can_device_access_peer(i, j)
                                print(f"[DeepEP Buffer] Rank {self.rank} P2P access GPU {i} → GPU {j}: {can_access}")
                            except Exception as p2p_e:
                                print(f"[DeepEP Buffer] Rank {self.rank} P2P check GPU {i} → GPU {j} failed: {p2p_e}")

                # Check if P2P is enabled for current device
                if device_count > 1:
                    try:
                        # Try to enable P2P access from current device to all others
                        for other_device in range(device_count):
                            if other_device != current_device:
                                if torch.cuda.can_device_access_peer(current_device, other_device):
                                    # Check if P2P is already enabled
                                    with torch.cuda.device(current_device):
                                        try:
                                            # This will fail if P2P is not enabled
                                            torch.cuda.set_device(current_device)
                                            print(f"[DeepEP Buffer] Rank {self.rank} P2P from GPU {current_device} to GPU {other_device}: available")
                                        except Exception as enable_e:
                                            print(f"[DeepEP Buffer] Rank {self.rank} P2P enable check failed: {enable_e}")
                                else:
                                    print(f"[DeepEP Buffer] Rank {self.rank} P2P from GPU {current_device} to GPU {other_device}: NOT available")
                    except Exception as p2p_enable_e:
                        print(f"[DeepEP Buffer] Rank {self.rank} P2P enable check failed: {p2p_enable_e}")

            except Exception as e:
                print(f"[DeepEP Buffer] Rank {self.rank} CUDA P2P check failed: {e}")

            # Check NVSHMEM environment variables
            print(f"[DeepEP Buffer] Rank {self.rank} checking NVSHMEM environment variables...")
            nvshmem_env_vars = [
                'NVSHMEM_DISABLE_P2P',
                'NVSHMEM_DISABLE_CUDA_VMM',
                'NVSHMEM_IB_ENABLE_IBGDA',
                'NVSHMEM_IBGDA_NUM_DCI',
                'NVSHMEM_IBGDA_NUM_DCT',
                'NVSHMEM_IBGDA_DCI_MAP_BY',
                'NVSHMEM_REMOTE_TRANSPORT',
                'NVSHMEM_DISABLE_NVLS',
                'NVSHMEM_BOOTSTRAP',
                'NVSHMEM_BOOTSTRAP_UID_SESSION_ID',
                'NVSHMEM_BOOTSTRAP_UID_SOCK_FAMILY',
                'NVSHMEM_ENABLE_NIC_PE_MAPPING',
                'NVSHMEM_HCA_LIST',
                'NVSHMEM_DEBUG',
                'NVSHMEM_INFO'
            ]

            print(f"[DeepEP Buffer] Rank {self.rank} NVSHMEM Environment Variables:")
            for var in nvshmem_env_vars:
                value = os.getenv(var, 'NOT_SET')
                print(f"  {var} = {value}")

            # Check communication hierarchy and network access
            print(f"[DeepEP Buffer] Rank {self.rank} communication hierarchy check:")
            print(f"  - Global rank: {self.rank}")
            print(f"  - RDMA rank: {rdma_rank}")
            print(f"  - Can generate unique ID: {can_generate_unique_id}")

            # Check process group properties
            try:
                pg_rank = self.group.rank()
                pg_size = self.group.size()
                print(f"  - Process group rank: {pg_rank}")
                print(f"  - Process group size: {pg_size}")
                print(f"  - Process group backend: {dist.get_backend(self.group)}")
            except Exception as e:
                print(f"  - Process group check failed: {e}")

            # Test all_gather with simple data before unique ID
            print(f"[DeepEP Buffer] Rank {self.rank} testing all_gather with simple data...")
            test_data = torch.full((128,), self.rank + 10, dtype=torch.uint8, device=f"cuda:{local_device_id}")
            test_gathered = torch.empty(self.group_size, 128, dtype=torch.uint8, device=f"cuda:{local_device_id}")

            dist.all_gather_into_tensor(test_gathered, test_data, group=self.group)

            print(f"[DeepEP Buffer] Rank {self.rank} test all_gather results:")
            for i in range(self.group_size):
                first_val = test_gathered[i][0].item()
                expected = i + 10
                print(f"  - [From Rank {self.rank}] Rank {i} data: first_val={first_val}, expected={expected}, match={first_val == expected}")

            # Use a robust tensor-based broadcast for the unique ID, avoiding pickle.
            uid_tensor = torch.empty(
                128, dtype=torch.uint8, device=f"cuda:{local_device_id}"
            )

            # Initialize tensor with zeros to ensure clean state
            uid_tensor.zero_()

            if self.rank == root_rdma_rank:
                print(f"[DeepEP Buffer] Rank {self.rank} generating NVSHMEM unique ID")

                if can_generate_unique_id:
                    # The root rank generates the proper NVSHMEM unique ID using the NVSHMEM API
                    # This will create a properly formatted bootstrap_uid_handle structure
                    uid_bytes = self.runtime.get_local_nvshmem_unique_id()

                    # 🔧 DEBUG: Check the generated unique ID content
                    print(f"[DeepEP Buffer] Generated unique ID length: {len(uid_bytes)}")
                    print(f"[DeepEP Buffer] Unique ID first 32 bytes: {uid_bytes[:32].hex()}")

                    # Check if the bootstrap address is properly set
                    if len(uid_bytes) >= 128:
                        # Extract version and first few bytes of internal data
                        version = struct.unpack('i', uid_bytes[:4])[0]
                        internal_start = uid_bytes[4:36].hex()
                        print(f"[DeepEP Buffer] Unique ID version: {version}")
                        print(f"[DeepEP Buffer] Internal data start: {internal_start}")

                        # Check if it looks like a valid bootstrap address
                        if version > 0 and any(b != 0 for b in uid_bytes[4:36]):
                            print(f"[DeepEP Buffer] ✅ Unique ID appears valid (non-zero content)")
                        else:
                            print(f"[DeepEP Buffer] ❌ Unique ID appears invalid (zero content)")

                    uid_tensor = torch.empty(128, dtype=torch.uint8, device=f"cuda:{local_device_id}")
                    uid_tensor[:len(uid_bytes)] = torch.tensor(list(uid_bytes), dtype=torch.uint8)
                    if len(uid_bytes) < 128:
                        uid_tensor[len(uid_bytes):] = 0  # Pad with zeros

                    print(f"[DeepEP Buffer] Root rank generated NVSHMEM unique ID, tensor shape: {uid_tensor.shape}")
                else:
                    print(f"[DeepEP Buffer] ❌ ERROR: Rank {self.rank} cannot generate unique ID (rdma_rank={rdma_rank})")
                    raise RuntimeError(f"Rank {self.rank} cannot generate NVSHMEM unique ID")
            else:
                print(f"[DeepEP Buffer] Rank {self.rank} waiting for unique ID from rank {root_rdma_rank}")

            print(f"[DeepEP Buffer] Rank {self.rank} starting all_gather (test-proven approach)")

            # Print broadcast information
            print(f"[DeepEP Buffer] Rank {self.rank} broadcasting NVSHMEM unique ID from rank {root_rdma_rank}")
            print(f"[DeepEP Buffer] Rank {self.rank} using process group: {self.group}")

            # Use synchronous broadcast with explicit device sync
            try:
                # Use all_gather instead of broadcast
                print(f"[DeepEP Buffer] Rank {self.rank} using all_gather instead of broadcast")

                # Create gathered tensor to collect unique IDs from all ranks
                gathered_uid_tensor = torch.empty(
                    self.group_size, 128, dtype=torch.uint8, device=f"cuda:{local_device_id}"
                )

                # Use all_gather_into_tensor instead of broadcast
                dist.all_gather_into_tensor(
                    gathered_uid_tensor, uid_tensor, group=self.group
                )

                # Check all gathered data before extraction
                print(f"[DeepEP Buffer] Rank {self.rank} all_gather result analysis:")
                print(f"  - Root rank index: {root_rdma_rank}")
                print(f"  - Gathered tensor shape: {gathered_uid_tensor.shape}")
                for i in range(self.group_size):
                    rank_data = gathered_uid_tensor[i]
                    non_zero_count = torch.count_nonzero(rank_data).item()
                    tensor_sum = rank_data.sum().item()
                    first_bytes = rank_data[:8].cpu().tolist()
                    print(f"  - [From Rank {self.rank}] Rank {i} data: non_zero={non_zero_count}, sum={tensor_sum}, first_8={first_bytes}")

                # Extract the unique ID from the root rank
                uid_tensor = gathered_uid_tensor[root_rdma_rank]

                print(f"[DeepEP Buffer] Rank {self.rank} all_gather completed")

                # Check received unique ID content for all ranks
                print(f"[DeepEP Buffer] Rank {self.rank} received unique ID check:")
                print(f"  - First 16 bytes: {uid_tensor[:16].cpu().tolist()}")
                print(f"  - All zeros?: {torch.all(uid_tensor == 0).item()}")
                print(f"  - Non-zero count: {torch.count_nonzero(uid_tensor).item()}")
                print(f"  - Tensor sum: {uid_tensor.sum().item()}")
                if torch.count_nonzero(uid_tensor).item() > 0:
                    print(f"  - ✅ Rank {self.rank} received valid unique ID")
                else:
                    print(f"  - ❌ Rank {self.rank} received zero content")

            except Exception as e:
                print(f"[DeepEP Buffer] Rank {self.rank} broadcast failed: {e}")
                raise RuntimeError(f"Failed to broadcast NVSHMEM unique ID from rank {self.rank}: {e}")

            print(f"[DeepEP Buffer] Rank {self.rank} NVSHMEM unique ID broadcast process completed (no barrier)")

            # All ranks convert the received tensor back to bytes.
            root_unique_id = bytearray(uid_tensor.cpu().tolist())

        self.runtime.sync(device_ids, ipc_handles, root_unique_id)
        assert self.runtime.is_available()

    def destroy(self):
        """
        Destroy the cpp runtime and release resources.
        
        """

        assert self.explicitly_destroy, '`explicitly_destroy` flag must be set'

        self.runtime.destroy()
        self.runtime = None

    @staticmethod
    def is_sm90_compiled():
        return deep_ep_cpp.is_sm90_compiled()

    @staticmethod
    def set_num_sms(new_num_sms: int) -> None:
        """
        Set the number of SMs to use in high-throughput kernels.

        Arguments:
            new_num_sms: the new number to be set.
        """

        assert new_num_sms % 2 == 0, "The SM count must be even"
        Buffer.num_sms = new_num_sms

    @staticmethod
    def capture() -> EventOverlap:
        """
        Capture a CUDA event on the current stream, i.e. `torch.cuda.current_stream()`.

        Returns:
            event: the captured event.
        """
        return EventOverlap(EventHandle())

    @staticmethod
    def get_low_latency_rdma_size_hint(
        num_max_dispatch_tokens_per_rank: int,
        hidden: int,
        num_ranks: int,
        num_experts: int,
    ) -> int:
        """
        Get a minimum size requirement for the RDMA buffer. The size calculation will be done with BF16.

        Arguments:
            num_max_dispatch_tokens_per_rank: the maximum number of tokens to dispatch, all the ranks must hold the same value.
            hidden: the hidden dimension of each token.
            num_ranks: the number of EP group ranks.
            num_experts: the number of all experts.

        Returns:
            size: the RDMA buffer size recommended.
        """
        return deep_ep_cpp.get_low_latency_rdma_size_hint(
            num_max_dispatch_tokens_per_rank, hidden, num_ranks, num_experts
        )

    def get_local_buffer_tensor(
        self,
        dtype: torch.dtype,
        size: Optional[torch.Size] = None,
        offset: int = 0,
        use_rdma_buffer: bool = False,
    ) -> torch.Tensor:
        """
        Get the raw buffer (slice supported) as a PyTorch tensor.

        Argument:
            dtype: the data type (PyTorch `dtype`) for the tensor.
            size: the slice size (by elements) to get from the buffer.
            offset: the offset of the beginning element.
            use_rdma_buffer: whether to return the RDMA buffer.
        """
        tensor = self.runtime.get_local_buffer_tensor(dtype, offset, use_rdma_buffer)
        if size is None:
            return tensor

        assert tensor.numel() >= size.numel()
        return tensor[: size.numel()].view(size)

    @staticmethod
    def get_dispatch_config(num_ranks: int) -> Config:
        """
        Get a recommended dispatch config.

        Argument:
            num_ranks: the number of ranks.

        Returns:
            config: the recommended config.
        """

        # TODO: automatically tune
        config_map = {
            2: Config(Buffer.num_sms, 24, 256, 6, 128),
            4: Config(Buffer.num_sms, 6, 256, 6, 128),
            8: Config(Buffer.num_sms, 6, 256, 6, 128),
            16: Config(Buffer.num_sms, 16, 288, 20, 128),
            24: Config(Buffer.num_sms, 8, 288, 32, 128),
            32: Config(Buffer.num_sms, 8, 288, 32, 128),
            64: Config(Buffer.num_sms, 20, 288, 28, 128),
            128: Config(Buffer.num_sms, 20, 560, 32, 128),
            144: Config(Buffer.num_sms, 32, 720, 12, 128),
            160: Config(Buffer.num_sms, 28, 720, 12, 128),
        }
        assert num_ranks in config_map, f"Unsupported number of EP ranks: {num_ranks}"
        return config_map[num_ranks]

    @staticmethod
    def get_combine_config(num_ranks: int) -> Config:
        """
        Get a recommended combine config.

        Argument:
            num_ranks: the number of ranks.

        Returns:
            config: the recommended config.
        """

        # TODO: automatically tune
        config_map = {
            2: Config(Buffer.num_sms, 10, 256, 6, 128),
            4: Config(Buffer.num_sms, 9, 256, 6, 128),
            8: Config(Buffer.num_sms, 4, 256, 6, 128),
            16: Config(Buffer.num_sms, 2, 288, 28, 128),
            24: Config(Buffer.num_sms, 1, 288, 20, 128),
            32: Config(Buffer.num_sms, 1, 288, 20, 128),
            64: Config(Buffer.num_sms, 1, 288, 20, 128),
            128: Config(Buffer.num_sms, 1, 560, 12, 128),
            144: Config(Buffer.num_sms, 2, 720, 8, 128),
            160: Config(Buffer.num_sms, 2, 720, 8, 128),
        }
        assert num_ranks in config_map, f"Unsupported number of EP ranks: {num_ranks}"
        return config_map[num_ranks]

    # noinspection PyTypeChecker
    def get_dispatch_layout(
        self,
        topk_idx: torch.Tensor,
        num_experts: int,
        previous_event: Optional[EventOverlap] = None,
        async_finish: bool = False,
        allocate_on_comm_stream: bool = False,
    ) -> Tuple[
        torch.Tensor, Optional[torch.Tensor], torch.Tensor, torch.Tensor, EventOverlap
    ]:
        """
        Calculate the layout required for later communication.

        Arguments:
            topk_idx: `[num_tokens, num_topk]`, dtype must be `torch.int64`, the expert indices selected by each token,
                `-1` means no selections.
            num_experts: the number of experts.
            previous_event: the event to wait before actually executing the kernel.
            async_finish: the current stream will not wait for the communication kernels to be finished if set.
            allocate_on_comm_stream: control whether all the allocated tensors' ownership to be on the communication stream.

        Returns:
            num_tokens_per_rank: `[num_ranks]` with `torch.int`, the number of tokens to be sent to each rank.
            num_tokens_per_rdma_rank: `[num_rdma_ranks]` with `torch.int`, the number of tokens to be sent to each RDMA
                rank (with the same GPU index), return `None` for intranode settings.
            num_tokens_per_expert: `[num_experts]` with `torch.int`, the number of tokens to be sent to each expert.
            is_token_in_rank: `[num_tokens, num_ranks]` with `torch.bool`, whether a token be sent to a rank.
            event: the event after executing the kernel (valid only if `async_finish` is set).
        """
        (
            num_tokens_per_rank,
            num_tokens_per_rdma_rank,
            num_tokens_per_expert,
            is_token_in_rank,
            event,
        ) = self.runtime.get_dispatch_layout(
            topk_idx,
            num_experts,
            getattr(previous_event, "event", None),
            async_finish,
            allocate_on_comm_stream,
        )
        return (
            num_tokens_per_rank,
            num_tokens_per_rdma_rank,
            num_tokens_per_expert,
            is_token_in_rank,
            EventOverlap(event),
        )

    # noinspection PyTypeChecker
    def dispatch(
        self,
        x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        handle: Optional[Tuple] = None,
        num_tokens_per_rank: Optional[torch.Tensor] = None,
        num_tokens_per_rdma_rank: Optional[torch.Tensor] = None,
        is_token_in_rank: Optional[torch.Tensor] = None,
        num_tokens_per_expert: Optional[torch.Tensor] = None,
        topk_idx: Optional[torch.Tensor] = None,
        topk_weights: Optional[torch.Tensor] = None,
        expert_alignment: int = 1,
        num_worst_tokens: int = 0,
        config: Optional[Config] = None,
        previous_event: Optional[EventOverlap] = None,
        async_finish: bool = False,
        allocate_on_comm_stream: bool = False,
    ) -> Tuple[
        Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        List[int],
        Tuple,
        EventOverlap,
    ]:
        """
        Dispatch tokens to different ranks, both intranode and internode settings are supported.
        Intranode kernels require all the ranks should be visible via NVLink.
        Internode kernels require the ranks in a node should be visible via NVLink, while the ranks with the same GPU
            index should be visible via RDMA.

        Arguments:
            x: `torch.Tensor` or tuple of `torch.Tensor`, for the first type, the shape must be `[num_tokens, hidden]`,
                and type must be `torch.bfloat16`; for the second type, the first element of the tuple must be shaped as
                `[num_tokens, hidden]` with type `torch.float8_e4m3fn`, the second must be `[num_tokens, hidden // 128]`
                 (requiring divisible) with type `torch.float`.
            handle: an optional communication handle, if set, the CPU will reuse the layout information to save some time.
            num_tokens_per_rank: `[num_ranks]` with `torch.int`, the number of tokens to be sent to each rank.
            num_tokens_per_rdma_rank: `[num_rdma_ranks]` with `torch.int`, the number of tokens to be sent to each RDMA
                rank (with the same GPU index), return `None` for intranode settings.
            is_token_in_rank: `[num_tokens, num_ranks]` with `torch.bool`, whether a token be sent to a rank.
            num_tokens_per_expert: `[num_experts]` with `torch.int`, the number of tokens to be sent to each expert.
            topk_idx: `[num_tokens, num_topk]` with `torch.int64`, the expert indices selected by each token,
                `-1` means no selections.
            topk_weights: `[num_tokens, num_topk]` with `torch.float`, the expert weights of each token to dispatch.
            expert_alignment: align the number of tokens received by each local expert to this variable.
            num_worst_tokens: the worst number of tokens to receive, if specified, there will be no CPU sync, and it
                will be CUDA-graph compatible. Please also notice that this flag is for intranode only.
            config: the performance tuning config.
            previous_event: the event to wait before actually executing the kernel.
            async_finish: the current stream will not wait for the communication kernels to be finished if set.
            allocate_on_comm_stream: control whether all the allocated tensors' ownership to be on the communication stream.

        Returns:
            recv_x: received tokens, the same type and tuple as the input `x`, but the number of tokens equals to the
                received token count.
            recv_topk_idx: received expert indices.
            recv_topk_weights: received expert weights.
            num_recv_tokens_per_expert_list: Python list shaped `[num_local_experts]`, the received token count by
                each local expert, aligned to the input `expert_alignment`. If `num_worst_tokens` is specified, the list
                will be empty.
            handle: the returned communication handle.
            event: the event after executing the kernel (valid only if `async_finish` is set).
        """
        # Default config
        config = self.get_dispatch_config(self.group_size) if config is None else config

        # Internode
        if self.runtime.get_num_rdma_ranks() > 1:
            assert (
                num_worst_tokens == 0
            ), "Internode dispatch does not support `num_worst_tokens > 0`"
            return self.internode_dispatch(
                x,
                handle,
                num_tokens_per_rank,
                num_tokens_per_rdma_rank,
                is_token_in_rank,
                num_tokens_per_expert,
                topk_idx,
                topk_weights,
                expert_alignment,
                config,
                previous_event,
                async_finish,
                allocate_on_comm_stream,
            )

        # Launch the kernel with cached or non-cached mode
        x, x_scales = x if isinstance(x, tuple) else (x, None)
        if handle is not None:
            assert topk_idx is None and topk_weights is None
            (
                rank_prefix_matrix,
                channel_prefix_matrix,
                recv_channel_prefix_matrix,
                recv_src_idx,
                is_token_in_rank,
                send_head,
            ) = handle
            num_recv_tokens = recv_src_idx.size(0)
            recv_x, recv_x_scales, _, _, _, _, _, _, _, _, event = (
                self.runtime.intranode_dispatch(
                    x,
                    x_scales,
                    None,
                    None,
                    None,
                    is_token_in_rank,
                    None,
                    num_recv_tokens,
                    rank_prefix_matrix,
                    channel_prefix_matrix,
                    expert_alignment,
                    num_worst_tokens,
                    config,
                    getattr(previous_event, "event", None),
                    async_finish,
                    allocate_on_comm_stream,
                )
            )
            return (
                (recv_x, recv_x_scales) if x_scales is not None else recv_x,
                None,
                None,
                None,
                None,
                EventOverlap(event),
            )
        else:
            assert (
                num_tokens_per_rank is not None
                and is_token_in_rank is not None
                and num_tokens_per_expert is not None
            )
            (
                recv_x,
                recv_x_scales,
                recv_topk_idx,
                recv_topk_weights,
                num_recv_tokens_per_expert_list,
                rank_prefix_matrix,
                channel_prefix_matrix,
                recv_channel_prefix_matrix,
                recv_src_idx,
                send_head,
                event,
            ) = self.runtime.intranode_dispatch(
                x,
                x_scales,
                topk_idx,
                topk_weights,
                num_tokens_per_rank,
                is_token_in_rank,
                num_tokens_per_expert,
                0,
                None,
                None,
                expert_alignment,
                num_worst_tokens,
                config,
                getattr(previous_event, "event", None),
                async_finish,
                allocate_on_comm_stream,
            )
            handle = (
                rank_prefix_matrix,
                channel_prefix_matrix,
                recv_channel_prefix_matrix,
                recv_src_idx,
                is_token_in_rank,
                send_head,
            )
            return (
                (recv_x, recv_x_scales) if x_scales is not None else recv_x,
                recv_topk_idx,
                recv_topk_weights,
                num_recv_tokens_per_expert_list,
                handle,
                EventOverlap(event),
            )

    # noinspection PyTypeChecker
    def combine(
        self,
        x: torch.Tensor,
        handle: Tuple,
        topk_weights: Optional[torch.Tensor] = None,
        config: Optional[Config] = None,
        previous_event: Optional[EventOverlap] = None,
        async_finish: bool = False,
        allocate_on_comm_stream: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], EventOverlap]:
        """
        Combine (reduce) tokens (addition **without** weights) from different ranks, both intranode and internode
            settings are supported.
        Intranode kernels require all the ranks should be visible via NVLink.
        Internode kernels require the ranks in a node should be visible via NVLink, while the ranks with the same GPU
            index should be visible via RDMA.

        Arguments:
            x: `[num_tokens, hidden]` with `torch.bfloat16`, the tokens to send for reducing to its original ranks.
            handle: a must-set communication handle, you can obtain this from the dispatch function.
            topk_weights: `[num_tokens, num_topk]` with `torch.float`, the tokens' top-k weights for reducing to its original ranks.
            config: the performance tuning config.
            previous_event: the event to wait before actually executing the kernel.
            async_finish: the current stream will not wait for the communication kernels to be finished if set.
            allocate_on_comm_stream: control whether all the allocated tensors' ownership to be on the communication stream.

        Returns:
            recv_x: the reduced token from its dispatched ranks.
            recv_topk_weights: the reduced top-k weights from its dispatch ranks.
            event: the event after executing the kernel (valid only if `async_finish` is set).
        """
        # Default config
        config = self.get_combine_config(self.group_size) if config is None else config

        # Internode
        if self.runtime.get_num_rdma_ranks() > 1:
            return self.internode_combine(
                x,
                handle,
                topk_weights,
                config,
                previous_event,
                async_finish,
                allocate_on_comm_stream,
            )

        # NOTES: the second `_` is for the sending side, so we should use the third one
        (
            rank_prefix_matrix,
            _,
            channel_prefix_matrix,
            src_idx,
            is_recv_token_in_rank,
            send_head,
        ) = handle

        # Launch the kernel
        result = self.runtime.intranode_combine(
            x,
            topk_weights,
            src_idx,
            rank_prefix_matrix,
            channel_prefix_matrix,
            send_head,
            config,
            getattr(previous_event, "event", None),
            async_finish,
            allocate_on_comm_stream,
        )

        event = None
        if topk_weights is not None:
            if isinstance(result, (list, tuple)) and len(result) == 3:
                recv_x, recv_topk_weights, event = result
            elif isinstance(result, (list, tuple)) and len(result) == 2:
                recv_x, event = result
                recv_topk_weights = None
            else:
                recv_x = result
                recv_topk_weights = None
        else:
            if isinstance(result, (list, tuple)) and len(result) == 2:
                recv_x, event = result
            else:
                recv_x = result
            recv_topk_weights = None

        return recv_x, recv_topk_weights, EventOverlap(event)

    # noinspection PyTypeChecker
    def internode_dispatch(
        self,
        x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        handle: Optional[Tuple] = None,
        num_tokens_per_rank: Optional[torch.Tensor] = None,
        num_tokens_per_rdma_rank: Optional[torch.Tensor] = None,
        is_token_in_rank: Optional[torch.Tensor] = None,
        num_tokens_per_expert: Optional[torch.Tensor] = None,
        topk_idx: Optional[torch.Tensor] = None,
        topk_weights: Optional[torch.Tensor] = None,
        expert_alignment: int = 1,
        config: Optional[Config] = None,
        previous_event: Optional[EventOverlap] = None,
        async_finish: bool = False,
        allocate_on_comm_stream: bool = False,
    ) -> Tuple[
        Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        List[int],
        Tuple,
        EventOverlap,
    ]:
        """
        Internode dispatch implementation, for more details, please refer to the `dispatch` docs.
        Normally, you should not directly call this function.
        """
        assert config is not None

        # Launch the kernel with cached or non-cached mode
        x, x_scales = x if isinstance(x, tuple) else (x, None)
        if handle is not None:
            assert topk_idx is None and topk_weights is None
            (
                is_token_in_rank,
                rdma_channel_prefix_matrix,
                gbl_channel_prefix_matrix,
                recv_rdma_channel_prefix_matrix,
                recv_rdma_rank_prefix_sum,
                recv_gbl_channel_prefix_matrix,
                recv_gbl_rank_prefix_sum,
                recv_src_meta,
                send_rdma_head,
                send_nvl_head,
            ) = handle
            num_recv_tokens = recv_src_meta.size(0)
            num_rdma_recv_tokens = send_nvl_head.size(0)
            recv_x, recv_x_scales, _, _, _, _, _, _, _, _, _, _, _, _, event = (
                self.runtime.internode_dispatch(
                    x,
                    x_scales,
                    topk_idx,
                    topk_weights,
                    None,
                    None,
                    is_token_in_rank,
                    None,
                    num_recv_tokens,
                    num_rdma_recv_tokens,
                    rdma_channel_prefix_matrix,
                    recv_rdma_rank_prefix_sum,
                    gbl_channel_prefix_matrix,
                    recv_gbl_rank_prefix_sum,
                    expert_alignment,
                    config,
                    getattr(previous_event, "event", None),
                    async_finish,
                    allocate_on_comm_stream,
                )
            )
            return (
                (recv_x, recv_x_scales) if x_scales is not None else recv_x,
                None,
                None,
                None,
                None,
                EventOverlap(event),
            )
        else:
            assert (
                num_tokens_per_rank is not None
                and is_token_in_rank is not None
                and num_tokens_per_expert is not None
            )
            (
                recv_x,
                recv_x_scales,
                recv_topk_idx,
                recv_topk_weights,
                num_recv_tokens_per_expert_list,
                rdma_channel_prefix_matrix,
                gbl_channel_prefix_matrix,
                recv_rdma_channel_prefix_matrix,
                recv_rdma_rank_prefix_sum,
                recv_gbl_channel_prefix_matrix,
                recv_gbl_rank_prefix_sum,
                recv_src_meta,
                send_rdma_head,
                send_nvl_head,
                event,
            ) = self.runtime.internode_dispatch(
                x,
                x_scales,
                topk_idx,
                topk_weights,
                num_tokens_per_rank,
                num_tokens_per_rdma_rank,
                is_token_in_rank,
                num_tokens_per_expert,
                0,
                0,
                None,
                None,
                None,
                None,
                expert_alignment,
                config,
                getattr(previous_event, "event", None),
                async_finish,
                allocate_on_comm_stream,
            )
            handle = (
                is_token_in_rank,
                rdma_channel_prefix_matrix,
                gbl_channel_prefix_matrix,
                recv_rdma_channel_prefix_matrix,
                recv_rdma_rank_prefix_sum,
                recv_gbl_channel_prefix_matrix,
                recv_gbl_rank_prefix_sum,
                recv_src_meta,
                send_rdma_head,
                send_nvl_head,
            )
            return (
                (recv_x, recv_x_scales) if x_scales is not None else recv_x,
                recv_topk_idx,
                recv_topk_weights,
                num_recv_tokens_per_expert_list,
                handle,
                EventOverlap(event),
            )

    # noinspection PyTypeChecker
    def internode_combine(
        self,
        x: torch.Tensor,
        handle: Union[tuple, list],
        topk_weights: Optional[torch.Tensor] = None,
        config: Optional[Config] = None,
        previous_event: Optional[EventOverlap] = None,
        async_finish: bool = False,
        allocate_on_comm_stream: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], EventOverlap]:
        """
        Internode combine implementation, for more details, please refer to the `combine` docs.
        Normally, you should not directly call this function.
        """
        assert config is not None

        # Unpack handle
        (
            is_combined_token_in_rank,
            _,
            _,
            rdma_channel_prefix_matrix,
            rdma_rank_prefix_sum,
            gbl_channel_prefix_matrix,
            gbl_rank_prefix_sum,
            src_meta,
            send_rdma_head,
            send_nvl_head,
        ) = handle

        # Launch the kernel
        result = self.runtime.internode_combine(
            x,
            topk_weights,
            src_meta,
            is_combined_token_in_rank,
            rdma_channel_prefix_matrix,
            rdma_rank_prefix_sum,
            gbl_channel_prefix_matrix,
            send_rdma_head,
            send_nvl_head,
            config,
            getattr(previous_event, "event", None),
            async_finish,
            allocate_on_comm_stream,
        )

        event = None
        if isinstance(result, (list, tuple)) and len(result) >= 3:
            combined_x, combined_topk_weights, event = result[:3]
        elif isinstance(result, (list, tuple)) and len(result) == 2:
            combined_x, event = result
            combined_topk_weights = None
        else:
            combined_x = result
            combined_topk_weights = None

        return combined_x, combined_topk_weights, EventOverlap(event)

    def clean_low_latency_buffer(
        self, num_max_dispatch_tokens_per_rank: int, hidden: int, num_experts: int
    ) -> None:
        """
        As low-latency kernels require part of the buffer to be zero-initialized, so it is vital to clean the buffer
            if the buffer is dirty at some time.
        For example, after running the normal dispatch/combine, you must run this function before executing any
            low-latency kernel.

        Arguments:
            num_max_dispatch_tokens_per_rank: the maximum number of tokens to dispatch, all the ranks must hold the same value.
            hidden: the hidden dimension of each token.
            num_experts: the number of all experts.
        """
        self.runtime.clean_low_latency_buffer(
            num_max_dispatch_tokens_per_rank, hidden, num_experts
        )

    # noinspection PyTypeChecker
    def low_latency_dispatch(
        self,
        x: torch.Tensor,
        topk_idx: torch.Tensor,
        num_max_dispatch_tokens_per_rank: int,
        num_experts: int,
        cumulative_local_expert_recv_stats: Optional[torch.Tensor] = None,
        use_fp8: bool = True,
        round_scale: bool = False,
        use_ue8m0: bool = False,
        async_finish: bool = False,
        return_recv_hook: bool = False,
    ) -> Tuple[
        Tuple[torch.Tensor, torch.Tensor], torch.Tensor, Tuple, EventOverlap, Callable
    ]:
        """
        A low-latency implementation for dispatching with IBGDA.
        This kernel requires all the ranks (no matter intranode or internode) should be visible via RDMA
            (specifically, IBGDA must be enabled).
        Warning: as there are only two buffers, and the returned tensors reuse the buffer, you cannot hold more than 2
            low-latency kernels' result tensors at a single moment.

        Arguments:
            x: `torch.Tensor` with `torch.bfloat16`, shaped as `[num_tokens, hidden]`, only several hidden shapes are
                supported. The number of tokens to be dispatched must be less than `num_max_dispatch_tokens_per_rank`.
            topk_idx: `torch.Tensor` with `torch.int64`, shaped as `[num_tokens, num_topk]`, only several top-k shapes
                are supported. `-1` indices (not selecting any expert) are supported.
            num_max_dispatch_tokens_per_rank: the maximum number of tokens to dispatch, all the ranks must hold the same value.
            num_experts: the number of all experts.
            cumulative_local_expert_recv_stats: a cumulative expert count tensor for statistics, which should have shape
                `[num_local_experts]` and be typed as `torch.int`. This is useful for online service EP load balance
                monitoring.
            use_fp8: whether to enable FP8 casting, with this, the received data will be a tuple of FP8 tensor and scaling factors.
            round_scale: whether round the scaling factors into power of 2.
            use_ue8m0: whether use UE8M0 as scaling factor format (available only with `round_scale=True`).
            async_finish: the current stream will not wait for the communication kernels to be finished if set.
            return_recv_hook: return a receiving hook if set. If set, the kernel will just do the RDMA request issues,
                but **without actually receiving the data**. You must call the received hook to make sure the data's arrival.
                If you do not set this flag, the kernel will ensure the data's arrival.

        Returns:
            recv_x: a tensor or tuple with received tokens for each expert.
                With `use_fp8=True`: the first element is a `torch.Tensor` shaped as
                `[num_local_experts, num_max_dispatch_tokens_per_rank * num_ranks, hidden]` with `torch.float8_e4m3fn`.
                The second tensor is the corresponding scales for the first element with shape
                `[num_local_experts, num_max_dispatch_tokens_per_rank * num_ranks, hidden // 128]` with `torch.float`,
                if `use_ue8m0=False`. With `use_ue8m0=True`, the second one is packed and shaped as
                `[num_local_experts, num_max_dispatch_tokens_per_rank * num_ranks, hidden // 512]` with type `torch.int`.
                Notice that, the last-two-dimension of the scaling tensors are in column-major for TMA compatibility.
                With `use_fp8=False`, the result would be a tensor shaped as
                `[num_local_experts, num_max_dispatch_tokens_per_rank * num_ranks, hidden]` with `torch.bfloat16`.
                Moreover, not all tokens are valid, only some of the `num_max_dispatch_tokens_per_rank * num_ranks` are,
                as we do not synchronize CPU received count with GPU (also not incompatible with CUDA graph if synced).
            recv_count: a tensor shaped `[num_local_experts]` with type `torch.int`, indicating how many tokens each
                expert receives. As mentioned before, not all tokens are valid in `recv_x`.
            handle: the communication handle to be used in the `low_latency_combine` function.
            event: the event after executing the kernel (valid only if `async_finish` is set).
            hook: the receiving hook function (valid only if `return_recv_hook` is set).
        """
        (
            packed_recv_x,
            packed_recv_x_scales,
            packed_recv_count,
            packed_recv_src_info,
            packed_recv_layout_range,
            event,
            hook,
        ) = self.runtime.low_latency_dispatch(
            x,
            topk_idx,
            cumulative_local_expert_recv_stats,
            num_max_dispatch_tokens_per_rank,
            num_experts,
            use_fp8,
            round_scale,
            use_ue8m0,
            async_finish,
            return_recv_hook,
        )
        handle = (
            packed_recv_src_info,
            packed_recv_layout_range,
            num_max_dispatch_tokens_per_rank,
            x.size(1),
            num_experts,
        )
        tensors_to_record = (
            x,
            topk_idx,
            packed_recv_x,
            packed_recv_x_scales,
            packed_recv_count,
            packed_recv_src_info,
            packed_recv_layout_range,
            cumulative_local_expert_recv_stats,
        )
        return (
            (packed_recv_x, packed_recv_x_scales) if use_fp8 else packed_recv_x,
            packed_recv_count,
            handle,
            EventOverlap(event, tensors_to_record if async_finish else None),
            hook,
        )

    # noinspection PyTypeChecker
    def low_latency_combine(
        self,
        x: torch.Tensor,
        topk_idx: torch.Tensor,
        topk_weights: torch.Tensor,
        handle: tuple,
        zero_copy: bool = False,
        async_finish: bool = False,
        return_recv_hook: bool = False,
        out: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, EventOverlap, Callable]:
        """
        A low-latency implementation for combining tokens (reduce **with weights**) with IBGDA.
        This kernel requires all the ranks (no matter intranode or internode) should be visible via RDMA
            (specifically, IBGDA must be enabled).
        Warning: as there are only two buffers, and the returned tensors reuse the buffer, you cannot hold more than 2
            low-latency kernels' result tensors at a single moment.

        Arguments:
            x: `[num_local_experts, num_max_dispatch_tokens_per_rank * num_ranks, hidden]` with `torch.bfloat16`,
                the local calculated tokens to be sent to this original rank and reduced.
            topk_idx: `[num_combined_tokens, num_topk]` with `torch.int64`, the expert indices selected by the dispatched
                tokens. `-1` indices (not selecting any expert) are supported. Note that, `num_combined_tokens` equals
                to the number of dispatched tokens.
            topk_weights: `[num_combined_tokens, num_topk]` with `torch.float`, the expert weights selected by the dispatched
                tokens. The received tokens will be reduced with the weights in this tensor.
            handle: the communication handle given by the `dispatch` function.
            zero_copy: whether the tensor is already copied into the RDMA buffer, should be cooperative
                with `get_next_low_latency_combine_buffer`.
            async_finish: the current stream will not wait for the communication kernels to be finished if set.
            return_recv_hook: return a receiving hook if set. If set, the kernel will just do the RDMA request issues,
                but **without actually receiving the data**. You must call the received hook to make sure the data's arrival.
                If you do not set this flag, the kernel will ensure the data's arrival.
            out: the in-place output tensor, if set, the kernel will write the result to this tensor and return it directly.

        Returns:
            combined_x: the reduced token tensor, with shape `[num_combined_tokens, hidden]` and type `torch.bfloat16`.
            event: the event after executing the kernel (valid only if `async_finish` is set).
            hook: the receiving hook function (valid only if `return_recv_hook` is set).
        """
        (
            src_info,
            layout_range,
            num_max_dispatch_tokens_per_rank,
            hidden,
            num_experts,
        ) = handle
        combined_x, event, hook = self.runtime.low_latency_combine(
            x,
            topk_idx,
            topk_weights,
            src_info,
            layout_range,
            num_max_dispatch_tokens_per_rank,
            num_experts,
            zero_copy,
            async_finish,
            return_recv_hook,
            out,
        )
        tensors_to_record = (
            x,
            topk_idx,
            topk_weights,
            src_info,
            layout_range,
            combined_x,
        )
        return (
            combined_x,
            EventOverlap(event, tensors_to_record if async_finish else None),
            hook,
        )

    def get_next_low_latency_combine_buffer(self, handle: object):
        """
        Get the raw registered RDMA buffer tensor for next low-latency combine, so that the next combine kernel can skip the copying.

        Arguments:
            handle: the communication handle given by the `dispatch` function.

        Returns:
            buffer: the raw RDMA low-latency buffer as a BF16 PyTorch tensor with shape
                `[num_local_experts, num_ranks * num_max_dispatch_tokens_per_rank, hidden]`, you should fill this buffer
                by yourself.
        """
        (
            src_info,
            layout_range,
            num_max_dispatch_tokens_per_rank,
            hidden,
            num_experts,
        ) = handle
        return self.runtime.get_next_low_latency_combine_buffer(
            num_max_dispatch_tokens_per_rank, hidden, num_experts
        )