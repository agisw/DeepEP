#!/usr/bin/env python3

import multiprocessing as mp
import os
import sys

import torch
import torch.distributed as dist

# 현재 디렉토리를 path에 추가
sys.path.append(".")


def test_4096_tokens(rank, world_size):
    """4096 토큰만 테스트하는 간단한 함수"""

    # CUDA 환경 설정
    os.environ["CUDA_HOME"] = "/usr/local/cuda-12.4"
    os.environ["PATH"] = "/usr/local/cuda-12.4/bin:" + os.environ.get("PATH", "")
    os.environ["LD_LIBRARY_PATH"] = (
        "/usr/local/cuda-12.4/lib64:/usr/local/cuda-12.4/targets/x86_64-linux/lib:"
        + os.environ.get("LD_LIBRARY_PATH", "")
    )

    # GPU 설정
    device = f"cuda:{rank}"
    torch.cuda.set_device(device)

    # 분산 환경 초기화
    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://localhost:29522",
        world_size=world_size,
        rank=rank,
    )

    print(f"🚀 [Rank {rank}] 4096 토큰 테스트 시작")

    try:
        # DeepEP import
        from inference_engine.inference.core.deepep_wrapper import DeepEPWrapper

        # 설정
        config = {
            "max_tokens": 8192,  # 4096 토큰을 위해 더 큰 버퍼
            "top_k": 6,
            "num_experts": 64,
            "hidden_size": 4096,
        }

        # DeepEP 초기화
        wrapper = DeepEPWrapper(config)
        wrapper.initialize(world_size, rank, 64, 4096)

        print(f"✅ [Rank {rank}] DeepEP 초기화 완료")

        # 4096 토큰 테스트
        total_tokens = 4096
        test_input_tensor = torch.randn(
            total_tokens, 4096, device=device, dtype=torch.bfloat16
        )
        test_topk_indices = torch.randint(0, 64, (total_tokens, 6), device=device)
        test_topk_weights = torch.softmax(
            torch.randn(total_tokens, 6, device=device, dtype=torch.float32), dim=-1
        )

        print(f"🔄 [Rank {rank}] 4096 토큰 Dispatch 시작...")

        # Dispatch 테스트
        dispatch_result = wrapper.dispatch(
            test_input_tensor, test_topk_indices, test_topk_weights
        )

        print(f"✅ [Rank {rank}] Dispatch 성공! 결과 길이: {len(dispatch_result)}")

        # Combine 테스트
        if len(dispatch_result) == 6:
            dispatched_tensor = dispatch_result[0]
            recv_topk_weights = dispatch_result[2]
            dispatch_handle = dispatch_result[4]

            print(f"🔄 [Rank {rank}] 4096 토큰 Combine 시작...")

            combine_result = wrapper.combine(
                dispatched_tensor, dispatch_handle, recv_topk_weights
            )

            print(f"🎉 [Rank {rank}] 4096 토큰 테스트 완전 성공!")
            return True

    except Exception as e:
        print(f"❌ [Rank {rank}] 4096 토큰 테스트 실패: {e}")
        return False
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def main():
    """메인 함수"""
    os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29522"

    world_size = 2
    processes = []

    print("🚀 4096 토큰 단일 테스트 시작")

    for rank in range(world_size):
        p = mp.Process(target=test_4096_tokens, args=(rank, world_size))
        p.start()
        processes.append(p)

    success_count = 0
    for p in processes:
        p.join()
        if p.exitcode == 0:
            success_count += 1
        print(f"Process {p.pid} completed with exit code: {p.exitcode}")

    print(f"🎯 결과: {success_count}/{world_size} 프로세스 성공")

    if success_count == world_size:
        print("🎉 4096 토큰 테스트 완전 성공!")
    else:
        print("❌ 일부 프로세스에서 실패")


if __name__ == "__main__":
    main()
