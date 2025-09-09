#!/usr/bin/env python3
"""
매우 간단한 DeepEP 테스트 - 작은 크기부터 시작
"""

import os
import signal
import sys
import time
from typing import Any, Dict

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

# DeepEP wrapper 경로 추가
sys.path.insert(0, "inference_engine/inference/core")


def setup_process(rank: int, world_size: int, master_addr: str, master_port: int):
    """개별 프로세스 설정"""
    print(f"🚀 [Rank {rank}] 프로세스 시작")

    # CUDA 환경 변수 설정
    os.environ["CUDA_HOME"] = "/usr/local/cuda-12.4"
    os.environ["PATH"] = f"/usr/local/cuda-12.4/bin:{os.environ.get('PATH', '')}"
    os.environ["LD_LIBRARY_PATH"] = (
        f"/usr/local/cuda-12.4/lib64:{os.environ.get('LD_LIBRARY_PATH', '')}"
    )

    # 분산 환경 변수 설정
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = str(master_port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"

    # CUDA 디바이스 설정
    if torch.cuda.is_available() and rank < torch.cuda.device_count():
        torch.cuda.set_device(rank)
        print(f"✅ [Rank {rank}] CUDA 디바이스 설정: cuda:{rank}")
    else:
        print(f"⚠️ [Rank {rank}] CUDA 디바이스 부족, CPU 사용")


def simple_test_worker(rank: int, world_size: int, master_addr: str, master_port: int):
    """매우 간단한 DeepEP 테스트"""
    try:
        setup_process(rank, world_size, master_addr, master_port)

        print(f"📦 [Rank {rank}] DeepEP Wrapper import 중...")
        from deepep_wrapper import DeepEPWrapper

        # 분산 환경 초기화
        print(f"🌐 [Rank {rank}] 분산 환경 초기화 중...")
        dist.init_process_group(
            backend="nccl",
            init_method=f"tcp://{master_addr}:{master_port}",
            world_size=world_size,
            rank=rank,
        )
        print(f"✅ [Rank {rank}] 분산 환경 초기화 완료")

        # 매우 작은 설정으로 시작
        config = {
            "max_tokens": 512,  # 매우 작게 시작 (기존 4096 -> 512)
            "top_k": 2,  # 작게 시작 (기존 6 -> 2)
            "num_experts": 8,  # 작게 시작 (기존 64 -> 8)
            "hidden_size": 512,  # 작게 시작 (기존 4096 -> 512)
            "intermediate_size": 1024,  # 작게 시작
            "moe_intermediate_size": 256,  # 작게 시작
        }

        print(f"🔧 [Rank {rank}] DeepEP Wrapper 생성 중...")
        wrapper = DeepEPWrapper(config)

        print(f"⚙️ [Rank {rank}] DeepEP 초기화 중...")
        wrapper.initialize(
            world_size=world_size,
            rank=rank,
            num_experts=config["num_experts"],
            hidden_size=config["hidden_size"],
        )

        print(f"✅ [Rank {rank}] DeepEP 초기화 완료!")

        # 매우 작은 텐서로 테스트
        device = f"cuda:{rank}" if torch.cuda.is_available() else "cpu"

        # 매우 작은 크기부터 시작
        test_sizes = [
            (1, 8),  # 8 토큰
            (1, 16),  # 16 토큰
            (1, 32),  # 32 토큰
        ]

        for batch_size, seq_len in test_sizes:
            total_tokens = batch_size * seq_len
            print(f"🧪 [Rank {rank}] 테스트: {total_tokens} 토큰")

            try:
                # 테스트 텐서 생성
                test_input = torch.randn(
                    total_tokens,
                    config["hidden_size"],
                    device=device,
                    dtype=torch.bfloat16,
                )
                test_topk_indices = torch.randint(
                    0,
                    config["num_experts"],
                    (total_tokens, config["top_k"]),
                    device=device,
                )
                test_topk_weights = torch.softmax(
                    torch.randn(
                        total_tokens,
                        config["top_k"],
                        device=device,
                        dtype=torch.float32,
                    ),
                    dim=-1,
                )

                # 동기화
                dist.barrier()

                # Dispatch 테스트
                print(f"🔄 [Rank {rank}] Dispatch 시작...")
                start_time = time.time()
                dispatch_result = wrapper.dispatch(
                    test_input, test_topk_indices, test_topk_weights
                )
                dispatch_time = time.time() - start_time
                print(f"✅ [Rank {rank}] Dispatch 완료: {dispatch_time:.4f}s")

                # 동기화
                dist.barrier()

                # Combine 테스트
                print(f"🔄 [Rank {rank}] Combine 시작...")
                if (
                    isinstance(dispatch_result, (list, tuple))
                    and len(dispatch_result) >= 6
                ):
                    dispatched_tensor = dispatch_result[0]
                    dispatch_handle = dispatch_result[4]
                    recv_topk_weights = (
                        dispatch_result[2]
                        if dispatch_result[2] is not None
                        else test_topk_weights
                    )

                    start_time = time.time()
                    combine_result = wrapper.combine(
                        dispatched_tensor, dispatch_handle, recv_topk_weights
                    )
                    combine_time = time.time() - start_time

                    combined = (
                        combine_result[0]
                        if isinstance(combine_result, (list, tuple))
                        else combine_result
                    )
                    print(
                        f"✅ [Rank {rank}] Combine 완료: {combine_time:.4f}s, 결과: {combined.shape}"
                    )
                else:
                    print(f"❌ [Rank {rank}] 예상치 못한 dispatch_result 구조")
                    continue

                # 동기화
                dist.barrier()

                print(f"🎉 [Rank {rank}] 성공! {total_tokens} 토큰 처리 완료")

            except Exception as e:
                print(f"❌ [Rank {rank}] 실패: {total_tokens} 토큰 - {str(e)[:100]}...")
                break  # 실패하면 더 큰 크기는 시도하지 않음

        return True

    except Exception as e:
        print(f"❌ [Rank {rank}] 전체 오류: {type(e).__name__}: {e}")
        import traceback

        traceback.print_exc()
        return False


def find_free_port():
    """사용 가능한 포트 찾기"""
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def main():
    """메인 함수"""
    print("🚀 ===== 간단한 DeepEP 테스트 =====")

    # CUDA 환경 설정
    os.environ["CUDA_HOME"] = "/usr/local/cuda-12.4"
    os.environ["PATH"] = f"/usr/local/cuda-12.4/bin:{os.environ.get('PATH', '')}"
    os.environ["LD_LIBRARY_PATH"] = (
        f"/usr/local/cuda-12.4/lib64:{os.environ.get('LD_LIBRARY_PATH', '')}"
    )

    # GPU 확인
    num_gpus = torch.cuda.device_count()
    print(f"🔍 사용 가능한 GPU 개수: {num_gpus}")

    if num_gpus < 2:
        print("❌ 최소 2개 GPU가 필요합니다.")
        return False

    # 설정
    world_size = 2
    master_addr = "localhost"
    master_port = find_free_port()

    print(f"🌐 마스터 주소: {master_addr}:{master_port}")

    # 멀티프로세싱 설정
    mp.set_start_method("spawn", force=True)

    # 프로세스 생성
    processes = []
    for rank in range(world_size):
        p = mp.Process(
            target=simple_test_worker, args=(rank, world_size, master_addr, master_port)
        )
        p.start()
        processes.append(p)
        print(f"✅ 프로세스 {rank} 시작")

    # 프로세스 완료 대기
    results = []
    for rank, p in enumerate(processes):
        p.join(timeout=60)  # 1분 타임아웃
        if p.is_alive():
            print(f"⚠️ 프로세스 {rank} 타임아웃, 강제 종료")
            p.terminate()
            p.join()
            results.append(False)
        else:
            results.append(p.exitcode == 0)
            print(f"✅ 프로세스 {rank} 완료 (exitcode={p.exitcode})")

    # 결과
    success_count = sum(results)
    print(f"\n🎯 결과: {success_count}/{len(results)} 성공")
    return success_count == len(results)


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n⚠️ 사용자 중단")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 오류: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
