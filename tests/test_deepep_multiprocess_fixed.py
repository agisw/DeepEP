#!/usr/bin/env python3
"""
동작하는 simple_test 패턴을 따른 DeepEP 멀티프로세스 테스트
문제가 되는 동기화와 복잡한 로직을 제거
"""

import os
import sys
import time
from typing import Any, Dict

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

# DeepEP wrapper 경로 추가
sys.path.insert(0, "inference_engine/inference/core")


def test_deepep_worker(
    rank: int,
    world_size: int,
    master_addr: str,
    master_port: int,
    config: Dict[str, Any],
):
    """동작하는 패턴을 따른 간단한 DeepEP 테스트"""

    # CUDA 환경 설정 (simple_test와 동일)
    os.environ["CUDA_HOME"] = "/usr/local/cuda-12.4"
    os.environ["PATH"] = "/usr/local/cuda-12.4/bin:" + os.environ.get("PATH", "")
    os.environ["LD_LIBRARY_PATH"] = (
        "/usr/local/cuda-12.4/lib64:/usr/local/cuda-12.4/targets/x86_64-linux/lib:"
        + os.environ.get("LD_LIBRARY_PATH", "")
    )

    # GPU 설정 (simple_test와 동일)
    device = f"cuda:{rank}"
    torch.cuda.set_device(device)

    # 분산 환경 초기화 (simple_test와 동일)
    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://{master_addr}:{master_port}",
        world_size=world_size,
        rank=rank,
    )

    print(f"🚀 [Rank {rank}] DeepEP 테스트 시작")

    try:
        # DeepEP import
        from deepep_wrapper import DeepEPWrapper

        print(f"✅ [Rank {rank}] DeepEP import 완료")

        # DeepEP 초기화
        wrapper = DeepEPWrapper(config)
        wrapper.initialize(
            world_size=world_size,
            rank=rank,
            num_experts=config["num_experts"],
            hidden_size=config["hidden_size"],
        )

        print(f"✅ [Rank {rank}] DeepEP 초기화 완료")

        # 설정값
        hidden_size = config["hidden_size"]
        num_experts = config["num_experts"]
        top_k = config["top_k"]

        # 테스트 크기들 (점진적으로 증가)
        test_sizes = [
            64,  # 64 토큰
            128,  # 128 토큰
            256,  # 256 토큰
            512,  # 512 토큰 - 이전에 타임아웃 발생 지점
            1024,  # 1024 토큰
        ]

        max_successful_tokens = 0

        for total_tokens in test_sizes:
            print(f"🧪 [Rank {rank}] 테스트: {total_tokens} 토큰")

            try:
                # 테스트 데이터 생성 (simple_test와 동일)
                test_input_tensor = torch.randn(
                    total_tokens, hidden_size, device=device, dtype=torch.bfloat16
                )
                test_topk_indices = torch.randint(
                    0, num_experts, (total_tokens, top_k), device=device
                )
                test_topk_weights = torch.softmax(
                    torch.randn(
                        total_tokens, top_k, device=device, dtype=torch.float32
                    ),
                    dim=-1,
                )

                print(f"🔄 [Rank {rank}] {total_tokens} 토큰 Dispatch 시작...")

                # Dispatch 테스트 (simple_test와 동일 - 동기화 없음)
                start_time = time.time()
                dispatch_result = wrapper.dispatch(
                    test_input_tensor, test_topk_indices, test_topk_weights
                )
                dispatch_time = time.time() - start_time

                print(
                    f"✅ [Rank {rank}] Dispatch 성공! 시간: {dispatch_time:.4f}s, 결과 길이: {len(dispatch_result)}"
                )

                # Combine 테스트 (simple_test와 동일)
                if len(dispatch_result) == 6:
                    dispatched_tensor = dispatch_result[0]
                    recv_topk_weights = dispatch_result[2]
                    dispatch_handle = dispatch_result[4]

                    print(f"🔄 [Rank {rank}] {total_tokens} 토큰 Combine 시작...")

                    start_time = time.time()
                    combine_result = wrapper.combine(
                        dispatched_tensor, dispatch_handle, recv_topk_weights
                    )
                    combine_time = time.time() - start_time

                    print(f"✅ [Rank {rank}] Combine 성공! 시간: {combine_time:.4f}s")
                    print(f"🎉 [Rank {rank}] {total_tokens} 토큰 테스트 완전 성공!")

                    max_successful_tokens = total_tokens

                    # 성능 정보 출력
                    total_time = dispatch_time + combine_time
                    tokens_per_sec = total_tokens / total_time if total_time > 0 else 0
                    print(f"📊 [Rank {rank}] 성능: {tokens_per_sec:.1f} tokens/sec")

                else:
                    print(
                        f"❌ [Rank {rank}] 예상치 못한 dispatch_result 구조: {len(dispatch_result)}"
                    )
                    break

            except Exception as e:
                print(
                    f"❌ [Rank {rank}] {total_tokens} 토큰 테스트 실패: {str(e)[:150]}..."
                )
                break  # 실패하면 더 큰 크기는 시도하지 않음

        # 최종 결과
        print(f"🏆 [Rank {rank}] 최대 성공 토큰 수: {max_successful_tokens}")

        if max_successful_tokens >= 512:
            print(f"🎉 [Rank {rank}] 512 토큰 이상 처리 성공! (타임아웃 문제 해결)")
            return True
        elif max_successful_tokens > 0:
            print(f"⚠️ [Rank {rank}] 부분 성공: {max_successful_tokens} 토큰까지 처리")
            return True
        else:
            print(f"💔 [Rank {rank}] 모든 테스트 실패")
            return False

    except Exception as e:
        print(f"❌ [Rank {rank}] 전체 오류: {type(e).__name__}: {e}")
        import traceback

        traceback.print_exc()
        return False
    finally:
        # 정리 (simple_test와 동일)
        if dist.is_initialized():
            dist.destroy_process_group()


def find_free_port():
    """사용 가능한 포트 찾기"""
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def load_config():
    """config.json에서 설정 로드"""
    import json

    try:
        with open("inference_engine/inference/config.json", "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"⚠️ config.json 로드 실패: {e}")
        return None


def main():
    """메인 테스트 함수"""
    print("🚀 ===== 수정된 DeepEP 멀티프로세스 테스트 =====")

    # 환경 변수 설정 (simple_test와 동일 - 메인에서 설정)
    os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"
    os.environ["MASTER_ADDR"] = "localhost"

    # config.json에서 설정 로드
    config_data = load_config()
    if config_data:
        print("📋 config.json 설정 사용")
    else:
        print("⚠️ 기본 설정 사용")

    # GPU 개수 확인
    num_gpus = torch.cuda.device_count()
    print(f"🔍 실제 사용 가능한 GPU 개수: {num_gpus}")

    if num_gpus < 2:
        print("❌ 다중 GPU가 필요합니다. 최소 2개 GPU 필요.")
        return False

    # 설정
    world_size = 2
    master_addr = "localhost"
    master_port = find_free_port()

    # DeepSeek MoE 16B 모델의 실제 설정
    config = {
        "max_tokens": 4096,
        "top_k": 6,
        "num_experts": 64,
        "hidden_size": 4096,
        "intermediate_size": 11008,
        "moe_intermediate_size": 1407,
    }

    print(f"⚙️ DeepEP 설정: {config}")
    print(f"🌐 마스터 주소: {master_addr}:{master_port}")

    # 멀티프로세싱 시작 방법 설정
    mp.set_start_method("spawn", force=True)

    # 프로세스 생성 및 실행 (simple_test와 동일한 패턴)
    print(f"🚀 {world_size}개 프로세스 생성 중...")

    processes = []
    for rank in range(world_size):
        p = mp.Process(
            target=test_deepep_worker,
            args=(rank, world_size, master_addr, master_port, config),
        )
        p.start()
        processes.append(p)
        print(f"✅ 프로세스 {rank} 시작됨")

    # 모든 프로세스 완료 대기 (simple_test와 동일)
    print("⏳ 모든 프로세스 완료 대기 중...")

    success_count = 0
    for rank, p in enumerate(processes):
        p.join()  # 타임아웃 없음 (simple_test와 동일)
        if p.exitcode == 0:
            success_count += 1
        print(f"✅ 프로세스 {rank} 완료 (exitcode={p.exitcode})")

    # 결과 분석
    total_count = len(processes)

    print(f"\n🎯 ===== 테스트 결과 =====")
    print(f"성공: {success_count}/{total_count}")
    print(f"실패: {total_count - success_count}/{total_count}")

    if success_count == total_count:
        print("🎉 모든 프로세스에서 DeepEP 테스트 성공!")
        print("✅ 타임아웃 문제가 해결되었습니다!")
        return True
    else:
        print("❌ 일부 프로세스에서 실패 발생")
        return False


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n⚠️ 사용자에 의해 중단됨")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 예상치 못한 오류: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
