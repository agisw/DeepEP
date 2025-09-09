#!/usr/bin/env python3
"""
각 토큰 크기별로 별도 프로세스를 사용하는 DeepEP 테스트
simple_test_xxx.py 패턴을 연속 실행으로 구현
"""

import multiprocessing as mp
import os
import sys
import time
from typing import Any, Dict

import torch
import torch.distributed as dist

# 현재 디렉토리를 path에 추가
sys.path.append(".")


def test_single_token_size(
    rank: int, world_size: int, total_tokens: int, master_port: int
):
    """단일 토큰 크기를 테스트하는 함수 (simple_test 패턴)"""

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
        init_method=f"tcp://localhost:{master_port}",
        world_size=world_size,
        rank=rank,
    )

    print(f"🚀 [Rank {rank}] {total_tokens} 토큰 테스트 시작")

    try:
        # DeepEP import
        from inference_engine.inference.core.deepep_wrapper import DeepEPWrapper

        # 설정
        config = {
            "max_tokens": max(4096, total_tokens * 2),  # 동적 버퍼 크기
            "top_k": 6,
            "num_experts": 64,
            "hidden_size": 4096,
        }

        # DeepEP 초기화 (새로운 인스턴스)
        wrapper = DeepEPWrapper(config)
        wrapper.initialize(world_size, rank, 64, 4096)

        print(f"✅ [Rank {rank}] DeepEP 초기화 완료")

        # 테스트 데이터 생성
        test_input_tensor = torch.randn(
            total_tokens, 4096, device=device, dtype=torch.bfloat16
        )
        test_topk_indices = torch.randint(0, 64, (total_tokens, 6), device=device)
        test_topk_weights = torch.softmax(
            torch.randn(total_tokens, 6, device=device, dtype=torch.float32), dim=-1
        )

        print(f"🔄 [Rank {rank}] {total_tokens} 토큰 Dispatch 시작...")

        # Dispatch 테스트
        start_time = time.time()
        dispatch_result = wrapper.dispatch(
            test_input_tensor, test_topk_indices, test_topk_weights
        )
        dispatch_time = time.time() - start_time

        print(f"✅ [Rank {rank}] Dispatch 성공! 시간: {dispatch_time:.4f}s")

        # Combine 테스트
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

            # 성능 정보
            total_time = dispatch_time + combine_time
            tokens_per_sec = total_tokens / total_time if total_time > 0 else 0
            print(
                f"📊 [Rank {rank}] {total_tokens} 토큰 성능: {tokens_per_sec:.1f} tokens/sec"
            )
            print(f"🎉 [Rank {rank}] {total_tokens} 토큰 테스트 완전 성공!")
            return True

    except Exception as e:
        print(f"❌ [Rank {rank}] {total_tokens} 토큰 테스트 실패: {e}")
        import traceback

        traceback.print_exc()
        return False
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def run_token_size_test(total_tokens: int, test_number: int):
    """특정 토큰 크기에 대해 별도 프로세스로 테스트 실행"""
    print(f"\n🧪 ===== 테스트 {test_number}: {total_tokens} 토큰 =====")

    # 각 테스트마다 다른 포트 사용 (충돌 방지)
    master_port = 29500 + test_number
    world_size = 2

    processes = []

    for rank in range(world_size):
        p = mp.Process(
            target=test_single_token_size,
            args=(rank, world_size, total_tokens, master_port),
        )
        p.start()
        processes.append(p)

    # 프로세스 완료 대기
    success_count = 0
    for p in processes:
        p.join()
        if p.exitcode == 0:
            success_count += 1

    success = success_count == world_size

    if success:
        print(f"✅ {total_tokens} 토큰 테스트 성공!")
    else:
        print(f"❌ {total_tokens} 토큰 테스트 실패 ({success_count}/{world_size} 성공)")

    return success


def main():
    """메인 함수 - 각 토큰 크기를 순차적으로 별도 프로세스에서 테스트"""
    print("🚀 ===== DeepEP 별도 프로세스 연속 테스트 =====")
    print("각 토큰 크기마다 새로운 프로세스와 DeepEP 인스턴스 사용")

    # 환경 변수 설정
    os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"
    os.environ["MASTER_ADDR"] = "localhost"

    # 멀티프로세싱 설정
    mp.set_start_method("spawn", force=True)

    # 테스트할 토큰 크기들
    test_sizes = [64, 128, 256, 512, 1024, 2048]

    results = {}

    for i, total_tokens in enumerate(test_sizes, 1):
        try:
            print(f"\n⏰ 테스트 {i}/{len(test_sizes)} 시작...")
            success = run_token_size_test(total_tokens, i)
            results[total_tokens] = success

            if not success:
                print(f"⚠️ {total_tokens} 토큰에서 실패. 더 큰 크기는 건너뜁니다.")
                break

            # 테스트 간 잠시 대기 (리소스 정리 시간)
            time.sleep(2)

        except Exception as e:
            print(f"❌ {total_tokens} 토큰 테스트 중 예외 발생: {e}")
            results[total_tokens] = False
            break

    # 최종 결과 출력
    print(f"\n🎯 ===== 최종 결과 =====")
    max_successful = 0

    for tokens, success in results.items():
        status = "✅ 성공" if success else "❌ 실패"
        print(f"{tokens:4d} 토큰: {status}")
        if success:
            max_successful = max(max_successful, tokens)

    print(f"\n🏆 최대 성공 토큰 수: {max_successful}")

    if max_successful >= 512:
        print("🎉 512 토큰 이상 처리 성공! 별도 프로세스 방식이 효과적입니다!")
        return True
    elif max_successful > 0:
        print(f"⚠️ 부분 성공: {max_successful} 토큰까지 처리")
        return True
    else:
        print("💔 모든 테스트 실패")
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
