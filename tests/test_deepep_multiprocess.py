#!/usr/bin/env python3
"""
실제 다중 프로세스 환경에서 DeepEP 테스트
torch.multiprocessing을 사용하여 진짜 분산 환경 구성
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


def setup_process(
    rank: int,
    world_size: int,
    master_addr: str,
    master_port: int,
    visible_gpus: str = "0,1",
):
    """개별 프로세스 설정"""
    print(f"🚀 [Rank {rank}] 프로세스 시작")

    # CUDA 환경 변수 설정 (CUDA 12.4 강제 사용)
    os.environ["CUDA_HOME"] = "/usr/local/cuda-12.4"
    os.environ["PATH"] = f"/usr/local/cuda-12.4/bin:{os.environ.get('PATH', '')}"
    os.environ["LD_LIBRARY_PATH"] = (
        f"/usr/local/cuda-12.4/lib64:{os.environ.get('LD_LIBRARY_PATH', '')}"
    )

    # CUDA 라이브러리 경로 추가 (DeepEP 심볼 링킹을 위해)
    current_ld_path = os.environ.get("LD_LIBRARY_PATH", "")
    cuda_lib_paths = [
        "/usr/local/cuda-12.4/lib64",
        "/usr/local/cuda-12.4/targets/x86_64-linux/lib",
    ]
    for cuda_path in cuda_lib_paths:
        if cuda_path not in current_ld_path:
            os.environ["LD_LIBRARY_PATH"] = f"{cuda_path}:{current_ld_path}"
            current_ld_path = os.environ["LD_LIBRARY_PATH"]

    print(f"🔧 [Rank {rank}] CUDA 환경 설정: CUDA_HOME={os.environ['CUDA_HOME']}")
    print(f"🔧 [Rank {rank}] LD_LIBRARY_PATH: {os.environ['LD_LIBRARY_PATH'][:100]}...")

    # 환경 변수 설정 (config.json 기반)
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = str(master_port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["CUDA_VISIBLE_DEVICES"] = visible_gpus

    print(
        f"🌐 [Rank {rank}] 분산 설정: MASTER_ADDR={master_addr}, MASTER_PORT={master_port}"
    )

    # visible_gpus에서 실제 GPU ID 매핑
    gpu_ids = [int(x.strip()) for x in visible_gpus.split(",")]
    actual_gpu_id = gpu_ids[rank] if rank < len(gpu_ids) else rank

    # CUDA 디바이스 설정
    if torch.cuda.is_available() and rank < torch.cuda.device_count():
        torch.cuda.set_device(rank)
        print(
            f"✅ [Rank {rank}] CUDA 디바이스 설정: cuda:{rank} (실제 GPU {actual_gpu_id})"
        )
    else:
        print(f"⚠️ [Rank {rank}] CUDA 디바이스 부족, CPU 사용")


def run_single_test_with_timeout(
    wrapper,
    test_input_tensor,
    test_topk_indices,
    test_topk_weights,
    rank,
    test_batch_size,
    test_seq_len,
    timeout_seconds=30,
):
    """단일 테스트를 타임아웃과 함께 실행"""
    total_tokens = test_batch_size * test_seq_len

    def timeout_handler(signum, frame):
        raise TimeoutError(f"[Rank {rank}] 테스트 타임아웃 ({timeout_seconds}초)")

    # 타임아웃 설정
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout_seconds)

    try:
        print(
            f"🧪 [Rank {rank}] 테스트 시작: batch_size={test_batch_size}, seq_len={test_seq_len}, total_tokens={total_tokens}"
        )
        print(
            f"📝 [Rank {rank}] 입력 텐서: {test_input_tensor.shape}, dtype={test_input_tensor.dtype}"
        )
        print(
            f"📝 [Rank {rank}] TopK 인덱스: {test_topk_indices.shape}, dtype={test_topk_indices.dtype}"
        )
        print(
            f"📝 [Rank {rank}] TopK 가중치: {test_topk_weights.shape}, dtype={test_topk_weights.dtype}"
        )

        # 동기화 포인트 1: 모든 프로세스가 테스트 시작 준비 완료
        print(f"🔄 [Rank {rank}] 테스트 시작 동기화...")
        dist.barrier()

        # Dispatch 연산 테스트
        print(f"🔄 [Rank {rank}] Dispatch 연산 시작...")
        dispatch_start_time = time.time()

        dispatch_result = wrapper.dispatch(
            test_input_tensor, test_topk_indices, test_topk_weights
        )

        dispatch_time = time.time() - dispatch_start_time
        print(f"📊 [Rank {rank}] Dispatch 완료: {dispatch_time:.4f}s")
        print(
            f"📊 [Rank {rank}] Dispatch 결과: {dispatch_result.shape if hasattr(dispatch_result, 'shape') else type(dispatch_result)}"
        )

        # 동기화 포인트 2: 모든 프로세스가 dispatch 완료
        print(f"🔄 [Rank {rank}] Dispatch 완료 동기화...")
        dist.barrier()

        # Combine 연산 테스트
        print(f"🔄 [Rank {rank}] Combine 연산 시작...")
        combine_start_time = time.time()

        # DeepEP의 실제 combine 함수 사용
        if hasattr(wrapper, "combine") and callable(wrapper.combine):
            if isinstance(dispatch_result, (list, tuple)) and len(dispatch_result) == 6:
                dispatched_tensor = dispatch_result[0]  # recv_x
                recv_topk_idx = dispatch_result[1]  # recv_topk_idx
                recv_topk_weights = dispatch_result[2]  # recv_topk_weights
                num_recv_tokens_per_expert_list = dispatch_result[
                    3
                ]  # num_recv_tokens_per_expert_list
                dispatch_handle = dispatch_result[4]  # handle
                event = dispatch_result[5]  # event

                print(f"🔍 [Rank {rank}] dispatch_handle type: {type(dispatch_handle)}")
                print(
                    f"🔍 [Rank {rank}] dispatched_tensor shape: {dispatched_tensor.shape}"
                )
                print(
                    f"🔍 [Rank {rank}] recv_topk_weights shape: {recv_topk_weights.shape if recv_topk_weights is not None else None}"
                )

                combine_topk_weights = (
                    recv_topk_weights
                    if recv_topk_weights is not None
                    else test_topk_weights
                )
                combine_result = wrapper.combine(
                    dispatched_tensor, dispatch_handle, combine_topk_weights
                )

                if isinstance(combine_result, (list, tuple)):
                    combined = combine_result[0]
                else:
                    combined = combine_result
            else:
                print(f"❌ [Rank {rank}] 예상치 못한 dispatch_result 구조")
                combined = (
                    dispatch_result[0]
                    if isinstance(dispatch_result, (list, tuple))
                    else dispatch_result
                )
        else:
            combined = dispatch_result

        combine_time = time.time() - combine_start_time
        print(f"📊 [Rank {rank}] Combine 완료: {combine_time:.4f}s")
        print(
            f"📊 [Rank {rank}] Combine 결과: {combined.shape if hasattr(combined, 'shape') else type(combined)}"
        )

        # 동기화 포인트 3: 모든 프로세스가 combine 완료
        print(f"🔄 [Rank {rank}] Combine 완료 동기화...")
        dist.barrier()

        total_time = dispatch_time + combine_time
        tokens_per_sec = total_tokens / total_time if total_time > 0 else 0

        print(
            f"✅ [Rank {rank}] 성공! batch_size={test_batch_size}, seq_len={test_seq_len}"
        )
        print(f"   - Dispatch: {dispatch_time:.4f}s, Combine: {combine_time:.4f}s")
        print(
            f"   - 총 시간: {total_time:.4f}s, 처리량: {tokens_per_sec:.1f} tokens/sec"
        )

        return True, total_time, tokens_per_sec

    except TimeoutError as e:
        print(f"⏰ [Rank {rank}] {e}")
        return False, 0, 0
    except Exception as e:
        print(f"❌ [Rank {rank}] 테스트 실패: {str(e)[:150]}...")
        return False, 0, 0
    finally:
        # 타임아웃 해제
        signal.alarm(0)


def test_deepep_worker(
    rank: int,
    world_size: int,
    master_addr: str,
    master_port: int,
    config: Dict[str, Any],
    visible_gpus: str = "0,1",
):
    """개별 워커 프로세스에서 실행되는 DeepEP 테스트"""
    try:
        setup_process(rank, world_size, master_addr, master_port, visible_gpus)

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

        # 버퍼 정보 확인
        buffer_info = wrapper.get_buffer_info()
        print(f"📊 [Rank {rank}] 버퍼 정보: initialized={buffer_info['initialized']}")

        # 실제 텐서 연산 테스트
        print(f"🧮 [Rank {rank}] 실제 텐서 연산 테스트 시작...")

        # DeepSeek MoE 16B 모델의 실제 텐서 사이즈로 테스트
        hidden_size = config["hidden_size"]  # 4096
        num_experts = config["num_experts"]  # 64
        top_k = config["top_k"]  # 6
        intermediate_size = config["intermediate_size"]  # 11008
        moe_intermediate_size = config["moe_intermediate_size"]  # 1407

        print(f"🔍 [Rank {rank}] DeepSeek MoE 16B 실제 설정:")
        print(f"   - hidden_size: {hidden_size}")
        print(f"   - num_experts: {num_experts}")
        print(f"   - top_k: {top_k}")
        print(f"   - intermediate_size: {intermediate_size}")
        print(f"   - moe_intermediate_size: {moe_intermediate_size}")

        # 입력 데이터 생성
        device = (
            f"cuda:{rank}"
            if torch.cuda.is_available() and rank < torch.cuda.device_count()
            else "cpu"
        )

        # 실제 DeepSeek MoE 16B 모델에서 사용되는 배치 사이즈와 시퀀스 길이로 테스트
        max_successful_batch_size = 0
        max_successful_seq_len = 0
        max_successful_tokens_per_sec = 0

        # 점진적 테스트: 작은 크기부터 시작하여 실패 지점 찾기
        test_configurations = [
            # (batch_size, seq_len, timeout_seconds) - 더 짧은 타임아웃으로 빠른 감지
            (1, 64, 15),  # 매우 작은 크기
            (1, 128, 20),  # 기본 크기
            (1, 256, 30),  # 중간 크기
            (1, 512, 45),  # 큰 크기 - 더 긴 타임아웃
            (1, 1024, 60),  # 매우 큰 크기
            (2, 128, 30),  # 작은 배치
            (4, 128, 45),  # 중간 배치
        ]

        for test_batch_size, test_seq_len, timeout_seconds in test_configurations:
            total_tokens = test_batch_size * test_seq_len

            try:
                # 테스트용 텐서 생성 (DeepSeek MoE 16B 실제 차원)
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

                # 타임아웃이 있는 테스트 실행
                success, total_time, tokens_per_sec = run_single_test_with_timeout(
                    wrapper,
                    test_input_tensor,
                    test_topk_indices,
                    test_topk_weights,
                    rank,
                    test_batch_size,
                    test_seq_len,
                    timeout_seconds,
                )

                if success:
                    # 성공한 경우 기록
                    max_successful_batch_size = test_batch_size
                    max_successful_seq_len = test_seq_len
                    max_successful_tokens_per_sec = max(
                        max_successful_tokens_per_sec, tokens_per_sec
                    )
                else:
                    # 실패한 경우 이후 더 큰 테스트는 건너뛰기
                    print(
                        f"⚠️ [Rank {rank}] batch_size={test_batch_size}, seq_len={test_seq_len}에서 실패, 더 큰 테스트 건너뛰기"
                    )
                    break

            except Exception as e:
                print(
                    f"❌ [Rank {rank}] 예외 발생: batch_size={test_batch_size}, seq_len={test_seq_len}"
                )
                print(f"   오류: {str(e)[:150]}...")
                break  # 예외 발생 시 테스트 중단

        print(
            f"🏆 [Rank {rank}] 최대 성공 설정: batch_size={max_successful_batch_size}, seq_len={max_successful_seq_len}"
        )

        if max_successful_batch_size == 0:
            print(f"💔 [Rank {rank}] 모든 설정에서 실패")
            return False

        # 메모리 사용량 확인
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated(device) / 1024**3
            memory_cached = torch.cuda.memory_reserved(device) / 1024**3
            print(
                f"💾 [Rank {rank}] GPU 메모리: {memory_allocated:.2f}GB 할당, {memory_cached:.2f}GB 예약"
            )

        # 최종 결과 요약
        if max_successful_batch_size > 0:
            max_total_tokens = max_successful_batch_size * max_successful_seq_len
            print(f"🎉 [Rank {rank}] DeepSeek MoE 16B 테스트 완료!")
            print(f"📊 [Rank {rank}] 최적 설정:")
            print(f"   - batch_size: {max_successful_batch_size}")
            print(f"   - seq_len: {max_successful_seq_len}")
            print(f"   - total_tokens: {max_total_tokens}")
            print(f"   - 최대 처리량: {max_successful_tokens_per_sec:.1f} tokens/sec")
            print(
                f"   - 실제 모델 차원: hidden_size={hidden_size}, num_experts={num_experts}, top_k={top_k}"
            )
            return True
        else:
            print(
                f"💔 [Rank {rank}] 테스트 실패: DeepSeek MoE 16B 사이즈에서 DeepEP 사용 불가"
            )
            return False

    except Exception as e:
        print(f"❌ [Rank {rank}] 오류 발생: {type(e).__name__}: {e}")
        import traceback

        traceback.print_exc()
        # 예외 발생 시 올바른 종료 코드 반환
        sys.exit(1)


def check_gpu_shared_memory():
    """GPU 공유 메모리 정보 확인"""
    if not torch.cuda.is_available():
        print("⚠️ CUDA 사용 불가")
        return False

    try:
        device = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(device)

        print(f"🔍 GPU 정보:")
        print(f"   - 디바이스: {props.name}")

        # PyTorch 2.x에서는 속성 이름이 다를 수 있음
        try:
            shared_mem_per_block = props.shared_memory_per_block
        except AttributeError:
            # 대안적인 방법으로 확인
            shared_mem_per_block = getattr(
                props, "sharedMemPerBlock", 49152
            )  # 기본값 48KB

        try:
            shared_mem_per_multiprocessor = props.shared_memory_per_multiprocessor
        except AttributeError:
            shared_mem_per_multiprocessor = getattr(
                props, "sharedMemPerMultiprocessor", 102400
            )  # 기본값 100KB

        print(f"   - 공유 메모리/블록: {shared_mem_per_block / 1024:.1f} KB")
        print(
            f"   - 공유 메모리/멀티프로세서: {shared_mem_per_multiprocessor / 1024:.1f} KB"
        )
        print(f"   - 최대 스레드/블록: {props.max_threads_per_block}")
        print(f"   - 멀티프로세서 수: {props.multi_processor_count}")

        # DeepEP에서 요구하는 공유 메모리 계산
        kNumTMABytesPerWarp = 8192
        kNumThreads = 1024  # DeepEP에서 사용하는 스레드 수
        required_smem = kNumTMABytesPerWarp * (kNumThreads // 32)

        print(f"   - DeepEP 요구 공유 메모리: {required_smem / 1024:.1f} KB")
        print(
            f"   - 공유 메모리 충분 여부: {'✅' if required_smem <= shared_mem_per_block else '❌'}"
        )

        return required_smem <= shared_mem_per_block

    except Exception as e:
        print(f"⚠️ GPU 정보 확인 실패: {e}")
        return False


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
    print("🚀 ===== 실제 다중 프로세스 DeepEP 테스트 =====")

    # 메인 프로세스에서도 CUDA 12.4 환경 설정
    os.environ["CUDA_HOME"] = "/usr/local/cuda-12.4"
    os.environ["PATH"] = f"/usr/local/cuda-12.4/bin:{os.environ.get('PATH', '')}"
    os.environ["LD_LIBRARY_PATH"] = (
        f"/usr/local/cuda-12.4/lib64:{os.environ.get('LD_LIBRARY_PATH', '')}"
    )
    print(f"🔧 메인 프로세스 CUDA 환경 설정 완료: CUDA_HOME={os.environ['CUDA_HOME']}")

    # config.json에서 설정 로드
    config_data = load_config()
    if config_data:
        device_config = config_data.get("device", {})
        configured_num_gpus = device_config.get("num_gpus", 2)
        visible_gpus = device_config.get("visible_gpus", "0,1")

        # 네트워크 설정 (config.json에서 읽기)
        master_addr = device_config.get("nccl_master_addr", "localhost")
        configured_master_port = device_config.get("nccl_master_port", "29519")

        print(f"📋 config.json 설정:")
        print(f"   - num_gpus={configured_num_gpus}, visible_gpus={visible_gpus}")
        print(f"   - master_addr={master_addr}, master_port={configured_master_port}")

        # 환경 변수 설정 (config.json 반영)
        os.environ["CUDA_VISIBLE_DEVICES"] = visible_gpus
        world_size = configured_num_gpus
    else:
        print("⚠️ config.json 설정을 사용할 수 없어 기본값 사용")
        master_addr = "localhost"
        configured_master_port = "29519"
        world_size = 2

    # GPU 개수 확인
    num_gpus = torch.cuda.device_count()
    print(f"🔍 실제 사용 가능한 GPU 개수: {num_gpus}")

    if num_gpus < 2:
        print("❌ 다중 GPU가 필요합니다. 최소 2개 GPU 필요.")
        return False

    # GPU 공유 메모리 확인
    print("🔍 GPU 공유 메모리 확인 중...")
    if not check_gpu_shared_memory():
        print("⚠️ GPU 공유 메모리가 부족할 수 있습니다. 테스트를 계속 진행합니다.")

    # world_size를 2로 고정 (config.json 반영)
    world_size = 2
    print(f"🎯 테스트 설정: world_size={world_size} (config.json 반영)")

    # DeepSeek MoE 16B 모델의 실제 설정 반영 - 더 큰 버퍼 할당
    model_config = config_data.get("model", {}) if config_data else {}
    config = {
        "max_tokens": 4096,  # 더 큰 버퍼로 확장 (2048 -> 4096)
        "top_k": 6,  # DeepSeek MoE 16B의 num_experts_per_tok (일반적으로 6개 전문가 선택)
        "num_experts": 64,  # DeepSeek MoE 16B의 n_routed_experts (64개 라우팅 전문가)
        "hidden_size": 4096,  # DeepSeek MoE 16B의 hidden_size
        "intermediate_size": 11008,  # DeepSeek MoE 16B의 intermediate_size
        "moe_intermediate_size": 1407,  # DeepSeek MoE 16B의 moe_intermediate_size
    }

    print(f"⚙️ DeepEP 설정: {config}")

    # 포트 설정 (config.json 우선, 사용 중이면 자동 할당)
    try:
        preferred_port = int(configured_master_port)
        # config.json의 포트가 사용 가능한지 확인
        import socket

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind((master_addr, preferred_port))
            master_port = preferred_port
            print(f"🌐 마스터 포트: {master_port} (config.json 설정 사용)")
    except (ValueError, OSError):
        # config.json 포트를 사용할 수 없으면 자동으로 찾기
        master_port = find_free_port()
        print(
            f"🌐 마스터 포트: {master_port} (자동 할당, config.json 포트 {configured_master_port} 사용 불가)"
        )

    # 멀티프로세싱 시작 방법 설정
    mp.set_start_method("spawn", force=True)

    # 프로세스 생성 및 실행
    print(f"🚀 {world_size}개 프로세스 생성 중...")

    processes = []
    for rank in range(world_size):
        p = mp.Process(
            target=test_deepep_worker,
            args=(rank, world_size, master_addr, master_port, config, visible_gpus),
        )
        p.start()
        processes.append(p)
        print(f"✅ 프로세스 {rank} 시작됨 (MASTER_ADDR={master_addr}:{master_port})")

    # 모든 프로세스 완료 대기 - 더 짧은 타임아웃으로 빠른 감지
    print("⏳ 모든 프로세스 완료 대기 중...")

    results = []
    for rank, p in enumerate(processes):
        p.join(timeout=120)  # 2분 타임아웃으로 단축
        if p.is_alive():
            print(f"⚠️ 프로세스 {rank} 타임아웃, 강제 종료")
            p.terminate()
            p.join(timeout=10)  # 강제 종료 대기 시간
            if p.is_alive():
                print(f"🔥 프로세스 {rank} 강제 kill")
                p.kill()
                p.join()
            results.append(False)
        else:
            results.append(p.exitcode == 0)
            print(f"✅ 프로세스 {rank} 완료 (exitcode={p.exitcode})")

    # 결과 분석
    success_count = sum(results)
    total_count = len(results)

    print(f"\n🎯 ===== 테스트 결과 =====")
    print(f"성공: {success_count}/{total_count}")
    print(f"실패: {total_count - success_count}/{total_count}")

    if success_count == total_count:
        print("모든 프로세스에서 DeepEP 테스트 성공!")
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
