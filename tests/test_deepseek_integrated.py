#!/usr/bin/env python3
"""
통합된 DeepSeek MoE + DeepEP 테스트
기존 deepseek.py에 통합된 DeepEP 기능을 테스트
"""

import logging
import os
import socket
import sys
import time
from typing import Any, Dict

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

# DeepSeek MoE (DeepEP 통합) import
sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(__file__), "..", "..", "..", "model_runner", "model", "deepseek"
    ),
)
from deepseek import create_deepseek_moe_with_deepep

# 로깅 설정
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class MockDeepSeekConfig:
    """DeepSeek MoE 16B 설정 Mock"""

    def __init__(self, small_size: bool = True):
        if small_size:
            # 테스트용 작은 설정
            self.vocab_size = 1000
            self.hidden_size = 512
            self.intermediate_size = 1024
            self.num_hidden_layers = 4
            self.num_attention_heads = 8
            self.num_key_value_heads = 8
            self.max_position_embeddings = 1024
            self.n_routed_experts = 8
            self.num_experts_per_tok = 2
            self.n_shared_experts = 1
            self.moe_intermediate_size = 256
        else:
            # 실제 크기 설정
            self.vocab_size = 102400
            self.hidden_size = 4096
            self.intermediate_size = 11008
            self.num_hidden_layers = 60
            self.num_attention_heads = 32
            self.num_key_value_heads = 32
            self.max_position_embeddings = 4096
            self.n_routed_experts = 64
            self.num_experts_per_tok = 6
            self.n_shared_experts = 2
            self.moe_intermediate_size = 1408

        # 공통 설정
        self.rms_norm_eps = 1e-6
        self.first_k_dense_replace = 1
        self.moe_layer_freq = 1
        self.scoring_func = "softmax"
        self.norm_topk_prob = True
        self.hidden_act = "silu"
        self.rms_norm_method = "triton"
        self.fused_rms_norm_method = "triton"
        self.attention_bias = False
        self.attention_dropout = 0.0
        self.max_seq_length = 1024 if small_size else 4096
        self.use_cache = True
        self.rope_theta = 10000.0


def find_free_port():
    """사용 가능한 포트 찾기"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def test_single_gpu():
    """단일 GPU 테스트 (DeepEP 없이)"""
    logger.info("🚀 ===== 단일 GPU 테스트 (DeepEP 없이) =====")

    try:
        # 작은 설정으로 테스트
        config = MockDeepSeekConfig(small_size=True)

        logger.info(f"📋 테스트 설정:")
        logger.info(f"   - hidden_size: {config.hidden_size}")
        logger.info(f"   - num_hidden_layers: {config.num_hidden_layers}")
        logger.info(f"   - n_routed_experts: {config.n_routed_experts}")
        logger.info(f"   - num_experts_per_tok: {config.num_experts_per_tok}")

        # 모델 생성 (DeepEP 없이, world_size=1)
        logger.info("🔧 DeepSeek MoE 모델 생성 중 (DeepEP 없이)...")
        model = create_deepseek_moe_with_deepep(
            config=config,
            pad_token_id=0,
            world_size=1,  # 단일 프로세스
            deepep_config=None,  # DeepEP 비활성화
        )

        # 디바이스 설정
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        logger.info(f"✅ 모델을 {device}로 이동")

        # DeepEP 상태 확인
        deepep_status = model.get_deepep_status()
        logger.info(f"🔍 DeepEP 상태: {deepep_status}")

        # KV cache 초기화
        batch_size = 2
        seq_len = 32
        logger.info(
            f"⚙️ KV cache 초기화 중: batch_size={batch_size}, max_seq_len={seq_len}"
        )
        model.model.initialize_kv_caches(
            batch_size=batch_size,
            max_seq_length=seq_len,
            num_kv_entries_per_page=16,
            kv_cache_memory_limit=1024 * 1024 * 512,  # 512MB 제한
        )

        # 테스트 데이터 생성
        input_ids = torch.randint(
            0, config.vocab_size, (batch_size, seq_len), device=device
        )

        logger.info(f"🧪 테스트 데이터: batch_size={batch_size}, seq_len={seq_len}")

        # Forward pass 테스트
        logger.info("🔄 Forward pass 테스트 시작...")
        model.eval()

        # Mock attention metadata 생성
        class MockAttentionMeta:
            def __init__(self):
                self.mode = "prefill"
                self.is_prompt = True
                self.request_ids = [0, 1]  # batch_size만큼 request ID 생성
                self.context_lens = [seq_len] * batch_size  # 각 요청의 컨텍스트 길이
                self.seq_lens = [seq_len] * batch_size  # 각 요청의 시퀀스 길이
                # KV cache page plans (올바른 형식: [(page_idx, n_tokens)])
                self.page_plans = [
                    [(i, 1) for i in range(seq_len)] for _ in range(batch_size)
                ]
                # Final offsets (각 요청의 마지막 위치)
                self.final_offsets = [seq_len - 1] * batch_size

        mock_meta_attention = MockAttentionMeta()

        with torch.no_grad():
            start_time = time.time()
            outputs = model(input_ids, meta_attention=mock_meta_attention)
            forward_time = time.time() - start_time

        logger.info(f"✅ Forward pass 완료: {forward_time:.4f}s")
        logger.info(f"📏 출력 shape: {outputs.shape}")

        # 출력 검증
        expected_shape = (batch_size, seq_len, config.vocab_size)
        if outputs.shape == expected_shape:
            logger.info("✅ 출력 shape 검증 성공")
        else:
            logger.error(f"❌ 출력 shape 불일치: {outputs.shape} != {expected_shape}")
            return False

        # NaN/Inf 검사
        if torch.isnan(outputs).any():
            logger.error("❌ 출력에 NaN 값 발견")
            return False
        if torch.isinf(outputs).any():
            logger.error("❌ 출력에 Inf 값 발견")
            return False
        logger.info("✅ NaN/Inf 검사 통과")

        logger.info("🎉 단일 GPU 테스트 성공!")
        return True

    except Exception as e:
        logger.error(f"❌ 단일 GPU 테스트 실패: {type(e).__name__}: {e}")
        import traceback

        traceback.print_exc()
        return False


def setup_process(rank: int, world_size: int, master_addr: str, master_port: int):
    """개별 프로세스 설정"""
    logger.info(f"🚀 [Rank {rank}] 프로세스 시작")

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
    os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"  # GPU 1, 2 사용

    # CUDA 디바이스 설정
    if torch.cuda.is_available() and rank < torch.cuda.device_count():
        torch.cuda.set_device(rank)
        logger.info(f"✅ [Rank {rank}] CUDA 디바이스 설정: cuda:{rank}")
    else:
        logger.warning(f"⚠️ [Rank {rank}] CUDA 디바이스 부족, CPU 사용")


def test_multi_gpu_worker(
    rank: int, world_size: int, master_addr: str, master_port: int
):
    """멀티 GPU 테스트 워커 (DeepEP 포함)"""
    try:
        setup_process(rank, world_size, master_addr, master_port)

        # 분산 환경 초기화
        logger.info(f"🌐 [Rank {rank}] 분산 환경 초기화 중...")
        dist.init_process_group(
            backend="nccl",
            init_method=f"tcp://{master_addr}:{master_port}",
            world_size=world_size,
            rank=rank,
        )
        logger.info(f"✅ [Rank {rank}] 분산 환경 초기화 완료")

        # DeepSeek 설정 생성 (작은 크기)
        config = MockDeepSeekConfig(small_size=True)

        # DeepEP 설정 생성
        deepep_config = {
            "max_tokens": 512,  # 작게 시작
            "top_k": config.num_experts_per_tok,
            "num_experts": config.n_routed_experts,
            "hidden_size": config.hidden_size,
        }

        logger.info(f"📋 [Rank {rank}] 설정:")
        logger.info(
            f"   - DeepSeek: {config.n_routed_experts} experts, {config.num_experts_per_tok} top-k"
        )
        logger.info(f"   - DeepEP: {deepep_config}")

        # DeepSeek MoE + DeepEP 모델 생성
        logger.info(f"🔧 [Rank {rank}] DeepSeek MoE + DeepEP 모델 생성 중...")
        model = create_deepseek_moe_with_deepep(
            config=config,
            pad_token_id=0,
            world_size=world_size,
            deepep_config=deepep_config,
        )

        # GPU로 이동
        device = f"cuda:{rank}" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        logger.info(f"✅ [Rank {rank}] 모델을 {device}로 이동")

        # DeepEP 초기화
        logger.info(f"⚙️ [Rank {rank}] DeepEP 초기화 중...")
        deepep_success = model.initialize_deepep(rank=rank)
        logger.info(f"🔍 [Rank {rank}] DeepEP 초기화 결과: {deepep_success}")

        # DeepEP 상태 확인
        deepep_status = model.get_deepep_status()
        logger.info(f"📊 [Rank {rank}] DeepEP 상태: {deepep_status}")

        # KV cache 초기화
        batch_size = 2
        seq_len = 16  # 작게 시작
        logger.info(
            f"⚙️ [Rank {rank}] KV cache 초기화 중: batch_size={batch_size}, max_seq_len={seq_len}"
        )
        model.model.initialize_kv_caches(
            batch_size=batch_size,
            max_seq_length=seq_len,
            num_kv_entries_per_page=16,
            kv_cache_memory_limit=1024 * 1024 * 256,  # 256MB 제한
        )

        # 동기화
        dist.barrier()

        # 테스트 데이터 생성
        input_ids = torch.randint(
            0, config.vocab_size, (batch_size, seq_len), device=device
        )

        logger.info(
            f"🧪 [Rank {rank}] 테스트 시작: batch_size={batch_size}, seq_len={seq_len}"
        )

        # Forward pass 테스트
        # Mock attention metadata 생성
        class MockAttentionMeta:
            def __init__(self):
                self.mode = "prefill"
                self.is_prompt = True
                self.request_ids = [0, 1]  # batch_size만큼 request ID 생성
                self.context_lens = [seq_len] * batch_size  # 각 요청의 컨텍스트 길이
                self.seq_lens = [seq_len] * batch_size  # 각 요청의 시퀀스 길이
                # KV cache page plans (올바른 형식: [(page_idx, n_tokens)])
                self.page_plans = [
                    [(i, 1) for i in range(seq_len)] for _ in range(batch_size)
                ]
                # Final offsets (각 요청의 마지막 위치)
                self.final_offsets = [seq_len - 1] * batch_size

        mock_meta_attention = MockAttentionMeta()

        with torch.no_grad():
            start_time = time.time()

            logger.info(f"🔄 [Rank {rank}] Forward pass 시작...")
            outputs = model(input_ids, meta_attention=mock_meta_attention)

            forward_time = time.time() - start_time
            logger.info(f"✅ [Rank {rank}] Forward pass 완료: {forward_time:.4f}s")
            logger.info(f"📏 [Rank {rank}] 출력 shape: {outputs.shape}")

        # 동기화
        dist.barrier()

        logger.info(f"🎉 [Rank {rank}] 테스트 완료!")
        return True

    except Exception as e:
        logger.error(f"❌ [Rank {rank}] 테스트 실패: {type(e).__name__}: {e}")
        import traceback

        traceback.print_exc()
        return False

    finally:
        # 분산 환경 정리
        try:
            if dist.is_initialized():
                dist.destroy_process_group()
        except Exception:
            pass


def test_multi_gpu():
    """멀티 GPU 테스트 (DeepEP 포함)"""
    logger.info("🚀 ===== 멀티 GPU 테스트 (DeepEP 포함) =====")

    # GPU 확인
    num_gpus = torch.cuda.device_count()
    logger.info(f"🔍 사용 가능한 GPU 개수: {num_gpus}")

    if num_gpus < 2:
        logger.warning("⚠️ 최소 2개 GPU가 필요합니다. 멀티 GPU 테스트 스킵")
        return True

    # 설정
    world_size = 2
    master_addr = "localhost"
    master_port = find_free_port()

    logger.info(f"🌐 마스터 주소: {master_addr}:{master_port}")

    # 멀티프로세싱 설정
    mp.set_start_method("spawn", force=True)

    # 프로세스 생성
    processes = []
    for rank in range(world_size):
        p = mp.Process(
            target=test_multi_gpu_worker,
            args=(rank, world_size, master_addr, master_port),
        )
        p.start()
        processes.append(p)
        logger.info(f"✅ 프로세스 {rank} 시작")

    # 프로세스 완료 대기
    results = []
    for rank, p in enumerate(processes):
        p.join(timeout=120)  # 2분 타임아웃
        if p.is_alive():
            logger.warning(f"⚠️ 프로세스 {rank} 타임아웃, 강제 종료")
            p.terminate()
            p.join()
            results.append(False)
        else:
            results.append(p.exitcode == 0)
            logger.info(f"✅ 프로세스 {rank} 완료 (exitcode={p.exitcode})")

    # 결과
    success_count = sum(results)
    logger.info(f"🎯 멀티 GPU 테스트 결과: {success_count}/{len(results)} 성공")

    return success_count == len(results)


def main():
    """메인 함수"""
    logger.info("🚀 ===== 통합된 DeepSeek MoE + DeepEP 테스트 =====")

    success_count = 0
    total_tests = 2

    # 1. 단일 GPU 테스트
    if test_single_gpu():
        success_count += 1
        logger.info("✅ 테스트 1/2 성공: 단일 GPU 테스트")
    else:
        logger.error("❌ 테스트 1/2 실패: 단일 GPU 테스트")

    # 2. 멀티 GPU 테스트
    if test_multi_gpu():
        success_count += 1
        logger.info("✅ 테스트 2/2 성공: 멀티 GPU 테스트")
    else:
        logger.error("❌ 테스트 2/2 실패: 멀티 GPU 테스트")

    # 결과
    logger.info(f"\n🎯 최종 결과: {success_count}/{total_tests} 테스트 성공")

    if success_count == total_tests:
        logger.info("🎉 모든 테스트 성공!")
        return True
    else:
        logger.error("❌ 일부 테스트 실패")
        return False


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        logger.warning("\n⚠️ 사용자 중단")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\n❌ 오류: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
