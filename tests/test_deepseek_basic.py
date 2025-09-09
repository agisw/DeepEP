#!/usr/bin/env python3
"""
DeepSeek MoE 기본 동작 테스트
DeepEP 없이 fallback 모드로 동작하는지 확인
"""

import logging
import os
import sys
import time
from typing import Any, Dict

import torch

# DeepSeek MoE + DeepEP 통합 모듈 import
sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(__file__), "..", "..", "..", "model_runner", "model", "deepseek"
    ),
)
from deepseek_with_deepep import create_deepseek_moe_with_deepep

# 로깅 설정
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class MockDeepSeekConfig:
    """DeepSeek MoE 16B 설정 Mock (작은 크기)"""

    def __init__(self):
        # 테스트용 작은 설정
        self.vocab_size = 1000
        self.hidden_size = 512
        self.intermediate_size = 1024
        self.num_hidden_layers = 4  # 작게 설정
        self.num_attention_heads = 8
        self.num_key_value_heads = 8
        self.max_position_embeddings = 1024
        self.rms_norm_eps = 1e-6

        # MoE 관련 설정 (작게 설정)
        self.n_routed_experts = 8  # 작게 설정
        self.num_experts_per_tok = 2  # 작게 설정
        self.n_shared_experts = 1  # 작게 설정
        self.moe_intermediate_size = 256  # 작게 설정
        self.first_k_dense_replace = 1
        self.moe_layer_freq = 1

        # Scoring 및 normalization
        self.scoring_func = "softmax"
        self.norm_topk_prob = True

        # Activation function
        self.hidden_act = "silu"

        # RMS Norm 설정
        self.rms_norm_method = "triton"
        self.fused_rms_norm_method = "triton"

        # Attention 설정
        self.attention_bias = False
        self.attention_dropout = 0.0

        # 기타 필요한 속성들
        self.max_seq_length = 1024
        self.use_cache = True
        self.rope_theta = 10000.0


def test_deepseek_basic():
    """DeepSeek MoE 기본 동작 테스트"""
    logger.info("🚀 ===== DeepSeek MoE 기본 동작 테스트 =====")

    try:
        # 설정 생성
        config = MockDeepSeekConfig()

        logger.info(f"📋 테스트 설정:")
        logger.info(f"   - hidden_size: {config.hidden_size}")
        logger.info(f"   - num_hidden_layers: {config.num_hidden_layers}")
        logger.info(f"   - n_routed_experts: {config.n_routed_experts}")
        logger.info(f"   - num_experts_per_tok: {config.num_experts_per_tok}")
        logger.info(f"   - n_shared_experts: {config.n_shared_experts}")

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

        # 모델 정보 출력
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"📊 모델 정보:")
        logger.info(f"   - 총 파라미터: {total_params:,}")
        logger.info(f"   - 학습 가능한 파라미터: {trainable_params:,}")

        # DeepEP 상태 확인
        deepep_status = model.get_deepep_status()
        logger.info(f"🔍 DeepEP 상태: {deepep_status}")

        # 테스트 데이터 생성
        batch_size = 4
        seq_len = 64
        input_ids = torch.randint(
            0, config.vocab_size, (batch_size, seq_len), device=device
        )

        logger.info(f"🧪 테스트 데이터: batch_size={batch_size}, seq_len={seq_len}")

        # Forward pass 테스트
        logger.info("🔄 Forward pass 테스트 시작...")
        model.eval()

        with torch.no_grad():
            start_time = time.time()
            outputs = model(input_ids)
            forward_time = time.time() - start_time

        logger.info(f"✅ Forward pass 완료: {forward_time:.4f}s")
        logger.info(f"📏 출력 shape: {outputs.shape}")
        logger.info(f"📏 예상 shape: {(batch_size, seq_len, config.vocab_size)}")

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

        # 여러 번 테스트 (일관성 확인)
        logger.info("🔄 일관성 테스트 시작 (3회 반복)...")
        times = []

        with torch.no_grad():
            for i in range(3):
                start_time = time.time()
                outputs = model(input_ids)
                end_time = time.time()
                times.append(end_time - start_time)
                logger.info(f"🔄 테스트 {i+1}/3: {times[-1]:.4f}s")

        # 통계 계산
        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)

        logger.info(f"📊 성능 통계:")
        logger.info(f"   - 평균: {avg_time:.4f}s")
        logger.info(f"   - 최소: {min_time:.4f}s")
        logger.info(f"   - 최대: {max_time:.4f}s")

        # 메모리 사용량 확인
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated(device) / (1024**2)
            memory_reserved = torch.cuda.memory_reserved(device) / (1024**2)
            logger.info(f"💾 GPU 메모리:")
            logger.info(f"   - 할당됨: {memory_allocated:.2f} MB")
            logger.info(f"   - 예약됨: {memory_reserved:.2f} MB")

        # 다른 크기로 테스트
        logger.info("🧪 다양한 크기로 테스트...")
        test_sizes = [
            (1, 16),  # 작은 크기
            (2, 32),  # 중간 크기
            (1, 128),  # 긴 시퀀스
        ]

        for batch_size, seq_len in test_sizes:
            logger.info(f"🔄 테스트 크기: batch_size={batch_size}, seq_len={seq_len}")
            input_ids = torch.randint(
                0, config.vocab_size, (batch_size, seq_len), device=device
            )

            with torch.no_grad():
                start_time = time.time()
                outputs = model(input_ids)
                test_time = time.time() - start_time

            expected_shape = (batch_size, seq_len, config.vocab_size)
            if outputs.shape == expected_shape:
                logger.info(
                    f"✅ 크기 {batch_size}x{seq_len} 테스트 성공: {test_time:.4f}s"
                )
            else:
                logger.error(
                    f"❌ 크기 {batch_size}x{seq_len} 테스트 실패: {outputs.shape} != {expected_shape}"
                )
                return False

        logger.info("🎉 모든 테스트 성공!")
        return True

    except Exception as e:
        logger.error(f"❌ 테스트 실패: {type(e).__name__}: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_moe_layer_individually():
    """MoE 레이어 개별 테스트"""
    logger.info("🔧 MoE 레이어 개별 테스트...")

    try:
        from deepseek_with_deepep import DeepSeekMoEWithDeepEP

        config = MockDeepSeekConfig()

        # MoE 레이어 생성 (DeepEP 없이)
        moe_layer = DeepSeekMoEWithDeepEP(config, deepep_wrapper=None)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        moe_layer = moe_layer.to(device)

        # 테스트 입력
        batch_size, seq_len = 2, 16
        hidden_states = torch.randn(
            batch_size, seq_len, config.hidden_size, device=device
        )

        logger.info(f"🧪 MoE 레이어 테스트: input_shape={hidden_states.shape}")

        # Forward pass
        with torch.no_grad():
            output = moe_layer(hidden_states)

        logger.info(f"✅ MoE 레이어 출력: {output.shape}")

        # 검증
        if output.shape == hidden_states.shape:
            logger.info("✅ MoE 레이어 개별 테스트 성공")
            return True
        else:
            logger.error(
                f"❌ MoE 레이어 출력 shape 불일치: {output.shape} != {hidden_states.shape}"
            )
            return False

    except Exception as e:
        logger.error(f"❌ MoE 레이어 개별 테스트 실패: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """메인 함수"""
    logger.info("🚀 ===== DeepSeek MoE 기본 테스트 시작 =====")

    success_count = 0
    total_tests = 2

    # 1. MoE 레이어 개별 테스트
    if test_moe_layer_individually():
        success_count += 1
        logger.info("✅ 테스트 1/2 성공: MoE 레이어 개별 테스트")
    else:
        logger.error("❌ 테스트 1/2 실패: MoE 레이어 개별 테스트")

    # 2. 전체 모델 테스트
    if test_deepseek_basic():
        success_count += 1
        logger.info("✅ 테스트 2/2 성공: 전체 모델 테스트")
    else:
        logger.error("❌ 테스트 2/2 실패: 전체 모델 테스트")

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
