"""Arm64 MoE test.

Tests fused_experts_cpu with W8A8 INT8 quantization, which is supported
on Arm64 via aarch64/moe.cpp (PR #16045). Additional quantization paths
(BF16, INT4) will be added here as Arm kernels land.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-b-test-cpu-arm64")

import itertools
import math
import platform
import unittest

import torch

from sglang.srt.layers.amx_utils import CPUQuantMethod
from sglang.test.cpu_test_utils import precision, torch_w8a8_per_column_fused_moe
from sglang.test.test_utils import CustomTestCase

kernel = torch.ops.sgl_kernel
IS_ARM64 = platform.machine().lower() in ("aarch64", "arm64")

torch.manual_seed(128)


class TestFusedExpertsInt8(CustomTestCase):
    M = [1, 3, 5, 6, 32, 64]
    N = [256, 512]
    K = [256, 512]
    E = [8]
    topk = [4]

    def _int8_moe(self, M, N, K, E, topk):
        dtype = torch.bfloat16
        int8_factor_for_scale = 1e-2
        int8_max = 127
        int8_min = -128

        a = torch.randn((M, K), dtype=dtype) / math.sqrt(K)
        w1_fp32 = (torch.rand((E, 2 * N, K), dtype=torch.float32) - 0.5) * 2
        w1 = (w1_fp32 * int8_max).clamp(min=int8_min, max=int8_max).to(torch.int8)
        w2_fp32 = (torch.rand((E, K, N), dtype=torch.float32) - 0.5) * 2
        w2 = (w2_fp32 * int8_max).clamp(min=int8_min, max=int8_max).to(torch.int8)
        w1_s = torch.rand(E, 2 * N, device=w1_fp32.device) * int8_factor_for_scale
        w2_s = torch.rand(E, K, device=w2_fp32.device) * int8_factor_for_scale

        score = torch.randn((M, E), dtype=dtype)
        score = torch.softmax(score, dim=-1, dtype=torch.float32)
        topk_weight, topk_ids = torch.topk(score, topk)

        ref_out = torch_w8a8_per_column_fused_moe(
            a, w1, w2, w1_s, w2_s, topk_weight, topk_ids, topk
        )

        inplace = not IS_ARM64

        def run(prepack, pack_fn):
            packed_w1 = pack_fn(w1) if prepack else w1
            packed_w2 = pack_fn(w2) if prepack else w2
            return kernel.fused_experts_cpu(
                a,
                packed_w1,
                packed_w2,
                topk_weight,
                topk_ids.to(torch.int32),
                inplace,
                CPUQuantMethod.INT8_W8A8,
                w1_s,
                w2_s,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                prepack,
            )

        atol = rtol = precision[ref_out.dtype]
        if IS_ARM64:
            atol = rtol = 0.03
        elif M > 35:
            atol = rtol = 0.02

        if IS_ARM64:
            raw_out = run(prepack=False, pack_fn=kernel.convert_weight_packed_i8mm)
            torch.testing.assert_close(ref_out, raw_out, atol=atol, rtol=rtol)

            packed_out = run(prepack=True, pack_fn=kernel.convert_weight_packed_i8mm)
            torch.testing.assert_close(ref_out, packed_out, atol=atol, rtol=rtol)
            torch.testing.assert_close(packed_out, raw_out, atol=atol, rtol=rtol)
        else:
            out = run(prepack=True, pack_fn=kernel.convert_weight_packed)
            torch.testing.assert_close(ref_out, out, atol=atol, rtol=rtol)

if __name__ == "__main__":
    unittest.main()
