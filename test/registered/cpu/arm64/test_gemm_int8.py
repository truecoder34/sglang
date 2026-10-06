"""Arm64 INT8 dense GEMM test for the i8mm weight pre-pack (convert_weight_packed_i8mm).

Covers aarch64/pack.cpp (byte-exact permute) and the packed-B kernels in aarch64/op.h
(i8mm_tile_packed_b, sdot_gemv_packed_b) via int8_scaled_mm_with_quant(..., is_vnni=True).
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=15, suite="base-b-test-cpu-arm64")

import itertools
import unittest

import torch

from sglang.test.cpu_test_utils import precision
from sglang.test.test_utils import CustomTestCase

kernel = torch.ops.sgl_kernel

torch.manual_seed(129)


def _ref_pack(w: torch.Tensor) -> torch.Tensor:
    oc, ic = w.shape[-2], w.shape[-1]
    rows = w.numel() // ic
    return (
        w.view(rows // 8, 4, 2, ic // 16, 2, 8)
        .permute(0, 3, 1, 4, 2, 5)
        .reshape(w.shape)
    )


class TestConvertWeightPackedI8mm(CustomTestCase):
    def test_exact_bytes_2d(self):
        w = torch.randint(-128, 128, (64, 32), dtype=torch.int8)
        packed = kernel.convert_weight_packed_i8mm(w)
        self.assertEqual(packed.shape, w.shape)
        self.assertEqual(packed.dtype, w.dtype)
        self.assertTrue(torch.equal(packed, _ref_pack(w)))

    def test_exact_bytes_3d_moe(self):
        w = torch.randint(-128, 128, (4, 32, 32), dtype=torch.int8)  # [E, 2N, K]
        packed = kernel.convert_weight_packed_i8mm(w)
        self.assertEqual(packed.shape, w.shape)
        self.assertTrue(torch.equal(packed, _ref_pack(w)))

    def test_rejects_bad_k(self):
        w = torch.randint(-128, 128, (8, 17), dtype=torch.int8)  # K % 16 != 0
        with self.assertRaises(RuntimeError):
            kernel.convert_weight_packed_i8mm(w)

    def test_rejects_bad_oc(self):
        w = torch.randint(-128, 128, (4, 16), dtype=torch.int8)  # OC % 8 != 0
        with self.assertRaises(RuntimeError):
            kernel.convert_weight_packed_i8mm(w)


class TestInt8GemmPackedB(CustomTestCase):
    M = [1, 2, 3, 4, 5, 7, 8, 17, 64]
    NK = [(16, 16), (64, 64), (256, 2048)]
    HAS_BIAS = [False, True]

    def _packed_vs_raw(self, M, N, K, has_bias):
        dtype = torch.bfloat16
        A = torch.randn((M, K), dtype=dtype) / 10
        Bq = torch.randint(-128, 128, (N, K), dtype=torch.int8)
        Bs = torch.rand(N) * 1e-2
        bias = torch.randn(N) if has_bias else None

        raw_out = kernel.int8_scaled_mm_with_quant(A, Bq, Bs, bias, dtype, False)

        Bq_packed = kernel.convert_weight_packed_i8mm(Bq)
        packed_out = kernel.int8_scaled_mm_with_quant(
            A, Bq_packed, Bs, bias, dtype, True
        )

        atol = rtol = precision[raw_out.dtype]
        torch.testing.assert_close(packed_out, raw_out, atol=atol, rtol=rtol)

        if M % 4 in (0, 1):
            self.assertTrue(torch.equal(packed_out, raw_out))

    def test_packed_vs_raw(self):
        for M, (N, K), has_bias in itertools.product(self.M, self.NK, self.HAS_BIAS):
            with self.subTest(M=M, N=N, K=K, has_bias=has_bias):
                self._packed_vs_raw(M, N, K, has_bias)


if __name__ == "__main__":
    unittest.main()