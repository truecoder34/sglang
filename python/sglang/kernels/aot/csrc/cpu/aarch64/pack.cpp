#include "../common.h"

// Pre-pack int8 weights for op::i8mm_matmul_packed_b
//   packed[nb][ks][p][h][ch][b] = weight[8 * nb + 2 * p + ch][16 * ks + 8 * h + b]
//   nb: 8-channel block, ks: 16-byte K step, p: channel pair, h: K half (zip1/zip2),
//   ch: channel within the pair, b: byte.
at::Tensor convert_weight_packed_i8mm(at::Tensor& weight) {
  CHECK_INPUT(weight);
  TORCH_CHECK(
      weight.scalar_type() == at::kChar,
      "convert_weight_packed_i8mm: expect int8 weight, got ",
      weight.scalar_type());
  TORCH_CHECK(
      weight.dim() == 2 || weight.dim() == 3,
      "convert_weight_packed_i8mm: expect 2D or 3D weight, got ",
      weight.dim(),
      "D");

  const int64_t OC = weight.size(-2);
  const int64_t IC = weight.size(-1);
  TORCH_CHECK(
      OC % 8 == 0 && IC % 16 == 0,
      "convert_weight_packed_i8mm: expect OC % 8 == 0 and IC % 16 == 0, got OC=",
      OC,
      ", IC=",
      IC);

  const int64_t rows = weight.numel() / IC;  // E * OC for 3D MoE weights
  return weight.view({rows / 8, 4, 2, IC / 16, 2, 8})
      .permute({0, 3, 1, 4, 2, 5})
      .contiguous()
      .view(weight.sizes());
}