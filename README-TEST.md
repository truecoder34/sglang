Testing guide for the Arm machine
1. Build
sudo apt-get install -y gcc g++ make cmake libnuma-dev numactl libtbb-dev google-perftools
python3 -m venv ~/venv-sgl && source ~/venv-sgl/bin/activate
pip install --upgrade pip uv
cat > ~/venv-sgl/uv.toml <<'EOF'
[[index]]
name = "torch"
url = "https://download.pytorch.org/whl/cpu"
EOF
export UV_CONFIG_FILE=~/venv-sgl/uv.toml

git clone <your fork URL> ~/sglang-fork/sglang   # or pull into an existing clone
cd ~/sglang-fork/sglang
git checkout arm-int8-wei-repack

(cd python && cp pyproject_cpu.toml pyproject.toml && uv pip install -e . && git checkout -- pyproject.toml)

export CMAKE_BUILD_PARALLEL_LEVEL=$(nproc)
(cd python/sglang/kernels/aot && cp pyproject_cpu.toml pyproject.toml && uv pip install -v . && git checkout -- pyproject.toml)

First two sanity checks — confirm the hardware actually has what the kernel needs, and confirm the new op registered:

grep -o -w -e i8mm -e bf16 -e asimddp /proc/cpuinfo | sort -u
python3 -c "import torch, sgl_kernel; print(hasattr(torch.ops.sgl_kernel, 'convert_weight_packed_i8mm'))"   # expect True

If i8mm/bf16 aren't in that grep output, stop — this is Graviton2/Ampere Altra/Kunpeng920-class hardware, and per the plan's §8 table, even today's unpacked Arm INT8 path can't run there (SIGILL), independent of anything we built.

2. Correctness
python3 test/registered/cpu/arm64/test_gemm_int8.py
python3 test/registered/cpu/arm64/test_moe.py
cd test && python3 run_suite.py --hw cpu --suite base-b-test-cpu-arm64   # full Arm suite — catches any regression elsewhere

test_gemm_int8.py is the one that actually proves the point: TestConvertWeightPackedI8mm checks the pack op's bytes exactly, and TestInt8GemmPackedB.test_packed_vs_raw checks torch.equal(packed_out, raw_out) for every M % 4 ∈ {0, 1} case — if that assertion fails, something in op.h's packed-B kernels doesn't match the layout pack.cpp actually produces, and you should stop before benchmarking.

3. Performance

3a. Kernel microbenchmark (this is the plan's own script, Step 6a — keep it out of the repo, just run it and paste results into the PR description):

cat > ~/bench_arm64_int8_gemm.py <<'PYEOF'
import itertools, time, torch
import sgl_kernel  # noqa: F401
torch.manual_seed(0)
k = torch.ops.sgl_kernel
HAS_PACK = hasattr(k, "convert_weight_packed_i8mm")

def bench(fn, iters=30, warmup=5):
    for _ in range(warmup):
        fn()
    t = time.perf_counter()
    for _ in range(iters):
        fn()
    return (time.perf_counter() - t) / iters

SHAPES = {
    "llama3b.qkv": (5120, 3072), "llama3b.o": (3072, 3072),
    "llama3b.gate_up": (16384, 3072), "llama3b.down": (3072, 8192),
    "qwen3moe.qkv": (5120, 2048), "qwen3moe.o": (2048, 4096),
}
for (name, (N, K)), M in itertools.product(SHAPES.items(), [1, 4, 16, 64, 256, 1024]):
    x = torch.randn(M, K, dtype=torch.bfloat16)
    w = torch.randint(-128, 128, (N, K), dtype=torch.int8)
    s = torch.rand(N) * 1e-2
    t_raw = bench(lambda: k.int8_scaled_mm_with_quant(x, w, s, None, torch.bfloat16, False))
    line = f"{name:16s} M={M:5d}  raw {t_raw*1e3:8.3f} ms  {2*M*N*K/t_raw/1e9:7.1f} GOP/s"
    if HAS_PACK:
        wp = k.convert_weight_packed_i8mm(w)
        t_pk = bench(lambda: k.int8_scaled_mm_with_quant(x, wp, s, None, torch.bfloat16, True))
        line += f"  packed {t_pk*1e3:8.3f} ms  speedup {t_raw/t_pk:5.2f}x"
    print(line, flush=True)
PYEOF
numactl -C 0-15 -m 0 python3 ~/bench_arm64_int8_gemm.py   # adjust the core range to your box

What to look for: speedup should be flat at ~1.0× for M=1 (no zip was happening there before either — if it regresses here, that's the gemv tail path misbehaving) and climb toward ~1.1-1.3× as M grows past 4, per the plan's §8 model. Pin threads and take the median of a few runs — the first run on a cold cache is noisy.

3b. End-to-end A/B on the same binary — this is the one that actually matters for a PR, since it's measured on the real serving path, not a microbenchmark:

export SGLANG_CPU_OMP_THREADS_BIND="0-31"   # adjust to your cores
for P in 0 1; do
  SGLANG_CPU_ARM64_INT8_PREPACK=$P python3 -m sglang.benchmark.one_batch \
    --model-path RedHatAI/Llama-3.2-3B-quantized.w8a8 --quantization w8a8_int8 \
    --device cpu --trust-remote-code --disable-radix \
    --batch-size 1 16 64 --input-len 512 --output-len 32
done

SGLANG_CPU_ARM64_INT8_PREPACK is the kill switch from Step 4 — toggling it 0/1 on the same binary is what makes this a clean A/B instead of a cross-build comparison. Report prefill TTFT and decode tok/s for each batch size. Repeat with nytopop/Qwen3-30B-A3B.w8a8 for the MoE path if you have ~64GB RAM free.

3c. Accuracy — confirms the packed path didn't silently change answers beyond the expected last-bit noise (plan §7 Step 6c, D4):

# launch sglang serve once with SGLANG_CPU_ARM64_INT8_PREPACK=0, once with =1
python3 -m sglang.test.few_shot_gsm8k --num-questions 200 --parallel 8

Expect the same score within ±1-2 points between the two runs. A bigger drift means the packed path is producing materially different numbers, not just last-bit noise from the M%4∈{2,3} rows (D4 already tells you those rows are allowed to differ at the last bit — that's expected, not a bug).