"""
Benchmark to measure overhead of conditionals (backend selection, gram vs standard choice)
for small matrices where the actual compute is fast and overhead may be significant.

Compares the full GramNewtonSchulz (which dynamically selects backend and algorithm)
against hardcoded versions that use a fixed backend (quack or pytorch) with no conditionals.

This helps determine whether torch.compile eliminates the branching cost or not.
"""

import torch
from gram_newton_schulz import GramNewtonSchulz, POLAR_EXPRESS_COEFFICIENTS
from quack.gemm_interface import gemm_symmetric, gemm, gemm_add


def benchmark_fn(fn, warmup=50, iters=200):
    """Benchmark a function using CUDA events for accurate GPU timing."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]

    for i in range(iters):
        start_events[i].record()
        fn()
        end_events[i].record()

    torch.cuda.synchronize()
    times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
    return times


# ============================================================================
# Hardcoded Quack backend versions (no conditionals)
# ============================================================================

def make_hardcoded_gram_quack(coefficients, reset_iterations):
    """Hardcoded gram NS using quack backend, no conditionals."""
    @torch.compile(fullgraph=True, mode="reduce-overhead")
    def hardcoded(X: torch.Tensor) -> torch.Tensor:
        X = X.to(torch.float32)
        X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
        X = X.to(torch.float16)

        R = gemm_symmetric(X, X.mT)
        batch_size = R.size(0)
        I = torch.eye(R.size(-1), device=X.device, dtype=X.dtype).unsqueeze(0).expand(batch_size, -1, -1).contiguous()
        Q = None

        for i, (a, b, c) in enumerate(coefficients):
            if i in reset_iterations and i != 0:
                X = gemm(Q, X, tuned=False)
                R = gemm_symmetric(X, X.mT)
                Q = None

            Z = gemm_symmetric(R, R, C=R, alpha=c, beta=b)
            if i == 0 or i in reset_iterations:
                Q = Z + a * I
            else:
                Q = gemm_symmetric(Q, Z, C=Q, beta=a)
            if i < len(coefficients) - 1 and i + 1 not in reset_iterations:
                RZ = gemm_symmetric(R, Z, C=R, beta=a)
                R = gemm_symmetric(Z, RZ, C=RZ, beta=a)

        X = gemm(Q, X, tuned=False)
        return X

    return hardcoded


def make_hardcoded_standard_quack(coefficients):
    """Hardcoded standard NS using quack backend, no conditionals."""
    @torch.compile(fullgraph=True, mode="reduce-overhead")
    def hardcoded(X: torch.Tensor) -> torch.Tensor:
        X = X.to(torch.float32)
        X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
        X = X.to(torch.float16)

        for a, b, c in coefficients:
            A = gemm_symmetric(X, X.mT)
            B = gemm_symmetric(A, A, C=A, alpha=c, beta=b)
            X = gemm_add(B, X, C=X, beta=a, tuned=False)

        return X

    return hardcoded


# ============================================================================
# Hardcoded PyTorch backend versions (no conditionals)
# ============================================================================

def make_hardcoded_gram_pytorch(coefficients, reset_iterations):
    """Hardcoded gram NS using pytorch backend, no conditionals."""
    @torch.compile(fullgraph=True, mode="reduce-overhead")
    def hardcoded(X: torch.Tensor) -> torch.Tensor:
        X = X.to(torch.float32)
        X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
        X = X.to(torch.float16)

        R = X @ X.mT
        batch_size = R.size(0)
        I = torch.eye(R.size(-1), device=X.device, dtype=X.dtype).unsqueeze(0).expand(batch_size, -1, -1).contiguous()
        Q = None

        for i, (a, b, c) in enumerate(coefficients):
            if i in reset_iterations and i != 0:
                X = Q @ X
                R = X @ X.mT
                Q = None

            Z = torch.baddbmm(R, R, R, alpha=c, beta=b)
            if i == 0 or i in reset_iterations:
                Q = Z + a * I
            else:
                Q = torch.baddbmm(Q, Q, Z, beta=a)
            if i < len(coefficients) - 1 and i + 1 not in reset_iterations:
                RZ = torch.baddbmm(R, R, Z, beta=a)
                R = torch.baddbmm(RZ, Z, RZ, beta=a)

        X = Q @ X
        return X

    return hardcoded


def make_hardcoded_standard_pytorch(coefficients):
    """Hardcoded standard NS using pytorch backend, no conditionals."""
    @torch.compile(fullgraph=True, mode="reduce-overhead")
    def hardcoded(X: torch.Tensor) -> torch.Tensor:
        X = X.to(torch.float32)
        X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
        X = X.to(torch.float16)

        for a, b, c in coefficients:
            A = X @ X.mT
            B = torch.baddbmm(A, A, A, alpha=c, beta=b)
            X = torch.baddbmm(X, B, X, beta=a)

        return X

    return hardcoded


# ============================================================================
# Main benchmark
# ============================================================================

def print_header(title):
    print(f"\n{'=' * 90}")
    print(f"  {title}")
    print(f"{'=' * 90}")


def print_table_header():
    print(f"{'Shape':<18} {'Full (ms)':<11} {'Quack (ms)':<12} {'PyTorch (ms)':<14} {'Overhead':<15}")
    print("-" * 70)


def main():
    device = "cuda"
    # Small matrices where overhead may be significant
    BATCH = 128
    shapes = [
        (BATCH, 256, 256),
        (BATCH, 256, 512),
        (BATCH, 256, 1024),
        (BATCH, 256, 2048),
        (BATCH, 256, 4096),
        (BATCH, 512, 4096),
        (BATCH, 256, 256),
        (BATCH, 512, 512),
        (BATCH, 1024, 1024),
    ]

    coefficients = POLAR_EXPRESS_COEFFICIENTS
    reset_iterations = [2]

    print_header("Conditional Overhead Benchmark: Full GramNewtonSchulz vs Hardcoded Backends")
    print("\nFull GramNewtonSchulz has runtime conditionals that choose:")
    print("  1. Backend: quack kernels (if min dim > 256) vs pytorch (otherwise)")
    print("  2. Algorithm: gram (if non-square) vs standard (if square)")
    print("  3. Transpose: if M > N")
    print("\nHardcoded versions skip all conditionals and use a fixed backend + algorithm.")

    # Build the full GramNewtonSchulz (with all conditionals, uses kernels)
    full_ns = GramNewtonSchulz(
        ns_coefficients=coefficients,
        gram_newton_schulz_reset_iterations=reset_iterations,
        ns_use_kernels=True,
    )

    # Build hardcoded versions
    hardcoded_gram_quack = make_hardcoded_gram_quack(coefficients, set(reset_iterations))
    hardcoded_gram_pytorch = make_hardcoded_gram_pytorch(coefficients, set(reset_iterations))
    hardcoded_standard_quack = make_hardcoded_standard_quack(coefficients)
    hardcoded_standard_pytorch = make_hardcoded_standard_pytorch(coefficients)

    # --- Gram path (non-square matrices) ---
    print_header("GRAM Newton-Schulz path (non-square matrices)")
    print_table_header()

    for shape in shapes:
        if shape[-2] == shape[-1]:
            continue

        X = torch.randn(shape, device=device)

        full_times = benchmark_fn(lambda: full_ns(X))
        quack_times = benchmark_fn(lambda: hardcoded_gram_quack(X))
        pytorch_times = benchmark_fn(lambda: hardcoded_gram_pytorch(X))

        full_med = sorted(full_times)[len(full_times) // 2]
        quack_med = sorted(quack_times)[len(quack_times) // 2]
        pytorch_med = sorted(pytorch_times)[len(pytorch_times) // 2]

        best_med = min(quack_med, pytorch_med)
        overhead = ((full_med - best_med) / best_med * 100) if best_med > 0 else 0

        print(f"{str(shape):<18} {full_med:<11.4f} {quack_med:<12.4f} {pytorch_med:<14.4f} {overhead:<+14.1f}%")

    # --- Standard path (square matrices) ---
    print_header("STANDARD Newton-Schulz path (square matrices)")
    print_table_header()

    for shape in shapes:
        if shape[-2] != shape[-1]:
            continue

        X = torch.randn(shape, device=device)

        full_times = benchmark_fn(lambda: full_ns(X))
        quack_times = benchmark_fn(lambda: hardcoded_standard_quack(X))
        pytorch_times = benchmark_fn(lambda: hardcoded_standard_pytorch(X))

        full_med = sorted(full_times)[len(full_times) // 2]
        quack_med = sorted(quack_times)[len(quack_times) // 2]
        pytorch_med = sorted(pytorch_times)[len(pytorch_times) // 2]

        best_med = min(quack_med, pytorch_med)
        overhead = ((full_med - best_med) / best_med * 100) if best_med > 0 else 0

        print(f"{str(shape):<18} {full_med:<11.4f} {quack_med:<12.4f} {pytorch_med:<14.4f} {overhead:<+14.1f}%")

    # # --- Compiled vs uncompiled ---
    # print_header("Effect of torch.compile (full GramNewtonSchulz with conditionals)")
    # print(f"{'Shape':<18} {'Compiled (ms)':<14} {'Uncompiled (ms)':<17} {'Speedup':<10}")
    # print("-" * 59)

    # uncompiled_ns = GramNewtonSchulz(
    #     ns_coefficients=coefficients,
    #     gram_newton_schulz_reset_iterations=reset_iterations,
    #     ns_use_kernels=True,
    #     compile_kwargs=None,
    # )

    # for shape in shapes:
    #     X = torch.randn(shape, device=device)

    #     compiled_times = benchmark_fn(lambda: full_ns(X))
    #     uncompiled_times = benchmark_fn(lambda: uncompiled_ns(X))

    #     comp_med = sorted(compiled_times)[len(compiled_times) // 2]
    #     uncomp_med = sorted(uncompiled_times)[len(uncompiled_times) // 2]
    #     speedup = uncomp_med / comp_med if comp_med > 0 else 0

    #     print(f"{str(shape):<18} {comp_med:<14.4f} {uncomp_med:<17.4f} {speedup:<10.2f}x")

    print("\n" + "=" * 90)
    print("INTERPRETATION:")
    print("  - 'Ovhd vs Quack/PyTorch' shows how much slower the full class is vs hardcoded.")
    print("  - Positive % = conditionals add overhead. Negative % = full class is somehow faster.")
    print("  - For small matrices (min dim <= 256), full class uses PyTorch backend.")
    print("  - For larger matrices (min dim > 256), full class uses quack backend.")
    print("  - Large compile speedup = Python overhead dominates without torch.compile.")
    print("=" * 90)


if __name__ == "__main__":
    main()
