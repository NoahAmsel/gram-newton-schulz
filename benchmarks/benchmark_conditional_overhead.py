"""
Benchmark to measure overhead of conditionals (backend selection, gram vs standard choice)
for small matrices where the actual compute is fast and overhead may be significant.

Compares:
1. Full GramNewtonSchulz (with all conditionals)
2. A "hardcoded" version that skips conditionals by calling the inner method directly

This helps isolate whether torch.compile eliminates the branching cost or not.
Uses the quack kernel backend for all tests.
"""

import torch
import time
from gram_newton_schulz import GramNewtonSchulz, StandardNewtonSchulz, POLAR_EXPRESS_COEFFICIENTS
from quack.gemm_interface import gemm_symmetric, gemm, gemm_add


def benchmark_fn(fn, warmup=50, iters=200):
    """Benchmark a function using CUDA events for accurate GPU timing."""
    # Warmup
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


def make_hardcoded_gram_ns(coefficients, reset_iterations):
    """
    Create a hardcoded version that does the same math as GramNewtonSchulz
    but without any runtime conditionals for backend/algorithm selection.
    Always uses quack backend, always does gram, never transposes.
    """
    @torch.compile(fullgraph=True, mode="reduce-overhead")
    def hardcoded(X: torch.Tensor) -> torch.Tensor:
        X = X.to(torch.float32)
        X /= (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
        X = X.to(torch.float16)

        # Gram Newton-Schulz with quack ops, no conditionals
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


def make_hardcoded_standard_ns(coefficients):
    """Hardcoded standard NS with no conditionals, using quack backend."""
    @torch.compile(fullgraph=True, mode="reduce-overhead")
    def hardcoded(X: torch.Tensor) -> torch.Tensor:
        X = X.to(torch.float32)
        X /= (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
        X = X.to(torch.float16)

        for a, b, c in coefficients:
            A = gemm_symmetric(X, X.mT)
            B = gemm_symmetric(A, A, C=A, alpha=c, beta=b)
            X = gemm_add(B, X, C=X, beta=a, tuned=False)

        return X

    return hardcoded


def main():
    device = "cuda"
    shapes = [
        (16, 256, 256),
        (16, 512, 512),
        (16, 1024, 1024),
        (16, 256, 1024),
    ]

    coefficients = POLAR_EXPRESS_COEFFICIENTS
    reset_iterations = [2]

    print("=" * 80)
    print("Conditional Overhead Benchmark")
    print("Comparing full GramNewtonSchulz (with conditionals) vs hardcoded (no conditionals)")
    print("=" * 80)

    # --- Gram Newton-Schulz (non-square, where gram is chosen) ---
    print("\n--- GRAM Newton-Schulz path (non-square matrices) ---")
    print(f"{'Shape':<20} {'Full (ms)':<12} {'Hardcoded (ms)':<16} {'Overhead (ms)':<15} {'Overhead %':<12}")
    print("-" * 75)

    gram_ns = GramNewtonSchulz(
        ns_coefficients=coefficients,
        gram_newton_schulz_reset_iterations=reset_iterations,
        ns_use_kernels=True,  # Use quack backend
    )
    hardcoded_gram = make_hardcoded_gram_ns(coefficients, set(reset_iterations))

    for shape in shapes:
        if shape[-2] == shape[-1]:
            continue  # gram path only chosen for non-square

        X = torch.randn(shape, device=device)

        full_times = benchmark_fn(lambda: gram_ns(X))
        hard_times = benchmark_fn(lambda: hardcoded_gram(X.unsqueeze(0) if X.ndim == 2 else X))

        full_med = sorted(full_times)[len(full_times) // 2]
        hard_med = sorted(hard_times)[len(hard_times) // 2]
        overhead = full_med - hard_med
        overhead_pct = (overhead / hard_med) * 100 if hard_med > 0 else 0

        print(f"{str(shape):<20} {full_med:<12.4f} {hard_med:<16.4f} {overhead:<15.4f} {overhead_pct:<12.1f}")

    # --- Standard Newton-Schulz (square matrices) ---
    print("\n--- STANDARD Newton-Schulz path (square matrices) ---")
    print(f"{'Shape':<20} {'Full (ms)':<12} {'Hardcoded (ms)':<16} {'Overhead (ms)':<15} {'Overhead %':<12}")
    print("-" * 75)

    standard_ns = GramNewtonSchulz(
        ns_coefficients=coefficients,
        ns_use_kernels=True,
        use_gram_newton_schulz=False
    )
    hardcoded_standard = make_hardcoded_standard_ns(coefficients)

    for shape in shapes:
        if shape[-2] != shape[-1]:
            continue  # standard path only for square

        X = torch.randn(shape, device=device)

        full_times = benchmark_fn(lambda: standard_ns(X))
        hard_times = benchmark_fn(lambda: hardcoded_standard(X.unsqueeze(0) if X.ndim == 2 else X))

        full_med = sorted(full_times)[len(full_times) // 2]
        hard_med = sorted(hard_times)[len(hard_times) // 2]
        overhead = full_med - hard_med
        overhead_pct = (overhead / hard_med) * 100 if hard_med > 0 else 0

        print(f"{str(shape):<20} {full_med:<12.4f} {hard_med:<16.4f} {overhead:<15.4f} {overhead_pct:<12.1f}")

    # --- Also test: compile with conditionals vs without compile ---
    print("\n--- Effect of torch.compile on conditional overhead ---")
    print(f"{'Shape':<20} {'Compiled (ms)':<14} {'Uncompiled (ms)':<17} {'Compile speedup':<15}")
    print("-" * 66)

    uncompiled_ns = GramNewtonSchulz(
        ns_coefficients=coefficients,
        gram_newton_schulz_reset_iterations=reset_iterations,
        ns_use_kernels=True,
        compile_kwargs=None,  # No compilation
    )

    for shape in shapes:
        X = torch.randn(shape, device=device)

        compiled_times = benchmark_fn(lambda: gram_ns(X) if shape[-2] != shape[-1] else standard_ns(X))
        uncompiled_times = benchmark_fn(lambda: uncompiled_ns(X))

        comp_med = sorted(compiled_times)[len(compiled_times) // 2]
        uncomp_med = sorted(uncompiled_times)[len(uncompiled_times) // 2]
        speedup = uncomp_med / comp_med if comp_med > 0 else 0

        print(f"{str(shape):<20} {comp_med:<14.4f} {uncomp_med:<17.4f} {speedup:<15.2f}x")

    print("\n" + "=" * 80)
    print("If 'Overhead %' is significant (>5%), conditionals are costly at this size.")
    print("If compile speedup is large, torch.compile is helping eliminate Python overhead.")
    print("=" * 80)


if __name__ == "__main__":
    main()
