// ============================================================================
// instrumentation_snippet.cu — minimal device-side timing snippets.
//
// Copy the pieces you need into a profiling harness based on
// `helpers/harness_template.cu`. Keep probes behind KERNEL_PROFILING and always
// validate conclusions with an uninstrumented NCU report.
//
// Compile the harness with:
//     nvcc -O2 -std=c++17 -lineinfo -DKERNEL_PROFILING \
//          -gencode=arch=compute_100,code=sm_100 \
//          my_kernel_harness.cu -o my_kernel_harness
//
// Suitable scenarios:
//   - per-CTA / per-warp phase timing histograms
//   - checking whether a specific async wait is exposed
//   - detecting tail CTAs or data-dependent loop imbalance
//
// Avoid:
//   - every-thread global timestamp writes
//   - hot-loop atomics
//   - using instrumented absolute runtime as production performance
// ============================================================================

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <vector>

// ---------------------------------------------------------------------------
// Device timer helpers
// ---------------------------------------------------------------------------

__device__ __forceinline__ unsigned long long read_clock64_probe() {
    // Best for same-thread / same-SM relative cycle deltas.
    return clock64();
}

__device__ __forceinline__ unsigned long long read_globaltimer_probe() {
    // Useful for coarse cross-SM event ordering. `%globaltimer` is target-specific
    // and documented for NVIDIA tools; use as a probe, not a long-term API.
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}

// ---------------------------------------------------------------------------
// Minimal per-CTA timing buffer
// ---------------------------------------------------------------------------

struct CtaPhaseStamps {
    unsigned long long t0;  // phase 0 begin
    unsigned long long t1;  // phase 0 end / phase 1 begin
    unsigned long long t2;  // phase 1 end
    unsigned int work;      // optional: elements / loop trips handled by CTA
};

// Add one extra kernel argument:
//
//     CtaPhaseStamps* __restrict__ stamps
//
// Allocate it in the harness:
//
//     CtaPhaseStamps* d_stamps = nullptr;
//     CUDA_CHECK(cudaMalloc(&d_stamps, grid.x * sizeof(CtaPhaseStamps)));
//     CUDA_CHECK(cudaMemset(d_stamps, 0, grid.x * sizeof(CtaPhaseStamps)));
//
// Copy it back after the kernel:
//
//     std::vector<CtaPhaseStamps> h_stamps(grid.x);
//     CUDA_CHECK(cudaMemcpy(h_stamps.data(), d_stamps,
//                           grid.x * sizeof(CtaPhaseStamps),
//                           cudaMemcpyDeviceToHost));

// Paste this pattern into the kernel around the region you want to time.
// Use __syncthreads() only if the phase is CTA-wide; synchronization changes
// performance and can hide/introduce pipeline bubbles.

#if 0
template <typename T>
__global__ void my_kernel_with_probes(const T* x, T* out,
                                      int n,
                                      CtaPhaseStamps* __restrict__ stamps) {
#if defined(KERNEL_PROFILING)
    __syncthreads();
    if (threadIdx.x == 0) {
        stamps[blockIdx.x].t0 = read_clock64_probe();
        stamps[blockIdx.x].work = 0;
    }
    __syncthreads();
#endif

    // ---------------- measured phase 0: e.g. tile load / compute loop -------
    unsigned int local_work = 0;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < n;
         i += gridDim.x * blockDim.x) {
        // ... your original work ...
        ++local_work;
    }

#if defined(KERNEL_PROFILING)
    // Debug only. Prefer one write per CTA/warp; avoid hot-loop atomics.
    atomicAdd(&stamps[blockIdx.x].work, local_work);
    __syncthreads();
    if (threadIdx.x == 0) {
        stamps[blockIdx.x].t1 = read_clock64_probe();
    }
    __syncthreads();
#endif

    // ---------------- measured phase 1: e.g. epilogue / store ---------------
    // ... your original work ...

#if defined(KERNEL_PROFILING)
    __syncthreads();
    if (threadIdx.x == 0) {
        stamps[blockIdx.x].t2 = read_clock64_probe();
    }
#endif
}
#endif

// ---------------------------------------------------------------------------
// Async pipeline phase markers
// ---------------------------------------------------------------------------
//
// For cp.async / TMA / WGMMA-HGMMA / tcgen05-style pipelines, measure phase
// boundaries rather than one big region:
//
//   load_issue_start -> load_wait_done -> mma_issue_start -> mma_wait_done
//
// Example placement:
//
//     stamps[cta].t0 = clock64();   // before async copy / TMA issue
//     ... issue async copies ...
//     ... wait for data ...
//     stamps[cta].t1 = clock64();   // after wait, before MMA issue
//     ... issue WGMMA/HGMMA/tcgen05 group ...
//     ... wait_group / depbar / mbarrier wait ...
//     stamps[cta].t2 = clock64();   // after exposed wait
//
// Interpret long intervals with NCU:
//   - long copy wait  -> check mbarrier/TMA/long-scoreboard/source counters
//   - long MMA wait   -> check tensor pipe, warpgroup_arrive/wait/dependency
//   - long epilogue   -> check store sectors, spills, ALU/XU pressure

// ---------------------------------------------------------------------------
// Host-side summary helper
// ---------------------------------------------------------------------------

[[maybe_unused]] static void print_cta_phase_summary(
    const std::vector<CtaPhaseStamps>& stamps) {
    if (stamps.empty()) {
        std::printf("no CTA stamps\n");
        return;
    }

    std::vector<unsigned long long> phase0, phase1, total;
    phase0.reserve(stamps.size());
    phase1.reserve(stamps.size());
    total.reserve(stamps.size());

    for (const CtaPhaseStamps& s : stamps) {
        phase0.push_back(s.t1 - s.t0);
        phase1.push_back(s.t2 - s.t1);
        total.push_back(s.t2 - s.t0);
    }

    auto summarize = [](const char* name,
                        const std::vector<unsigned long long>& values) {
        auto [mn, mx] = std::minmax_element(values.begin(), values.end());
        long double sum = 0.0;
        for (auto v : values) sum += static_cast<long double>(v);
        std::printf("%-12s min=%10llu avg=%10.1Lf max=%10llu cycles\n",
                    name, *mn, sum / static_cast<long double>(values.size()), *mx);
    };

    summarize("phase0", phase0);
    summarize("phase1", phase1);
    summarize("total", total);

    std::printf("first CTAs:\n");
    for (int i = 0; i < std::min<int>(8, stamps.size()); ++i) {
        const CtaPhaseStamps& s = stamps[i];
        std::printf("  cta=%03d work=%u phase0=%llu phase1=%llu total=%llu\n",
                    i, s.work, s.t1 - s.t0, s.t2 - s.t1, s.t2 - s.t0);
    }
}
