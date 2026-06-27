# Kernel Internal Instrumentation

Use this document when the user asks whether a CUDA / Triton / CuTe / CUTLASS
kernel can measure a specific code region, pipeline phase, or source interval
from inside the kernel.

The short answer:

- **Region timing:** yes, with device-side timestamps such as `clock64()` or
  inline PTX reads of `%globaltimer`.
- **Region-level hardware counters:** usually no. Nsight Compute counters are
  collected by the profiler, not by normal device code. Use NCU SourceCounters,
  PM sampling, or split the code into separately profiled kernels/phases.

Instrumentation is a probe. Keep it behind a compile-time flag and remove it
from final performance measurements unless the instrumentation overhead is the
thing being studied.

Minimal copy-paste snippets live at
[`../helpers/instrumentation_snippet.cu`](../helpers/instrumentation_snippet.cu).
Use them with a harness based on
[`../helpers/harness_template.cu`](../helpers/harness_template.cu).

---

## Method Selection

| Method | Suitable when | Avoid when | What it tells you |
|---|---|---|---|
| `clock64()` interval timing | Timing a region within one thread, warp, or CTA on one SM | Comparing absolute times across SMs, or measuring very short regions where the timer overhead dominates | Cycle delta for the instrumented thread/CTA path |
| `%globaltimer` inline PTX | You need a device-side timer closer to global nanoseconds across SMs | Long-term portable code; PTX documents it as target-specific and intended for NVIDIA tools | Approximate global timestamp deltas |
| Per-CTA / per-warp timing buffer | You need a distribution: tail CTAs, imbalance, phase duration variance | Every thread writing timestamps; that will destroy the workload | Which CTAs/warps are slow and where phase variance appears |
| Phase markers/counters | You need to count branch path frequency, loop trip counts, queue depth, retry counts | Hot inner loops where atomics or global stores perturb scheduling | Control-flow and workload-shape evidence |
| NCU SourceCounters + PM sampling | You need stall reasons, pipe utilization, source/SASS hotspots, or bubbles | Kernels too short for sampling, missing `-lineinfo`, unsupported PM sampling | Hardware-backed attribution without editing kernel logic |
| Split-kernel / differential profiling | A region boundary is clean and can be separated without changing semantics too much | Strong producer/consumer overlap or cache-state dependence makes the split unrealistic | Approximate region cost by A/B timing and NCU comparisons |
| Host-side NVTX ranges | You need to mark launches, high-level phases, or framework regions in Nsight Systems/NCU filtering | Annotating inside `__device__` code; NVTX is not a device-code range API | Which host region launched which kernels |
| Binary instrumentation such as NVBit | Research/debug: instruction coverage, dynamic instruction tracing, special probes | Normal optimization loops; overhead and complexity are high | Instruction-level dynamic traces or custom counters |

---

## CUDA / CuTe / CUTLASS: Timing A Code Region

For CUDA C++ kernels, including CuTe/CUTLASS kernels where you can edit device
code, use `clock64()` for low-friction cycle timing.

```cpp
__device__ __forceinline__ unsigned long long read_clock64() {
    return clock64();
}

template <typename T>
__global__ void kernel_with_probe(const T* x, T* y, unsigned long long* timing) {
#if defined(KERNEL_PROFILING)
    unsigned long long t0 = 0;
    if (threadIdx.x == 0) t0 = read_clock64();
#endif

    // Region to measure.
    // For example: load tile, issue MMA group, wait, epilogue, or store.

#if defined(KERNEL_PROFILING)
    if (threadIdx.x == 0) {
        unsigned long long t1 = read_clock64();
        timing[blockIdx.x] = t1 - t0;
    }
#endif
}
```

Use this for:

- A quick per-CTA timing histogram.
- Comparing two versions of the same code region.
- Detecting tail CTAs or variable loop trip counts.

Do not use this alone to claim root cause. Pair it with NCU metrics. A slow
region could be slow due to memory dependencies, barriers, tensor pipe waits, or
scheduler starvation; the timestamp only says "how long", not "why".

### CTA-Level Region Timing

If the measured region should include all threads in a CTA, place synchronization
around the region. This changes behavior, so use it only for probes.

```cpp
#if defined(KERNEL_PROFILING)
__syncthreads();
unsigned long long t0 = 0;
if (threadIdx.x == 0) t0 = clock64();
__syncthreads();
#endif

// CTA-wide measured region.

#if defined(KERNEL_PROFILING)
__syncthreads();
if (threadIdx.x == 0) {
    unsigned long long t1 = clock64();
    timing[blockIdx.x] = t1 - t0;
}
#endif
```

Use this for:

- A phase where all threads participate, such as tile load, shared-memory
  transpose, CTA-level reduction, or epilogue store.

Avoid this when:

- The original code deliberately overlaps work across warps or uses warp
  specialization. Extra `__syncthreads()` can remove the very overlap you are
  trying to measure.

### `%globaltimer`

When cross-SM timestamp comparability matters more than cycle-level overhead,
read `%globaltimer` via inline PTX.

```cpp
__device__ __forceinline__ unsigned long long read_globaltimer() {
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}
```

Use this for:

- Coarse event ordering across CTAs/SMs.
- Debugging load imbalance or persistent-kernel work-queue timing.

Avoid this when:

- You need a stable public semantic across GPU generations. The PTX ISA
  documents `%globaltimer` as target-specific and intended for NVIDIA tools.

---

## Instrumenting Async Pipelines

For `cp.async`, TMA, WGMMA/HGMMA, and Blackwell `tcgen05` pipelines, measure
phase boundaries, not just one large region.

Typical phase labels:

```text
load_issue_start
load_commit
load_wait_done
mma_issue_start
mma_commit
mma_wait_done
epilogue_start
store_done
```

For WGMMA-style code:

```cpp
#if defined(KERNEL_PROFILING)
if (threadIdx.x == 0) stamps[blockIdx.x * 4 + 0] = clock64();
#endif

// wgmma.mma_async / cute::gemm tiled mma issue

#if defined(KERNEL_PROFILING)
if (threadIdx.x == 0) stamps[blockIdx.x * 4 + 1] = clock64();
#endif

// wgmma.commit_group
// independent work, if any
// wgmma.wait_group

#if defined(KERNEL_PROFILING)
if (threadIdx.x == 0) stamps[blockIdx.x * 4 + 2] = clock64();
#endif
```

Interpretation:

- `issue` interval: front-end / instruction issue cost of the async operation.
- `wait` interval: exposed latency after overlap opportunities.
- High wait time plus NCU `mbarrier`, `wait`, `warpgroup_arrive`, `long_scoreboard`,
  or tensor-pipe oscillation is evidence of a pipeline bubble.

Use this for:

- Checking whether prefetch depth is enough.
- Comparing 2-stage vs 3-stage vs 4-stage pipelines.
- Verifying that producer/consumer or warp-specialized phases overlap.

Avoid this when:

- The timestamp store itself disrupts register allocation or scheduling. Always
  compare an uninstrumented build after forming the hypothesis.

---

## Phase Counters

Sometimes the right probe is a count, not a timestamp.

```cpp
struct KernelDebugCounters {
    unsigned long long branch_a;
    unsigned long long branch_b;
    unsigned long long loop_iters;
};

__device__ __forceinline__ void debug_add(unsigned long long* p,
                                          unsigned long long v) {
#if defined(KERNEL_PROFILING)
    atomicAdd(p, v);
#endif
}
```

Use this for:

- Data-dependent branches.
- Loop trip count distributions.
- Work stealing or persistent-kernel queue behavior.
- Verifying that a supposedly rare slow path is actually rare.

Avoid atomics in the hot inner loop. Prefer one write per CTA or one write per
warp to a preallocated debug buffer, then reduce on the host.

---

## Triton Kernels

Triton does not expose CUDA C++ `clock64()` directly, but `tl.inline_asm_elementwise`
can emit inline PTX. The exact constraints can vary by Triton version, so treat
this as a probe pattern to validate in the local environment.

```python
import triton
import triton.language as tl

@triton.jit
def kernel(x, y, timing, BLOCK: tl.constexpr):
    pid = tl.program_id(0)

    t0 = tl.inline_asm_elementwise(
        "mov.u64 $0, %clock64;",
        constraints="=l",
        args=[],
        dtype=tl.uint64,
        is_pure=False,
        pack=1,
    )

    # Region to measure.

    t1 = tl.inline_asm_elementwise(
        "mov.u64 $0, %clock64;",
        constraints="=l",
        args=[],
        dtype=tl.uint64,
        is_pure=False,
        pack=1,
    )

    tl.store(timing + pid, t1 - t0)
```

Use this for:

- Quick region timing in a Triton program.
- Program-ID-level timing histograms.

Avoid this when:

- You need source-level NCU attribution. For Triton, the more robust route is
  often to dump generated PTX/SASS, or rebuild a minimal standalone CUDA
  harness when possible.

---

## NVTX And Profiler Ranges

NVTX is useful for host-side ranges:

```cpp
nvtxRangePushA("decode_step");
my_kernel<<<grid, block, smem, stream>>>(...);
nvtxRangePop();
```

Use this for:

- Marking framework phases.
- Filtering NCU/NSYS to kernels launched inside a host-side range.
- Separating data loading, preprocessing, kernel launch, and postprocessing.

Avoid this for:

- Device-side code regions. NVTX is not a `__device__` range annotation API.

NCU can use NVTX filters at the CLI level; see `ncu --help` for
`--nvtx`, `--nvtx-include`, and `--nvtx-exclude`.

---

## Prefer NCU For Hardware Cause

Internal timestamps identify a slow interval. NCU identifies why it is slow.

Use NCU when the question is:

- Which pipe is underutilized?
- Are tensor cores waiting for data?
- Is the wait on long scoreboard, short scoreboard, mbarrier, warpgroup arrive,
  MIO throttle, or no instruction?
- Is there a tail wave or phase oscillation?
- Which SASS/source line has the stall samples?

Minimum collection:

```bash
ncu --set full \
    --section PmSampling \
    --section PmSampling_WarpStates \
    -k "regex:KERNEL_REGEX" -c 1 \
    -o "$PROFILE_RUN_DIR/reports/full_<tag>" \
    ./harness [args]

ncu --set source \
    --section SourceCounters \
    -k "regex:KERNEL_REGEX" -c 1 \
    -o "$PROFILE_RUN_DIR/reports/source_<tag>" \
    ./harness [args]
```

Then correlate:

| Timestamp finding | NCU evidence to check |
|---|---|
| Long tile-load phase | global/shared load sectors, L2 hit rate, long scoreboard, LSU/L1TEX throughput |
| Long wait after async copy/TMA | mbarrier/wait stalls, TMA pipe activity, PM-sampling oscillation |
| Long wait after WGMMA/HGMMA issue | tensor pipe utilization, warpgroup arrive/wait/dependency stalls, eligible warps |
| Long epilogue | store sectors, local-memory spills, ALU/XU pipe, predicate/divergence |
| Tail CTAs much slower | PM-sampling tail, per-SM active-cycle variance, launch waves, work distribution |

---

## Reporting Requirements

When using instrumentation in a profiling report, include:

1. Whether instrumentation was compiled in (`KERNEL_PROFILING`, debug branch,
   extra buffers, extra synchronizations).
2. Which threads wrote timestamps or counters.
3. Timer source: `clock64()` or `%globaltimer`.
4. Output schema: one value per CTA, per warp, per phase, or global aggregate.
5. Expected overhead and how it might perturb scheduling.
6. Uninstrumented NCU run used to validate that the observed bottleneck still
   exists without probes.

Never compare an instrumented kernel's absolute runtime against a production
kernel unless the report explicitly accounts for probe overhead.

---

## References

- CUDA inline PTX assembly: <https://docs.nvidia.com/cuda/inline-ptx-assembly/>
- PTX ISA special registers (`%clock64`, `%globaltimer`):
  <https://docs.nvidia.com/cuda/parallel-thread-execution/>
- Nsight Compute CLI PM sampling, warp sampling, and NVTX filters:
  <https://docs.nvidia.com/nsight-compute/NsightComputeCli/>
- Triton inline assembly:
  <https://triton-lang.org/main/python-api/generated/triton.language.inline_asm_elementwise.html>
