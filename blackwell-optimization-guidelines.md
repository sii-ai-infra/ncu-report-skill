# Blackwell 专属优化准则

## 目标平台

本文档的所有准则和建议均针对以下开发环境：

- **GPU 架构：** NVIDIA Blackwell — B200（Compute Capability 10.0，双 die 共 148 SM）
- **CUDA Toolkit：** 13.2
- **编译目标：** `sm_100a`（`-arch=sm_100a` 或 `-gencode arch=compute_100a,code=sm_100a`）

编译命令示例：
```bash
nvcc -arch=sm_100a -lineinfo -O3 -o my_kernel my_kernel.cu
```

---

## Blackwell B200 架构关键参数速查

| 参数 | B200 数值 | 对比 H100 |
|------|----------|----------|
| SM 数量 | 148（2 die × 74 SM） | 132 |
| 每 SM 最大 warp 数 | 64 | 64 |
| 每 SM 寄存器文件 | 64K × 32-bit | 64K × 32-bit |
| 每线程最大寄存器 | 255 | 255 |
| 每 SM 最大 thread block 数 | 32 | 32 |
| Shared Memory 每 SM（可配置） | 最高 228 KB（可用 227 KB） | 最高 228 KB |
| **Tensor Memory (TMEM) 每 SM** | **256 KB（512 列 × 128 lane × 32-bit）** | **无** |
| L2 Cache | 126 MB（GB200） | 50 MB |
| HBM 容量 | 192 GB HBM3e | 80 GB HBM3 |
| HBM 带宽 | 8 TB/s | 3.35 TB/s |
| FP4 Tensor（dense/sparse） | 9 / 18 PFLOPS | 不支持 |
| FP8 Tensor（dense/sparse） | 4.5 / 9 PFLOPS | 1.98 / 3.96 PFLOPS |
| FP16/BF16 Tensor（dense/sparse） | 2.25 / 4.5 PFLOPS | 0.99 / 1.98 PFLOPS |
| TF32 Tensor（dense/sparse） | 1.13 / 2.25 PFLOPS | 0.49 / 0.99 PFLOPS |
| FP64 Tensor | 45 TFLOPS | 67 TFLOPS |
| NVLink 带宽 | 1.8 TB/s（NVLink 5） | 900 GB/s（NVLink 4） |
| 最大 Cluster size（portable / non-portable） | 8 / 16 | 8 / 16 |

---

## Blackwell 独有特性与编程要点

### 1. 第 5 代 Tensor Core 与 tcgen05 指令

Blackwell 的 Tensor Core 与前代有根本性不同。核心变化：

**从 warpgroup MMA 到 warp-level 单线程 MMA：** Hopper 的 `wgmma` 以 128 线程（4 个 warp 的 warpgroup）为单位发起 MMA，而 Blackwell 的 `tcgen05.mma` 由**单个线程**代表整个 CTA 发起。这大幅降低了单条指令延迟（比 Hopper 低 3-11 倍），但要求新的编程模型。

**累加器存放在 TMEM 而非寄存器：** 线程不再"拥有" MMA 结果。`tcgen05.mma` 的输出写入 TMEM，需要通过 `tcgen05.ld` 显式搬回寄存器做后处理（epilogue）。

**操作数来源：** A 矩阵可以来自 Shared Memory 或 TMEM，B 矩阵来自 Shared Memory。D（累加器）始终在 TMEM。

**编程模式（通过 inline PTX）：**
```cuda
// TMEM 分配（由单个 warp 执行）
asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
             : : "r"(smem_addr), "r"(num_columns));

// MMA 指令（由单个线程，即 elected leader 发射）
asm volatile("tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, %4;"
             : : "r"(tmem_addr), "l"(a_desc), "l"(b_desc), "r"(idesc), "b"(pred));

// 从 TMEM 加载结果到寄存器
asm volatile("tcgen05.ld.sync.aligned.32x32b.x1.b32 %0, [%1];"
             : "=r"(reg) : "r"(tmem_addr));

// TMEM 释放
asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
             : : "r"(tmem_addr), "r"(num_columns));
```

**实践建议：**
- 对于 GEMM 类 kernel，优先使用 CUTLASS 4.x 的 Blackwell 原生路径，而非手写 tcgen05。手写 tcgen05 需要管理 leader election、mbarrier 同步、TMEM 分配/释放等大量细节。
- 手写 tcgen05 kernel 时，tile 操作数建议按 64×64 元素对齐以最大化 TMEM 带宽。
- `tcgen05.mma` 与 `tcgen05.cp`（smem→tmem）形成隐式 pipeline——按发射顺序执行。利用这一特性可以省去部分同步操作。

### 2. Tensor Memory (TMEM)

TMEM 是 Blackwell 新增的片上内存空间，每个 SM 256 KB，专用于 Tensor Core 数据通路。

**关键特性：**
- 读带宽 ~16 TB/s/SM，写带宽 ~8 TB/s/SM（远超 shared memory）
- 组织为 512 列 × 128 lane × 32-bit
- 只能通过专用指令访问：`tcgen05.ld`、`tcgen05.st`、`tcgen05.cp`（smem→tmem）
- 不能被传统 load/store 指令（`ld.shared`、`ld.global` 等）访问
- 需要显式分配和释放（类似 malloc/free）

**TMEM vs 寄存器 vs Shared Memory：**

| 特性 | 寄存器 | Shared Memory | TMEM |
|------|--------|--------------|------|
| 容量/SM | 64 KB | 最高 228 KB | 256 KB |
| 延迟 | 1 cycle | ~20-30 cycles | 专用通路 |
| 谁能访问 | 单个线程 | block 内所有线程 | Tensor Core + 显式搬运 |
| 用途 | 通用计算 | 线程间共享数据 | MMA 累加器 + operand staging |

**实践建议：**
- TMEM 主要用于 GEMM/MMA 的累加器，不要尝试将其用作通用暂存。
- 中间累加结果应尽量保持在 TMEM 中，减少 TMEM↔寄存器搬运。只在 epilogue（如 bias add、activation）时才将结果搬到寄存器。
- TMEM 分配必须由单个 warp 执行，且同一个 warp 负责分配和释放。分配列数必须是 2 的幂且≥32。
- TMEM 分配存在竞争的可能（多个 block 同时分配），高性能 kernel 应实现带重试和 nanosleep 退避的分配逻辑。

### 3. CTA Pair（2CTA）— 双 SM 协作

Blackwell 中，同一 TPC（Texture Processing Cluster）内的两个 SM 可以组成 **CTA pair** 协作执行 MMA。这通过 `cta_group::2` 修饰符启用。

**好处：**
- 两个 CTA 共享输入操作数，将每个 CTA 的 shared memory 带宽需求减半
- 有效的 MMA M 维度翻倍（如单 CTA M=128，2CTA 则 M=256）
- 这是利用满 Tensor Core 吞吐量的必要条件——单 SM 的 MMA（M=64）只能达到 ~50% 理论峰值

**实践建议：**
- 对于大型 GEMM，尽量启用 2CTA 模式以充分利用 Tensor Core。
- 使用 Thread Block Cluster（cluster_size=2）来确保 CTA pair 映射到同一 TPC。
- 2CTA 模式下 TMA 数据加载也可以跨 CTA 分摊。

### 4. 低精度数据类型：FP4 / FP6 / FP8 与 Block Scaling

Blackwell Tensor Core 原生支持的精度格式（吞吐量关系）：

| 格式 | 说明 | Dense 吞吐量 | 相对 FP16 |
|------|------|-------------|----------|
| NVFP4 | E2M1 + block-16 E4M3 scale + per-tensor FP32 scale | 9 PFLOPS | 4x |
| MXFP4 | E2M1 + block-32 E8M0 scale | 9 PFLOPS | 4x |
| MXFP6 (E3M2 / E2M3) | 6-bit float + block scaling | 4.5 PFLOPS | 2x |
| MXFP8 / FP8 (E4M3 / E5M2) | 8-bit float | 4.5 PFLOPS | 2x |
| FP16 / BF16 | 标准 16-bit | 2.25 PFLOPS | 1x |
| TF32 | 19-bit（10-bit mantissa） | 1.13 PFLOPS | 0.5x |
| FP64 | 双精度 | 45 TFLOPS | 0.02x |

**FP4 与 FP6 吞吐量相同的原因：** FP4 和 FP8 共享相同物理 Tensor Core 电路，FP4 实现 2x 吞吐是因为每个 cycle 能处理双倍数量的元素。FP6 和 FP8 共享电路，因此 FP6 吞吐量等于 FP8（不是 FP4 和 FP8 的中间值）。

**NVFP4 vs MXFP4：**
- NVFP4 使用更小的 block size（16 vs 32）和更精确的 scale format（E4M3 vs E8M0），通常精度更好
- MXFP4 是 OCP 社区标准，跨平台兼容性更好
- cuBLAS 13.2 对 B200 上的 NVFP4 已有优化支持

**实践建议：**
- **LLM 推理优先考虑 NVFP4（W4A4）或 FP8（W8A8）：** NVFP4 将模型显存占用降低 ~1.8x（相比 FP8），同时保持接近 FP8 的精度。
- **训练推荐 BF16 + FP8 混合：** BF16/FP16 用于累加保证稳定性，FP8 用于前向/反向的矩阵乘法。
- **不要盲目用最低精度：** FP4 在某些层可能导致显著精度退化（实测平均 ~8.2% perplexity 退化），应该逐层选择精度。
- **编译注意：** 使用 NVFP4/MXFP4 需要 CUTLASS 4.x 或 cuBLAS 的相应 API。CUDA 13.2 的 cuBLAS 已支持 Grouped GEMM 的 MXFP8。

### 5. 硬件解压引擎（Decompression Engine）

B200 内置硬件解压引擎，支持 LZ4、Snappy、Zstandard、GZIP、Bitcomp、ANS 等格式，吞吐量可达 ~539 GB/s（100 MB 块，亚毫秒延迟）。

**使用场景：**
- 大型 LLM 推理：将量化权重以压缩格式存储在 HBM 中，解压引擎在加载时自动解压
- 稀疏矩阵运算：将稀疏数据以 RLE/Bitcomp 压缩存储，流式解压到 TMEM 做运算
- 通过 NVIDIA nvCOMP 库可移植编程

**实践建议：**
- 当模型权重的 HBM 读取是瓶颈时（memory-bound 推理），使用硬件解压可以有效节省带宽。
- 解压引擎与 SM 并行工作，不占用 SM 计算资源。

### 6. Thread Block Cluster 与 Distributed Shared Memory (DSMEM)

继承自 Hopper 的特性，在 Blackwell 上得到增强：

- B200 支持 portable cluster size 最大 8，non-portable 最大 16（需要 `cudaFuncAttributeNonPortableClusterSizeAllowed`）
- Cluster 内的 block 可以直接读写其他 block 的 shared memory（DSMEM）
- DSMEM 与 L2 cache 带宽可以同时使用（叠加带宽）

**DSMEM 访问最佳实践：**
- 访问模式应与 global memory 类似：合并、对齐到 32 字节段
- 避免非 unit stride 访问——如需要，先拷到 local shared memory 再随机访问

### 7. CUDA 13.2 特定功能

- **cuTile（CUDA Tile）：** 新的 tile-level 编程模型，现已支持 Ampere/Ada/Blackwell（sm_80+）。提供 Python 和 C++ 两种接口，可自动生成 tcgen05 + TMA + pipeline 等底层代码。对于新 kernel 开发，cuTile 是一个比手写 PTX 更高效的起点。
- **cuBLAS Grouped GEMM：** 13.2 新增 MXFP8 Grouped GEMM 实验性支持，结合 CUDA Graph 可实现无 host 同步的 device-side shape 处理（MoE 场景 4x 加速）。
- **cuSOLVER FP64 仿真：** 利用 INT8 Tensor Core 仿真 FP64 运算，对 QR/LU/Cholesky 等分解有显著加速。适合 B200 上 INT8 吞吐量远超 FP64 的场景。
- **Nsight Compute 新功能：** 报告聚类（Clustering）、寄存器依赖可视化、PM Sampling 改进。

### 8. Blackwell 上的关键性能准则调整

与 Hopper 相比，以下准则的权重需要调整：

**寄存器压力（准则 6）更宽松：** TMEM 承担了 MMA 累加器存储，原本占大量寄存器的 accumulator 现在不再需要寄存器。对 GEMM 类 kernel，有效可用寄存器增加了。

**Shared Memory 带宽压力降低：** 2CTA 模式下两个 SM 共享 operand，每个 SM 的 shared memory 带宽需求减半。因此对 tiling 策略的 shared memory 带宽约束放宽。

**Pipeline 设计更关键（准则 15）：** Blackwell Tensor Core 吞吐翻倍，但 shared memory 容量没变。需要更精细的 pipeline（3-4 stage）才能喂饱 Tensor Core。如果 PM Sampling 显示 Tensor pipe 利用率有波动，优先检查 pipeline 深度和 prefetch 策略。

**L2 Cache 大幅增加（126 MB）：** 更多 working set 可以留在 L2 内。对于需要跨 block 共享数据的场景（如 multi-head attention 的 KV cache），可以利用 L2 persistence 策略显式管理热数据驻留。

---

## 与通用准则的关系

本文只保留 Blackwell / B200 / sm_100a 专属内容：tcgen05、TMEM、2CTA、FP4/FP6/FP8 block scaling、硬件解压、CUDA 13.2 Blackwell 路径，以及相对 Hopper/通用 CUDA 的性能准则调整。

通用 CUDA kernel 编程准则已拆到 [`cuda-kernel-general-guidelines.md`](cuda-kernel-general-guidelines.md)。使用时先按通用准则检查并行度、访存、寄存器、同步、pipeline 等，再回到本文检查 Blackwell 专属机会。

## 写 Kernel 前的 Blackwell 专项检查清单

在完成通用 CUDA kernel 检查后，再逐项确认这些 Blackwell/B200 特有问题：

### Blackwell 专项（B200 / sm_100a）
0. **编译目标正确吗？** 是否使用了 `-arch=sm_100a` 而非旧架构？
0. **能用 TMEM 吗？** 如果是 GEMM/MMA 类 kernel，累加器应在 TMEM 中而非寄存器。
0. **能用 2CTA 吗？** 大型 GEMM 应启用 CTA pair（`cta_group::2`）以达到 100% Tensor Core 利用率。
0. **数据类型选对了吗？** LLM 推理考虑 NVFP4/FP8，训练考虑 BF16+FP8 混合。FP4 吞吐量是 FP16 的 4 倍。
0. **能用 cuTile 吗？** CUDA 13.2 的 cuTile 支持 Blackwell，可自动生成 tcgen05+TMA pipeline 代码。
0. **L2 persistence 有用吗？** B200 有 126 MB L2，对于热数据（如 KV cache）可以用 L2 persistence 策略驻留。
