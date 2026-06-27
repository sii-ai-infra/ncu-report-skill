# GPU Kernel 编程准则 Skill

此文件原来同时包含 Blackwell 专属内容与通用 CUDA kernel 优化准则。现在已拆分为两份更聚焦的 companion 文档：

1. [`blackwell-optimization-guidelines.md`](blackwell-optimization-guidelines.md) — Blackwell / B200 / sm_100a 专属优化准则：tcgen05、TMEM、2CTA、FP4/FP6/FP8 block scaling、硬件解压、CUDA 13.2 Blackwell 路径等。
2. [`cuda-kernel-general-guidelines.md`](cuda-kernel-general-guidelines.md) — 通用 CUDA kernel 优化准则：并行度、访存合并、shared memory、bank conflict、warp divergence、寄存器压力、pipeline、同步等。

使用建议：

- 先用通用准则检查 kernel 的基本性能结构。
- 如果目标 GPU 是 Blackwell / B200，再额外检查 Blackwell 专属机会和约束。
- NCU profiling 的诊断仍以 `reference/05-analysis-dimensions.md` 和 `reference/06-diagnosis-playbook.md` 为主；这两份 companion 文档用于提出修复方向和设计新 kernel。
