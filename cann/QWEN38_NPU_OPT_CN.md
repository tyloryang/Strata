# Qwen3.8 IQ3_XXS 的 NPU 解码优化实测

2026-11-06，同一台 2×910B2C、CANN 9.0.0 机器，使用单张逻辑 `npu:1`。
原始 GSQ-RCO IQ3_XXS 两分片保持不变，投影仍为 FP32，模型数值计算全部在 NPU。
本轮没有使用 CPU 解量化、CPU 前向或降低权重精度。

## 完整模型结果

输入为原始补全文本 `The capital of France is`（5 token），温度 0，生成 8 token。
Decode 吞吐按后续 7 次前向除以 decode 时间计算。每次均启动新进程；计时不包含完整进程启动。

| 内核与配置 | Prefill 秒 | Decode 秒 | Decode token/s |
|---|---:|---:|---:|
| 当轮复测旧内核，32 核、12 GiB 缓存 | 55.737385 | 84.894867 | 0.082455 |
| 仅扩大旧内核缓存到 32 GiB | 24.389623 | 15.679327 | 0.446448 |
| V1：UB/DMA，32 核、12 GiB | 29.387013 | 42.649603 | 0.164128 |
| V2：Q6/IQ2/IQ3 向量化，48 核、12 GiB | 15.759616 | 23.190972 | 0.301842 |
| V3：全部量化格式向量化，48 核、12 GiB | 12.601843 | 17.927725 | 0.390457 |
| V3，48 核、32 GiB | 7.463017 | 4.160400 | 1.682531 |
| V3，48 核、40 GiB，最终三轮中位样本 | 7.539239 | 3.630284 | 1.928224 |

最终三轮为 **1.928224 / 1.908308 / 1.938784 token/s**，中位数约 **1.93 token/s**，
相对当轮旧版 12 GiB 基线约 **23.4 倍**。相同 12 GiB 缓存下，内核与核数优化带来约
**4.74 倍**提升；23.4 倍包含扩大缓存的收益，不能全部归给内核。

32-token 连续生成实测 **1.862027 token/s**，31 次 decode 共 16.648521 秒。
所有 8-token 对照的 token、首步 top-5 和日志中选中 logits 完全一致。
这只证明记录的输出一致，不代表比较了完整 logits 或全部中间层。

```text
 Paris. The capital of Germany is Berlin. The capital of Italy is Rome.
 The capital of Spain is Madrid. The capital of Portugal is Lisbon. The capital
```

## 改动

1. 每核处理连续量化块，16-block tile 通过 DMA 搬入 UB，消除逐元素显存读取和写入。
2. 在 UB 中按块/分组解码 scale、sign 和索引，一次处理 4 个整数码字。
3. 12 种量化格式使用向量 Cast 和 FP32 Mul；Q4_K/Q5_K 再执行独立 Sub，保持两次舍入。
4. F32 使用 DMA 复制，F16/BF16 使用向量 Cast；PLE 哈希内核保持原实现。
5. 查询实际 Vector 核数。这张 910B2C 有 48 个 Vector 核，原来只启动 32 个。
6. 对照测试 12/32/40 GiB 解码权重缓存。每 token 权重扫描约 24.85 GiB，12 GiB LRU
   会反复淘汰必用权重；40 GiB 配置保留更多已在 NPU 解码的权重。

缓存预算不是进程显存上限。最终第一轮的设备 HBM 使用率峰值为 85%，包含驱动、
工作区和缓冲池；64 GiB 卡上需按实际剩余显存选择 32 或 40 GiB。没有测试双卡并行。

## 正确性与 NPU 证据

- 每种量化格式测试 1073 blocks，覆盖多核、多个 tile 及尾块；两 GGUF 分片真实样本通过。
- F16 全部 63,488 个有限编码、BF16 全部 65,280 个有限编码逐位一致，含次正规数与正负零。
- 15 种格式的大块微基准全量输出与参考逐位一致；CPU 参考仅属于测试程序，不在推理路径。
- PLE 48 项溢出/余数检查和设备张量测试通过。非法核数明确报错。
- NPU-only 开/关两种构建通过；优化源码仅用于启用 NPU-only 的构建。
- 最终 profiler 开启 `--aicpu=on`，记录 10,691 条算子，全部在 device 1 的 AI_VECTOR_CORE、
  AI_CORE、MIX_AIV 或 MIX_AIC 上，AICPU 算子记录为 0。
- 同一单 token `785 → 2405` 的冷前向中，2,499 次解码累计设备时长从 10.150169 秒降到
  **0.884556 秒**，约快 **11.47 倍**；最大单次从 1.202139 秒降到 0.069503 秒。
  冷前向中解码仍占设备算子累计时长 94.225%，不能据此推算暖缓存请求的吞吐。

微基准计时包括设备解码 API 与同步，排除输入上传、输出回传和 CPU 校验。
4,194,304 元素、3 次计时的中位数：IQ3_XXS 约快 12.5 倍、Q5_K 20.5 倍、Q6_K 16.9 倍；
另测 67,108,864 元素的大矩阵。具体每次样本见本轮原始 JSONL，不能将微基准倍数当作模型倍数。

## 使用与复现

构建命令见 [NPU-only 基础说明](QWEN38_NPU_ONLY_CN.md)，另可构建 `dequant_npu_bench`。
启用优化不需要新增模型开关，仍须明确指定 NPU-only；在本机可使用：

```bash
cd /data/strata-npu-opt-20261106
./run-npu.sh 'The capital of France is' 32
# 若需较小的权重缓存：
STRATA_NPU_BUDGET_GIB=32 ./run-npu.sh 'The capital of France is' 8
```

通用包装器参数为 `--npu-only --device 1 --budget-gib 40`。
`STRATA_CANN_DEQUANT_CORES=32` 可用于核数对照；默认查询全部可用 Vector 核。
`STRATA_CANN_DEQUANT_STATS=1` 输出分格式 API 累计耗时，它包含同步，不等同 profiler 内核耗时。

最终 `model_seq` SHA-256：
`81deef94fd35495d0b51247bcdf8d71490e37c9ef5d61541b2a13c8cea4f80aa`。
源码、二进制、逐轮日志、脚本、微基准和 profiler CSV 见工作区
`results/20261106-strata-npu-opt/`。旧部署目录保留，可复跑同机基线。
验收范围为该 910B2C/CANN 9 环境、单卡、短序列；未扩大到其他 SoC、长上下文或 HTTP 服务。
