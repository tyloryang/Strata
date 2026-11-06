# Qwen3.8 IQ3_XXS 的原生 NPU 执行路径

2026-11-06，在 2×910B2C、CANN 9.0.0 的机器上，已用逻辑 `npu:1` 运行
Qwen3.8-Flash-Next GSQ-RCO IQ3_XXS。模型权重解码、激活、注意力、递归状态和路由
都在 NPU 执行。CPU 保留文件读取、分词、元数据、设备调用调度及最终输出处理。
这里的 NPU-only 指模型数值计算，不能把它解释为主机 CPU 使用率为零。

这是新的可选路径。原有混合参考路径仍保留，启动时必须明确选择 NPU-only。

后续设备解码优化已实测达到约 **1.93 token/s**，三轮重复测试通过；
配置和逐轮对照见 [NPU 解码优化实测](QWEN38_NPU_OPT_CN.md)。下文性能数值保留初版基线。

## 构建与启动

要求 CANN 的 ACLNN 库、AscendC/Bisheng 编译器和 kernel CMake 文件已安装。
从 Strata 源码根目录执行：

```bash
source /usr/local/Ascend/cann/set_env.sh
export ASCEND_HOME_PATH=/usr/local/Ascend/cann-9.0.0
cmake -S cann -B build-npu -DCMAKE_BUILD_TYPE=Release \
  -DSTRATA_CANN_NPU_ONLY=ON
cmake --build build-npu --target model_seq npu_tensor_parity dequant_npu_parity -j8

python3 cann/python/native_generate.py \
  --engine ./build-npu/model_seq \
  --model /path/to/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf \
  --npu-only --device 1 --budget-gib 12 \
  --prompt 'The capital of France is' --max-tokens 8
```

两个 GGUF 分片必须在同一目录。Python 包装器仅做分词、启动 C++ 进程和解码文本，
不加载 Torch，也不通过其他推理服务生成。

直接运行二进制时设置 `STRATA_CANN_NPU_ONLY=1`；`STRATA_CANN_DEVICE` 指定设备，
`STRATA_BUDGET_GB` 指定解码后的权重缓存预算（GiB），不是进程显存上限。
NPU-only 当前只支持 `qwen4exp` 和温度 0。构建未启用、模型架构不符、量化类型不支持
或非零温度采样都会明确报错，不会改用 CPU 模型计算。

## 执行位置

| 内容 | NPU 路径 |
|---|---|
| GGUF 权重 | 原始压缩字节 H2D，AscendC 解码为设备 FP32 |
| embedding、PLE 表行 | 所需行在设备解码；PLE uint64 哈希也由设备内核执行 |
| HC、归一化、RoPE、激活、卷积 | ACLNN 设备算子组合 |
| GDN | 设备批量矩阵乘及状态更新，状态常驻 HBM |
| Attention、KV、index | 设备矩阵乘、softmax、gather；KV/index 设备 FP16 存储 |
| MoE | 设备 softmax/top-k、权重归一化、专家加权；仅读回专家 ID 控制文件读取 |
| 输出 | 设备 top-k/logsumexp；仅读回候选 token、候选分数与 logZ |

`qwen38_npu.cpp` 不调用 CPU `load_tensor_slice()` 或 host 浮点前向。
设备激活由 `npu_tensor.cpp` 管理，支持广播、非连续视图和设备复制。
`dequant_npu.cpp` 只上传原始字节；`ascendc/dequant_kernel.cpp` 完成位提取、码本查表和缩放。
正式推理不读回中间浮点激活。显式开启 `STRATA_CANN_TRACE` 时会额外复制诊断张量，
这些复制不参与推理计算；本次吞吐运行关闭 trace。

实际模型混合了 IQ2_XXS、IQ2_XS、IQ2_S、IQ3_XXS、IQ3_S、IQ4_NL、IQ4_XS、
Q2_0、Q4_K、Q5_K、Q6_K、F32、F16 和 BF16；设备解码均已覆盖，另支持 Q8_0。

## 已完成的实机检查

- 设备张量测试：非连续切片/转置、设备复制、广播矩阵乘、归约、top-k、类型转换及激活检查通过。
- NPU 解码：12 种量化格式的测试数据、F32/F16/BF16、两分片中各类型真实权重样本均逐位一致。
- PLE 哈希：48 个包括 uint64 溢出和大模数的检查逐项一致。
- 模型连续生成 8 个 token，与独立 CPU 浮点诊断参考相同：

```text
输入：The capital of France is
输出： Paris. The capital of Germany is Berlin
```

一轮完整前向的 CANN profiler 记录了 10,691 条设备算子记录，类型为 AI_VECTOR_CORE、
AI_CORE、MIX_AIV 和 MIX_AIC；AICPU 算子记录为 0。采集时开启了 `--aicpu=on`。
矩阵乘、设备解码、归一化、激活、Gather 等都在设备记录中。
最终模型运行的 `npu-smi info -t usages -i 1` 共采样 101 次，NPU 利用率峰值 100%，
采样算术均值 83.35%；AICPU 利用率采样均为 0。采样包含请求启动和结束阶段。
完整日志、最终二进制校验值和各轮时间见工作区 `results/20261106-strata-npu-only/`。

最终连续生成的 prefill 为 55.290317 s，decode 为 84.412630 s，即 0.082926 token/s
（7 次后续 decode / decode 时间）。这是单张卡、12 GiB 解码权重缓存、FP32 投影的
移植基线，不能代表优化后的性能。Profiler 显示首个前向中设备解码占设备算子累计耗时
约 99.5%；初版解码核主要使用 NPU 标量指令。当轮后续优化已增加向量化，
设备解码与矩阵乘仍是分开的内核。

## 本轮解决的问题

1. AscendC 禁止 unsigned→float 的直接标量转换：对已知小范围字段先转 int32 再转 FP32。
2. 厂商 kernel CMake 强制可执行文件以 PIE 链接：将本项目相关静态库及可执行文件启用 PIC/PIE 编译。
3. `aclCreateTensorList()` 接管张量描述符所有权：修正 concat 后重复销毁 descriptor 导致的 IndexSelect 错误和崩溃，并增加生命周期回归测试。
4. 禁止执行模式悄悄降级：不支持的模型、构建配置、量化类型和随机采样均拒绝执行。

本次验证范围是原生 CLI 短序列、单卡运行。双卡并行、HTTP 服务切换和长上下文性能
尚未验收；QSA 长序列的同分 top-k 顺序仍需单独对照。CPU 测试参考仅用于离线比较，
不是 NPU-only 推理路径的一部分。
