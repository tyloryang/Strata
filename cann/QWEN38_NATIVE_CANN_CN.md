# Strata 原生 C++ 在 910B2C 上运行 IQ3_XXS

2026-11-06 实测：**可以生成文本**。使用本仓库 `cann/` 的 C++ `Model` 和
`model_seq` 二进制，读取原始 Qwen3.8-Flash-Next GSQ-RCO IQ3_XXS 两分片。
运行过程不加载 Torch，也不调用 llama.cpp 或其他推理服务。

这仍是移植正确性路径。矩阵投影使用 CANN FP32 在 NPU 执行；归一化、路由、
卷积、递归状态更新和注意力状态计算使用 CPU FP32。**尚未实现全算子 NPU 化，
也尚未接入 `strata launch` 的后端自动选择。**

## 从哪里修起

修改前直接运行原有 `model_seq`，返回退出码 1：

```text
FAILED: model: missing tensor output_norm.weight
```

原加载路径按 Qwen3.6 的 40 层、2048 hidden 和普通 RMSNorm/残差处理；实际文件
是 `qwen4exp`，有 48 层、2560 hidden，并使用 HyperConnection 输出归一化。
修复后，`Model` 按 GGUF 架构分发，Qwen3.8 进入新的 `src/qwen38.cpp`，
原 `qwen35moe` 计算分支保持独立。

新路径包括双分片读取、PLE 的 uint64 哈希与按行解码、HyperConnection、GDN、
GQA/RoPE、MoE 512 选 10、共享专家和输出投影。QSA 稀疏选择代码已接入，
但本次短序列未触发 2048-token 以上的稀疏分支，不能据此认定长上下文验证通过。

## 构建和运行

使用服务器上已安装的 CANN 9.0 工具链，两个 GGUF 文件放在同一目录，名称保留
`00001-of-00002` 和 `00002-of-00002`。从仓库根目录运行：

```bash
STRATA_CANN_DEVICE=1 STRATA_BUDGET_GB=12 \
  bash cann/tests/run_qwen38_native.sh \
  /data/models/IQ3_XXS/IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf \
  785 8
```

`STRATA_CANN_DEVICE=1` 选择逻辑第二张卡；单卡机器设为 `0`。
`STRATA_BUDGET_GB` 是权重缓存预算，按 GiB 计算，不是整个进程的 HBM 上限。
临时缓冲区、池化缓冲和运行时也会占用 HBM。

也可以直接运行已构建二进制：

```bash
source /usr/local/Ascend/cann/set_env.sh
STRATA_CANN_DEVICE=1 STRATA_BUDGET_GB=12 \
  ./build-cann/model_seq /path/to/00001-of-00002.gguf 785 8 0
```

参数依次为模型第一分片、逗号分隔的输入 token、生成数量和温度；本次验收温度为 0。
这是固定生成长度的原始 completion 工具，不自动套聊天模板。

### 直接输入文字

`native_generate.py` 只负责 Strata 分词和 C++ 子进程输入输出，不参与模型推理：

```bash
python3 cann/python/native_generate.py \
  --engine ./build-cann/model_seq \
  --model /path/to/00001-of-00002.gguf \
  --prompt 'The capital of France is' \
  --max-tokens 8 --device 1 --budget-gib 12
```

也可用 `--tokens 785` 代替 `--prompt`，精确复现下面的验收输入。
此文本包装器依赖仓库分词器所用的 `regex`，不依赖 `torch`/`torch_npu`。

## 本次实机结果

单张 Ascend 910B2C、CANN 9.0.0、驱动 26.0.rc1；使用双卡机器的逻辑 `npu:1`，
12 GiB 权重缓存、FP32 投影。原有 8080 和 18791 服务未停止，验收后健康检查均为 200。

输入 `[785]`，连续贪心生成 8 个 token：

```text
2405, 314, 1070, 995, 978, 303, 279, 2956
ither of them had been in the room
```

| 检查 | 实测结果 |
|---|---|
| C++ 原生构建 | `model_seq` 构建成功 |
| 运行退出码 | 0 |
| NPU 进程 | `npu-smi` 观察到 `model_seq`，采样进程占用约 17086 MiB |
| NPU 投影调用 | 8 token 累计 17704 次 CANN matmul |
| 输入处理时间 | 20.270402 s，包含首轮输出投影 |
| decode 时间 | 132.883181 s |
| decode 速度 | 0.052678 tok/s，7 次后续 decode / decode 时间 |
| 总请求耗时 | 约 153.2 s，开启了层与 logits trace |
| 贪心 token | 与参考路径 8/8 一致 |
| 每步前 5 名候选 | 8/8 同序 |
| 48 层 prefill 残差最大相对 L2 | 0.000242768 |
| 8 步 logits 最大相对 L2 | 0.000263039 |
| 前 5 名最大 logprob 差 | 0.001031667 |
| 常驻进程重置 | 两次相同请求均输出 token 2405，LOGP 均为 14.166368，退出码 0 |

此次对照对象是仓库内独立的 Torch/CANN FP32 路径；没有重新运行独立 CPU oracle。
逐层残差只比较了首个输入的 48 层，8 次生成比较的是完整 logits 和前 5 名候选。

12 GiB 缓存下本次报告的实际命中数为 0，持续解码和上传权重仍有明显开销。
没有做吞吐优化，也没有完成长上下文、多请求并发、聊天质量或 Qwen3.6 回归验收。

## 对照复现

原生执行设置 `STRATA_CANN_TRACE=/path/to/native-trace`，会保存
`p0-l0.f32` 等残差文件和 `p0-logits.f32` 等词表输出。
参考路径使用 `qwen38.py` 的 `--dump-layers`、`--dump-logits`、`--output`。
两边必须使用相同的模型、输入 token、生成长度和温度 0。

```bash
python3 cann/tests/compare_qwen38_native.py \
  --native /path/to/native-trace \
  --oracle /path/to/torch-layer-trace \
  --oracle-result /path/to/torch-result.json \
  --oracle-logits /path/to/torch-logits.f32
```

脚本要求前 5 名顺序一致、生成 token 一致，并检查残差、logits 和 logprob 误差；
缺文件、NaN、形状错误和超阈值均返回失败。

## 当前部署位置

服务器的独立源目录为 `/data/strata-native-20261106`，二进制为其中的
`build/model_seq`，实测记录为 `native-eight.log`、`comparison.json`、`reset.log`。
这些是本次服务器路径，其他机器请按实际部署位置调整。

本次实测二进制 SHA-256：
`7885a89ba51d9389aa22b1cb1f31a3a937dcd491a4bfaca49782cd2d88ff569d`。

现有 `18791` HTTP 服务和 Strata 监控中的请求统计仍属于 Torch 后端。
本次首先验证原生 C++ 引擎可运行，没有把这次命令行测试冒充为已切换的 HTTP 服务。
