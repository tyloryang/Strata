# Qwen3.8 IQ3_XXS 在双 Ascend 910B2C 上的 Strata 部署手册

- 更新时间：2026-11-06（UTC+8）
- 适用环境：2 × Ascend 910B2C、CANN 9.0、Qwen3.8-Flash-Next GSQ-RCO IQ3_XXS
- 实现：Strata 原生 C++/CANN 开发路径 `model_seq-grouped`

这份手册按本次实机部署记录整理，覆盖引擎适配、构建、NPU-only 验证、API 接入和压测。它不是 Strata 桌面版安装指南；Ascend 路径仍是独立的开发端口。文中主机名、SSH 用户和 API Key 都用占位符表示。

## 结果速览

| 测量 | 本次 2 × 910B2C |
|---|---:|
| 常驻权重直接解码，中位数 | 10.949 token/s |
| 旧版 CANN 引擎直接解码中位数 | 2.174 token/s |
| 同机优化幅度 | 5.04× |
| API 聚合吞吐，并发 1 / 2 / 4 | 8.918 / 9.279 / 9.317 token/s |
| API 正式请求成功率 | 21 / 21 |
| NPU AICore 压测峰值 | NPU 0：61%；NPU 1：65% |

本次没有达到 100 token/s。直接解码和 API 聚合吞吐是两种不同口径，不能混为一谈。逐请求数据、原始 JSON、构建脚本和压测脚本见[实测报告与附件](../bench/results/2026-11-06-community-2x-ascend-910b2c-qwen38-iq3-xxs/README.md)。

## 1. 部署结构

```text
客户端 / SDK
    │  本机 http://127.0.0.1:18790/v1
    │  SSH 本地转发
    ▼
远端 API 鉴权/指标网关 127.0.0.1:18791
    │
    ├── Strata CANN 推理 API 127.0.0.1:18790
    │       └── model_seq-grouped → NPU 0、NPU 1
    │
    └── llama.cpp tokenizer 服务 127.0.0.1:18789
            └── 仅分词/解码文本；本次配置 --n-gpu-layers 0
```

模型的权重解码、激活、注意力、递归状态、路由和矩阵计算都在两张 NPU 上。CPU 仍负责文件 I/O、tokenizer、进程调度和 HTTP 层；“NPU-only”表示模型张量计算不回退到 CPU，并不表示主机 CPU 完全不参与请求处理。

这是 `cann/` 下独立的原生 NPU 开发路径，不是 CUDA 版引擎或 Strata 桌面/Terminal 安装包，也还没有接入通用 `serve/` 启动器。当前 HTTP API 由单独的服务包装层接入该 CANN 引擎。

本次客户端通过 SSH 隧道访问已运行的 API 网关。公网 TLS 端点没有完成端到端验证。目标机上的 API 服务管理单元和 TLS 私钥不在公开仓库中；后文给出已验证的服务检查、SSH 隧道和 API 调用方式，不伪造一个未经实机验证的 systemd 配置。要在新机器上完整复刻 API 常驻服务，需要由部署方配置该服务包装层和进程管理器；引擎编译及 NPU 推理命令在本手册中可复现。

## 2. 部署前准备

### 2.1 确认硬件和 CANN

在目标 Linux 主机确认两张卡都可见，并加载 CANN 9.0 环境：

```bash
npu-smi info
source /usr/local/Ascend/cann/set_env.sh
export ASCEND_HOME_PATH=/usr/local/Ascend/cann-9.0.0
```

实测设备为 NPU 0 和 NPU 1，每卡 HBM 65,536 MB。该轮没有记录操作系统版本、驱动版本和 CPU 型号；部署到其他机器时请先记录这些值，不要据此假设驱动兼容性。

### 2.2 准备源代码、GGML/CANN 依赖和模型

引擎代码已合入 Strata，IQ3_XXS 双卡优化提交为 `c62972c`。可从仓库取该代码版本：

```bash
git clone https://github.com/tyloryang/Strata.git /data/strata-npu-vector-v4-20261106
cd /data/strata-npu-vector-v4-20261106
git checkout c62972c
```

构建需要 CANN/AscendC 工具链，以及 Strata CMake 配置使用的 llama.cpp/ggml 源码和已构建库。本次机器上的位置如下；其他机器按实际安装位置替换：

| 组件 | 本次路径 |
|---|---|
| Strata 源码 | `/data/strata-npu-vector-v4-20261106` |
| llama.cpp/ggml 源码 | `/data/llama.cpp-master` |
| ggml 构建库 | `/data/build-cann/bin` |
| IQ3_XXS 模型入口分片 | `/data/models/IQ3_XXS/IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf` |

两个 GGUF 分片必须保持在同一目录。将模型路径记入 shell 变量，并确认分片存在：

```bash
export STRATA_ROOT=/data/strata-npu-vector-v4-20261106
export MODEL=/data/models/IQ3_XXS/IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf
ls -lh "${MODEL%00001-of-00002.gguf}"*gguf
```

`env.sh` 是实测机上的本地环境初始化文件，不包含在公开仓库中。已有同类部署环境可沿用它；新机器至少要先加载 CANN，并确认 `bisheng`、CMake、编译器和 ggml 库能被当前 shell 找到。

## 3. Strata 如何适配 Qwen3.8 和 IQ3_XXS

这不是把 CUDA kernel 改一个设备号。Strata 在 `cann/` 下增加了原生 C++/CANN 执行路径，按模型结构、GGUF 权重和 NPU kernel 分层适配：

1. **识别正确的模型架构。** GGUF 标记为 `qwen4exp`，有 48 层、hidden size 2560，并包含 HyperConnection、PLE、GDN、QSA 和 MoE（512 个专家、每 token 选 10 个）。这些结构不能按普通 Qwen3/Qwen3.6 残差模型处理。
2. **读取两分片 GGUF。** 加入 Qwen3.8 元数据和张量映射，并支持从两个 GGUF 分片按需读取模型张量；入口传 `00001-of-00002.gguf`。
3. **把量化解码放到 AscendC。** IQ3_XXS 使用码本/位域编码，不能当作普通整数矩阵直接乘。AscendC kernel 在 NPU 上解析压缩权重；packed-weight 模式让压缩权重保持常驻 HBM。
4. **把完整模型图接到 CANN 设备张量。** 权重投影、激活、注意力、GDN 状态、路由和输出在设备侧执行。`STRATA_CANN_NPU_ONLY=1` 用于拒绝不支持的配置，避免静默回退成 CPU 模型计算。
5. **针对 decode 瓶颈融合和分组。** fused matvec 把解量化与点积放到同一条 NPU 路径；deferred submission 减少小算子逐次提交开销；grouped experts 将选中专家按批组织计算。两张设备通过 `STRATA_CANN_DEVICES=0,1` 参与推理。

主要实现入口包括 [`qwen38.cpp`](src/qwen38.cpp)、[`qwen38_npu.cpp`](src/qwen38_npu.cpp)、[`tensor_load.cpp`](src/tensor_load.cpp)、[`dequant_kernel.cpp`](ascendc/dequant_kernel.cpp)、[`fused_matvec.hpp`](ascendc/fused_matvec.hpp) 和 [`model_seq.cpp`](tests/model_seq.cpp)。

## 4. 编译双卡 NPU-only 引擎

加载实测机环境并设置项目路径。以下命令与报告附件中的 `build-grouped.sh` 一致：

```bash
cd "$STRATA_ROOT"
source ./env.sh

cmake -S cann -B build-grouped \
  -DCMAKE_BUILD_TYPE=Release \
  -DSTRATA_CANN_NPU_ONLY=ON \
  -DSTRATA_CANN_GGML_DIR=/data/llama.cpp-master \
  -DSTRATA_CANN_GGML_LIB_DIR=/data/build-cann/bin

cmake --build build-grouped -j8
cp build-grouped/model_seq model_seq-grouped
```

如果源码、ggml 或构建目录不在本次路径，先修改 `-DSTRATA_CANN_GGML_DIR` 和 `-DSTRATA_CANN_GGML_LIB_DIR`。本次执行选项以 `STRATA_CANN_NPU_ONLY=ON` 编译，并将生成的 `model_seq` 复制为便于区分的 `model_seq-grouped`。

## 5. 设置运行参数并先做小样本验证

正式压测前先固定双 NPU、NPU-only 和优化开关：

```bash
export STRATA_CANN_DEVICES=0,1
export STRATA_CANN_NPU_ONLY=1
export STRATA_CANN_PACKED_WEIGHTS=1
export STRATA_CANN_FUSED_MATVEC=1
export STRATA_CANN_DENSE_CACHE=1
export STRATA_CANN_DEFERRED=1
export STRATA_CANN_GROUPED_EXPERTS=1
export STRATA_BUDGET_GB=3
```

运行一条短生成，确认二进制、模型和 CANN 环境能协同工作：

```bash
python3 cann/python/native_generate.py \
  --engine ./model_seq-grouped \
  --model "$MODEL" \
  --devices 0,1 \
  --npu-only \
  --packed-weights \
  --fused-matvec \
  --budget-gib 3 \
  --prompt 'The capital of France is' \
  --max-tokens 8
```

该包装器直接读取 GGUF tokenizer，不导入 Torch。预期结果中 `device` 包含 `npu:0` 和 `npu:1`、`npu_only` 为 `true`，并有生成文本和 timing。NPU 利用率可另开终端观察：

```bash
npu-smi info
```

本次 API 冒烟输出样本为 `Paris. The capital of Germany is Berlin`。示例文本用于确认生成链路，不作为模型质量评测。

## 6. 跑 IQ3_XXS 分组 kernel 门禁

编译完成后，先运行 parity 测试，再跑 shared 和 per-rank 两种布局的 15 次微基准：

```bash
./build-grouped/grouped_matvec_parity

./build-grouped/grouped_matvec_parity \
  --types IQ2_XXS,IQ2_XS,IQ3_XXS,IQ3_S,IQ2_S,Q2_0,IQ4_NL \
  --rows 640 --cols 2560 --selected 10 --input shared \
  --bench --warmup 3 --repeats 15 --deferred \
  > grouped-bench-gate.log 2>&1

./build-grouped/grouped_matvec_parity \
  --types IQ2_XXS,IQ2_XS,IQ3_XXS,IQ3_S,IQ2_S,Q2_0,IQ4_NL \
  --rows 2560 --cols 640 --selected 10 --input per-rank \
  --bench --warmup 3 --repeats 15 --deferred \
  > grouped-bench-down.log 2>&1

grep GROUPED_ALL_PASS grouped-bench-gate.log grouped-bench-down.log
```

实测两份日志都以 `GROUPED_ALL_PASS` 结束。IQ3_XXS shared 形状 `640 × 2560` 的 grouped 中位数为 0.250802 ms，分离 matvec 为 0.400811 ms（1.598×）；per-rank `2560 × 640` 为 0.312476 ms 对 0.413801 ms（1.324×）。微基准只表示这些 kernel 形状的耗时，不等于整模型 token/s。

## 7. 复现常驻模型 decode 对比

`bench_resident.py` 在一个 `model_seq` 进程里连续执行 case，每个 case 前重置序列状态；它测的是模型内部的 resident decode，不经过 HTTP/API 网关。用例定义在 `resident-cases.json`。

```bash
cd "$STRATA_ROOT/bench/results/2026-11-06-community-2x-ascend-910b2c-qwen38-iq3-xxs"
export STRATA_CANN_DEVICES=0,1
export STRATA_CANN_NPU_ONLY=1
export STRATA_CANN_PACKED_WEIGHTS=1
export STRATA_CANN_FUSED_MATVEC=1
export STRATA_CANN_DENSE_CACHE=1
export STRATA_CANN_DEFERRED=1
export STRATA_CANN_GROUPED_EXPERTS=1
export STRATA_BUDGET_GB=3

python3 bench_resident.py \
  --engine "$STRATA_ROOT/model_seq-grouped" \
  --model "$MODEL" \
  --out resident-grouped \
  --cases resident-cases.json
```

这轮固定提示词为 5 tokens，每次生成 32 tokens，每个配置重复 3 次。旧版基线二进制 `model_seq-fused-v3b` 是对比时使用的历史构建；候选二进制由本手册第 4 节构建。结果 JSON 记录引擎、运行环境、生成 token、logits 与 timing。

| 配置 | 中位 decode | 三次范围 |
|---|---:|---:|
| `model_seq-fused-v3b`，deferred/grouped 关闭 | 2.174328 token/s | 2.159737–2.189527 |
| `model_seq-grouped`，deferred/grouped 开启 | 10.949095 token/s | 10.942993–10.968804 |

本组优化把该固定 workload 的解码速度提高 5.04×。因为二进制和运行选项同时变化，这个结果衡量的是整组优化，不是单独某一个开关的因果增益。

## 8. 检查已部署 API、建 SSH 隧道并调用

### 8.1 远端确认服务

在 NPU 主机执行：

```bash
cd /data/strata-npu-vector-v4-20261106
source ./env.sh
npu-smi info
ps -eo pid,args | grep '[m]odel_seq-grouped'
ss -lntp | grep -E ':(18789|18790|18791|443)\b' || true
curl -fsS http://127.0.0.1:18790/health
```

端口用途：远端 `18789` 是 tokenizer helper；远端 `18790` 是 Strata CANN API；远端 `18791` 是 API Key 鉴权与指标网关。机器上的服务单元不在公开材料中，若端口已有进程，不要重复启动同一服务或覆盖现有配置。

### 8.2 从 Windows 建立 SSH 隧道

在本机 PowerShell 启动隧道并保持窗口运行：

```powershell
ssh -CNg -o ExitOnForwardFailure=yes `
  -L 127.0.0.1:18790:127.0.0.1:18791 `
  -p <SSH_PORT> <SSH_USER>@<NPU_HOST>
```

SSH 密码由客户端提示输入，不要放进命令、脚本或手册。之后本机 API Base URL 为 `http://127.0.0.1:18790/v1`。

### 8.3 API 冒烟和监控

把 `<API_KEY>` 换成部署时配置的值，不要提交到文件或命令历史：

```powershell
curl.exe -sS http://127.0.0.1:18790/v1/models `
  -H "Authorization: Bearer <API_KEY>"

curl.exe -sS http://127.0.0.1:18790/v1/completions `
  -H "Authorization: Bearer <API_KEY>" `
  -H "Content-Type: application/json" `
  -d '{"model":"qwen3.8-flash-next-iq3-xxs-strata-cann","prompt":"The capital of France is","max_tokens":8,"temperature":0}'
```

浏览器指标页：`http://127.0.0.1:18790/monitor`。JSON 和 Prometheus 指标分别为 `/metrics` 与 `/metrics?format=prometheus`，请求都需带 Bearer Key。公网 TLS 地址没有端到端验收，本手册以 SSH 隧道为已验证访问方式。

## 9. 运行 API 并发压测

确保 SSH 隧道保持连接后，在 Strata 仓库根目录运行已发布脚本：

```powershell
python -u .\bench\results\2026-11-06-community-2x-ascend-910b2c-qwen38-iq3-xxs\api-loadtest.py
```

脚本用隐藏输入读取 API Key；先预热 1 个 8-token 请求，然后对并发 1、2、4 各发每 worker 3 个正式请求，每请求最多生成 24 tokens。每秒采样一次网关 NPU 指标，结果写入 `api-loadtest.json`。

正式请求使用提示词 `The capital of France is`（实测 5 tokens）、`temperature=0`；因此这组结果是短提示、小输出容量测试，不代表长文档或长上下文服务能力。

| 并发 | 成功 / 失败 | 聚合输出吞吐 | 客户端 P95 |
|---:|---:|---:|---:|
| 1 | 3 / 0 | 8.918 token/s | 2.674 s |
| 2 | 6 / 0 | 9.279 token/s | 5.221 s |
| 4 | 12 / 0 | 9.317 token/s | 10.325 s |

21 个正式请求全部成功，共生成 504 tokens。36 个指标样本中，两卡 AICore 峰值分别为 61% 和 65%；最后采样的 HBM 使用量为 NPU 0：59,289 / 65,536 MB、NPU 1：38,882 / 65,536 MB。该脚本的聚合吞吐按一档内生成 token 数除以整批墙钟时间计算，客户端 P95 包含排队和 API 开销；它不是纯 kernel decode 数字，也不是 `strata bench run` 标准基准。

## 10. 和 2 × NVIDIA A100 的实测对比

同一用户的 [Strata-A100-Deploy 仓库](https://github.com/tyloryang/Strata-A100-Deploy)记录了 Qwen3.8-Flash-Next IQ3_XXS 在 2 × A100-SXM4-40GB 上的离线部署。该报告给出单请求 decode 105–111 token/s；并发 8 时，短提示自然中文（in≈800、8 槽）聚合 89.3 token/s，长提示（in=2000、4 槽）聚合 59.5 token/s。A100 环境为 Strata 0.1.45、CUDA 12.4、驱动 550.78；测试机于 2026-11-02 回收，数字来自当时记录。[部署记录](https://github.com/tyloryang/Strata-A100-Deploy/blob/main/docs/%E9%83%A8%E7%BD%B2%E6%96%87%E6%A1%A3.md) · [压测方法与记录](https://github.com/tyloryang/Strata-A100-Deploy/blob/main/docs/%E5%8E%8B%E6%B5%8B%E6%96%87%E6%A1%A3.md)

| 指标 | 双 910B2C / CANN 9.0 | 双 A100-40GB / CUDA 12.4 | 说明 |
|---|---:|---:|---|
| 单请求 / resident decode | 10.949 token/s | 105–111 token/s | 910 是当前原生 CANN 引擎 5-token prompt、32-token 输出；A100 是其部署记录的单请求值，软件和投机配置不同 |
| 短提示并发聚合 | API 并发 4：9.317 token/s | 自然中文 in≈800、并发 8：89.3 token/s | 约 9.6× 数值差距，负载和并发不匹配，不能当成严格性能比 |
| 长提示并发聚合 | 本次未测相同场景 | in=2000、并发 8：59.5 token/s | 910 没有对应的 2000-token 压测数据 |

若只看公开报告中的单请求 decode 数字，A100 为 910 当前值的约 **9.6–10.1 倍**。这说明目前这版 910 路径离 A100 项目达到的速度还有明显距离，但不能把该倍数直接解释为芯片算力比：两边 Strata 引擎实现、并发调度、请求长度、投机解码配置和测量方法都不一致，且 A100 数字是 4 天前的历史记录，并未在同一脚本、同一 prompt 和同一软件提交上重跑。

不能把上述差距直接归因为芯片。A100 项目使用 Strata CUDA 引擎、双卡分层、专家常驻和多轮并发参数调优；910 这里是新接入的 CANN 原生执行路径，当前 API 压测也只有 1–4 并发的短请求。两边引擎成熟度、kernel 融合、调度和测量流程都有差异。

就 910 数据本身，可确认 grouped/deferred 等优化使 resident decode 相比本机旧版提升 5.04×；API 并发 1 到 4 时吞吐只从 8.918 增到 9.317 token/s，P95 延迟增加。该轮 AICore 采样峰值未达到 100%，只能说明这些采样没有观测到满载，不能据此锁定唯一瓶颈或推断性能上限。下一轮应在两边统一模型文件、prompt/output 长度、预热、并发和 API 计时口径，并同步采集逐算子 profiler 与 NPU 利用率。

## 11. 已知边界

- 当前最优测得 resident decode 为约 10.95 token/s；本轮没有达到 100 token/s。
- API 压测仅 21 个短请求，不是长时间稳定性、最大上下文或生产 SLA 验收。
- 操作系统、910 驱动和 CPU 型号未记录；新机器需补齐环境清单。
- 公开附件包含运行/压测脚本和结果 JSON，但 CANN 环境初始化 `env.sh`、机器服务单元、API Key、TLS 私钥不发布。
- Qwen3.8 的长上下文 QSA 同分选择、桌面安装包集成以及公网 TLS 入口不属于本轮验收范围。
