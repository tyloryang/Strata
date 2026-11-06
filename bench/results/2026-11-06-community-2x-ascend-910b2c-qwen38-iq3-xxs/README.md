# Strata CANN：双 Ascend 910B2C 上的 Qwen3.8 IQ3_XXS

实测日期：2026-11-06（UTC+8）。这是对 Strata 原生 CANN 开发路径的定向测量，不是 `strata bench run` 标准基准，也不代表桌面版或 Terminal 安装包已支持 Ascend。

按步骤从环境检查、Strata CANN 构建、Qwen3.8 适配到 API 压测的部署说明见[双 910B2C 部署手册](../../../cann/QWEN38_IQ3_XXS_910B2C_DEPLOY_CN.md)，其中也列出与 2×A100 CUDA 部署记录的口径差异。

## 环境与配置

| 项目 | 实测值 |
|---|---|
| NPU | 2 × Ascend 910B2C，设备 0、1；每卡 HBM 65,536 MB |
| CANN | 9.0 |
| 模型 | Qwen3.8-Flash-Next GSQ-RCO IQ3_XXS，GGUF 两分片 |
| 引擎 | Strata 原生 CANN `model_seq-grouped` |
| 引擎源码提交 | `c62972c` |
| 推理执行 | `STRATA_CANN_NPU_ONLY=1`；模型张量计算在两张 NPU 上，主机负责文件 I/O、tokenizer、调度 |
| 设备选项 | packed weights、fused matvec、dense cache、deferred submission、grouped experts；权重缓存预算 3 GiB |
| OS、驱动、CPU 型号 | 此轮报告中未记录 |

模型入口分片为 `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf`。完整配置和逐次结果见本目录的 JSON 附件。运行中每秒采样网关指标；两张卡都观察到 AICore 活动。

## 部署与访问

### 在 NPU 主机上检查、构建

```bash
cd /data/strata-npu-vector-v4-20261106
source ./env.sh
npu-smi info
ps -eo pid,args | grep '[m]odel_seq-grouped'
ss -lntp | grep -E ':(18789|18790|18791|443)\b' || true
curl -fsS http://127.0.0.1:18790/health
```

若源码、CANN 环境及 ggml 库已部署，运行随报告提供的 [`build-grouped.sh`](build-grouped.sh)：

```bash
bash ./build-grouped.sh
```

该脚本以 Release 模式、`STRATA_CANN_NPU_ONLY=ON` 编译 `model_seq-grouped`。脚本里的 `/data` 路径对应本次机器目录；其他机器需按本机路径调整。

### 在客户端建立 SSH 隧道

在本机 PowerShell 保持以下命令运行，替换主机和 SSH 用户：

```powershell
ssh -CNg -L 18790:127.0.0.1:18791 -p <SSH_PORT> <SSH_USER>@<NPU_HOST>
```

本地 API Base URL 为 `http://127.0.0.1:18790/v1`。本次验证通过 SSH 隧道；公网 TLS 地址没有完成端到端验证。

### API 冒烟与指标

```powershell
curl.exe -sS http://127.0.0.1:18790/v1/models `
  -H "Authorization: Bearer <API_KEY>"

curl.exe -sS http://127.0.0.1:18790/v1/completions `
  -H "Authorization: Bearer <API_KEY>" `
  -H "Content-Type: application/json" `
  -d '{"model":"qwen3.8-flash-next-iq3-xxs-strata-cann","prompt":"The capital of France is","max_tokens":8,"temperature":0}'

curl.exe -sS http://127.0.0.1:18790/metrics `
  -H "Authorization: Bearer <API_KEY>"

curl.exe -sS "http://127.0.0.1:18790/metrics?format=prometheus" `
  -H "Authorization: Bearer <API_KEY>"
```

监控页面：`http://127.0.0.1:18790/monitor`。本次 8-token 冒烟请求输出为 ` Paris. The capital of Germany is Berlin`。密钥只通过 Bearer 头传入，不保存在报告、脚本或结果 JSON 中。

## 同机解码对比

固定提示词 `The capital of France is`（5 tokens），每请求生成 32 tokens。权重在同一个引擎进程中保持驻留；每次请求通过 `model.reset()` 清空序列状态。每种配置连续测 3 次。

| 配置 | 引擎 | Deferred | Grouped experts | 解码速度中位数 | 三次范围 |
|---|---|---:|---:|---:|---:|
| 基线 | `model_seq-fused-v3b` | 关 | 关 | 2.174328 token/s | 2.159737–2.189527 |
| 优化 | `model_seq-grouped` | 开 | 开 | 10.949095 token/s | 10.942993–10.968804 |

在本次固定工作负载下，优化配置约为基线的 **5.04×**。基线和优化配置的逐请求 token、logits、耗时与环境变量分别保存在 [`resident-baseline.json`](resident-baseline.json) 和 [`resident-grouped.json`](resident-grouped.json)。比较时改变了引擎二进制，并开启 deferred submission 与 grouped experts；同机、同模型、同提示词、同生成长度。

IQ3_XXS grouped matvec 微基准也通过 CPU 双精度 oracle、分离 matvec 对照、重复运行和无效 ID 检查：

| 输入布局 | 形状 | Grouped 中位数 | 分离 matvec 中位数 | 加速比 |
|---|---:|---:|---:|---:|
| shared | 640 × 2560 | 0.250802 ms | 0.400811 ms | 1.598× |
| per-rank | 2560 × 640 | 0.312476 ms | 0.413801 ms | 1.324× |

每种布局 15 次测量，日志见 [`grouped-bench-gate.log`](grouped-bench-gate.log) 与 [`grouped-bench-down.log`](grouped-bench-down.log)。两份日志均以 `GROUPED_ALL_PASS` 结束。

## API 并发压测

运行 [`api-loadtest.py`](api-loadtest.py)：

```powershell
python -u .\bench\results\2026-11-06-community-2x-ascend-910b2c-qwen38-iq3-xxs\api-loadtest.py
```

脚本隐藏提示输入 API Key；先发 1 个 8-token 预热请求，再对并发 1、2、4 各执行每 worker 3 个请求，每次最多生成 24 tokens。正式压测 21 个请求全部成功，共生成 504 tokens；报告采集 36 个 NPU 指标样本，峰值活动请求为 4。

| 并发 | 请求成功/失败 | 批次生成吞吐 | 客户端延迟 P95 |
|---:|---:|---:|---:|
| 1 | 3 / 0 | 8.918 token/s | 2.674 s |
| 2 | 6 / 0 | 9.279 token/s | 5.221 s |
| 4 | 12 / 0 | 9.317 token/s | 10.325 s |

批次吞吐按该批生成 token 总数除以整批墙钟时间计算；客户端 P95 包含排队和 API 开销，不是纯 kernel decode 速率。并发从 1 增至 4 后，聚合吞吐约停在 9.3 token/s，P95 延迟则增加。

压测期间每卡 AICore 峰值为 NPU 0 **61%**、NPU 1 **65%**；最后采样分别为 59% 和 65%。最后采样的 HBM 使用量为 NPU 0 **59,289 / 65,536 MB**、NPU 1 **38,882 / 65,536 MB**。完整原始记录在 [`api-loadtest.json`](api-loadtest.json)。

## 限制与复现范围

- 当前最佳直接引擎 decode 结果约 **10.95 token/s**；API 并发压测聚合约 **9.32 token/s**，本次没有达到 100 token/s。
- API 测试是 21 个短请求的并发容量检查，不是长时间稳定性测试，也不是 Strata 标准 benchmark 协议。
- 公开报告只给出占位符主机名、SSH 用户及 API Key。服务本身使用的进程管理单元和 TLS 私钥不随报告发布；报告包含引擎构建、隧道、API 和指标复现命令。
- 旧的单卡 910B2C 1.93 token/s 结果采用另一硬件数量和工作负载，不能与本报告的双卡数字直接比较。
