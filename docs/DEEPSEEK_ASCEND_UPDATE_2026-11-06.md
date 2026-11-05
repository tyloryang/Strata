# DeepSeek / Codex 在昇腾机上的更新记录

记录时间：2026-11-06 00:07 CST（2026-11-05 16:07 UTC），由 Muse 上机核实后记录。
机器：<服务器地址>，新容器 c86d60b5759f，x86_64，openEuler 24.03，128 核，128Gi 内存，
2× Ascend 910B2C 各 64GB，CANN 9.0.0，npu-smi 26.0.rc1，/data 118G。
本文件只记录已核实的内容；机上的 `srv_*.sh` 等脚本本体仍在机器上，不在 git 里。

## 1. 模型下载（ModelScope，正在进行）

命令（13:25 UTC 启动，进程 PID 1339）：

```
modelscope download --model ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF \
  --include IQ3_XXS/* --include mmproj-Qwen3.8-Flash-Next-BF16.gguf \
  --local_dir /data/models/IQ3_XXS
```

- 分片 2（28.80GB）：已下完，15:36 UTC 去掉 `.incomplete` 后缀。
- 分片 1（47.04GB）：16:08 UTC 时 40.5G/47.0G，86%，约 6.8–7.1 MB/s，日志自估剩约 15 分钟。
- mmproj（0.91GB）：已下完。
- 整套 76.75GB：约 68.4G/76.75G（89%），预计 00:25 CST 前后下完。
- 另有 Qwen3-0.6B 的 F16 GGUF（1.5GB，`/data/models/qwen3-0.6b-f16.gguf`），已用于冒烟测试。

## 2. llama.cpp + CANN（已跑通 0.6B）

- 源码：`/data/llama.cpp-master`（tar 包 `/data/llama.cpp-master.tar.gz` 38MB）。
- 构建：`/data/build-cann`，13:29 UTC 第一次 100% 完成；14:59 UTC 第二次构建也 100%
  完成（UI 资源因 HF 超时没下到，建成无内置 UI 版，不影响跑模型）。
- 补丁：`/data/llamacpp-patch.tar.gz`（83KB），内容为 `ggml/src/ggml-cann/` 的
  aclnn_ops / acl_tensor / ggml-cann 等文件；应用脚本 `/root/srv_apply_patch.sh`。
- 冒烟（Qwen3-0.6B，CANN 卸载已验证）：
  - llama-cli：prompt 约 1387–16230 t/s，生成约 181–186 t/s。
  - llama-server：8081 端口，4 槽、2048 上下文，跑过任务后正常退出。
  - 22:10 CST 又起一轮双卡分层测试 `--split-mode layer -ts 0.5,0.5`。
  - server 日志提示：CANN 不支持 Flash Attention over K/V rows，已自动禁用。

## 3. Qwen3.8-Flash-Next 的 serve / 验收 / 测速脚本（已写好，未跑大模型）

- `/data/serve_qwen38.sh`：llama-server，默认 ctx 32768、端口 8080、双卡
  `--split-mode layer`，`-lm mmap -lzm on`（28.8GB 的 n-gram 表只按行读，留在磁盘上
  mmap），vision 默认 off（`VISION=on` 才加载 mmproj）。
- `/data/accept_qwen38.sh`：验收脚本，`/health`、`/v1/models`、一次中文 chat
  （“用一句话说明什么是混合专家模型”，max_tokens 96，temperature 0），再查两张卡是否都在忙。
- `/data/measure_qwen38.sh`：基线测速，打印加载时间、prefill/decode 速率、每卡 HBM、
  以及所有被移出设备的 fused op。
- `/data/status_page.py`：状态页，`http://127.0.0.1:8188`。

## 4. 机上 36 个 srv_*.sh 脚本（/root/，13:44–16:04 UTC 陆续写出）

按用途分组（只列文件名与一句话用途，内容以机上文件为准）：

- 下载：`srv_start_download.sh`、`srv_dl_status.sh`、`srv_dl_rate.sh`、
  `srv_dl_rate2.sh`、`srv_dl_rate3.sh`、`srv_find_partial.sh`、
  `srv_range_test.sh`（试 ranged 请求是否比 modelscope CLI 快）。
- 构建与补丁：`srv_build_x86.sh`、`srv_build_dump.sh`、`srv_build_dump2.sh`、
  `srv_apply_patch.sh`、`srv_check_build.sh`、`srv_flags.sh`。
- 冒烟与分层：`srv_smoke_06b.sh`、`srv_smoke_4b.sh`、`srv_split_test.sh`、
  `srv_split_verify.sh`、`srv_split_verbose.sh`、`srv_server_test.sh`、
  `srv_measure_06b.sh`、`srv_speed_cmp.sh`、`srv_show_logs.sh`。
- GGUF / 张量检查：`srv_dump_gguf.sh`、`srv_dump_tensors.sh`、`srv_shard2.sh`
  （核对分片 2 是否含分片 1 缺的 output_norm / PLE 等张量）、
  `srv_partial_load.sh`（用不完整分片试 loader 是否认得架构与 metadata）、
  `srv_show_floats.sh`（float metadata 被截断的问题）、`srv_test_topk.sh`。
- 算子与调度：`srv_cann_ops.sh`、`srv_op_sigs.sh`、`srv_sched_probe.sh`。
- 状态与收尾：`srv_preflight.sh`、`srv_preflight_final.sh`、
  `srv_status_page.sh`、`srv_page_check.sh`。

## 5. Strata CANN 的 parity 工作（机上已构建）

- 源码包：`/data/cann-src.tar.gz`（717KB），解压到 `/data/strata-src/`，
  内含 `cann/`（`src/`、`tests/`、`include/`、`patches/`、`python/`）。
- 构建：`/data/build-strata/`，产物 `port_check`、`gguf_dump`、`libstrata_cann.a`，
  以及 parity 测试：attn_cache / attn_layer / batched / gdn / gdn_block /
  gdn_fused / gdn_state / gguf / layer / model_forward / model_seq /
  moe / quant / rope / topk / cann_parity / bench_op。
- 文档（`/data/strata-src/cann/`）：`ASCEND_910B_PORT_STATUS.md`、
  `QWEN38_ARCH_GAP.md`、`QWEN38_CANN_OP_PATCH.md`、`QWEN38_GDN_DELTA.md`、
  `QWEN38_IQ3_XXS_ASCEND.md`（含中文版 `_CN`）、`QWEN38_PLE_QSA.md`、
  `QWEN38_TENSORS.md`，补丁 `patches/llama-cpp-cann-topk.patch`，
  Python 部分 `python/`（`serve.py`、`qwen38.py`、`weights.py`、`model_spec.py` 等）。
- 注意：这份 `cann/` 与本分支 `ascend-cann` 的 Phase 1（GEMM + 注意力，86b286cf）
  是平行的工作，不是同一个提交历史，见第 6 节。

## 6. GitHub fork 上的 Codex 分支（已同步，2026-11-06 00:07 核实）

Fork：https://github.com/tyloryang/Strata

- `codex/ascend910-qwen38-iq3-xxs-squashed`，头 05b724a5
  `cann: keep float metadata; 1e-6 was being truncated to 0`。
  相对上游 ffa80d73 共 13 个提交：Qwen3.8 IQ3_XXS 后端、中文指南、NPU 指标、
  真实张量表与两处修正、PLE head dim 是 160 不是 320、float metadata 修正等。
- `codex/ascend910-qwen38-iq3-xxs`（未 squash），头 7a99f616
  `docs: document Qwen3.8 Ascend backend and sanitize host notes`。
  其中记录的实测数字（以分支内提交信息为准，未在 Muse 这边复测）：
  批量 GDN 12 tokens 17.8s → 14.7s；专家 slab 并行解码 5 tokens 37s → 12.7s；
  多线程反量化 5-token prompt 166s → 41s；权重常驻 166s → 64s；
  serve 模式 + OpenAI 兼容端点 18790；fp32 状态 5/5 与 llama.cpp 一致。
- 相对本分支 `ascend-cann`：92 个文件改动，+13362 / -3212 行。
  其中 `cann/tests/model_forward.cpp`（902 行）、`model_seq.cpp`、`moe_parity.cpp`、
  `serve_engine.py`、`verify_all.sh` 等为新增；本分支的
  `test_cann_attention.cpp`、`test_cann_gemm.cpp`、`docs/CANN_PORT_PLAN.md`
  在该分支中被删掉（由 parity 套件取代）。
- `main` 分支也被推到 a3869dd1（与未 squash 分支同头）。

## 7. 监控

- 机上监控：`/data/deepseek-monitor.sh`，PID 7296（13:35 UTC 起），每 60 秒写
  `/data/deepseek-monitor.log`（进程 + NPU + 端口 + 内存/磁盘 + 下载进度）。
- <旧 Ascend 机>上的监控 PID 24755 仍在那台机上跑，未迁移。

## 8. 待确认

- 分片 1 下完后是否立即跑 `serve_qwen38.sh` + `accept_qwen38.sh`，还是先跑
  `measure_qwen38.sh` 做基线——由用户定。
- Codex 分支与本分支 `ascend-cann` 的合并方式未定：两边都改了 `cann/`，
  直接合会冲突，需先对齐测试套件（Phase 1 的 test_cann_* vs parity 套件）。
- `/root/ascend/atrace` 下有 4 组 atrace 跟踪（14:52–14:59 UTC），未细读。

## Update 2026-11-06 00:23 CST

The scripts themselves are now in this repo under `docs/ascend-deepseek/` (36 `srv_*.sh`, `serve/accept/measure_qwen38.sh`, the on-box Python files, `status_page.py`), collected after the IQ3_XXS download completed and size-verified. See that directory's README for provenance and caveats.
