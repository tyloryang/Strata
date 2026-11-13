# Community benchmark: 2x NVIDIA A100-SXM4-40GB, IQ3_XXS, 262K context

Measured on 2026-11-02. This is a record of the last configuration that
completed a full benchmark round on this machine, not a recommended
configuration. The configuration changed several times on the same day, the
machine has since been retired, and the numbers below are single runs, so
do not treat them as a settled conclusion.

## Hardware and software

- GPU: 2x NVIDIA A100-SXM4-40GB, driver 550.78
- CPU: 32 cores
- RAM: 94 GiB
- OS: CentOS 7
- Deployment: Strata in Docker, host port 8800 mapped to container port 8080
- Model: Qwen3.8-Flash-Next IQ3_XXS
- Context: 262,144 tokens

## Model and configuration

Config file: `strata-iq3_xxs.json` (23:28 version, 2026-11-02). Credentials
have been removed from the configuration reproduced here.

```text
--pack /data/Strata-data/packs/iq3_xxs
--native /data/Strata-data/models/IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf
--ple-gguf /data/Strata-data/models/IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00002-of-00002.gguf
--expert-profile /data/Strata-data/data/expert-profile.bin
--expert-cache auto
--resident-experts
--prefill auto:32768
--spec 4
--spec-min-p 0.5
--mtp <mtp runtime>
--max-context 262144
--kv int8
--kv-resident 32768
--ple-io mmap
--layer-split auto
--parallel 8
--conversation-cache-mib 32768
--conversation-cache-slots 8
--conversation-cache-min-tokens 4096
--reasoning-budget-tokens 4096
```

Other settings in the same configuration:

- ESP control vector: scale 1.0, layers 4-44, project / per-layer
- Sampling: temperature 1.0, top_p 0.95, top_k 20
- Reasoning: effort medium, `reasoning_close_retry` true, `fit_max_tokens` true
- `tail-skip 0`, adapt overlap on
- `STRATA_BATCH_MTP_SERIAL=0` (pipeline mode)

Important: the file still listed `--batch-mtp`, but in pipeline grouping
mode the engine disables batch-mtp. There is no "parallel MTP" mode on a
layer-split machine: it is either serial window + `--batch-mtp`, or
pipeline grouping without batch-mtp.

## Method

- Tool: evalscope perf, prompt 2000 tokens, output 100 tokens
- Thinking off (`reasoning_effort none`), temperature 0, seed 1234
- Concurrency arms: 1 / 2 / 3 / 4 / 6 / 8
- Round: c2000-v14, 2026-11-02 23:32-23:37, 220.25 s, 104/104 requests
  successful
- This round ran under the 23:28 configuration above. Earlier rounds on
  the same day ran under different configurations (different
  `--layer-split`, `--ple-io`, conversation-cache size and
  `STRATA_BATCH_MTP_SERIAL`), so do not mix their numbers with this one.

## Results

Total average: **43.29 tok/s**.

| Concurrency | Output tok/s | TTFT mean (ms) | TPOT mean (ms) |
| ---: | ---: | ---: | ---: |
| 1 | 29.50 | 1959 | 14.46 |
| 2 | 39.57 | 1930 | 29.33 |
| 3 | 47.89 | 2820 | 34.69 |
| 4 | 51.27 | 3631 | 42.06 |
| 6 | 34.00 | 4801 | 128.74 |
| 8 | 52.16 | 6032 | 88.26 |

Per-run data are in `results.json` in this folder.

## Correctness and limitations

- Single run only, no retest. The total average was +18.5% against the
  previous round (v13, 36.54 tok/s, `--layer-split 28`), with the low
  concurrency arms recovering most, but that comparison is also
  single-run against single-run.
- Concurrency 6 is anomalous (mean latency 17.43 s, TPOT p99 332 ms,
  worse than concurrency 8) and was probably polluted by a concurrent
  long thinking request. Do not use it as a configuration score.
- Spec Accept Rate stayed at only about 1-2.1% in this round, so
  speculative decoding had effectively not taken effect here, despite
  `--spec 4` and MTP being configured.
- Earlier the same day, two longer-prompt rounds (in4096-out256 and
  in8192-out256, thinking on) measured 65.77 tok/s and 49.04 tok/s in
  total. Those used different workloads and thinking settings and are
  not directly comparable with the c2000-v14 numbers above.
- No quality / recall check was run in this round. No hipBLASLt-style
  tuning table applies here: that mechanism is AMD HIP only, and the
  CUDA backend on A100 has no equivalent tuning-table file.
