# DeepSeek / Codex Ascend scripts (<服务器地址>)

Operational scripts written by the DeepSeek / Codex side while running
Qwen3.8-Flash-Next (IQ3_XXS) on the rented Ascend box
`<服务器地址>`, container `<容器 ID>`
(x86_64, 2 x Ascend 910B2C 64GB, CANN 9.0.0, openEuler 24.03).

Collected from the box 2026-11-06 ~00:23 CST, at which point the IQ3_XXS
download had just finished (both shards size-verified against the
ModelScope file list: shard 1 47,039,860,096 bytes, shard 2
28,800,138,432 bytes, mmproj 907,543,008 bytes).

## What is here

- `srv/` — 36 `srv_*.sh` helper scripts from `/root/` on the box:
  preflight / network diagnostics, download start/status/rate, llama.cpp
  build + patch, 0.6B smoke / split tests, GGUF/tensor dumps, CANN op
  probes, scheduling probes, status page. See
  `../DEEPSEEK_ASCEND_UPDATE_2026-11-06.md` for the per-script inventory.
- `qwen38/` — the three scripts for the big model, from `/data/`:
  `serve_qwen38.sh` (llama-server, CTX 32768, port 8080, layer split,
  mmap + lazy mode, vision off by default), `accept_qwen38.sh`
  (health / models / one timed Chinese chat turn / NPU busy check),
  `measure_qwen38.sh` (load time, prefill/decode, HBM, off-device ops).
  At collection time these had been written but **not yet run** on
  Qwen3.8 — the download had only just completed.
- `python/` — the on-box Python work from
  `/data/strata-src/cann/python/`: `serve.py`, `qwen38.py`,
  `weights.py`, `model_spec.py`, `prepare_model.py`,
  `status_page.py`, `requirements.txt`. This overlaps the `cann/python/`
  tree on the `codex/ascend910-qwen38-iq3-xxs` branch; kept here as the
  exact on-box copies.
- `status_page.py` — the status page served on the box
  (`http://127.0.0.1:8188`).

## What is NOT here

- Model files (76.75 GB IQ3_XXS set — stays on the box).
- Logs (`/data/logs/`, `/data/deepseek-monitor.log`).
- The llama.cpp patch tarball (`/data/llamacpp-patch.tar.gz`), the
  `cann-src.tar.gz` / `/data/strata-src/` C sources, build directories
  (`/data/build-cann`, `/data/build-strata`), and the atrace traces
  (`/root/ascend/atrace/`).
- Host credentials, SSH details, and internal network addresses used
  only for diagnostics are not in these scripts; the scripts were
  checked for secrets before upload and contain none.

## Status / caveats

- These are the other side's scripts, reproduced verbatim. Muse did
  **not** write them and has **not** verified that each one runs
  cleanly — only that the ones already executed on the box produced
  the results recorded in `../DEEPSEEK_ASCEND_UPDATE_2026-11-06.md`
  (two 100% llama.cpp builds, 0.6B smoke at ~182 tok/s generation,
  double-card layer-split test).
- Paths inside the scripts are absolute to the box (`/root/`,
  `/data/`); they are not runnable from this repo as-is.
- Do not merge this directory into the engine code — it is a record
  of the on-box operational work, not part of the Strata CANN port.
