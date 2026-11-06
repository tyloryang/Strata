# Strata CANN serving stack (Machine C, 2x Ascend 910B2C)

Runs the grouped-experts native engine (`model_seq-grouped`, serve mode)
behind an authenticated TLS gateway.

- `strata_iq3_api.py` — launcher for the OpenAI-compatible API on
  `127.0.0.1:18790` (engine `strata-cann`, tokenizer via the CPU
  llama-server on `127.0.0.1:18789`, used as the reference/oracle).
- `api_gateway.py` — TLS + Bearer-token gateway in front of `:18790`.
  Env: `STRATA_API_CLIENT_IP` (only this client IP is allowed),
  `STRATA_API_CERT`, `STRATA_API_KEY_FILE`, `STRATA_API_TOKEN_FILE`,
  `STRATA_API_MONITOR_HTML`. Serves `/metrics` and `monitor.html`.
- `monitor.html` — live NPU metrics / monitor page (token entered in the
  browser, kept in that tab only).

Secrets are never committed: the Bearer token, TLS key and cert live only
on the machine (`/data/strata-npu-vector-v4-20261106/api-token`,
`api-gateway.crt`, `api-gateway.key`).

Measured 2026-11-06: grouped-resident decode 10.89–10.97 t/s sustained
(32-token cases, exit 0); see `../bench/results/grouped-resident.json`.
