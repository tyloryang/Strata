# Ascend: the CANN backend

Strata on Huawei Ascend 910B/910C through its CANN backend. This page
is for people with an Ascend card and for anyone building the engine
from source. It follows the shape of `docs/AMD_HIP.md`.

## Status

Experimental, Phase 0. It does not serve a model yet. The only thing
built so far is `strata_cann_probe`, a no-model self-test that
initialises the CANN runtime and prints the visible NPU count. Do not
expect a model to load or answer on this backend until a later phase
says so in `docs/CANN_PORT_PLAN.md`.

Verified 2026-11-04 on the first target (910B2C, CANN 9.0.0): the
probe compiles with g++ 12.3.1 against `libascendcl` and runs with
exit 0, printing `CANN runtime ok, NPU count=1` and
`current device=0`.

## Supported cards

| Card | Status |
| --- | --- |
| Ascend 910B2C (single card, 64 GiB HBM) | First verified target, 2026-11-04: CANN 9.0.0, one device visible as `/dev/davinci14`, x86_64 openEuler 24.03 container host. Probe only, no model yet. |
| Ascend 910B (other variants) | Not yet verified. |
| Ascend 910C (2 dies per board) | Not yet verified, no target machine yet. |

The first target is x86_64, not Kunpeng aarch64. Check `uname -m`
on your own box before assuming either.

## Quick start (probe only)

On a machine with CANN installed:

```sh
cmake -S cann -B build-cann -G Ninja
cmake --build build-cann
./build-cann/strata_cann_probe
```

Exit 0 means the runtime sees at least one NPU. If configure fails,
it says CANN was not found; install CANN 9.0.0 (the version on the
first verified target) or pass `-DSTRATA_CANN_ROOT=<path>`.

From the top level, the same pattern as the SYCL backend:

```sh
cmake -S . -B build -DSTRATA_ENABLE_CANN=ON
```

Without that flag the top-level build never enters `cann/`, and the
default CUDA/HIP/SYCL builds are unchanged.

## While the port is in progress

The ready-made way to run a model on this hardware today is not
Strata: vLLM-Ascend supports Qwen3.6-35B-A3B (including its Gated
DeltaNet layers). Use **vLLM-Ascend v0.22.1** on a CANN 9.0.0 box
(the version alignment on the first target); v0.23 needs CANN 9.0.1
and v0.30 needs CANN 9.2.0. The BF16 weights of Qwen3.6-35B-A3B
do not fit a single 64 GiB card; the W8A8 weights are the only
single-card candidate and are officially unverified on a single 910B.
