"""Text/token CLI for Strata's native C++ model_seq engine; no Torch import."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools'))
from strata_tokenizer import Tokenizer


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--engine', type=Path, default=ROOT / 'build-cann/model_seq')
    p.add_argument('--model', type=Path, required=True, help='original shard 00001-of-00002.gguf')
    prompt = p.add_mutually_exclusive_group(required=True)
    prompt.add_argument('--prompt', help='raw text completion prompt, without a chat template')
    prompt.add_argument('--tokens', help='comma-separated prompt token IDs')
    p.add_argument('--max-tokens', type=int, default=8)
    device = p.add_mutually_exclusive_group()
    device.add_argument('--device', type=int, help='one NPU ordinal; default: environment or 0')
    device.add_argument('--devices', help='one or two distinct NPU ordinals, for example 0,1; default: environment')
    p.add_argument('--budget-gib', type=float, help='weight budget; default: STRATA_BUDGET_GB or 12')
    p.add_argument('--npu-only', action='store_true', help='require device weight decoding and device-resident model computation')
    p.add_argument('--packed-weights', action='store_true', help='retain original compressed weights in NPU HBM')
    p.add_argument('--fused-matvec', action='store_true', help='fuse NPU dequantization and dot products; requires packed weights')
    args = p.parse_args()
    env = dict(os.environ)
    if args.device is not None:
        env['STRATA_CANN_DEVICE'] = str(args.device)
        env.pop('STRATA_CANN_DEVICES', None)
    elif args.devices is not None:
        env['STRATA_CANN_DEVICES'] = args.devices
    selected = env.get('STRATA_CANN_DEVICES', env.get('STRATA_CANN_DEVICE', '0'))
    parts = selected.split(',')
    if not 1 <= len(parts) <= 2 or any(not re.fullmatch(r'[0-9]+', part) for part in parts):
        p.error('devices must contain one or two nonnegative integer ordinals')
    devices = [int(part) for part in parts]
    if len(set(devices)) != len(devices):
        p.error('device ordinals must be distinct')
    env['STRATA_CANN_DEVICE'] = str(devices[0])
    if 'STRATA_CANN_DEVICES' in env:
        env['STRATA_CANN_DEVICES'] = ','.join(map(str, devices))
    if args.budget_gib is not None:
        env['STRATA_BUDGET_GB'] = str(args.budget_gib)
    env.setdefault('STRATA_BUDGET_GB', '12')
    try:
        budget_gib = float(env['STRATA_BUDGET_GB'])
    except ValueError:
        p.error('budget must be numeric')
    if args.max_tokens < 1 or not 3 <= budget_gib <= 48:
        p.error('max-tokens must be positive and budget between 3 and 48 GiB')
    if args.npu_only:
        env['STRATA_CANN_NPU_ONLY'] = '1'
    if args.packed_weights:
        env['STRATA_CANN_PACKED_WEIGHTS'] = '1'
    if args.fused_matvec:
        env['STRATA_CANN_FUSED_MATVEC'] = '1'
    npu_only = env.get('STRATA_CANN_NPU_ONLY') == '1'
    packed_weights = env.get('STRATA_CANN_PACKED_WEIGHTS') == '1'
    fused_matvec = env.get('STRATA_CANN_FUSED_MATVEC') == '1'
    dense_cache = env.get('STRATA_CANN_DENSE_CACHE') == '1'
    deferred = env.get('STRATA_CANN_DEFERRED') == '1'
    grouped = env.get('STRATA_CANN_GROUPED_EXPERTS') == '1'
    if deferred and not fused_matvec:
        p.error('STRATA_CANN_DEFERRED=1 requires fused matvec')
    if grouped and not fused_matvec:
        p.error('STRATA_CANN_GROUPED_EXPERTS=1 requires fused matvec')
    if dense_cache and not fused_matvec:
        p.error('STRATA_CANN_DENSE_CACHE=1 requires fused matvec')
    if fused_matvec and not packed_weights:
        p.error('fused matvec requires --packed-weights or STRATA_CANN_PACKED_WEIGHTS=1')
    if (len(devices) > 1 or packed_weights or fused_matvec) and not npu_only:
        p.error('multiple devices and packed/fused execution require --npu-only or STRATA_CANN_NPU_ONLY=1')
    tokenizer = Tokenizer.from_gguf(args.model)
    tokens = [int(x) for x in args.tokens.split(',')] if args.tokens is not None else tokenizer.encode(args.prompt, parse_special=True)
    if not tokens or any(t < 0 or t >= len(tokenizer.tokens) for t in tokens):
        p.error('prompt token IDs must be nonempty and inside the model vocabulary')
    generated, timing = [], {}
    with subprocess.Popen([str(args.engine.resolve()), str(args.model.resolve()), ','.join(map(str,tokens)),
                           str(args.max_tokens), '0'], stdout=subprocess.PIPE, text=True, env=env) as process:
        try:
            for line in process.stdout:
                print(line.rstrip(), file=sys.stderr, flush=True)
                if line.startswith('GEN '):
                    generated.append(int(line.split()[1]))
                elif line.startswith('TIMING '):
                    timing = {key: float(value) for key,value in re.findall(r'(\w+)=([0-9.]+)',line)}
            code = process.wait()
        except BaseException:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            raise
    if code:
        raise SystemExit(code)
    if not generated or not timing:
        raise RuntimeError('native engine returned no generation/timing record')
    print(json.dumps(dict(backend='strata-cann-native', device=','.join(f'npu:{d}' for d in devices),
                          devices=devices, npu_only=npu_only, packed_weights=packed_weights,
                          fused_matvec=fused_matvec, dense_cache=dense_cache, deferred_submission=deferred,
                          grouped_experts=grouped, budget_gib=budget_gib,
                          prompt_tokens=tokens, tokens=generated, text=tokenizer.decode(generated),
                          timing=timing, execution=('AscendC weight decoding; CANN device activations and state; host I/O and scheduling'
                              if npu_only else
                              'CANN FP32 projections; CPU FP32 state and elementwise operations')),
                     ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
