"""Compare native model_seq traces with the independent Torch/CANN FP32 path.

Both traces must use the same prompt, generation length, model files and decoding
settings. This checks prefill residuals at every layer and all decode logits.
"""
import argparse
import json
from pathlib import Path

import numpy as np


def compare(native, oracle, oracle_result, oracle_logits, tolerance=0.002):
    result = json.loads(oracle_result.read_text())
    expected = np.fromfile(oracle_logits, dtype=np.float32).reshape(len(result['tokens']), -1)
    records = []
    worst_layer = worst_logits = worst_logprob = 0.0
    for step, token in enumerate(result['tokens']):
        position = len(result['prompt_tokens']) - 1 + step
        actual = np.fromfile(native / f'p{position}-logits.f32', dtype=np.float32)
        reference = expected[step]
        if actual.shape != reference.shape or not np.isfinite(actual).all():
            raise AssertionError(f'invalid logits at position {position}')
        top = np.argsort(actual)[-5:][::-1]
        ref_top = np.argsort(reference)[-5:][::-1]
        if top.tolist() != ref_top.tolist() or top[0] != token:
            raise AssertionError(f'top-five mismatch at {position}: {top} vs {ref_top}')
        log_z = np.log(np.exp(actual.astype(np.float64)-actual.max()).sum())+actual.max()
        ref_z = np.log(np.exp(reference.astype(np.float64)-reference.max()).sum())+reference.max()
        logprob_error = float(np.max(np.abs((actual[top]-log_z)-(reference[top]-ref_z))))
        logits_error = float(np.linalg.norm(actual-reference)/max(np.linalg.norm(reference),1e-12))
        worst_logits = max(worst_logits, logits_error)
        worst_logprob = max(worst_logprob, logprob_error)
        records.append(dict(position=position, token=int(token), top5=top.tolist(),
                            logits_relative_l2=logits_error, top5_logprob_error=logprob_error))
    count = 0
    for path in sorted(oracle.glob('position-*-layer-*.npy')):
        parts = path.stem.split('-')
        position, layer = int(parts[1]), int(parts[3])
        reference = np.load(path).reshape(-1)
        actual = np.fromfile(native / f'p{position}-l{layer}.f32', dtype=np.float32)
        if actual.shape != reference.shape or not np.isfinite(actual).all():
            raise AssertionError(f'invalid residual at {position}/{layer}')
        error = float(np.linalg.norm(actual-reference)/max(np.linalg.norm(reference),1e-12))
        worst_layer = max(worst_layer,error)
        if error > tolerance:
            raise AssertionError(f'residual relative L2 {error} exceeds {tolerance} at {position}/{layer}')
        count += 1
    # The Torch CLI's layer callback covers prefill; its logit dump covers every
    # generated token. Do not claim a decode-layer trace that it did not produce.
    if count != len(result['prompt_tokens'])*48:
        raise AssertionError(f'incomplete layer traces: {count}')
    if worst_logits > tolerance or worst_logprob > tolerance:
        raise AssertionError(f'logit/logprob errors {worst_logits}/{worst_logprob} exceed {tolerance}')
    return dict(passed=True, layer_comparisons=count, generated_tokens=result['tokens'],
                text=result['text'], max_layer_relative_l2=worst_layer,
                max_logits_relative_l2=worst_logits, max_top5_logprob_error=worst_logprob, steps=records)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--native', type=Path, required=True)
    p.add_argument('--oracle', type=Path, required=True)
    p.add_argument('--oracle-result', type=Path, required=True)
    p.add_argument('--oracle-logits', type=Path, required=True)
    p.add_argument('--tolerance', type=float, default=0.002)
    args = p.parse_args()
    print(json.dumps(compare(args.native,args.oracle,args.oracle_result,args.oracle_logits,args.tolerance),
                     ensure_ascii=False,indent=2))
