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
    p.add_argument('--device', type=int, default=0)
    p.add_argument('--budget-gib', type=float, default=12)
    args = p.parse_args()
    if args.max_tokens < 1 or args.device < 0 or not 3 <= args.budget_gib <= 48:
        p.error('max-tokens must be positive, device nonnegative, and budget between 3 and 48 GiB')
    tokenizer = Tokenizer.from_gguf(args.model)
    tokens = [int(x) for x in args.tokens.split(',')] if args.tokens is not None else tokenizer.encode(args.prompt, parse_special=True)
    if not tokens or any(t < 0 or t >= len(tokenizer.tokens) for t in tokens):
        p.error('prompt token IDs must be nonempty and inside the model vocabulary')
    env = dict(os.environ, STRATA_CANN_DEVICE=str(args.device), STRATA_BUDGET_GB=str(args.budget_gib))
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
    print(json.dumps(dict(backend='strata-cann-native', device=f'npu:{args.device}',
                          prompt_tokens=tokens, tokens=generated, text=tokenizer.decode(generated),
                          timing=timing, execution='CANN FP32 projections; CPU FP32 state and elementwise operations'),
                     ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
