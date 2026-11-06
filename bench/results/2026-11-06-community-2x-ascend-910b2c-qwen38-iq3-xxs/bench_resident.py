"""Repeated requests against one loaded model_seq; preserves raw output and errors."""
import argparse
import json
import os
from pathlib import Path
import re
import selectors
import subprocess
import time

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--engine', required=True)
p.add_argument('--model', required=True)
p.add_argument('--out', required=True)
p.add_argument('--cases', required=True)
args = p.parse_args()
cases = json.loads(Path(args.cases).read_text(encoding='utf-8'))
prefix = Path(args.out)
results = {'engine': str(Path(args.engine).resolve()),
           'environment': {k: v for k, v in os.environ.items() if k.startswith('STRATA_')},
           'cases': []}
with prefix.with_suffix('.stderr.log').open('w') as err, prefix.with_suffix('.stdout.log').open('w') as log:
    start = time.monotonic()
    process = subprocess.Popen([args.engine, args.model, '0', 'serve'], stdin=subprocess.PIPE,
                               stdout=subprocess.PIPE, stderr=err, bufsize=0)
    selector = selectors.DefaultSelector()
    selector.register(process.stdout, selectors.EVENT_READ)
    pending = bytearray()

    def read_line(timeout=600):
        deadline = time.monotonic() + timeout
        while b'\n' not in pending:
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not selector.select(remaining):
                raise TimeoutError('model output timeout')
            block = os.read(process.stdout.fileno(), 65536)
            if not block:
                raise RuntimeError(f'model stdout ended: exit={process.poll()}')
            pending.extend(block)
        line, _, rest = pending.partition(b'\n')
        pending[:] = rest
        text = line.decode('utf-8', 'replace')
        log.write(text + '\n')
        log.flush()
        return text

    try:
        while read_line() != 'READY':
            pass
        results['load_ready_s'] = time.monotonic() - start
        for case in cases:
            request = f"RUN {case['generate']} 0 {','.join(map(str, case['tokens']))}\n"
            log.write('# ' + case['name'] + '\n')
            process.stdin.write(request.encode())
            process.stdin.flush()
            sample = {**case, 'generated_tokens': [], 'selected_logits': []}
            request_start = time.monotonic()
            while True:
                line = read_line()
                if line == 'END':
                    break
                if line.startswith('GEN '):
                    _, token, value = line.split()
                    sample['generated_tokens'].append(int(token))
                    sample['selected_logits'].append(float(value))
                elif line.startswith('TIMING '):
                    sample['timing'] = {k: float(v) for k, v in re.findall(r'(\w+)=([0-9.]+)', line)}
                elif line.startswith('ERR '):
                    raise RuntimeError(line)
            sample['request_wall_s'] = time.monotonic() - request_start
            assert len(sample['generated_tokens']) == case['generate']
            assert 'timing' in sample
            results['cases'].append(sample)
            prefix.with_suffix('.json').write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding='utf-8')
            print(case['name'], sample['timing'], flush=True)
        process.stdin.write(b'QUIT\n')
        process.stdin.flush()
        process.stdin.close()
        process.wait(timeout=60)
        if process.returncode:
            raise RuntimeError(f'model exit {process.returncode}')
    finally:
        selector.close()
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
