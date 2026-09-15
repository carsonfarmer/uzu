"""Measure the complete fusion/async combination against both stock controls."""
import argparse
import fcntl
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--prompts', nargs='+', choices=['short', 'rust', 'long'],
                        default=['short', 'rust', 'long'])
    parser.add_argument('--tokens', type=int, default=128)
    parser.add_argument('--runs', type=int, default=10,
                        help='Multiple of ten for balanced order; zero for a correctness smoke run')
    args = parser.parse_args()
    if args.runs < 0 or args.runs % 10 or args.tokens < 2:
        parser.error('runs must be a nonnegative multiple of ten; tokens must be at least two')
    # Keep research processes on this machine from competing for the same GPU.
    with open('/tmp/north-metal-research-gpu.lock', 'a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        run(args)


def run(args):
    import mlx.core as mx
    from transformers import AutoTokenizer
    from reference import ROOT, load
    from layers import variants
    from decode import generate

    if mx.__version__ != '0.32.2':
        raise RuntimeError('This exact arithmetic is validated against MLX 0.32.2')
    model, config = load()
    paths = variants(model)
    base = ROOT / 'work/models/North-Mini-Code-1.0-4bit'
    tokenizer = AutoTokenizer.from_pretrained(str(base), local_files_only=True)
    tokenizer.chat_template = (base / 'chat_template.jinja').read_text()
    prompts = {
        'short': 'Return only Python code for merging two sorted lists. Include assertions for empty inputs and duplicate values.',
        'rust': 'Return only Rust code for finding the first duplicate in a u32 slice, with tests for empty input and duplicates.',
        'long': '\n'.join(f'def helper_{i}(x): return x + {i}' for i in range(128)) + '\nReturn only Python code for merging two sorted lists. Include assertions for empty inputs and duplicate values.',
    }
    modes = {
        'original_host': ('original', 'host'),
        'original_async': ('original', 'async'),
        'fused_host': ('fused', 'host'),
        'fused_async': ('fused', 'async'),
        'prepared_async': ('prepared', 'async'),
    }
    eos = config['eos_token_id']
    eos = {eos} if isinstance(eos, int) else set(eos)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    here = Path(__file__).resolve().parent
    # Exclusive creation prevents accidentally erasing an earlier run.
    with output.open('x') as file:
        def save(row):
            file.write(json.dumps(row) + '\n')
            file.flush()

        sources = sorted(here.glob('*.py')) + sorted(here.glob('*.h'))
        sources += sorted((here / 'provenance').glob('*.json'))
        runtime = ROOT / 'work/mlx-vlm'
        loaded_sources = {}
        for name, module in list(sys.modules.items()):
            path = getattr(module, '__file__', None)
            if name.startswith('mlx_vlm.') and path and Path(path).suffix == '.py':
                loaded_sources[str(Path(path).relative_to(runtime))] = hashlib.sha256(Path(path).read_bytes()).hexdigest()
        manifest = json.loads((here / 'provenance/source.json').read_text())['sha256']
        for name, digest in loaded_sources.items():
            assert manifest.get(name) == digest, ('Unverified runtime source', name)
        save(dict(kind='provenance', args=vars(args), mlx=mx.__version__, device=mx.device_info(),
                  packages={d.metadata['Name']: d.version for d in importlib.metadata.distributions()},
                  batching_environment={k: os.environ.get(k) for k in ('MLX_MAX_MB_PER_BUFFER', 'MLX_MAX_OPS_PER_BUFFER')},
                  sources={str(s.relative_to(here)): hashlib.sha256(s.read_bytes()).hexdigest() for s in sources},
                  loaded_runtime_sources=loaded_sources,
                  scope='128 output tokens by default: first from untimed prefill, 127 full-model timed decode steps, including full-vocabulary logits, argmax, all token readbacks and final GPU drain. Fixed length; early EOS rejected. Stock host control reads argmax.item directly without a separate logits eval. Async controls use identical submission. No outlier removal.'))

        def call(ids, path, submission, retain=False):
            model.model.layers = paths[path]
            return generate(model, ids, args.tokens, submission, eos, retain=retain)

        try:
            for label in args.prompts:
                ids = tokenizer.apply_chat_template(
                    [dict(role='user', content=prompts[label])], tokenize=True,
                    return_dict=False, add_generation_prompt=True, reasoning=False, skip_thinking=True)
                expected, reference, _, _ = call(ids, 'original', 'host', retain=True)
                for name, settings in modes.items():
                    tokens, actual, _, _ = call(ids, *settings, retain=True)
                    assert tokens == expected, name
                    assert len(reference) == len(actual) == args.tokens - 1
                    for step, (ref, got) in enumerate(zip(reference, actual)):
                        assert ref.shape == got.shape and ref.dtype == got.dtype
                        unequal = int(mx.sum(ref.view(mx.uint8) != got.view(mx.uint8)).item())
                        save(dict(kind='correctness', prompt=label, variant=name, step=step,
                                  shape=list(ref.shape), dtype=str(ref.dtype), unequal_bytes=unequal))
                        assert unequal == 0, (name, step, unequal)
                    del actual
                    gc.collect()
                del reference
                gc.collect()
                print(label, 'full-logit gate passed', args.tokens - 1, flush=True)
                for rep in range(-1, args.runs):
                    names = list(modes)
                    if rep >= 0:
                        shift = rep % len(names)
                        names = names[shift:] + names[:shift]
                        if (rep // len(names)) % 2:
                            names.reverse()
                    for position, name in enumerate(names):
                        tokens, _, elapsed, intervals = call(ids, *modes[name])
                        assert tokens == expected, name
                        save(dict(kind='generation', prompt=label, prompt_tokens=len(ids), variant=name,
                                  repetition=rep, position=position, warmup=rep < 0, token_ids=tokens,
                                  decode_steps=args.tokens - 1, decode_seconds=elapsed,
                                  decode_tokens_per_second=(args.tokens - 1) / elapsed,
                                  host_iteration_seconds=intervals))
                        print(label, name, rep, round((args.tokens - 1) / elapsed, 3), flush=True)
                        gc.collect()
        finally:
            mx.synchronize()
            model.model.layers = paths['original']


if __name__ == '__main__':
    main()
