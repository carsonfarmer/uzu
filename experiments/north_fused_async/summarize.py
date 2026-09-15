"""Report every measured repetition; pair variants within the same round."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import random
import statistics as stats


MODES = ('original_host', 'original_async', 'fused_host', 'fused_async', 'prepared_async')
COMPARISONS = (
    ('fused_async', 'original_host'),
    ('prepared_async', 'original_host'),
    ('fused_host', 'original_host'),
    ('fused_async', 'fused_host'),
    ('fused_async', 'original_async'),
    ('prepared_async', 'original_async'),
    ('prepared_async', 'fused_async'),
)


def summarize(path):
    records = [json.loads(line) for line in path.read_text().splitlines()]
    provenance = records[0]
    args = provenance['args']
    assert args['runs'] > 0 and args['runs'] % 10 == 0
    checks = [r for r in records if r['kind'] == 'correctness']
    assert all(r['unequal_bytes'] == 0 for r in checks)
    generations = [r for r in records if r['kind'] == 'generation' and not r['warmup']]
    report = {'input': path.name, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
              'method': 'Paired geometric mean of baseline seconds / candidate seconds. 95% percentile bootstrap over complete rounds, 20,000 resamples, seed 20260914. No measured samples excluded. Intervals describe these rounds on one shared desktop, not cross-device generalization.',
              'full_logit_array_comparisons': len(checks), 'measured_generations': len(generations),
              'prompts': {}}
    for prompt in args['prompts']:
        rows = [r for r in generations if r['prompt'] == prompt]
        assert len(rows) == args['runs'] * len(MODES)
        assert len({tuple(r['token_ids']) for r in rows}) == 1
        assert all(len(r['token_ids']) == args['tokens'] and r['decode_steps'] == args['tokens'] - 1 for r in rows)
        table = {}
        medians, ranges = {}, {}
        for mode in MODES:
            selected = [r for r in rows if r['variant'] == mode]
            assert {r['repetition'] for r in selected} == set(range(args['runs']))
            assert Counter(r['position'] for r in selected) == Counter({i: args['runs'] // 5 for i in range(5)})
            gates = [r for r in checks if r['prompt'] == prompt and r['variant'] == mode]
            assert len(gates) == args['tokens'] - 1
            assert {r['step'] for r in gates} == set(range(args['tokens'] - 1))
            table[mode] = {r['repetition']: r['decode_seconds'] for r in selected}
            rates = [(args['tokens'] - 1) / r['decode_seconds'] for r in selected]
            medians[mode] = stats.median(rates)
            ranges[mode] = [min(rates), max(rates)]
        comparisons = {}
        for candidate, baseline in COMPARISONS:
            ratios = [table[baseline][rep] / table[candidate][rep] for rep in range(args['runs'])]
            logs = [math.log(ratio) for ratio in ratios]
            rng = random.Random(20260914)
            boot = sorted(math.expm1(stats.mean(rng.choices(logs, k=len(logs)))) * 100 for _ in range(20000))
            comparisons[f'{candidate}/{baseline}'] = {
                'gain_percent': math.expm1(stats.mean(logs)) * 100,
                'bootstrap_95_percent': [boot[500], boot[19499]],
                'paired_gain_percent': [(r - 1) * 100 for r in ratios],
            }
        report['prompts'][prompt] = {'prompt_tokens': rows[0]['prompt_tokens'],
                                    'median_tokens_per_second': medians,
                                    'range_tokens_per_second': ranges,
                                    'comparisons': comparisons}
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = summarize(args.input)
    with args.output.open('x') as file:
        json.dump(report, file, indent=2)
        file.write('\n')
    for prompt, values in report['prompts'].items():
        print(prompt, {name: round(value, 3) for name, value in values['median_tokens_per_second'].items()})
        for name, comparison in values['comparisons'].items():
            print(' ', name, round(comparison['gain_percent'], 3),
                  [round(x, 3) for x in comparison['bootstrap_95_percent']])
