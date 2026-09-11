"""Print medians and paired relative changes; never infer model speed from these."""
import json
import statistics
import sys
for path in sys.argv[1:]:
    with open(path) as source:
        data = json.load(source)
    print(path)
    assert all(check['bit_mismatches'] == 0 for check in data['checks'])
    for iterations in sorted({s['iterations'] for s in data['samples']}):
        samples = [s for s in data['samples'] if s['iterations'] == iterations]
        baseline = [s for s in samples if not s['optimized']]
        fused = [s for s in samples if s['optimized']]
        ratios = [next(f['gpu_us'] for f in fused if f['repetition']==b['repetition'])/b['gpu_us'] for b in baseline]
        print(f'  iterations={iterations}: GPU median us baseline={statistics.median(s["gpu_us"] for s in baseline):.3f}, fused={statistics.median(s["gpu_us"] for s in fused):.3f}; paired median change={(statistics.median(ratios)-1)*100:+.2f}%')
        print(f'    wall median us baseline={statistics.median(s["wall_us"] for s in baseline):.3f}, fused={statistics.median(s["wall_us"] for s in fused):.3f}; thermal states={sorted({s["thermal_state"] for s in samples})}')
