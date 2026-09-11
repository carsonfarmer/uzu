"""Summarize all retained runs; synthetic scheduling costs are not model speed."""
import json, statistics, sys
for path in sys.argv[1:]:
    data = json.load(open(path))
    print(path, 'correctness runs:', data['correctness_runs'])
    baseline = {s['repetition']:s for s in data['samples'] if s['variant']=='phased'}
    for name,workers in dict.fromkeys((s['variant'],s['workers']) for s in data['samples']):
        rows = [s for s in data['samples'] if s['variant']==name and s['workers']==workers]
        gpu = statistics.median(s['gpu_us'] for s in rows)
        wall = statistics.median(s['wall_us'] for s in rows)
        ratio = statistics.median(s['gpu_us']/baseline[s['repetition']]['gpu_us'] for s in rows)
        print(f'{name:14s} workers={workers:3d} GPU={gpu:9.3f}us wall={wall:9.3f}us paired/phased={ratio:.3f} thermal={sorted({s["thermal_state"] for s in rows})}')
