import json,statistics,sys
for file in sys.argv[1:]:
 data=json.load(open(file));print(file)
 baseline={r['repetition']:r for r in data['samples'] if r['variant']=='phased'}
 for name,w in dict.fromkeys((r['variant'],r['workers']) for r in data['samples']):
  rows=[r for r in data['samples'] if (r['variant'],r['workers'])==(name,w)]
  ratio=statistics.median(r['gpu_us']/baseline[r['repetition']]['gpu_us'] for r in rows)
  print(name,w,'gpu_us',round(statistics.median(r['gpu_us'] for r in rows),2),'wall_us',round(statistics.median(r['wall_us'] for r in rows),2),'paired_ratio',round(ratio,3))
