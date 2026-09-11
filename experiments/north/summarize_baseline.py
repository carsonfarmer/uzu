import json,statistics,sys
from pathlib import Path
rows=[json.loads(line) for line in Path(sys.argv[1]).read_text().splitlines()]
summary=[]
for name in dict.fromkeys(r['prompt'] for r in rows):
 rs=[r for r in rows if r['prompt']==name and not r['warmup']]
 if not rs:continue
 summary.append({'prompt':name,'runs':len(rs),'prompt_tokens':sorted(set(r['prompt_tokens'] for r in rs)),
  'generated_tokens':[r['generated_tokens'] for r in rs],'median_decode_tps':statistics.median(r['decode_tokens_per_second'] for r in rs),
  'median_ttft_seconds':statistics.median(r['ttft_seconds'] for r in rs),'median_wall_seconds':statistics.median(r['wall_seconds'] for r in rs),
  'peak_memory_bytes':max(r['peak_memory_bytes'] for r in rs),'identical_tokens':all(r['token_ids']==rs[0]['token_ids'] for r in rs),
  'finish_reasons':sorted(set(r['finish_reason'] for r in rs))})
print(json.dumps(summary,indent=2))
