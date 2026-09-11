"""Repeat deterministic coding requests against the local Uzu OpenAI server."""
import argparse
import json
import time
import urllib.request

parser = argparse.ArgumentParser()
parser.add_argument('--url', default='http://127.0.0.1:18000')
parser.add_argument('--runs', type=int, default=5)
parser.add_argument('--tokens', type=int, default=128)
parser.add_argument('--prompt', choices=['short', 'long', 'rust'])
args = parser.parse_args()
# Repeated source context grows the input; report actual server token counts.
prompts = [
 ('short', 'Write a Python function that merges two sorted lists. Include one test.'),
 ('long', '\n'.join(f'def helper_{i}(x): return x + {i}' for i in range(128)) + '\nWrite a Python function that merges two sorted lists. Include one test.'),
 ('rust', 'Write a Rust function that finds the first duplicate in a slice of u32. Include tests for an empty slice and a duplicate.'),
]
for label, prompt in prompts:
 if args.prompt and label != args.prompt:
  continue
 for repetition in range(-1, args.runs):
  body = dict(model='local', messages=[dict(role='user',content=prompt)], temperature=0, max_tokens=args.tokens, enable_thinking=False, stream=False)
  req = urllib.request.Request(args.url+'/v1/chat/completions',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
  start=time.perf_counter()
  with urllib.request.urlopen(req,timeout=600) as response: result=json.load(response)
  print(json.dumps(dict(prompt=label,repetition=repetition,warmup=repetition<0,wall_seconds=time.perf_counter()-start,response=result)),flush=True)
