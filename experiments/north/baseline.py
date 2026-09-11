"""Greedy text decoding; separate prefill/decode timing, fresh KV per request."""
import argparse,gc,json,time
from pathlib import Path
import mlx.core as mx
from transformers import AutoTokenizer
from reference import ROOT,load
parser=argparse.ArgumentParser()
parser.add_argument('--runs',type=int,default=3)
parser.add_argument('--tokens',type=int,default=192)
parser.add_argument('--output',default='experiments/north/results/baseline.jsonl')
parser.add_argument('--prompt',choices=['short','long','rust'])
args=parser.parse_args()
path=ROOT/'work/models/North-Mini-Code-1.0-4bit'
print('Loading pinned North model...',flush=True)
start=time.perf_counter()
model,config=load(path)
print('Loaded',time.perf_counter()-start,'seconds; active GB',mx.get_active_memory()/1e9,flush=True)
tokenizer=AutoTokenizer.from_pretrained(str(path),local_files_only=True)
tokenizer.chat_template=(path/'chat_template.jinja').read_text()
eos=config['eos_token_id']
eos={eos} if isinstance(eos,int) else set(eos)
prompts=[
 ('short','Return only Python code for merging two sorted lists. Include assertions for empty inputs and duplicate values.'),
 ('long','\n'.join(f'def helper_{i}(x): return x + {i}' for i in range(128))+'\nReturn only Python code for merging two sorted lists. Include assertions for empty inputs and duplicate values.'),
 ('rust','Return only Rust code for finding the first duplicate in a u32 slice, with tests for empty input and duplicates.'),
]
out=ROOT/args.output;out.parent.mkdir(parents=True,exist_ok=True)
with out.open('w') as f:
 for label,prompt in prompts:
  if args.prompt and label!=args.prompt:continue
  ids=tokenizer.apply_chat_template([dict(role='user',content=prompt)],tokenize=True,return_dict=False,add_generation_prompt=True,reasoning=False,skip_thinking=True)
  for repetition in range(-1,args.runs):
   cache=model.make_cache();mx.reset_peak_memory()
   start=time.perf_counter()
   # Chunk prefill to bound scratch allocation.
   for offset in range(0,len(ids),256):
    logits=model(mx.array([ids[offset:offset+256]]),cache=cache).logits
    mx.eval(logits)
   token=int(mx.argmax(logits[0,-1]).item())
   ttft=time.perf_counter()-start
   generated=[token];step_times=[]
   while len(generated)<args.tokens and token not in eos:
    t=time.perf_counter()
    logits=model(mx.array([[token]]),cache=cache).logits
    token=int(mx.argmax(logits[0,-1]).item())
    step_times.append(time.perf_counter()-t)
    generated.append(token)
   elapsed=time.perf_counter()-start
   row=dict(prompt=label,repetition=repetition,warmup=repetition<0,prompt_tokens=len(ids),generated_tokens=len(generated),token_ids=generated,
    text=tokenizer.decode(generated,skip_special_tokens=True),finish_reason='stop' if token in eos else 'length',
    ttft_seconds=ttft,decode_seconds=sum(step_times),decode_steps=len(step_times),
    decode_tokens_per_second=len(step_times)/sum(step_times) if step_times else None,
    wall_seconds=elapsed,peak_memory_bytes=mx.get_peak_memory(),active_memory_bytes=mx.get_active_memory(),step_seconds=step_times)
   f.write(json.dumps(row)+'\n');f.flush()
   print(label,repetition,round(row['decode_tokens_per_second'],2),'tok/s',len(ids),'prompt',len(generated),'output',row['finish_reason'],flush=True)
   del logits,cache;gc.collect();mx.clear_cache()
