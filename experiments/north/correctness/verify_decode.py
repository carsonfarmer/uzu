"""Fail immediately unless the custom full decoder matches original logits.

Both models advance through the same tokens, with independent fresh caches.
Exact logits plus equal stopping decisions imply identical free generation.
This is a correctness run; its alternating/check-heavy wall time is not a
throughput benchmark.
"""
import argparse,gc,hashlib,json,sys,time
from pathlib import Path
import mlx.core as mx
from transformers import AutoTokenizer
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'quantized'))
from reference import ROOT,load
from patch import variants
p=argparse.ArgumentParser()
p.add_argument('--tokens',type=int,default=512)
p.add_argument('--modes',nargs='+',choices=['exact','scheduled','prefetch','staged','fast'],default=['exact'])
p.add_argument('--workers',type=int,default=160)
p.add_argument('--rows',type=int,default=16)
p.add_argument('--output',default='experiments/north/correctness/results/full-exact-v1.jsonl')
a=p.parse_args();model,config=load();paths=variants(model,modes=tuple(a.modes),workers=a.workers,rows=a.rows)
base=ROOT/'work/models/North-Mini-Code-1.0-4bit'
tok=AutoTokenizer.from_pretrained(str(base),local_files_only=True)
tok.chat_template=(base/'chat_template.jinja').read_text()
eos=config['eos_token_id'];eos={eos} if isinstance(eos,int) else set(eos)
prompts={
 'short':'Return only Python code for merging two sorted lists. Include assertions for empty inputs and duplicate values.',
 'long':'\n'.join(f'def helper_{i}(x): return x + {i}' for i in range(128))+'\nReturn only Python code for merging two sorted lists. Include assertions for empty inputs and duplicate values.',
 'rust':'Return only Rust code for finding the first duplicate in a u32 slice, with tests for empty input and duplicates.'}
out=ROOT/a.output;out.parent.mkdir(parents=True,exist_ok=True)
with out.open('w') as f:
 def save(r):f.write(json.dumps(r)+'\n');f.flush()
 save(dict(kind='provenance',mlx=mx.__version__,device=mx.device_info(),token_cap=a.tokens,args=vars(a),
     scheduler_sources={n:hashlib.sha256((ROOT/'experiments/north/persistent'/n).read_bytes()).hexdigest() for n in ['branch.py','down.h','prefetch.py','fast.py']},
     sources={n:hashlib.sha256((ROOT/'experiments/north/quantized'/n).read_bytes()).hexdigest() for n in ['kernel.h','kernels.py','exact.py','patch.py']}))
 try:
  for label,prompt in prompts.items():
   ids=tok.apply_chat_template([dict(role='user',content=prompt)],tokenize=True,return_dict=False,
        add_generation_prompt=True,reasoning=False,skip_thinking=True)
   caches={};logits={}
   for mode in paths:
    model.model.layers=paths[mode];caches[mode]=model.make_cache()
    for offset in range(0,len(ids),256):
     logits[mode]=model(mx.array([ids[offset:offset+256]]),cache=caches[mode]).logits;mx.eval(logits[mode])
   assert all(bool(mx.all(logits['original'].view(mx.uint8)==logits[mode].view(mx.uint8)).item()) for mode in a.modes),'prefill'
   token=int(mx.argmax(logits['original'][0,-1]).item());generated=[token]
   start=time.perf_counter()
   while len(generated)<a.tokens and token not in eos:
    for mode in paths:
     model.model.layers=paths[mode]
     logits[mode]=model(mx.array([[token]]),cache=caches[mode]).logits;mx.eval(logits[mode])
    for mode in a.modes:
     bits=logits[mode].view(mx.uint8)!=logits['original'].view(mx.uint8)
     if bool(mx.any(bits).item()):
      d=logits[mode].astype(mx.float32)-logits['original'].astype(mx.float32)
      row=dict(kind='failure',prompt=label,step=len(generated),mode=mode,unequal_bytes=int(mx.sum(bits).item()),
         max_abs=float(mx.max(mx.abs(d)).item()))
      save(row);raise AssertionError(row)
    token=int(mx.argmax(logits['original'][0,-1]).item());generated.append(token)
    save(dict(kind='step',prompt=label,step=len(generated)-1,exact_logits=True,bitwise=True,modes=a.modes,token=token))
    if (len(generated)-1)%64==0:print(label,len(generated)-1,'exact decode steps',flush=True)
   save(dict(kind='completion',prompt=label,prompt_tokens=len(ids),generated_tokens=len(generated),
       exact_decode_steps=len(generated)-1,token_ids=generated,text=tok.decode(generated,skip_special_tokens=True),
       finish_reason='stop' if token in eos else 'length',check_wall_seconds=time.perf_counter()-start))
   print(label,'PASS',len(generated)-1,'exact decode steps',flush=True)
   del caches,logits;gc.collect();mx.clear_cache();logits={}
 finally:model.model.layers=paths['original']
