"""Warm a full decoder, then wait for an external profiler to attach."""
import argparse,json,os,sys,time
from pathlib import Path
import mlx.core as mx
from transformers import AutoTokenizer
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'quantized'))
from reference import ROOT,load
from patch import variants
p=argparse.ArgumentParser();p.add_argument('--mode',choices=['original','native_compiled','exact','fast','scheduled'],required=True)
p.add_argument('--signal',required=True);p.add_argument('--steps',type=int,default=128)
a=p.parse_args();model,config=load();paths=variants(model,workers=160,rows=16,modes=() if a.mode=='original' else (a.mode,))
model.model.layers=paths[a.mode]
base=ROOT/'work/models/North-Mini-Code-1.0-4bit'
tok=AutoTokenizer.from_pretrained(str(base),local_files_only=True)
tok.chat_template=(base/'chat_template.jinja').read_text()
prompt='Return only Python code for merging two sorted lists. Include assertions for empty inputs and duplicate values.'
ids=tok.apply_chat_template([dict(role='user',content=prompt)],tokenize=True,return_dict=False,add_generation_prompt=True,reasoning=False,skip_thinking=True)
def prefill():
 c=model.make_cache();v=model(mx.array([ids]),cache=c).logits
 return c,int(mx.argmax(v[0,-1]).item())
cache,token=prefill()
for _ in range(32):token=int(mx.argmax(model(mx.array([[token]]),cache=cache).logits[0,-1]).item())
cache,token=prefill()
signal=Path(a.signal)
signal.with_suffix('.ready').write_text(str(os.getpid()))
print('READY',os.getpid(),flush=True)
while not signal.exists():time.sleep(.05)
start=time.perf_counter()
for _ in range(a.steps):token=int(mx.argmax(model(mx.array([[token]]),cache=cache).logits[0,-1]).item())
print(json.dumps(dict(mode=a.mode,steps=a.steps,seconds=time.perf_counter()-start)),flush=True)
