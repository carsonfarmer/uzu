"""Try the corrected Metal branches on a prompt, optionally verifying logits."""
import argparse,json,sys,time
from pathlib import Path
import mlx.core as mx
from transformers import AutoTokenizer
sys.path.insert(0,str(Path(__file__).resolve().parent/'quantized'))
from reference import ROOT,load
from patch import variants
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--prompt',required=True)
p.add_argument('--mode',choices=['original','native_compiled','exact','fast'],default='exact')
p.add_argument('--tokens',type=int,default=512)
p.add_argument('--verify',action='store_true',help='Compare every raw logit byte with a separate original-model cache; timing includes verification')
a=p.parse_args();assert a.tokens>0
model,config=load();paths=variants(model,workers=160,rows=16,modes=() if a.mode=='original' else (a.mode,))
base=ROOT/'work/models/North-Mini-Code-1.0-4bit'
tok=AutoTokenizer.from_pretrained(str(base),local_files_only=True)
tok.chat_template=(base/'chat_template.jinja').read_text()
ids=tok.apply_chat_template([dict(role='user',content=a.prompt)],tokenize=True,
 return_dict=False,add_generation_prompt=True,reasoning=False,skip_thinking=True)
eos=config['eos_token_id'];eos={eos} if isinstance(eos,int) else set(eos)
modes=[a.mode]+(['original'] if a.verify and a.mode!='original' else [])
caches={mode:model.make_cache() for mode in modes};logits={};checks=0
def forward(tokens):
 global checks
 for mode in modes:
  model.model.layers=paths[mode]
  logits[mode]=model(mx.array([tokens]),cache=caches[mode]).logits;mx.eval(logits[mode])
 if len(modes)>1:
  assert bool(mx.all(logits[a.mode].view(mx.uint8)==logits['original'].view(mx.uint8)).item()),'Logit bytes differ'
  checks+=1
 return int(mx.argmax(logits[a.mode][0,-1]).item())
start=time.perf_counter()
for offset in range(0,len(ids),256):token=forward(ids[offset:offset+256])
prefill=time.perf_counter()-start;generated=[token];decode_start=time.perf_counter()
while len(generated)<a.tokens and token not in eos:
 token=forward([token]);generated.append(token)
elapsed=time.perf_counter()-decode_start
print(tok.decode(generated,skip_special_tokens=True))
print(json.dumps(dict(mode=a.mode,prompt_tokens=len(ids),generated_tokens=len(generated),
 finish_reason='stop' if token in eos else 'length',logit_byte_checks=checks,
 prefill_seconds=prefill,decode_seconds=elapsed,
 note='Verification and first-use compilation are included; use benchmark_decode.py for throughput comparisons.')),file=sys.stderr)
