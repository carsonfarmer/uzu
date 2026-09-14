"""Raw full-logit gate and counterbalanced end-to-end decode measurement."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import mlx.core as mx
from transformers import AutoTokenizer
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
sys.path.insert(0,str(HERE.parent/'quantized'))
from reference import ROOT,load
from patch import variants
from layers import PreparedBranch
from lean import LeanBranch

p=argparse.ArgumentParser()
p.add_argument('--output',required=True)
p.add_argument('--prompts',nargs='+',default=['short','rust','long'])
p.add_argument('--tokens',type=int,default=128)
p.add_argument('--runs',type=int,default=6)
p.add_argument('--check-steps',type=int,default=127)
p.add_argument('--rows',type=int,default=32)
p.add_argument('--router-rows',type=int,default=4)
p.add_argument('--normalize',action='store_true')
p.add_argument('--candidate',choices=['prep','lean'],default='prep')
a=p.parse_args()
model,config=load()
paths=variants(model,modes=('exact',),workers=160,rows=16)
paths['candidate']=[paths['original'][0]]+[(PreparedBranch(l,a.rows,a.router_rows,a.normalize) if a.candidate=='prep' else LeanBranch(l)) for l in paths['original'][1:]]
modelpath=ROOT/'work/models/North-Mini-Code-1.0-4bit'
tok=AutoTokenizer.from_pretrained(str(modelpath),local_files_only=True)
tok.chat_template=(modelpath/'chat_template.jinja').read_text()
prompts={
 'short':'Return only Python code for merging two sorted lists. Include assertions for empty inputs and duplicate values.',
 'rust':'Return only Rust code for finding the first duplicate in a u32 slice, with tests for empty input and duplicates.',
 'long':'\n'.join(f'def helper_{i}(x): return x + {i}' for i in range(128))+'\nReturn only Python code for merging two sorted lists. Include assertions for empty inputs and duplicate values.'}
eos=config['eos_token_id'];eos={eos} if isinstance(eos,int) else set(eos)
output=Path(a.output);output.parent.mkdir(parents=True,exist_ok=True)
with output.open('w') as f:
    def save(row):f.write(json.dumps(row)+'\n');f.flush()
    source_files=list(HERE.glob('*.py'))+list(HERE.glob('*.h'))+[HERE.parent/'quantized'/n for n in ('patch.py','exact.py','kernel.h','kernels.py')]+[HERE.parent/'reference.py',ROOT/'work/mlx-vlm/mlx_vlm/models/cohere2_moe/language.py']
    save(dict(kind='provenance',args=vars(a),mlx=mx.__version__,device=mx.device_info(),
        batching_environment={key:os.environ.get(key) for key in ('MLX_MAX_OPS_PER_BUFFER','MLX_MAX_MB_PER_BUFFER')},
        sources={str(s.relative_to(ROOT)):hashlib.sha256(s.read_bytes()).hexdigest() for s in source_files},
        scope='One shared model; native prefill; full-logit byte checks precede separate free-running timing. Includes token submission, cache updates, full logits, eval and argmax. Excludes model load, prefill and warmup.'))
    def prefill(ids,name):
        model.model.layers=paths[name];cache=model.make_cache()
        for i in range(0,len(ids),256):
            logits=model(mx.array([ids[i:i+256]]),cache=cache).logits;mx.eval(logits)
        return cache,logits
    for label in a.prompts:
        ids=tok.apply_chat_template([dict(role='user',content=prompts[label])],tokenize=True,return_dict=False,add_generation_prompt=True,reasoning=False,skip_thinking=True)
        if a.check_steps:
            states={name:prefill(ids,name) for name in paths}
            ref=states['original'][1]
            for name,(_,value) in states.items():
                assert bool(mx.all(ref.view(mx.uint8)==value.view(mx.uint8)).item()),name
            token=int(mx.argmax(ref[0,-1]).item())
            for step in range(a.check_steps):
                logits={}
                for name in paths:
                    model.model.layers=paths[name]
                    logits[name]=model(mx.array([[token]]),cache=states[name][0]).logits
                    mx.eval(logits[name])
                for name in ('exact','candidate'):
                    unequal=int(mx.sum(logits[name].view(mx.uint8)!=logits['original'].view(mx.uint8)).item())
                    save(dict(kind='correctness',prompt=label,prompt_tokens=len(ids),step=step,variant=name,unequal_bytes=unequal))
                    assert unequal==0,(label,step,name,unequal)
                token=int(mx.argmax(logits['original'][0,-1]).item())
            del states,logits,ref;gc.collect()
            print(label,'full-logit gate passed',a.check_steps,flush=True)
        expected=None
        for rep in range(-1,a.runs):
            order=['exact','candidate'] if rep%2==0 else ['candidate','exact']
            for name in order:
                cache,logits=prefill(ids,name)
                token=int(mx.argmax(logits[0,-1]).item());tokens=[token];steps=[]
                while len(tokens)<a.tokens and token not in eos:
                    start=time.perf_counter()
                    logits=model(mx.array([[token]]),cache=cache).logits
                    mx.eval(logits)
                    token=int(mx.argmax(logits[0,-1]).item())
                    steps.append(time.perf_counter()-start);tokens.append(token)
                if expected is None:expected=tokens
                assert tokens==expected,(label,rep,name)
                save(dict(kind='generation',prompt=label,prompt_tokens=len(ids),variant=name,repetition=rep,warmup=rep<0,token_ids=tokens,step_seconds=steps,decode_steps=len(steps),decode_tokens_per_second=len(steps)/sum(steps)))
                print(label,name,rep,round(len(steps)/sum(steps),3),flush=True)
                del cache,logits;gc.collect()
model.model.layers=paths['original']
