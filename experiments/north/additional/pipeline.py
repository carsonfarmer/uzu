"""Isolate host submission overlap while retaining full logits and exact kernels.

Uses one-token lookahead for CPU/GPU submission, not speculative model tokens.
Every next token still depends on the previous step's complete logits/argmax.
"""
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
sys.path.insert(0,str(HERE.parent));sys.path.insert(0,str(HERE.parent/'quantized'))
from reference import ROOT,load
from patch import variants

p=argparse.ArgumentParser()
p.add_argument('--output',required=True)
p.add_argument('--prompts',nargs='+',default=['short','rust','long'])
p.add_argument('--tokens',type=int,default=128)
p.add_argument('--runs',type=int,default=6)
p.add_argument('--check',action=argparse.BooleanOptionalAction,default=True)
a=p.parse_args()
model,config=load();paths=variants(model,modes=('exact',),workers=160,rows=16)
base=ROOT/'work/models/North-Mini-Code-1.0-4bit'
tok=AutoTokenizer.from_pretrained(str(base),local_files_only=True)
tok.chat_template=(base/'chat_template.jinja').read_text()
prompts={
 'short':'Return only Python code for merging two sorted lists. Include assertions for empty inputs and duplicate values.',
 'rust':'Return only Rust code for finding the first duplicate in a u32 slice, with tests for empty input and duplicates.',
 'long':'\n'.join(f'def helper_{i}(x): return x + {i}' for i in range(128))+'\nReturn only Python code for merging two sorted lists. Include assertions for empty inputs and duplicate values.'}
eos=config['eos_token_id'];eos={eos} if isinstance(eos,int) else set(eos)
output=Path(a.output);output.parent.mkdir(parents=True,exist_ok=True)
with output.open('w') as f:
    def save(row):f.write(json.dumps(row)+'\n');f.flush()
    sources=[Path(__file__).resolve(),HERE/'gpu_run.py',HERE.parent/'reference.py']+[HERE.parent/'quantized'/n for n in ('patch.py','exact.py','kernel.h','kernels.py')]
    save(dict(kind='provenance',args=vars(a),mlx=mx.__version__,device=mx.device_info(),
        batching_environment={k:os.environ.get(k) for k in ('MLX_MAX_MB_PER_BUFFER','MLX_MAX_OPS_PER_BUFFER')},
        sources={str(s.relative_to(ROOT)):hashlib.sha256(s.read_bytes()).hexdigest() for s in sources},
        scope='Fixed-length decode. All full logits computed. No speculative model steps. Early EOS is rejected for this experiment. Async path overlaps host construction of the next dependent graph with GPU execution; synchronous path is the established exact control.'))
    def generate(ids,path,asynchronous,retain=False):
        model.model.layers=paths[path];cache=model.make_cache()
        for i in range(0,len(ids),256):
            logits=model(mx.array([ids[i:i+256]]),cache=cache).logits;mx.eval(logits)
        first=int(mx.argmax(logits[0,-1]).item());current=mx.array(first,mx.int32)
        tokens=[first];pending=None;kept=[];intervals=[]
        start=time.perf_counter()
        for step in range(a.tokens-1):
            t=time.perf_counter()
            logits=model(current.reshape(1,1),cache=cache).logits
            current=mx.argmax(logits[0,-1])
            if asynchronous:
                mx.async_eval(logits,current)
                if pending is not None:tokens.append(int(pending.item()))
                pending=current
            else:
                mx.eval(logits);tokens.append(int(current.item()))
                # Match the established host-token submission path.
                current=mx.array(tokens[-1],mx.int32)
            if retain:kept.append(logits)
            intervals.append(time.perf_counter()-t)
        if asynchronous:tokens.append(int(pending.item()))
        mx.synchronize()
        elapsed=time.perf_counter()-start
        assert len(tokens)==a.tokens
        assert not (set(tokens[:-1])&eos),'Early EOS: fixed-length benchmark inapplicable'
        return tokens,kept,elapsed,intervals
    modes={'exact_sync':('exact',False),'exact_async':('exact',True),'original_async':('original',True)}
    for label in a.prompts:
        ids=tok.apply_chat_template([dict(role='user',content=prompts[label])],tokenize=True,return_dict=False,add_generation_prompt=True,reasoning=False,skip_thinking=True)
        expected=None
        if a.check:
            expected,ref,_,_=generate(ids,'original',False,True)
            for name,(path,async_mode) in modes.items():
                tokens,got,_,_=generate(ids,path,async_mode,True)
                assert tokens==expected,name
                for step,(r,v) in enumerate(zip(ref,got)):
                    unequal=int(mx.sum(r.view(mx.uint8)!=v.view(mx.uint8)).item())
                    save(dict(kind='correctness',prompt=label,variant=name,step=step,unequal_bytes=unequal))
                    assert unequal==0,(name,step,unequal)
                del got;gc.collect()
            del ref;gc.collect()
            print(label,'full-logit pipeline gate passed',a.tokens-1,flush=True)
        for rep in range(-1,a.runs):
            names=list(modes);shift=(rep+1)%len(names);names=names[shift:]+names[:shift]
            if rep%2:names.reverse()
            for name in names:
                tokens,_,elapsed,intervals=generate(ids,*modes[name])
                if expected is None:expected=tokens
                assert tokens==expected,name
                save(dict(kind='generation',prompt=label,prompt_tokens=len(ids),variant=name,repetition=rep,warmup=rep<0,token_ids=tokens,decode_steps=a.tokens-1,decode_seconds=elapsed,decode_tokens_per_second=(a.tokens-1)/elapsed,host_iteration_seconds=intervals))
                print(label,name,rep,round((a.tokens-1)/elapsed,3),flush=True)
                gc.collect()
model.model.layers=paths['original']
