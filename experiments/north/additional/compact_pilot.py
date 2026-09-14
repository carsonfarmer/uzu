"""Corrected historical host-token control and balanced five-way comparison.

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
from layers import PreparedBranch
from compact import CompactPrepared

p=argparse.ArgumentParser()
p.add_argument('--output',required=True)
p.add_argument('--prompts',nargs='+',default=['short','rust','long'])
p.add_argument('--tokens',type=int,default=128)
p.add_argument('--runs',type=int,default=6)
p.add_argument('--check',action=argparse.BooleanOptionalAction,default=True)
a=p.parse_args()
assert a.runs%6==0, 'Three-way order requires complete six-round balanced cycles'
model,config=load();paths=variants(model,modes=('exact',),workers=160,rows=16)
paths['prepared']=[paths['original'][0]]+[PreparedBranch(l,64,8,True) for l in paths['original'][1:]]
paths['compact']=[paths['original'][0]]+[CompactPrepared(l) for l in paths['original'][1:]]
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
    sources=[Path(__file__).resolve(),HERE/'gpu_run.py',HERE/'layers.py',HERE/'compact.py',HERE/'prep.py',HERE/'prep.h',HERE.parent/'reference.py',HERE.parent/'full_layer/benchmark_decode.py']+[HERE.parent/'quantized'/n for n in ('patch.py','exact.py','kernel.h','kernels.py')]
    save(dict(kind='provenance',args=vars(a),mlx=mx.__version__,device=mx.device_info(),
        batching_environment={k:os.environ.get(k) for k in ('MLX_MAX_MB_PER_BUFFER','MLX_MAX_OPS_PER_BUFFER')},
        sources={str(s.relative_to(ROOT)):hashlib.sha256(s.read_bytes()).hexdigest() for s in sources},
        scope='Fixed-length decode. All full logits computed. No speculative model steps. Early EOS is rejected for this experiment. Async path overlaps host construction of the next dependent graph with GPU execution; host control matches full_layer/benchmark_decode.py direct argmax.item loop with mx.array([[token]]), without a separate logits eval. All modes include final drain.'))
    def generate(ids,path,submission,retain=False):
        model.model.layers=paths[path];cache=model.make_cache()
        for i in range(0,len(ids),256):
            logits=model(mx.array([ids[i:i+256]]),cache=cache).logits;mx.eval(logits)
        first=int(mx.argmax(logits[0,-1]).item());current=mx.array(first,mx.int32);token=first
        tokens=[first];pending=None;kept=[];intervals=[]
        start=time.perf_counter()
        for step in range(a.tokens-1):
            t=time.perf_counter()
            if submission=='async':
                logits=model(current.reshape(1,1),cache=cache).logits
                current=mx.argmax(logits[0,-1])
                mx.async_eval(logits,current)
                if pending is not None:tokens.append(int(pending.item()))
                pending=current
            else:
                # Historical benchmark_decode.py host-token loop, unchanged.
                logits=model(mx.array([[token]]),cache=cache).logits
                if submission=='host':
                    token=int(mx.argmax(logits[0,-1]).item())
                elif submission=='joint':
                    next_token=mx.argmax(logits[0,-1])
                    mx.eval(logits,next_token)
                    token=int(next_token.item())
                else:raise ValueError(submission)
                tokens.append(token)
            if retain:kept.append(logits)
            intervals.append(time.perf_counter()-t)
        if submission=='async':tokens.append(int(pending.item()))
        mx.synchronize()
        elapsed=time.perf_counter()-start
        assert len(tokens)==a.tokens
        assert not (set(tokens[:-1])&eos),'Early EOS: fixed-length benchmark inapplicable'
        return tokens,kept,elapsed,intervals
    modes={'exact_host':('exact','host'),'prepared_async':('prepared','async'),'compact_async':('compact','async')}
    for label in a.prompts:
        ids=tok.apply_chat_template([dict(role='user',content=prompts[label])],tokenize=True,return_dict=False,add_generation_prompt=True,reasoning=False,skip_thinking=True)
        expected=None
        if a.check:
            expected,ref,_,_=generate(ids,'original','host',retain=True)
            for name,settings in modes.items():
                tokens,got,_,_=generate(ids,*settings,retain=True)
                assert tokens==expected,name
                for step,(r,v) in enumerate(zip(ref,got)):
                    unequal=int(mx.sum(r.view(mx.uint8)!=v.view(mx.uint8)).item())
                    save(dict(kind='correctness',prompt=label,variant=name,step=step,unequal_bytes=unequal))
                    assert unequal==0,(name,step,unequal)
                del got;gc.collect()
            del ref;gc.collect()
            print(label,'full-logit pipeline gate passed',a.tokens-1,flush=True)
        for rep in range(-1,a.runs):
            names=list(modes)
            if rep>=0:
                shift=rep%len(names);names=names[shift:]+names[:shift]
                if (rep//len(names))%2:names.reverse()
            for name in names:
                tokens,_,elapsed,intervals=generate(ids,*modes[name])
                if expected is None:expected=tokens
                assert tokens==expected,name
                save(dict(kind='generation',prompt=label,prompt_tokens=len(ids),variant=name,repetition=rep,warmup=rep<0,token_ids=tokens,decode_steps=a.tokens-1,decode_seconds=elapsed,decode_tokens_per_second=(a.tokens-1)/elapsed,host_iteration_seconds=intervals))
                print(label,name,rep,round((a.tokens-1)/elapsed,3),flush=True)
                gc.collect()
model.model.layers=paths['original']
