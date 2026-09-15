"""Fresh matched-async prepared/phase-queue/fine-DAG decode with shared weights."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent));sys.path.insert(0,str(HERE.parent/'quantized'));sys.path.insert(0,str(HERE.parent/'additional'))
import mlx.core as mx
from transformers import AutoTokenizer
from reference import load,ROOT
from layers import PreparedBranch
from shared_pack import pack_shared
from whole_pass.pack import pack_caches
from whole_pass.kernel import run_whole_pass,SIZE
from cut_whole import CutState

p=argparse.ArgumentParser();p.add_argument('--output',required=True)
p.add_argument('--tokens',type=int,default=32);p.add_argument('--runs',type=int,default=6)
p.add_argument('--workers',type=int,default=32);p.add_argument("--scheduler",choices=["scan","affinity","progress","prefetch_early","prefetch_late","window_early","window_late"],default="scan")
a=p.parse_args();assert a.runs%10==0
model,config=load();original=list(model.layers)
print('Packing shared model weights',flush=True);weights=pack_shared(model)
prepared=[original[0]]+[PreparedBranch(l,64,8,True) for l in original[1:]]
base=ROOT/'work/models/North-Mini-Code-1.0-4bit';tok=AutoTokenizer.from_pretrained(str(base),local_files_only=True);tok.chat_template=(base/'chat_template.jinja').read_text()
ids=tok.apply_chat_template([dict(role='user',content='Return only Python code for merging two sorted lists. Include assertions for empty inputs and duplicate values.')],tokenize=True,return_dict=False,add_generation_prompt=True,reasoning=False,skip_thinking=True)
eos=config['eos_token_id'];eos={eos} if isinstance(eos,int) else set(eos)
assert len(ids)+a.tokens<1024,'Long-context SDPA algorithm needs a separate exactness implementation'
output=Path(a.output);output.parent.mkdir(parents=True,exist_ok=True)
with output.open('w') as f:
    def save(row):f.write(json.dumps(row)+'\n');f.flush()
    sources=[q for folder in (HERE,HERE.parent/'whole_pass',HERE.parent/'quantized',HERE.parent/'full_layer',HERE.parent/'additional',HERE.parent/'persistent') for q in folder.iterdir() if q.suffix in ('.py','.h')]
    save(dict(kind='provenance',args=vars(a),source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),device=mx.device_info(),mlx=mx.__version__,active_bytes=mx.get_active_memory(),prompt_tokens=len(ids),sources={str(q.relative_to(ROOT)):hashlib.sha256(q.read_bytes()).hexdigest() for q in sources},scope='Same async submission, complete logits and final drain. Packed weights shared with baseline modules; prefill/initial cache conversion excluded, decode cache growth included. Short context only pending long SDPA work.'))
    def generate(mode,retain=False):
        model.model.layers=original;cache=model.make_cache();logits=model(mx.array([ids]),cache=cache).logits;mx.eval(logits)
        first=int(mx.argmax(logits[0,-1]).item());current=mx.array(first,mx.int32)
        tokens=[first];pending=None;kept=[];states=[]
        if mode.startswith('cut'):
            position=len(ids);capacity=max(256,cache[0].keys.shape[2]);cut=CutState(weights,cache,position,int(mode[3:]),capacity)
        else:model.model.layers=prepared if mode=='prepared' else original
        start=time.perf_counter()
        for step in range(a.tokens-1):
            if mode.startswith('cut'):
                h=model.model.embed_tokens(current.reshape(1,1));logits,st=cut.advance(h,workers=a.workers,retain=retain)
                if retain:states.append(st)
            else:logits=model(current.reshape(1,1),cache=cache).logits
            current=mx.argmax(logits[0,-1]);mx.async_eval(logits,current)
            if pending is not None:tokens.append(int(pending.item()))
            pending=current
            if retain:kept.append(logits)
        tokens.append(int(pending.item()));mx.synchronize();elapsed=time.perf_counter()-start
        assert len(tokens)==a.tokens and not set(tokens[:-1])&eos
        return tokens,kept,elapsed,states
    expected,refs,_,_=generate('original',True)
    modes=['prepared','cut1','cut2','cut4','cut8']
    for name in modes:
        tokens,got,_,states=generate(name,True);assert tokens==expected
        for step,(r,v) in enumerate(zip(refs,got)):
            unequal=int(mx.sum(r.view(mx.uint8)!=v.view(mx.uint8)).item())
            row=dict(kind='correctness',variant=name,step=step,unequal_bytes=unequal)
            if name.startswith('cut'):
                row['queue_counters_match']=all(int(st[1].item())==int(st[2].item()) for st in states[step])
                assert row['queue_counters_match'],row
            save(row);assert unequal==0,row
        print(name,'gate passed',flush=True)
    del refs,got,states;gc.collect()
    for rep in range(-1,a.runs):
        order=modes[max(0,rep)%len(modes):]+modes[:max(0,rep)%len(modes)]
        if rep>=0 and (rep//len(modes))%2:order.reverse()
        for name in order:
            tokens,_,elapsed,_=generate(name);assert tokens==expected
            save(dict(kind='generation',variant=name,repetition=rep,warmup=rep<0,token_ids=tokens,decode_steps=a.tokens-1,decode_seconds=elapsed,decode_tokens_per_second=(a.tokens-1)/elapsed))
            print(name,rep,round((a.tokens-1)/elapsed,3),flush=True);gc.collect()
model.model.layers=original
