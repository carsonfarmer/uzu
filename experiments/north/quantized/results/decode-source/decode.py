"""Full-model teacher-forced correctness and paired greedy decode experiment."""
import argparse,gc,hashlib,json,sys,time
from pathlib import Path
import mlx.core as mx
from transformers import AutoTokenizer
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from reference import ROOT,load
from patch import variants

p=argparse.ArgumentParser()
p.add_argument('--tokens',type=int,default=256)
p.add_argument('--runs',type=int,default=3)
p.add_argument('--teacher-steps',type=int,default=64)
p.add_argument('--prompts',nargs='+',default=['short','long','rust'])
p.add_argument('--output',default='experiments/north/quantized/results/decode.jsonl')
a=p.parse_args()
model,config=load();paths=variants(model,chunk=64,workers=128)
path=ROOT/'work/models/North-Mini-Code-1.0-4bit'
tok=AutoTokenizer.from_pretrained(str(path),local_files_only=True)
tok.chat_template=(path/'chat_template.jinja').read_text()
eos=config['eos_token_id'];eos={eos} if isinstance(eos,int) else set(eos)
prompts={
 'short':'Return only Python code for merging two sorted lists. Include assertions for empty inputs and duplicate values.',
 'long':'\n'.join(f'def helper_{i}(x): return x + {i}' for i in range(128))+'\nReturn only Python code for merging two sorted lists. Include assertions for empty inputs and duplicate values.',
 'rust':'Return only Rust code for finding the first duplicate in a u32 slice, with tests for empty input and duplicates.'}
prior={r['prompt']:r['token_ids'] for r in map(json.loads,(ROOT/'experiments/north/results/baseline.jsonl').read_text().splitlines()) if not r['warmup']}
output=ROOT/a.output;output.parent.mkdir(parents=True,exist_ok=True)
f=output.open('w')
def save(row):f.write(json.dumps(row)+'\n');f.flush()
def prefill(ids,mode):
    model.model.layers=paths[mode];cache=model.make_cache()
    for pos in range(0,len(ids),256):
        logits=model(mx.array([ids[pos:pos+256]]),cache=cache).logits
        mx.eval(logits)
    return cache,logits

def measure(ids,mode):
    mx.reset_peak_memory();start=time.perf_counter()
    cache,logits=prefill(ids,mode)
    token=int(mx.argmax(logits[0,-1]).item());ttft=time.perf_counter()-start
    generated=[token];steps=[]
    while len(generated)<a.tokens and token not in eos:
        t=time.perf_counter();logits=model(mx.array([[token]]),cache=cache).logits
        token=int(mx.argmax(logits[0,-1]).item());steps.append(time.perf_counter()-t)
        generated.append(token)
    row=dict(ttft_seconds=ttft,decode_steps=len(steps),decode_seconds=sum(steps),
        decode_tokens_per_second=len(steps)/sum(steps),step_seconds=steps,
        wall_seconds=time.perf_counter()-start,token_ids=generated,
        generated_tokens=len(generated),finish_reason='stop' if token in eos else 'length',
        peak_memory_bytes=mx.get_peak_memory(),text=tok.decode(generated,skip_special_tokens=True))
    del cache,logits;gc.collect()
    return row
save(dict(kind='provenance',device=mx.device_info(),mlx=mx.__version__,args=vars(a),
    model_revision='dfbe084dfa26e241345af99ca32848f38fd865f9',
    source_revision='cdc745ad8a32d162f6d8e9d08be256910d663ac2',
    source_sha256={name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in ['decode.py','patch.py','kernels.py','kernel.h']},
    specialization=dict(chunk=64,workers=128,interleave=True,patched_layers=list(range(1,49))),
    note='Only single-token branches replaced; original prefill and layer 0. All variants share loaded weight buffers. One warmup and rotated variant order. Teacher-forced paths use identical baseline tokens; greedy paths free-run.'))
try:
    for label in a.prompts:
        ids=tok.apply_chat_template([dict(role='user',content=prompts[label])],tokenize=True,
            return_dict=False,add_generation_prompt=True,reasoning=False,skip_thinking=True)
        if a.teacher_steps:
            caches={};initial={}
            for mode in paths:
                caches[mode],initial[mode]=prefill(ids,mode)
            assert all(bool(mx.all(v==initial['original']).item()) for v in initial.values())
            del initial
            for step,token in enumerate(prior[label][:a.teacher_steps]):
                logits={}
                for mode in paths:
                    model.model.layers=paths[mode]
                    logits[mode]=model(mx.array([[token]]),cache=caches[mode]).logits[0,-1].astype(mx.float32)
                    mx.eval(logits[mode])
                ref=logits['original'];ref_logp=mx.log(mx.softmax(ref))
                reference_top=int(mx.argmax(ref).item())
                for mode in ['native_compiled','custom']:
                    v=logits[mode];d=v-ref
                    kl=mx.sum(mx.exp(ref_logp)*(ref_logp-mx.log(mx.softmax(v))))
                    row=dict(kind='teacher',prompt=label,step=step,variant=mode,
                        reference_top=reference_top,candidate_top=int(mx.argmax(v).item()),
                        max_abs=float(mx.max(mx.abs(d)).item()),
                        relative_l2=float((mx.linalg.norm(d)/mx.linalg.norm(ref)).item()),
                        kl=float(kl.item()),finite=bool(mx.all(mx.isfinite(v)).item()))
                    assert row['finite'],row
                    save(row)
            del caches,logits;gc.collect();mx.clear_cache()
            print(label,'teacher checks saved',flush=True)
        for rep in range(-1,a.runs):
            order=list(paths);shift=(rep+1)%len(order);order=order[shift:]+order[:shift]
            for mode in order:
                row=measure(ids,mode)
                row.update(kind='generation',prompt=label,prompt_tokens=len(ids),
                    variant=mode,repetition=rep,warmup=rep<0)
                save(row)
                print(label,mode,rep,round(row['decode_tokens_per_second'],2),'tok/s',row['generated_tokens'],flush=True)
finally:
    model.model.layers=paths['original'];f.close()
