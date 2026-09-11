"""Generate the corrected, auditable result packet from completed raw runs."""
import hashlib,json,statistics
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

BASE=Path(__file__).resolve().parent;NORTH=BASE.parent;OUT=NORTH/'share';OUT.mkdir(exist_ok=True)
median=statistics.median
def jsonl(path):return [json.loads(line) for line in path.read_text().splitlines()]
raw=jsonl(BASE/'results/benchmark-final-v4.jsonl')
gate=jsonl(BASE/'results/full-fast-v4.jsonl')
earlier_gate=jsonl(BASE/'results/full-bitwise-v3.jsonl')
assert raw[0]['source_sha256']==gate[0]['sources']
assert raw[0]['scheduler_sources']==gate[0]['scheduler_sources']
for name in ['kernel.h','kernels.py','exact.py']:
 assert raw[0]['source_sha256'][name]==earlier_gate[0]['sources'][name]
assert all(r['bitwise'] and r['exact_logits'] for r in gate if r['kind']=='step')
assert sum(r['kind']=='step' for r in gate)==1424 and not any(r['kind']=='failure' for r in gate)
rows=[r for r in raw if r['kind']=='generation' and not r['warmup']]
modes=['original','native_compiled','exact','fast'];prompts=['short','long','rust']
assert len(rows)==len(modes)*len(prompts)*3
summary=[]
for prompt in prompts:
 reference={r['repetition']:r for r in rows if r['prompt']==prompt and r['variant']=='original'}
 control={r['repetition']:r for r in rows if r['prompt']==prompt and r['variant']=='native_compiled'}
 for mode in modes:
  group=[r for r in rows if r['prompt']==prompt and r['variant']==mode]
  assert len(group)==3 and all(r['token_ids']==reference[r['repetition']]['token_ids'] for r in group)
  rates=[r['decode_tokens_per_second'] for r in group]
  pairs=[r['decode_tokens_per_second']/reference[r['repetition']]['decode_tokens_per_second'] for r in group]
  compiled_pairs=[r['decode_tokens_per_second']/control[r['repetition']]['decode_tokens_per_second'] for r in group]
  summary.append(dict(prompt=prompt,variant=mode,median_tps=median(rates),min_tps=min(rates),max_tps=max(rates),
   paired_ratio_vs_original=median(pairs),paired_ratios_vs_original=pairs,
   paired_ratio_vs_compiled=median(compiled_pairs),paired_ratios_vs_compiled=compiled_pairs,
   median_ttft_seconds=median(r['ttft_seconds'] for r in group),
   median_peak_memory_bytes=median(r['peak_memory_bytes'] for r in group)))
result=dict(decode=summary,measured_requests=len(rows),all_tokens_match=True,
 logit_gate=dict(steps_per_variant=1424,total_variant_steps=7120,variants=['exact','scheduled','prefetch','staged','fast'],
  prefill_matches=True,raw_byte_equality=True,generated_tokens=[512,512,403]),
 benchmark_provenance=raw[0],raw_files={str(p.relative_to(NORTH)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [BASE/'results/benchmark-final-v4.jsonl',BASE/'results/benchmark-all-v3.jsonl',BASE/'results/full-fast-v4.jsonl',BASE/'results/full-bitwise-v3.jsonl']})
(OUT/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
def row(prompt,mode):return next(r for r in summary if r['prompt']==prompt and r['variant']==mode)
def gains(mode,baseline='original'):
 return [100*(row(p,mode)['paired_ratio_vs_'+baseline]-1) for p in prompts]
names={'original':'Original MLX','native_compiled':'Compiled MLX','exact':'Fused Metal','fast':'Persistent Metal'}

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,'axes.spines.left':False})
fig=plt.figure(figsize=(12,7.2));bg='#f7f7f2';fig.patch.set_facecolor(bg)
fig.text(.07,.935,'North Mini Code on an M4 Pro',fontsize=25,weight='bold',color='#152d2b')
fig.text(.07,.885,'48 GB Mac · affine 4-bit weights · batch 1 · MLX 0.32.2',fontsize=13,color='#53635f')
ax=fig.add_axes([.075,.30,.625,.51]);ax.set_facecolor(bg)
x=np.arange(3);width=.19;colors=['#aab3b1','#647e8a','#197b60','#bd8134']
for index,mode in enumerate(modes):
 values=np.array([row(p,mode)['median_tps'] for p in prompts]);low=np.array([row(p,mode)['min_tps'] for p in prompts]);high=np.array([row(p,mode)['max_tps'] for p in prompts])
 bars=ax.bar(x+(index-1.5)*width,values,width,color=colors[index],label=names[mode],
  yerr=[values-low,high-values],capsize=2,error_kw={'elinewidth':.8,'alpha':.65})
 for bar,value in zip(bars,values):ax.text(bar.get_x()+width/2,value+1.6,f'{value:.1f}',ha='center',fontsize=9)
ax.set_xticks(x,['Python','Python · long input','Rust']);ax.set_ylabel('Decode tokens / second')
ax.set_ylim(0,max(r['max_tps'] for r in summary)*1.19);ax.grid(axis='y',alpha=.16);ax.set_axisbelow(True)
ax.tick_params(axis='both',length=0);ax.legend(loc='upper left',bbox_to_anchor=(-.01,-.14),ncol=2,frameon=False,fontsize=10)
fig.text(.755,.70,'1,424',fontsize=31,weight='bold',color='#197b60')
fig.text(.755,.64,'decode steps with\nidentical logit bytes',fontsize=12,color='#203d36',linespacing=1.5)
fig.text(.755,.49,'48',fontsize=31,weight='bold',color='#203d36')
fig.text(.755,.45,'MoE layers integrated',fontsize=11,color='#203d36')
fig.text(.755,.34,'Same tokens.\nSame stopping behavior.',fontsize=11,weight='bold',color='#197b60',linespacing=1.6)
fig.text(.07,.13,'Medians + min–max of 3 runs per prompt; 256-token cap. Independent full-logit checks use a 512-token cap.',fontsize=9,color='#53635f')
fig.text(.07,.087,'Branch integrations into the full decoder. QKV, attention, routing and layer transitions still use MLX.',fontsize=9,color='#53635f')
fig.text(.07,.044,'Inspired by cohere.com/blog/megakernels · source, raw runs and controlled ablations included · September 2026',fontsize=9,color='#53635f')
fig.savefig(OUT/'north-m4-pro-results.png',dpi=180,facecolor=bg)
fig.savefig(OUT/'north-m4-pro-results.svg',facecolor=bg)

table=['| Prompt | Original MLX | Compiled MLX | Fused Metal | Persistent Metal |',
 '|---|---:|---:|---:|---:|']
for p in prompts:table.append('| '+p+' | '+' | '.join(f"{row(p,m)['median_tps']:.2f}" for m in modes)+' |')
gain_table=['| Prompt | Fused / original | Persistent / original | Fused / compiled | Persistent / compiled |','|---|---:|---:|---:|---:|']
for p in prompts:gain_table.append('| '+p+' | '+' | '.join(f"{100*(row(p,m)['paired_ratio_vs_'+b]-1):+.1f}%" for b in ['original','compiled'] for m in ['exact','fast'])+' |')
old=json.loads((BASE/'results/summary-all-v3.json').read_text())
ablation=['| Prompt | Original | Compiled | Fused | First scheduler | Prefetch | Stage after ready |','|---|---:|---:|---:|---:|---:|---:|']
for p in prompts:ablation.append('| '+p+' | '+' | '.join(f"{next(r['median_tps'] for r in old['decode'] if r['prompt']==p and r['variant']==m):.2f}" for m in ['original','native_compiled','exact','scheduled','prefetch','staged'])+' |')
mechanisms=[];early_starts=[]
for layer in [1,7]:
 for mode,candidate,control in [('fast','w64_fine','w64_all_experts'),('staged','w64_prefetch','w64_after_ready')]:
  data=json.loads((NORTH/'persistent'/f'ablation-{mode}-layer{layer}-v5.json').read_text())
  baseline={s['repetition']:s['wall_us'] for s in data['samples'] if s['variant']==control}
  selected=[s for s in data['samples'] if s['variant']==candidate]
  assert len(baseline)==len(selected)==29
  assert all(c['unequal_bytes']==[0]*4 and c['errors']==0 and c['task_visits_min']==c['task_visits_max']==1 for c in data['checks'])
  change=100*(median(s['wall_us']/baseline[s['repetition']] for s in selected)-1)
  mechanisms.append(dict(layer=layer,mode=mode,paired_latency_change_percent=change,pairs=29))
  if mode=='fast':early_starts.extend(c['first_down_completed_up_tiles'] for c in data['checks'] if c['variant']=='w64_fine')
mechanism_table=['| Layer fixture | Per-expert readiness vs all experts | Prefetch vs load after ready |','|---|---:|---:|']
for layer in [1,7]:mechanism_table.append('| '+str(layer)+' | '+' | '.join(f"{next(r['paired_latency_change_percent'] for r in mechanisms if r['layer']==layer and r['mode']==mode):+.1f}%" for mode in ['fast','staged'])+' |')
result['branch_ablations']=mechanisms
confirmation=jsonl(BASE/'results/benchmark-fast-counterbalanced-v5.jsonl')
assert confirmation[0]['source_sha256']==gate[0]['sources'] and confirmation[0]['scheduler_sources']==gate[0]['scheduler_sources']
confirm_rows=[r for r in confirmation if r['kind']=='generation' and not r['warmup']]
assert len(confirm_rows)==12
confirm_ratios=[]
for prompt in prompts:
 for rep in [0,1]:
  pair={r['variant']:r for r in confirm_rows if r['prompt']==prompt and r['repetition']==rep}
  assert pair['original']['token_ids']==pair['fast']['token_ids']
  ratio=pair['fast']['decode_tokens_per_second']/pair['original']['decode_tokens_per_second']
  confirm_ratios.append(dict(prompt=prompt,repetition=rep,ratio=ratio))
assert all(r['ratio']>1 for r in confirm_ratios)
result['counterbalanced_confirmation']=confirm_ratios
(OUT/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
report=f'''# Correct, faster North Mini Code decoding on M4 Pro

The original prototype was incomplete and contained accumulation-order errors.
Those errors are fixed. Both the fused branch and the optimized persistent
scheduler now have repeatable full-model comparisons with identical generated
tokens. The persistent path also passes 1,424 complete-logit byte comparisons,
plus prefill; the fused path passes the same gate at the preceding checkpoint
with unchanged kernel arithmetic.

## Full-model results

Decode tokens per second, median of three measured runs per prompt:

{chr(10).join(table)}

Median paired throughput changes (pair by prompt and repetition):

{chr(10).join(gain_table)}

The ratio of medians and median paired ratio are different statistics; the
percentages above use paired ratios. All 36 measured requests use identical
token sequences across variants. Output cap is 256; the first token belongs
to prefill, and decode throughput measures the subsequent 255 forward passes.
The inputs contain 136, 1,672 and 140 tokens. Model load, tokenization and
warmup compilation are excluded. These are decode timings within complete
model generation, not end-to-end serving speedups. TTFT and full wall times
are retained in the raw JSONL. Desktop clocks are unlocked; ranges and every
paired observation are available in `summary.json` and the raw runs.

An additional confirmation uses only original MLX and the persistent path,
with two measured repetitions per prompt: one in each order. All six pairs
again improve, by {100*(min(r['ratio'] for r in confirm_ratios)-1):.1f}–{100*(max(r['ratio'] for r in confirm_ratios)-1):.1f}%, with matching tokens. This separate confirmation is
not pooled into the headline table. Its sources match the full-logit gate.

## What was wrong

The old attention projection accumulated products in a different per-lane
order and used a different reduction tree. The old down projection split K
and reassociated partial sums. Those changes altered BF16 rounding and then
logits and greedy tokens. Component isolation found exact up/gate/activation
values but mismatches in down and attention. Preserving the native MLX
accumulation order, reduction tree and rounding boundaries removed the errors.
No tolerance was loosened and the model weights were not changed.

The selected persistent scheduler further reduces overhead by having one
thread acquire each dependency, publishing visibility through a device-memory
threadgroup barrier, staging the hidden vector once per threadgroup, and
caching readiness that cannot change back. Full-model timing omits diagnostic
visit counters; correctness stress runs enable them and require every task
to execute exactly once. Error/progress checks remain in the timed route.

## What the Cohere ideas showed here

The implementation follows [Cohere’s article](https://cohere.com/blog/megakernels)
and [pinned code](https://github.com/cohere-ai/cohere-megakernel/tree/67d0b9ca22ea3652796b715d1d1863459e0e2c3c).
Independent MoE and attention-output tiles share a grid. A persistent ready
queue starts each expert’s down projection when its own hidden tiles are ready.
Counters directly record starts before all experts finish. Separate variants
load the same down weights into threadgroup memory before or after readiness.

The first implementation of those mechanisms had substantial overhead. Its
complete, correctness-passing full-model comparison is retained separately:

{chr(10).join(ablation)}

These are a separate run session; do not subtract its rates from the final
table as if they were paired measurements. Prefetch did not beat the best
fused branch in this implementation. The pinned Cohere batch-1 schedule also
uses zero explicit MoE prefetch stages, while heavily prefetching QKV/router
weights. Our down-weight staging experiment does not reproduce that later
QKV/router overlap, and does not establish its performance on Apple GPUs.

Controlled branch ablations use 29 timed pairs after two warmups on each of
two captured-layer fixtures, varying activations and expert IDs. These numbers
are changes in latency, so negative is faster:

{chr(10).join(mechanism_table)}

Both sides include workspace initialization, the final join and diagnostic
counters. Each intermediate byte matches and every task runs exactly once.
For the fine-grained scheduler, the first down job starts with only {min(early_starts)}–{max(early_starts)} of
96 hidden tiles complete across these fixtures; the control waits for all 96.
For the prefetch variant, counters confirm weights are staged while their
activation dependencies are still unready. This distinguishes a measured
mechanism from merely reducing the number of launches.

The target-process GPU traces are retained locally, with aggregate summaries
in `correctness/results/profile-*.json`. The final original/fast traces record
roughly 75 versus 51 Compute-active command buffers per generated token and
17.50 versus 16.41 ms of Compute activity per step. Instrumentation substantially
increases CPU/driver overhead, so those trace wall times are not throughput
results. These traces do not provide per-shader timings or bandwidth counters;
the uninstrumented paired runs above are the performance evidence.

## Correctness, scope and reproduction

Five corrected variants each pass 1,424 decode steps, totaling 7,120
full-vocabulary byte comparisons, plus prefill checks. The three continuations
contain 512, 512 and 403 generated tokens; Rust stops naturally, and the two
Python requests reach the cap. This establishes equivalence on the checked
workloads, not broad coding-quality evaluation. Additional scheduler stress
varies activations, expert IDs and worker counts, including one worker and
oversubscription. Source hashes and snapshots accompany the gates and timings.
Metal API and GPU shader validation also pass for the dependency and staging
variants. The arbitrary-prompt runner passes raw-logit verification with both
the fused and persistent routes on an additional Python request.

This is a branch integration into all 48 MoE layers, not one kernel for the
entire forward pass. Original MLX still handles layer 0, prefill, normalization,
QKV, attention, routing and the output head. The checkpoint is the community
affine-W4/group64 conversion, not Cohere’s H100 BF16 setup. The only tested
device is this M4 Pro (20 GPU cores, 48 GB), macOS 27.0 / 26A5425a, MLX 0.32.2.

Use `experiments/north/quantized/README.md` for installation and commands,
`experiments/north/correctness/COHERE_REFERENCE.md` for the mechanism mapping,
and `experiments/north/generate.py --mode exact --verify --prompt '...'` to
try your own prompt. Use `--mode fast` for the persistent scheduler.

Model: `mlx-community/North-Mini-Code-1.0-4bit`, revision
`dfbe084dfa26e241345af99ca32848f38fd865f9`. Reference source: MLX-VLM revision
`cdc745ad8a32d162f6d8e9d08be256910d663ac2`. Model files are downloaded and
verified separately; the result packet contains no weights. Nothing has been
posted, pushed or sent to Cohere.
'''
(NORTH.parents[1]/'NORTH_RESULTS.md').write_text(report)
(OUT/'RESULTS.md').write_text(report)
print('\n'.join(table));print('\n'.join(gain_table))
