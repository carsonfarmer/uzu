"""Regenerate numerical summary and the shareable figure from saved raw runs."""
import json,statistics
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

BASE=Path(__file__).resolve().parent
OUT=BASE.parent/'share';OUT.mkdir(exist_ok=True)
median=statistics.median
bf=[];q=[]
for layer,prefix in [(1,'branch-real'),(7,'branch-layer7')]:
    for rep in range(1,4):
        raw=json.loads((BASE.parent/f'results/{prefix}-{rep}.json').read_text())
        control={r['repetition']:r['gpu_us'] for r in raw['samples'] if r['variant']=='phased'}
        selected=[r for r in raw['samples'] if r['variant']=='static' and r['workers']==128]
        small=[r for r in raw['samples'] if r['variant']=='static' and r['workers']==20]
        assert len(selected)==len(control)==15
        bf.append(dict(layer=layer,run=rep,phased_gpu_us=median(control.values()),
            selected_gpu_us=median(r['gpu_us'] for r in selected),
            paired_ratio=median(r['gpu_us']/control[r['repetition']] for r in selected),
            small_pool_ratio=median(r['gpu_us']/control[r['repetition']] for r in small)))
        raw=json.loads((BASE/f'results/final-layer{layer}-{rep}.json').read_text())
        control={r['repetition']:r['wall_us'] for r in raw['samples'] if r['variant']=='native_compiled'}
        medians={name:median(r['wall_us'] for r in raw['samples'] if r['variant']==name)
                 for name in dict.fromkeys(r['variant'] for r in raw['samples'])}
        ratios={name:median(r['wall_us']/control[r['repetition']] for r in raw['samples'] if r['variant']==name)
                 for name in medians}
        q.append(dict(layer=layer,run=rep,wall_us=medians,paired_ratio_vs_native_compiled=ratios,
            checks=raw['checks']))
rows=list(map(json.loads,(BASE/'results/decode.jsonl').read_text().splitlines()))
generation=[r for r in rows if r['kind']=='generation' and not r['warmup']]
teacher=[r for r in rows if r['kind']=='teacher']
assert len(generation)==27 and len(teacher)==384
summary=[];numerics=[]
for label in ['short','long','rust']:
    reference={r['repetition']:r for r in generation if r['prompt']==label and r['variant']=='original'}
    for mode in ['original','native_compiled','custom']:
        subset=[r for r in generation if r['prompt']==label and r['variant']==mode]
        rates=[r['decode_tokens_per_second'] for r in subset]
        firstdiff=[]
        for r in subset:
            tokens=reference[r['repetition']]['token_ids']
            firstdiff.append(next((i for i,(a,b) in enumerate(zip(tokens,r['token_ids'])) if a!=b),
                None if len(tokens)==len(r['token_ids']) else min(len(tokens),len(r['token_ids']))))
        summary.append(dict(prompt=label,variant=mode,median_tps=median(rates),min_tps=min(rates),max_tps=max(rates),
            median_ttft_seconds=median(r['ttft_seconds'] for r in subset),
            median_peak_bytes=median(r['peak_memory_bytes'] for r in subset),
            tokens=[r['generated_tokens'] for r in subset],first_different_token_zero_based=firstdiff,
            repeat_tokens_identical=all(r['token_ids']==subset[0]['token_ids'] for r in subset),
            paired_tps_ratio=median(r['decode_tokens_per_second']/reference[r['repetition']]['decode_tokens_per_second'] for r in subset)))
    for mode in ['native_compiled','custom']:
        subset=[r for r in teacher if r['prompt']==label and r['variant']==mode]
        numerics.append(dict(prompt=label,variant=mode,steps=len(subset),
            greedy_matches=sum(r['reference_top']==r['candidate_top'] for r in subset),
            exact_logits=sum(r['max_abs']==0 for r in subset),max_relative_l2=max(r['relative_l2'] for r in subset),
            median_relative_l2=median(r['relative_l2'] for r in subset),
            max_abs=max(r['max_abs'] for r in subset),max_kl=max(r['kl'] for r in subset),
            all_finite=all(r['finite'] for r in subset)))
result=dict(bf16_branch=bf,quantized_branch=q,decode=summary,teacher=numerics)
(OUT/'summary.json').write_text(json.dumps(result,indent=2)+'\n')

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,
                     'axes.spines.right':False,'axes.titleweight':'bold'})
fig,axes=plt.subplots(1,3,figsize=(16,7),gridspec_kw={'width_ratios':[.9,1,1.35]})
fig.patch.set_facecolor('#f7f7f4')
for ax in axes:ax.set_facecolor('#f7f7f4')
fig.suptitle('Testing megakernel ideas on Apple M4 Pro',x=.07,ha='left',y=.96,fontsize=23,weight='bold')
fig.text(.07,.895,'North Mini Code · 48 GB Mac · batch 1 · MLX 0.32.2 · local experiment, September 2026',fontsize=12,color='#444444')
colors=['#276d85','#87a9b0','#cf713e']
ax=axes[0]
for layer,x in [(1,0),(7,1)]:
    vals=[100*(r['paired_ratio']-1) for r in bf if r['layer']==layer]
    ax.scatter(np.linspace(x-.1,x+.1,3),vals,s=60,color=colors[0],zorder=3)
    ax.hlines(median(vals),x-.2,x+.2,color=colors[0],linewidth=3)
ax.axhline(0,color='#999999',linewidth=1)
ax.set_xticks([0,1],['Layer 1','Layer 7']);ax.set_xlim(-.55,1.55);ax.set_ylim(-12,1)
ax.set_ylabel('GPU time change vs phased (%)')
ax.set_title('1. BF16 branch schedule',loc='left',fontsize=13,pad=15)
ax.text(0,-.24,'Static 128 workers; same arithmetic.\nThree runs per captured layer.\nMaterialized BF16 expert weights.',transform=ax.transAxes,va='top',fontsize=10,color='#444444')
ax.grid(axis='y',alpha=.15)
ax=axes[1]
for layer,x in [(1,0),(7,1)]:
    vals=[100*(r['paired_ratio_vs_native_compiled']['c64_interleaved128_compiled']-1) for r in q if r['layer']==layer]
    ax.scatter(np.linspace(x-.1,x+.1,3),vals,s=60,color=colors[2],zorder=3)
    ax.hlines(median(vals),x-.2,x+.2,color=colors[2],linewidth=3)
ax.axhline(0,color='#999999',linewidth=1)
ax.set_xticks([0,1],['Layer 1','Layer 7']);ax.set_xlim(-.55,1.55)
ax.set_ylabel('Wall time change vs compiled MLX (%)')
ax.set_title('2. Packed 4-bit branch',loc='left',fontsize=13,pad=15)
ax.text(0,-.24,'64-feature tiles; 128 mixed workers.\nBoth paths compiled in MLX.\nPositive values mean slower.',transform=ax.transAxes,va='top',fontsize=10,color='#444444')
ax.grid(axis='y',alpha=.15)
ax=axes[2];x=np.arange(3);width=.24
for i,(mode,label) in enumerate([('original','Original MLX'),('native_compiled','Compiled control'),('custom','Custom Metal')]):
    vals=[next(r['median_tps'] for r in summary if r['variant']==mode and r['prompt']==p) for p in ['short','long','rust']]
    lo=[next(r['min_tps'] for r in summary if r['variant']==mode and r['prompt']==p) for p in ['short','long','rust']]
    hi=[next(r['max_tps'] for r in summary if r['variant']==mode and r['prompt']==p) for p in ['short','long','rust']]
    bars=ax.bar(x+(i-1)*width,vals,width,color=colors[i],label=label,
        yerr=[np.array(vals)-lo,np.array(hi)-vals],capsize=2,error_kw={'elinewidth':.8,'alpha':.6})
    for bar,v in zip(bars,vals):ax.text(bar.get_x()+width/2,v+.7,f'{v:.1f}',ha='center',fontsize=9)
ax.set_xticks(x,['Short Python','Long Python','Rust']);ax.set_ylabel('Decode tokens / second');ax.set_ylim(0,65)
ax.set_title('3. Full-model generation',loc='left',fontsize=13,pad=15)
ax.legend(loc='upper left',bbox_to_anchor=(0,-.12),fontsize=8,frameon=False,ncol=3)
ax.text(0,-.24,'Median + min/max of 3 runs; 256-token cap.\nAfter warmup; all 48 MoE branches patched.\nCustom continuations differ from the reference.',transform=ax.transAxes,va='top',fontsize=10,color='#444444')
ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
fig.subplots_adjust(left=.07,right=.98,top=.78,bottom=.3,wspace=.5)
fig.text(.07,.035,'A branch scheduling gain did not produce faster or equivalent full-model decoding. Full raw runs and source accompany this figure.',fontsize=11,weight='bold')
fig.savefig(OUT/'north-m4-pro-results.png',dpi=150,facecolor=fig.get_facecolor())
fig.savefig(OUT/'north-m4-pro-results.svg',facecolor=fig.get_facecolor())
for r in summary:print(r['prompt'],r['variant'],round(r['median_tps'],3),r['first_different_token_zero_based'])
for r in numerics:print(r)
for r in q:print('quantized',r['layer'],r['run'],r['wall_us'],r['paired_ratio_vs_native_compiled']['c64_interleaved128_compiled'])
