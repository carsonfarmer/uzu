"""Summarize target-process Compute intervals from an xctrace XML export.

Do not equate command-buffer counts with kernel-dispatch counts. This trace
template does not provide per-shader durations unless shader profiling is on.
Only aggregate target-process facts are written to the shareable result.
"""
import argparse,json,xml.etree.ElementTree as ET
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--gpu',required=True);p.add_argument('--pid',type=int,required=True)
p.add_argument('--steps',type=int,required=True);p.add_argument('--label',required=True);p.add_argument('--output',required=True)
a=p.parse_args();root=ET.parse(a.gpu).getroot()
refs={e.get('id'):e for e in root.iter() if e.get('id')}
def resolve(e):return refs[e.get('ref')] if e.get('ref') else e
intervals=[];command_buffers=set()
for row in root.iter('row'):
 v=[resolve(e) for e in row]
 proc=v[10];pid=proc.find('pid');pid=resolve(pid) if pid is not None else None
 if pid is None or int(pid.text)!=a.pid or v[2].text!='Compute':continue
 start=int(v[0].text);intervals.append((start,start+int(v[1].text)))
 command_buffers.add(v[15].text)
assert intervals,'No target Compute events'
merged=[]
for start,end in sorted(intervals):
 if merged and start<=merged[-1][1]:merged[-1][1]=max(end,merged[-1][1])
 else:merged.append([start,end])
active=sum(end-start for start,end in merged);span=merged[-1][1]-merged[0][0]
out=dict(label=a.label,steps=a.steps,compute_active_ms=active/1e6,compute_span_ms=span/1e6,
 compute_active_ms_per_step=active/1e6/a.steps,compute_active_fraction_of_span=active/span,
 command_buffers=len(command_buffers),command_buffers_per_step=len(command_buffers)/a.steps,
 compute_intervals=len(intervals),
 note='Instrumented 128-step short-prompt run, separate from uninstrumented throughput. Union of target Compute intervals. No per-kernel timing, memory bandwidth or hardware-counter claim. Gaps include CPU work, waits and desktop contention.')
Path(a.output).write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
