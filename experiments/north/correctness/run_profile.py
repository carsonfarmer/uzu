"""Collect a warmed full-model GPU trace without tracing model loading."""
import argparse,hashlib,json,subprocess,sys,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--mode',required=True);p.add_argument('--prefix',required=True)
a=p.parse_args();prefix=Path(a.prefix).resolve()
assert not prefix.with_suffix('.ready').exists() and not prefix.with_suffix('.trace').exists(),'Choose a fresh output prefix'
signal=prefix.with_suffix('.go');log=prefix.with_suffix('.log');trace=prefix.with_suffix('.trace')
tracelog=prefix.with_suffix('.xctrace.log');here=Path(__file__).resolve().parent
child=recorder=None
try:
 with log.open('w') as out:
  child=subprocess.Popen([sys.executable,str(here/'profile_decode.py'),'--mode',a.mode,'--signal',str(signal)],stdout=out,stderr=subprocess.STDOUT)
  deadline=time.monotonic()+180
  while not prefix.with_suffix('.ready').exists():
   if child.poll() is not None:raise RuntimeError(log.read_text())
   if time.monotonic()>deadline:raise TimeoutError('Model warmup')
   time.sleep(.05)
  pid=int(prefix.with_suffix('.ready').read_text())
  with tracelog.open('w') as recording_out:
   recorder=subprocess.Popen(['xcrun','xctrace','record','--template','Metal System Trace','--attach',str(pid),'--time-limit','30s','--output',str(trace),'--no-prompt'],stdout=recording_out,stderr=subprocess.STDOUT)
   deadline=time.monotonic()+25
   while 'Starting recording' not in tracelog.read_text():
    if recorder.poll() is not None:raise RuntimeError(tracelog.read_text())
    if time.monotonic()>deadline:raise TimeoutError('Profiler attach')
    time.sleep(.05)
   time.sleep(1)
   signal.touch();child.wait(timeout=120);recorder.wait(timeout=90)
   assert child.returncode==recorder.returncode==0,(child.returncode,recorder.returncode)
 print(log.read_text(),flush=True)
 gpu=prefix.with_suffix('.gpu.xml')
 subprocess.run(['xcrun','xctrace','export','--input',str(trace),'--xpath','/trace-toc/run[@number="1"]/data/table[@schema="metal-gpu-intervals"]','--output',str(gpu)],check=True)
 subprocess.run([sys.executable,str(here/'summarize_trace.py'),'--gpu',str(gpu),'--pid',str(pid),'--steps','128','--label',a.mode,'--output',str(prefix.with_suffix('.summary.json'))],check=True)
finally:
 for process in [child,recorder]:
  if process is not None and process.poll() is None:
   process.terminate();process.wait(timeout=10)
