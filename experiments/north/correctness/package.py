"""Build a source-and-evidence bundle without weights or system traces."""
import hashlib,json,zipfile
from pathlib import Path
NORTH=Path(__file__).resolve().parents[1];ROOT=NORTH.parents[1];SHARE=NORTH/'share'
files=[ROOT/'NORTH_RESULTS.md',ROOT/'STATUS.md']
files += [NORTH/name for name in ['README.md','reference.py','prepare.py','baseline.py','export_layer.py','generate.py']]
files += [NORTH/'quantized'/name for name in ['README.md','REVIEW.md','NOTICE.md','requirements.txt','kernel.h','kernels.py','exact.py','patch.py']]
files += [NORTH/'results'/name for name in ['model-provenance.json','source-provenance.json','experiment-provenance.json','layer-provenance.json','layer7-provenance.json','baseline.jsonl']]
for directory in [NORTH/'correctness',NORTH/'persistent']:
 files += [p for p in directory.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix in ['.py','.h','.md','.json','.jsonl','.log','.txt']]
files += [SHARE/name for name in ['SHARE.md','RESULTS.md','summary.json','north-m4-pro-results.png','north-m4-pro-results.svg']]
files=sorted(set(files));assert all(p.is_file() for p in files)
manifest={str(p.relative_to(ROOT)):dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in files}
archive=SHARE/'north-m4-pro-research.zip'
historical=NORTH/'history/failed-prototype-packet.zip'
if archive.exists() and not historical.exists():archive.replace(historical)
with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
 for p in files:z.write(p,p.relative_to(ROOT))
 z.writestr('MANIFEST.json',json.dumps(manifest,indent=2)+'\n')
 z.writestr('README.md','''# North Mini Code on Apple M4 Pro

Start with NORTH_RESULTS.md and experiments/north/share/north-m4-pro-results.png.
Reproduction commands: experiments/north/quantized/README.md.
Cohere mechanism mapping: experiments/north/correctness/COHERE_REFERENCE.md.
Tweet drafts: experiments/north/share/SHARE.md.

This bundle contains corrected code and raw evidence. It contains no model
weights or raw system traces. The preparation script downloads and verifies
the pinned public model and runtime source separately. The experiment was
tested on a 48 GB M4 Pro with macOS 27.0 and MLX 0.32.2. All model and GPU runs
require a suitable Apple device; only this specific device has been tested.

The source snapshots under correctness/results identify the exact versions
used for each recorded gate and benchmark. Older failed-prototype evidence
is retained for diagnosis and is not pooled into the final estimates.
''')
with zipfile.ZipFile(archive) as z:
 assert z.testzip() is None
 for name,record in manifest.items():
  data=z.read(name)
  assert len(data)==record['bytes'] and hashlib.sha256(data).hexdigest()==record['sha256'],name
digest=hashlib.sha256(archive.read_bytes()).hexdigest()
(SHARE/'packet-sha256.txt').write_text(digest+'  '+archive.name+'\n')
print(json.dumps(dict(archive=str(archive),bytes=archive.stat().st_size,files=len(files)+2,sha256=digest),indent=2))
