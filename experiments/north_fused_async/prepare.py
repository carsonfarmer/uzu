"""Fetch or verify pinned public inputs. No credentials or Git auth helpers."""
import argparse,hashlib,json,subprocess,tarfile,tempfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
REV='cdc745ad8a32d162f6d8e9d08be256910d663ac2'
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--download',action='store_true',help='Download missing pinned sources and ~18.5 GB checkpoint')
a=p.parse_args()
source=ROOT/'work/mlx-vlm'
manifest=json.loads((Path(__file__).parent/'provenance/model.json').read_text())

def fetch(url,path,resume=False):
    command=['curl','-q','-fL','--retry','4','--connect-timeout','20','--max-time','1800']
    if resume:command+=['-C','-']
    subprocess.run(command+['-o',str(path),url],check=True)

if not source.exists():
    if not a.download:raise SystemExit('Missing work/mlx-vlm; use --download.')
    source.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(dir=source.parent) as tmp:
        archive=Path(tmp)/'source.tar.gz'
        fetch(f'https://codeload.github.com/Blaizzy/mlx-vlm/tar.gz/{REV}',archive)
        with tarfile.open(archive) as tar:tar.extractall(tmp,filter='data')
        (Path(tmp)/f'mlx-vlm-{REV}').rename(source)
# A file manifest also works for the public source archive, without .git.
source_manifest=json.loads((Path(__file__).parent/'provenance/source.json').read_text())
for name,digest in source_manifest['sha256'].items():
    with (source/name).open('rb') as f:actual=hashlib.file_digest(f,'sha256').hexdigest()
    if actual!=digest:raise SystemExit('Source hash mismatch: '+name)
print('Verified pinned runtime source',REV,flush=True)

model=ROOT/'work/models/North-Mini-Code-1.0-4bit';model.mkdir(parents=True,exist_ok=True)
for entry in manifest['files']:
    if entry is None:continue
    path=model/entry['file']
    if not path.exists() or path.stat().st_size!=entry['bytes']:
        if not a.download:raise SystemExit('Missing/incomplete '+str(path)+'; use --download.')
        # Partial files are kept separate from final verified model files.
        partial=path.with_name(path.name+'.partial')
        fetch(f"https://huggingface.co/{manifest['repo']}/resolve/{manifest['revision']}/{entry['file']}",partial,resume=True)
        with partial.open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
        if partial.stat().st_size!=entry['bytes'] or digest!=entry['sha256']:
            raise SystemExit('Download integrity failure: '+entry['file'])
        partial.replace(path)
    with path.open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
    if digest!=entry['sha256']:raise SystemExit('Model hash mismatch: '+entry['file'])
    print('Verified',entry['file'],flush=True)
print('All pinned inputs verified.')
