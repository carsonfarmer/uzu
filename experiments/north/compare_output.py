"""Compare the standalone Metal branch to two independently evaluated MLX references."""
import argparse,json
from pathlib import Path
import numpy as np
root=Path(__file__).resolve().parents[2]
parser=argparse.ArgumentParser()
parser.add_argument('--tensors',default='work/north-layer')
parser.add_argument('--metal-output',default='work/north-metal/output.bin')
parser.add_argument('--result',default='experiments/north/results/real-layer-numerics.json')
args=parser.parse_args()
def read(path):
 return (np.fromfile(path,dtype=np.uint16).astype(np.uint32)<<16).view(np.float32).astype(np.float64)
result={}
actual=read(root/args.metal_output)
for name in ['materialized_output','output']:
 expected=read(root/args.tensors/f'{name}.bin')
 diff=actual-expected
 result[name]={'max_abs':float(abs(diff).max()),'relative_l2':float(np.linalg.norm(diff)/np.linalg.norm(expected)),
  'cosine':float(np.dot(actual,expected)/(np.linalg.norm(actual)*np.linalg.norm(expected))),
  'reference_max_abs':float(abs(expected).max()),'finite':bool(np.isfinite(actual).all()),'elements':actual.size}
print(json.dumps(result,indent=2))
(root/args.result).write_text(json.dumps(result,indent=2)+'\n')
