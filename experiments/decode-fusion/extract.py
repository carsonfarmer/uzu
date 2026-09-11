"""Extract layer 0 tensors for the isolated BF16/W4 prototype, without model dependencies."""
import argparse
import json
import struct
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('model', type=Path)
parser.add_argument('output', type=Path)
args = parser.parse_args()
with args.model.open('rb') as source:
    header_size = struct.unpack('<Q', source.read(8))[0]
    header = json.loads(source.read(header_size))
    prefix = 'decoder.transformer.layers.0.mlp.'
    tensors = {
        'weights': ('up_projection.weights.quantized.weights', 'U8', [18432, 1280]),
        'scales': ('up_projection.weights.quantized.scales', 'BF16', [18432, 80]),
        'zeros': ('up_projection.weights.quantized.zero_points', 'U8', [18432, 40]),
        'output_factors': ('up_projection.weights.incoherence_signs.output_signs', 'I32', [18432]),
        'input_factors': ('down_projection.weights.incoherence_signs.input_signs', 'I32', [9216]),
    }
    spec = json.loads(header['__metadata__'][prefix+'up_projection.weights.quantized.spec'])
    assert spec == dict(type='IntSpec', bits=4, group_size=32, is_symmetric=False, layout='output_input'), spec
    args.output.mkdir(parents=True, exist_ok=True)
    for name, (key, dtype, shape) in tensors.items():
        tensor = header[prefix+key]
        assert tensor['dtype'] == dtype and tensor['shape'] == shape, tensor
        start, end = tensor['data_offsets']
        source.seek(8+header_size+start)
        data = source.read(end-start)
        assert len(data) == end-start, 'model download incomplete for this tensor'
        (args.output / (name+'.bin')).write_bytes(data)
