"""Load the pinned text model implementation without importing multimedia frontends."""
import importlib,json,sys,types
from pathlib import Path
import mlx.core as mx
import mlx.nn as nn

ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT/'work/mlx-vlm/mlx_vlm'
# Only bypass package __init__ side effects; model/math files are unmodified.
for name,path in [('mlx_vlm',SOURCE),('mlx_vlm.models',SOURCE/'models'),('mlx_vlm.models.cohere2_moe',SOURCE/'models/cohere2_moe')]:
 module=types.ModuleType(name);module.__path__=[str(path)];sys.modules[name]=module
ModelConfig=importlib.import_module('mlx_vlm.models.cohere2_moe.config').ModelConfig
LanguageModel=importlib.import_module('mlx_vlm.models.cohere2_moe.language').LanguageModel

def load(path=ROOT/'work/models/North-Mini-Code-1.0-4bit'):
 config=json.loads((path/'config.json').read_text())
 model=LanguageModel(ModelConfig.from_dict(config))
 weights={}
 for file in sorted(path.glob('model-*.safetensors')): weights.update(mx.load(str(file)))
 # The conversion wraps the language model; this harness instantiates it directly.
 weights={k.removeprefix('language_model.'):v for k,v in weights.items()}
 weights=model.sanitize(weights)
 quant=config['quantization']
 nn.quantize(model,group_size=quant['group_size'],bits=quant['bits'],mode=quant['mode'],
             class_predicate=lambda p,m: hasattr(m,'to_quantized') and p+'.scales' in weights)
 model.load_weights(list(weights.items()),strict=True)
 model.eval()
 mx.eval(model.parameters())
 return model,config

if __name__=='__main__':
 print('text model imports succeeded',mx.__version__)
