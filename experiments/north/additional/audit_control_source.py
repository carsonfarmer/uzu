"""Check the historical host model/token expressions and surrounding loop path."""
import ast
import hashlib
import json
from pathlib import Path

root=Path(__file__).resolve().parents[3]
old=root/'experiments/north/full_layer/benchmark_decode.py'
new=Path(__file__).with_name('corrected_control.py')
a=ast.parse(old.read_text());b=ast.parse(new.read_text())
measure=next(n for n in a.body if isinstance(n,ast.FunctionDef) and n.name=='measure')
old_loop=next(n for n in measure.body if isinstance(n,ast.While))
expected=[old_loop.body[1],old_loop.body[2]]
loop=next(n for n in ast.walk(b) if isinstance(n,ast.For) and ast.unparse(n.target)=='step')
assert len(loop.body)==4
assert ast.unparse(loop.body[0])=='t = time.perf_counter()'
branch=loop.body[1]
assert isinstance(branch,ast.If) and ast.unparse(branch.test)=="submission == 'async'"
assert len(branch.orelse)==3
assert ast.unparse(branch.orelse[2])=='tokens.append(token)'
same=lambda x,y:ast.dump(x,include_attributes=False)==ast.dump(y,include_attributes=False)
assert same(branch.orelse[0],expected[0])
host=branch.orelse[1]
assert isinstance(host,ast.If) and ast.unparse(host.test)=="submission == 'host'"
assert len(host.body)==1 and same(host.body[0],expected[1])
assert ast.unparse(loop.body[2].test)=='retain'
assert ast.unparse(loop.body[2].body[0])=='kept.append(logits)'
assert ast.unparse(loop.body[3])=='intervals.append(time.perf_counter() - t)'
print(json.dumps(dict(historical_file=str(old.relative_to(root)),
    historical_sha256=hashlib.sha256(old.read_bytes()).hexdigest(),
    corrected_file=str(new.relative_to(root)),corrected_sha256=hashlib.sha256(new.read_bytes()).hexdigest(),
    matched_timed_expressions=[ast.unparse(n) for n in expected],
    host_branch_contains_only_direct_argmax_assignment=True,
    surrounding_loop_has_no_additional_eval=True,
    scope='AST equality of historical model/token expressions and host branch. Prefill and common final drain are separate; timing-wrapper administration is included for every variant.'),indent=2))
