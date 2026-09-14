"""CPU task DAG and stealable affinity lists for exact North layer arithmetic.

This is a schedule specification, not yet an executable Metal decoder. Attention
O-projection depends on all heads: splitting its reduction would alter exactness.
Future immutable weight reservations do not count as activation-ready tasks.
"""
from dataclasses import dataclass,asdict
import random

@dataclass(frozen=True)
class Task:
    id: int
    layer: int
    op: str
    tile: int
    deps: tuple[int,...]

def layer_graph(layers=2, prep_rows=64, router_rows=8):
    tasks=[]
    def add(layer,op,tile,deps):
        i=len(tasks);tasks.append(Task(i,layer,op,tile,tuple(deps)));return i
    previous=[]
    for layer in range(layers):
        norm=add(layer,'norm',0,previous)
        q=[add(layer,'q',i,[norm]) for i in range(4096//prep_rows)]
        k=[add(layer,'k',i,[norm]) for i in range(512//prep_rows)]
        v=[add(layer,'v',i,[norm]) for i in range(512//prep_rows)]
        router=[add(layer,'router',i,[norm]) for i in range(128//router_rows)]
        route=add(layer,'route',0,router)
        rope_q=[add(layer,'rope_q',h,q[h*128//prep_rows:(h+1)*128//prep_rows]) for h in range(32)]
        kv=[add(layer,'rope_kv',h,k[h*128//prep_rows:(h+1)*128//prep_rows]+v[h*128//prep_rows:(h+1)*128//prep_rows]) for h in range(4)]
        att=[add(layer,'attention',h,[rope_q[h],kv[h//8]]) for h in range(32)]
        front=[[add(layer,'expert_front',slot*12+j,[norm,route]) for j in range(12)] for slot in range(8)]
        # Sixteen128-row O tiles, sixteen128-row down tiles per selected expert.
        o=[add(layer,'oproj',j,att) for j in range(16)]
        down=[[add(layer,'expert_down',slot*16+j,front[slot]) for j in range(16)] for slot in range(8)]
        previous=[add(layer,'join',j,[o[j]]+[d[j] for d in down]) for j in range(16)]
    return tasks

def affinity_lists(tasks,workers):
    """Continuous RR cursor, then same-op dependencies bias equal-load choices."""
    lists=[[] for _ in range(workers)];owners={};cursor=0
    for t in tasks:
        minimum=min(map(len,lists))
        eligible=[(cursor+i)%workers for i in range(workers) if len(lists[(cursor+i)%workers])==minimum]
        parent_owners=[owners[d] for d in t.deps]
        worker=max(eligible,key=lambda w:parent_owners.count(w))
        lists[worker].append(t.id);owners[t.id]=worker;cursor=(worker+1)%workers
    return lists

def simulate(tasks,workers,resident,seed):
    """Random completion / partial residency stress of ready fallback policy.

    No assumptions about all workers being resident. A preferred-list blocked
    task cannot pin a worker: it steals ANY ready task. Real memory publication
    and GPU forward progress are outside this CPU model.
    """
    rng=random.Random(seed);lists=affinity_lists(tasks,workers)
    done=set();claimed=set();running={};visits=[0]*len(tasks);steals=0
    active=rng.sample(range(workers),resident)
    while len(done)<len(tasks):
        for w in active:
            if w in running:continue
            ready=lambda i:i not in claimed and all(d in done for d in tasks[i].deps)
            candidate=next((i for i in lists[w] if ready(i)),None)
            if candidate is None:
                candidate=next((t.id for t in tasks if ready(t.id)),None)
                if candidate is not None:steals+=1
            if candidate is not None:
                claimed.add(candidate);running[w]=candidate;visits[candidate]+=1
        if not running:raise AssertionError('DAG stalled with unfinished tasks')
        w=rng.choice(list(running));done.add(running.pop(w))
    assert all(v==1 for v in visits)
    return dict(workers=workers,resident=resident,seed=seed,tasks=len(tasks),steals=steals,exactly_once=True)

if __name__=='__main__':
    import json
    tasks=layer_graph()
    trials=[simulate(tasks,w,r,s) for w in (1,20,32,64) for r in sorted({1,max(1,w//2),w}) for s in range(3)]
    print(json.dumps(dict(scope='CPU dependency/claim model only, not GPU liveness or model exactness',tasks=[asdict(t) for t in tasks],trials=trials),indent=2))
