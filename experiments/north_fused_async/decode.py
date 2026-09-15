"""Fixed-length greedy decode, including full logits, argmax, readback and drain.

The async path enqueues the next dependent graph before reading the previous
GPU token on the CPU. This standard MLX pattern is not speculative decoding.
Prefill and its first output token are outside the timer. Early EOS is rejected;
this is a reproducible decode runner, not a streaming chat frontend.
"""
import time
import mlx.core as mx

def generate(model,ids,tokens,submission,eos,retain=False):
    if tokens < 2 or not ids:
        raise ValueError("Use a nonempty prompt and at least two output tokens")
    if submission not in ("host", "async"):
        raise ValueError("submission must be host or async")
    count=tokens
    cache=model.make_cache()
    for i in range(0,len(ids),256):
        logits=model(mx.array([ids[i:i+256]]),cache=cache).logits;mx.eval(logits)
    first=int(mx.argmax(logits[0,-1]).item());current=mx.array(first,mx.int32);token=first
    tokens=[first];pending=None;kept=[];intervals=[]
    start=time.perf_counter()
    for step in range(count-1):
        t=time.perf_counter()
        if submission=='async':
            logits=model(current.reshape(1,1),cache=cache).logits
            current=mx.argmax(logits[0,-1])
            mx.async_eval(logits,current)
            if pending is not None:tokens.append(int(pending.item()))
            pending=current
        else:
            # Historical benchmark_decode.py host-token loop, unchanged.
            logits=model(mx.array([[token]]),cache=cache).logits
            if submission=='host':
                token=int(mx.argmax(logits[0,-1]).item())
            else:raise ValueError(submission)
            tokens.append(token)
        if retain:kept.append(logits)
        intervals.append(time.perf_counter()-t)
    if submission=='async':tokens.append(int(pending.item()))
    mx.synchronize()
    elapsed=time.perf_counter()-start
    assert len(tokens)==count
    assert not (set(tokens[:-1])&eos),'Early EOS: fixed-length benchmark inapplicable'
    return tokens,kept,elapsed,intervals
