"""Round-robin future task reservations across Q, K, V, and router operations."""
from tuned_tail import build_source as base
from ready_tail import U

def job_order(ticket,rows=32,router_rows=8):
    q=4096//rows;k=512//rows;r=128//router_rows;n=min(k,r)
    if ticket<4*n:
        kind=ticket%4;tile=ticket//4
        return (0,q,q+k,q+2*k)[kind]+tile
    ticket-=4*n
    for offset,size in ((0,q),(q,k),(q+k,k),(q+2*k,r)):
        left=size-n
        if ticket<left:return offset+n+ticket
        ticket-=left
    raise ValueError('ticket out of range')

def build_source(register=False):
    h,s=base(register)
    assert s.count('owned=job;')==2
    s=s.replace('owned=job;','owned=north_prep_job(job,ROWS,ROUTER_ROWS);')
    h+='''
inline uint north_prep_job(uint ticket,uint rows,uint router_rows) {
 uint q=4096/rows,k=512/rows,r=128/router_rows,n=min(k,r);
 if(ticket<4*n){uint kind=ticket%4,tile=ticket/4;return (kind==0?0:(kind==1?q:(kind==2?q+k:q+2*k)))+tile;}
 ticket-=4*n;
 if(ticket<q-n)return n+ticket;ticket-=q-n;
 if(ticket<k-n)return q+n+ticket;ticket-=k-n;
 if(ticket<k-n)return q+k+n+ticket;ticket-=k-n;
 return q+2*k+n+ticket;
}
'''
    needle=f'if(AUDIT)atomic_fetch_add_explicit(state+{U["EARLY_ROUNDS"]},1u,memory_order_relaxed);'
    assert needle in s
    s=s.replace(needle,needle+f'''if(AUDIT){{uint kind=owned<4096/ROWS?0:(owned<4608/ROWS?1:(owned<5120/ROWS?2:3));atomic_fetch_add_explicit(state+{U['ERROR']+1}+kind,1u,memory_order_relaxed);}}''')
    return h,s

if __name__=='__main__':
    import json
    results=[]
    for rows in (32,64):
        order=[job_order(i,rows) for i in range(5120//rows+16)]
        assert sorted(order)==list(range(len(order)))
        results.append(dict(rows=rows,all_jobs_exactly_once=True,first32=order[:32]))
    print(json.dumps(dict(scope='CPU task permutation check only',results=results),indent=2))
