"""Same ready task engine, but future weight prefixes stay in thread registers.

The exact consumer lane owns its retained weights. This removes threadgroup
staging traffic and changes resource pressure; it is not assumed to be faster.
"""
from ready_tail import build_source as shared_source

def build_source():
    h,s=shared_source()
    old='threadgroup bfloat staged[(ROWS*PREFIX)>0?ROWS*PREFIX:1];'
    assert old in s
    # Dense threads own4 rows per phase. Router threads own4 per phase and
    # disjoint1024-element K blocks. Unused slots are never consumed.
    s=s.replace(old,'''constexpr uint dense_slots=(ROWS/32)*4*(PREFIX/32);
constexpr uint router_slots=(ROUTER_ROWS/4)*4*((PREFIX+1023)/1024)*4;
bfloat staged[(dense_slots+router_slots)>0?dense_slots+router_slots:1];''')
    h=h.replace('threadgroup const bfloat* staged','thread const bfloat* staged')
    h=h.replace('threadgroup bfloat* staged','thread bfloat* staged')
    h=h.replace('staged[(phase*32+sg*4+rr)*prefix+k+j]',
                'staged[(phase*4+rr)*(prefix/32)+(k/128)*4+j]')
    h=h.replace('staged[(phase*4+rr)*prefix+k+j]',
                'staged[(phase*4+rr)*((prefix+1023)/1024)*4+(k/1024)*4+j]')
    h=h.replace('uint router_rows,uint prefix,uint tid,','uint router_rows,uint prefix,uint tid,uint sg,uint lane,')
    old='for(uint i=tid;i<count*prefix;i+=256)staged[i]=w[(first+i/prefix)*2048+i%prefix];'
    assert old in h
    h=h.replace(old,'''if(job<qjobs+2*kvjobs) {
  for(uint phase=0;phase<rows/32;phase++)for(uint rr=0;rr<4;rr++)
   for(uint k=lane*4;k<prefix;k+=128)for(uint j=0;j<4;j++)
    staged[(phase*4+rr)*(prefix/32)+(k/128)*4+j]=w[(first+phase*32+sg*4+rr)*2048+k+j];
 } else {
  for(uint phase=0;phase<router_rows/4;phase++)for(uint rr=0;rr<4;rr++)
   for(uint k=sg*128+lane*4;k<prefix;k+=1024)for(uint j=0;j<4;j++)
    staged[(phase*4+rr)*((prefix+1023)/1024)*4+(k/1024)*4+j]=w[(first+phase*4+rr)*2048+k+j];
 }''')
    s=s.replace('PREFIX,tid,next_qw','PREFIX,tid,sg,lane,next_qw')
    return h,s
