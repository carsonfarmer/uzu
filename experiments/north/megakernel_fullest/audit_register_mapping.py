"""CPU audit of register prefix ownership against the unchanged consumer loops."""
import json
checks=[]
for rows,prefix in ((32,128),(32,256),(64,128)):
    rr_count=8;slots=rows//32*4*(prefix//32)+rr_count//4*4*((prefix+1023)//1024)*4
    reads=0
    for router in (False,True):
        for sg in range(8):
            for lane in range(32):
                loaded={}
                phases=rr_count//4 if router else rows//32
                ks=range(sg*128+lane*4,prefix,1024) if router else range(lane*4,prefix,128)
                for phase in range(phases):
                    for rr in range(4):
                        for k in ks:
                            for j in range(4):
                                index=(phase*4+rr)*((prefix+1023)//1024)*4+(k//1024)*4+j if router else (phase*4+rr)*(prefix//32)+(k//128)*4+j
                                address=(phase*4+rr)*2048+k+j if router else (phase*32+sg*4+rr)*2048+k+j
                                assert index not in loaded and index<slots
                                loaded[index]=address
                # Walk the full original2048-element K reduction, reading staging only for prefix.
                fullks=range(sg*128+lane*4,2048,1024) if router else range(lane*4,2048,128)
                for phase in range(phases):
                    for k in fullks:
                        for rr in range(4):
                            for j in range(4):
                                if k+j>=prefix:continue
                                index=(phase*4+rr)*((prefix+1023)//1024)*4+(k//1024)*4+j if router else (phase*4+rr)*(prefix//32)+(k//128)*4+j
                                expected=(phase*4+rr)*2048+k+j if router else (phase*32+sg*4+rr)*2048+k+j
                                assert loaded[index]==expected;reads+=1
    checks.append(dict(rows=rows,prefix=prefix,checked_reads=reads,max_thread_slots=slots,pass_mapping=True))
print(json.dumps(dict(scope='CPU address/ownership audit only; GPU compiler allocation and numerical exactness unverified',checks=checks),indent=2))
