import numpy as np

def reformatTree(thisNode, t, upnext):
    #% binarize
    
    kids = t.kids[:,thisNode]
    kids = kids(np.where(kids > 0)[0]);
    kkk = t.isLeafnode[kids[0]]
    
    while len(kids) == 1 and kkk != 1 :
        kkids = t.kids[:,kids[0]]
        kkids = kkids[np.where(kkids > 0)[0]]
        
        t.pp[kids[0]] = -1;
        t.pp[kkids] = thisNode;
        t.kids[:len[kkids],thisNode] = kkids;
        
        kids = kkids;
        kkk = t.isLeafnode[kids[0]]
    
    
    numnode = 0;
    kkk = t.isLeafnode[kids[0]];
    if len(kids) == 1 and kkk :
        t.isLeafnode[thisNode] = 1;
        t.pp[kids[0]] = -1;
        t.kids[:,thisNode] = 0;
        inc = 0;
        numnode = 1;    
    else:
        inc = 0;
    
        for k in range(len(kids)):
            kkk = t.isLeafnode[kids[k]];
            if not kkk:
                [thisinc, thisnumnode, newt] = reformatTree(kids(k), t, upnext+inc);
                inc = inc+ thisinc;
                t = newt;
                numnode = numnode+thisnumnode;
            else:
                numnode = numnode+1;
            
        
    
        next = upnext + inc;
        n = len(kids);
        last = kids[-1];
        start = n-1;
        while n >= 2:
            if (n == 2):
                next = thisNode;
            else:
                next = next + 1;
                inc = inc+1;
            
            
            t.pp[last] = next;
            t.pp[kids[start]] = next;
            
            t.kids[:, next] = 0;
            t.kids[1, next] = kids[start];
            t.kids[2, next] = last;
    
    
            last = next;
            start = start-1;
            n = n - 1;
        
    
    
    newt = t;
    
    return [inc, numnode, newt] 