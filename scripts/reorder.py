import numpy as np

def reorder(thisNode, t, nextleaf, nextnode, opp, okids, opos):

    nnextleaf = nextleaf;
    nnextnode = nextnode - 1;
    nkids = okids;
    
    pp = opp;
    pos = opos;
    
    kids = t.kids[:,thisNode];
    kids = kids[np.where(kids > 0)[0]];
    
    for k in [0,1] :
        kkk = t.isLeafnode[kids[k]]
        if kkk :
            pp[nnextleaf] = nextnode;
            nkids[nextnode,k] = nnextleaf;            
            nnextleaf = nnextleaf+1;
        else:
            pp[nnextnode] = nextnode;
            nkids[nextnode,k] = nnextnode;            
            
            [pp, nnextleaf, nnextnode, nkids, pos] = reorder(kids(k), t, nnextleaf, nnextnode, pp, nkids, pos);
        
    
    return [pp, nnextleaf, nnextnode, nkids, pos] 