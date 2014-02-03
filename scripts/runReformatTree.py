#%% reformat the tree structures for use

import numpy as np
import scipy.stats as stats
from scripts.tree2 import tree2
from scripts.reformatTree import reformatTree
from scripts.reorder import reorder

def runReformatTree(allSTree, allSStr):
    
    numinstance = len(allSTree);
    empty = [];
    for instance in range( 0,numinstance):
        if instance % 1000 == 0:
            print 'Sentence: ', instance
        
        # get embeddings
        n = len(allSTree[instance]);
        
        cnt = 0;
        for j in range( 0,len(allSStr[instance])):
            if not (len(allSStr[instance][j])):
                cnt = cnt+1;
            

        if cnt < 2:  # words in sentence:
            empty.append(instance)
            continue
        
        
        t = tree2();
        t.pp = np.zeros(n);
        t.pp[:n] = allSTree[instance]
        mostkids = len(np.where(allSTree[instance] == stats.mode(allSTree[instance]))) # largest number of kids one node has
        t.kids = np.zeros((mostkids,n))
        for i in range(n):
            tempkids = np.where(allSTree[instance] == i);
            t.kids[0:len(tempkids),i] = tempkids;
        
        
        # binarize
        [inc, numnode, newt] = reformatTree(1, t, n+1);
        
        opp = np.zeros(2*numnode-1);
        okids = np.zeros((2*numnode-1,2));
        opos = np.zeros(2*numnode-1);
        
        [pp, nnextleaf, nnextnode, nkids, pos] = reorder(1, newt, 1, 2*numnode-1, opp, okids, opos);
        

        newstr = range(1,numnode);
        nxt = 1;
        for i in range(len(allSTree[instance])):
            newstr[nxt] = allSStr[instance][i];                
            nxt = nxt + 1;
            
        allSStr[instance] = newstr;       
        allSTree[instance] = pp;
        
        
        
    