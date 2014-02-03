'''
Created on Dec 16, 2013

@author: bhanu
'''

#%%
#% Input file
#
#% Parse input with Stanford Parser
#% From Stanford Parser README:
#% On a Unix system you should be able to parse the English test file with the
#% following command:
#%    ./lexparser.sh input.txt > parsed.txt
#
#%things edited:
#% changed runReformat tree and used allSOStr for my indices, it's not what
#% it used to be now.

import iais.data.config as config
import numpy as np
import scipy.stats as stats
import re


class tree2(object) :
    
    # parent pointers
    pp = [];
    nodeNames=None
    nodeFeatures=None
    nodeOut=None
    leafFeatures=[];
    isLeafnode = [];
    #        % the parent pointers do not save which is the left and right child of each node, hence:
    #        % numNodes x 2 matrix of kids, [0 0] for leaf nodes
    kids = [];
    #        % matrix (maybe sparse) with L x S, L = number of unique labels, S= number of segments
    nodeLabels=[];
    score=0;
    nodeScores=[];
    pos=[]
    
    
    def getTopNode(self):
        return np.where(self.pp==0)[0];    
    
    def getKids(self,node):
    #            %kids = find(obj.pp==node);
        return self.kids[:,node]    
    
    #TODO: maybe compute leaf-node-ness once and then just check for it
    def isLeaf(self,node):
            return self.isLeafnode[node];



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


def reformatTree(thisNode, t, upnext):
    #% binarize
    
    kids = t.kids[:,[thisNode]]
    kids = kids[np.where(kids > 0)][0];
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


def runReformatTree(allSTree, allSStr):    
    numinstance = len(allSTree);
    empty = [];
    for instance in range( 0,numinstance):
        if instance % 1000 == 0:
            print 'Sentence: ', instance
        
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
            tempkids = np.where(allSTree[instance] == i)[0];
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
#        allSKids[instance] = nkids


def create_pp_trees():
    inputFile =  config.corpus_path+'wiki_trees.txt'
    
    allSStr = []
    allSTree = []
    fid = open(inputFile, 'r');
    fileLines = fid.readlines()
    fid.close()
    
    sTree = []
    c = []
    cc = []
    for i in range(len(fileLines)):
        if (i % 1000) == 0:
            print 'Line Number: ', i
        
        if len(fileLines[i]) == 0:
            continue        
        
        if fileLines[i] == 'SENTENCE_SKIPPED_OR_UNPARSABLE':
            allSStr.append([])
            allSTree.append([])
            continue        
        
        if fileLines[i] == 'Sentence skipped: no PCFG fallback.':
            cc.append(len(allSTree))
            continue        
        
        line = fileLines[i].split()
        if len(line) == 0:
            continue
        
        if len(sTree) == 0:        
            sStr = ['']                    
            if (fileLines[i][0] == '(')  and (fileLines[i][1] == '(') :
                if(len(line) > 1):
                    line = ['(' + line[0][1:] + line[1:] ]
                else:
                    line = ['(' + ')'    ]           
            
            sTree= [0];
            lastParents = [1];
            currentParent = 1;
            if len(line)>2 :
                line = line[2:]
            else :
                continue;
        
        lineLength = len(line);
        s=0;    
        for s in range(lineLength):
            startsBranch = line[s][0] == '('
            nextIsWord = False
            if(s<lineLength-2 and (line[s+1][0] == '(') ):
#                if(not line[s+1][0] == '('):
                nextIsWord = True
            elif(s<lineLength-1 and line[s+1][-1] == ')'):
                nextIsWord = True
#            nextIsWord = s < lineLength and (line[s+1][-1] == ')') or (not (line[s+1][0] == '(') and  s < lineLength - 1)
    #        % internal nodes
            if startsBranch and not nextIsWord :
                sTree = [currentParent] + sTree
                sStr.append('');
                currentParent=len(sTree);
                lastParents += [currentParent]
                s = s+1;
                continue;
            
            
            if startsBranch and nextIsWord:
                numWords = 1;
                
                startBIdx = [ma.start() for ma in re.finditer('(?=\()', line[s+numWords])]
                endBIdx = [ma.start() for ma in re.finditer('(?=\))', line[s+numWords])]
#                startBIdx = str(line[s+numWords]).index('(')
#                endBIdx = str(line[s+numWords]).index(')')
                while len(endBIdx) <= len(startBIdx):
                    word = line[s+numWords]
    
                    sStr.append([word])
    
                    sTree = [currentParent] + sTree
                    
                    numWords = numWords+1;
                    assert(s+numWords <= lineLength);
                    startBIdx = [ma.start() for ma in re.finditer('(?=\()', line[s+numWords])]
                    endBIdx = [ma.start() for ma in re.finditer('(?=\))', line[s+numWords])]
                
                
                if len(startBIdx) > 0:
                    word = line[s+numWords][startBIdx+1:endBIdx-1]
                else:
                    word = line[s+numWords][0:endBIdx[0]]
                
                word = str(word).lower();
                
                sStr.append([word])
                sTree = [currentParent] + sTree
                s=s+numWords+1;
                lastParents=lastParents[0:-(len(endBIdx)-len(startBIdx))+1];
                if len(lastParents) == 0:
                    allSStr.append(sStr)                
                    allSTree.append(sTree)                
                    s=s+1;                       
                    sStr = [];                    
                    sTree= [];                
                    continue
                
                currentParent = lastParents[-1];
                continue            
    
    runReformatTree(allSTree, allSStr)
    print
    
    
if __name__ == '__main__':
    create_pp_trees()


