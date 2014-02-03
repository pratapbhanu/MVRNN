import numpy as np

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

                
        
        
        
        