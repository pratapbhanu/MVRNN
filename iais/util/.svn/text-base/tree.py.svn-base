'''
Created on Oct 20, 2013

@author: bhanu
'''

import numpy as np


def addOuterContext(thisTree, indicies, sentLen, params):
    n = params.numOutContextWords;
    wsz = params.wordSize;
    wszI = np.eye(wsz);
    
    #element 1
    o1 = np.array(range(max([indicies[0]-n,0]), indicies[0]), dtype='int32')
    if len(o1) == 0:
        o1 = np.array([indicies[0]], dtype='int32')
    
    o1L = len(o1)
    thisTree.poolMatrix = np.vstack((
                                    np.hstack((thisTree.poolMatrix, np.zeros((thisTree.poolMatrix.shape[0],wsz)))),
                                    np.hstack((np.zeros((o1L*wsz,thisTree.poolMatrix.shape[1])), np.tile((1.0/o1L)*wszI,(o1L,1))))
                                    ))
    
    # element 2
    o2 = np.array(range(indicies[1]+1, min([indicies[1]+n+1,sentLen])), dtype='int32')
    if len(o2) == 0:
        o2 = np.array([indicies[1]], dtype='int32')
    o2L = len(o2)
    
    thisTree.poolMatrix = np.vstack((
                                     np.hstack((thisTree.poolMatrix, np.zeros((thisTree.poolMatrix.shape[0],wsz)))), 
                                     np.hstack((np.zeros((o2L*wsz,thisTree.poolMatrix.shape[1])), np.tile((1.0/o2L)*wszI,(o2L,1))))
                                     ))
    
    thisTree.nodePath = np.concatenate((thisTree.nodePath, o1, o2))
    thisTree.pooledVecPath = np.hstack((thisTree.pooledVecPath, np.mean(thisTree.nodeAct_a[:,o1],1).reshape(wsz,1), np.mean(thisTree.nodeAct_a[:,o2],1).reshape(wsz,1)))
    return thisTree


def addNN(thisTree, Wv, sNN, params):
    wsz = params.wordSize
    wszI = np.eye(wsz)
    thisTree.NN_vecs = np.hstack((np.mean(Wv[:,sNN[:params.NN,0]],1).reshape(wsz,1), np.mean(Wv[:,sNN[:params.NN,1]],1).reshape(wsz,1)))
    thisTree.poolMatrix = np.vstack((
                                     np.hstack((thisTree.poolMatrix, np.zeros((thisTree.poolMatrix.shape[0],wsz)))),
                                     np.hstack((np.zeros((wsz,thisTree.poolMatrix.shape[1])), (1.0/params.NN)*wszI))
                                    ))
    
    thisTree.poolMatrix = np.vstack((
                                     np.hstack((thisTree.poolMatrix, np.zeros((thisTree.poolMatrix.shape[0],wsz)))),
                                     np.hstack((np.zeros((wsz,thisTree.poolMatrix.shape[1])), (1.0/params.NN)*wszI))
                                    ))
    return thisTree


def addInnerContext(thisTree, indicies, Wv, sentLen, params):
    wsz = params.wordSize;
    n = params.numInContextWords;
    
    # element1 
    o1 = np.arange(indicies[0]+1, min([indicies[0]+n+1,sentLen]), dtype='int32')    
    if len(o1) < n :
        o1 = np.concatenate((o1, np.tile(sentLen,(n-len(o1)))))
    
    thisTree.poolMatrix = np.vstack((
                                     np.hstack((thisTree.poolMatrix, np.zeros((thisTree.poolMatrix.shape[0],wsz*n)))),
                                     np.hstack((np.zeros((wsz*n,thisTree.poolMatrix.shape[1])),  np.eye(wsz*n)))
                                    ))
    
    # element 2
    o2 = np.arange(max([indicies[1]-n,0]),indicies[1], dtype='int32')
    vecs = thisTree.nodeAct_a[:,o2]
    o2L = len(o2)
    if o2L < n : # use PADDING for this case
        o2 = np.hstack((np.zeros(n-o2L), o2))
        vecs = np.hstack((np.tile(Wv[:,[1]],(1,n-o2L)), vecs))
    
    thisTree.poolMatrix = np.vstack((
                                     np.hstack((thisTree.poolMatrix, np.zeros((thisTree.poolMatrix.shape[0],wsz*n)))),
                                     np.hstack((np.zeros((wsz*n,thisTree.poolMatrix.shape[1])), np.eye(wsz*n)))
                                     ))
    
    thisTree.nodePath = np.concatenate((thisTree.nodePath, o1, o2))
    thisTree.pooledVecPath = np.hstack((thisTree.pooledVecPath,thisTree.nodeAct_a[:,o1],vecs))
    return thisTree


def findPath(thisTree, indicies):

    elem1 = indicies[0]
    elem2 = indicies[1]    
    
    parent = thisTree.pp[elem1]
    path1 = []
    while parent != -1 :
        path1.append(parent)
        parent = thisTree.pp[parent]
    
    path1.append(-1)
    
    path2 = []
    parent = thisTree.pp[elem2]
    while not (parent in path1) :
        path2.append(parent)
        parent = thisTree.pp[parent]
    
    for i, node in enumerate(path1):
        if(node == parent):
            break
    fullPath = path1[:i+1] + path2[::-1]

    return np.array(fullPath)

def getInternalFeatures(path, sStr, indicies, feat_mean, feat_std):
    # features [path length, path depth1, path depth2, sentence length, length between elements]
    path = np.concatenate((np.array([0]),path,np.array([0])))
    # get depth
    depth1 = np.max(path)
    depth2 = len(path) - depth1 + 1;
    # account for missing leafs
    features = np.array([len(path), depth1, depth2, len(sStr), (indicies[1] - indicies[0])])    
    features = features - feat_mean
    features = np.divide(features, feat_std)    
    return features

class Tree(object):
    '''
    classdocs
    '''

    def __init__(self):
        self.pp = [];
        self.nodeNames = None
        self.nodeFeatures = None
        self.nodeOperators = None

        self.nodeAct_z=[];
        self.nodeAct_a=[];
        
        # the inputs to the parent 
        self.ParIn_z=[]; # empty for leaf nodes
        self.ParIn_a=[];

        self.nodeOp_Z=[];
        self.nodeOp_A=[];
        
        self.nodeContextL = [];
        self.nodeContextR = [];
        self.nodeContextNumL = [];
        self.nodeContextNumR = [];
        
        # the parent pointers do not save which is the left and right child of each node, hence:
        # numNodes x 2 matrix of kids, [0 0] for leaf nodes
        self.kids = [];
        # matrix (maybe sparse) with L x S, L = number of unique labels, S= number of segments
        # ground truth:
        self.nodeLabels=[];
        # categories: computed activations (not softmaxed)
        self.catAct = [];
        self.catOut = [];
        # computed category
        self.nodeCat = [];
        self.pos = {};
        # if we have the ground truth, this vector tells us which leaf labels were correctly classified
        self.nodeCatsRight=0;
        self.isLeafVec = [];
        
        self.score=0;
        
        self.nodePath = [];
        self.poolMatrix = [];
        self.y = [];
        
        self.nodeVecDeltas = [];
        
        self.pooledVecPath = [];
        self.features = [];
        self.NN_vecs = [];
        self.NN_deltas = [];
    
    def numLeafs(self):
        num = (len(self.pp)+1)/2;
        return num
        
    def getTopNode(self):
        node_id = np.where(self.pp==-1)[0]
        return node_id[0]
    
    
    def getKids(self,node):
        #%kids = find(obj.pp==node);
        kids = self.kids[node,:]
        return kids
    
    #TODO: maybe compute leaf-node-ness once and then just check for it
    def isLeaf(self,node): 
        l = np.where(self.pp == node)[0] #bhanu
        if(len(l)>0):
            return False
        else:
            return True
            
        
    
        
