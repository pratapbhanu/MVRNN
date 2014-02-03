'''
Created on Oct 21, 2013

@author: bhanu
'''
import numpy as np
from iais.util.tree import Tree, findPath, getInternalFeatures

class Params(object):
    '''
    classdocs
    '''


    def __init__(self, data, wordSize=50, rankWo=2, externalFeatures=False):
        '''
        @param data: object of RNNDataCorpus
        @param wordSize: dimension of the word vectors
        @param rankWo: rank for low matrix approximation of word matrices
        '''
        self.regC = 7e-3;
        self.regC_Wcat = 5e-4;
        self.regC_WcatFeat = 5e-7;
        self.regC_Wv = 1e-5;
        self.regC_Wo = 1e-6;
        self.regC_Tran = 1e-8;
        
        self.wordSize = wordSize;
        self.rankWo = rankWo;
#        self.NN = 3;  #bhanu
        self.NN = 0
        
        self.numInContextWords = 3; # number inner context words
        self.numOutContextWords = 3; # number outter context words
        
        self.categories = len(data.categories)
        self.nWords = len(data.words)
        
        self.hyp = 1; self.NER = 1; self.POS = 1  
        if(not externalFeatures):
            self.hyp = 0; self.NER = 0; self.POS = 0  
        
        self._setInternalFeatureStats(data)    
        self.fanIn = self._getFanIn()
        
        self.nWords_reduced = None
            
    def _getFanIn(self):   
        #for top node
        l = 1;    
        # for the elements
        l = l + 2;    
        # for the averaged outer words
        l = l + 2;    
        # for all the inner context words
        l = l + self.numInContextWords*2;    
        # for averaged nearest neighbors
#        l = l + 2;     #bhanu
        numFeats = len(self.features_std)    
        fanIn = (l*self.wordSize) + numFeats + self.hyp + self.NER + self.POS;    
        return fanIn
    
    def _setInternalFeatureStats(self, data):
        # features [path length, path depth1, path depth2, sentence length, length between elements]
        numFeatures = 5;
        
        self.features_mean = 0;
        self.features_std = np.ones(numFeatures);
        
        totaltrees = 0
        for s in range(len(data.allSStr)):
            nverbs = len(data.verbIndices[s].flatten())
            sentLength = len(data.allSStr[s].flatten())
            totaltrees += nverbs*sentLength
        
        allFeatures = np.zeros((totaltrees,numFeatures));
        
        thisTree = Tree()
        for s in range(len(data.allSStr)):
            thisTree.pp = data.allSTree[s].flatten()
            verbIndexesList = data.verbIndices[s].flatten()
            for  vid in verbIndexesList:
                for wid in range(len(data.allSStr[s])):    
                    indices = [vid, wid]        
                    fullPath = findPath(thisTree, indices)
                    allFeatures[s,:] = getInternalFeatures(fullPath, data.allSNum[s], indices, self.features_mean, self.features_std);
        
        # Find the mean
        self.features_mean = np.mean(allFeatures,axis=0); # check this
        allFeatures = allFeatures - self.features_mean
        # Find the standard deviation
        self.features_std  = np.std(allFeatures, axis=0)
    
    def setNumReducedWords(self, nWords_reduced):
        self.nWords_reduced = nWords_reduced
    
    def resetNumReducedWords(self):
        self.nWords_reduced = None
        
        