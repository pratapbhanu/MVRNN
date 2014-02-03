'''
Created on Oct 21, 2013

@author: bhanu
'''

import scipy.io as sio
from iais.util.utils import build_dict_from_wordlist
import numpy as np
import iais.data.config as config

class RNNDataCorpus(object):
    '''
    classdocs
    '''


    def __init__(self):
        '''
        Constructor
        '''
        self.allSStr = None
        self.allSNum = None
        self.allSKids = None
        self.allIndicies = None
        self.allSTree = None
        self.allSNN = None
        self.words = None
        self.sentenceLabels = None
        self.word_dict = None
        self.categories = None
        self.verbIndices = None
    
    def load_data(self, load_file, nExamples=-1):
        keys = ['allSStr', 'allSNum', 'allSTree', 'allSNN', 'words', 'allSKids', 'allIndicies', 'sentenceLabels', 'categories']
        data = sio.loadmat(load_file,variable_names=keys)
        
        self.allSStr = data.get('allSStr').flatten()
        self.allSNum = data.get('allSNum').flatten()
        self.allSKids = data.get('allSKids').flatten()
        self.allSTree = data.get('allSTree').flatten()
        self.allIndicies = data.get('allIndicies')
        self.allSNN = data.get('allSNN').flatten()
        self.sentenceLabels = data.get('sentenceLabels').flatten()
        
        if(nExamples != -1):
            self.allSStr = self.allSStr[:nExamples]
            self.allSNum = self.allSNum[:nExamples]
            self.allSKids = self.allSKids[:nExamples]
            self.allSTree = self.allSTree[:nExamples]
            self.allIndicies = self.allIndicies[:nExamples,:]
            self.allSNN = self.allSNN[:nExamples]
            self.sentenceLabels = self.sentenceLabels[:nExamples]
        
        self.words = data.get('words').flatten()
        words = []
        for word in self.words:
            words.append(str(word[0]))       
        self.words = words  
        
        self.categories = data.get('categories').flatten()
        categories = []
        for cat in self.categories: 
            categories.append(str(cat[0])) 
        self.categories = categories     
        
        self.word_dict = build_dict_from_wordlist(self.words)
 
        #flatten 2d arrays
        map(self._flatten_data, [self.allSNum, self.allSTree, self.allSStr])
        
        #convert dtype to int32
        map(self._convertDtype, [self.allIndicies, self.allSTree, self.allSKids, self.allSNum, self.allSNN])
        
        #adjust indexing: index from 0 instead of 1
        map(self._adjustIndexing, [self.allSNum, self.allSTree, self.sentenceLabels, self.allIndicies, self.allSKids, self.allSNN])
        
        
    def _flatten_data(self, dataList):
        for i in range(len(dataList)):
            dataList[i] = dataList[i].flatten()
    
    def _adjustIndexing(self, dataList):
        for i in range(len(dataList)):
            dataList[i] -= 1
            
    def _convertDtype(self, dataList, dtype='int64'):
        for i in range(len(dataList)):
            dataList[i] = np.asanyarray(dataList[i], dtype=dtype)
        
    
    def ndoc(self):
        return len(self.allSNum)
    
    def copy_into_minibatch(self, rnnData_mini, sentIdx):
            rnnData_mini.allSStr = self.allSStr[sentIdx]
            rnnData_mini.allSNum = self.allSNum[sentIdx]
            rnnData_mini.allSKids = self.allSKids[sentIdx]
#            rnnData_mini.allIndicies = self.allIndicies[sentIdx]
            rnnData_mini.allSTree = self.allSTree[sentIdx]
#            rnnData_mini.allSNN = self.allSNN[sentIdx]
            rnnData_mini.words = self.words
            rnnData_mini.sentenceLabels = self.sentenceLabels[sentIdx]
#            rnnData_mini.word_dict = self.word_dict
            rnnData_mini.categories = self.categories
            rnnData_mini.verbIndices = self.verbIndices[sentIdx]
    
    def getSubset(self, idocNext=0, ipart=0, nparts=5, sequenceIndex=None):
        """ get the iparts-th subset from npart possible subsets 
            
            idocNext: the next document to use
            ipart: get the i-th part
            nparts: total number of equalsized parts
            sequenceIndex: a permutation of the indices of self. elements. Used for random shuffling
        
        """
        rnnData_mini = RNNDataCorpus();
        if ipart == 0 and idocNext != 0: raise ValueError('ipart=0 and idocNext!=0')
        if sequenceIndex == None:
            sequenceIndex = range(self.ndoc())
        if len(sequenceIndex) != self.ndoc():
            raise ValueError('len(sequenceIndex)!=self.ndoc', len(sequenceIndex), self.ndoc())
        if nparts > self.ndoc():
            raise ValueError('nparts> self.ndoc()', nparts, self.ndoc())
        if ipart < 0 or ipart >= nparts:
            raise ValueError('ipart>=nparts', ipart, nparts)
        idocFirst = idocNext
        ndocRest = self.ndoc() - idocFirst
        size = ndocRest / (nparts - ipart)
        if size <= 0: raise ValueError('size<=0')
        idocNext = idocNext + size
        if ipart == nparts - 1 and idocNext != self.ndoc():
            raise ValueError('ipart == nparts-1 and idocNext != self.ndoc()', ipart, nparts, idocNext, self.ndoc())
        iseq = [sequenceIndex[i] for i in range(idocFirst, idocNext)]
        self.copy_into_minibatch(rnnData_mini, iseq)
    
        return (rnnData_mini, idocNext)
    
    def load_data_srl(self, load_file, nExamples=-1):
        keys = ['allSStr', 'allSNum', 'allSTree',  'allSKids', 'sentenceLabels', 'categories', 'verbIndices']
        data = sio.loadmat(load_file,variable_names=keys)
        
        self.allSStr = data.get('allSStr').flatten()
        self.allSNum = data.get('allSNum').flatten()
        self.allSKids = data.get('allSKids').flatten()
        self.allSTree = data.get('allSTree').flatten()
        self.sentenceLabels = data.get('sentenceLabels').flatten()
        self.categories = data.get('categories').flatten()
        self.verbIndices = data.get('verbIndices').flatten()
        
        
        if(nExamples != -1):
            self.allSStr = self.allSStr[:nExamples]
            self.allSNum = self.allSNum[:nExamples]
            self.allSKids = self.allSKids[:nExamples]
            self.allSTree = self.allSTree[:nExamples]
            self.sentenceLabels = self.sentenceLabels[:nExamples]
            self.verbIndices = self.verbIndices[:nExamples]
        
        self.words = sio.loadmat(config.cw_embeddings_file).get('words').flatten()
        words = []
        for word in self.words:
            words.append(str(word).strip())       
        self.words = words  
        
        #flatten 2d arrays
        map(self._flatten_data, [self.allSNum, self.allSTree, self.verbIndices, self.allSStr])
        
        #convert dtype to int32
        map(self._convertDtype, [self.allSTree, self.allSKids, self.allSNum,  self.verbIndices, self.sentenceLabels])
        
        #adjust indexing: index from 0 instead of 1
        map(self._adjustIndexing, [self.allSNum, self.allSTree, self.allSKids])
        
        