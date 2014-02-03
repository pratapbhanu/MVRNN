'''
Created on Oct 18, 2013

@author: bhanu
'''


import numpy as np
from scipy.optimize import lbfgsb
from iais.data.params import Params
from iais.data.rnn_data import RNNDataCorpus
from iais.network.mvrnn import unroll_theta, costFn
import scipy.io as sio
from iais.util.utils import getRelevantWords, evaluate
import socket
import iais.data.config as config


def set_minibatch(rnnData, rnnData_mini, ibatch, nBatches, trainSentIdx):
    sizePerBatch = len(trainSentIdx)/nBatches
    minibatchIdx = trainSentIdx[ibatch*sizePerBatch: sizePerBatch*(ibatch+1)]
    rnnData.copy_into_minibatch(rnnData_mini, minibatchIdx)
#    [Wv_batch, Wo_batch, allWordInds] = getRelevantWords(rnnData_mini, Wv,Wo,params) 
#    return [rnnData_mini, Wv_batch, Wo_batch, allWordInds] 
    

def train():
    
    np.random.seed(131742)
    #get sentences, trees and labels
    nExamples = -1
    print "loading data.."
    rnnData = RNNDataCorpus()
    rnnData.load_data(load_file=config.train_data, nExamples=nExamples)  
    
    #initialize params
    print "initializing params"
    params = Params(data=rnnData, wordSize=50, rankWo=2)

    #define theta
    #one vector for all the parameters of mvrnn model:  W, Wm, Wlabel, L, Lm
    n = params.wordSize; fanIn = params.fanIn; nWords = params.nWords; nLabels = params.categories; rank=params.rankWo
    Wo = 0.01*np.random.randn(n + 2*n*rank, nWords) #Lm, as in paper
    Wo[:n,:] = np.ones((n,Wo.shape[1])) #Lm, as in paper
    Wcat = 0.005*np.random.randn(nLabels, fanIn) #Wlabel, as in paper
#    Wv = 0.01*np.random.randn(n, nWords)
#    WO = 0.01*np.random.randn(n, 2*n)
#    W = 0.01*np.random.randn(n, 2*n+1)
    
    
    #load pre-trained weights here
    mats = sio.loadmat(config.pre_trained_weights)
    Wv = mats.get('Wv')  #L, as in paper
    W = mats.get('W') #W, as in paper
    WO = mats.get('WO') #Wm, as in paper
    
    
    sentencesIdx = np.arange(rnnData.ndoc())
    np.random.shuffle(sentencesIdx)
    nTrain = 4*len(sentencesIdx)/5
    trainSentIdx = sentencesIdx[0:nTrain]
    testSentIdx = sentencesIdx[nTrain:]
    batchSize = 5 
    nBatches = len(trainSentIdx)/batchSize
    evalFreq = 5  #evaluate after every 5 minibatches
    nTestSentEval = 50 #number of test sentences to be evaluated
    
   
    rnnData_train = RNNDataCorpus()
    rnnData.copy_into_minibatch(rnnData_train, trainSentIdx)
    
    rnnData_test = RNNDataCorpus()
    if(len(testSentIdx) > nTestSentEval):
#        np.random.shuffle(testSentIdx)  #choose random test examples
        thisTestSentIdx = testSentIdx[:nTestSentEval]
    else:
        thisTestSentIdx = testSentIdx
    rnnData.copy_into_minibatch(rnnData_test, thisTestSentIdx)
    
    
#    [Wv_test, Wo_test, _] = getRelevantWords(rnnData_test, Wv,Wo,params) 
    [Wv_trainTest, Wo_trainTest, all_train_idx] = getRelevantWords(rnnData, Wv,Wo,params) #sets nWords_reduced, returns new arrays    
    theta = np.concatenate((W.flatten(), WO.flatten(), Wcat.flatten(), Wv_trainTest.flatten(), Wo_trainTest.flatten()))
    
    #optimize    
    print "starting training..."
    nIter = 100
    rnnData_minibatch = RNNDataCorpus()
    for i in range(nIter):        
        #train in minibatches
#        ftrain = np.zeros(nBatches)
#        for ibatch in range(nBatches):            
#            set_minibatch(rnnData, rnnData_minibatch, ibatch, nBatches, trainSentIdx)
            
#            print 'Iteration: ', i, ' minibatch: ', ibatch
        tunedTheta, fbatch_train, _ = lbfgsb.fmin_l_bfgs_b(func=costFn, x0=theta, fprime=None, args=(rnnData_train, params), approx_grad=0, bounds=None, m=5,
                                        factr=1000000000000000.0, pgtol=1.0000000000000001e-5, epsilon=1e-08,
                                        iprint=3, maxfun=1, disp=0)
          
        #map parameters back
        W[:,:], WO[:,:], Wcat[:,:], Wv_trainTest, Wo_trainTest = unroll_theta(tunedTheta, params)
        Wv[:,all_train_idx] = Wv_trainTest
        Wo[:,all_train_idx] = Wo_trainTest
        
#        ftrain[ibatch] = fbatch_train  
        theta = tunedTheta  #for next iteration         
        
        print "========================================"
        print "XXXXXXIteration ", i, 
        print "Average cost: ", np.average(fbatch_train)
        evaluate(Wv,Wo,W,WO,Wcat,params, rnnData_test)
        print "========================================"                  
  
        #save weights
        save_dict = {'Wv':Wv, 'Wo':Wo, 'Wcat':Wcat, 'W':W, 'WO':WO}
        sio.savemat(config.saved_params_file+'_lbfgs_iter'+str(i), mdict=save_dict)
        print "saved tuned theta. "


if __name__ == '__main__':
    train()
