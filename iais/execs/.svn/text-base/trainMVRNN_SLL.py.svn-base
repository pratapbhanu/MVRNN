'''
Created on Nov 5, 2013

@author: bhanu
'''
import numpy as np
from iais.data.params import Params
from iais.data.rnn_data import RNNDataCorpus
from iais.network.mvrnn_sll import  costFn, init_transitions, MVRNNSLL
from iais.util.utils import getRelevantWords
from iais.optimization.optimizer import StochasticGradientDescent
import iais.data.config as config


    
def train():
    SEED = 13742
    load_model = False
    custom_load = False
    np.random.seed(SEED)
    #get sentences, trees and labels
    nExamples = 5
    print "loading data.."
    rnnData_train = RNNDataCorpus()
    rnnData_train.load_data_srl(load_file=config.train_data_srl, nExamples=nExamples)  
    rnnData_dev = RNNDataCorpus()
    rnnData_dev.load_data_srl(load_file=config.dev_data_srl, nExamples=nExamples)
    print "Number of sentences loaded in training data: ", rnnData_train.ndoc()
    #initialize params
    print "initializing params"
    params = Params(data=rnnData_train, wordSize=52, rankWo=2)
    n = params.wordSize; fanIn = params.fanIn; nWords = params.nWords; nLabels = params.categories; rank=params.rankWo
    if(load_model):
        modelfile = config.saved_params_file+'SGD_SLL300'
        rnn = MVRNNSLL.load(modelfile)
        print 'loaded model : ', modelfile
    elif(custom_load):        
        modelfile = config.saved_params_file+'SGD_SLL300'
        print "loading customized model..", modelfile
#        d = 2#extra features for wordvectors 
#        Wo = 0.01*np.random.randn(n + 2*n*rank, nWords) #Lm, as in paper
#        Wo[:n,:] = np.ones((n,Wo.shape[1])) #Lm, as in paper
#        Wcat = 0.005*np.random.randn(nLabels, fanIn) #Wlabel, as in paper
#        Wv = 0.01*np.random.randn(n, nWords)
#        WO = 0.01*np.random.randn(n, 2*n)
#        W = 0.01*np.random.randn(n, 2*n+1)
        #load pre-trained weights here
        oldrnn = MVRNNSLL.load(modelfile)
#        Wv[:-d,:] = oldrnn.Wv
        categories = [x.strip() for x in rnnData_train.categories]
        Tran = init_transitions(dict(zip(categories, range(len(categories)))), 'iob')       
        rnn = MVRNNSLL(oldrnn.W, oldrnn.WO, oldrnn.Wcat, oldrnn.Wv, oldrnn.Wo, Tran)
    else:
        #define theta
        #one vector for all the parameters of mvrnn model:  W, Wm, Wlabel, L, Lm
#        n = params.wordSize; fanIn = params.fanIn; nWords = params.nWords; nLabels = params.categories; rank=params.rankWo
#        Wo = 0.01*np.random.randn(n + 2*n*rank, nWords) #Lm, as in paper
#        Wo[:n,:] = np.ones((n,Wo.shape[1])) #Lm, as in paper
#        Wcat = 0.005*np.random.randn(nLabels, fanIn) #Wlabel, as in paper        
#        #load pre-trained weights here
##        mats = sio.loadmat(config.saved_params_file)
#        oldrnn = MVRNNSLL.load(modelfile)
#        Wv = oldrnn.Wv  #L, as in paper
#        W = oldrnn..get('W') #W, as in paper
#        WO = mats.get('WO') #Wm, as in paper
        Wo = 0.01*np.random.randn(n + 2*n*rank, nWords) #Lm, as in paper
        Wo[:n,:] = np.ones((n,Wo.shape[1])) #Lm, as in paper
        Wcat = 0.005*np.random.randn(nLabels, fanIn) #Wlabel, as in paper
        Wv = 0.01*np.random.randn(n, nWords)
        WO = 0.01*np.random.randn(n, 2*n)
        W = 0.01*np.random.randn(n, 2*n+1)
        categories = [x.strip() for x in rnnData_train.categories]
        Tran = init_transitions(dict(zip(categories, range(len(categories)))), 'iob')
        rnn = MVRNNSLL(W, WO, Wcat, Wv, Wo, Tran)
        
#    rnn = MVRNNSLL(W, WO, Wcat, Wv, Wo, Tran)    
    [_, _, all_train_idx] = getRelevantWords(rnnData_train, rnn.Wv, rnn.Wo, params) #sets nWords_reduced, returns new arrays
    params.setNumReducedWords(len(all_train_idx))    
    theta = rnn.getTheta(params, all_train_idx)
    
    
    #optimize    
    print "starting training using SGD..."
    nIter = 500
    optimizer = StochasticGradientDescent(niter=nIter , learning_rate=0.01, learningrateFactor=1.0, printAt10Iter='.', printAt100Iter='\n+')    
    
    
    optimizer.minimizeBatches(rnn=rnn, rnnData_train=rnnData_train, allTrainSentIdx=all_train_idx, params=params, x0=theta, func=costFn, fprime=None, 
                              rnnData_test=rnnData_dev, initialSetSize=1, niter=nIter, seed=SEED,
                              modelFileName=config.saved_params_file+'SGD_SLL', printStatistics=True, modelSaveIter=100, nIterInPart=1,  
                              nFetch=-1, rnd=None, nodeid=-1)                
  


if __name__ == '__main__':
    train()
