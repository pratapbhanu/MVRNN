'''
Created on Oct 17, 2013

@author: bhanu
'''
from iais.data.rnn_data import RNNDataCorpus
import iais.data.config as config
import scipy.io as sio
from iais.data.params import Params
from iais.util.utils import create_preds_file
import sys
from iais.network.mvrnn import MVRNN
import cPickle

def test():
    
    # load testing data
    print "loading test data.."
    rnnData = RNNDataCorpus()
    rnnData.load_data_srl(load_file=config.test_data_srl, nExamples=10) 
#    rnnData.load_data(load_file=config.test_data, nExamples=-1) 
    
    print 'loading trained model : ', sys.argv[1]
#    mats = sio.loadmat(config.model_path+str(sys.argv[1]))
#    Wv = mats.get('Wv')  #L, as in paper
#    W = mats.get('W') #W, as in paper
#    WO = mats.get('WO') #Wm, as in paper
#    Wo = mats.get('Wo')
#    Wcat = mats.get('Wcat')    
#    n = Wv.shape[0]
#    r = (Wo.shape[0] - n)/(2*n)    
#    rnn = MVRNN(W, WO, Wcat, Wv, Wo)
    with open(config.model_path+sys.argv[1], 'r') as loadfile:
        rnn = cPickle.load(loadfile)
    
    n = rnn.Wv.shape[0]
    r = (rnn.Wo.shape[0] - n)/(2*n)    
    
    print "initializing params.."
    params = Params(data=rnnData, wordSize=n, rankWo=r)
    
    print "evaluating.."
    predictLabels = rnn.evaluate(params, rnnData)

    print "creating labels file .."
    create_preds_file(predictLabels, rnnData.categories, rnnData.sentenceLabels,
                          predictions_file=config.results_path+'preds_srl.txt', 
                          testKeys_file='srl_test_keys.txt')


if __name__ == '__main__':
    test()