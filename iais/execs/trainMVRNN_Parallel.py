'''
Created on Nov 2, 2013

@author: bhanu
'''
import playdoh
from time import gmtime, strftime
import numpy as np
import multiprocessing as mp
import time
from iais.network.mvrnn import costFn,  MVRNN
from iais.data.rnn_data import RNNDataCorpus
import scipy.io as sio
from iais.util.utils import getRelevantWords
import cPickle
from iais.data.params import Params
import iais.data.config as config 
from iais.optimization.optimizer import StochasticGradientDescent
import sys

def trainParallelSGD(nuse):
    """ build the cluster. and then start the optimization""" 
    print "time to build the cluster"
    #machines = ['127.0.0.1']
    all_machines = [  '10.1.20.16',  
                         
                       '10.1.20.7', '10.1.20.3',
                      '10.1.20.13', 
                     '10.1.20.4',  '10.1.20.6','10.1.20.12','10.1.20.20','10.1.20.17',
                     '10.1.20.5', '10.1.20.8', '10.1.20.9'
#                     , '127.0.0.1'
                     ]
    machines = all_machines[-3:]
    print machines
#        command = "/home/bhanu/workspace/structMlpPlane_original/open_playdoh" #+ " ".join(machines)
#        subprocess.call([command] + machines)  
    nNodesPerMachine = 3 #this should be same as the number of cpus in open_playdoh script
    totalNodes = nNodesPerMachine * len(machines) #n cpus/nodes per machine + one master cpu/node
         
    topology = []               # Define a star topology, for master-worker model
    for i in xrange(1, totalNodes): # 0 is master
        topology.append(('toMaster_' + str(i), i, 0))
        topology.append(('fromMaster_' + str(i), 0, i))
    
    print "Build a cluster with nMachines=" + str(len(machines)) + " nNodesPerMachine="+str(nNodesPerMachine)
    # Start the task
    
    task = playdoh.start_task(SGDParallel, # name of the task class
                  cpu=totalNodes, # total number of computing units to use on several machines
                  topology=topology, #topology of computing units to communicate with each other
                  machines=machines, #list of machines to be used
                  args=(nuse, totalNodes)) # arguments of the ``initialize`` method
    print "parallel training task launched on cluster.. "
    print "see console of master node: ", machines[0]



def test_mvrnn_sgd_pll():
    """     
    --- to start in a terminal: ---
    open a terminal
    cd to workspace/MVRNN/
    export PYTHONPATH=`pwd`
    python iais/execs/trainMVRNN_Parallel.py
    
    profiling:
    python -m cProfile -o profile.txt -s 'cumtime' tests/structure/testMlpTurianWikipedia.py
    python -m cProfile tests/structure/testMlpTurianWikipedia.py
    """
   
    nuse=-1 #depending on this, trainTest_$nuse.pkl and mlp_$nuse.pkl files are loaded from cluster_model_path and cluster_data_path dirs
    trainParallelSGD(nuse)    
    


class SGDParallel(playdoh.ParallelTask):
    '''  ''' 
    
    def initialize(self, 
                   nuse,
                   totalNodes
                   ):
        """ This method is called on each node when playdoh.start_task method is called.  
            Each node initializes its own model.
        """
        
        self.seed = 1742
        self.nuse = nuse
        self.totalNodes = totalNodes #including master node
        self.debug = True
        self.nIter = 1000
        self.nFetch = 15 #after how many steps(minibatch processing) should a worker fetch new parameters from master
        self.rnd = np.random.RandomState(self.seed)
           
            
    def start(self):
        if self.index == 0:
            self.startMasterNode()
        else:
            self.startWorkerNode()
            
    def startWorkerNode(self):
        '''Receives updated parameters from the master, keeps updating local parameters using its own subset of training data, pushes local parameters to 
            master, all asynchronously  '''        
        
        
        self.initMVRNN()  #also initializes training and test sets
        self.func = costFn
        self.fprime = None        
        self.optimizer = StochasticGradientDescent(niter=self.nIter , learning_rate=0.01, learningrateFactor=1.0, printAt10Iter='.', printAt100Iter='\n+')
        genObj = None   
        while True:
                #fetch action and parameters from masters
                action, master_rnn = self.pop('fromMaster_' + str(self.index))
                if(action == "finish"):
                    print "Node: ",self.index," Received finish action from master, exiting..."
                    return
                if self.debug:
                    print "Node: ", self.index, " Fetched new action & parameters from Master.. " + str(master_rnn.W[0,0])
                
                self.params.resetNumReducedWords() #to unroll all the words
#                W, WO, Wcat, Wv, Wo = unroll_theta(theta, self.params)
                self.copy_into_rnn(master_rnn.getParamsList())  #copy to local rnn to global                
                theta = self.rnn.getTheta(self.params, self.all_train_idx)
                self.params.setNumReducedWords(len(self.all_train_idx))  #set number of reduced words, to be used by costfn for unrolling
                start_time = time.clock()                
                
                try:
                    genObj.next()
                    
                except (AttributeError, StopIteration) :
                    
                    genObj =  self.optimizer.minimizeBatchesPll(rnn=self.rnn, rnnData_train=self.rnnData_train, allTrainSentIdx=self.all_train_idx, 
                                                             params=self.params, x0=theta, func=costFn, fprime=None, 
                                                             rnnData_test=self.rnnData_dev, initialSetSize=1, niter=1, seed=self.seed,
                                                             modelFileName='', printStatistics=False, modelSaveIter=10, nIterInPart=1,  
                                                             nFetch=self.nFetch, rnd=self.rnd, nodeid=self.index)  #optimize this theta and save it in self.rnn 
                except:
                    raise
                end_time = time.clock()                
                
#                [W, WO, Wcat, Wv, Wo] = self.rnn 
#                theta = np.concatenate((W.flatten(), WO.flatten(), Wcat.flatten(), Wv.flatten(), Wo.flatten()))  # current local optimal theta

                #push local theta to master                
                self.push('toMaster_' + str(self.index), (self.rnn, None)) #push the theta value #pushing a tuple object #playdoh bug
                if self.debug:
                    print "Node:", self.index, " Execution time for ", self.nFetch, " minibatches: ", (end_time - start_time)/60, 'minutes'
                    print "Node:", self.index, " Pushed local parameters to Master.."
 

    def async_eval(self, itr, testSet, result_file=None):
        if testSet != None:
#            if len(testSet.allSNum) > 100:
#                rnnData_test_mini = RNNDataCorpus()
#                testSet.copy_into_minibatch(rnnData_test_mini, range(100))
#            else: 
            rnnData_test_mini = testSet
            self.rnn.evaluate( self.params, rnnData_test_mini)             


    def async_save(self, itr):
        timeStr = strftime("%Y-%m-%d_%H-%M-%S", gmtime()) # for generating names
        modelFileName = config.saved_params_file+"_"+str(itr)+'_'+timeStr
        #save weights
        with open(modelFileName, 'wb') as wf:
            cPickle.dump(self.rnn, wf, protocol=-1)
#        [W, WO, Wcat, Wv, Wo] = self.rnn 
#        save_dict = {'Wv':Wv, 'Wo':Wo, 'Wcat':Wcat, 'W':W, 'WO':WO}
#        sio.savemat(modelFileName, mdict=save_dict)
        print 'Master## saved trained model to ', modelFileName

    def startMasterNode(self):
        '''Parameter server, receives gradients from workers, updates parameters of model, sends updated parameters to workers ''' 
        
        nDocsTest = 150
        printStatistics = True
        action = ["continue", "finish"]
        evalFreq = 10
        saveFreq = 5
        
        self.initMVRNN()  #also initializes training and test sets     
        nparts = self.rnnData_dev.ndoc()/nDocsTest if self.rnnData_dev.ndoc() > nDocsTest else 1
        testSeq = range(self.rnnData_dev.ndoc())
        
        
        #start optimization by pushing intial params
        for iNode in xrange(1, self.totalNodes):
                if self.debug:
                    print "Master## Pushing parameters to " + str(iNode) + " theta[0]= " + str(self.rnn.W[0,0])
                self.push('fromMaster_' + str(iNode),(action[0],self.rnn)) # send updated params to workers
                
        #start iterations and evaluate after completion of set of iterations
        for itr in range(self.nIter):
#            workerW = np.zeros(self.rnn.W.shape); workerWO = np.zeros(self.rnn.WO.shape); workerWcat = np.zeros(self.rnn.Wcat.shape); 
#            workerWv = np.zeros(self.rnn.Wv.shape); workerWo = np.zeros(self.rnn.Wo.shape)
            workerParams = []
            for p in self.rnn.getParamsList():
                workerParams.append(np.zeros(p.shape))
            
            self.rnd.shuffle(testSeq)
            (testSetPart, _) = self.rnnData_dev.getSubset(idocNext=0, ipart=0, nparts=nparts, sequenceIndex=testSeq)
            
            if self.debug:
                    print "Master## Iteration: ",itr ,"Waiting to receive params from workers..."          
            for iNode in xrange(1, self.totalNodes):  #get gradients from all workers                          
                receivedRNN, _ = self.pop('toMaster_' + str(iNode)) # poll the workers for result
#                print str(receivedRNN)
                receivedParams = receivedRNN.getParamsList()
                for workerParam, receivedParam in zip(workerParams, receivedParams):
                    workerParam += receivedParam
                
#                workerW = workerW + rW
#                workerWO = workerWO + rWO
#                workerWcat = workerWcat + rWcat
#                workerWv = workerWv + rWv
#                workerWo = workerWo + rWo
                if self.debug:
                    print "Master## Iteration: ",itr ,"Received theta from: " + str(iNode) + " theta[0]="+str(receivedParams[0][0,0])
            #average all the received parameters
            for i in range(len(workerParams)):
                workerParams[i] /= (self.totalNodes-1)
#            workerW = workerW /(self.totalNodes - 1)
#            workerWO = workerWO /(self.totalNodes - 1)
#            workerWcat = workerWcat /(self.totalNodes - 1)
#            workerWv = workerWv /(self.totalNodes - 1)
#            workerWo = workerWo /(self.totalNodes - 1) #one is master rest are workers
            
            #copy into master's rnn
#            workerRnn = [workerW, workerWO, workerWcat, workerWv, workerWo] 
            self.copy_into_rnn(workerParams)
            
            if(self.debug):
                print "Master## Average Params : ", str(workerParams[0][0,0])
            for iNode in xrange(1, self.totalNodes):                
                self.push('fromMaster_' + str(iNode),(action[0],self.rnn)) # send updated params to workers 
                if self.debug:
                    print "Master## Sending new parameters to Worker ", iNode,    "Param[0]:  ", str(self.rnn.W[0,0])       
            if printStatistics and ((itr % evalFreq) == 0) :
                p = mp.Process(target=self.async_eval, args=(itr,self.rnnData_dev))
                p.start()
            if((itr % saveFreq == 0)): # save mlp after every 5 iterations of nFetch cycles.           
                p = mp.Process(target=self.async_save, args=(itr,))
                p.start()
                   
        #finish optimization and close workers
        for iNode in xrange(1, self.totalNodes):
                self.push('fromMaster_' + str(iNode),(action[1],None)) # send updated params to workers
    
    def get_result(self):
        if self.index == 0:
            return self.rnn 
    
    def copy_into_rnn(self, workerRnnParams):
        for dst, src in zip(self.rnn.getParamsList(), workerRnnParams):
            dst[:,:] = src
        
                
    def initMVRNN(self):
        print "Node: ",self.index," Loading training and dev sets.."         
        self.rnnData_train = RNNDataCorpus()
        self.rnnData_train.load_data_srl(config.train_data_srl, nExamples=self.nuse)
        self.rnnData_dev = RNNDataCorpus()
        self.rnnData_dev.load_data_srl(config.dev_data_srl, nExamples=self.nuse)
        modelfilename = config.saved_params_file+'SGD_SRLiter305'
        print "Node: ", self.index," Loading model: ", modelfilename
#        mats =   sio.loadmat(config.saved_params_file+'iter120.mat')
#        Wv = mats.get('Wv')  #L, as in paper
#        W = mats.get('W') #W, as in paper
#        WO = mats.get('WO') #Wm, as in paper
#        Wo = mats.get('Wo')
#        Wcat = mats.get('Wcat')
#        n = Wv.shape[0]
#        r = (Wo.shape[0] - n)/(2*n)
        with open(modelfilename, 'r') as loadfile:
            self.rnn = cPickle.load(loadfile)#MVRNN(W, WO, Wcat, Wv, Wo)
        n = self.rnn.Wv.shape[0]
        r = (self.rnn.Wo.shape[0] - n)/(2*n)
        print "Node: ",self.index, "initializing params.."
        self.params = Params(data=self.rnnData_train, wordSize=n, rankWo=r)
        
#        #to be removed
#        Wcat = 0.005*np.random.randn(self.params.categories, self.params.fanIn)
#        self.rnn.Wcat = Wcat
        
        
        if(self.index == 0):
            print "Master## Total trees in training set: ", self.rnnData_train.ndoc()
            print "Master## nFetch: ", self.nFetch
        
        if(self.index!=0):        
            self.rnnData_train = self.get_subset(self.rnnData_train, self.index)
            self.rnnData_dev = None # workers don't need test set and trainTest, so to free memory release them        
            [_, _, self.all_train_idx] = getRelevantWords(self.rnnData_train, self.rnn.Wv,self.rnn.Wo,self.params) #sets nWords_reduced, returns new arrays  
            #set this to none, as unrolling of theta will take all words into account.  
#        self.theta = np.concatenate((W.flatten(), WO.flatten(), Wcat.flatten(), Wv_trainTest.flatten(), Wo_trainTest.flatten()))
        
        

    def get_subset(self, rnnData, workerIndex):
        ''' data : complete train or test set for which a subset of docs is to be calculated for this worker  '''
        rnnData_mini = RNNDataCorpus()
        dataSize = rnnData.ndoc()
        sizePerNode = dataSize/(self.totalNodes-1)      #exclude master node              
        startPos = (workerIndex-1) * sizePerNode
        if(workerIndex != self.totalNodes-1):     #last node on a machine gets all the remaining data
            endPos = startPos+sizePerNode
        else:
            endPos = dataSize  
        rnnData.copy_into_minibatch(rnnData_mini, range(startPos, endPos))        
        return rnnData_mini



if __name__ == '__main__':
    test_mvrnn_sgd_pll()