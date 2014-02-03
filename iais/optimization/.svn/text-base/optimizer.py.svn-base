'''
Created on 19.03.2013

@author: gpaass
'''
# -*- coding: utf8 -*-

import numpy as np
import time
import sys
import inspect
from iais.data.rnn_data import RNNDataCorpus
from iais.network.mvrnn import unroll_theta
import scipy.io as sio
import cPickle


class StochasticGradientDescent(object):
    def __init__(self, niter=0 , learning_rate=0.001, learningrateFactor=1.0, printAt10Iter='.', printAt100Iter='\n+'):
        """ init stochastic gradient minimizer 
            Parameters
            ----------
            niter : number of iterations
            learning_rate : initial learning rate
            learningrateFactor : multiply learning rate after each iteration with this factor
        
        """
        self._niter = niter
        self._learning_rate = learning_rate
        self._learningrateFactor = learningrateFactor
        self._printAt10Iter = printAt10Iter
        self._printAt100Iter = printAt100Iter
        self._trainTime = 0.0
        self._funValuesTrain = []
        self.lastIDoc = 0


    def getTime(self):
        """ return total execution time """
        return self._trainTime

    def getMinimizerName(self):
        """ return string with name of Minimizer """
        return 'Stochastic Gradient Descent'

    def resetTime(self):
        """ reset total execution time to 0"""
        self._trainTime = 0.0

    def getFvalues(self):
        """ return computed function values for each optimization """
        return self._funValuesTrain


    def minimize(self, rnnData_mini=None, params=None, x0=None, func=None, fprime=None, niter= -1):
        """ minimize by Stochastic Gradient Descent 
            Parameters
            ----------
            func : callable f(x,*args)
                Function to minimise.
            x0 : ndarray
                Initial guess.
            fprime : callable fprime(x,*args)
                The gradient of `func`.  If None, then `func` returns the function
                value and the gradient (``f, g = func(x, *args)``), unless
                `approx_grad` is True in which case `func` returns only ``f``.
            args : sequence
                Arguments to pass to `func` and `fprime`.
    
            Returns
            -------
            x : array_like
                Estimated position of the minimum.
            f : float
                Value of `func` at the minimum.
            """

        """
        Stochastic Gradient Descent with a batch size
        """
        # print 'training MLP with StocGradDescent with learning rate=' + str(learning_rate) + ' and batch size=' + str(batch_size)

        start = int(time.time())

        if niter <= 0:
            niter = self._niter
        for itr in range(niter):
            funVal, d_loss_wrt_params = func(x0, rnnData_mini, params)
            x0 -= self._learning_rate * d_loss_wrt_params
            if (itr + 1) % 100 == 0 and self._printAt100Iter != '':
                sys.stdout.write(self._printAt100Iter)
            elif (itr + 1) % 10 == 0 and self._printAt10Iter != '':
                sys.stdout.write(self._printAt10Iter)
            if self._learningrateFactor != 1.0:
                self._learning_rate = self._learningrateFactor * self._learning_rate

#        funVal = func(x0, rnnData_mini)  # used just for printing loglikelihood
        self._funValuesTrain.append(funVal)

        end = int(time.time())
        timeUsed = end - start
        self._trainTime += timeUsed
        return (x0, funVal, None)


    def minimizeBatches(self, rnn, rnnData_train, allTrainSentIdx, params, x0, func, fprime=None, rnnData_test=None, initialSetSize=5, niter=100, seed=17,
                              modelFileName='', printStatistics=True, modelSaveIter=1, nIterInPart=1,  
                        nFetch=-1, rnd=None, nodeid=-1):
        """ 
            train in batches and optimize the learning rate. The number of batches remains constant
            after each successful validation the learning rate is multiplied by factor incLearnRate 
            after each unsuccesful validation the previous best result is restored and the learning rate is multiplied by factor decLearningRate <1

        Parameters
        ----------
        trainSet: MlpDataCorpus with training data
        x0 : ndarray
            Initial guess.
        func : callable f(x,*args)
            Function to minimise.
        fprime : callable fprime(x,*args)
            The gradient of `func`.  If None, then `func` returns the function
            value and the gradient (``f, g = func(x, *args)``), unless
            `approx_grad` is True in which case `func` returns only ``f``.
            rnd: random number generator
        initialSetSize: size of initial training and test set
        niter: number of outer iterations. over all training set sizes 
        seed: seed for shuffling training/testset
        modelFileName: name of file of saved model
        printStatistics: print statistics on params and derivs in every connection and observer
        modelSaveIter: save model every n major iterations 
        nIterInParts=5: number of inner iterations. -1: use prespecified number
        """
        # print arguments
        frame = inspect.currentframe()
        args, _, _, values = inspect.getargvalues(frame)
#        print 'minimizeBatches', [(i, values[i]) for i in args]
        ndoc = rnnData_train.ndoc()
        if not isinstance(rnnData_train, RNNDataCorpus):
            raise ValueError('not isinstance(trainSet, RNNDataCorpus)', type(rnnData_train))
        if initialSetSize > ndoc:
            raise ValueError('initialSetSize>trainSet.ndoc()', initialSetSize, ndoc)
        nsubsetTrain = max(1, ndoc / initialSetSize)  # each subset should contain 30 observations
        if(rnd == None):  #bhanu
            rnd = np.random.RandomState(seed)
        subIter = -1
         
        trainSeq = range(ndoc)
        fTrainBest = np.zeros(nsubsetTrain) * float('nan')
        #--- iterations over different training set sizes
        for itr in np.arange(niter):
            rnd.shuffle(trainSeq)  # new random sequence for subset selection from trainingset
            fTrainSum = 0
            idocNext = self.lastIDoc  # next doc to use
            print "starting sgd with idoc ", idocNext, "nsubsets = ", nsubsetTrain, "ndoc = ", ndoc
            for isubSet in range(nsubsetTrain):
                (rnnData_mini, idocNext) = rnnData_train.getSubset(idocNext=idocNext, ipart=isubSet, nparts=nsubsetTrain, sequenceIndex=trainSeq)
                nnobsTrain = rnnData_mini.ndoc()

                subIter += 1
                (x1, _fTrain, d) = self.minimize(rnnData_mini=rnnData_mini, params=params, x0=x0, func=func, fprime=fprime, niter=nIterInPart)
                fTrain = _fTrain / nnobsTrain; fTrainSum += fTrain
                
                #map parameters back
                rnn.setTheta(x1, params, allTrainSentIdx)
                
                print '===== iter = %d fTrain = %f subIter = %d isubset = %d ' % (itr, fTrain, subIter, isubSet)
                x0 = x1  # current optimal parameter
                if np.isnan(fTrainBest[isubSet]) or fTrainBest[isubSet] > fTrain:
                    fTrainBest[isubSet] = fTrain
                if((isubSet+1)%modelSaveIter == 0 and modelFileName != ''):
                    rnn.save(modelFileName+str(isubSet))
                    print 'saved model to ', modelFileName            
                    print 'XXXXX subiter = %d Mean fTrain = %f ' % (itr, fTrainSum / modelSaveIter)
                    fTrainSum = 0
                    if printStatistics:
                        if rnnData_test != None:
#                            if len(rnnData_test.allSNum) > 100:
#                                rnnData_test_mini = RNNDataCorpus()
#                                rnnData_test.copy_into_minibatch(rnnData_test_mini, range(100))
#                            else: 
#                                rnnData_test_mini = rnnData_test
                            rnn.evaluate(params, rnnData_test)  # used just for printing loglikelihood
#            if (itr == niter - 1 or (itr + 1) % modelSaveIter == 0) and modelFileName!='': #bhanu
##                save_dict = {'Wv':rnn.Wv, 'Wo':rnn.Wo, 'Wcat':rnn.Wcat, 'W':rnn.W, 'WO':rnn.WO}
##                sio.savemat(modelFileName+'iter'+str(itr), mdict=save_dict)
#                #save weights
#                with open(modelFileName+'iter'+str(itr), 'wb') as wf:
#                    cPickle.dump(rnn, wf, protocol=-1)
#                print "saved tuned theta. "


    def minimizeBatchesPll(self, rnn, rnnData_train, allTrainSentIdx, params, x0, func, fprime=None, rnnData_test=None, initialSetSize=5, niter=100, seed=17,
                                  modelFileName='', printStatistics=True, modelSaveIter=1, nIterInPart=1,  
                            nFetch=-1, rnd=None, nodeid=-1):
        """ 
            train in batches and optimize the learning rate. The number of batches remains constant
            after each successful validation the learning rate is multiplied by factor incLearnRate 
            after each unsuccesful validation the previous best result is restored and the learning rate is multiplied by factor decLearningRate <1

        Parameters
        ----------
        trainSet: MlpDataCorpus with training data
        x0 : ndarray
            Initial guess.
        func : callable f(x,*args)
            Function to minimise.
        fprime : callable fprime(x,*args)
            The gradient of `func`.  If None, then `func` returns the function
            value and the gradient (``f, g = func(x, *args)``), unless
            `approx_grad` is True in which case `func` returns only ``f``.
            rnd: random number generator
        initialSetSize: size of initial training and test set
        niter: number of outer iterations. over all training set sizes 
        seed: seed for shuffling training/testset
        modelFileName: name of file of saved model
        printStatistics: print statistics on params and derivs in every connection and observer
        modelSaveIter: save model every n major iterations 
        nIterInParts=5: number of inner iterations. -1: use prespecified number
        """
        # print arguments
        frame = inspect.currentframe()
        args, _, _, values = inspect.getargvalues(frame)
#        print 'minimizeBatches', [(i, values[i]) for i in args]
        ndoc = rnnData_train.ndoc()
        if not isinstance(rnnData_train, RNNDataCorpus):
            raise ValueError('not isinstance(trainSet, RNNDataCorpus)', type(rnnData_train))
        if initialSetSize > ndoc:
            raise ValueError('initialSetSize>trainSet.ndoc()', initialSetSize, ndoc)
        nsubsetTrain = max(1, ndoc / initialSetSize)  # each subset should contain 30 observations
        if(rnd == None):  #bhanu
            rnd = np.random.RandomState(seed)
        subIter = -1

        trainSeq = range(ndoc)
        fTrainBest = np.zeros(nsubsetTrain) * float('nan')
        #--- iterations over different training set sizes
        for itr in np.arange(niter):
            rnd.shuffle(trainSeq)  # new random sequence for subset selection from trainingset
            fTrainSum = 0
            idocNext = self.lastIDoc  # next doc to use
            print "Node ",nodeid," starting sgd with idoc ", idocNext, "nsubsets = ", nsubsetTrain, "ndoc = ", ndoc, 'nFetch = ', nFetch
            for isubSet in range(nsubsetTrain):
                if((((isubSet+1) % nFetch) == 0 ) and (nFetch!=-1)):  #bhanu
                    yield #return #return and fetch new parameters from master
                (rnnData_mini, idocNext) = rnnData_train.getSubset(idocNext=idocNext, ipart=isubSet, nparts=nsubsetTrain, sequenceIndex=trainSeq)
                nnobsTrain = rnnData_mini.ndoc()

                subIter += 1
                (x1, _fTrain, d) = self.minimize(rnnData_mini=rnnData_mini, params=params, x0=x0, func=func, fprime=fprime, niter=nIterInPart)
                fTrain = _fTrain / nnobsTrain; fTrainSum += fTrain
                
                #map parameters back
                rnn.setTheta(x1, params, allTrainSentIdx)
                
                print 'Node ',nodeid,'===== iter = %d fTrain = %f subIter = %d isubset = %d ' % (itr, fTrain, subIter, isubSet)
                x0 = x1  # current optimal parameter
                if np.isnan(fTrainBest[isubSet]) or fTrainBest[isubSet] > fTrain:
                    fTrainBest[isubSet] = fTrain
#                if(isubSet%1200 == 0):
#                    mlp.save(modelFileName+str(isubSet))
#                    print 'saved model to ', modelFileName
            
            print 'Node ',nodeid,'XXXXX iter = %d Mean fTrain = %f ' % (itr, fTrainSum / nsubsetTrain)


