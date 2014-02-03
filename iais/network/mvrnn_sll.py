'''
Created on Oct 17, 2013

@author: bhanu
'''
import numpy as np
import re
from sklearn.metrics.metrics import f1_score, precision_score, recall_score
from iais.network.mvrnn import setVerbnWordDistanceFeat, \
    forwardPropTree, backpropAll
import cPickle
import sys
from iais.util.tree import findPath, getInternalFeatures, addOuterContext,\
    addInnerContext


class MVRNNSLL(object):
    def __init__(self, W, WO, Wcat, Wv, Wo, Tran):
        self.W = W
        self.WO = WO
        self.Wcat = Wcat
        self.Wv = Wv
        self.Wo = Wo
        self.Tran = Tran #transition matrix
    
    def setTheta(self, theta, params, allWordIndx):
        W, WO, Wcat, Wv, Wo, Tran = unroll_theta(theta, params)
        self.W[:,:] = W
        self.WO[:,:] = WO
        self.Wcat[:,:] = Wcat
        self.Wv[:,allWordIndx] = Wv
        self.Wo[:,allWordIndx] = Wo
        self.Tran[:,:] = Tran
        
    def getTheta(self, params, allWordIndx):
        Wv_trainTest = self.Wv[:,allWordIndx]
        Wo_trainTest = self.Wo[:,allWordIndx]
        theta = np.concatenate(((self.W.flatten(), self.WO.flatten(), self.Wcat.flatten(), 
                                Wv_trainTest.flatten(), Wo_trainTest.flatten(), self.Tran.flatten())))
        return theta
        
    
    def evaluate(self, params, rnnDataTest):
        predictLabels = []
        trueLabels = []        
        
        allSNum = rnnDataTest.allSNum
        allSTree = rnnDataTest.allSTree
        allSStr = rnnDataTest.allSStr
        verbIndices = rnnDataTest.verbIndices
        sentenceLabels = rnnDataTest.sentenceLabels
    
        ndoc = rnnDataTest.ndoc()
        print "Total number of trees/sentences to be evaluated: ", ndoc
        for s in range(ndoc):              
            if(s % 100 == 0) :
                print "Processing sentences ", s , ' - ', s+100   
            thissentVerbIndices = verbIndices[s]  
            sStr = allSStr[s]; sNum = allSNum[s]; sTree = allSTree[s]
            labels = sentenceLabels[s]
            if((len(sNum) == 1) or (len(thissentVerbIndices)==0) or (labels.shape[1] != len(sStr))):
                continue  #only one word in a sent, no verbs for this sent, tokens and labels mismatch                 
            for nverb, vid in enumerate(thissentVerbIndices):
                scoresMat = np.zeros((len(sStr), self.Wcat.shape[0]))
                for wid in range(len(sStr)):
                    indices = np.array([vid, wid])   
                    setVerbnWordDistanceFeat(self.Wv, sNum, vid, wid, params) 
                    tree = forwardPropTree(self.W, self.WO, self.Wcat, self.Wv, self.Wo, sNum, sTree, sStr, sNN=None, indicies=None,  params=params) 
                    calPredictions(tree, self.Wcat, self.Wv, indices, sStr, params) #updates score, nodepath etc for this verb, word pair
                    scoresMat[wid,:] = tree.score
                pred_answer = viterbi(scoresMat, self.Tran)
                true_answer = labels[nverb,:]
                for i in range(len(pred_answer)):
                    predictLabels.append(pred_answer[i])
                    trueLabels.append(true_answer[i])
                #TODO : calculate predicted labels     
        
        f1 = f1_score(y_true=trueLabels, y_pred=predictLabels, pos_label=None)#, labels=all_labels)
        p = precision_score(y_true=trueLabels, y_pred=predictLabels, pos_label=None)#, labels=all_labels)
        r = recall_score(y_true=trueLabels, y_pred=predictLabels, pos_label=None)#), labels=all_labels)
        print "XXXXXXX F1 = ", f1
        print "XXXXXXX P = ", p
        print "XXXXXXX R = ", r
        print 
        return predictLabels
    
    def getParamsList(self):
        return [self.W, self.WO, self.Wcat, self.Wv, self.Wo, self.Tran]
    
    def save(self, modelFileName):
        with open(modelFileName, 'wb') as wf:
            cPickle.dump(self, wf, protocol=-1)
    @staticmethod
    def load(modelFileName):
        with open(modelFileName, 'rb') as rf:
            rnn = cPickle.load(rf)
        return rnn
                
def unroll_theta(theta, params):
    '''Order of unrolling theta:
        W, Wm, Wlabel, L, Lm
    and in terms of socher code vocabulary:  W, WO, Wcat, Wv, Wo
     '''    
    n = params.wordSize; nLabels=params.categories; rank=params.rankWo;  fanIn=params.fanIn; nCat = params.categories
    if(params.nWords_reduced != None): #reduced vocab is being used
        nWords = params.nWords_reduced
    else:
        nWords = params.nWords
    W = np.array(theta[:n*(2*n+1)]).reshape(n, 2*n+1)
    WO = np.array(theta[n*(2*n+1): n*(2*n+1) + n*2*n ]).reshape((n, 2*n))
    Wcat = np.array(theta[n*(2*n+1) + n*2*n: n*(2*n+1) + n*2*n + nLabels*fanIn]).reshape(nLabels, fanIn)
    Wv = np.array(theta[n*(2*n+1) + n*2*n  + nLabels*fanIn : n*(2*n+1) + n*2*n  + nLabels*fanIn + n*nWords]).reshape(n, nWords)
    Wo = np.array(theta[n*(2*n+1) + n*2*n  + nLabels*fanIn + n*nWords : 
                        n*(2*n+1) + n*2*n  + nLabels*fanIn + n*nWords + (2*n*rank+n)*nWords ]).reshape(2*n*rank+n, nWords)
    Tran = np.array(theta[n*(2*n+1) + n*2*n  + nLabels*fanIn + n*nWords + (2*n*rank+n)*nWords : ]).reshape(nCat+1, nCat)
    return W, WO, Wcat, Wv, Wo, Tran
   

def backpropPool(tree, topDelta, Wcat, params):

    
    numFeatures = len(params.features_std)
    
    catInput = np.concatenate((tree.pooledVecPath.flatten(),tree.features)) #,tree.NN_vecs.flatten()))  #bhanu
    catInput = np.tanh(catInput)
#    topDelta = tree.y - groundTruth; topDelta = topDelta.reshape(len(tree.y),1)
    topDelta = topDelta.transpose()
    df_s_Wcat = topDelta.dot(catInput.reshape(1,len(catInput))) #to check #bhanu
#    df_s_Tran = grad_Tran #To check
    
    nodeVecDeltas = np.zeros(tree.nodeAct_a.shape)
    paddingDelta = np.zeros((tree.nodeAct_a.shape[0],1))
    
    #% Get the deltas for each node to pass down
    nodeDeltas = np.dot(np.dot(tree.poolMatrix,np.transpose(Wcat[:,:-numFeatures])), topDelta) #TO check #bhanu
    s = nodeDeltas.shape[0]
    nodeDeltas = nodeDeltas.reshape(params.wordSize,s/params.wordSize)
    
    #% Pass out the deltas to each node
    for ni in range(len(tree.nodePath)):
        node = tree.nodePath[ni]
        if node == -1:
            paddingDelta = paddingDelta + np.multiply(nodeDeltas[:,ni],(1 - tree.nodeAct_a[:,node]**2)) #added tanh layer
        else:
#            if tree.isLeafVec[node]:
#                nodeVecDeltas[:,node] = nodeVecDeltas[:,node] + nodeDeltas[:,ni]
#            else:
#                nodeVecDeltas[:,node] = nodeVecDeltas[:,node] + np.multiply(nodeDeltas[:,ni],(1 - tree.nodeAct_a[:,node]**2))
            nodeVecDeltas[:,node] = nodeVecDeltas[:,node] + np.multiply(nodeDeltas[:,ni],(1 - tree.nodeAct_a[:,node]**2))
   
    #% Backprop into the nearest neighbors
    NN_deltas = nodeDeltas[:,ni:]
    return [df_s_Wcat, nodeVecDeltas, NN_deltas, paddingDelta]


def viterbi(scores, transitions, allow_repeats=True):
        """
        Performs a Viterbi search over the scores for each tag using
        the transitions matrix.
        """
        
        path_scores = np.empty_like(scores)
        path_backtrack = np.empty_like(scores, np.int)
        
        # now the actual Viterbi algorithm
        # first, get the scores for each tag at token 0
        # the last row of the transitions table has the scores for the first tag
        path_scores[0,:] = scores[0,:] + transitions[-1]
        
        for i in range(1, scores.shape[0]):
            
            # each line contains the score until each tag t plus the transition to each other tag t'
            prev_score_and_trans = (path_scores[i - 1] + transitions[:-1].T).T
            
            # find the previous tag that yielded the max score
            path_backtrack[i,:] = prev_score_and_trans.argmax(0)
            path_scores[i,:] = prev_score_and_trans[path_backtrack[i,:], np.arange(scores.shape[1])] + scores[i,:]
            
        # now find the maximum score for the last token and follow the backtrack
        answer = np.empty(len(scores), dtype=np.int)
        answer[-1] = path_scores[-1,:].argmax()
        answer_score = path_scores[-1,:][answer[-1]]
        previous_tag = path_backtrack[-1,:][answer[-1]]
        
        for i in range(scores.shape[0] - 2, 0, -1):
            answer[i] = previous_tag
            previous_tag = path_backtrack[i][previous_tag]
        
        answer[0] = previous_tag
        return answer

def calculate_all_scores(scores, Tran):
        """
        Calculates a matrix with the scores for all possible paths at all given
        points (tokens).
        In the returning matrix, all_scores[i][j] means the sum of all scores 
        ending in token i with tag j
        """
        # logadd for first token. the transition score of the starting tag must be used.
        # it turns out that logadd = log(exp(score)) = score
        # (use long double because taking exp's leads to very very big numbers)
        scores = np.longdouble(scores)
        scores[0] += Tran[-1]
        
        # logadd for the following tokens
        transitions = Tran[:-1].T
        for token, _ in enumerate(scores[1:], start=1):
            logadd = np.log(np.sum(np.exp(scores[token - 1] + transitions), 1))
            scores[token] += logadd
         
#        scores[np.where(scores==np.inf)]   = 1000 #bhanu
        return scores #np.inf a very large number is replaced with a finite large number

def calPredictions(thisTree, Wcat, Wv, indicies, sStr, params):
    wsz = params.wordSize
    # vecPath - contains the vectors along the node path
    # nodePath - contains the nodes along the node path
    thisTree.nodePath = findPath(thisTree, indicies) 
    
    # Get the internal features
    thisTree.features = getInternalFeatures(thisTree.nodePath, sStr, indicies, params.features_mean, params.features_std)
    
    # Add only the top node
    thisTree.nodePath = np.max(thisTree.nodePath)
    thisTree.pooledVecPath = thisTree.nodeAct_a[:,thisTree.nodePath].reshape(wsz,1)  
    
    # Add the elements
    thisTree.pooledVecPath = np.hstack((thisTree.pooledVecPath, thisTree.nodeAct_a[:,[indicies[0]]], thisTree.nodeAct_a[:,[indicies[1]]]))
    thisTree.nodePath = np.array([thisTree.nodePath, indicies[0], indicies[1]], dtype='int32')
    thisTree.poolMatrix = np.eye(3*wsz)
    
    # add outer context words
    thisTree = addOuterContext(thisTree, indicies, len(sStr), params);
    
    # add inner context words
    thisTree = addInnerContext(thisTree,indicies, Wv, len(sStr), params);
    
    # add in nearest neighbors
#    thisTree = addNN(thisTree, Wv, sNN, params);  #bhanu 
    
    catInput = np.concatenate((thisTree.pooledVecPath.flatten(), thisTree.features))#, thisTree.NN_vecs.flatten())) #bhanu 
    
    #adding tanh layer
    nonLinearCatInput = np.tanh(catInput)
       
    thisTree.score = np.dot(Wcat, np.transpose(nonLinearCatInput)) #score for each label/tag
    
    
    num = np.exp(thisTree.score)
    thisTree.y = np.divide(num, np.sum(num))  #conditional probs for each tag/label



def calculate_sll_grad(scores, all_scores, trueLabels, Tran, Wcat):
    # initialize gradients
    net_gradients = np.zeros_like(scores, np.float)
    trans_gradients = np.zeros_like(Tran, np.float)
    
    # compute the gradients for the last token
    exponentials = np.exp(all_scores[-1])
    exp_sum = np.sum(exponentials)
    net_gradients[-1] = -exponentials / exp_sum
    
    transitions_t = Tran[:-1].T
    
    # now compute the gradients for the other tokens, from last to first
    for token in range(len(scores) - 2, -1, -1):
        
        # matrix with the exponentials which will be used to find the gradients
        # sum the scores for all paths ending with each tag in token "token"
        # with the transitions from this tag to the next
        exp_matrix = np.exp(all_scores[token] + transitions_t).T
        
        # the sums of exps, used to calculate the softmax
        # sum the exponentials by column
        denominators = exp_matrix.sum(0)
#        denominators[np.where(denominators == np.inf)] = 10 #bhanu
        # softmax is the division of an exponential by the sum of all exponentials
        # (yields a probability)
        softmax = exp_matrix / denominators
        
        # multiply each value in the softmax by the gradient at the next tag
        grad_times_softmax = net_gradients[token + 1] * softmax
        trans_gradients[:-1, :]  += grad_times_softmax
        
        # sum all transition gradients by line to find the network gradients
        net_gradients[token] = np.sum(grad_times_softmax, 1)
    
    # find the gradients for the starting transition
    # there is only one possibility to come from, which is the sentence start
    trans_gradients[-1] = net_gradients[0]
    
    # now, add +1 to the correct path
    last_tag = Wcat.shape[0]
    for token, tag in enumerate(trueLabels):
        net_gradients[token][tag] += 1        
        trans_gradients[last_tag][tag] += 1
        last_tag = tag
    
    return net_gradients, trans_gradients


def cost_oneSent(W, WO, Wcat, Wv, Wo, Tran, sNum,sTree, sStr, sNN, labels, verbIndices, params):
    ''' Returns sentence level loglikelihood cost for this sentence'''     
    
    cost = 0.0 #cost of one sentence = sum of cost of classification of each word for each verb    
    Sdf_s_Wv = None; Sdf_s_Wo = None; Sdf_s_W = None; Sdf_s_WO = None; Sdf_s_Wcat = None; Sdf_s_Tran = None
    
    #forward propagation for each verb in this sentence
    for nverb, vid in enumerate(verbIndices): 
        scoresMat = np.zeros((len(sStr), Wcat.shape[0]))  
        input_values = np.zeros((len(sStr), Wcat.shape[1]))
        forwardPropTrees = []  
        for wid in range(len(sStr)):
            indices = np.array([vid, wid])   
            setVerbnWordDistanceFeat(Wv, sNum, vid, wid, params) 
            tree = forwardPropTree(W, WO, Wcat, Wv, Wo, sNum, sTree, sStr, sNN,  params=params) #calculate nodevectors and matrices        
            calPredictions(tree, Wcat, Wv, indices, sStr, params) #updates score, nodepath etc for this verb, word pair
            scoresMat[wid,:] = tree.score
            forwardPropTrees.append(tree)
            input_values[wid,:] = np.tanh(np.concatenate((tree.pooledVecPath.flatten(), tree.features))) #this should be same as catInput

        #calculate sentence-level-loglikelihood cost
        #cost = logadd(score for all possible paths) - score(correct path)
        correct_path_score = 0
        last_label = Wcat.shape[0] #last row is the score of starting from a tag
        for i, this_label in enumerate(labels[nverb,:]):
            correct_path_score += scoresMat[i,this_label] + Tran[last_label, this_label]
            last_label = this_label            
        all_scores = calculate_all_scores(scoresMat, Tran)
        error  = np.log(np.sum(np.exp(all_scores[-1]))) - correct_path_score
        cost += error
    
        #calculate derivative of cost function
        grad_Wcat, df_s_Tran = calculate_sll_grad(scoresMat, all_scores, labels[nverb,:], Tran, Wcat)
        
        #calculate df_s_Wcat and df_s_Tran
#        top_delta = np.tile(grad_Wcat, [Wcat.shape[1], 1, 1]).T
#        df_s_Wcat = np.multiply(top_delta, input_values).sum(1)
#        top_delta = top_delta.T.sum(1)  #for backpropagating down each tree
#        df_s_Wcat = np.zeros(Wcat.shape)
#        for i in range(grad_Wcat.shape[0]):
#            df_s_Wcat += grad_Wcat[[i],:].T.dot(input_values[[i],:])
##        df_s_Wcat /= grad_Wcat.shape[0]
#             
#       
        if(Sdf_s_Tran == None):
            Sdf_s_Tran = np.zeros(df_s_Tran.shape) #Sdf_s_Wcat = np.zeros(df_s_Wcat.shape); 
#        Sdf_s_Wcat += df_s_Wcat
        Sdf_s_Tran += df_s_Tran
        
        #calculate hidden layer gradients
#        numFeatures = len(params.features_std)
        #do backpropagation  for this verb  
        for i, ftree in enumerate(forwardPropTrees):
            #calculate deltas for nodes, delta_m and delta_h
#            xx = np.dot(ftree.poolMatrix,np.transpose(Wcat[:,:-numFeatures]))
#            nodeDeltas = xx.dot(grad_Wcat[i]) 
            [df_s_Wcat, ftree.nodeVecDeltas, ftree.NN_deltas, paddingDelta] = backpropPool(ftree, grad_Wcat[[i],:], Wcat, params)                                                                                            
    
            deltaDown_vec = np.zeros((params.wordSize,1))
            deltaDown_op = np.zeros((params.wordSize,params.wordSize))
            
            topNode = ftree.getTopNode();
            [df_s_Wv, df_s_Wo, df_s_W, df_s_WO] = backpropAll(ftree, W, WO, Wo, params, deltaDown_vec, deltaDown_op, topNode, Wv.shape[1], None, None)
            
            #Backprop into Padding
            df_s_Wv[:,0] = df_s_Wv[:,0] + paddingDelta.flatten()
            
            if(Sdf_s_Wv == None):
                Sdf_s_Wv = np.zeros(df_s_Wv.shape); Sdf_s_Wo = np.zeros(df_s_Wo.shape); Sdf_s_W = np.zeros(df_s_W.shape); 
                Sdf_s_WO = np.zeros(df_s_WO.shape); Sdf_s_Wcat = np.zeros(df_s_Wcat.shape)
            
            Sdf_s_Wv = Sdf_s_Wv + df_s_Wv
            Sdf_s_Wo = Sdf_s_Wo + df_s_Wo
            Sdf_s_W = Sdf_s_W + df_s_W
            Sdf_s_WO = Sdf_s_WO + df_s_WO
            Sdf_s_Wcat = Sdf_s_Wcat + df_s_Wcat
#            Sdf_s_Tran = Sdf_s_Tran + df_s_Tran
        
        #scale cost and derivative by the number of forward trees created for this verb
        #divide by number of sentences 
        numTrees = len(forwardPropTrees)
        Sdf_s_Wcat = (1.0/numTrees) * Sdf_s_Wcat
        Sdf_s_W = (1.0/numTrees) * Sdf_s_W
        Sdf_s_Wv   = (1.0/numTrees) * Sdf_s_Wv
        Sdf_s_WO   = (1.0/numTrees) * Sdf_s_WO
        Sdf_s_Wo   = (1.0/numTrees) * Sdf_s_Wo 
#        Sdf_s_Tran = (1.0/numTrees) * Sdf_s_Tran
    
    #scale w.r.t. number of verbs in this sentence
    numVerbs = verbIndices.shape[0]
    cost = (1.0/numVerbs) * cost
    Sdf_s_Wcat = (1.0/numVerbs) * Sdf_s_Wcat
    Sdf_s_W = (1.0/numVerbs) * Sdf_s_W
    Sdf_s_Wv   = (1.0/numVerbs) * Sdf_s_Wv
    Sdf_s_WO   = (1.0/numVerbs) * Sdf_s_WO
    Sdf_s_Wo   = (1.0/numVerbs) * Sdf_s_Wo    
    Sdf_s_Tran   = (1.0/numVerbs) * Sdf_s_Tran      
    
    return [Sdf_s_Wv, Sdf_s_Wo, Sdf_s_W, Sdf_s_WO, Sdf_s_Wcat, Sdf_s_Tran, cost]
    
def costFn(theta, rnnData, params):
    '''returns avergage sentence level loglikelihood cost and its gradient over the given training set 
    Note: it always uses reduced vocabulary for unrolling theta
    '''
    #    Xs,ys,ts = trainSet
    allSNum = rnnData.allSNum
    allSTree = rnnData.allSTree
    allSStr = rnnData.allSStr
    sentenceLabels = rnnData.sentenceLabels
    verbIndices = rnnData.verbIndices
    
    W, WO, Wcat, Wv, Wo, Tran = unroll_theta(theta, params)
    numSent = len(allSNum);
    numFeats = len(params.features_std);
    wsz = params.wordSize;

    Wv_df = np.zeros(Wv.shape)
    Wo_df = np.zeros(Wo.shape)
    W_df = np.zeros(W.shape)
    WO_df = np.zeros(WO.shape)
    Wcat_df = np.zeros(Wcat.shape) 
    Tran_df = np.zeros(Tran.shape)
    
    totalCost = 0.0  

    for t in range(numSent):
        if((len(allSNum[t]) == 1) or (len(verbIndices[t])==0) or (sentenceLabels[t].shape[1] != len(allSStr[t]))):
            continue  #only one word in a sent, no verbs for this sent, tokens and labels mismatch 
        #calculate gradients and cost for this sentence and predicate  
        try:    
            [df_s_Wv,df_s_Wo,df_s_W,df_s_WO, df_s_Wcat, df_s_Tran, cost] = cost_oneSent(W, WO, Wcat, Wv, Wo, Tran, 
                                                                                        allSNum[t],allSTree[t], allSStr[t],
                                                                            None, sentenceLabels[t], verbIndices[t], params)
        except :
            print "Error occurred", allSStr[t]
            print sys.exc_info()[0]
            raise
        
        totalCost += cost
        Wcat_df = Wcat_df + df_s_Wcat
        W_df = W_df + df_s_W
        WO_df = WO_df + df_s_WO
        Wv_df = Wv_df + df_s_Wv
        Wo_df = Wo_df + df_s_Wo
        Tran_df = Tran_df + df_s_Tran
        
         
    #divide by number of sentences 
    cost = (1.0/numSent)*totalCost
    Wcat_df = (1.0/numSent) * Wcat_df
    W_df = (1.0/numSent) * W_df
    Wv_df   = (1.0/numSent) * Wv_df
    WO_df   = (1.0/numSent) * WO_df
    Wo_df   = (1.0/numSent) * Wo_df
    Tran_df = (1.0/numSent) * Tran_df
    
    #Add regularization to gradient
    Wcat_df = Wcat_df + np.hstack((params.regC_Wcat*Wcat[:,:-numFeats], params.regC_WcatFeat*Wcat[:,-numFeats:]))
    Wv_df = Wv_df+ params.regC_Wv * Wv
    W_df  = W_df + params.regC  * np.hstack((W[:,:-1], np.zeros((W.shape[0],1))))
    Wo_df = Wo_df + params.regC_Wo * np.vstack(((Wo[:wsz,:] - np.ones((wsz,Wo.shape[1]))), Wo[wsz:,:]))
    WO_df = WO_df + params.regC * WO
    Tran_df = Tran_df + params.regC_Tran * Tran 

    # Add regularization to cost    
    cost = cost +  params.regC_Wcat/2.0 * np.sum(np.sum(Wcat[:,-numFeats]**2)) 
    cost = cost +  params.regC_WcatFeat/2.0 * np.sum(np.sum(Wcat[:,-numFeats:]**2))   
    cost = cost +  params.regC_Wv/2.0 * np.sum(np.sum(Wv**2)) 
    cost = cost +  params.regC/2.0 * np.sum(np.sum(W[:,-1]**2)) 
    cost = cost +  params.regC_Wo/2.0 * np.sum(np.sum(np.vstack((Wo[:wsz,:] - np.ones((wsz,Wo.shape[1])), Wo[wsz:,:]**2)))) 
    cost = cost +  params.regC/2.0 * np.sum(WO**2)
    cost = cost +  params.regC_Tran/2.0 * np.sum(Tran**2) #TODO : check if we need separate regC for Tran

    #return cost and gradient
    grad = np.concatenate((W_df.flatten(), WO_df.flatten(), Wcat_df.flatten(), Wv_df.flatten(), Wo_df.flatten(), Tran_df.flatten()))
    return cost, grad

def init_transitions(tag_dict, scheme):
    """
    This function initializes the tag transition table setting 
    very low values for impossible transitions. 
    :param tag_dict: The tag dictionary mapping tag names to the
    network output number.
    :param scheme: either iob or iobes.
    """
    scheme = scheme.lower()
    assert scheme in ('iob', 'iobes'), 'Unknown tagging scheme: %s' % scheme
    transitions = []
    postrans = 0; neutraltrans = -1; negtrans = -1000
    
    # since dict's are unordered, let's take the tags in the correct order
    tags = sorted(tag_dict, key=tag_dict.get)
    
    # transitions between tags
    for tag in tags:
        
        if tag == 'O' or tag == 'OTHER':
            # next tag can be O, V or any B
            trans = lambda x: postrans if re.match('B|S|V', x) \
                                else neutraltrans if (x == 'O' or x=='OTHER') else negtrans
#        elif tag == 'OTHER':
#            pass 
        elif tag[0] in 'IB':
            block = tag[2:]
            if scheme == 'iobes':
                # next tag can be I or E (same block)
                trans = lambda x: postrans if re.match('(I|E)-%s' % block, x) else negtrans
            else:
                # next tag can be O, I (same block) or B (new block)
                trans = lambda x: postrans if re.match('I-%s' % block, x) or re.match('B-(?!%s)' % block, x) \
                                    else neutraltrans if (x == 'O' or x=='OTHER') else negtrans
        
        elif tag[0] in 'ES':
            # next tag can be O, S (new block) or B (new block)
            block = tag[2:]
            trans = lambda x: postrans if re.match('(S|B)-(?!%s)' % block, x) \
                                else neutraltrans if (x == 'O' or x=='OTHER') else negtrans

        else:
            raise ValueError('Unknown tag: %s' % tag)
        
        transitions.append([trans(next_tag) for next_tag in tags])  
    
    # starting tag
    # it can be O or any B/S
    trans = lambda x: postrans if x[0] in 'OBS' else negtrans  #this takes into account 'OTHER' tag too #bhanu
    transitions.append([trans(next_tag) for next_tag in tags])
    
    return np.array(transitions, np.float)

