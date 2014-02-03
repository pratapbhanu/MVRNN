'''
Created on Oct 17, 2013

@author: bhanu
'''
import numpy as np 
from iais.util.tree import Tree, findPath, getInternalFeatures, addOuterContext,\
    addInnerContext
from sklearn.metrics.metrics import f1_score, precision_score, recall_score
import cPickle


class MVRNN(object):
    def __init__(self, W, WO, Wcat, Wv, Wo):
        self.W = W
        self.WO = WO
        self.Wcat = Wcat
        self.Wv = Wv
        self.Wo = Wo
        
    
    def setTheta(self, theta, params, allWordIndx):
        W, WO, Wcat, Wv, Wo = unroll_theta(theta, params)
        self.W[:,:] = W
        self.WO[:,:] = WO
        self.Wcat[:,:] = Wcat
        self.Wv[:,allWordIndx] = Wv
        self.Wo[:,allWordIndx] = Wo
        
    def evaluate(self, params, rnnDataTest):
#        predictLabels = np.zeros(len(rnnDataTest.allSNum), dtype='int32')
#        probabilities = np.zeros(len(rnnDataTest.allSNum))
        predictLabels = []
        trueLabels = []
        
        allSNum = rnnDataTest.allSNum
        allSTree = rnnDataTest.allSTree
        allSStr = rnnDataTest.allSStr
        verbIndices = rnnDataTest.verbIndices
#        allSNN = rnnDataTest.allSNN
#        allIndicies = rnnDataTest.allIndicies
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
                for wid in range(len(sStr)):
                    indices = np.array([vid, wid])   
                    truelabel = labels[nverb, wid]  
                    setVerbnWordDistanceFeat(self.Wv, sNum, vid, wid, params) 
                    tree = forwardPropTree(self.W, self.WO, self.Wcat, self.Wv, self.Wo, sNum, sTree, sStr, sNN=None, indicies=None,  params=params) 
                    trueLabels.append(truelabel)       
                    calPredictions(tree, self.Wcat, self.Wv, indices, sStr, params) #updates score, nodepath etc for this verb, word pair
                    predictedLabel = np.argmax(tree.y)
                    predictLabels.append(predictedLabel)
        
        f1 = f1_score(y_true=trueLabels, y_pred=predictLabels, pos_label=None)#, labels=all_labels)
        p = precision_score(y_true=trueLabels, y_pred=predictLabels, pos_label=None)#, labels=all_labels)
        r = recall_score(y_true=trueLabels, y_pred=predictLabels, pos_label=None)#), labels=all_labels)
        print "XXXXXXX F1 = ", f1
        print "XXXXXXX P = ", p
        print "XXXXXXX R = ", r
        print 
        return predictLabels
    
    def getTheta(self, params, allWordIndx):
        Wv_trainTest = self.Wv[:,allWordIndx]
        Wo_trainTest = self.Wo[:,allWordIndx]
        return np.concatenate(((self.W.flatten(), self.WO.flatten(), self.Wcat.flatten(), 
                                Wv_trainTest.flatten(), Wo_trainTest.flatten())))
    
    def getParamsList(self):
        return [self.W, self.WO, self.Wcat, self.Wv, self.Wo]
    
    def save(self, modelFileName):
        with open(modelFileName, 'wb') as wf:
            cPickle.dump(self, wf, protocol=-1)
    

def unroll_theta(theta, params):
    '''Order of unrolling theta:
        W, Wm, Wlabel, L, Lm
    and in terms of socher code vocabulary:  W, WO, Wcat, Wv, Wo
     '''    
    n = params.wordSize; nLabels=params.categories; rank=params.rankWo;  fanIn=params.fanIn;
    if(params.nWords_reduced != None): #reduced vocab is being used
        nWords = params.nWords_reduced
    else:
        nWords = params.nWords
    W = np.array(theta[:n*(2*n+1)]).reshape(n, 2*n+1)
    WO = np.array(theta[n*(2*n+1): n*(2*n+1) + n*2*n ]).reshape((n, 2*n))
    Wcat = np.array(theta[n*(2*n+1) + n*2*n: n*(2*n+1) + n*2*n + nLabels*fanIn]).reshape(nLabels, fanIn)
    Wv = np.array(theta[n*(2*n+1) + n*2*n  + nLabels*fanIn : n*(2*n+1) + n*2*n  + nLabels*fanIn + n*nWords]).reshape(n, nWords)
    Wo = np.array(theta[n*(2*n+1) + n*2*n  + nLabels*fanIn + n*nWords : ]).reshape(2*n*rank+n, nWords)
    return W, WO, Wcat, Wv, Wo


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
    thisTree.score = np.dot(Wcat, np.transpose(catInput)) #score for each label/tag
    
    num = np.exp(thisTree.score)
    thisTree.y = np.divide(num, np.sum(num))  #conditional probs for each tag/label
    


def forwardPropTree(W, WO, Wcat, Wv, Wo, sNum,sTree, sStr=None, sNN=None, indicies=None, params=None):
    
    wsz = params.wordSize
    r = params.rankWo
    
    words = np.where(sNum>=0)[0]
    numTotalNodes = len(sNum)
    
    allV = Wv[:,sNum[words]]
    allO = Wo[:,sNum[words]]
    
    thisTree = Tree()
    # set tree structure of tree
    thisTree.pp = sTree  #to check
    # set which nodes are leaf nodes
    thisTree.isLeafVec = np.zeros(numTotalNodes);
    thisTree.isLeafVec[words] = 1;
    
    thisTree.nodeNames = np.arange(len(sTree))
    thisTree.nodeLabels = sNum;
    
    # the inputs to the parent
    thisTree.ParIn_z = np.zeros((wsz,numTotalNodes)) # empty for leaf nodes
    thisTree.ParIn_a = np.zeros((wsz,numTotalNodes))
    
    #node vectors
    thisTree.nodeAct_a = np.zeros((wsz, numTotalNodes))
    # the new operators
    thisTree.nodeOp_A = np.zeros((wsz**2,numTotalNodes))
    # the scores for each decision
    thisTree.score = np.zeros(numTotalNodes);
    # the children of each node (for speed)
    thisTree.kids = np.zeros((numTotalNodes,2), dtype='int32');
    
    
    # initialize the vectors and operators of the words (leaf nodes)
    thisTree.nodeAct_a[:,words] = allV;
    
    for thisWordNum in range(len(words)):
        diag_a = np.diag(allO[:wsz,thisWordNum])
        U = allO[wsz:wsz*(1+r),thisWordNum].reshape(wsz,r)
        V = allO[wsz*(1+r):,thisWordNum].reshape(wsz, r)
        A = diag_a + np.dot(U, np.transpose(V))  
        A = A.reshape(wsz**2)
        thisTree.nodeOp_A[:, thisWordNum] = A
    
    toMerge = np.zeros(shape=(words.shape), dtype='int32')
    toMerge[:] = words[:]    
    while len(toMerge)>1 :
        # find unpaired bottom leaf pairs (initially words) that share parent
        i=-1;
        foundGoodPair = False
        while (not foundGoodPair )  :
            i += 1
            if sTree[toMerge[i]]==sTree[toMerge[i+1]]:
                foundGoodPair = True                 
            
        newParent = sTree[toMerge[i]] 
        kid1 = toMerge[i]
        kid2 = toMerge[i+1]
        thisTree.kids[newParent,:] = [kid1, kid2];
        # set new parent to be possible merge candidate
        toMerge[i] = newParent;
        # delete other kid
        toMerge = np.delete(toMerge,i+1)
        
        a = thisTree.nodeAct_a[:,kid1];
        A = thisTree.nodeOp_A[:,kid1].reshape(wsz,wsz)
        b = thisTree.nodeAct_a[:,kid2];
        B = thisTree.nodeOp_A[:,kid2].reshape(wsz,wsz)
        
        l_a = np.dot(B,a)
        r_a = np.dot(A,b)
        C = np.concatenate((l_a,r_a, np.ndarray([1])))
        thisTree.nodeAct_a[:,newParent] = np.tanh(np.dot(W,C))
        
        P_A =  (np.dot(WO,np.vstack((A,B)))).reshape(wsz**2)
        
        # save all this for backprop:
        thisTree.ParIn_a[:,kid1] = l_a
        thisTree.ParIn_a[:,kid2] = r_a
        thisTree.nodeOp_A[:,newParent] = P_A   
    
    return thisTree
    

def backpropPool(tree, label, Wcat, params, classWeight=1.0):

    I = np.eye(len(tree.y))
    groundTruth = I[:,label]

    
    numFeatures = len(params.features_std)
    
    catInput = np.concatenate((tree.pooledVecPath.flatten(),tree.features ))#,tree.NN_vecs.flatten()))  #bhanu
    
    topDelta = tree.y - groundTruth; 
    if(label!=34 or label!=0):  #category labels for 'O' and 'OTHER' tags
        topDelta = topDelta*classWeight
    else:
        topDelta = topDelta*(1-classWeight)
    topDelta = topDelta.reshape(len(tree.y),1);
    df_s_Wcat = topDelta.dot(catInput.reshape(len(catInput),1).transpose())
    
    nodeVecDeltas = np.zeros(tree.nodeAct_a.shape)
    paddingDelta = np.zeros((tree.nodeAct_a.shape[0],1))
    
    #% Get the deltas for each node to pass down
    nodeDeltas = np.dot(np.dot(tree.poolMatrix,np.transpose(Wcat[:,:-numFeatures])), topDelta)
    s = nodeDeltas.shape[0]
    nodeDeltas = nodeDeltas.reshape(params.wordSize,s/params.wordSize)
    
    #% Pass out the deltas to each node
    for ni in range(len(tree.nodePath)):
        node = tree.nodePath[ni]
        if node == -1:
            paddingDelta = paddingDelta + nodeDeltas[:,ni]
        else:
            if tree.isLeafVec[node]:
                nodeVecDeltas[:,node] = nodeVecDeltas[:,node] + nodeDeltas[:,ni]
            else:
                nodeVecDeltas[:,node] = nodeVecDeltas[:,node] + np.multiply(nodeDeltas[:,ni],(1 - tree.nodeAct_a[:,node]**2))
   
    #% Backprop into the nearest neighbors
    NN_deltas = nodeDeltas[:,ni:]
    return [df_s_Wcat, nodeVecDeltas, NN_deltas, paddingDelta]


def backpropAll(thisTree,W,WO,Wo,params,deltaUp_vec,deltaUp_op,thisNode,numWordsInBatch,indicies=None,NN=None):

    wsz = params.wordSize
    r = params.rankWo
    
#    numWords = thisTree.numLeafs()
    
    df_Wv = np.zeros((wsz,numWordsInBatch))#sparse.csr_matrix((wsz, numWordsInBatch), dtype='float64')    
    df_Wo = np.zeros((wsz+2*r*wsz, numWordsInBatch))#sparse.csr_matrix((wsz+2*r*wsz,numWordsInBatch), dtype='float64') 
    
    #%%%%%%%%%%%%%%
    #% df_W
    #% add here: delta's from your pooled matrix instead of deltaDownAddScore
    deltaVecDownFull = (deltaUp_vec + thisTree.nodeVecDeltas[:,[thisNode]])
    
    kids = thisTree.getKids(thisNode);
    kidsParInLR = np.zeros((wsz,2))
    kidsParInLR[:,0] = thisTree.ParIn_a[:,kids[0]]
    kidsParInLR[:,1] = thisTree.ParIn_a[:,kids[1]]
    
    kidsAct = np.vstack((kidsParInLR[:,[0]],kidsParInLR[:,[1]], np.ones((1,1)))) 
    df_W =  np.dot(deltaVecDownFull,np.transpose(kidsAct))
    
    #%%%%%%%%%%%%%%
    #% df_W
    kidsOps = np.zeros((wsz*wsz,2))
    kidsOps[:,0] = thisTree.nodeOp_A[:,kids[0]]
    kidsOps[:,1] = thisTree.nodeOp_A[:,kids[1]]
    
    
    deltaOpDownFull = deltaUp_op;
    df_WO = np.dot(deltaOpDownFull , np.transpose(np.vstack((kidsOps[:,0].reshape(wsz,wsz), kidsOps[:,1].reshape(wsz,wsz)))))
    
    WOxDeltaUp = np.dot(np.transpose(WO) , deltaOpDownFull)
    
    Wdelta_bothKids = np.dot(np.transpose(W),deltaVecDownFull)
    Wdelta_bothKids = Wdelta_bothKids[:2*wsz]
    Wdelta_bothKids= Wdelta_bothKids.reshape(wsz,2)
    
    otherKid = np.zeros(2)
    otherKid[0] = 1;
    otherKid[1] = 0;
    
    kidsActLR = np.zeros((wsz,2))
    kidsActLR[:,0] = thisTree.nodeAct_a[:,kids[0]]
    kidsActLR[:,1] = thisTree.nodeAct_a[:,kids[1]]
    
    # collect deltas from each children (they cross influence each other via operators)
    deltaDown_vec = np.zeros((wsz,2))
    deltaDown_op = np.zeros((wsz*wsz,2))
    for c in [0, 1]:
        delta_intoMatrixVec = Wdelta_bothKids[:,c]
        deltaDown_op[:,otherKid[c]] = np.dot(delta_intoMatrixVec,np.transpose(kidsActLR[:,c]))
        otherChildOp = kidsOps[:,otherKid[c]].reshape(wsz, wsz)
        if thisTree.isLeafVec[kids[c]]:
            deltaDown_vec[:,c] = np.dot(np.transpose(otherChildOp) , delta_intoMatrixVec)
        else:
            deltaDown_vec[:,c] = np.multiply(np.dot(np.transpose(otherChildOp) ,delta_intoMatrixVec), (1 - kidsActLR[:,c]**2))
            
    for c in [0, 1] :
        if thisTree.isLeaf(kids[c]):
            thisWordNum = thisTree.nodeLabels[kids[c]]
            df_Wv[:,thisWordNum] = df_Wv[:,thisWordNum]+deltaDown_vec[:,c] + thisTree.nodeVecDeltas[:,kids[c]]
            #bhanu
#            NN_ind = -1;
#            if kids[c] == indicies[0]:
#                NN_ind = 0;
#            elif kids[c] == indicies[1]:
#                NN_ind = 1;
#            
#            if NN_ind > -1:
#                el_NN = NN[:,NN_ind]
#                for i in range(params.NN):
#                    df_Wv[:,el_NN[i]] = df_Wv[:,el_NN[i]] + thisTree.NN_deltas[:,NN_ind]
                
                        
            deltaDown_op[:,c] = (deltaDown_op[:,c].reshape(wsz,wsz) + WOxDeltaUp[(c)*wsz:(c+1)*wsz,:]).flatten()
            
            numP = wsz*r
            dAlpha = np.diag(deltaDown_op[:,c].reshape(wsz,wsz)).reshape(wsz,1)
        
            wordWo_v = Wo[wsz*(1+r):,thisWordNum]
            WoASD_v = wordWo_v.reshape(wsz,r)
            dWo_u = np.dot(deltaDown_op[:,c].reshape(wsz,wsz),WoASD_v)
            final_dWo_u = dWo_u.reshape(numP, 1)
                
            wordWo_u = Wo[wsz:wsz*(1+r),thisWordNum]
            WoASD_u = wordWo_u.reshape(wsz,r)
            dWo_v = np.dot(np.transpose(deltaDown_op[:,c].reshape(wsz,wsz)), WoASD_u)
            final_dWo_v = dWo_v.reshape(numP,1)
            df_Wo[:,thisWordNum] = df_Wo[:,thisWordNum] + np.vstack((dAlpha.reshape(wsz,1), final_dWo_u, final_dWo_v)).flatten()
        else:
            deltaDown_op[:,c] = (deltaDown_op[:,c].reshape(wsz,wsz) + WOxDeltaUp[(c)*wsz:(c+1)*wsz,:]).flatten()
            [df_Wv_new,df_Wo_new,df_W_new,df_WO_new] = backpropAll(thisTree,W,WO,Wo,params,deltaDown_vec[:,[c]],
                                                                   deltaDown_op[:,[c]].reshape(wsz,wsz),
                                                                   kids[c],numWordsInBatch,indicies,NN)
            df_Wv = df_Wv + df_Wv_new;
            df_Wo = df_Wo + df_Wo_new;
            df_W  = df_W  + df_W_new;
            df_WO = df_WO + df_WO_new;    

    return [df_Wv,df_Wo,df_W,df_WO]


def setVerbnWordDistanceFeat(Wv, sNum, vid, wid, params):
    words = np.where(sNum>=0)[0]
    wsz = params.wordSize
    row = wsz - 2
    maxdistance = 10.0
    for i in range(len(words)):
        wordtoword = i - wid
        wordtoverb = i - vid
        if(np.abs(wordtoword) > maxdistance):
            wordtoword = maxdistance
        if(np.abs(wordtoverb) > maxdistance):
            wordtoverb = maxdistance
        
        Wv[row,sNum[words[i]]] = wordtoword/maxdistance
        Wv[row+1,sNum[words[i]]] = wordtoverb/maxdistance
        

def cost_oneSent(W, WO, Wcat, Wv, Wo, sNum, sTree, sStr, sNN, labels, verbIndices, params):
      
    #do forward propagation and sum up classification errors at each node
    cost = 0.0 #cost of one sentence = sum of cost of classification of each word for each verb
    forwardPropTrees = []; goldlabels = []
#    print "SSTR: ", sStr
    for nverb, vid in enumerate(verbIndices):
        for wid in range(len(sStr)):
            indices = np.array([vid, wid])   
            truelabel = labels[nverb, wid]
            setVerbnWordDistanceFeat(Wv, sNum, vid, wid, params) 
            tree = forwardPropTree(W, WO, Wcat, Wv, Wo, sNum, sTree, sStr, sNN,  params=params) #calculate nodevectors and matrices        
            calPredictions(tree, Wcat, Wv, indices, sStr, params) #updates score, nodepath etc for this verb, word pair
            cost += -np.log(tree.y[truelabel]) 
            forwardPropTrees.append(tree)
            goldlabels.append(truelabel)
    
    #do backpropagation
    Sdf_s_Wv = None; Sdf_s_Wo = None; Sdf_s_W = None; Sdf_s_WO = None; Sdf_s_Wcat = None
    for ftree, label in zip(forwardPropTrees, goldlabels):
        [df_s_Wcat, ftree.nodeVecDeltas, ftree.NN_deltas, paddingDelta] = backpropPool(ftree, label, Wcat, params, classWeight=((0.7*len(sStr))/len(sStr)))

        deltaDown_vec = np.zeros((params.wordSize,1))
        deltaDown_op = np.zeros((params.wordSize,params.wordSize))
        
        topNode = ftree.getTopNode();
        [df_s_Wv,df_s_Wo,df_s_W,df_s_WO] = backpropAll(ftree,W,WO,Wo,params,deltaDown_vec,deltaDown_op,topNode,Wv.shape[1],None,None)
        
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
        
    #scale cost and derivative by the number of forward trees created for this sentence
    #divide by number of sentences 
    numSent = len(forwardPropTrees)
    cost = (1.0/numSent)*cost
    Sdf_s_Wcat = (1.0/numSent) * Sdf_s_Wcat
    Sdf_s_W = (1.0/numSent) * Sdf_s_W
    Sdf_s_Wv   = (1.0/numSent) * Sdf_s_Wv
    Sdf_s_WO   = (1.0/numSent) * Sdf_s_WO
    Sdf_s_Wo   = (1.0/numSent) * Sdf_s_Wo
        
    
    return [Sdf_s_Wv, Sdf_s_Wo, Sdf_s_W, Sdf_s_WO, Sdf_s_Wcat, cost]

    
def costFn(theta, rnnData, params):
    '''returns avergage classification cost and its gradient over the given training set 
    Note: it always uses reduced vocabulary for unrolling theta
    '''
    
    #    Xs,ys,ts = trainSet
    allSNum = rnnData.allSNum
    allSTree = rnnData.allSTree
    allSStr = rnnData.allSStr
    allSNN = rnnData.allSNN
#    allIndicies = rnnData.allIndicies 
    sentenceLabels = rnnData.sentenceLabels
    verbIndices = rnnData.verbIndices
    
    W, WO, Wcat, Wv, Wo = unroll_theta(theta, params)
    numSent = len(allSNum);
    numFeats = len(params.features_std);
    wsz = params.wordSize;

    Wv_df = np.zeros(Wv.shape)
    Wo_df = np.zeros(Wo.shape)
    W_df = np.zeros(W.shape)
    WO_df = np.zeros(WO.shape)
    Wcat_df = np.zeros(Wcat.shape) 
    totalCost = 0.0  
    

    for t in range(numSent):
        if((len(allSNum[t]) == 1) or (len(verbIndices[t])==0) or (sentenceLabels[t].shape[1] != len(allSStr[t]))):
            continue  #only one word in a sent, no verbs for this sent, tokens and labels mismatch                 
        [df_s_Wv,df_s_Wo,df_s_W,df_s_WO, df_s_Wcat,cost] = cost_oneSent(W, WO, Wcat, Wv, Wo, allSNum[t],allSTree[t], allSStr[t],
                                                                        None, sentenceLabels[t], verbIndices[t], params)  #bhanu
        totalCost += cost
        Wcat_df = Wcat_df + df_s_Wcat
        W_df = W_df + df_s_W
        WO_df = WO_df + df_s_WO
        Wv_df = Wv_df + df_s_Wv
        Wo_df = Wo_df + df_s_Wo
         
    #divide by number of sentences 
    cost = (1.0/numSent)*totalCost
    Wcat_df = (1.0/numSent) * Wcat_df
    W_df = (1.0/numSent) * W_df
    Wv_df   = (1.0/numSent) * Wv_df
    WO_df   = (1.0/numSent) * WO_df
    Wo_df   = (1.0/numSent) * Wo_df
    
    #Add regularization to gradient
    Wcat_df = Wcat_df + np.hstack((params.regC_Wcat*Wcat[:,:-numFeats], params.regC_WcatFeat*Wcat[:,-numFeats:]))
    Wv_df = Wv_df+ params.regC_Wv * Wv
    W_df  = W_df + params.regC  * np.hstack((W[:,:-1], np.zeros((W.shape[0],1))))
    Wo_df = Wo_df + params.regC_Wo * np.vstack(((Wo[:wsz,:] - np.ones((wsz,Wo.shape[1]))), Wo[wsz:,:]))
    WO_df = WO_df + params.regC * WO

    # Add regularization to cost    
    cost = cost +  params.regC_Wcat/2.0 * np.sum(np.sum(Wcat[:,-numFeats]**2)
                                                 ) + params.regC_WcatFeat/2.0 * np.sum(np.sum(Wcat[:,-numFeats:]**2)
                                                 ) + params.regC_Wv/2.0 * np.sum(np.sum(Wv**2)
                                                 ) + params.regC/2.0 * np.sum(np.sum(W[:,-1]**2)
                                                 ) + params.regC_Wo/2.0 * np.sum(np.sum(np.vstack((Wo[:wsz,:] - np.ones((wsz,Wo.shape[1])), Wo[wsz:,:]**2)))
                                                 ) + params.regC/2.0 * np.sum(WO**2)

    #return cost and gradient
    grad = np.concatenate((W_df.flatten(), WO_df.flatten(), Wcat_df.flatten(), Wv_df.flatten(), Wo_df.flatten()))
#    print "grad: ", np.sum(grad**2)
#    print "cost : ", cost
#    print 
    return cost, grad

