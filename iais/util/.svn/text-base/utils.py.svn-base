'''
Created on Oct 21, 2013

@author: bhanu
'''

import numpy as np
from nltk.tree import Tree
from iais.util.tokenizer import Tokenizer
from sklearn.metrics import f1_score, precision_score, recall_score
from iais.network.mvrnn import forwardPropTree



def getRelevantWords(rnnData_mini, Wv,Wo,params):
    # Becasuse not all words are used in our dictionary, only the relvant words
    # in these sentences will be used to save memory.     
    #allIndicies=rnnData.allIndicies
    # add to our batch the padding and nearest neighbors
    allWordInds = set()
    allWordInds.add(0)  #add padding
    for s in range(len(rnnData_mini.allSNum)): 
        this_sent_uniqueIds = np.unique(np.hstack((rnnData_mini.allSNum[s])))#, rnnData_mini.allSNN[s][:params.NN,:].flatten()))) #bhanu
        for wid in this_sent_uniqueIds:
            allWordInds.add(wid)
    allWordInds = np.array(list(allWordInds), dtype='int32')
    
    
    allWordInds = np.sort(allWordInds)[1:] # ignore -2
#    all2Batch = containers.Map(num2cell(allWordInds),num2cell(1:length(allWordInds)));
    all2Batch = dict(zip(allWordInds, np.arange(len(allWordInds))))
    # get index of mid context padding, start sentence is always 1
    Wv_batch = Wv[:,allWordInds]
    Wo_batch = Wo[:,allWordInds]
#    print 'Updating allSNum to new batch numbers...'
    allSNum_batch = rnnData_mini.allSNum
#    allSNN_batch = rnnData_mini.allSNN #bhanu d
    # update all indicies for the batch
    for s in range(len(allSNum_batch)):
        for w in range(len(allSNum_batch[s])):
            if allSNum_batch[s][w]>0:
                allSNum_batch[s][w] = all2Batch.get(allSNum_batch[s][w])
            
        
#        allSNN_batch[s][params.NN:,:] = []  #bhanu to check
#        for e in [0, 1] :
#            for nn in range(params.NN):
#                allSNN_batch[s][nn,e] = all2Batch.get(allSNN_batch[s][nn,e])

                
    #set number of words in the reduced vocab, for unrolling theta
#    params.nWords_reduced = len(allWordInds)
    
    return [Wv_batch, Wo_batch, allWordInds]



#    print "Weights = ", W[0,0]
#    print "WO = ", WO[0,0]
#    print "Wcat = ", Wcat[0,0]


def build_dict_from_wordlist(words):
    word_dict  = {}
    idx = 0
    for word in words:
        if(word_dict.has_key(word)):
            continue
        else:
            word_dict[word] = idx
            idx+=1
    return word_dict


def generate_features_CW(CWEmbedsFile, text_reader):
    """
    Generates vectors of real numbers, to be used as word features.
    Vectors are initialized with Collobert and Weston word embeddings. Returns a 2-dim numpy array.
    """    
    cw_embs = {}
    with open(CWEmbedsFile, 'r') as ef:
        for line in ef:
            word, emb = line.split('\t')
            cw_embs[word] = np.fromstring(emb.strip(), sep=' ')
    
    word_dict = text_reader.word_dict
    num_features = len(cw_embs.get('PADDING'))
    num_vectors = len(word_dict)
    
    table = np.zeros(shape=(num_vectors, num_features))
    for word, i in word_dict.items():
        if(word == word_dict.padding_left or word == word_dict.padding_right):
            table[i,:] = cw_embs.get('PADDING')
        elif(word == word_dict.rare):
            table[i,:] = cw_embs.get('UNKNOWN')
        else:
            if(cw_embs.has_key(word)):
                table[i,:] = cw_embs.get(word)
            else:
                table[i,:] = cw_embs.get('UNKNOWN')     

    return table

def build_dictionary(X):
    word_dict = {}
    idx = 0
#    tokenizer = Tokenizer()
    for sent in X:        
        tokens = sent #tokenizer.getTokens(sent, replaceDigits=True)
        for tok in tokens:
            if(word_dict.has_key(tok.strip())):
                continue
            else:
                word_dict[tok.strip()] = idx
                idx+=1
    return word_dict


def codify_sentences(X, word_dict):
    newX = []
    tokenizer = Tokenizer()
    for sent in X:   
        word_ids = []     
        tokens = tokenizer.getTokens(sent, replaceDigits=True)
        for tok in tokens:
            if(word_dict.has_key(tok.strip())):
                word_ids.append(word_dict.get(tok.strip()))
            else:
                word_ids.append(word_dict.get('UNKNOWN'))
        newX.append(word_ids)
    return newX

def create_trees_nltk(filename):    
    f = open(filename, "r")

    response = f.readlines(); f.close()
    valid_tree_texts = []   
    tree_text = '' 
    for line in response:
        line = line.strip()
        if(line == ""):
            valid_tree_texts.append(tree_text)
            tree_text = ""            
        else:
            tree_text += line+" "        
    trees = [Tree.parse(line) for line in valid_tree_texts]
    
    for i in range(len(trees)):
        trees[i].chomsky_normal_form() 
    
    return trees

def create_preds_file(predictLabel, categories, sentenceLabels,  predictions_file, testKeys_file):
    
    proposedFile = open(predictions_file,'w')
    
#    if len(sentenceLabels) == 0:
#        print 'No labels, will just output predictions'
#        for j in range(len(predictLabel)):
#            line =  str(j) + '\t' + categories[predictLabel[j]] +'\n'
#            proposedFile.write(line)
#            proposedFile.flush()        
#        proposedFile.close()        
    
#    answerFile = open(testKeys_file,'w');
#    outFileStr =  os.path.join(config.results_path, 'results'+'.txt')
    
    id_to_cat = dict()
    for cid, cat in enumerate(categories):
        id_to_cat[cid] = cat.strip()
    
    for j in range(len(predictLabel)):
        line_preds = str(j) + '\t' + id_to_cat.get(predictLabel[j]) + '\n'
#        line_keys = str(j) + '\t' + categories[sentenceLabels[j]] + '\n'
        proposedFile.write(line_preds); proposedFile.flush()
#        answerFile.write(line_keys); answerFile.flush()    
    proposedFile.close()
#    answerFile.close()   
    
#    cmdList = [config.PROJECT_HOME+'/scripts/semeval2010_task8_scorer-v1.2.pl ', predictions_file, testKeys_file,  ' > ' , outFileStr];
#    subprocess.call(cmdList)
    

    

