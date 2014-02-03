'''
Created on Nov 23, 2013

@author: bhanu
'''
import iais.data.config as config
import scipy.io as sio
import numpy as np
from iais.util.tree import Tree
from iais.data.rnn_data import RNNDataCorpus
from iais.io.read_corpus import read_sent_chunktags, read_srl
 


def writeVectors():

    vecFileName = config.results_path+"vectors.out"
    vecFile = open(vecFileName, 'w')
    
    mats = sio.loadmat(config.corpus_path+'vars.normalized.100.mat')    
    We_orig = mats.get('We')
    
    params = sio.loadmat(config.corpus_path+'params_rae.mat')
    W1 = params.get('W1')
    W2 = params.get('W2')
    b1 = params.get('b1')
    We = params.get('We')
    b = params.get('b')
    W = params.get('W')
    
    hiddenSize = 100
    
    nExamples = 5
    print "loading data.."
    rnnData_train = RNNDataCorpus()
    rnnData_train.load_data_srl(load_file=config.train_data_srl, nExamples=nExamples)  
    
    print 'writing vectors to: ', vecFileName
    for ii in range(len(rnnData_train.allSNum)):       
        
        sNum = rnnData_train.allSNum[ii]
        sStr = rnnData_train.allSStr[ii]
        sTree = rnnData_train.allSTree[ii]
        sKids = rnnData_train.allSKids[ii]
        
        words_indexed = np.where(sNum >= 0)[0]
        #L is only the part of the embedding matrix that is relevant for this sentence
        #L is deltaWe
        if We.shape[1] != 0:
            L = We[:, words_indexed]
            words_embedded = We_orig[:, words_indexed] + L;
        else :
            words_embedded = We_orig[:, words_indexed]
#        sl = words_embedded.shape[1]
        
        tree = Tree()
        tree.pp = all#np.zeros(((2*sl-1),1))
        tree.nodeScores = np.zeros(len(sNum))
#        tree.nodeNames = np.arange(1,(2*sl-1))
        tree.kids = np.zeros((len(sNum),2))
        
        tree.nodeFeatures = np.zeros((hiddenSize, len(sNum)))
        tree.nodeFeatures[:,:len(words_indexed)] = words_embedded;
        
        toMerge = np.zeros(shape=(words_indexed.shape), dtype='int32')
        toMerge[:] = words_indexed[:]    
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
            tree.kids[newParent,:] = [kid1, kid2];
            # set new parent to be possible merge candidate
            toMerge[i] = newParent;
            # delete other kid
            toMerge = np.delete(toMerge,i+1)
            
            c1 = tree.nodeFeatures[:,kid1]
            c2 = tree.nodeFeatures[:,kid2]
            
            p = np.tanh(np.dot(W1,c1) + np.dot(W2,c2) + b1.flatten())           
            
            tree.nodeFeatures[:,newParent] = p;
        
        vec = tree.nodeFeatures[-1]
        vecFile.write(" ".join([str(x) for x in vec])+'\n')
    
    vecFile.close()
    print "finished! "

def get_sent_phrases_srl(srl_iob_file, chunk_tags_file):
    ''' returns list of new sentences(reduced tokens) and list of phrases for each sentence.
        In list of new sentences, each new sentence contains either the single word or the phrase
        '''

    postags, chktags = read_sent_chunktags(chunk_tags_file)
    sentences_tags_verbs = read_srl(srl_iob_file)
    new_sent_tags_verbs = []
    new_sent_pos_chk = []
    for s, stv in enumerate(sentences_tags_verbs):
        sent, tagsList, verbIds = stv
        chktag = chktags[s]
        postag = postags[s]
        new_sent = [] 
        
        #construct new sent
        i = 0
        while( i < len(chktag) - 1):
            thistag = chktag[i]
            nexttag = chktag[i+1]
            if(thistag.startswith('B-') and  nexttag.startswith('I-')
               and thistag != 'B-VP'):
                phrase = [sent[i]] 
                j = i+1
                while(j < len(chktag) and  chktag[j].startswith('I-')):
                    phrase.append(sent[j])
                    j += 1
                new_sent.append(phrase)                    
                i = j
                
            else:
                new_sent.append([sent[i]])
                i += 1
                    
            if(i == len(chktag) -1):
                new_sent.append([sent[i]])
        
        #construct new pos n chk tags
        offset = 0
        new_posTags = []
        new_chkTags = []
        for i, phrase in enumerate(new_sent):
            offset+=len(phrase)
            new_posTags.append(postag[offset-1]) 
            new_chkTags.append(chktag[offset-1])        
        #construct  verbIds for this new sent
        new_verbIds = []
        for verbId in verbIds:
            offset = 0
            for i, phrase in enumerate(new_sent):
                if(offset == verbId):
                    new_verbIds.append(i)
                    break
                offset += len(phrase)
        #construct tags
        new_tagsList =[]
        for nv, verbId in enumerate(verbIds):
            new_tags = []
            tags = tagsList[nv]
            offset = 0
            for i, phrase in enumerate(new_sent):
                new_tags.append(tags[offset])
                offset += len(phrase)
            new_tagsList.append(new_tags)
        
        new_sent_tags_verbs.append((new_sent, new_tagsList, new_verbIds))   
        new_sent_pos_chk.append((new_posTags, new_chkTags))     
#        all_new_sents.append(new_sent)  
#        all_new_verbIds.append(new_verbIds)
#        all_new_tags.append(new_tagsList)      
    
    return new_sent_tags_verbs, new_sent_pos_chk
                

def write_phrases_srl():             
    _, all_phrases = get_sent_phrases_srl()
    with open(config.corpus_path+'srl_phrases', 'w') as wf:
        for phrases in all_phrases:
            for phrase in phrases:
                wf.write(" ".join(phrase)+"\n")

def write_phrase_vectors():   
    vecFileName = config.results_path+"vectors_srl.out"
    vecFile = open(vecFileName, 'w')
    
    mats = sio.loadmat(config.corpus_path+'vars.normalized.100.mat')    
    We_orig = mats.get('We')
    words = mats.get('words')
    
    words = words.flatten()
    keys = [str(words[i][0]).strip() for i in range(len(words))]
    values = range(len(words))
    word_dict = dict(zip(keys, values))
    
    params = sio.loadmat(config.corpus_path+'params_rae.mat')
    W1 = params.get('W1')
    W2 = params.get('W2')
    b1 = params.get('b1')
    We = params.get('We')
    b = params.get('b')
    W = params.get('W')
    
    _, all_phrases = get_sent_phrases_srl()
    for phrases in all_phrases:
        for phrase in phrases:
            if(len(phrase) < 2):
                continue
            wids = [word_dict.get(word.strip()) if word_dict.has_key(word.strip()) else 
                    word_dict.get("*UNKNOWN*") for word in phrase]
        
            i = 0; p = None
            try:
                while(i < len(wids)): #merge sequentially, works well for smaller phrases
                    if(p == None):
                        kid1 = wids[i]; kid2 = wids[i+1]
                        c1 = We_orig[:, kid1]
                        c2 = We_orig[:, kid2]
                        p = np.tanh(np.dot(W1,c1) + np.dot(W2,c2) + b1.flatten())  
                        i = i+2
                    else:
                        nextKid = wids[i]
                        c2 = We_orig[:, nextKid]
                        c1 = p
                        p = np.tanh(np.dot(W1,c1) + np.dot(W2,c2) + b1.flatten()) 
                        i += 1
            
                if(p != None):
                    vecFile.write(" ".join([str(x) for x in p])+'\n')
            except:
                print
    vecFile.close()
        

def get_phrase_vector(phrase, W1, W2, b1, We, word_dict):
    
    wids = [word_dict.get(word.strip()) if word_dict.has_key(word.strip()) else 
                    word_dict.get("*UNKNOWN*") for word in phrase]
    if(len(wids) == 1):
        wid = wids[0]
        return We[:,wid]

    i = 0; p = None    
    while(i < len(wids)): #merge sequentially, works well for smaller phrases
        if(p == None):
            kid1 = wids[i]; kid2 = wids[i+1]
            c1 = We[:, kid1]
            c2 = We[:, kid2]
            p = np.tanh(np.dot(W1,c1) + np.dot(W2,c2) + b1.flatten())  
            i = i+2
        else:
            nextKid = wids[i]
            c2 = We[:, nextKid]
            c1 = p
            p = np.tanh(np.dot(W1,c1) + np.dot(W2,c2) + b1.flatten()) 
            i += 1

    return p
    
    

def get_features(srl_iob_file, chunks_tags_file):
    
    outfile = open(config.corpus_path+"srl_vec_features.train", 'w')
    phnlabels = open(config.corpus_path+"srl_phrases_labels.train", 'w')
    
    mats = sio.loadmat(config.corpus_path+'vars.normalized.100.mat')    
    We_orig = mats.get('We')
    words = mats.get('words')
    
    words = words.flatten()
    keys = [str(words[i][0]).strip() for i in range(len(words))]
    values = range(len(words))
    word_dict = dict(zip(keys, values))
    
    params = sio.loadmat(config.corpus_path+'params_rae.mat')
    W1 = params.get('W1')
    W2 = params.get('W2')
    b1 = params.get('b1')
    new_sents, _ = get_sent_phrases_srl(srl_iob_file, chunks_tags_file)
    sentences_tags_verbs = read_srl(srl_iob_file)
    for new_sent, sentence_tags_verbs in zip(new_sents, sentences_tags_verbs):
        sent, taglists, verbIds = sentence_tags_verbs
        for i, verbId in enumerate(verbIds):
            tags = taglists[i]
            offset = 0
            for wordOrPhrase in new_sent:
                try:
#                    wpvec = get_phrase_vector(wordOrPhrase, W1, W2, b1, We_orig, word_dict)
#                    verbVec = get_phrase_vector(sent[verbId], W1, W2, b1, We_orig, word_dict)
                    label = tags[offset]
                    offset += len(wordOrPhrase)
#                    row = " ".join([str(x) for x in wpvec]) + "\t" + " ".join([str(x) for x in verbVec]) \
#                            + "\t"+ label + '\n'
#                    outfile.write(row)
                    phnlabels.write(" ".join(wordOrPhrase)+ "\t" + label + "\n")
                except:
                    print
    outfile.close()
                 
    

if __name__ == "__main__":
#    writeVectors()
#    write_phrases_srl()
#    write_phrase_vectors()
    get_features(config.corpus_path+"srl_iob.train", config.corpus_path+"synt.train.gold")
    
    
    
    