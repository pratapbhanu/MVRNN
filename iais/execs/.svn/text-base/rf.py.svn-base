'''
Created on Nov 24, 2013

@author: bhanu
'''
from sklearn.ensemble.forest import RandomForestClassifier
import numpy as np
import iais.data.config as config
import scipy.io as sio
from iais.network.rae import get_phrase_vector, get_sent_phrases_srl
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score
import cPickle
from sklearn.linear_model.logistic import LogisticRegression
from iais.io.read_corpus import read_sent_chunktags
from sklearn.preprocessing import OneHotEncoder




def encode_cats(categorical_features):
    thisX = np.array(categorical_features)
    newX = np.zeros(thisX.shape)
    for c in range(thisX.shape[1]):
        cat_set = set()
        for key in thisX[:,c]:
            cat_set.add(key)
        cat_dict = dict(zip(cat_set, range(len(cat_set))))
        for i in range(thisX.shape[0]):
            newX[i,c] = cat_dict.get(thisX[i,c])
    return newX
            


def set_features(train_file, chk_train_file, X, y, nExamples):
    mats = sio.loadmat(config.corpus_path + 'vars.normalized.100.mat')
    We_orig = mats.get('We')
    words = mats.get('words')
    words = words.flatten()
    keys = [str(words[i][0]).strip() for i in range(len(words))]
    values = range(len(words))
    word_dict = dict(zip(keys, values))
    params = sio.loadmat(config.corpus_path + 'params_rae.mat')
    W1 = params.get('W1')
    W2 = params.get('W2')
    b1 = params.get('b1')
#    sentences_tags_verbs = read_srl(train_file)
    sentences_tags_verbs, sent_pos_chk = get_sent_phrases_srl(train_file, chk_train_file)
    categorical_features = []
#    all_pos_tags,all_chk_tags = read_sent_chunktags(chk_train_file)
    for s, stv in enumerate(sentences_tags_verbs[:nExamples]):
        sent, tagslist, verbIds = stv
        pos_tags, chk_tags = sent_pos_chk[s]
        
#        new_sent = new_sents[i]
        for nv, verbId in enumerate(verbIds):
            trueLabels = tagslist[nv]
            for i, phrase in enumerate(sent):
                #features of verb 
                verbVec = get_phrase_vector(sent[verbId], W1, W2, b1, We_orig, word_dict)
                #features of word to tag  
                phraseVec = get_phrase_vector(phrase, W1, W2, b1, We_orig, word_dict)
                #nBasePhrases between verb and phrase to tag
                nBasePhrases = verbId - i
                #pos tag of verb
#                vpostag = abs(hash(pos_tags[verbId].strip()))
                vpostag = pos_tags[verbId].strip()
                #pos tag of head word in phrase
#                ppostag = abs(hash(pos_tags[i].strip()))
                ppostag = pos_tags[i].strip()
                #pos tag of 2 words before the phrase
                if(i-1 < 0):
                    pbpostag = "None"
                else:
#                    pbpostag = abs(hash(pos_tags[i-1].strip()))
                    pbpostag = pos_tags[i-1].strip()
                if(i-2 < 0):
                    pb2postag = "None"
                else:
#                    pb2postag = abs(hash(pos_tags[i-2].strip()))
                    pb2postag = pos_tags[i-2].strip()
                #pos tag of 2 words after the phrase
                if(i+1 >= len(sent)):
                    papostag = "None"
                else:
#                    papostag = abs(hash(pos_tags[i+1].strip()))
                    papostag = pos_tags[i+1].strip()
                if(i+2 >= len(sent)):
#                    pa2postag = notag
                    pa2postag = "None"
                else:
#                    pa2postag = abs(hash(pos_tags[i+2].strip()))
                    pa2postag = pos_tags[i+2]
                #chunk tag of phrase
#                pchktag = abs(hash(chk_tags[i].strip()))
                pchktag = chk_tags[i].strip()
                vchktag = chk_tags[i].strip()
#                #number of words in phrase
                nwords = len(phrase)                
#                #number of verbs
                nverbs = len(verbIds)
                #position w.r.t verb
                if(verbId - i > 0):
                    poswrtverb = "before" #before
                elif(verbId -i < 0):
                    poswrtverb = "after" #after
                else:
                    poswrtverb = 'onVerb' #verb
                #top node feature 
                #number of verb phrase chunks
                
                categorical_features.append([vpostag, ppostag, pbpostag, pb2postag, papostag, pa2postag, pchktag, poswrtverb, vchktag, nwords, nverbs])
                
                X.append(np.concatenate((verbVec, phraseVec, [nBasePhrases])))
                y.append(trueLabels[i].strip())
    
    #transform categorical features into onehot encoding
#    int_features = encode_cats(categorical_features)
#    enc = OneHotEncoder()
#    int_features = enc.fit_transform(int_features)
#    nX = np.hstack((X, int_features.todense()))
    return np.array(X), np.array(y), categorical_features
            

                

def train_rf():
    
    train_file = config.corpus_path+"srl_iob.train"
    chk_train_file = config.corpus_path+"synt.train.gold"
    dev_file = config.corpus_path+"srl_iob.dev"
    chk_dev_file = config.corpus_path+"synt.dev.gold"
    
    print "creating features..."
    X = []; y = []
    Xdev = []; ydev = []

    X, y, cat_feat = set_features(train_file, chk_train_file, X, y, -1)
    Xdev, ydev, cat_feat_dev = set_features(dev_file, chk_dev_file, Xdev, ydev, 500)
    all_cat_feat = np.vstack((cat_feat, cat_feat_dev))
    int_features = encode_cats(all_cat_feat)
    enc = OneHotEncoder()
    int_features = enc.fit_transform(int_features)
    int_features = int_features.todense()
    X = np.hstack((X, int_features[:X.shape[0],:]))
    Xdev = np.hstack((Xdev, int_features[X.shape[0]:,:]))
    
    
#    cats = set()
#    for label in y:
#        cats.add(label)
#    for label in ydev:
#        cats.add(label)        
#    labels_dict = dict(zip(cats, range(len(cats))))
    
    #load categories    
    nMostFrequentCats = 34
    with open(config.corpus_path+'srl_Freq.categories', 'r') as rf:
        categories = rf.readlines()      
    category_keys = [x.strip() for x in categories]        
    cat_dict= dict(zip(category_keys[:nMostFrequentCats], range(nMostFrequentCats)))
    cat_dict["OTHER"] = nMostFrequentCats    
    freq_cats = category_keys[:nMostFrequentCats] + ['OTHER']
    freq_cats_dict = dict(zip(freq_cats, range(len(freq_cats))))
    
    for i in range(len(y)):
        if(freq_cats_dict.has_key(y[i].strip())):
            y[i] = freq_cats_dict.get(y[i].strip())
        else:
            y[i] = freq_cats_dict.get("OTHER")
    for i in range(len(ydev)):
        if(freq_cats_dict.has_key(ydev[i].strip())):
            ydev[i] = freq_cats_dict.get(ydev[i].strip())
        else:
            ydev[i] = freq_cats_dict.get("OTHER")
    
##################################################################    
#    classifier = RandomForestClassifier(n_estimators=100, 
#                                        verbose=2,
#                                        n_jobs=1,
#                                        min_samples_split=1,
#                                        random_state=0                                   
#                                        )    
###########################################################                                    
    classifier = LogisticRegression(penalty='l2', C=0.8, random_state=0)
#    X = np.array(X)
#    X -= np.mean(X, axis=0)
#    X /= np.std(X, axis=0)
############################################################    
    print "fitting classifier...", classifier
    classifier.fit(X, y)
    print "making predictions..."    
    ypred = classifier.predict(Xdev)
    accuracy = accuracy_score(ydev, ypred)
    precision = precision_score(ydev, ypred, pos_label=None, average=None)
    recall = recall_score(ydev, ypred,pos_label=None, average=None)
    f1 = f1_score(ydev, ypred,pos_label=None, average=None)
    print "Accuracy:" , accuracy
    print "Precision: ", precision
    print "Recall : ", recall
    print "F1 : ", f1
    
#    print "saving classifier.."
#    cPickle.dump(classifier, open(config.model_path+"rf_"+str(accuracy)+".pkl", 'wb'), protocol=-1)
    

if __name__ == '__main__':
    train_rf()