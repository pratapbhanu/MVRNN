'''
Created on Oct 31, 2013

@author: bhanu
'''
from collections import defaultdict
import iais.data.config as config
import scipy.io as sio
import numpy as np
import cPickle
import os

def read_srl(filename):
    """
    Reads an SRL file and returns the training data. The file
    should be divided in columns, separated by tabs and/or whitespace.
    First column: tokens
    Second column: - for non-predicates, anything else for predicates.
    Third and next columns: the SRL IOBES tags for each token concerning
    each predicate (3rd column for 1st predicate, 4th for the 2nd, and
    so on).
    
    :returns: a list of tuples in the format (tokens, tags, predicates)
    """
    sentences = []
    pred_texts = []
    with open(filename, 'rb') as f:
        token_num = 0
        sentence = []
        tags = []
        predicates = []
        pred_text = []
        
        for line in f:
            line = unicode(line, 'utf-8').strip()
            
            if line == '':
                # last sentence ended
                sentences.append((sentence, tags, predicates))
                pred_texts.append(pred_text)
                sentence = []
                tags = []
                predicates = []; pred_text = []
                token_num = 0
                continue
            
#            parts = line.split('\t')
            parts = line.split()
            sentence.append(parts[0].strip())
            
            # check if this is a predicate
            pred_text.append(parts[1].strip())
            if parts[1].strip() != '-':
                predicates.append(token_num)
                
            
            # initialize the expected roles
            if tags == []:
                num_preds = len(parts) - 2
                tags = [[] for _ in range(num_preds)]
            
            for i, role in enumerate(parts[2:]):
                # the SRL tags
                tags[i].append(role)
            
            token_num += 1
        
    
    if sentence != []:
        sentences.append((sentence, tags, predicates))
    if(pred_text != []):
        pred_texts.append(pred_text)
    _, tail = os.path.split(filename)
    wf = open(config.corpus_path+tail+".verbs", 'w')
    cPickle.dump(pred_texts, wf, protocol=-1)
    wf.close()    
    
    return sentences


def convert_iob_to_iobes(iob_tags):
    """Converts a sequence of IOB tags into IOBES tags."""
    iobes_tags = []
    
    # check each tag and its following one. A None object is appended 
    # to the end of the list
    for tag, next_tag in zip(iob_tags, iob_tags[1:] + [None]):
        if tag == 'O':
            iobes_tags.append('O')
        elif tag.startswith('B'):
            if next_tag is not None and next_tag.startswith('I'):
                iobes_tags.append(tag)
            else:
                iobes_tags.append('S-%s' % tag[2:])
        elif tag.startswith('I'):
            if next_tag == tag:
                iobes_tags.append(tag)
            else:
                iobes_tags.append('E-%s' % tag[2:])
        else:
            raise ValueError("Unknown tag: %s" % tag)
    
    return iobes_tags

def fix_invalid_tags(iob_tags):
    fixed_iob_tags  = []
    for prevtag, thistag, next_tag in zip([None]+iob_tags[:-1], iob_tags, iob_tags[1:]+[None]):
        if(thistag == 'O'):
            fixed_iob_tags.append('O')
        elif(thistag.startswith('B')):
                fixed_iob_tags.append(thistag)
        elif(thistag.startswith('I')):
            if(prevtag is not None ):
                if(prevtag.startswith('O')):
                    fixed_iob_tags.append('O')
                elif(prevtag.startswith('B') and prevtag[2:]==thistag[2:]):
                    fixed_iob_tags.append(thistag)
                else:
                    fixed_iob_tags.append('O')
            else:
                fixed_iob_tags.append('O')
    return fixed_iob_tags
                    

def convert_iobes_to_bracket(tag):
    """
    Convert tags from the IOBES scheme to the CoNLL bracketing.
    
    Example:
    B-A0 -> (A0*
    I-A0 -> *
    E-A0 -> *)
    S-A1 -> (A1*)
    O    -> *
    """
    if tag.startswith('I') or tag.startswith('O'):
        return '*'
    if tag.startswith('B'):
        return '(%s*' % tag[2:]
    if tag.startswith('E'):
        return '*)'
    if tag.startswith('S'):
        return '(%s*)' % tag[2:]
    else:
        raise ValueError("Unknown tag: %s" % tag)

def prop_conll(verbs, props, sent_length, sent=None):
    """
    Returns the string representation for a single sentence
    using the CoNLL format for evaluation.
    
    :param verbs: list of tuples (position, token)
    :param props: list of lists with IOBES tags.
    """
    # defaultdict to know what to print in the verbs column
    verb_dict = defaultdict(lambda: '-', verbs)
    lines = []
    
    for i in range(sent_length):
        verb = verb_dict[i]        
        args = [convert_iobes_to_bracket(x[i]) for x in props]
#        args = [x[i] for x in props]
        if(sent!=None):
            lines.append('\t'.join([sent[i]]+[verb] + args))
        else:
            lines.append('\t'.join([verb] + args))
    
    # add a line break at the end
    result = '%s\n' % '\n'.join(lines) 
    return result.encode('utf-8')


def create_srl_data():
        
        inMats = ['allTrainData_srl', 'allDevData_srl', 'allTestData_srl']
        insrls = ['srl_iob.train', 'srl_iob.dev', 'srl_iob.test']
        outMats = ['FinalTrainData_srl', 'FinalDevData_srl', 'FinalTestData_srl']
        
        #find unique categories:        
        categories = set()      
        for filename in insrls:  
            sentences_tags_verbs = read_srl(config.corpus_path+filename)        
            for stv in sentences_tags_verbs:
                _,tagslist,_ = stv
                for tags in tagslist:
                    for tag in tags:
                        tag = tag.strip()
                        categories.add(tag)
        categories = list(categories)
        catIds = range(len(categories))
        cat_dict = dict(zip(categories, catIds))        
        
        for infmat,infsrl, outfmat in zip(inMats, insrls, outMats):
            print 'processing ', infmat
            keys = ['allSStr', 'allSNum', 'allSTree', 'words', 'allSKids']
            data = sio.loadmat(config.corpus_path+infmat,variable_names=keys)
            filename = config.corpus_path+infsrl
            sentences_tags_verbs = read_srl(filename)
    #        allSStr = data.get('allSStr').flatten()
            allSNum = data.get('allSNum').flatten()
            allSKids = data.get('allSKids').flatten()
            allSTree = data.get('allSTree').flatten()
            newAllIndicies = []
    #        newAllSStr = []
            newAllSNum = []
            newAllSKids = []
            newAllSTree = []
            sentenceLabels = []
            verbIndices = []
            for i, s_ts_vs in enumerate(sentences_tags_verbs): 
                sent, tags, verbidx = s_ts_vs
                verbIndices.append(verbidx)
                for nv, vid in enumerate(verbidx):# for each verb
                    for wid in range(len(sent)): #for each word in the sentence
    #                    newAllSStr.append(sent)
                        newAllSTree.append(allSTree[i].flatten())
                        newAllSNum.append(allSNum[i].flatten())
                        newAllSKids.append(allSKids[i])
                        newAllIndicies.append([vid, wid]) #element1 = verb, element2 = word
                        sentenceLabels.append(cat_dict.get(str(tags[nv][wid]).strip()))
            save_dict = {'allSStr':[], 'allSNum':newAllSNum, 'allSTree':newAllSTree, 'sentenceLabels':sentenceLabels, 
                         'allIndicies':newAllIndicies, 'allSKids':newAllSKids, 'categories':categories, 'verbIndices':verbIndices}
            
#            newAllSNum = np.asanyarray(newAllSNum, dtype='int64')
#            newAllSTree = np.asanyarray(newAllSTree, dtype='int64')
#            sentenceLabels = np.asanyarray(sentenceLabels, dtype='int64')
#            newAllIndicies = np.asanyarray(newAllIndicies, dtype='int64')
#            newAllSKids = np.asanyarray(newAllSKids, dtype='int64')
#            verbIndices = np.asanyarray(verbIndices, dtype='int64')
            
            sio.savemat(config.corpus_path+outfmat, mdict=save_dict)
            print "saved "+outfmat

def create_srl_data2():
    inMats = ['allTrainData_srl', 'allDevData_srl', 'allTestData_srl']
    insrls = ['srl_iob.train', 'srl_iob.dev', 'srl_iob.test']
    outMats = ['FinalTrainData_srl2', 'FinalDevData_srl2', 'FinalTestData_srl2']
    
    #load categories
    with open(config.corpus_path+'srl_Freq.categories', 'r') as rf:
        categories = rf.readlines()
    
    nMostFrequentCats = 34
    category_keys = [x.strip() for x in categories]        
    cat_dict= dict(zip(category_keys[:nMostFrequentCats], range(nMostFrequentCats)))
    cat_dict["OTHER"] = nMostFrequentCats
    
    freq_cats = category_keys[:nMostFrequentCats] + ['OTHER']
        
    for infmat,infsrl, outfmat in zip(inMats, insrls, outMats):
        print 'processing ', infmat
        keys = ['allSStr', 'allSNum', 'allSTree', 'words', 'allSKids']
        data = sio.loadmat(config.corpus_path+infmat,variable_names=keys)
        filename = config.corpus_path+infsrl
        sentences_tags_verbs = read_srl(filename)
        allSStr = data.get('allSStr').flatten()
        allSNum = data.get('allSNum').flatten()
        allSKids = data.get('allSKids').flatten()
        allSTree = data.get('allSTree').flatten()
        sentenceLabels = []
        verbIndices = []
        for i, s_ts_vs in enumerate(sentences_tags_verbs): 
            sent, tags, verbidx = s_ts_vs
            verbIndices.append(verbidx)
            thissentLabels = []
            for nv, vid in enumerate(verbidx):# for each verb
                labels4thisverb = []
                for wid in range(len(sent)): #for each word in the sentence    
                    tagkey = str(tags[nv][wid]).strip()
                    if(cat_dict.has_key(tagkey)):                
                        labels4thisverb.append(cat_dict.get(tagkey))
                    else:
                        labels4thisverb.append(cat_dict.get("OTHER"))
                thissentLabels.append(labels4thisverb)
            sentenceLabels.append(thissentLabels)
        save_dict = {'allSStr':allSStr, 'allSNum':allSNum, 'allSTree':allSTree, 'sentenceLabels':sentenceLabels, 
                      'allSKids':allSKids, 'categories':freq_cats, 'verbIndices':verbIndices}
        
        sio.savemat(config.corpus_path+outfmat, mdict=save_dict)
        print "saved "+outfmat
                

def create_evalfile_srl():
    sentences_tags_verbs = read_srl(config.corpus_path+'srl_iob.test') 
#    rnnData_test = RNNDataCorpus()
#    rnnData_test.load_data_srl(config.test_data_srl, nExamples=nExamples)
    srl_preds = []
    with open(config.results_path+'preds_srl.txt') as srl_preds_file :
        for line in srl_preds_file:
            _, tag = line.split('\t')
            srl_preds.append(tag.strip())   
    
    i = 0
    verb_texts = cPickle.load(open(config.corpus_path+"srl_predicates.test", 'r'))
    for sid, stv in enumerate(sentences_tags_verbs):
        sent, _, verbidx = stv
        verb_text = verb_texts[sid]
#        if(len(srl_preds) < i + len(verbidx)*len(sent)):
#            return
        verbs = []; predtagsLists = []
        for nv, vid in enumerate(verbidx):# for each verb
            verbs.append((vid, verb_text[vid]))
            predtagsLists.append(srl_preds[i:i+len(sent)]) #for each word in the sentence
            i  = i + len(sent)   
        predtagsLists1 = [fix_invalid_tags(predtags) for predtags in predtagsLists]     
        tags = [convert_iob_to_iobes(predtags) for predtags in predtagsLists1]
        
        print prop_conll(verbs, tags, len(sent), sent=None)
                
def read_sent_chunktags(chunk_tags_file):
    chk_tags = []
    pos_tags = []
    with open(chunk_tags_file, 'rb') as f:       
        chktags = []
        postags = []
        for line in f:
            line = unicode(line, 'utf-8').strip()            
            if line == '':
                # last sentence ended
                chk_tags.append(chktags) 
                pos_tags.append(postags)               
                chktags = []; postags = []                
                continue
            
            parts = line.split()
            postags.append(parts[0].strip())
            chktags.append(parts[1].strip())
            
    return pos_tags, chk_tags


