# -*- coding: utf-8 -*-
'''
Created on 12.03.2013

@author: ashah
'''

import nltk
import re
            


class Tokenizer(object):  
    def __init__(self):
        self.sentence_tokenizer = nltk.sent_tokenize
        self.word_tokenizer = nltk.word_tokenize
        self.digit = re.compile("[0-9]") 
    
    def getTokens(self,text, replaceDigits=True):
        text = text.encode('ascii','ignore')
        if(replaceDigits):
            text = self.digit.sub("9",text) #replace all digits with 9
        tokens = []
        sentences = self.sentence_tokenizer(text)
        for sentence in sentences:
            words = self.word_tokenizer(sentence)
            tokens = tokens + words
        return tokens
        