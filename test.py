from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from collections import Counter
from nltk.stem import WordNetLemmatizer 

import nltk
import os
import string
import numpy as np
import copy
import pandas as pd
import pickle
import re
import math


def convert_lower_case(data):
    return np.char.lower(data)

def remove_stop_words(data):
    stop_words = stopwords.words('english')
    words = word_tokenize(str(data))
    new_text = ""
    for w in words:
        if w not in stop_words and len(w) > 1:
            new_text = new_text + " " + w
    return new_text

def remove_punctuation(data):
    symbols = "!\"#$%&()*+-./:;<=>?@[\]^_`{|}~\n"
    for i in range(len(symbols)):
        data = np.char.replace(data, symbols[i], ' ')
        data = np.char.replace(data, "  ", " ")
    data = np.char.replace(data, ',', '')
    return data

def remove_apostrophe(data):
    return np.char.replace(data, "'", "")


def lemmatize(data):

	words = word_tokenize(str(data))

	lemmatizer = WordNetLemmatizer() 

	temp_str=""

	for w in words:
		temp_str = temp_str+" "+lemmatizer.lemmatize(w)

	return temp_str

def convert_numbers(data):
    tokens = word_tokenize(str(data))
    new_text = ""
    for w in tokens:
        try:
            w = num2words(int(w))
        except:
            a = 0
        new_text = new_text + " " + w
    new_text = np.char.replace(new_text, "-", " ")
    return new_text

def preprocess(data):
    data = convert_lower_case(data)
    data = remove_punctuation(data) #remove comma seperately
    data = remove_apostrophe(data)
    data = remove_stop_words(data)
    data = convert_numbers(data)
    data = lemmatize(data)
    data = remove_punctuation(data)
    data = convert_numbers(data)
    data = lemmatize(data) #needed again as we need to stem the words
    data = remove_punctuation(data) #needed again as num2word is giving few hypens and commas fourty-one
    data = remove_stop_words(data) #needed again as num2word is giving stop words 101 - one hundred and one
    return data




def doc_freq(word):
    c = 0
    try:
        c = DF[word]
    except:
        pass
    return c





########__main##########################
filedirectories = os.listdir('data/20ng/20news-bydate-train/')

processed_text = []

filenames=[]

for directory in filedirectories:

	filedirectory = 'data/20ng/20news-bydate-train/'+str(directory)+'/'

	allfiles = os.listdir(filedirectory)

	for i in allfiles:
		
		filenames.append(i)
		file = open(filedirectory+i, 'r', encoding="utf8", errors='ignore')
		text = file.read().strip()
		file.close()

		processed_text.append(word_tokenize(str(preprocess(text))))

	 
#print(processed_text)

docs = len(processed_text)

DF = {}

for i in range(docs):
    tokens = processed_text[i]
    for w in tokens:
        try:
            DF[w].add(i)
        except:
            DF[w] = {i}

    
for i in DF:
    DF[i] = len(DF[i])

total_vocab = [x for x in DF] 

print(DF)


doc = 0

tf_idf = {}

for i in range(docs):
    
    tokens = processed_text[i]
    
    counter = Counter(tokens) #finds unique words with their frequency
    words_count = len(tokens)
    
    for token in np.unique(tokens):
        
        tf = counter[token]/words_count
        df = doc_freq(token)
        idf = np.log((docs+1)/(df+1))
        
        tf_idf[doc, token] = tf*idf

    doc += 1


print(doc, len(filenames))

















