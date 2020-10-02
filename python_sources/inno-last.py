#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import pandas as pd
import numpy as np
import nltk
import spacy
import os
from sklearn.model_selection import train_test_split
import sklearn_crfsuite
from sklearn_crfsuite import scorers
from sklearn_crfsuite import metrics
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


train = pd.read_csv('../input/innovation-2/archive 3/full_train_with_pos.csv')
test = pd.read_csv('../input/innovation-2/archive 3/full_test_with_pos.csv')


# In[ ]:


train.fillna('thing',inplace = True)
test.fillna('thing',inplace = True)


# In[ ]:


test['tag'] = 'M'


# In[ ]:


class SentenceGetter(object):
    
    def __init__(self, data):
        self.n_sent = 1
        self.data = data
        self.empty = False
        agg_func = lambda s: [(w,p,t) for w, p, t in zip(s["Word"].values.tolist(),
                                                           s["tag_pos"].values.tolist(),
                                                           s["tag"].values.tolist())]
        self.grouped = self.data.groupby("Sent_ID").apply(agg_func)
        self.sentences = [s for s in self.grouped]
    
    def get_next(self):
            s = self.grouped[self.n_sent]
            self.n_sent += 1
            return s


# In[ ]:


getter_df_train = SentenceGetter(train)
getter_df_test = SentenceGetter(test)


# In[ ]:


sent = getter_df_train.get_next()
print(sent)


# In[ ]:


sentences_train = getter_df_train.sentences
sentences_test = getter_df_test.sentences


# In[ ]:


def word2features(sent, i):
    word = sent[i][0]
    postag = sent[i][1]
    
    features = {
        'bias': 1.0, 
        'word.lower()': word.lower(), 
        'word[-3:]': word[-3:],
        'word[-2:]': word[-2:],
        'no_of_capital' : sum(1 for c in word if c.isupper()),
        'hyphen' : word.count('-'),
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
        'postag': postag,
        'postag[:2]': postag[:2],
    }
    if i > 0:
        word1 = sent[i-1][0]
        postag1 = sent[i-1][1]
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper(),
            '-1:postag': postag1,
            '-1:postag[:2]': postag1[:2],
        })
    else:
        features['BOS'] = True
    if i < len(sent)-1:
        word1 = sent[i+1][0]
        postag1 = sent[i+1][1]
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:word.istitle()': word1.istitle(),
            '+1:word.isupper()': word1.isupper(),
            '+1:postag': postag1,
            '+1:postag[:2]': postag1[:2],
        })
    else:
        features['EOS'] = True

    return features

def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]

def sent2labels(sent):
    return [label for token, postag, label in sent]

def sent2tokens(sent):
    return [token for token, postag, label in sent]


# In[ ]:


X_train = [sent2features(s) for s in sentences_train]
y_train = [sent2labels(s) for s in sentences_train]


# In[ ]:


X_test = [sent2features(s) for s in sentences_test]


# # Uncomment for model training

# In[ ]:


# crf = sklearn_crfsuite.CRF(
#     algorithm='lbfgs',
#     c1=0.05,
#     c2=0.05,
#     max_iterations=200,
#     all_possible_transitions=True
# )
# crf.fit(X_train, y_train)


# In[ ]:


y_pred = crf.predict(X_test)


# In[ ]:


result = []
for i in y_pred:
    for j in i:
        result.append(j)


# In[ ]:


pd.Series(result).to_csv('new_one.csv')


# # ##The model output was obtained here it was combined with the output from Stanford core nlp model which is uploaded in the kaggle kernel
# # and they were ensembled with the other model output which was trained without the pos tag
# # the properties file that was used to train the model is
# # trainFile = /Users/al20018340/Desktop/Hackathons/Innoplexus/pos_tag/stanford_train_2.txt
# # serializeTo = /Users/al20018340/Desktop/Hackathons//Innoplexus/pos_tag/ner-model_2.ser.gz
# # map = word=0,answer=1
# 
# # useClassFeature=true
# # useWord=true
# # useNGrams=true
# # noMidNGrams=true
# # maxNGramLeng=7
# # usePrev=true
# # useNext=true
# # useSequences=true
# # usePrevSequences=true
# # maxLeft=1
# # useTypeSeqs=true
# # useTypeSeqs2=true
# # useTypeySequences=true
# # wordShape=chris2useLC
# # useDisjunctive=true
# # saveFeatureIndexToDisk=true
# # qnSize=25
# # maxQNItr=200
# # tolerance=0.0005
# 
# 

# In[ ]:




