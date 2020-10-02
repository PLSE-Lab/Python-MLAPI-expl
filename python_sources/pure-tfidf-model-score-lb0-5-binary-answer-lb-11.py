#!/usr/bin/env python
# coding: utf-8

# The start in his most basic form...
# -----

# In[ ]:


import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
train = pd.read_csv('../input/train.csv')
print(train.head(10))

def mapfunctie(row):
    #rint(row)
    if(row==1):
        rij='dubbeltje'
    if(row==0):
        rij='leegaartje'
    return rij

train['mapp']=train['is_duplicate'].apply(mapfunctie)
        
train=train.dropna(axis=0, how='any')
train['test']=train['question1']+' '+train['question2']+train['mapp']  # leeg prevents sum of two questions sentence is empty

traintit=train[train['test']>'']  # remove all empties

count_vectorizer = CountVectorizer()
count_vectorizer.fit_transform(traintit['test'])  #Learn vocabulary and idf, return term-document matrix.
freq_term_matrix = count_vectorizer.transform(traintit['test']) #Transform documents to document-term matrix. Uses the vocabulary and document frequencies (df) learned by fit (or fit_transform) This is equivalent to fit followed by transform

tfidf = TfidfTransformer(norm="l2")
tf_idf_matrix = tfidf.fit_transform(freq_term_matrix)

print('Questions x Words', tf_idf_matrix.shape)
#als je similariteit wilt zien...
print('Q similarity',tf_idf_matrix[:14].dot(tf_idf_matrix[:14].T).todense().round(2) )

count_vectorizer.vocabulary_['dubbeltje']
      


# In[ ]:


print('analyse Question 1 : what is the step by step guide to invest in the stockmarket in Japan /Vietnam')
print('tfidf value of Q0 of leegaartje',tf_idf_matrix[0,45219])


isdup=pd.DataFrame(tf_idf_matrix[:,count_vectorizer.vocabulary_['leegaartje']].todense())
isdup['dupli']=tf_idf_matrix[:,count_vectorizer.vocabulary_['dubbeltje']].todense()
print(isdup.shape)
print(tf_idf_matrix.shape)

vietnam=tf_idf_matrix[:,count_vectorizer.vocabulary_['vietnam'] ].todense() # all questions with Vietnam
japan=tf_idf_matrix[:,count_vectorizer.vocabulary_['japan'] ].todense()
print('wordvector shape',vietnam.shape)  #all questions with Japan
print('Correlation Vietnam - japan',vietnam.T.dot(japan))

diffvect=pd.DataFrame(vietnam)
diffvect['japan']=japan
vc=vietnam.T.dot(isdup)
print('Vietnam 52% duplicate vrs nondup',vc[0,1]/vc[0,0])
jc=japan.T.dot(isdup)
print('Japan 57% dupl vrs nondup',jc[0,1]/jc[0,0])

print('------------------------------------')
print('analyse Question 7 : What should i do to be a great/good geologist')
print('print wordvectors is duplicate 0 of leegaartje: is dup',tf_idf_matrix[7,45219])

good=tf_idf_matrix[:,count_vectorizer.vocabulary_['good'] ].todense() # all questions with Vietnam
great=tf_idf_matrix[:,count_vectorizer.vocabulary_['great'] ].todense() #all questions with Japan
print('Correlation good - great very high...',good.T.dot(great))

vg=good.T.dot(isdup)
vr=great.T.dot(isdup)
print('good 80% duplicate versus nondup',vg[0,1]/vg[0,0])
print('great 110% duplicate versus nondup',vr[0,1]/vr[0,0])
print('remember this is not the real correlation, its indication of similarity' )
print(isdup.T.dot(isdup))


# In[ ]:


# Lets redo it but splitted... and use the existing vocabulary

count1_vectorizer = CountVectorizer(vocabulary=count_vectorizer.vocabulary_)
count1_vectorizer.fit_transform(train['question1'])  #Learn vocabulary and idf, return term-document matrix.
freq1_term_matrix = count_vectorizer.transform(train['question1'])
count2_vectorizer = CountVectorizer(vocabulary=count_vectorizer.vocabulary_)
count2_vectorizer.fit_transform(train['question2'])
freq2_term_matrix = count_vectorizer.transform(train['question2']) #Transform documents to document-term matrix. Uses the vocabulary and document frequencies (df) learned by fit (or fit_transform) This is equivalent to fit followed by transform


tfidf1 = TfidfTransformer(norm="l2")
tf1_idf_matrix = tfidf1.fit_transform(freq1_term_matrix)
tfidf2 = TfidfTransformer(norm="l2")
tf2_idf_matrix = tfidf2.fit_transform(freq2_term_matrix)

print('Questions1 x Words', tf1_idf_matrix.shape)
print('Questions2 x Words', tf2_idf_matrix.shape)
#als je similariteit wilt zien...
#print('Q similarity',tf1_idf_matrix[:10].dot(tf2_idf_matrix[:10].T) )

print(tf1_idf_matrix[:10].dot(tf2_idf_matrix[:10].T).diagonal())
print(train[:10].question1)


# In[ ]:


test = pd.read_csv('../input/test.csv')[:50000]
#we dont' need the complete training set to do this tfidf comparison !

test=test.dropna(axis=0, how='any')
test['test']=test['question1']+' '+test['question2']  # leeg prevents sum of two questions sentence is empty
testtit=test[test['test']>'']  # remove all empties

count_vectorizer = CountVectorizer()
count_vectorizer.fit_transform(testtit['test'])  #Learn vocabulary and idf, return term-document matrix.
freq_term_matrix = count_vectorizer.transform(testtit['test']) #Transform documents to document-term matrix. Uses the vocabulary and document frequencies (df) learned by fit (or fit_transform) This is equivalent to fit followed by transform

# Lets redo it but splitted... and use the existing vocabulary

count3_vectorizer = CountVectorizer(vocabulary=count_vectorizer.vocabulary_)
count3_vectorizer.fit_transform(test['question1'])  #Learn vocabulary and idf, return term-document matrix.
freq3_term_matrix = count_vectorizer.transform(test['question1'])
count4_vectorizer = CountVectorizer(vocabulary=count_vectorizer.vocabulary_)
count4_vectorizer.fit_transform(test['question2'])
freq4_term_matrix = count_vectorizer.transform(test['question2']) #Transform documents to document-term matrix. Uses the vocabulary and document frequencies (df) learned by fit (or fit_transform) This is equivalent to fit followed by transform


tfidf3 = TfidfTransformer(norm="l2")
tf3_idf_matrix = tfidf3.fit_transform(freq3_term_matrix)
tfidf4 = TfidfTransformer(norm="l2")
tf4_idf_matrix = tfidf4.fit_transform(freq4_term_matrix)

print('Test Questions1 x Words', tf3_idf_matrix.shape)
print('Test Questions2 x Words', tf4_idf_matrix.shape)
#als je similariteit wilt zien...
#print('Q similarity',tf1_idf_matrix[:10].dot(tf2_idf_matrix[:10].T) )

print(tf3_idf_matrix.dot(tf4_idf_matrix.T).diagonal().round(2))


# In[ ]:


count_vectorizer.vocabulary_


# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
import nltk
from nltk import word_tokenize
import re

from sklearn.model_selection import train_test_split as tts
from gensim.models import doc2vec
import gensim
import json

import sys

STOP_WORDS = nltk.corpus.stopwords.words()


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# Okay, learning from my previous venture [Cosine Similarity using TFIDF Weighting](https://www.kaggle.com/antriksh5235/quora-question-pairs/cosine-similarity-using-tfidf-weighting) I decided it was best to drop the questions less than a length of 10 and the stop words in sentences in order to avoid a lot of computation.
# 
# The following function was written to take care of all of this.

# In[ ]:


def clean_sentence(sent):
    regex = re.compile('([^\s\w]|_)+')
    sentence = regex.sub('', sent).lower()
    sentence = sentence.split(" ")

    for word in list(sentence):
        if word in STOP_WORDS:
            sentence.remove(word)

    sentence = " ".join(sentence)
    return sentence


# this simplifies abit the calculation, and gives the same result, there are more than ten different similarities calculations

# In[ ]:


import math
import numpy as np

def cosine(v1, v2):
    """
            v1 and v2 are two vectors (can be list of numbers) of the same dimensions. Function returns the cosine distance between those
            which is the ratio of the dot product of the vectors over their RS.
    """
    v1 = np.array(v1)
    v2 = np.array(v2)

    return np.dot(v1, v2.T) / np.dot(abs(v1),abs(v2.T))


# This one is simple enough, just dropping all the nans.

# In[ ]:


data = pd.read_csv('../input/train.csv')
data = data.dropna(how="any")


# I was first going for merging the two sentence sets into one so that the system is trained over all the words/sentences/document vectors and I would not be able to do it without concatenating both of these.

# In[ ]:


def concatenate(data):
    X_set1 = data['question1']
    X_set2 = data['question2']
#    y = data['is_duplicate']
    X = X_set1.append(X_set2, ignore_index=True)
    
    return X


# The above method is being run here to clean the sentences one by one.

# In[ ]:


print('Cleaning data, this might take long')
for col in ['question1', 'question2']:
    data[col] = data[col].apply(clean_sentence)


# I am splitting into training and testing set for now. The test given with this competition is very large and I only want to use it once I have tested out implementations.

# In[ ]:


print('Splitting data to train and test sets.')
y = data['is_duplicate']
X_train, X_test, y_train, y_test = tts(data[['id','question1', 'question2']], y, test_size=0.3)

X_train.head()


# Like I said, I am compiling my understanding of gensim from a lot of sources and one of them used multiprocessing, stating that it might be painfully slow doing otherwise.

# In[ ]:


import multiprocessing
cores = multiprocessing.cpu_count()
assert gensim.models.doc2vec.FAST_VERSION > -1
print(cores) 


# This is where the initial usage of gensim begins. Notice that I am yielding output from the __iter__ method, which is actually why I wrote this modification of the gensim LabeledLineSentence class. Every question is a document and every document has to be tagged, which is where I am using the id. I might as well start using the question Ids once I figure out how I am going to use the or I'll just append a '_q1' and '_q2' for each to know which is which and compare the same set of questions for every ID.
# 
# Once done, I hope this somehow helps in boosting performance. The only reason I went for doc2vec instead of word2vec was it might be able to capture the semantic relations within the sentence/question.

# In[ ]:


from gensim.models.doc2vec import Doc2Vec
from gensim.models import doc2vec
class LabeledLineSentence(object):

    def __init__(self, doc_list, labels_list):
        self.labels_list = labels_list
        self.doc_list = doc_list

    def __iter__(self):
        for idx, doc in enumerate(self.doc_list):
            yield doc2vec.TaggedDocument(words=word_tokenize(doc),
                                         tags=[self.labels_list[idx]])


# The iterator returns a yield of a TaggedDocument every time the Doc2Vec.build_vocab() function requests it. Had I not given the iterator with the Doc2Vec, I would have to call another model1.build_vocab(it) function just to perform the initialisation. This seemed a quick getaway from writing more lines.

# In[ ]:


X = concatenate(X_train)
labels = []
for label in X_train['id'].tolist():
    labels.append('SENT_%s_1' % label)
for label in X_train['id'].tolist():
    labels.append('SENT_%s_2' % label)

docs = LabeledLineSentence(X.tolist(), labels)
it = docs.__iter__()
model1 = Doc2Vec(it, size=12, window=8, min_count=5, workers=4)


# Okay, the more documents you have, the better. To better measure the similarity of documents, it is undoubtable that your model needs to have a very established vector for each.
# 
# Also, instead of running for the normal 10-20 epochs that people usually have for training Doc2vec models, I tried 100 epochs just in case. Let's see how the results turn up on the test sets I extracted.

# In[ ]:


for epoch in range(10):
    model1.train(it, total_examples=model1.corpus_count, epochs=model1.iter)
    model1.alpha -= 0.0002  # decrease the learning rate
    model1.min_alpha = model1.alpha  # fix the learning rate, no deca
    model1.train(it, total_examples=model1.corpus_count, epochs=model1.iter)


# Now finally for the similarity. I hope it turns out to be some good.

# In[ ]:


X_test.index = np.arange(0, X_test['question1'].shape[0])
y_test.index = np.arange(0, X_test['question1'].shape[0])
#print(X_test)
count = 0
for i in range(X_test['question1'].shape[0]):
    doc1 = word_tokenize(X_test['question1'][i])
    doc2 = word_tokenize(X_test['question2'][i])
    doceq= [w for w in doc1 if w in doc2]
    docdif1= [w for w in doc1 if w not in doc2]
    docdif2= [w for w in doc2 if w not in doc1]
    
    docvec1 = model1.infer_vector(doc1)
    docvec2 = model1.infer_vector(doc2)
    docveced = model1.infer_vector(doceq)
    docvecdi = model1.infer_vector(docdif1+docdif2)

    #print(docvec1)
    #print(docvec2)

    print(cosine(docvec1, docvec2), doc1,doc2, y_test[i])
    print(cosine(docveced, docvecdi), doceq,docdif1,docdif2, y_test[i])
    print(cosine(docdif1, docdif2), docdif1,docdif2, y_test[i])
    if count>20:
        break
    count+=1


# In[ ]:




