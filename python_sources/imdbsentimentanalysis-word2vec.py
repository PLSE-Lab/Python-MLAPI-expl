#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import html
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


path = "/kaggle/input/aclimdb/aclImdb/"
positiveFiles = [x for x in os.listdir(path+"train/pos/")
                 if x.endswith(".txt")]
negativeFiles = [x for x in os.listdir(path+"train/neg/")
                if x.endswith(".txt")]
testFiles = [x for x in os.listdir(path+"test/") 
             if x.endswith(".txt")]


# In[ ]:


#positiveFiles


# In[ ]:


positiveReviews, negativeReviews, testReviews = [], [], []
for pfile in positiveFiles:
    with open(path+"train/pos/"+pfile, encoding="latin1") as f:
        positiveReviews.append(f.read())
for nfile in negativeFiles:
    with open(path+"train/neg/"+nfile, encoding="latin1") as f:
        negativeReviews.append(f.read())
for tfile in testFiles:
    with open(path+"test/"+tfile, encoding="latin1") as f:
        testReviews.append(f.read())


# In[34]:


print(len(positiveReviews))
print(len(negativeReviews))
print(len(testReviews))


# In[ ]:


# testReviews


# In[ ]:


reviews = pd.concat([pd.DataFrame({"review":positiveReviews, "label":1,
                                   "file":positiveFiles}),
                    pd.DataFrame({"review":negativeReviews, "label":0,
                                   "file":negativeFiles}),
                    pd.DataFrame({"review":testReviews, "label":-1,
                                   "file":testFiles})
                    ], ignore_index=True).sample(frac=1, random_state=1)
                    


# In[ ]:


reviews.shape


# In[ ]:


reviews[0:10]


# In[ ]:


from nltk.corpus import stopwords
import re


# In[ ]:


stopWords = stopwords.words('english')


# In[ ]:


def CleanData(sentence):
    processedList = ""
    
    #convert to lowercase and ignore special charcter
    sentence = re.sub(r'[^A-Za-z0-9\s.]', r'', str(sentence).lower())
    sentence = re.sub(r'\n', r' ', sentence)
    
    sentence = " ".join([word for word in sentence.split() if word not in stopWords])
    
    return sentence


# In[ ]:


reviews.info()


# In[ ]:


reviews['review'][0]


# In[ ]:


CleanData(reviews['review'][0])


# In[ ]:


reviews['review'] = reviews['review'].map(lambda x: CleanData(x))


# In[ ]:


reviews['review'].head()


# Going to use [gensim](https://radimrehurek.com/gensim/models/word2vec.html) library to train word2vec model. Gensim accepts input in form of list of lists, where each internal list consists of review sentence.  
#   
# Each review in our data may have more than one sentence. We'll split each sentence and create a list of sentences to pass it to gensim.

# In[ ]:


tmp_corpus = reviews['review'].map(lambda x:x.split('.'))


# In[ ]:


from tqdm import tqdm


# In[ ]:


#corpus [[w1, w2, w3,...],[...]]
corpus = []

for i in tqdm(range(len(reviews))):
    for line in tmp_corpus[i]:
        words = [x for x in line.split()]
        corpus.append(words)


# In[ ]:


len(corpus)


# In[ ]:


#removing blank list
corpus_new = []
for i in range(len(corpus)):
    if (len(corpus[i]) != 0):
        corpus_new.append(corpus[i])


# In[ ]:


# corpus[1:100]


# In[ ]:


num_of_sentences = len(corpus_new)
num_of_words = 0
for line in corpus_new:
    num_of_words += len(line)

print('Num of sentences - %s'%(num_of_sentences))
print('Num of words - %s'%(num_of_words))


# In[ ]:


from gensim.models import Word2Vec


# In[ ]:


# sg - skip gram |  window = size of the window | size = vector dimension
size = 100
window_size = 2 # sentences weren't too long, so
epochs = 100
min_count = 2
workers = 4

model = Word2Vec(corpus_new)


# In[ ]:


model.build_vocab(sentences= corpus_new, update=True)

for i in range(5):
    model.train(sentences=corpus_new, epochs=50, total_examples=model.corpus_count)
    


# In[31]:


#save model
model.save('w2v_model')


# In[32]:


model = Word2Vec.load('w2v_model')


# In[35]:


model.wv.most_similar('movie')


# In[39]:


reviews.head()


# In[40]:


reviews = reviews[["review", "label", "file"]].sample(frac=1,
                                                      random_state=1)
train = reviews[reviews.label!=-1].sample(frac=0.6, random_state=1)
valid = reviews[reviews.label!=-1].drop(train.index)
test = reviews[reviews.label==-1]


# In[108]:


print(train.shape)
print(valid.shape)
print(test.shape)


# In[109]:


valid.head()


# In[110]:


num_features = 100


# In[111]:


index2word_set = set(model.wv.index2word)


# In[112]:


model = model


# In[113]:


def featureVecorMethod(words):
    featureVec = np.zeros(num_features, dtype='float32')
    nwords = 0
    
    for word in words:
        if word in index2word_set:
            nwords+= 1
            featureVec = np.add(featureVec, model[word])
            
    #average of feature vec
    featureVec = np.divide(featureVec, nwords)
    return featureVec


# In[114]:


def getAvgFeatureVecs(reviews):
    counter = 0
    
    reviewFeatureVecs = np.zeros((len(reviews), num_features), dtype='float32')
    for review in reviews:
#         print(review)
        if counter%1000 == 0:
            print("Review %d of %d"%(counter, len(reviews)))
            
        reviewFeatureVecs[counter] = featureVecorMethod(review)
        counter = counter+1
    return reviewFeatureVecs


# In[115]:


clean_train_reviews = []
for review in train['review']:
#     print(review)
    clean_train_reviews.append(list(CleanData(review).split()))
# print(len(clean_train_reviews))\

trainDataVecs = getAvgFeatureVecs(clean_train_reviews)


# In[117]:


len(valid['review'])


# In[118]:


clean_test_reviews = []
for review in valid['review']:
#     print(review)
    clean_test_reviews.append(list(CleanData(review).split()))
# print(len(clean_train_reviews))\

testDataVecs = getAvgFeatureVecs(clean_test_reviews)


# In[121]:


print(len(print(len(testDataVecs))))
print(len(testDataVecs))


# In[122]:


import sklearn


# In[123]:


from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(n_estimators=100)

print("fitting data")
forest = forest.fit(trainDataVecs, train['label'])


# In[134]:


# valid.index


# In[124]:


result = forest.predict(testDataVecs)


# In[136]:


output = pd.DataFrame(data={"id":valid.index, "sentiment": result})


# In[141]:


from sklearn.metrics import accuracy_score


# In[142]:


accuracy_score(valid['label'], result)


# In[ ]:




