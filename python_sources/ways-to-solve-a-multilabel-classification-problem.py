#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd 
from sklearn.pipeline import Pipeline
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer


# In[2]:


train_data=pd.read_csv("../input/train.csv")
test_data=pd.read_csv("../input/test.csv")
final_out=pd.read_csv("../input/sample_submission.csv")


# In[3]:


train_data.head()


# The problem is a multi label problem:
# 
# **What is a multilabel problem?**
# 
# A multilabel problem is a classificatoin problem in which the input can be mapped into different classes. This can be better considered with the movie genres. We would have seen many labels for the movies in theatres. A single movie will be classified as 'Romance", "Comedy" genres at the same time.
# 
# **How to Solve the Problem?**
# We can approach this in 3 ways:
# - one v/s All
# - one v/s one
# - Error Correcting Output Codes

# **One v/s All**
# 
# This strategy, also known as one-vs-all, is implemented in OneVsRestClassifier. The strategy consists in fitting one classifier per class. For each classifier, the class is fitted against all the other classes. Advantage of this approach is its interpretability.
# Since each class is represented by one and only one classifier, it is possible to gain knowledge about the class by inspecting its corresponding classifier. This is the most commonly used strategy and is a fair default choice.
# So we shall go by this method
# 
# The classifier we are using here is a *Linear Support Vector Machines:*
# 
# we can try other SVM variants like: ["c-svm","nu-svm","Bound Constraint c-svm","weston watkins multiclass svm"] and more
# 
# 

# **Embeddings**
# 
# LIke any other machine learning algorithms these classifiers cannot work on the texts. so we need to find a way to convert thetexts into numerical equivalents
# We have two ways to do that :
# - TFIDF
# - Word2Vec
# 
# I think these two terminologies are much familar to every1. hence we can see a way to implement them
# 

# **Model 1: Word2Vec with OneV/s All**
# ![](http://)

# *Step 1 :Generate the word embeddings*
# 
#  we can generate word embeddings for all the words in the text and the training and the test set

# In[4]:


combined_df=train_data['comment_text'].append(test_data['comment_text'])


# In[5]:


from gensim.models import Word2Vec
Vocab_list=(combined_df.apply(lambda x:str(x).strip().split())
                         )
models=Word2Vec(Vocab_list,size=100)

WordVectorz=dict(zip(models.wv.index2word,models.wv.vectors))


# **Fit these things into a pipeline**
# 
# We need to make a class which implements the fit and the transform method(as per scikit learns Pipeline class documentation)
# So how should we transform a sentence representation to fit the pipeline. 
# 
# The easiest transformation is to move from a word embedding level to a sentence level embedding:
# 
# **Sentence level embedding:**
# Calculate average of all the word vectors that make up the sentence and  this average vector will be a repressentation for the sentence.

# In[6]:


class AverageEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        # if a text is empty we should return a vector of zeros
        # with the same dimensionality as all the other vectors
        self.dim =100 # because we use 100 embedding points 

    def fit(self, X, y):
        return self

    def transform(self, X):
        return np.array([
            np.mean([self.word2vec[w] for w in words if w in self.word2vec]
                    or [np.zeros(self.dim)], axis=0)
            for words in X
        ])


# In[ ]:


pipe1=Pipeline([("wordVectz",AverageEmbeddingVectorizer(WordVectorz)),("multilabel",OneVsRestClassifier(LinearSVC(random_state=0)))])


# In[10]:


y_train=train_data[[i for i in train_data.columns if i not in ["comment_text","id"]]]


# In[ ]:


pipe1.fit(train_data['comment_text'],y_train)


# **Prediction and Submission File Generation Model 1**
# 

# In[ ]:


predicted1=pipe1.predict(test_data['comment_text'])
label_cols=train_data.columns[2:]
submid = pd.DataFrame({'id': final_out["id"]})
submission = pd.concat([submid, pd.DataFrame(predicted1, columns = label_cols)], axis=1)
submission.to_csv('submission_W2v_m1.csv', index=False)


# **Model -2:TFiDf with OneV/s All**

# In[16]:


pipe2=Pipeline([('TFidf',TfidfVectorizer()),("multilabel",OneVsRestClassifier(LinearSVC(random_state=0)))])


# In[17]:


pipe2.fit(train_data['comment_text'],y_train)


# In[20]:


predicted2=pipe2.predict(test_data['comment_text'])
label_cols=train_data.columns[2:]
submid = pd.DataFrame({'id': final_out["id"]})
submission = pd.concat([submid, pd.DataFrame(predicted2, columns = label_cols)], axis=1)
submission.to_csv('submission_TfIdf_m1.csv', index=False)


# **Conculsion  from Model1 and Model2:**
# 
# Model 1 LB_score: 0.563
# Model2 LB_score:0.754
# 
# These approaches provide a decent begining to  a multilabel classification problems.
# 

# In[ ]:




