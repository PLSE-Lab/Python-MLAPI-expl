#!/usr/bin/env python
# coding: utf-8

# In this notebook we are going to build a simple model for **Real or Not? NLP with Disaster Tweets Competition** that got me roughly .8 on the leaderboard but if you want to go higher I'll suggest using BERT, there are tons of great notebooks on kaggle that implements BERT and what I'd reccommend is reading about this model online then take a look at the implementation, anyways let's get started and what we are going to do here is just vectorizing sentences using spacy then give the vectorized sentences to a fine tuned LightGBM model. 

# ![](http://)**What is spaCy ?**
# spaCy is a free open-source library for Natural Language Processing in Python. It features NER, POS tagging, dependency parsing, word vectors and more. but in this notebook we are interested in only text vectorization using spaCy

# **What is LightGBM ? ** It's an open-source Tree Based Gradient Boosting Framework.It was developed by Microsoft, it's pretty popular along with CatBoost and XGBoost that uses roughly same methodology

# In[ ]:


import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from tqdm import tqdm


# In[ ]:


import spacy
spacy_model = spacy.load('en_core_web_lg') # 'lg' means large and 'en' means english so we are saying import the large spacy english model
#there is small model too just change 'lg' to 'sm', note that the large model vectorize the text to 300d whereas the small model vectorize the text to 96d


# In[ ]:


train = pd.read_csv('../input/nlp-getting-started/train.csv') #load the data
test = pd.read_csv('../input/nlp-getting-started/test.csv')
sb = pd.read_csv('../input/nlp-getting-started/sample_submission.csv')


# In[ ]:


spacy_vectorizer = lambda text: spacy_model(text).vector # vectorize the data
vectorized_train_documents = []
for i in tqdm(range(train.shape[0])):
    vectorized_train_documents.append(spacy_vectorizer(train.iloc[i].text.lower()))
print('Vectorizing the Training documents is DONE!')

vectorized_test_documents = []
for i in tqdm(range(test.shape[0])):
    vectorized_test_documents.append(spacy_vectorizer(test.iloc[i].text.lower()))
print('Vectorizing the Testing documents is DONE!')


# In[ ]:


xtrain_spacy = np.array(vectorized_train_documents) #put the data in the right format
xtest_spacy = np.array(vectorized_test_documents)
ytrain = train.target.values.reshape(-1, 1)


# In[ ]:


lgbmc = LGBMClassifier(learning_rate=.05, n_estimators=150,) # train the model
lgbmc.fit(xtrain_spacy, ytrain)


# In[ ]:


import os #save the predictions on the test set
os.chdir('/kaggle/working/')
sb['target'] = lgbmc.predict(xtest_spacy)
sb.to_csv('SubmitMe!.csv', index=False)


# <center>**Stay home and Machine Learn :))**</center>
