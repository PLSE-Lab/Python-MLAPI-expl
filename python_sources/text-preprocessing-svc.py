#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# ### Feature Enginnering

# In[ ]:


train=pd.read_csv("/kaggle/input/nlp-getting-started/train.csv")
test=pd.read_csv("/kaggle/input/nlp-getting-started/test.csv")


# In[ ]:


train.head()


# In[ ]:


train = train[['text','target']]
test = test[['id','text']]


# In[ ]:


from nltk.stem import PorterStemmer
import nltk
from nltk.corpus import stopwords
stopwords = set(stopwords.words("english"))
import re


# Remove stopwords (or, are, is etc) from data 

# In[ ]:


train['text'] = train['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stopwords)]))
test['text'] = test['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stopwords)]))


# In[ ]:


corpus_train = train['text']
corpus_test = test['text']


# In[ ]:


def replace(text):
    text = text.str.replace(r'^.+@[^\.].*\.[a-z]{2,}$'," ")  # remove emailaddress
    text = text.str.replace(r'\W+'," ")     # remove symbols
    text = text.str.replace(r' '," ")       # remove punctuations
    text = text.str.replace('\d+'," ")      # remove numbers
    text = text.str.lower()                 # remove capital letters as they does not make any effect
    return text


# In[ ]:


corpus_train = replace(corpus_train)
corpus_test = replace(corpus_test)


# In[ ]:


import nltk
nltk.download('wordnet')
from textblob import Word


# Remove rare words from text that are not been used oftenly

# In[ ]:


freq = pd.Series(' '.join(corpus_train).split()).value_counts()[-19500:]
corpus_train = corpus_train.apply(lambda x: " ".join(x for x in x.split() if x not in freq))


# In[ ]:


freq.head()


# In[ ]:


freq = pd.Series(' '.join(corpus_test).split()).value_counts()[-10000:]
corpus_test = corpus_test.apply(lambda x: " ".join(x for x in x.split() if x not in freq))


# Visualise most occuring words from training corpus

# In[ ]:


from wordcloud import WordCloud 
import matplotlib.pyplot as plt
def wordcloud(text):
    wordcloud = WordCloud(
        background_color='white',
        max_words=500,
        max_font_size=30, 
        scale=3,
        random_state=5
    ).generate(str(corpus_train))
    fig = plt.figure(figsize=(15, 12))
    plt.axis('off')
    plt.imshow(wordcloud)
    plt.show()
    
wordcloud(corpus_train)


# In[ ]:


import seaborn as sns
target = train['target']
sns.countplot(target)


# Unlike humans, machine cannot understand raw text. Hence need to convert text into corresponding numerical form.<br>
# Tfidfvectorizer count each word occurence from document 

# In[ ]:


from sklearn.feature_extraction.text import TfidfVectorizer

Tfidf_vect = TfidfVectorizer(max_features = 7000)
Tfidf_vect.fit(corpus_train)
X_train = Tfidf_vect.transform(corpus_train)
X_test = Tfidf_vect.transform(corpus_test)


# ### Hyperparameter Tuning 
# Parameters are conditions or settings that are to be defined in models. These changing of parameters according to the need is called *Hyperparameter Tuning*. 
# Technically parameters passed within algorithm are not best parameters for every dataset. 
# Hence to choose the best parameters hyperparameter tuning is done
# Hyperparameter tuning are of two types *Grid SearchCV* and *Random SearchCV.*
#  
# *Grid Search* is the approach where every parameter is selected from grid list specified and tried on the model and the best one can be interpreted. We will use Grid Search approach in this problem.<br>
# Where in *Random Search*, search the parameter randomly sepcified randomly choosed according to the specifity of model with in range. 
# ![](https://i.stack.imgur.com/cIDuR.png)
# Above diagram states that Grid Search explores at every fix distinct places while Random Search has no such fix trials

# In[ ]:


from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
parameters = { 
    'gamma': [0.001, 0.01, 0.1, 0.4, 0.6, 0.7, 'auto'], # for complex decision boundary (mainly used for rbf kerel)
    
    'kernel': ['rbf','linear'], # used for different type of data
                                # linear - when data is easy to classify 
                                # rbf - when data is too complex
    
    'C': [0.001, 0.01, 0.1, 1, 1.5, 2, 3, 10], # inverse weight on regularization parameter 
                                               # (how finely to classify, decreasing will prevent overfititing and vice versa)
}
model = GridSearchCV(SVC(), parameters, cv=10, n_jobs=-1).fit(X_train, target)
model.cv_results_['params'][model.best_index_]
y_val_pred = model.predict(X_test)


# Above hyperparameter tuning is time consuming so putting the results directly we get,

# Here we will use **SVC (Support Vector Classifier)** <br>
# SVC aims to fit data that is provided with returning best fit hyperplane that divides <br>
# the data between classes while prediction helps to sort which class features belongs to.

# In[ ]:


from sklearn.svm import SVC
SVM = SVC(C=1.0, kernel='linear', gamma='auto')
SVM.fit(X_train,target)
SVM_predictions = SVM.predict(X_test)


# ### Prediction

# In[ ]:


file_submission = pd.read_csv("/kaggle/input/nlp-getting-started/sample_submission.csv")
file_submission.target = SVM_predictions
file_submission.to_csv("submission.csv", index=False)


# In[ ]:




