#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import seaborn as sns
import os
print(os.listdir("../input"))
from tqdm import tqdm_notebook as tqdm

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import (RandomForestClassifier,GradientBoostingClassifier,AdaBoostClassifier)

from sklearn.metrics import accuracy_score
import warnings 
warnings.filterwarnings("ignore") 
pd.set_option('display.max_colwidth', 200)
# Any results you write to the current directory are saved as output.


# ## Reading Data : 

# In[ ]:


df = pd.read_csv('../input/imdb_master.csv', encoding='latin')
df.head()


# In[ ]:


df.drop(['Unnamed: 0', 'file'], axis=1, inplace = True)


# In[ ]:


df.head()


# In[ ]:


df['label'].unique()


# In[ ]:


df['label'].value_counts()


# In[ ]:


# Taking only positive and negative datapoints. 
df = df[df['label'] != 'unsup']


# In[ ]:


df['label'] = df['label'].map({'neg' : 0, 'pos' : 1})


# In[ ]:


train = df[df['type'] == 'train'].drop(['type'], axis= 1)
test  = df[df['type'] == 'test'].drop(['type'], axis = 1)


# In[ ]:


train.head()


# ## Text Pre-processing 

# In[ ]:


import re
def decontracted(phrase):
     # specific    
    phrase = re.sub(r"won't", "will not", phrase)    
    phrase = re.sub(r"can\'t", "can not", phrase) 
    # general    
    phrase = re.sub(r"n\'t", " not", phrase)  
    phrase = re.sub(r"\'re", " are", phrase)  
    phrase = re.sub(r"\'s", " is", phrase)    
    phrase = re.sub(r"\'d", " would", phrase)  
    phrase = re.sub(r"\'ll", " will", phrase)   
    phrase = re.sub(r"\'t", " not", phrase)   
    phrase = re.sub(r"\'ve", " have", phrase)   
    phrase = re.sub(r"\'m", " am", phrase)   
    return phrase 


# In[ ]:


# https://gist.github.com/sebleier/554280 
# we are removing the words from the stop words list: 'no', 'nor', 'not'
stopwords= ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're" , "you've",            "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself',             'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'th ey', 'them', 'their',            'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "tha t'll", 'these', 'those',             'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'ha d', 'having', 'do', 'does',             'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of',             'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after',            'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'ove r', 'under', 'again', 'further',            'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any' , 'both', 'each', 'few', 'more',            'most', 'other', 'some', 'such', 'only', 'own', 'same', 'so', 'than', 'too', 'very',             's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'no w', 'd', 'll', 'm', 'o', 're',             've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn',            "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'might n', "mightn't", 'mustn',            "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wa sn', "wasn't", 'weren', "weren't",             'won', "won't", 'wouldn', "wouldn't"] 


# In[ ]:


def preprocess_text(dff, feature):   
       preprocessed_text = []   
       for sentance in tqdm(dff[feature].values):       
           sent = decontracted(sentance)      
           sent = sent.replace('\\r', ' ')     
           sent = sent.replace('\\"', ' ')     
           sent = sent.replace('\\n', ' ')     
           sent = re.sub('[^A-Za-z0-9]+', ' ', sent)      
           # https://gist.github.com/sebleier/554280     
           sent = ' '.join(e for e in sent.split() if e not in stopwords)    
           preprocessed_text.append(sent.lower().strip()) 
       return preprocessed_text 


# In[ ]:


train_prep_reviews = preprocess_text(train, 'review')
test_prep_reviews = preprocess_text(test, 'review')


# ## **Vectorizing Text Data**

# **1. Bag of Words**

# In[ ]:


vectorizer = CountVectorizer(min_df = 10)
train_bow = vectorizer.fit_transform(train_prep_reviews)
test_bow = vectorizer.transform(test_prep_reviews)
print("Shape of train matrix after BOW : ",train_bow.shape)
print("Shape of test matrix after BOW : ",test_bow.shape) 


# **2. Tf-Idf**

# In[ ]:


vectorizer = TfidfVectorizer(min_df = 10)
train_tfidf = vectorizer.fit_transform(train_prep_reviews)
test_tfidf = vectorizer.transform(test_prep_reviews)
print("Shape of train matrix after Tfidf : ",train_tfidf.shape)
print("Shape of test matrix after Tfidf : ",test_tfidf.shape) 


# ## Model on BOW

# In[ ]:


X_tr = train_bow
y_tr = train['label']
X_ts = test_bow
y_ts = test['label']


# In[ ]:


print(X_tr.shape)
print(y_tr.shape)
print(X_ts.shape)
print(y_ts.shape)


# In[ ]:





# In[ ]:


models = [RandomForestClassifier(random_state=77),
          GradientBoostingClassifier(random_state=77),
          AdaBoostClassifier(random_state=77)]

from sklearn.model_selection import cross_val_score, GridSearchCV

for model in models:
    score = cross_val_score(model, X_tr, y_tr, cv=5)
    msg = ("{0}:\n\tMean accuracy on development set\t= {1:.3f} "
           "(+/- {2:.3f})".format(model.__class__.__name__,
                                  score.mean(),
                                  score.std()))
    print(msg)
    
    # Fit the model on the dev set and predict and eval independent set
    model.fit(X_tr, y_tr)
    pred_eval = model.predict(X_ts)
    acc_eval = accuracy_score(y_ts, pred_eval)
    print("\tAccuracy on evaluation set\t\t= {0:.3f}".format(acc_eval))


# ## Model on Tf-idf

# In[ ]:


X_tr = train_tfidf
y_tr = train['label']
X_ts = test_tfidf
y_ts = test['label']


# In[ ]:


print(X_tr.shape)
print(y_tr.shape)
print(X_ts.shape)
print(y_ts.shape)


# In[ ]:


models = [RandomForestClassifier(random_state=77),
          GradientBoostingClassifier(random_state=77),
          AdaBoostClassifier(random_state=77)]

from sklearn.model_selection import cross_val_score, GridSearchCV

for model in models:
    score = cross_val_score(model, X_tr, y_tr, cv=5)
    msg = ("{0}:\n\tMean accuracy on development set\t= {1:.3f} "
           "(+/- {2:.3f})".format(model.__class__.__name__,
                                  score.mean(),
                                  score.std()))
    print(msg)
    
    # Fit the model on the dev set and predict and eval independent set
    model.fit(X_tr, y_tr)
    pred_eval = model.predict(X_ts)
    acc_eval = accuracy_score(y_ts, pred_eval)
    print("\tAccuracy on evaluation set\t\t= {0:.3f}".format(acc_eval))

