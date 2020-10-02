#!/usr/bin/env python
# coding: utf-8

# <h1>Natural Language Processing, EDA & Prediction of Tweets</h1>
# <img src='data:image/png;base64,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'>

# In[ ]:


import numpy as np 
import pandas as pd 
import nltk
from nltk.corpus import stopwords
import string as s
import re
from textblob import TextBlob
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, GridSearchCV, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from xgboost import XGBClassifier
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# <h1>Load Dataset & Get an informations about it</h1>

# In[ ]:


train_df=pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')
test_df=pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')
def data_info(d):
    print('number of variables: ',d.shape[1])
    print('number of tweets: ',d.shape[0])
    print('variables names: ')
    print(d.columns)
    print('variables data-types: ')
    print(d.dtypes)
    print('missing values: ')
    c=d.isnull().sum()
    print(c[c>0])


# In[ ]:


data_info(train_df)


# In[ ]:


data_info(test_df)


# <h1>EDA</h1>

# In[ ]:


plt.figure(figsize=(8,8))
train_df['target'].value_counts().plot.pie(autopct='%.2f%%')
plt.title('Disaster or Not Distribution')
plt.ylabel('')
plt.show()


# In[ ]:


train_df[train_df['target']==1].loc[:4,'text']


# In[ ]:


train_disasters={'earthquake':0,'fire':0,'fires':0,'shelter':0}
test_disasters={'earthquake':0,'fire':0,'fires':0,'shelter':0}
for i,j in zip(train_df['text'],test_df['text']):
    if 'earthquake' in i.split():
        train_disasters['earthquake']+=1
    elif 'earthquake' in j.split():
        test_disasters['earthquake']+=1
    if 'fire' in i.split():
        train_disasters['fire']+=1
    elif 'fire' in j.split():
        test_disasters['fire']+=1
    if 'fires' in i.split():
        train_disasters['fires']+=1
    elif 'fires' in j.split():
        test_disasters['fires']+=1
    if 'shelter' in i.split():
        train_disasters['shelter']+=1
    elif 'shelter' in j.split():
        test_disasters['shelter']+=1


# In[ ]:


print('number of tweets that fires,fire and earthquacke mentioned in train data: ',train_disasters)
print('number of tweets that fires,fire and earthquacke mentioned in test data: ',test_disasters)


# <h1>Text Preprocessing</h1>

# In[ ]:


def tokenization(text):
    lst=text.split()
    return lst
train_df['text']=train_df['text'].apply(tokenization)
test_df['text']=test_df['text'].apply(tokenization)


# In[ ]:


def lowercasing(lst):
    new_lst=[]
    for i in lst:
        i=i.lower()
        new_lst.append(i)
    return new_lst
train_df['text']=train_df['text'].apply(lowercasing)
test_df['text']=test_df['text'].apply(lowercasing)    


# In[ ]:


def remove_punctuations(lst):
    new_lst=[]
    for i in lst:
        for j in s.punctuation:
            i=i.replace(j,'')
        new_lst.append(i)
    return new_lst
train_df['text']=train_df['text'].apply(remove_punctuations)
test_df['text']=test_df['text'].apply(remove_punctuations)            


# In[ ]:


def remove_numbers(lst):
    nodig_lst=[]
    new_lst=[]
    for i in lst:
        for j in s.digits:    
            i=i.replace(j,'')
        nodig_lst.append(i)
    for i in nodig_lst:
        if i!='':
            new_lst.append(i)
    return new_lst
train_df['text']=train_df['text'].apply(remove_numbers)
test_df['text']=test_df['text'].apply(remove_numbers)     


# In[ ]:


def remove_stopwords(lst):
    stop=stopwords.words('english')
    new_lst=[]
    for i in lst:
        if i not in stop:
            new_lst.append(i)
    return new_lst
train_df['text']=train_df['text'].apply(remove_stopwords)
test_df['text']=test_df['text'].apply(remove_stopwords)  


# In[ ]:


def remove_spaces(lst):
    new_lst=[]
    for i in lst:
        i=i.strip()
        new_lst.append(i)
    return new_lst
train_df['text']=train_df['text'].apply(remove_spaces)
test_df['text']=test_df['text'].apply(remove_spaces)  


# In[ ]:


'''
def correct_spelling(lst):
    new_lst=[]
    for i in lst:
        i=TextBlob(i).correct()
        new_lst.append(i)
    return new_lst
train_df['text']=train_df['text'].apply(correct_spelling)
test_df['text']=test_df['text'].apply(correct_spelling)  
'''


# In[ ]:


lemmatizer=nltk.stem.WordNetLemmatizer()
def lemmatzation(lst):
    new_lst=[]
    for i in lst:
        i=lemmatizer.lemmatize(i)
        new_lst.append(i)
    return new_lst
train_df['text']=train_df['text'].apply(lemmatzation)
test_df['text']=test_df['text'].apply(lemmatzation)  


# <h1>Converting Text to Features</h1>

# <h2>Count Vectorizing</h2>

# In[ ]:


train_df['text']=train_df['text'].apply(lambda x: ''.join(i+' ' for i in x))
test_df['text']=test_df['text'].apply(lambda x: ''.join(i+' ' for i in x))


# In[ ]:


train_df.head()


# In[ ]:


from sklearn.feature_extraction.text import CountVectorizer
vec=CountVectorizer(ngram_range=(1,2))
train_1=vec.fit_transform(train_df['text'])
test_1=vec.transform(test_df['text'])


# <h1>TF-IDF</h1>

# In[ ]:


from sklearn.feature_extraction.text import TfidfVectorizer
tfidf=TfidfVectorizer(ngram_range=(1,2))
train_2=tfidf.fit_transform(train_df['text'])
test_2=tfidf.transform(test_df['text'])


# In[ ]:


def select_model(x,y):
    models=[{
        'name':'Multinomial Naive Bayes',
        'estimator':MultinomialNB(),
        'hyperparameters':{
            'alpha':np.arange(0.1,1,0.01)
        }}]
    for i in models:
        print(i['name'])
        gs=GridSearchCV(i['estimator'], param_grid=i['hyperparameters'], cv=10, scoring='f1')
        gs.fit(x,y)
        print(gs.best_score_)
        print(gs.best_params_)
select_model(train_2, train_df['target'])


# <h1>Prediction</h1>

# In[ ]:


clf_NB=MultinomialNB()
clf_NB.fit(train_2,train_df['target'])
pred=clf_NB.predict(test_2)
pred[:15]


# <h1>Submission</h1>

# In[ ]:


submission_df = {"id":test_df['id'],
                 "target":pred}
submission = pd.DataFrame(submission_df)
submission.to_csv('submission_df.csv',index=False)

