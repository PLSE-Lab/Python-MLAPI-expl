#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import re
import string
from sklearn.linear_model import LogisticRegression


# In[ ]:



from keras.preprocessing.text import Tokenizer
from keras.utils.vis_utils import plot_model
from keras.models import Sequential
from keras.layers import Dense


# In[ ]:


from collections import Counter


# In[ ]:



from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
vocab = Counter()


# In[ ]:


train  = pd.read_csv('../input/nlp-getting-started/train.csv')
test  = pd.read_csv('../input/nlp-getting-started/test.csv')


# In[ ]:


#train[train['target']==1]['keyword'].value_counts()


# In[ ]:


train = train.drop(['keyword','location'],axis=1)
test = test.drop(['keyword','location'],axis=1)


# In[ ]:


test


# In[ ]:


train['text']=train['text'].map(lambda x : x.split())
test['text']=test['text'].map(lambda x : x.split())


# In[ ]:



train['text'] = train['text'].map(lambda x:  [word for word in x if not word.startswith(('http','@'))])
re_punc = re.compile('[%s]'%re.escape(string.punctuation))
train['text'] = train['text'].map(lambda x:  [re_punc.sub('',w) for w in x])
train['text'] = train['text'].map(lambda x: [word for word in x if word.isalpha()])
train['text'] = train['text'].map(lambda x: [w.lower() for w in x])    
train['text'] = train['text'].map(lambda x:  [w for w in x if not w in stop_words ]) 
train['text'] = train['text'].map(lambda x: [w for w in x if len(w)>1])


# In[ ]:



test['text'] = test['text'].map(lambda x:  [word for word in x if not word.startswith(('http','@'))])
re_punc = re.compile('[%s]'%re.escape(string.punctuation))
test['text'] = test['text'].map(lambda x:  [re_punc.sub('',w) for w in x])
test['text'] = test['text'].map(lambda x: [word for word in x if word.isalpha()])
test['text'] = test['text'].map(lambda x: [w.lower() for w in x])    
test['text'] = test['text'].map(lambda x:  [w for w in x if not w in stop_words ]) 
test['text'] = test['text'].map(lambda x: [w for w in x if len(w)>1])


# In[ ]:


for line in train['text']:
    vocab.update(line)


# In[ ]:


#new_voc = vocab.copy()
#for key,val in vocab.items():
 #   if val<5:
  #      del new_voc[key]


# In[ ]:


o = list(vocab)


# In[ ]:


line = list(train['text'][0])
line


# In[ ]:


count =0
arr = []
for line in train['text']:
    vec = np.zeros(len(vocab))
    for word in line:
        
        if vocab[word]!=0:
            vec[o.index(word)]=(line.count(word)/len(line))*np.log(len(train)/vocab[word])
    arr.append(vec)
    count+=1
        #df.append(list(vec))
X_train = np.array(arr)


# In[ ]:


count =0
arr = []
for line in test['text']:
    vec = np.zeros(len(vocab))
    for word in line:
        
        if vocab[word]!=0:
            vec[o.index(word)]=(line.count(word)/len(line))*np.log(len(test)/vocab[word])
    arr.append(vec)
    count+=1
        #df.append(list(vec))
X_test = np.array(arr)


# In[ ]:


X_train


# In[ ]:


X_train.shape


# In[ ]:


X_train[0:5,0:5]


# In[ ]:


y_train = list(train['target'])
y_train[0:5]


# In[ ]:


from sklearn.model_selection import GridSearchCV


# In[ ]:


param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000] }
GridSearchCV(cv=None,estimator=LogisticRegression(C=1.0, intercept_scaling=1,   
               dual=False, fit_intercept=True, penalty='l2', tol=0.0001),
             param_grid={'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]})
clf = GridSearchCV(LogisticRegression(penalty='l2',max_iter=1000), param_grid,)


# In[ ]:


clf.fit(X_train,y_train)
pred = clf.predict(X_test)
print(clf)
print(pred[0:5])


# In[ ]:


sub = []
for w in pred:
    if w==1:
        sub.append(1)
    else :
        sub.append(0)


# In[ ]:


sub[0:10]


# In[ ]:


df = pd.read_csv('../input/nlp-getting-started/sample_submission.csv')
df['target']=sub


# In[ ]:


df.to_csv('submission.csv',index=False)

