#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[ ]:


dir = '/kaggle/input/nlp-getting-started'
print(dir)


# In[ ]:


train_master = pd.read_csv(os.path.join(dir,'train.csv'))
test_master = pd.read_csv(os.path.join(dir,'test.csv'))


# In[ ]:


print(train_master.head())


# In[ ]:


train = train_master.copy()
test = test_master.copy()


# In[ ]:


train['keyword'].unique()


# In[ ]:


print(len(train['keyword'].unique()))
print(len(test['keyword'].unique()))


# In[ ]:


train['keyword'].loc[train['keyword'].notna()] = train['keyword'].loc[train['keyword'].notna()].apply(lambda x: ' '.join(x.split('%20')))
test['keyword'].loc[test['keyword'].notna()] = test['keyword'].loc[test['keyword'].notna()].apply(lambda x: ' '.join(x.split('%20')))


# In[ ]:


print(train.shape)
split = train.shape[0]
temp = train.append(test, sort=False)


# In[ ]:


print(temp.head())


# In[ ]:


from nltk.stem import PorterStemmer 
from nltk.tokenize import word_tokenize 
   
ps = PorterStemmer()

temp['keyword'].loc[temp['keyword'].notna()] = temp['keyword'].loc[temp['keyword'].notna()].apply(lambda x: ps.stem(x))


# In[ ]:


train['keyword'] = temp['keyword'][:split]
test['keyword'] = temp['keyword'][split:]


# In[ ]:


print(train.shape)
print(test.shape)


# In[ ]:


print(len(train['keyword'].unique()))
print(len(test['keyword'].unique()))


# In[ ]:


print(set(train['keyword']).symmetric_difference(set(test['keyword'])))


# In[ ]:


print(set(train['keyword']) - set(test['keyword']))


# In[ ]:


print(set(test['keyword']) - set(train['keyword']))


# In[ ]:


print(train['keyword'].unique())


# In[ ]:


fig, ax = plt.subplots(figsize=(40,8))
sns.countplot(x="keyword", hue="target", data=train, ax=ax)
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
fig.show()


# In[ ]:


train['keyword'] = np.where(train['keyword'].isna(),'None',train['keyword'])
test['keyword'] = np.where(test['keyword'].isna(),'None',test['keyword'])


# In[ ]:


# from sklearn.feature_extraction.text import TfidfVectorizer

# vectorizer = TfidfVectorizer()

# train['keyword'] = vectorizer.fit_transform(train['keyword']).toarray()
# test['keyword'] = vectorizer.fit_transform(test['keyword']).toarray()


# In[ ]:


X_train = train['keyword']
y_train = train['target']

X_test = test['keyword']


# In[ ]:


temp = X_train.append(X_test)
temp = pd.get_dummies(temp)

print(temp.shape)


# In[ ]:


X_train = temp.iloc[:7613]
X_test = temp.iloc[7613:]


# In[ ]:


print(X_train.shape)
print(y_train.shape)


# In[ ]:


from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()

gnb.fit(X_train, y_train)


# In[ ]:


gnb.score(X_train, y_train)


# In[ ]:


predictions = gnb.predict(X_test)


# In[ ]:


submission = pd.DataFrame()
submission['id'] = test['id']


# In[ ]:


submission['target'] = predictions


# In[ ]:


print(submission.head())


# In[ ]:


submission.to_csv('../working/submission.csv', encoding='utf-8', index=False)


# In[ ]:




