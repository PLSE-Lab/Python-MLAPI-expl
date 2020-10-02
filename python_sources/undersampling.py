#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
from numpy import mean
plt.style.use('ggplot')


# ### In this notebook I will train a model on an imbalanced dataset by undersampling the data

# In[ ]:


data = pd.read_csv('../input/creditcardfraud/creditcard.csv')


# In[ ]:


print(data)


# In[ ]:


data.isnull().sum()


# In[ ]:


sb.countplot(x='Class', data=data)


# ### Here you can see our data is very imbalanced
# Training models on inbalanced data can result in overfitting. If our model just guessed that every transaction was a legitiment transaction, it would have a high accuracy. To combat this, we will undersample our data.

# In[ ]:


data.Class[data.Class == 1].count()


# In[ ]:


data.Class[data.Class == 0].count()


# First lets scale our time and amount features

# In[ ]:


from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()
data['Amount'] = scaler.fit_transform(data['Amount'].values.reshape(-1,1))
data['Time'] = scaler.fit_transform(data['Time'].values.reshape(-1,1))


# In[ ]:


data.Time.describe()


# ### Now lets split our data. We will use 80% of our data to train on. We do this before undersampling so we don't train on data we will be testing on

# In[ ]:


train=data.sample(frac=0.8,random_state=200)
test=data.drop(train.index)


# ### Now we undersample our negatives so we have the same amount of positive and negative fradulent transactions 

# In[ ]:


positives = train[train.Class == 1]
negatives = train[train.Class == 0]


# In[ ]:


negativeSample = negatives.sample(positives.Class.count())


# In[ ]:


df = pd.concat([negativeSample, positives], axis=0)


# In[ ]:


sb.countplot(x='Class', data=df)


# ### Now that our data is no longer inbalanced, we can start training

# In[ ]:



X = df.drop('Class', axis=1)
y = df['Class']


# In[ ]:


from sklearn.linear_model import LogisticRegression
clf = LogisticRegression()
clf.fit(X, y)


# ### Note: it is important to train on the original dataset, not the undersampled dataset. Also, since we split our data before undersampling it, the data we trained on is not a subset of the data we test on

# In[ ]:


X = test.drop('Class', axis=1)
y = test['Class']
score = clf.score(X, y)
print(score)


# ### Hope I did everything correct, leave a comment with advice or if I messed something up. Toodles
