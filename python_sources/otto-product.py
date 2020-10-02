#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from sklearn.utils import shuffle


# ## Import Dataset

# In[ ]:


train_csv = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')


# In[ ]:


train_csv = shuffle(train_csv)
train_csv=train_csv


# ## Datasets:-

# ### Train Data:-

# In[ ]:


train_csv.head()


# # Visualization:-

# ## Count Plot:-
# This is number of outcome of each type

# In[ ]:


order = sorted(set(train_csv['target']))
sns.countplot(x='target', data=train_csv,order=order)
plt.grid()
plt.title("No of Product of Each Class")
plt.figure(num=None, figsize=(20, 30), dpi=80, facecolor='w', edgecolor='k')


# ## Weight Of Each Feature:-
# This is the sum of all the featues.

# In[ ]:


wt = train_csv.sum()
wt.drop(['target','id']).sort_values().plot(kind='barh', figsize=(15,20))
plt.grid()
plt.title("Weight Of Features")


# ## Correation Analysis:-

# In[ ]:


df = train_csv.drop(['id','target'],axis=1).corr()
sns.heatmap(df)
plt.title("Correation Analysis")


# ## Covariance Analysis:-

# In[ ]:


df.var().sort_values().plot(kind='barh', figsize=(15,20))
plt.grid()
plt.title("Covariance Analysis")


# ## Preprocessing:-

# In[ ]:


train_csv.describe()


# In[ ]:


X = train_csv
Y = train_csv['target']
del X['target']
del X['id']


# In[ ]:


X.describe()


# ### Label Encoding:-

# In[ ]:


from sklearn import preprocessing
le = preprocessing.LabelEncoder()
le.fit(Y.values.tolist())
label=le.transform(Y)
print(list(le.classes_))
print(label)


# ## Feature Extraction:-

# In[ ]:


noOfFeature = 45


# In[ ]:


from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
import timeit
start = timeit.default_timer()
clf = RandomForestClassifier()
rfe = RFE(clf, noOfFeature)
fit = rfe.fit(X, label)
print("Time take %.2f "%(timeit.default_timer()-start))
print(("Num Features: %d") % fit.n_features_)
print(("Selected Features: %s") % fit.support_)
print(("Feature Ranking: %s") % fit.ranking_)
features = []
for i , j in zip(X.columns,fit.support_):
    if j == True:
        features.append(str(i))


# In[ ]:


print(features)


# ## Model Creation:-

# In[ ]:


from sklearn.model_selection import cross_val_score
import timeit
from xgboost import XGBClassifier
from statistics import mean
train_csv = pd.read_csv('../input/train.csv')


# # Original Dataset:-

# In[ ]:


start = timeit.default_timer()
clf=XGBClassifier(base_score=0.5, colsample_bylevel=1, colsample_bytree=.8,subsample=0.5,
       gamma=0, learning_rate=0.1, max_delta_step=0, max_depth=5,
       missing=None, n_estimators=100, nthread=2,
       objective='multi:softprob', reg_alpha=0, reg_lambda=1,
       scale_pos_weight=1, seed=27, silent=True)
scores = cross_val_score(clf,X[features], label, cv=2)
print("Time take %.2f "%(timeit.default_timer()-start))
print("Accuracy: %0.4f (+/- %0.4f)" % (scores.mean(), scores.std() * 2))


# In[ ]:


xg = XGBClassifier(base_score=0.5, colsample_bylevel=1, colsample_bytree=.8,subsample=.2,
       gamma=0,learning_rate=0.1,max_delta_step= 4,max_depth=5,
       missing=None,n_estimators= 400,nthread=2,
       objective='multi:softprob', reg_alpha=0, reg_lambda=1,
       scale_pos_weight=1, seed=27,silent=1)
start = timeit.default_timer()
scores = cross_val_score(xg,X[features], label, cv=2)
print("Time take %.2f "%(timeit.default_timer()-start))
print("Accuracy: %0.4f (+/- %0.4f)" % (scores.mean(), scores.std() * 2))


# In[ ]:


start = timeit.default_timer()
xg.fit(X[features], label)
print("Time take to fit the data %.2f "%(timeit.default_timer()-start))


# In[ ]:


start = timeit.default_timer()
pre = xg.predict(test[features])
print("Time take predict output %.2f "%(timeit.default_timer()-start))

