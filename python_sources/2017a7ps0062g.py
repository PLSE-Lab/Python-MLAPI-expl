#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


cd /kaggle/input/eval-lab-1-f464-v2


# In[ ]:


df=pd.read_csv("train.csv")


# In[ ]:


df.head(10)


# In[ ]:


df.isnull().any().any()


# In[ ]:


df.isnull().sum()


# In[ ]:


df.describe()


# In[ ]:


df.info()


# In[ ]:


df1 = df[df.isna().any(axis=1)]


# In[ ]:


df1


# In[ ]:


df.head()


# In[ ]:


df.fillna(value=df.mean(),inplace=True)


# In[ ]:


df.isnull().any().any()


# In[ ]:


df.corr()


# In[ ]:


X_encoded = pd.get_dummies(df['type'])
X_encoded.head()


# In[ ]:


df['new'] = X_encoded['new']
df['old'] = X_encoded['old']


# In[ ]:


Selected_features = ['id','feature1','feature2','feature3','feature4','feature5','feature6','feature7','feature8','feature9','feature10','feature11','new','old']


# In[ ]:


#cols = df.columns.tolist()


# In[ ]:


#cols = cols[:-2] + cols[-1:] +cols[-2:-1] 


# In[ ]:


#df.columns = cols
#df.head()


# In[ ]:


df.drop(['type'], axis=1).head()


# In[ ]:


df.corr()


# In[ ]:


test = pd.read_csv('test.csv')
y = df['rating'].copy()
#x_test = test[Selected_features]


# In[ ]:


from sklearn import ensemble
from sklearn.metrics import mean_squared_error
params = {'n_estimators': 500, 'max_depth': 4, 'min_samples_split': 2,
          'learning_rate': 0.01, 'loss': 'ls'}
clf = ensemble.GradientBoostingRegressor(**params)
X_train = df[Selected_features]
y_train = df['rating']
clf.fit(X_train, y_train)


# In[ ]:


test.fillna(value=test.mean(),inplace=True)
X_encoded = pd.get_dummies(test['type'])
X_encoded.head()


# In[ ]:


test['new'] = X_encoded['new']
test['old'] = X_encoded['old']


# In[ ]:


test = test.drop(['type'], axis=1)
test.corr()


# In[ ]:


test.isnull().any().any()
selected = ['id','feature1','feature2','feature3','feature4','feature5','feature6','feature7','feature8','feature9','feature10','feature11','new','old']
rating = clf.predict(test[selected])


# In[ ]:


import sys
rating = np.round(rating)
np.set_printoptions(threshold=sys.maxsize)
rating.astype(int)


# In[ ]:


df_ans = pd.DataFrame()
test = pd.read_csv('test.csv')
df_ans['id'] = test['id']
df_ans['rating'] = rating


# In[ ]:


#df_ans.to_csv ('/kaggle/Output/2017A7PS0062G.csv')
get_ipython().system('cd /kaggle')


# In[ ]:


get_ipython().system('mkdir /kaggle/Output')


# In[ ]:


df_ans.to_csv ('/kaggle/Output/2017A7PS0062G.csv')

