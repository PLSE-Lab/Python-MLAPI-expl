#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


df_train = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')


# In[ ]:


df_train.head()


# In[ ]:


df_test.head()


# In[ ]:


df_train.info()


# In[ ]:


df_train.describe()


# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


# In[ ]:


df_train, df_cv = train_test_split(df_train, test_size = 0)


# In[ ]:


df_trainy = df_train['target']
df_trainx = df_train.drop(['target', 'ID'], axis = 1)


# In[ ]:


#df_cvy = df_cv['target']
#df_cvx = df_cv.drop(['target', 'ID'], axis = 1)


# In[ ]:


from sklearn.decomposition import PCA


# In[ ]:


pca = PCA(n_components=512)


# In[ ]:


pca.fit(df_trainx)


# In[ ]:


plt.plot(pca.explained_variance_ratio_)
plt.show()


# In[ ]:


plt.plot(pca.singular_values_)
plt.show()


# In[ ]:


print(np.sum(pca.explained_variance_ratio_))


# In[ ]:


df_trainx = pd.DataFrame(pca.transform(df_trainx))
#df_cvx = pd.DataFrame(pca.transform(df_cvx))
df_trainx.head()


# In[ ]:


from sklearn.linear_model import LinearRegression as Lr
from sklearn.ensemble import RandomForestRegressor as Rfr
from sklearn.ensemble import GradientBoostingRegressor as Gbr
from xgboost import XGBRegressor as Xgb


# In[ ]:


lr = Lr(normalize=True)


# In[ ]:


lr.fit(df_trainx, df_trainy)


# In[ ]:


lr.score(df_trainx, df_trainy)


# In[ ]:


rf = Rfr()


# In[ ]:


rf.fit(df_trainx, df_trainy)


# In[ ]:


rf.score(df_trainx, df_trainy)


# In[ ]:


#rf.score(df_cvx, df_cvy)


# In[ ]:


gbr = Gbr()


# In[ ]:


gbr.fit(df_trainx, df_trainy)


# In[ ]:


gbr.score(df_trainx, df_trainy)


# In[ ]:


#gbr.score(df_cvx, df_cvy)


# In[ ]:


xgb = Xgb()


# In[ ]:


xgb.fit(df_trainx, df_trainy)


# In[ ]:


xgb.score(df_trainx, df_trainy)


# In[ ]:


#xgb.score(df_cvx, df_cvy)


# In[ ]:


submit = pd.DataFrame(df_test['ID'])


# In[ ]:


df_test = df_test.drop(['ID'], axis = 1)


# In[ ]:


df_test.shape


# In[ ]:


df_test = pd.DataFrame(pca.transform(df_test))


# In[ ]:


sub01 =pd.Series(np.abs((gbr.predict(df_test) + xgb.predict(df_test) + lr.predict(df_test) + rf.predict(df_test)) / 4), name='target')


# In[ ]:


submit = submit.join(sub01)


# In[ ]:


submit.to_csv('../working/submit01.csv', index=False)


# In[ ]:


submit.drop(['target'], axis = 1, inplace=True)


# In[ ]:


sub02 = rf.predict(df_test)


# In[ ]:


submit = submit.join(sub02)


# In[ ]:


submit.to_csv('../working/submit02.csv', index=False)


# In[ ]:





# In[ ]:




