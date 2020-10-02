#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.preprocessing import MinMaxScaler


# In[ ]:


dftest = pd.read_csv('/kaggle/input/eval-lab-1-f464-v2/test.csv')
# dftest.head()


# In[ ]:


df = pd.read_csv('/kaggle/input/eval-lab-1-f464-v2/train.csv')
df.head(5)


# In[ ]:


def enc(x):
    if x == "new":
        return 0;
    elif x =="old":
        return 1;
    else:
        return x


# In[ ]:


# df.dtypes
df['type'] = df['type'].apply(enc);
# df.head(10)
dftest['type'] = dftest['type'].apply(enc);


# In[ ]:


{np.all(np.isfinite(df)),np.any(np.isnan(df))}


# In[ ]:


{np.all(np.isfinite(dftest)),np.any(np.isnan(dftest))}


# In[ ]:


df.replace([np.inf, -np.inf], np.nan);
df.replace(np.nan, 0, inplace=True);


# In[ ]:


dftest.replace([np.inf, -np.inf], np.nan);
dftest.replace(np.nan, 0, inplace=True);
# can do df.mean(


# In[ ]:


{np.all(np.isfinite(df)),np.any(np.isnan(df))}


# In[ ]:


{np.all(np.isfinite(dftest)),np.any(np.isnan(dftest))}


# In[ ]:


df.drop_duplicates(keep=False,inplace=True) 


# In[ ]:


df.head()
X = df.iloc[:,1:-1]
# X.head()
Xtest =dftest.iloc[:,1:];
Xtest.head()


# In[ ]:


Y  = df.iloc[:,-1:]
# Y.head()

REGRESSION
# In[ ]:


# regressor = RandomForestRegressor(n_estimators=2000,random_state = 0)
regressor = ExtraTreesRegressor(n_estimators=2000,random_state = 0)
# regressor = RandomForestRegressor(n_estimators=10)
regressor.fit(X,Y.values.ravel())


# In[ ]:


ylol = regressor.predict(X)


# In[ ]:


ylol = ylol.round().astype(int)
s = Y.values;
count = 0


# In[ ]:


for i in range(len(ylol)):
    if(ylol[i] == s[i]):
        count = count+1;


# In[ ]:


count/len(ylol)


# In[ ]:


regressor.score(X,Y)


# In[ ]:


Ypred = regressor.predict(Xtest);


# In[ ]:


Ypred


# In[ ]:


Ypred = Ypred.round().astype(int)


# In[ ]:


Ypred


# In[ ]:


dftemp = dftest['id']


# In[ ]:


dftemp.columns ={"id"};
dftemp.head()


# In[ ]:


dftemp2= pd.DataFrame(Ypred)
dftemp2.columns = {"rating"};
dftemp2.head()


# In[ ]:


dftemp2.replace([np.inf, -np.inf], np.nan);
dftemp2.replace(np.nan, 1, inplace=True);
{np.all(np.isfinite(dftemp2)),np.any(np.isnan(dftemp2))}


# In[ ]:


df_final = pd.concat([dftemp,dftemp2.astype(int)],axis = 1)


# In[ ]:


df_final.head()


# In[ ]:


df_final.to_csv("ninth_submission.csv",index = False)


# In[ ]:




