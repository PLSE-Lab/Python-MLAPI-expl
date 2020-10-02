#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


dftrain = pd.read_csv('/kaggle/input/eval-lab-1-f464-v2/train.csv')
dftest = pd.read_csv('/kaggle/input/eval-lab-1-f464-v2/test.csv')


# In[ ]:


dftrain=dftrain.fillna(dftrain.mean())
dftest=dftest.fillna(dftest.mean())


# In[ ]:


dftrain = pd.get_dummies(dftrain, columns=["type"])
dftest = pd.get_dummies(dftest, columns=["type"])


# In[ ]:


features = ['feature1','feature2','feature3','feature4','feature5','feature6','feature7','feature8','feature9',
            'feature10','feature11','type_new']


# In[ ]:


from sklearn.ensemble import ExtraTreesRegressor


# In[ ]:


etr = ExtraTreesRegressor(n_estimators=2000)


# In[ ]:


etr.fit(dftrain[features],dftrain['rating'])


# In[ ]:


y_pred=etr.predict(dftest[features])


# In[ ]:


y_pred


# In[ ]:


y_pred=y_pred.round().astype('int64')


# In[ ]:


ta = []


# In[ ]:


for i in range(len(y_pred)):
    ta.append([dftrain['id'][i],y_pred[i]])


# In[ ]:


ta=pd.DataFrame(ta)


# In[ ]:


ta


# In[ ]:


ta['id']=ta[0]
ta['rating']=ta[1]


# In[ ]:


ta=ta.drop(0,axis=1)
ta=ta.drop(1,axis=1)


# In[ ]:


ta


# In[ ]:


ta.to_csv('ans.csv',index=False)


# In[ ]:




