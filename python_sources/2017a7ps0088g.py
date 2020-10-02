#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing as pp
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


df=pd.read_csv("/kaggle/input/eval-lab-1-f464-v2/train.csv")


# In[ ]:


df_new  = df.drop(['id'],axis=1)


# In[ ]:


y=df_new['rating']


# In[ ]:


X = df_new.drop(['rating'],axis=1)


# In[ ]:


numX = X.drop(['type'],axis=1)


# In[ ]:


CatX = X['type']


# In[ ]:


numX.fillna(numX.mean(),inplace = True)


# In[ ]:


normalizernx = pp.Normalizer();


# In[ ]:


numXnorm = normalizernx.fit_transform(numX)


# In[ ]:


X_train = pd.DataFrame(numXnorm,columns=list(numX.columns))


# In[ ]:


X_train["type"]=CatX 


# In[ ]:


X_train["type"]=X["type"].map({"new":1,"old":0})


# In[ ]:


X_train


# In[ ]:


X_test = pd.read_csv('/kaggle/input/eval-lab-1-f464-v2/test.csv')


# In[ ]:


X_test['type']=X_test['type'].map({'new':1,'old':0})


# In[ ]:


catXtest=X_test['type']


# In[ ]:


X_test=X_test.drop(['type'],axis=1)
X_test = X_test.drop(['id'],axis=1)


# In[ ]:


X_test.fillna(X_test.mean(),inplace = True)


# In[ ]:


norm2=pp.Normalizer()


# In[ ]:


X_test


# In[ ]:


X_test=norm2.fit_transform(X_test)


# In[ ]:


X_test = pd.DataFrame(X_test,columns=list(numX.columns))


# In[ ]:


X_test


# In[ ]:


X_test['type']=catXtest


# In[ ]:


X_test


# In[ ]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# Initialize and train
clf1 = DecisionTreeClassifier(max_depth = 10)
clf2 = RandomForestClassifier(n_estimators=50,max_depth=15)


# In[ ]:





# In[ ]:


clf2.fit(X_train,y)


# In[ ]:


y_pred = clf2.predict(X_test)


# In[ ]:


new=pd.read_csv("/kaggle/input/eval-lab-1-f464-v2/test.csv")


# In[ ]:


dfpref = pd.DataFrame({'id':new['id'],"rating":y_pred})


# In[ ]:


dfpref.head()


# In[ ]:


dfpref.to_csv ("sub3.csv", header=True,index=False)


# In[ ]:


dfpref.head()


# In[ ]:




