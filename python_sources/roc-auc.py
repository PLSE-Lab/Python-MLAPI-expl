#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


import numpy as np
import pandas as pd


# In[ ]:


data=pd.read_csv("../input/Social_Network_Ads.csv")


# In[ ]:


data


# In[ ]:


data['User ID'].isnull().sum()


# In[ ]:


data.corr()


# In[ ]:


data1=data.drop(['User ID'],axis=1)
data1


# In[ ]:


from sklearn.preprocessing import LabelEncoder
lb=LabelEncoder()
data1.iloc[:,0:1]=lb.fit_transform(data1.iloc[:,0:1])
data1


# In[ ]:


X=data1.iloc[:,:-1]
y=data1.iloc[:,-1]


# In[ ]:


X


# In[ ]:


y


# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# In[ ]:


Xtrain,Xtest,ytrain,ytest=train_test_split(X,y,test_size=0.2,random_state=42)


# In[ ]:


sc=StandardScaler()


# In[ ]:


Xtrain=sc.fit_transform(Xtrain)


# In[ ]:


Xtest=sc.fit_transform(Xtest)


# In[ ]:


from sklearn.linear_model import LogisticRegression


# In[ ]:


lr=LogisticRegression(random_state=42)


# In[ ]:


lr.fit(Xtrain,ytrain)


# In[ ]:


ypred=lr.predict(Xtest)


# In[ ]:


ypred


# In[ ]:


pd.crosstab(ytest,ypred)


# In[ ]:


from sklearn.metrics import f1_score,roc_auc_score


# In[ ]:


f1_score(ytest,ypred)


# In[ ]:


roc_auc_score(ytest,ypred)


# In[ ]:


from sklearn import metrics


# In[ ]:


fpr,tpr,threshold=metrics.roc_curve(ytest,ypred,pos_label=2)


# In[ ]:


fpr


# In[ ]:


tpr


# In[ ]:


threshold


# In[ ]:


import matplotlib.pyplot as plt


# In[ ]:


y_pred_proba = lr.predict_proba(Xtest)[::,1]
fpr, tpr, _ = metrics.roc_curve(ytest,  y_pred_proba) 
auc = metrics.roc_auc_score(ytest, y_pred_proba) 
plt.plot(fpr,tpr,label="data 1, auc="+str(auc)) 
plt.legend(loc=4) 
plt.show()


# In[ ]:




