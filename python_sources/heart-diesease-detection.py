#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


data=pd.read_csv("../input/heart.csv")


# In[ ]:


data.head(5)


# In[ ]:


categorical=data.select_dtypes(exclude='number')


# In[ ]:


categorical.shape


# In[ ]:


numerical=data.select_dtypes(include='number')
numerical.shape


# In[ ]:


data.isnull().sum()


# In[ ]:


data.info()


# In[ ]:


corr=data.corr()
plt.figure(figsize=(10,5))
sns.heatmap(corr,annot=True)


# In[ ]:


data['target'].unique()


# In[ ]:


Y=data['target']
X=data.drop('target',axis=1)


# In[ ]:


from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=1)


# In[ ]:


from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()
lr.fit(X_train,Y_train)


# In[ ]:


pred=lr.predict(X_test)


# In[ ]:


from sklearn.metrics import confusion_matrix
cm=confusion_matrix(Y_test,pred)
print(cm)
from sklearn.metrics import classification_report
report = classification_report(Y_test, pred)
print(report)


# In[ ]:


print(pred)


# In[ ]:


pred_labels=pred.tolist()


# In[ ]:


subm=pd.DataFrame({'index':,'target':pred_labels})
subm.to_csv('output.csv',index=False)


# In[ ]:




