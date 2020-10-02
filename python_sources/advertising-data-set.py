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


Data = pd.read_csv("/../kaggle/input/advertising/advertising.csv")
Data.head()


# In[ ]:


Data.describe()


# In[ ]:


Data.info()


# In[ ]:


Data.corr()


# In[ ]:


Data.City.value_counts()


# In[ ]:


import seaborn as sns


# In[ ]:


sns.heatmap(Data.corr());


# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,auc,roc_curve,precision_score


# In[ ]:


X = Data[['Daily Time Spent on Site','Age','Area Income','Daily Internet Usage']]


# In[ ]:


y =Data['Clicked on Ad']


# In[ ]:


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=42,stratify=y)
X_train.shape,X_test.shape,y_train.shape,y_test.shape


# **Base line Accuracy**

# In[ ]:


y.value_counts()


# In[ ]:


model = LogisticRegression()


# In[ ]:


model.fit(X_train,y_train)


# In[ ]:


model.score(X_train,y_train)


# In[ ]:


pred_test = model.predict(X_test)


# In[ ]:


accuracy_score(y_test,pred_test)


# In[ ]:


pred_train = model.predict_proba(X_train)
pred_train[:]


# In[ ]:


pred_train[:,1]


# In[ ]:


fpr,tpr,t = roc_curve(y_train,pred_train[:,1],pos_label=1)       


# In[ ]:


import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib as npl
l1 = []
for i in range(len(fpr)-1):
    l1.append([(fpr[i],tpr[i]),(fpr[i+1],tpr[i+1])])
#print(l1)

lc = LineCollection(l1,cmap='hsv')
plt.figure(figsize=(10,5))
fig, ax = plt.subplots()
line=ax.add_collection(lc)
lc.set_array(t[1:])

plt.colorbar(line, ticks=np.arange(0,1,0.1))

plt.title('ROC Curve')
plt.xlabel('False Postive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')


# In[ ]:


pred_test_prob = model.predict_proba(X_test)
pred_test_prob[:5]


# In[ ]:


pred_t_test = np.where(pred_test_prob[:,1]>=0.8,1,0)
pred_t_test


# In[ ]:


accuracy_score(y_test,pred_t_test)


# In[ ]:




