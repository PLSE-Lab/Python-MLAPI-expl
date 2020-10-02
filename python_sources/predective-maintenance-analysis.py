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
import plotly.offline as py
py.init_notebook_mode(connected=True)
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.offline as offline

import warnings
warnings.filterwarnings('ignore')


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


train =pd.read_csv('../input/train_label.csv')


# In[ ]:


train.head() # to check the head of the dataset


# In[ ]:


train.tail()


# In[ ]:


train.info()


# In[ ]:


train.describe()


# In[ ]:


train.isnull().sum() # found one nan value in the dataset 


# In[ ]:


train[train['label']==1].count()


# In[ ]:


train[train['label']==0].count()


# In[ ]:


from datetime import datetime
data = [go.Scatter(x=train.date, y=train['label'])]

py.iplot(data, filename = 'time-series-simple')


# In[ ]:


train.fillna(0,inplace=True) #by using fillna we can replace nan with 0


# In[ ]:


train.info()


# In[ ]:


X=train.drop('label',axis=1)
y=train.label


# In[ ]:


X.shape


# In[ ]:


y.shape


# In[ ]:


import datetime as dt
X['date'] = pd.to_datetime(X['date'])
X['date']=X['date'].map(dt.datetime.toordinal)

X['date'].head()


# ** Using train test split function because test data showing some error to read it **

# In[ ]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=60,test_size=0.20) 


# In[ ]:


from sklearn.tree import DecisionTreeClassifier
decision=DecisionTreeClassifier()
decision.fit(X_train,y_train)


# In[ ]:


y_predict=decision.predict(X_test)


# In[ ]:


from sklearn.metrics import roc_auc_score
roc_auc = roc_auc_score(y_test, y_predict)
roc_auc


# Check Out the confusion matrix to find out the true positive and true negative 

# In[ ]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_predict)
cm


# In[ ]:


from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt 
fpr, tpr, thresholds = roc_curve(y_test, decision.predict_proba(X_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, color='darkorange', label='Decision Tree (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
random=RandomForestClassifier()
random.fit(X_train,y_train)


# In[ ]:


pre=random.predict(X_test)


# In[ ]:


from sklearn.metrics import roc_auc_score
roc_auc = roc_auc_score(y_test, pre)
roc_auc


# In[ ]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, pre)
cm


# In[ ]:


from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt 
fpr, tpr, thresholds = roc_curve(y_test, random.predict_proba(X_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, color='darkorange', label='random forest (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()


# 
