#!/usr/bin/env python
# coding: utf-8

# **Follow my Github account: https://github.com/satyamuralidhar/Kaggle-HeartDisease_UCI   **
# 
# iam using gradient boosting techinque and ROC , AUC 

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
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# In[ ]:


df = pd.read_csv('/kaggle/input/heart-disease-uci/heart.csv')


# In[ ]:


df.isnull().sum()


# In[ ]:


df


# In[ ]:


sns.barplot(x=df['age'],y=df['target'])


# In[ ]:


plt.hist(x=df['age'],histtype='bar')


# In[ ]:


plt.hist(x=df['thalach'],histtype='bar',color='green')


# In[ ]:


plt.hist(x=df['chol'],histtype='bar',color='yellow')


# In[ ]:


from sklearn.preprocessing import StandardScaler , LabelEncoder
scaler = StandardScaler()


# In[ ]:


label = LabelEncoder()
train = df.iloc[:,:-1]
train


# In[ ]:


train['oldpeak'] = label.fit_transform(train['oldpeak'])


# In[ ]:


target = df['target']


# In[ ]:


X_scaled = scaler.fit_transform(train)


# In[ ]:


from sklearn.metrics import roc_curve , roc_auc_score , confusion_matrix , accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
grad = GradientBoostingClassifier()
X_train,X_test,y_train,y_test = train_test_split(X_scaled,target,test_size=0.4,random_state=120)
grad.fit(X_train,y_train)
y_pred=grad.predict(X_test)
accuracy_score(y_pred,y_test)


# In[ ]:


auc = roc_auc_score(y_test,y_pred)
confusion = confusion_matrix(y_test,y_pred)
tp = confusion[0][0]
fp = confusion[0][1]
fn = confusion[1][0]
tn = confusion[1][1]


# In[ ]:


from sklearn.metrics import plot_confusion_matrix
disp = plot_confusion_matrix(grad,X_test,y_test,cmap=plt.cm.Blues,normalize=None)
disp.confusion_matrix


# In[ ]:



# finding accuracy 
accuracy = (tp+tn)/(tp+tn+fp+fn)
accuracy


# In[ ]:


fpr , tpr , thresholds = roc_curve(y_test,y_pred)
plt.plot(fpr,tpr,color = 'darkblue',label = 'ROC')
plt.plot([0,1],[0,1],color='orange',linestyle='--',label="ROC Curve(area=%0.2f)"%auc)
plt.xlabel('False + ve rate')
plt.ylabel('True +ve rate')
plt.legend()
plt.show()


# In[ ]:




