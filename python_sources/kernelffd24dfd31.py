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
from sklearn.tree import DecisionTreeClassifier 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.naive_bayes import GaussianNB

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


data = pd.read_csv("../input/train.csv")


# In[ ]:


data.shape


# In[ ]:


test1 = pd.read_csv("../input/test.csv")


# In[ ]:


test1.shape


# In[ ]:


test1.head()


# In[ ]:


data.head(10)


# In[ ]:


data.describe()


# In[ ]:


data['target'].value_counts()


# In[ ]:


data['target'].value_counts()[1]/(data['target'].value_counts()[1]+data['target'].value_counts()[0])


# In[ ]:


plt.pie(data['target'].value_counts())


# In[ ]:


data_model = data.drop(data[['ID_code']], axis=1)
test_x = test1.drop(test1[['ID_code']],axis=1)


# In[ ]:


from sklearn.model_selection import train_test_split
train, val = train_test_split(data_model,
                             test_size=0.3,
                             random_state=100)

train_x = train.drop('target', axis=1)
train_y = train['target']

val_x = val.drop('target', axis=1)
val_y = val['target']

model = DecisionTreeClassifier(random_state=100, max_depth=10)
model.fit(train_x, train_y)

pred_test = model.predict(val_x)
pred_results = pd.DataFrame({
    'actual':val_y,
    'predicted':pred_test
})
pred_results
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
accuracy_score(pred_results['actual'], pred_results['predicted'])

tn, fp, fn, tp = confusion_matrix(pred_results['actual'], 
                                 pred_results['predicted']).ravel()

print(classification_report(pred_results['actual'], 
                            pred_results['predicted']))


# In[ ]:


from numpy.linalg import eig

evalues, evectors = eig(train_x.corr())

# No. of PCS to return
print((np.cumsum(sorted(evalues)[::-1] / sum(evalues) * 100) < 95).sum())

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(train_x)

train_x_scaled = scaler.transform(train_x)
val_x_scaled = scaler.transform(val_x)
test_x_scaled = scaler.transform(test_x)


# In[ ]:


pct = PCA(n_components=189)
pct.fit(train_x_scaled)

cols = ['PC%d' % i for i in range(1, 190)]
#print(cols)
train_pcs = pd.DataFrame(pct.transform(train_x_scaled), columns=cols)
val_pcs = pct.transform(val_x_scaled)
test_pcs = pct.transform(test_x_scaled)


# In[ ]:


model_nb = GaussianNB()
model_nb.fit(train_pcs, train_y)
pred_test = model_nb.predict(val_pcs)
pred_results = pd.DataFrame({
    'actual':val_y,
    'predicted':pred_test
})
pred_results
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
accuracy_score(pred_results['actual'], pred_results['predicted'])

tn_ab, fp_ab, fn_ab, tp_ab = confusion_matrix(pred_results['actual'], 
                                 pred_results['predicted']).ravel()


# In[ ]:


print(classification_report(pred_results['actual'], 
                            pred_results['predicted']))


# In[ ]:


test1['target'] = model_nb.predict(test_pcs)
test1[['ID_code', 'target']].to_csv('GaussianNB_submission.csv', index=False)


# In[ ]:




