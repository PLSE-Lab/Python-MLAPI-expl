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


data = pd.read_csv("/kaggle/input/udacity-mlcharity-competition/census.csv")
display(data.describe())
display(data.head(n=10))
data.isnull().sum()


# In[ ]:


data['income'] = data['income'].apply(lambda x: 1 if x=='>50K' else 0)
data = pd.get_dummies(data, columns=['workclass', 'education_level', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country'])


# In[ ]:


# removing label
y = data.pop('income')
display(y.head())
display(data.head())


# In[ ]:


from sklearn.model_selection import train_test_split
# splitting datasets
X_train, X_test, y_train, y_test = train_test_split(data, y, train_size=0.8)


# In[ ]:


from sklearn.ensemble import AdaBoostClassifier
model = AdaBoostClassifier()
model.fit(X_train, y_train)


# In[ ]:


y_predict = model.predict(X_test)


# In[ ]:


from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_curve
accuracy_score(y_test, y_predict)


# In[ ]:


len(y[y == 0]) / len(y)


# 10% better then just guessing they always make more than 50k.

# In[ ]:


f1_score(y_test, y_predict)


# In[ ]:


confusion_matrix(y_test, y_predict)


# In[ ]:


import matplotlib.pyplot as plt


# In[ ]:


y_probas = model.predict_proba(X_test)[::,1]
fpr, tpr, _ = roc_curve(y_test, y_probas)
plt.plot(fpr, tpr)
plt.legend(loc=4)
plt.show()


# So we see a nice curve to use predictive power here. Say we wanted to minimize False Negatives (maximize recall), where its more important to not to falsely classify an individual as not making over 50, while a higher FPR gives how many false positives. I would say we keep the FPR  around the .4 mark to get the most benefit in properly identifying all people making more than 50K

# In[ ]:


results = pd.DataFrame({'fpr':fpr,'tpr': tpr, 'threshold':_})


# In[ ]:


results[results['tpr']> .9]


# In[ ]:


X_submission = pd.read_csv('/kaggle/input/udacity-mlcharity-competition/test_census.csv')
X_submission = pd.get_dummies(X_submission, columns=['workclass', 'education_level', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country'])
X_submission.head()
X_submission = X_submission.dropna(axis=1)


# In[ ]:


y_pred_test = model.predict(X_submission)


# In[ ]:


y_pred_df = pd.DataFrame(y_pred_test, columns=['income'])


# In[ ]:


y_pred_df


# In[ ]:


y_pred_df.to_csv('submission.csv', index_label='id')


# 
