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
filenames = ['week2-task/datasets_33180_43520_heart.csv']
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        df = pd.read_csv(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


from sklearn.utils import shuffle
df = shuffle(df)


# In[ ]:


import seaborn as sns


# In[ ]:


sns.heatmap(df.isnull(),yticklabels=False)


# In[ ]:


thal = pd.get_dummies(df['thal'],drop_first=True)
slope = pd.get_dummies(df['slope'],drop_first=True)
ca = pd.get_dummies(df['ca'],drop_first=True)


# In[ ]:


df.drop(['thal','slope','ca'],axis=1,inplace=True)
df.head()


# In[ ]:


df = pd.concat([df,thal,slope,ca],axis=1)


# In[ ]:


df.head()


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(df.drop('target',axis=1), 
                                                    df['target'], test_size=0.30, 
                                                    random_state=101)


# In[ ]:


from sklearn.ensemble import RandomForestClassifier


# In[ ]:


from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[ ]:


classifier = RandomForestClassifier(n_estimators=128)
classifier.fit(X_train, y_train)


# In[ ]:


predictions = classifier.predict(X_test)


# In[ ]:


from sklearn.metrics import confusion_matrix
accuracy=confusion_matrix(y_test,predictions)


# In[ ]:


accuracy


# In[ ]:


from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,predictions)
accuracy

