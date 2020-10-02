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
for dirname, _, filenames in os.walk('/kaggle/input/illumina-south-asia-pacific'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 16:08:35 2019

@author: deepakkolekar
"""

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

get_ipython().run_line_magic('matplotlib', 'inline')
np.random.seed(1)

df = pd.read_csv('/kaggle/input/illumina-south-asia-pacific/AMAAM.csv',sep= ',')
df.head()

Xy = np.array(df)
X = Xy[:,:-1].astype(float)
print(X.shape)

y = Xy[:,-1]

np.unique(y,return_counts=True)

from sklearn.preprocessing import OneHotEncoder
onehot_encoder = OneHotEncoder(sparse=False,categories='auto')

y = onehot_encoder.fit_transform(y.reshape(-1,1))

y = np.argmax(y,axis=1)
print(y.shape)

from sklearn.utils import shuffle
X, y = shuffle(X, y)

from sklearn.preprocessing import MinMaxScaler
X = MinMaxScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test,y_pred)                
print(accuracy)

from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from sklearn.svm import SVC
clf = SVC(random_state=0)
clf.fit(X_train, y_train)
#cnf_matrix = confusion_matrix(y_test, y_pred)
#print(cnf_matrix)
plot_confusion_matrix(clf, X_test, y_test)
plt.show()
#plot_confusion_matrix(model,y_test,y_pred)

