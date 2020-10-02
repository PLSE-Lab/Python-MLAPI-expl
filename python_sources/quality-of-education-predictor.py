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


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import gc
import json
import math
import cv2
import PIL
from PIL import Image

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import OneHotEncoder
from tqdm import tqdm
from sklearn.decomposition import PCA
import os
import imagesize
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


data = pd.read_csv("../input/world-university-rankings/cwurData.csv")


# In[ ]:


data.head()


# #  **DATA VISUALIZATION**

# In[ ]:


plt.figure(figsize=(12, 5))
plt.hist(data['publications'].values, bins=200)
plt.title('publication w.r.t count')
plt.xlabel('publication')
plt.ylabel('Count')
plt.show()


# In[ ]:


data.plot.scatter('patents','year')


# In[ ]:


sns.countplot(data.publications)


# In[ ]:


data.plot.scatter('publications','year')


# In[ ]:


plt.figure(figsize=(12, 18))
sns.barplot(y=data['country'], x=data['alumni_employment'], palette="deep")


# In[ ]:


plt.figure(figsize=(12, 18))
sns.barplot(y=data['country'], x=data['publications'], palette="deep")


# In[ ]:


X = data[[ 'quality_of_faculty', 'publications', 'patents']]
y= data['quality_of_education']

X = X.drop(data[data['country'] =='USA'].index)
y = y.drop(data[data['country'] =='USA'].index)


# In[ ]:


X.head()


# In[ ]:


y.head()


# In[ ]:


X.shape,y.shape


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score


# In[ ]:


X_train,X_validation,Y_train,Y_validation=train_test_split(X,y,test_size=0.2,random_state=0)


# In[ ]:


X_train.shape, X_test.shape, y_train.shape, y_test.shape


# In[ ]:


sc=StandardScaler()
X_train_std=sc.fit_transform(X_train)
X_validation_std=sc.transform(X_validation)


# In[ ]:


lr=LogisticRegression()
lr.fit(X_train_std,Y_train)
y_validation_pred=lr.predict(X_validation_std)
print(accuracy_score(y_validation_pred,Y_validation))

