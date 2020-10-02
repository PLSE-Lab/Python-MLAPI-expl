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


import pandas as pd
import numpy as np
import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report,confusion_matrix


# In[ ]:


cdc=pd.read_csv('/kaggle/input/creditcardfraud/creditcard.csv')


# In[ ]:



cdc.info()
cdc.describe()


# In[ ]:



cdc.tail()


# In[ ]:


cdc.head()


# In[ ]:


cdc['Class'].value_counts()


# In[ ]:


cdc.isna().sum()


# In[ ]:


scale=StandardScaler()


# In[ ]:


scale.fit(cdc.drop(['Class','Time','Amount'],axis=1))


# In[ ]:


scaled_features=scale.transform(cdc.drop(['Class','Time','Amount'],axis=1))


# In[ ]:


cdc_feat=pd.DataFrame(scaled_features,columns=cdc.columns[1:-2])


# In[ ]:


cdc_feat.head(10)


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(scaled_features,cdc['Class'],test_size=0.30)


# In[ ]:


knn=KNeighborsClassifier(n_neighbors=1)


# In[ ]:


knn.fit(X_train,y_train)


# In[ ]:


pred=knn.predict(X_test)


# In[ ]:


print(confusion_matrix(y_test,pred))


# In[ ]:


print(classification_report(y_test,pred))

