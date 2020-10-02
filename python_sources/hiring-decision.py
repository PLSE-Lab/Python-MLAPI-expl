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
import os
import numpy as np


# In[ ]:


data = pd.read_csv('../input/hirirng-decision/Datapoint.csv')


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data[['Years_of_Experience','German_Language_Level']],data.Decision,test_size=0.1)


# In[ ]:


from sklearn.linear_model import LogisticRegression

model = LogisticRegression()


# In[ ]:


model.fit(X_train,y_train)


# In[ ]:


model.predict(X_test)


# In[ ]:


a = model.predict([[8,6]])

if a ==1 : 
    print("candidate is selected")
    
else:
    print("candidate is not selected")


# In[ ]:


b = model.predict([[8,1]])

if b ==1 : 
    print("candidate is selected")
    
else:
    print("candidate is not selected")


