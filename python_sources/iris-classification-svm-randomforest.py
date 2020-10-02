#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


ds = pd.read_csv('../input/IRIS.csv')
print (ds.columns)
print (len(ds.columns))
print (ds.head())


# In[ ]:


sns.pairplot(data=ds, hue='species', palette='Set2')


# In[ ]:


x = ds.iloc[ :, :-1]
y = ds.iloc[:, 4]
print(x.shape)
print (y.shape)


# In[ ]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.20)
print (x_train.shape)
print (x_test.shape)
print("...............")

print (y_train.shape)
print (y_test.shape)


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
rft = RandomForestClassifier()
rft.fit(x_train,y_train)
print (rft.score(x_test,y_test))


# In[ ]:


from sklearn.metrics import classification_report

pred = rft.predict(x_test)
print (classification_report(y_test, pred))


# In[ ]:


from sklearn.svm import SVC
model=SVC()
model.fit(x_train, y_train)
print (model.score(x_test,y_test))


# In[ ]:


pred=model.predict(x_test)
print(classification_report(y_test, pred))

