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

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[ ]:


df = pd.read_csv("../input/ginf.csv")


# In[ ]:


df.head()


# In[ ]:


y = (df["fthg"]+df["ftag"] > 2.5)


# In[ ]:


X = df[["odd_h","odd_d","odd_a"]]


# In[ ]:


from sklearn.model_selection import train_test_split
train_X, val_X, train_y, val_y = train_test_split(X, y,random_state = 0)


# In[ ]:


from sklearn.linear_model import LogisticRegression
my_model = LogisticRegression()
my_model.fit(train_X, train_y)
my_model.score(val_X, val_y)

