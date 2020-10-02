#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.|


# In[ ]:


df = pd.read_csv("../input/creditcard.csv")
df.head()
#https://www.kaggle.com/joparga3/in-depth-skewed-data-classif-93-recall-acc-now
#https://www.kaggle.com/arathee2/achieving-100-accuracy
#https://www.kaggle.com/currie32/predicting-fraud-with-tensorflow


# In[ ]:


df.Class.unique()


# In[ ]:


df['Class'].value_counts()


# In[ ]:


df1 = df[['Time','V1','V2','Amount','Class']]
df1


# In[ ]:


sns.pairplot(df1,hue='Class',palette='Dark2') #Drawing the scatterplo


# In[ ]:




