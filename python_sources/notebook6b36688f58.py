#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn import svm
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


df = pd.read_csv('../input/HR_comma_sep.csv')
print(df.images)
#digits = datasets.load_digits()
#print(len(digits.data))
#print(digits.data)
#print(digits.target)
#print(digits.images)


# In[ ]:


column_names = df.columns.tolist()
print("Column Names")
print(column_names)
df.shape
df.head()

