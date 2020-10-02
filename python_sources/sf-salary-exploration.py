#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import sqlite3
import random

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))


# # Some Exploration of SF Salaries

# In[ ]:


salaries = pd.read_csv('../input/Salaries.csv')
salaries.info()


# In[ ]:


salaries = salaries.convert_objects(convert_numeric=True)


# In[ ]:


# Some Data Munging and Type Conversion
df = df[(df.BasePay != 'Not Provided') & (df.BasePay != '0.00')].copy()

for column in ["BasePay", "OvertimePay", "Benefits", "TotalPay", "TotalPayBenefits"]:
    df[column] = df[column].map(float)


# In[ ]:


salaries = salaries.drop('Notes', axis=1)


# In[ ]:


salaries.describe()


# In[ ]:


# i am using seaborn to change aesthetics of the plots
sns.set_style("whitegrid")

# matplotlib.pyplot is the main module that provides the plotting API
x = [np.random.uniform(100) for _ in range(200)]
y = [np.random.uniform(100) for _ in range(200)]
plt.scatter(x,y)


# In[ ]:


sns.jointplot(x='X', y='Y', data=pd.DataFrame({'X': x, 'Y': y}))


# In[ ]:




