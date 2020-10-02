#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# ## Objective
# #### Predict  Canada's per capita income for years after 2016
# beginner level

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model


# In[ ]:


df=pd.read_csv("../input/canada-per-capita-income-single-variable-data-set/canada_per_capita_income.csv")
df.tail()


# #### check for null values

# In[ ]:


df.isnull().sum()


# #### Plot scatter plot year vs Per capita

# In[ ]:


plt.scatter(x=df['year'],y=df['per capita income (US$)'],marker="+")
plt.xlabel("Year")
plt.ylabel("per capita income")


# #### For Linear regression: y=mx+b
# fit the model

# In[ ]:


reg=linear_model.LinearRegression()
reg.fit(df[['year']],df[["per capita income (US$)"]])
reg.predict([[2000]])

where coef or slope ie m in y=mx+b is ...
# In[ ]:


reg.coef_


# intercept ie b is...

# In[ ]:


reg.intercept_


# In[ ]:


plt.xlabel("Year")
plt.ylabel("per capita income")
plt.scatter(x=df['year'],y=df['per capita income (US$)'],marker="+")
plt.plot(df.year,reg.predict(df[["year"]]),color="blue")


# This is just a beginner level notebook for understanding
