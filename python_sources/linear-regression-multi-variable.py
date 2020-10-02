#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
from sklearn import linear_model


# In[ ]:


df = pd.read_csv('../input/hiring.csv')
df


# In[ ]:


rego = linear_model.LinearRegression()
rego.fit(df.drop("salary",axis='columns'), df.salary)


# In[ ]:


rego.predict([[0,7,5]])


# In[ ]:


rego.coef_


# In[ ]:




