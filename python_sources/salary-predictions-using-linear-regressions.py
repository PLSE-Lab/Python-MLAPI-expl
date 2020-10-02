#!/usr/bin/env python
# coding: utf-8

# # Importing Libraries

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

sns.set()


# # Loading Data

# In[ ]:


df = pd.read_csv('../input/salary-data-simple-linear-regression/Salary_Data.csv')
df.head()


# # Checking Data

# In[ ]:


df.isnull().sum()


# In[ ]:


x1 = df['YearsExperience']
y = df['Salary']


# In[ ]:


plt.scatter(x1,y)
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()


# # Here We Go Again (OLS)

# In[ ]:


x = sm.add_constant(x1)
results = sm.OLS(y,x).fit()
results.summary()


# # Model Visualization
# 

# In[ ]:


plt.scatter(x1,y)
yhat = 25790 + x1*9449.9623
fig = plt.plot(x1,yhat, lw=4, c='orange', label='Regression Line')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()


# # Predictions with Random Years of Experience

# In[ ]:


new_data = pd.DataFrame({'conts': 1,'YearsExperience': [9, 12, 15, .5]})
predictions = results.predict(new_data)
predictions


# In[ ]:




