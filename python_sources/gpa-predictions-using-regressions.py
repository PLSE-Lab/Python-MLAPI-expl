#!/usr/bin/env python
# coding: utf-8

# **Importing Libraries**

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import missingno
import seaborn as sns
import statsmodels.api as sm
sns.set()


# **Load Data**

# In[ ]:


df = pd.read_csv('../input/101-simple-linear-regressioncsv/1.01. Simple linear regression.csv')
df.head(3)


# **Data Wrangling**

# In[ ]:


df.info()


# In[ ]:


df.describe()


# In[ ]:


missingno.matrix(df)


# In[ ]:


df.shape


# **Data Explonatory**

# In[ ]:


y = df['GPA']
x1 = df[['SAT']]


# In[ ]:


plt.scatter(y,x1)
plt.xlabel('GPA', fontsize=20)
plt.ylabel('SAT', fontsize=20)
plt.show()


# In[ ]:


x = sm.add_constant(x1)
results = sm.OLS(y,x).fit()
results.summary()


# **Regressions Formulation**
# 
# y = b0 + x1b1
# 
# from OLS results above we know that
# 
# 
# b0(contant) = 0.275
# 
# b1 = 0.0017
# 
# thus, y = 0.275 + x1*0.0017

# **Creating Dummy SAT for Predictions**

# In[ ]:


new_data = pd.DataFrame({'conts':1, 'SAT': [1900,1987,1690]})
new_data 


# **Predictions**

# In[ ]:


predictions = results.predict(new_data)
predictions


# In[ ]:


pred_df = pd.DataFrame({'predictions': predictions})
joined = new_data.join(pred_df)
joined


# In[ ]:




