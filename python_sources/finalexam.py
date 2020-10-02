#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import os
print(os.listdir("../input"))


# In[ ]:


data = pd.read_csv("../input/Advertising.csv")
data.head()


# In[ ]:


data.shape


# In[ ]:


x = data['TV']
y = data['sales']
plt.scatter(x,y)
plt.show()


# **MODEL DEVELOPING**

# In[ ]:


import statsmodels.api as sm
x1 = sm.add_constant(x)
model = sm.OLS(y,x1)
model_fit = model.fit()
print(model_fit.params)
beta_0 = model_fit.params[0] # to use for predicting data
beta_1 = model_fit.params[1] # N.A


# **SUMMARY OF THE FITTED SIMPLE LINEAR REGRESSION MODEL**

# In[ ]:


model_fit.summary()


# **USING THE MODEL FOR PREDICTING DATA**

# In[ ]:


x_new = pd.DataFrame({'TV':[50]})
x_new.head()


# **PREDICT DATA**

# In[ ]:


predected_y = beta_0 + (beta_1 * x_new)
print(predected_y)


# **PLOTING THE LEAST SQUARE LINE**

# In[ ]:



x_new = pd.DataFrame({'TV':[data.TV.min(),data.TV.max()]})
x_new.head()


# In[ ]:


predected_y = beta_0 + (beta_1 * x_new)
print(predected_y)


# In[ ]:


plt.scatter(x,y)
plt.plot(x_new,predected_y,c = 'red', linewidth=2)


# **CONFIDENCE IN OUR MODEL**

# In[ ]:


model_fit.conf_int()


# **HYPOTHESIS TESTING AND P-VALUES**

# null hypothesis = there is no relation between TV ads and Sales (b1 = 0)
# alternative hypothesis = there is relation between TV ads and Sales (b1 is not eqaul to zero)
# if p-value is less than 0.05 then there is a relation between TV ads and Sales
# if p-values > 0.05 then there is no relation

# In[ ]:


model_fit.pvalues


# here p-value < 0.05 so there is a relation between TV ads and Sales.
# Generally p-value of b0 (Beta_0) is ignored

# **R-SQUARED VALUE**
# 
# it is most useful as a tool for comparing different models

# In[ ]:


model_fit.rsquared


# **MULTIPLE LINEAR REGRESSION**

# In[ ]:


x = data[['TV','radio','newspaper']]
y = data['sales']
x1 = sm.add_constant(x)
model = sm.OLS(y,x1)
model_fit = model.fit()
print(model_fit.params)


# **SUMMARY OF THE FITTED MODEL**

# In[ ]:


#
model_fit.pvalues
model_fit.summary()

