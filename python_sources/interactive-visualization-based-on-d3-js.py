#!/usr/bin/env python
# coding: utf-8

# ### [Interactive Visualisation][1] ! based on dc.js (d3.js + crossfilter) 
# 
# Very helpful to understand profile of people leaving the company respectively after 3, 4 and 5 years. 
# 
# [![Interactive Visualization][2]](http://www.demos.donneesbrutes.com/who-will-quit-next?kaggle)
# 
# Next step, **survival analysis** :
# 
# - cox model with time-varying covariates
# - Tree-structured survival model
# 
#   [1]: http://www.demos.donneesbrutes.com/who-will-quit-next
#   [2]: http://www.demos.donneesbrutes.com/img/hr.png

# In[ ]:


import pandas
import sklearn
import pandas


# In[ ]:


df = pandas.read_csv('../input/HR_comma_sep.csv')
df.shape
df.describe()

