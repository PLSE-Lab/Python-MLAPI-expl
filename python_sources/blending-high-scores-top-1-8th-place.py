#!/usr/bin/env python
# coding: utf-8

# # Blending High Scores
# ## Upvote if you like

# In[ ]:


# Blending with https://www.kaggle.com/alfredmaboa/advanced-regression-techniques-regularization


# In[ ]:


import numpy as np
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt


# In[ ]:


sub1 = pd.read_csv('../input/housesbumissions/own_technique.csv')
sub2 = pd.read_csv('../input/housesbumissions/advanced_regression_techniques.csv')
temp = pd.read_csv('../input/housesbumissions/own_technique.csv')


# In[ ]:


print( sub1['SalePrice'].describe() )
print( sub2['SalePrice'].describe() )


# In[ ]:


sns.set()
plt.hist(sub1['SalePrice'],bins=100)
plt.show()


# In[ ]:


sns.set()
plt.hist(sub2['SalePrice'],bins=100)
plt.show()


# In[ ]:


temp['SalePrice'] = 0.70*sub1['SalePrice'] + 0.30*sub2['SalePrice'] 
temp.to_csv('blend_submission.csv', index=False )


# In[ ]:


sns.set()
plt.hist(temp['SalePrice'],bins=100)
plt.show()

