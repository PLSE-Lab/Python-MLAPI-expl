#!/usr/bin/env python
# coding: utf-8

# The objective is to find whether educating women could lead to saving more female children.The graphs clearly shows that the better sex ratios can only be achieved by educating the females.

# In[ ]:


import numpy as np 
import pandas as pd

data= pd.read_csv('../input/cities_r2.csv')
print(data.head(0))
#creating features to compare
Young_Female = data['0-6_population_female']
feature_female = data[['literates_female','female_graduates','effective_literacy_rate_female']]


# In[ ]:


import matplotlib.pyplot as plt
plt.figure(figsize=(8,15), dpi=80)
for i, col in enumerate(feature_female.columns):  
      plt.subplot(3, 1, i+1) 
      plt.plot(data[col], Young_Female,'mo')
      plt.title(col)
      plt.xlabel(col)
      plt.ylabel('Young Female Population' )


# In[ ]:


The above graphs clearly shows that the better sex ratios can only be achieved by educating the females.

