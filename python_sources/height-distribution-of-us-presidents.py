#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn; seaborn.set()
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


data = pd.read_csv("../input/president_heights.csv")
data.head()


# In[ ]:


heights = np.array(data['height(cm)'])
print(heights)


# In[ ]:


print("Mean height       : ",heights.mean())
print("Standard deviation: ",heights.std())
print("Mininum height    : ",heights.min())
print("Maxinum height    : ",heights.max())


# In[ ]:


print("25th percentile: ",np.percentile(heights,25))
print("Median         : ",np.percentile(heights,50))
print("75th percentile: ",np.percentile(heights,75))


# In[ ]:


plt.hist(heights)
plt.title('Height Distribution of US Presidents')
plt.xlabel('height (cm)')
plt.ylabel('number');


# # I'm new here,it's my first notebook,please chell me up!

# In[ ]:




