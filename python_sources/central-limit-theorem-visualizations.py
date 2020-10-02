#!/usr/bin/env python
# coding: utf-8

# # The central limit theorem is crucial to statistics. It states that the distribution of sample means aproaches a normal distribution (bell curve), as the sample size increases (>30 is the rule of thumb), assuming that all samples are identical in size, despite the population distribution shape.
# ![](https://upload.wikimedia.org/wikipedia/commons/7/7b/IllustrationCentralTheorem.png)
# 
# 
# Source: https://upload.wikimedia.org/wikipedia/commons/7/7b/IllustrationCentralTheorem.png

# In[ ]:


import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np 
from scipy.stats import norm 
import seaborn as sns 
plt.style.use('ggplot')
get_ipython().run_line_magic('matplotlib', 'inline')


# # I will be using the housing data to demonstrate this important statistical concept. 

# In[ ]:


data = pd.read_csv('../input/california-housing-prices/housing.csv')
data.head()


# # We are using the median house value for our distribution

# In[ ]:


median_house_value = np.array(data.median_house_value)


# # Plotting the original distribution, it is skewed to the right. 

# In[ ]:


plt.figure(figsize=(12, 8))
plt.title('Median House Value Distribution', size=18)
plt.xlabel('Value in $', size=18)
sns.distplot(median_house_value, fit=norm, color='blue', kde=False)


# # Take the sample means

# In[ ]:


sample_num = 1000
sample_size = 30

mean_sample_values = []

for i in range(sample_num):
    sample_mean = np.mean(np.random.choice(median_house_value, sample_size, replace=True))
    mean_sample_values.append(sample_mean)   


# # Plotting the sample means distribution, which approaches a normal distribution. As we can see, the central limit theorem is indeed true. 

# In[ ]:


plt.figure(figsize=(12, 8))
plt.title('Sample Mean of Median House Value Distribution', size=18)
plt.xlabel('Value in $', size=18)
sns.distplot(mean_sample_values, fit=norm, color='blue', kde=False)

