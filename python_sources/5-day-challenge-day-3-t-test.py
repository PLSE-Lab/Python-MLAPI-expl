#!/usr/bin/env python
# coding: utf-8

# From Rachel Tatman: https://www.kaggle.com/rtatman/the-5-day-data-challenge

# In[13]:


from scipy.stats import ttest_ind
import pandas as pd
import matplotlib.pyplot as plt


# In[14]:


# Load dataset
cereal_data = pd.read_csv('../input/cereal.csv')


# In[15]:


# Describe data
cereal_data.describe()


# In[16]:


# See sample of data
cereal_data.sample(5)


# In[29]:


# Compare sodium between K and G
# Get sodium data for cereal manufacturer K 
mfr_K_sodium_data = cereal_data.loc[cereal_data['mfr'] == 'K']['sodium']
mfr_K_sodium_data.sample(5)


# In[30]:


# Get sodium data for cereal manufacturer C
mfr_G_sodium_data = cereal_data.loc[cereal_data['mfr'] == 'G']['sodium']
mfr_G_sodium_data.sample(5)


# In[38]:


# Calculate standard deviation sodium data for cereal manufacturer K
mfr_K_sodium_data.std()


# In[39]:


# Calculate standard deviation sodium data for cereal manufacturer G
mfr_G_sodium_data.std()


# Clearly, sodium level data for manufacturer's K and G have different standard deviations. 

# In[32]:


# T-Test between K and G and use unequal variance
ttest_ind(mfr_K_sodium_data, mfr_G_sodium_data, equal_var=False)


# In[37]:


mfr_K_sodium_data.hist()
plt.title('Sodium Levels in Cereal Manufacturer K')
plt.xlabel('Amount of Sodium')
plt.ylabel('Frequency')


# In[35]:


mfr_G_sodium_data.hist()
plt.title('Sodium Levels in Cereal Manufacturer G')
plt.xlabel('Amount of Sodium')
plt.ylabel('Frequency')


# In[ ]:




