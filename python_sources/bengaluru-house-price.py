#!/usr/bin/env python
# coding: utf-8

# In this section we are going to set our enviornment by importing necessary libraries, as well as loading our data in pandas dataframe.

# In[ ]:


# import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


df = pd.read_csv('../input/Bengaluru_House_Data.csv')
df.head()


# In[ ]:


print("This dataset has {} rows and {} columns.".format(*df.shape))
print("This dataset contains {} duplicates.".format(df.duplicated().sum()))


# **Checking missing values in each columns**

# In[ ]:


df.isnull().sum()


# **Replace missing values with 0 and unknown**

# In[ ]:


new_df = df.fillna({'society': 'unknown',
                   'balcony': 0, 'bath': 0,
                   'size': 'unknown', 'location': 'unknown'})
new_df.head()


# In[ ]:


new_df.nunique()


# In[ ]:


# Check the types of data
new_df.dtypes


# In[ ]:


new_df.isnull().values.sum()


# In[ ]:


new_df.describe()


# In[ ]:


# Finding out the correlation between the features
correlations =new_df.corr()
correlations['price']


# In[ ]:


cor_target = abs(correlations['price'])

# Display features with correlation < 0.1
removed_features = cor_target[cor_target < 0.1]
removed_features


# In[ ]:


fig_1 = plt.figure(figsize=(10, 8))
new_correlations = new_df.corr()
sns.heatmap(new_correlations, annot=True, cmap='Accent', annot_kws={'size': 8})
plt.title('Pearson Correlation Matrix')
plt.show()


# To be continued
