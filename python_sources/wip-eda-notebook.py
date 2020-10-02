#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams["figure.figsize"] = (16,9)
import seaborn as sns
import scipy
import os
print(os.listdir("../input"))


# In[2]:


df = pd.read_csv('../input/train.csv')
df.info()


# In[3]:


df.head()


# ## Deal probability
# The histogram of target values show a clear abundance of low values, making this dataset a pretty umbalanced one.

# In[4]:


hist_kws={"alpha": 0.3}
sns.distplot(df.deal_probability, hist_kws=hist_kws)
plt.title('Deal Probability distribution')
plt.margins(0.02)
plt.show()


# ## Price

# In[5]:


# We sample the dataframe & limit the extreme prices (+ no NaN !)
df_dropped = df.loc[df.index[df.price < 500000]].sample(n=50000)


# It appears that there is little impact of price on deal probability seeing that:
# - huge confidence intervals for linear regression
# - statistically significant p value for a pearson correlation close to 0
# 
# Besides, the slopes we see for the linear regressions may be explained by:
# - the scatter motive looks like a grid (indicating a non uniform distribution of values after our sampling)
# - a higher concentration of low/low values (same as above)

# In[6]:


sns.lmplot('price', 'deal_probability', hue='user_type', data=df_dropped, fit_reg=True, size=10, aspect=2, scatter_kws={'s': 10, 'alpha':0.3})
plt.xlabel('Ad price')
plt.ylabel('Deal Probability')
plt.margins(0.01)
plt.show()
print(f'Pearson correlation : {scipy.stats.pearsonr(df_dropped.price.values, df_dropped.deal_probability.values)}')


# The histogram of the full dataset prices confirms the hypothesis of many low prices offers

# In[7]:


sns.distplot(df.price.dropna(), kde=False)
plt.margins(0.02)
plt.show()


# A kernel density estimation shows more clearly peaks on X.10k prices, which probably correspond with psychological thresholds.

# In[8]:


sns.kdeplot(df_dropped.price)
plt.margins(0.02)
plt.show()


# We can try to trim even further the price.

# In[9]:


df_trimmed = df.loc[df.index[df.price < 40000]].sample(n=50000)


# In[10]:


sns.kdeplot(df_trimmed.price)
plt.margins(0.02)
plt.show()


# ## User type EDA
# We colored our scatter plot above by `user_type` (Company, Private, Shop).
# The Private type is much more present in the dataset.

# In[11]:


print('Counts:\n')
for user_type in df.user_type.unique():
    print(f"{user_type} users : {df[df.user_type == user_type].shape[0]}")


# We may want to know if a user_type leans towards to a specific range of prices.

# In[12]:


sns.violinplot('user_type', 'price', data=df_dropped)


# As we have huge tails on our violin plot, we may want to use again our trimmed dataframe (price < 40k). Private & Company user_type seem to have fairly similar price distribution

# In[13]:


sns.violinplot('user_type', 'price', data=df_trimmed)


# The KDE of `deal_probability` behave the same way as the price : Shop `user_type` are a little bit different than the other two.

# In[14]:


hist_kws={"alpha": 0.2}
sns.distplot(df[df.user_type == 'Private']['deal_probability'], label='Private', hist_kws=hist_kws)
sns.distplot(df[df.user_type == 'Company']['deal_probability'], label='Company', hist_kws=hist_kws)
sns.distplot(df[df.user_type == 'Shop']['deal_probability'], label='Shop', hist_kws=hist_kws)
sns.distplot(df.deal_probability, label='All', hist_kws=hist_kws)
plt.title('Deal Probability distribution')
plt.legend()
plt.margins(0.02)
plt.show()


# ## Categorical features

# In[15]:


print('Counts\n')
print(f'region {len(df.region.unique())}')
print(f'city {len(df.city.unique())}')
print(f'parent cat {len(df.parent_category_name.unique())}')
print(f'cat {len(df.category_name.unique())}')


# In[ ]:




