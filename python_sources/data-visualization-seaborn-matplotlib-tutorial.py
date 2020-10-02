#!/usr/bin/env python
# coding: utf-8

# ***
# ### **I hope you find this kernel useful and your <font color="red"><b>UPVOTES</b></font> would be highly appreciated**
# ***

# In[ ]:


import pandas as pd
import numpy as np


# In[ ]:


drinks_df = pd.read_csv("../input/drinks-by-country/drinksbycountry.csv")
drinks_df.head()


# # Data exploration

# In[ ]:


drinks_df.info()


# In[ ]:


drinks_df.describe()


# In[ ]:


drinks_df.isna().sum()


# # Data visualization

# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


sns.pairplot(drinks_df, hue='continent', height=3, aspect=1)


# In[ ]:


sns.pairplot(drinks_df, hue='continent', height=3, diag_kind="hist");


# In[ ]:


drinks_df.hist(edgecolor='black', linewidth=1.2, figsize=(12, 8));


# In[ ]:


plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
sns.violinplot(x='continent', y='beer_servings', data=drinks_df)

plt.subplot(2, 2, 2)
sns.violinplot(x='continent', y='spirit_servings', data=drinks_df)

plt.subplot(2, 2, 3)
sns.violinplot(x='continent', y='wine_servings', data=drinks_df)

plt.subplot(2, 2, 4)
sns.violinplot(x='continent', y='total_litres_of_pure_alcohol', data=drinks_df);


# In[ ]:


drinks_df.boxplot(by='continent', figsize=(12, 8));


# In[ ]:


pd.plotting.scatter_matrix(drinks_df, figsize=(12, 10));


# In[ ]:


sns.pairplot(drinks_df)


# In[ ]:


plt.figure(figsize=(12, 8))
sns.heatmap(drinks_df.corr(), linewidths=1, annot=True)

