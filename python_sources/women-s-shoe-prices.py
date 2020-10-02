#!/usr/bin/env python
# coding: utf-8

# **Women's Shoe Prices**:
# 
# This is a data about 10,000 women's shoes and their associated information from here.
# 
# The description of each columns can be found here.
# 
# Here is the questions is going to be answered:
# 
# 1. What is the average price of each distinct brand listed?
# 2. Which brands have the highest prices? 
# 3. Which ones have the widest distribution of prices?
# 4. Is there a typical price distribution (e.g., normal) across brands or within specific brands?

# In[ ]:


# Importing the Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


# Importing the Data
shoes = pd.read_csv('../input/womens-shoes-prices/7210_1.csv')
shoes.head()


# **DATA CLEANING**

# In[ ]:


# Here only choose 5 columns interesting.
columns = ['brand', 'prices.amountMin', 'prices.amountMax', 'prices.isSale', 'prices.currency']


# In[ ]:


shoes = shoes[columns]
shoes.dropna(inplace=True)
shoes['prices.amountAverage'] = (shoes['prices.amountMin'] + shoes['prices.amountMax']) / 2


# In[ ]:


shoes['prices.currency'].value_counts()


# In[ ]:


# Show some not USD shoes
shoes[shoes['prices.currency'] != 'USD'].head(10)


# In[ ]:


# Transform different currency to USD
for row in shoes.itertuples():
    if row._5 == 'CAD':
        shoes['prices.amountMin'][row.Index] *= 0.73
        shoes['prices.amountMax'][row.Index] *= 0.73
        shoes['prices.amountAverage'][row.Index] *= 0.73
    elif row._5 == 'EUR':
        shoes['prices.amountMin'][row.Index] *= 1.1
        shoes['prices.amountMax'][row.Index] *= 1.1
        shoes['prices.amountAverage'][row.Index] *= 1.1
    elif row._5 == 'AUD':
        shoes['prices.amountMin'][row.Index] *= 0.75
        shoes['prices.amountMax'][row.Index] *= 0.75
        shoes['prices.amountAverage'][row.Index] *= 0.75
    elif row._5 == 'GPB':
        shoes['prices.amountMin'][row.Index] *= 1.3
        shoes['prices.amountMax'][row.Index] *= 1.3
        shoes['prices.amountAverage'][row.Index] *= 1.3


# In[ ]:


# Make sure it did the transformation
shoes[shoes['prices.currency'] != 'USD'].head(10)


# **What is the average price of each distinct brand listed? Which brands have the highest prices?**
# 
# The brands which have highest price are list here, like "JewelsObsession", "Valentino", "Simone Rocha" etc.

# In[ ]:


data = shoes.groupby('brand')['prices.amountAverage'].mean().sort_values(ascending=False).head(10)
ax = data.plot(kind='barh', figsize=(10, 6))
ax.invert_yaxis()
plt.xlabel('Price(USD)')
plt.title('Most experience average price brand')


# In[ ]:


data = shoes.groupby('brand')['prices.amountAverage'].max().sort_values(ascending=False).head(10)
ax = data.plot(kind='barh', figsize=(10, 6))
ax.invert_yaxis()
plt.xlabel('Price(USD)')
plt.title('Most experience single price brand')


# **Which ones have the widest distribution of prices?**
# 
# As is shown, brands have widest distribution are "JewelsObsession", "Gucci", "Ralph Lauren" etc, with descending order.

# In[ ]:


grouped = shoes.groupby('brand')['prices.amountAverage']
data = grouped.apply(lambda x:x.max() - x.min()).sort_values(ascending=False).head(10)
ax = data.plot(kind='barh', figsize=(10, 6))
ax.invert_yaxis()
plt.xlabel('Price(USD)')
plt.title('Most widest distributed price brand')


# **Is there a typical price distribution (e.g., normal) across brands or within specific brands?**

# In[ ]:


shoes['prices.amountAverage'].hist(bins=50, figsize=(10, 6))
plt.title('Prices Distribution across brands')


# In[ ]:


shoes['brand'].value_counts().head(10)


# In[ ]:


fig, axs = plt.subplots(2, 5, figsize=(15, 6))
for idx, brand in enumerate(shoes['brand'].value_counts()[0:10].index):
    axs[idx//5, idx%5].hist(shoes[shoes['brand'] == brand]['prices.amountAverage'], bins=20)
    axs[idx//5, idx%5].set_title(brand)
plt.suptitle('Price Distributions of Specific Brand')
plt.tight_layout()
fig.subplots_adjust(top=0.88)


# **Now, I can say: prices across brands didn't show a normal distribution. However, in specific brands like "Ralph Lauren" and "Skechers", there are normal distribution in their prices.**

# In[ ]:




