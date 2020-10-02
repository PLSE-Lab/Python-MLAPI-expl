#!/usr/bin/env python
# coding: utf-8

# # Apple Store
# <br>
# **Contents**
# 1. [Storing data](#1)
# 2. [Comparison of Price vs. Size](#2)
# 3. [Number of apps in each category](#3)

# In[ ]:


import numpy as np
import pandas as pd
pd.set_option('max_rows', 9)
import matplotlib.pyplot as plt
import seaborn as sns
import os
print(os.listdir("../input"))


# <span id="1"></span>
# # Storing data

# In[ ]:


apps = pd.read_csv('../input/AppleStore.csv')
apps


# <span id="2"></span>
# # Comparison of Price vs. Size

# In[ ]:


price = apps.loc[:, ['track_name', 'price', 'size_bytes']]
price


# In[ ]:


plt.scatter(price.price, price.size_bytes / (1024 ** 2))

plt.title('Price vs. Size')
plt.xlabel('Price (USD)')
plt.ylabel('Size (MB)')

plt.show()


# Clean data by removing outliers (any app with price >= $50)

# In[ ]:


price_cleaned = price.loc[(price.price <= 50)]
price_cleaned


# In[ ]:


plt.scatter(price_cleaned.price, price_cleaned.size_bytes / (1024 ** 2))

plt.title('Price vs. Size (without outliers)')
plt.xlabel('Price (USD)')
plt.ylabel('Size (MB)')

plt.show()


# <span id="3"></span>
# # Number of apps in each category

# In[ ]:


plt.figure(figsize=(10, 7))
ax = sns.countplot(y=apps.prime_genre)
ax.set(xlabel='Number of apps', ylabel='Category')
plt.show()

