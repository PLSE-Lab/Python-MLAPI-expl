#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams['figure.figsize']=(10, 6)


# # tl;dr

# 1. The mean number of products per user decreases over time. It takes a sizeable dive across all products from 2015-06-28 to 2015-07-28.
# 2. `ind_cco_fin_ult1` is the most-owned product.
# 3. The total number of products owned across all accounts increases over time. In effect, as times moves forward, we are adding more users, but each one of these users is owning less and less.

# In[ ]:


def read_csv_random_sample(path, nrows):
    total_rows_in_file_minus_header = sum(1 for line in open(path)) - 1
    skip_mask = random.sample(
        population=range(1, total_rows_in_file_minus_header + 1),
        k=total_rows_in_file_minus_header - nrows
    )
    return pd.read_csv(path, skiprows=skip_mask)


# In[ ]:


train = read_csv_random_sample(path="../input/train_ver2.csv", nrows=5000000)


# # Define product columns

# In[ ]:


PRODUCT_COLUMNS = [column for column in train.columns if column.endswith('ult1')]


# # How does the average number of products per user evolve over time?

# In[ ]:


train = train.assign(sum_of_products_owned = lambda df: df[PRODUCT_COLUMNS].sum(axis=1))


# In[ ]:


train.groupby('fecha_dato')['sum_of_products_owned'].mean()


# # How does individual product ownership evolve over time?

# In[ ]:


for product in PRODUCT_COLUMNS:
    train.groupby('fecha_dato')[product].mean().plot()


# ## Now, without the most popular product, `ind_cco_fin_ult1`

# In[ ]:


for product in PRODUCT_COLUMNS:
    if product != 'ind_cco_fin_ult1':
        train.groupby('fecha_dato')[product].mean().plot()


# # How does the total number of products owned across all accounts evolve over time?

# In[ ]:


train.groupby('fecha_dato')['sum_of_products_owned'].sum().plot(kind='bar')

