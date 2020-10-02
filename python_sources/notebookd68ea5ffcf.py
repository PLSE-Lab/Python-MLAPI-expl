#!/usr/bin/env python
# coding: utf-8

# My first exploration! Hope i do it right!

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
order_products_train = pd.read_csv('../input/order_products__train.csv')
order_products_prior = pd.read_csv('../input/order_products__prior.csv')
orders = pd.read_csv('../input/orders.csv')
products = pd.read_csv('../input/products.csv')
aisles = pd.read_csv('../input/aisles.csv')
departments = pd.read_csv('../input/departments.csv')


# In[ ]:


grouped = orders.groupby("order_id")["order_hour_of_day"].aggregate("sum").reset_index()
grouped = grouped.order_hour_of_day.value_counts()

plt.bar(grouped.index,grouped.values)


# In[ ]:


grouped = orders.groupby("user_id")["order_dow"].aggregate("sum").reset_index()
grouped = grouped.order_dow.value_counts()

plt.bar(grouped.index,grouped.values)


# In[ ]:


grouped = orders.groupby("user_id")["order_number"].aggregate("sum").reset_index()
grouped = grouped.order_number.value_counts()

plt.bar(grouped.index,grouped.values)

