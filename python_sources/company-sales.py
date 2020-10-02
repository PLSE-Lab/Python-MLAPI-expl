#!/usr/bin/env python
# coding: utf-8

# This Kernel shows the **evolution of sales** per salesperson in our company.
# 
# This is our raw sales information:

# In[1]:


## Shows sales in a text format

import csv
with open('../input/sales.csv', newline='') as csvfile:
  numReader = csv.reader(csvfile, delimiter=',')
  for row in numReader:
    print(', '.join(row))


# This is a chart comparing our Salespeople:

# In[17]:


## Create a per-quarter chart of the data
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
sales = pd.read_csv('../input/sales.csv')
sales_chart = sales.groupby("Quarter").sum().plot(kind='line')

