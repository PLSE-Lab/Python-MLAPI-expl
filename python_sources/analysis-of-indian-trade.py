#!/usr/bin/env python
# coding: utf-8

# # Indian Trade Data
# This notebook has the purpose of analyze the Indian import and export trying to understand possible patterns useful to create predictive models. The used dataset can be found in [https://www.kaggle.com/lakshyaag/india-trade-data](http://).

# ## Importing the libraries

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# ## Reading the data
# First of all, it is necessary to analyze the data and get some informations about them before start to use them. The code snippet below shows which is the csv files in the input directory.

# In[ ]:


# Input data files are available in the "../input/" directory.
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


import_data = pd.read_csv("/kaggle/input/india-trade-data/2018-2010_import.csv")
export_data = pd.read_csv("/kaggle/input/india-trade-data/2018-2010_export.csv")


# ## Exploring the data

# In[ ]:


import_data.head()


# In[ ]:


export_data.head()


# ### Understanding the fields
# * **HSCode**: also known as **Harmonized System** (HS) or **Harmonized Commodity Description and Coding System**, is an internationally standardized system of names and numbers to classify traded products.
# * **Commodity**: is the description of the commodity related to the HSCode.
# * **value**: has the values of import or export (in million US$).
# * **country** and **year**: have the countries and years the trades have made and all data are sorted by them.

# ## Generating charts

# In[ ]:


plt.figure(figsize= (15,5))
sns.lineplot(x='year',y='value', data=import_data, label='Imports')
sns.lineplot(x='year',y='value', data=export_data, label='Exports')
plt.title('Values of Indian imports and exports', fontsize=16)
plt.xlabel('Year')
plt.ylabel('Value in million US$')
plt.show()

