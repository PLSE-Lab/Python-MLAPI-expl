#!/usr/bin/env python
# coding: utf-8

# **Load Data**
# Load your data as a Pandas DataFrame.
# 
# DataFrame is a Multidimensional Array
# 
# *data.shape* lets you check how many rows and columns is in your dataset

# In[ ]:


import pandas
#names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = pandas.read_csv('../input/diabetes.csv')
print(data.shape)


# **Look At Sample Of Data**
# 

# In[ ]:


data.head(10)


# **BASIC DESCRIPTIVE STATISTICS FOR UNDERSTANDING DATA**

# Applied Machine Leaning works best with numbers. *data.dtypes* helps you check if any attribute might need convertion.

# In[ ]:


data.dtypes


# In[ ]:


data.describe()


# **Class Distribution**
# 
# Balance matters in Applied Machine Learning. Imbalanced Classification Datasets might lead to biased models

# In[ ]:


data.groupby('Outcome').size()


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt

data.hist()
plt.show()

