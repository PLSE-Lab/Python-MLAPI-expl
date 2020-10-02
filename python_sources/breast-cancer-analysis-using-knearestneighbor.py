#!/usr/bin/env python
# coding: utf-8

# In[ ]:


data = pd.read_csv('../input/breast-cancer-csv/breastCancer.csv')


# # Load Libraries

# In[ ]:


from sklearn.datasets import load_breast_cancer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# # Allocate variable to dataset and print description

# In[ ]:


cells = load_breast_cancer()
print(cells.DESCR)


# # Identify Target Names

# In[ ]:


print(cells.target_names)


# # Identify and show data type. 

# In[ ]:


type(cells.data)


# In[ ]:


cells.data


# # Quantify (nSamples, nFeatures) i.e.  rows and columns

# In[ ]:


cells.data.shape


# In[ ]:





# # Import pandas  
# # Switch Datasets for more instances  
# # Read last 10 instances of new dataframe.

# In[ ]:


import pandas as pd
raw_data=pd.read_csv('../input/breast-cancer-csv/breastCancer.csv')
raw_data.tail(10)


# # Install  for visualization we don't have permissions to load mods on kaggle but the following line of code will work like butter on your local machine using Anaconda Jupyter! I will show code and a link to image of output!

# !pip install mglearn  
# 
# import mglearn  
# 
# mglearn.plots.plot_knn_classification(n_neighbors=5)  
# 
# 
# ![kNN mglearn plot](https://www.dropbox.com/s/tp3ebx4ud7vnrfj/kNNmglearnplot.png?dl=0)  
