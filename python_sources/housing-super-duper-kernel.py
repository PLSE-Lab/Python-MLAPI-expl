#!/usr/bin/env python
# coding: utf-8

# # Predicting Housing Prices in CA, USA

# ## Introduction

# In this study we choose the dataset of the California Housing Prices as can be found in [this](https://www.kaggle.com/harrywang/housing#housing.csv) link. The dataset is based on data from the 1990 California census.

# <a>

# ## Data Preparation

# ## Models Presentation

# ## Solution Presentation

# ## Model Finetuning

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O 
import matplotlib.pyplot as plt # various graphs
from sklearn.model_selection import train_test_split
from pandas.plotting import scatter_matrix
import os #system i/o


project_dir="../input"
housing = pd.read_csv(project_dir+"/housing.csv")


# ## Data Exploration

# After having import the dataset, and having named it *housing*, let's take a close look at the top five rows via the _head_ command:

# In[ ]:


housing.head()


# Each district is represented by a single row, and holds 10 attributes as can be seen via the _info_ command. In our dataset there are 20640 rows spanning in 10 attributes:

# In[ ]:


housing.info()


# We utilize the _info()_ method in order to obtain a quick description of the data. Namely we get the total numbers of rows, along with each attribute's type. Additionally we obtain the number of non-null values. Let us note that the type of these 9 attributes is numeric. Concerning the __total_bedrooms__ there are 20433 out of 20640 rows, implying the existence of 207 NA's/Null values for these regions. We will get back to the value imputation at a later step.

# By using the _describe()_ method, we obtain the range of the numeric values along with metrics such as: the *mean*, *std*, *min* and *max* and of course the percentiles for 25%, 50, 75% ($1^{st}$, $2^{nd}$ and $3^{rd} $ quartile) of the total population. In our example, 25% of the total districts exhibit **housing_median_age** less than 18 years, while 50% are lower that 29 years and 75% lower than 37 years respectively.

# In[ ]:


housing.describe()


# Concerning the **ocean_proximity**, its type is categorical and in order to find out what categories exist and how many districts belong to each category. The meaning of this attribute has to do (very roughly) whether each block group is near the ocean, near the Bay area, inland or on an island. we utilize the *value_counts()* method:

# In[ ]:


housing["ocean_proximity"].value_counts()


# Sometimes it is more convenient to obtain a better glimpse of the data by checking the graphical distributions of the attributes. Luckily we make use of the intrisic capabilities of *matplotlib* module:

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
housing.hist(bins=50,figsize=(30,24))
plt.show()


# As far as the **ocean_proximity** is concerned, its distribution can be depicted in the following graph:

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
housing['ocean_proximity'].value_counts().plot(kind='bar')
plt.show()


# In[ ]:


housing.plot(kind="scatter",x="longitude",y="latitude",alpha=0.3,s=housing["population"]/100,label="population",figsize=(15,12),c="median_house_value",cmap=plt.get_cmap("jet"),colorbar=True)
plt.show()


# ### Checking for correlations

# In[ ]:


corr_mat=housing.corr()
corr_mat


# In[ ]:


corr_mat["median_house_value"].sort_values(ascending=False)


# In[ ]:


sub_attributes=["median_house_value","median_income","total_rooms","housing_median_age"]
scatter_matrix(housing[sub_attributes],figsize=(20,15))
plt.show()

