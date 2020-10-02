#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import os as os
import seaborn as sns
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


#load the dataset 
haberman = pd.read_csv("../input/haberman.csv")


# In[ ]:


haberman.head()


# In[ ]:


haberman.columns=['age', 'Op_year', 'axil_nodes', 'Surv_status']


# In[ ]:


haberman.columns


# In[ ]:


haberman.isnull().values.any()


# In[ ]:


haberman['Surv_status'].value_counts()


# In[ ]:


haberman.shape


# In[ ]:


haberman.dtypes


# Observation:-
# 1. We have 4 columns & 306 rows which have integer values
# 2. Class_1 have 225 elements and Class_2 have 81 elements
# 3. There is no null value in dataset

# **Pair Plot**

# In[ ]:


sns.pairplot(haberman,hue='Surv_status',palette='Set1',vars=["age", "Op_year", "axil_nodes"],size=2.5)
plt.show()


# **Observation **
# No one column feature is important to classifie the distinguish between class_1 and class_2

#  **Univariate Analysis**

# In[ ]:


sns.FacetGrid(haberman,hue='Surv_status',size=5)   .map(sns.distplot,'age')   .add_legend()
plt.show()


# In[ ]:


sns.FacetGrid(haberman,hue='Surv_status',size=5).map(sns.distplot,'Op_year').add_legend()
plt.show()


# In[ ]:


sns.FacetGrid(haberman,hue='Surv_status',size=5).map(sns.distplot,'axil_nodes').add_legend()
plt.show()


# **Observation**
# 1. We found that age and Op_year and not good feature both are looking same
# 2. But axillary nodes is much difference in distribution of classification 

# In[ ]:


#haberman_1 defines class_1 elements 
haberman_1=haberman[haberman['Surv_status']==1]
#haberman_2 defines class_2 elements
haberman_2=haberman[haberman['Surv_status']==2]


# In[ ]:


count_1, bin_edges_1=np.histogram(haberman_1['axil_nodes'],bins=25,density=True)
count_2,bin_edges_2=np.histogram(haberman_2['axil_nodes'],bins=25,density=True)
pdf_1=count_1/sum(count_1)
pdf_2=count_2/sum(count_2)
cdf_1=np.cumsum(pdf_1)
cdf_2=np.cumsum(pdf_2)
plt.plot(bin_edges_1[1:],pdf_1)
plt.plot(bin_edges_1[1:],cdf_1)
plt.plot(bin_edges_2[1:],pdf_2)
plt.plot(bin_edges_2[1:],cdf_2)
plt.legend(["Survived > 5 years pdf_1", "Survived > 5 years cdf_1", 
            "Survived > 5 years pdf_2", "Survived < 5 years cdf_2" ])
plt.xlabel('axil_node')
plt.show()


# **Observation **
# 1. If axil_node <47 then patient is survived
# 2. If axil_node >47 then patient is not survived

# In[ ]:


sns.boxplot(data=haberman,x='Surv_status',y='age')
plt.show()


# In[ ]:


sns.boxplot(data=haberman,x='Surv_status',y='Op_year')
plt.show()


# In[ ]:


sns.boxplot(data=haberman,x='Surv_status',y='axil_nodes')
plt.show()


# In[ ]:


sns.violinplot(data=haberman,x='Surv_status',y='age')
plt.show()


# In[ ]:


sns.violinplot(data=haberman,x='Surv_status',y='axil_nodes')
plt.show()


# In[ ]:


sns.violinplot(data=haberman,x='Surv_status',y='Op_year')
plt.show()


# In[ ]:


sns.jointplot(data=haberman,x='age',y='Op_year',kind='kde')
plt.show()


# **Observation**
# * we have found that lots of patients  who has gone for operation between Year 60 to 65 

# In[ ]:




