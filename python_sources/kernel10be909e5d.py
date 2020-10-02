#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# # **DataSet information:**
# 
# * Data Description The Haberman's survival dataset contains cases from a study that was conducted between 1958 and 1970 at the University of Chicago's 
# * Billings Hospital on the survival of patients who had undergone surgery for breast cancer.
# 
# 
# # **Objective**
# 
# * To predict whether the patient will survive after 5 years or not based upon the patient's age, year of treatment and the number of positive lymph nodes

# In[ ]:


import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

# reading the dataset as pandas dataframe
cols_names=['Age','Year_of_operation','Axillary_nodes_detected','Survival_status']
df_cancer=pd.read_csv('/kaggle/input/habermans-survival-data-set/haberman.csv',names=cols_names)
df_cancer.head()


# In[ ]:


#checking the info
df_cancer.info()


# In[ ]:


#checking the shape
df_cancer.shape


# In[ ]:


# Q3) Numerical statics of the features 
df_cancer.describe().transpose()


# In[ ]:


#Checking the value of the Survival status

df_cancer.Survival_status.value_counts()


# In[ ]:


df_cancer.Survival_status=df_cancer['Survival_status'].map({1:"Yes",2:"No"})
df_cancer.Survival_status=df_cancer.Survival_status.astype('category')
df_cancer.info()


# In[ ]:


df_cancer.head()
df_cancer.describe()


# # Univarite analysis

# In[ ]:


sns.FacetGrid(df_cancer,hue='Survival_status',height=5)    .map(sns.distplot,'Age')    .add_legend()
plt.show()


# In[ ]:


sns.FacetGrid(df_cancer,hue='Survival_status',height=5).map(sns.distplot,'Year_of_operation').add_legend()
plt.show()


# In[ ]:


sns.FacetGrid(df_cancer,hue='Survival_status',height=5, aspect=2).map(sns.distplot,'Axillary_nodes_detected').add_legend()
plt.show()


# # PDF and CDF

# In[ ]:


df_survival=df_cancer.loc[df_cancer.Survival_status=='Yes']
df_death=df_cancer.loc[df_cancer.Survival_status=='No']

#survival People
counts,bin_edges=np.histogram(df_survival.Axillary_nodes_detected,bins=10,density=True)
pdf=counts/(sum(counts))
print(pdf)
print(bin_edges)
cdf=np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf)
plt.plot(bin_edges[1:],cdf)

#died people

counts,bin_edges=np.histogram(df_death.Axillary_nodes_detected,bins=10,density=True)
pdf=counts/sum(counts)
print(pdf)
cdf=np.cumsum(pdf);
plt.plot(bin_edges[1:],pdf)
plt.plot(bin_edges[1:],cdf)
plt.show()


# # Observations: 
#     
#    1. people less than 5 axillary nodes will survive more, as per the CDF we can see the survival chance is 80%

# In[ ]:


sns.boxplot(x='Survival_status',y='Axillary_nodes_detected',data=df_cancer)


# In[ ]:


sns.boxplot(x='Survival_status',y='Year_of_operation',data=df_cancer)


# # Observations:
# 
# 1. people having higher survival rate in the operated year 1966, compared to other years like 1959-1961

# In[ ]:


df_survival['Axillary_nodes_detected'].describe().transpose()


# In[ ]:


df_death['Axillary_nodes_detected'].describe().transpose()


# # Observation: 
# 
# 1. 75% of people survived with 3 or less axillary nodes detected
# 2. people who are not survived having more axillary nodes.

# 

# 

# 

# 
