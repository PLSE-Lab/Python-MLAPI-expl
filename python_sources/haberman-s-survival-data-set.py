#!/usr/bin/env python
# coding: utf-8

# In[ ]:



# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# The haberman dataset contains cases from a study that was conducted between 1958 and 1970 at the University of Chicago's Billings Hospital on the survival of patients who had undergone surgery for breast cancer.

# In[ ]:


data = pd.read_csv('../input/haberman.csv',names=['Age','OpYear','axilNodes', 'Survival_status'])
data.head(n=20).T


# Added the names of the columns to increase the readablitity.

# Changeing the datatype of survival_status to bool dtype

# In[ ]:


data['Survival_status'] = data['Survival_status'].map({1: 0 , 2 : 1})


# In[ ]:


data.head(n=20).T


# In[ ]:


data.info()


# In[ ]:


data.describe()


# In[ ]:


data.shape


# The data contains total 306 cases and four features

# In[ ]:


data['Survival_status'].value_counts().unique()


# More cases of surival than the other (Unbalanced data)

# In[ ]:


data['Survival_status'].value_counts(normalize = True)


# 73% cases are of survival and 23% cases are of unfortunate cases.

# In[ ]:


np.sum(data.isna())


# we got no NA values in all columns

# In[ ]:


#what is the mean age?
data['Age'].mean()


# In[ ]:


data['OpYear'][data['OpYear'].value_counts().max()]


# Maximum no of surrgery were in year 67.

# EDA
# 1 Univariate
# 2 Bivariate
# 3 Multivariate

# In[ ]:


import seaborn as sns


# In[ ]:


sns.pairplot(data=data, hue='Survival_status')


# Data is not linearly separated.

# The more the Dist Plots are separated the better.

# In[ ]:


sns.FacetGrid(data, hue="Survival_status", height = 5).map(sns.distplot, 'Age').add_legend()


# The people btw the range of 40-60 are more likely to not make more than 5 years after their surgury

# The PDF are not well seperated.But we can say people with the age of around 50 are more likely to survive.

# In[ ]:


sns.FacetGrid(data, hue="Survival_status", height = 5).map(sns.distplot, 'OpYear').add_legend()


# 

# In[ ]:


sns.FacetGrid(data, hue="Survival_status", height = 5).map(sns.distplot, 'axilNodes').add_legend()


# The lower number of axilNodes(0-3) corresepond to the survival of the patient.

# In[ ]:





# In[ ]:


plt.figure(figsize=(15,10))
plt.figure(1)
plt.subplot(211)
counts, bin_edges = np.histogram(data['axilNodes'], bins = 10 , density =True)
pdf = counts / sum(counts)
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:], pdf)
plt.plot(bin_edges[1:], cdf)
plt.xlabel('axilNodes')
plt.grid()

plt.subplot(212)
counts, bin_edges = np.histogram(data['Age'], bins = 10 , density =True)
pdf = counts / sum(counts)
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:], pdf)
plt.plot(bin_edges[1:], cdf)
plt.xlabel('Age')
plt.grid()


# In[ ]:





# In[ ]:


sns.boxplot(x= data['Survival_status'], y = data['Age'] )


# the top and down whskers are Q1/Q5 (+/-)IQR(inter quartile range) * 1.5.
# Q2 25th %ile
# Q3 50th %ile
# Q4 75th %ile

# In[ ]:


sns.boxplot(x= data['Survival_status'], y = data['OpYear'])


# 

# In[ ]:


sns.boxplot(x= data['Survival_status'], y = data['axilNodes'])


# 

# Violin plot are boxplot with the PDF

# In[ ]:


sns.violinplot(x= data['Survival_status'], y = data['Age'])


# People btw the age of 45-60 are more prone to die from surgery

# In[ ]:


sns.violinplot(x= data['Survival_status'], y = data['OpYear'])


# 

# In[ ]:


sns.violinplot(x= data['Survival_status'], y = data['axilNodes'])


# The axilNodes are more dense btw 0-5 whereas it is less dense for the other case.

# In[ ]:





# In[ ]:





# In[ ]:




