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


# In[ ]:


#reading csv file from local folder
survivalds = pd.read_csv("../input/habermans-survival-data-set/haberman.csv", header=None, names = ['Age', 'Op_Year', 'axil_nodes', 'Surv_status'])


# In[ ]:


print("columns ",survivalds.columns)


# In[ ]:


# check columns and observations
survivalds.shape


# In[ ]:


# columns 
survivalds.columns


# In[ ]:


# finding target variable (Surv_status) distribution
survivalds.Surv_status.value_counts()


# In[ ]:


#finding percentage of Surv_status distribution
survivalds.Surv_status.value_counts()/survivalds.Surv_status.shape[0]

##### Data is distributed 73.5 and 26.5 so data is balanced


# In[ ]:


# finding na fields
survivalds.isna().sum().sum()


# In[ ]:


# EDA libraries
import seaborn as sns
import matplotlib.pyplot as plt


# In[ ]:


sns.boxplot(data=survivalds,x="Surv_status",y="axil_nodes")
plt.show()


# In[ ]:


print(survivalds.axil_nodes.mean() ,"---",survivalds.axil_nodes.median())
survivalds[survivalds.axil_nodes < 4].Surv_status.value_counts()/survivalds[survivalds.axil_nodes < 4].Surv_status.count()


# In[ ]:


sns.boxplot(data=survivalds,x="Surv_status",y="Age")
plt.show()


# In[ ]:


sns.boxplot(data=survivalds,x="Surv_status",y="Op_Year")
plt.show()


# In[ ]:


sns.violinplot(data=survivalds,x="Surv_status",y="axil_nodes")
plt.show()


# In[ ]:


sns.violinplot(data=survivalds,x="Surv_status",y="Age")
plt.show()


# In[ ]:


sns.violinplot(data=survivalds,x="Surv_status",y="Op_Year")
plt.show()


# In[ ]:


survivalds.dtypes


# In[ ]:


survivalds_1 = survivalds.loc[survivalds["Surv_status"] == 1];
survivalds_2 = survivalds.loc[survivalds["Surv_status"] == 2];

#print(iris_setosa["petal_length"])
plt.plot(survivalds_1["Age"], np.zeros_like(survivalds_1['Age']), 'o')
plt.plot(survivalds_2["Age"], np.zeros_like(survivalds_2['Age']), 'o')

plt.show()


# In[ ]:


survivalds[(survivalds.Age > 40) & (survivalds.Age < 60)].Surv_status.value_counts()/survivalds[(survivalds.Age > 40) & (survivalds.Age < 60)].Surv_status.count()


# In[ ]:


plt.plot(survivalds_1["axil_nodes"], np.zeros_like(survivalds_1['axil_nodes']), 'o')
plt.plot(survivalds_2["axil_nodes"], np.zeros_like(survivalds_2['axil_nodes']), 'o')

plt.show()


# In[ ]:


plt.plot(survivalds_1["Op_Year"], np.zeros_like(survivalds_1['Op_Year']), 'o')
plt.plot(survivalds_2["Op_Year"], np.zeros_like(survivalds_2['Op_Year']), 'o')

plt.show()


# In[ ]:


sns.FacetGrid(survivalds, hue="Surv_status", size=5)    .map(sns.distplot, "Age")    .add_legend();
plt.show();


# In[ ]:


sns.FacetGrid(survivalds, hue="Surv_status", size=5)    .map(sns.distplot, "Op_Year")    .add_legend();
plt.show();


# In[ ]:


sns.FacetGrid(survivalds, hue="Surv_status", size=5)    .map(sns.distplot, "axil_nodes")    .add_legend();
plt.show();


# In[ ]:


counts, bin_edges = np.histogram(survivalds['Age'], bins=10, 
                                 density = True)
pdf = counts/(sum(counts))
print(pdf);
print(bin_edges);
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf);
plt.plot(bin_edges[1:], cdf)


counts, bin_edges = np.histogram(survivalds['Age'], bins=20, 
                                 density = True)
pdf = counts/(sum(counts))
plt.plot(bin_edges[1:],pdf);

plt.show();


# In[ ]:


counts, bin_edges = np.histogram(survivalds['axil_nodes'], bins=10, 
                                 density = True)
pdf = counts/(sum(counts))
print(pdf);
print(bin_edges);
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf);
plt.plot(bin_edges[1:], cdf)


counts, bin_edges = np.histogram(survivalds['axil_nodes'], bins=20, 
                                 density = True)
pdf = counts/(sum(counts))
plt.plot(bin_edges[1:],pdf);

plt.show();


# In[ ]:


counts, bin_edges = np.histogram(survivalds['Op_Year'], bins=10, 
                                 density = True)
pdf = counts/(sum(counts))
print(pdf);
print(bin_edges);
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf);
plt.plot(bin_edges[1:], cdf)


counts, bin_edges = np.histogram(survivalds['Op_Year'], bins=20, 
                                 density = True)
pdf = counts/(sum(counts))
plt.plot(bin_edges[1:],pdf);

plt.show();


# In[ ]:


np.percentile(survivalds.axil_nodes,np.arange(0,125,25)) #[0.,  0.,  1.,  4., 52]

survivalds.axil_nodes.mean() #4

survivalds.axil_nodes.median() # 1

survivalds.sort_values(by='axil_nodes').axil_nodes.unique()


# In[ ]:


survivalds[survivalds.axil_nodes < 2].Surv_status.value_counts()/survivalds[survivalds.axil_nodes < 2].Surv_status.shape[0]


# In[ ]:


plt.hist(bins=10,data=survivalds,x='Surv_status')

