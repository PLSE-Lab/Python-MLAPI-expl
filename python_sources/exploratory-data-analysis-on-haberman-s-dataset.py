#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

cols = ['Age', 'Operation_year', 'Axil_nodes', 'Survival_status']
hb = pd.read_csv('../input/haberman.csv', names = cols)

# Input data files are available in the "../input/" directory.


# In[ ]:


print(hb.shape)


# In[ ]:


print(hb.columns)


# In[ ]:


hb["Survival_status"].value_counts()


# In[ ]:


#statistical parameters
hb.describe()


# **Observations**
# a) Number of records are 306
# 
# b) The median age of patients is 52 and the range being 30 to 83 
# 
# c) 75% of the patients have less than 5 axil nodes and whereas, 25% of the patients have no axil nodes at all.
# 
# d) Classes are imbalanced with 225 survived and 81 dead. 

# In[ ]:


#Bivariate Analysis using scatter plots 
sns.set_style("whitegrid");
sns.FacetGrid(hb, hue="Survival_status", size=4)    .map(plt.scatter, "Axil_nodes", "Operation_year")    .add_legend();
plt.show();


# In[ ]:


#Bivariate Analysis using Pairwise plots 
plt.close();
sns.set_style("whitegrid");
sns.pairplot(hb, hue="Survival_status", size=3);
plt.show()


# In[ ]:


#Univariate analysis
patient_survived = hb.loc[hb["Survival_status"] == 1];
patient_died = hb.loc[hb["Survival_status"] == 2];
plt.plot(patient_survived["Axil_nodes"], np.zeros_like(patient_survived['Axil_nodes']), 'o')
plt.plot(patient_died["Axil_nodes"], np.zeros_like(patient_died['Axil_nodes']), 'o')
plt.show()


# In[ ]:


sns.FacetGrid(hb, hue="Survival_status", size=5)    .map(sns.distplot, "Axil_nodes")    .add_legend();
plt.show();


# In[ ]:


sns.FacetGrid(hb, hue="Survival_status", size=5)    .map(sns.distplot, "Age")    .add_legend();
plt.show();


# In[ ]:


sns.FacetGrid(hb, hue="Survival_status", size=5)    .map(sns.distplot, "Operation_year")    .add_legend();
plt.show();


# In[ ]:


##Univariate analysis using PDF and CDF
counts, bin_edges = np.histogram(patient_survived['Axil_nodes'], bins=10, 
                                 density = True)
pdf = counts/(sum(counts))
print(pdf);
print(bin_edges);
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf);
plt.plot(bin_edges[1:], cdf)


counts, bin_edges = np.histogram(patient_died['Axil_nodes'], bins=10, 
                                 density = True)
pdf = counts/(sum(counts))
print(pdf);
print(bin_edges)
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf)
plt.plot(bin_edges[1:], cdf)
plt.show();


# In[ ]:


#Mean, Variance, Std-deviation,  
print("Means:")
print(np.mean(patient_survived["Axil_nodes"]))
print(np.mean(patient_died["Axil_nodes"]))
print("\nStd-dev:");
print(np.std(patient_survived["Axil_nodes"]))
print(np.std(patient_died["Axil_nodes"]))

#Median, Quantiles, Percentiles, IQR.
print("\nMedians:")
print(np.median(patient_survived["Axil_nodes"]))
print(np.median(patient_died["Axil_nodes"]))

print("\nQuantiles:")
print(np.percentile(patient_survived["Axil_nodes"],np.arange(0,100,25)))
print(np.percentile(patient_died["Axil_nodes"],np.arange(0,100,25)))

print("\n90th Percentiles:")
print(np.percentile(patient_survived["Axil_nodes"],90))
print(np.percentile(patient_died["Axil_nodes"],90))

from statsmodels import robust
print ("\nMedian Absolute Deviation")
print(robust.mad(patient_survived["Axil_nodes"]))
print(robust.mad(patient_died["Axil_nodes"]))


# In[ ]:


#Univariate analysis using Boxplot
sns.boxplot(x='Survival_status',y='Axil_nodes', data=hb)
plt.show()
sns.boxplot(x='Survival_status',y='Age', data=hb)
plt.show()
sns.boxplot(x='Survival_status',y='Operation_year', data=hb)
plt.show()


# In[ ]:


#Univariate analysis using Violinplot
sns.violinplot(x='Survival_status',y='Axil_nodes', data=hb)
plt.show()
sns.violinplot(x='Survival_status',y='Axil_nodes', data=hb)
plt.show()
sns.violinplot(x='Survival_status',y='Age', data=hb)
plt.show()


# In[ ]:


sns.jointplot(x="Operation_year", y="Age", data=hb, kind="kde");
plt.show();
sns.jointplot(x="Axil_nodes", y="Age", data=hb, kind="kde");
plt.show();


# **Conclusions from EDA**
# -> Patients who survived have lower no. of auxilary nodes compared patoents who din't survive.
# 
# -> Out of all 3 features, axil_nodes is most important.
# 
# -> Most no. of surgeries were performed between 1960 - 64
# 
# -> Most of the patients who had undergone surgery were aged between 42-60 years.
# 
# -> People with auxilary nodes less than 5 had higher chances of survival
# 
# -> People with ages less than 40 had 90% chances of survival

# In[ ]:




