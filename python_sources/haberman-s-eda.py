#!/usr/bin/env python
# coding: utf-8

# # Haberman's Survival Data Set
# ## Objective
# Perform an exploratory data analysis on breast cancer patients
# 
# ## Dataset description
# Attributes information
# a. Age of patient at time of operation (numerical)
# b. Patient's year of operation (year - 1900, numerical)
# c. Number of positive axillary nodes detected (numerical)
# d. Survival status (class attribute)
# 
#    * 1 = the patient survived for 5 year or longer
#    * 2 = the patient survived for less than 5 years
#                    
# 

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


col =['Age','Year-of-operation','Pos-axillary-nodes','Status']
df = pd.read_csv('../input/haberman.csv',names = col)


# In[ ]:


df.columns


# In[ ]:


df.head()


# In[ ]:


#Number of data points and features
print(df.shape)


# In[ ]:


#Number of classes and points per class
df['Status'].value_counts()

#It is an imbalanced dataset


# ## Conclusions
# 
# 1. There are __306__ observations with __4__ features in the data set
# 2. It is an imbalanced dataset with
#    * __225__ patients belonging to status 1, those who survived for 5 years and longer 
#    * __81__ patients belonging to status 2, those who survived for less than 5 years

# In[ ]:


sns.set_style('whitegrid');
sns.FacetGrid(df, hue = 'Status',size = 4).map(plt.scatter,'Age', 'Pos-axillary-nodes').add_legend();
plt.show();


# # Conclusion
# 
# * Using scatter plot we cannot distinguish people who survived and who didn't survive

# In[ ]:


print("Mean age of patients who survived:",(round(np.mean(df[df['Status'] == 1]['Age']))))
print("Mean age of patients who didnot survive:",(round(np.mean(df[df['Status']==2]['Age']))))


# In[ ]:


sns.pairplot(data = df, hue = 'Status', size = 3)


# In[ ]:


sns.FacetGrid(df, hue = 'Status', size = 5).map(sns.distplot, "Age").add_legend();
plt.show();


# In[ ]:


sns.FacetGrid(df, hue = 'Status', size = 5).map(sns.distplot, "Year-of-operation").add_legend();
plt.show();


# In[ ]:


sns.FacetGrid(df, hue = 'Status', size = 5).map(sns.distplot, "Pos-axillary-nodes").add_legend();
plt.show();


# # Conclusions
# *  Pos-axillary-nodes is a useful feature to identify the status,survival of cancer patients
# *  Age and Year of operation have overlapping curves which makes difficult for classifying the status
# *  The survived people mostly fall into zero pos-axillary-nodes
# *  Mean age of patients who survived is 52 years and who didn't survive is 54 years
# 

# In[ ]:


sur = df[df['Status'] == 1]
not_sur = df[df['Status'] == 2]


# In[ ]:


counts,bin_edges = np.histogram(not_sur['Pos-axillary-nodes'],bins = 30, density = True)
pdf = counts/(sum(counts))
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf)
plt.plot(bin_edges[1:],cdf)
plt.legend()

counts,bin_edges = np.histogram(sur['Pos-axillary-nodes'],bins = 30, density = True)
pdf = counts/(sum(counts))
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf)
plt.plot(bin_edges[1:],cdf)

plt.legend(['sur_pdf','not_sur_pdf','sur_cdf','not_sur_cdf'])
plt.xlabel('Pos-axillary-nodes')
plt.grid()


# In[ ]:


sns.boxplot(x = 'Status', y = 'Pos-axillary-nodes' , data = df)
plt.show()


# In[ ]:


sns.violinplot(x = 'Status', y = 'Pos-axillary-nodes' , data = df)
plt.show()


# In[ ]:


print("\n90th Percentiles:")
print(np.percentile(sur["Pos-axillary-nodes"],90))
print(np.percentile(not_sur["Pos-axillary-nodes"],90))


# # Conclusions
# * From the boxplot and violin plots its evident that most people who survived cancer have zero positive axillary nodes
