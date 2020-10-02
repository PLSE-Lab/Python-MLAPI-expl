#!/usr/bin/env python
# coding: utf-8

# # Dataset Description
# 
# Number of Instances: 305
# 
# Number of Attributes: 4 (including the class attribute)
# 
# Attribute Information:
# 
# Age of patient at time of operation (numerical)
# 
# Patient's year of operation (year - 1900, numerical)
# 
# Number of positive axillary nodes detected (numerical)
# 
# Survival status (class attribute):
# 
# 1 = the patient survived 5 years or longer
# 
# 2 = the patient died within 5 year

# In[4]:


import pandas as pd
import numpy as np
import matplotlib.pyplot  as plt
import seaborn as sns

path='../input/haberman.csv'
hebarman=pd.read_csv(path)


# In[5]:


#No. of data point and features
print(hebarman.shape)
#There are 305 observation in dataset


# In[6]:


#adding columns names
hebarman.columns = ["age","year","axillary_node","survival_status"]
print(hebarman.columns)


# In[7]:


hebarman['survival_status'].value_counts()


# # Observation(s):
# 
# 1) Number of observation : 305
# 
# 2) The dataset is classified into two classes (Survived-1 and not survived-2)
# 
# 3) 225 patients of class 1, the patient survived 5 years or longer 
# 
# 4) 81 patients of class 2, the patient died within 5 year.
# 
# 5) It is imbalance dataset as the number of element in both class are unequal
# 
# 6) Number of Attribute is 4.
# 

# # 2-D Scatter plot

# In[8]:


#Plotting plain scatter plot between axillary_node and survival
hebarman.plot(kind="scatter",x="axillary_node",y="survival_status")
plt.show()


# In[9]:


sns.set_style("whitegrid")
sns.FacetGrid(hebarman,hue='survival_status',size=10).map(plt.scatter,"age","axillary_node").add_legend()
plt.show()


# # Observation:
# 
# 1)  Most of the patient having Axillary_node less than 3

# # Pair-Plot

# In[10]:


sns.set_style("whitegrid")
sns.pairplot(hebarman,hue='survival_status',size=3)
plt.show()


# # Observation's:
# 
# 1) There is a considerable overlap
# 
# 2) Any Pair-Plot is not giving clear idea, They are not linearly separable.
# 
# 

# # Histogram,PDF,CDF

# In[11]:


sns.FacetGrid(hebarman,hue='survival_status',size=5).map(sns.distplot,'year').add_legend()
plt.show()


# In[12]:


sns.FacetGrid(hebarman,hue='survival_status',size=5).map(sns.distplot,'age').add_legend()
plt.show()


# In[13]:


sns.FacetGrid(hebarman,hue='survival_status',size=5).map(sns.distplot,'axillary_node').add_legend()
plt.show()


# # Observation's:
# 
# 1)  The patient having axillary_node less than 2 have higher chance of survival.
#   
# 2)  Age and Year are not good features for useful insights as the distibution is more similar for both people who survived and also dead.
# 

# In[16]:


survive = hebarman.loc[hebarman['survival_status']==1]
not_survive = hebarman.loc[hebarman['survival_status']==2]


# In[17]:


counts, bin_edges = np.histogram(survive['axillary_node'], bins=10, 
                                 density = True)
pdf = counts/(sum(counts))

#compute CDF
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf)
plt.plot(bin_edges[1:], cdf)
plt.legend(['Pdf for the patients who survive more than 5 years',
            'Cdf for the patients who survive more than 5 years'])



plt.show();


# In[18]:


counts, bin_edges = np.histogram(not_survive['axillary_node'], bins=10, 
                                 density = True)
pdf = counts/(sum(counts))

#compute CDF
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf)
plt.plot(bin_edges[1:], cdf)
plt.legend(['Pdf for the patients who died within 5 years',
            'Cdf for the patients who died within 5 years'])



plt.show();


# # Observation:
# 
# 1)  From both the cdf we can conclude that person having axillary_node 46 are not survived

# In[19]:


print(survive.describe())


# In[20]:


print(not_survive.describe())


# # Observation's:
# 
# 1)  In Both the table everything is almost same except the mean of axillary_node .
# 
# 
# 2) 75 percentage of patient having axillary_node <=2 survived.
# 
# 
# 3)  The maximum and minimum age of patient who survived is 77 and 30 years respectively.
# 
# 
# 4)  The maximum and minimum age of patient who dies is 83 and 34 years respectively.
# 

# # Box Plot

# In[21]:


sns.boxplot(x='survival_status',y='axillary_node', data=hebarman)
plt.show()


# # Observation:
# 
# 
# 1)  75 percentage of patient having axillary_node <= 2 survived.
# 
# 2) We can see some outlier in box plot (Black point above the whisker)

# # Violin Plot

# In[22]:


sns.violinplot(x='survival_status',y='axillary_node', data=hebarman,size=8)
plt.show()


# # Observation:
# 
# 1)  From this plot we conclude that the most patient who survived had axillary_node <= 2.

# # Conclusion:
# 
# 1)  Year of operations  have no effect on survival status
# 
# 2)  Numbers of axillary-node effect the survival status. 
# 
# 3)  There are outlier in axillary_node in the dataset.
# 

# In[ ]:




