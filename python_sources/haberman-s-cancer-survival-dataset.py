#!/usr/bin/env python
# coding: utf-8

# # Haberman's Survival Dataset
# 
# Dataset:['https://www.kaggle.com/gilsousa/habermans-survival-data-set']
# * Simple dataset contains patient records who had undergone surgery for breast cancer.
# * Objective: Label new patients belonging to one of the 2 classes-Survived or Dead
# 

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

habers=pd.read_csv('../input/haberman.csv')


# In[ ]:


#(Q) What are the column names?
#(Q) How many data points and features?
habers.info()


# * Datapoints:306
# * Features:4
# * Shape:(306,4)

# In[ ]:


habers.shape


# In[ ]:


habers['status'].value_counts()


# * Dataset is imbalenced
# * Staus(feature)
#    * 1=Survived
#    * 2=Dead

# ### Pair-Plot

# In[ ]:


sns.set_style('whitegrid')
sns.pairplot(habers,hue='status',size=3)
plt.show()


# ##### Observation:
# Not much can be concluded from the pair-plot as we are not able to distinguish between classes. However, Auxilary Nodes(detected_pos_nodes) plot is showing some classification

# ### 1-D Scatter Plot

# In[ ]:


habers["status"] = habers["status"].apply(lambda y: "Survived" if y == 1 else "Dead")
survived=habers.loc[habers['status'] == 'Survived']
dead=habers.loc[habers['status'] == 'Dead']

plt.plot(survived['detected_pos_nodes'], np.zeros_like(survived['detected_pos_nodes']),'o')
plt.plot(dead['detected_pos_nodes'], np.zeros_like(dead['detected_pos_nodes']),'x')
plt.show()


# ##### Observation:
# Many overlaps between class atribute and hence cannot be seperated

# ### Describe

# In[ ]:


survived.describe()


# In[ ]:


dead.describe()


# ### Histogram, PDF, CDF

# In[ ]:


sns.FacetGrid(habers, hue='status', size=5)     .map(sns.distplot, 'age')     .add_legend()

plt.show()


# In[ ]:


sns.FacetGrid(habers, hue='status', size=5)     .map(sns.distplot, 'year_of_op')     .add_legend()

plt.show()


# In[ ]:


sns.FacetGrid(habers, hue='status', size=5)     .map(sns.distplot, 'detected_pos_nodes')     .add_legend()

plt.show()


# In[ ]:


#CDF of Auxilary nodes
#Survived

counts, bin_edges = np.histogram(survived['detected_pos_nodes'], bins=10, 
                                 density = True)
pdf = counts/(sum(counts))
print(pdf);
print(bin_edges)

#compute CDF
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf)
plt.plot(bin_edges[1:], cdf)



plt.show();


# ##### Observation:
# * Most patients who survived had Auxilary nodes < 10(~90%)

# In[ ]:


#CDF of Age
#Survived
counts, bin_edges = np.histogram(survived['age'], bins=10, 
                                 density = True)
pdf = counts/(sum(counts))
print(pdf);
print(bin_edges)

#compute CDF
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf)
plt.plot(bin_edges[1:], cdf)



plt.show();


# In[ ]:


#CDF of Year of Operation
#Survived

counts, bin_edges = np.histogram(survived['year_of_op'], bins=10, 
                                 density = True)
pdf = counts/(sum(counts))
print(pdf);
print(bin_edges)

#compute CDF
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf)
plt.plot(bin_edges[1:], cdf)



plt.show();


# In[ ]:


#CDF of Auxilary Nodes
#Dead

counts, bin_edges = np.histogram(dead['detected_pos_nodes'], bins=10, 
                                 density = True)
pdf = counts/(sum(counts))
print(pdf);
print(bin_edges)

#compute CDF
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf)
plt.plot(bin_edges[1:], cdf)



plt.show();


# ##### Observation:
# Most Patients having Auxilar Nodes > 27 Died(~98%)

# In[ ]:


#CDF of Age
#Dead

counts, bin_edges = np.histogram(dead['age'], bins=10, 
                                 density = True)
pdf = counts/(sum(counts))
print(pdf);
print(bin_edges)

#compute CDF
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf)
plt.plot(bin_edges[1:], cdf)



plt.show();


# In[ ]:


#CDF of Year of Operation
#Dead

counts, bin_edges = np.histogram(dead['year_of_op'], bins=10, 
                                 density = True)
pdf = counts/(sum(counts))
print(pdf);
print(bin_edges)

#compute CDF
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf)
plt.plot(bin_edges[1:], cdf)



plt.show();


# In[ ]:


sns.boxplot(x='status',y='age', data=habers)
plt.show()
sns.boxplot(x='status',y='year_of_op', data=habers)
plt.show()
sns.boxplot(x='status',y='detected_pos_nodes', data=habers)
plt.show()


# ###### Observations:
# * Those who were operated after 1965 have higher chances of survival.
# * Those having Auxilary Node > 3 have greater chances of dying.
# * Those in the age group 30-35 are more likely to survive.
# * Those in age group 76-83 are most likely to die.

# ### Conclusion:
# * ##### Survival
#     * Auxilary Nodes < 3 can survive.(Aux Node < 10=~90% Survival Rate)
#     * Auxilar Nodes < 3 and operated after 1965 have greater chances of survival.
#     * Patients having Auxilar nodes < 3, treated after 1965 and in age group 30-35 will difinately survive.
# * ##### Dead
#     * Patients having Auxilary Nodes > 3 are likely to die(Aux Nodes > 10=~99% Death Rate).
#     * Auxilary Nodes > 3 and treated before 1965 are more likely to die.
#     * Patients having Auxilar node > 3, treated before 1965 and in age group 76-83 will difinately die.
#     

# In[ ]:




