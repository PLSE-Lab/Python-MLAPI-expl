#!/usr/bin/env python
# coding: utf-8

# In[ ]:




import numpy as np 
import pandas as pd 
import os
print(os.listdir("../input"))


# In[ ]:


haber=pd.read_csv('../input/haberman.csv', header=None, names=['age', 'year', 'nodes', 'status'])
haber.head()


# In[ ]:


#Features of the data
print(haber.columns)
# No. of features
print(len(haber.columns)) 


# In[ ]:


# The dimensions of the dataset
haber.shape


# In[ ]:


# Total no. of points
haber.size


# In[ ]:


# No. of unique values in the datasets for the status feature, which basically indicates the no. of classes.
haber['status'].unique().tolist()


# In[ ]:


# No of data points per class

haber.loc[haber["status"] == 1, 'status'] = "Survived_5"
haber.loc[haber["status"] == 2, 'status'] = "Died_5"
haber["status"].value_counts()


# About the features:
# 
# age : It represents the age of the patient(person)
# 
# year : It basically represents the year of operation  from 1900
# 
# nodes: Number of positive axillary nodes detected.
# 
# status: Indicates whether patient(person) survived or not
# 
# Survived_5 = the patient survived 5 years or longer 
# 
# Died_5 = the patient died within 5 year
# 
# # Objective
# 
#  To classify whether the patients who has undergone surgery(operation) survived or died based on the number
#  of nodes found in their body as well as their age

# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns

sns.FacetGrid(haber, hue="status", size=6)    .map(sns.distplot, "nodes")    .add_legend();
plt.show();


# ## Observations:
# 
# 1) It is quite evident that when nodes=0, more number of people survived as compared to number of deaths.
# 
# 2) P(0) is nearly 0.3 for survival when the number of nodes is 0. Therefore more than 3 times are the chances of survival when the nodes=0.
# 
# 3) When the number of nodes >= 3(or 4 approx.) the person is more likely to die. So we take 3(or 4 approx.) as a thershold we can classify between the two based on their probability value. When less than 3 the person is more likely to survive, so purely based on probability of fatality based on nodes the datapoints can be classified.

# In[ ]:


sns.FacetGrid(haber, hue="status", size=6)    .map(sns.distplot, "age")    .add_legend();
plt.show();


# In[ ]:


sns.boxplot(x='status',y='nodes', data=haber)
plt.show()


# In[ ]:


haber.loc[haber["status"]=="Survived_5"]["nodes"].describe()


# In[ ]:


haber.loc[haber["status"]=="Died_5"]["nodes"].describe()


# ## Observations:
# 1) Clearly when the number of nodes is >=3(or 4 approx) we can classify more than 50% of points accurately when it comes to those who have died. Ofcourse there will be an error of more than 25%.
# 
# 2) However the number of outliers in the case of those who have survived is quite high(dots indicating so).
# 
# 3) Almost 75% of total who have survived is have number of nodes less than or equal to 3 with a mean of 2.79. So the above stated threshold can be used.

# In[ ]:


sns.violinplot(x='status',y='nodes', data=haber)
plt.show()


# In[ ]:


# Calculate the CDF
import numpy as np
count,bins = np.histogram(haber.loc[haber["status"]=="Survived_5"]["nodes"], bins=10, density = True)
pdf = count/(sum(count))
cdf = np.cumsum(pdf)
plt.plot(bins[1:],pdf)
plt.plot(bins[1:], cdf)

count,bins = np.histogram(haber.loc[haber["status"]=="Died_5"]["nodes"], bins=10, density = True)
pdf = count/(sum(count))
cdf = np.cumsum(pdf)
plt.plot(bins[1:],pdf)
plt.plot(bins[1:], cdf)

plt.xlim(-10,40)
plt.show()


# ## Observations
# 1) Since the CDF of 'Survived' is always above than that of 'Died', max of cdfs of the two can be used as a classifying factor for a given number of nodes.

# In[ ]:


sns.set_style("whitegrid");
sns.pairplot(haber, hue="status", size=3);
plt.show()


# ## Observations
# 1) No combinations of features gives a clear distinction between the two classes as the number of outliers are quite large in case of those who have survived. 

# In[ ]:




