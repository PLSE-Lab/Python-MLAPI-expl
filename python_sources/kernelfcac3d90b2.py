#!/usr/bin/env python
# coding: utf-8

#  ### EDA on Haberman's cancer data set
#  <pre>
# source : [https://www.kaggle.com/gilsousa/habermans-survival-data-set/data](http://)
# This data set contains 306 instances of operations performed inorder to treat cancer patients
# Objective:Our motive is trying to classify any new instance to predict the survival after 5 years
# features of the data set : Age of the patient,year in which the operation was undertaken,count of positive axillary nodes detected.
# here the positive axillary nodes refers the no.of nodes that have been identified as nodes that are damaged due to cancer.
# 
# Dependant variable: survival- categorical variable with 2-survived for less than 5 years.else 1.
#                     
# </pre>

# In[2]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.style as style


# In[3]:


import os
print(os.curdir)
#print(os.__doc__)
print(os.listdir())


# In[4]:


#pd.read_csv.__code__.co_varnames
df=pd.read_csv("../input/haberman.csv",header=None)
df.columns=['age','year','nodes','status']
print(df.info())
print(df.describe())
print(df)
temp=df


# In[5]:


print(df['status'].value_counts())
df[df['status']==1].count()/306


# In[6]:


df['nodes'][df["nodes"]<5].value_counts()


# #### Observation:
# <pre>
# -> clearly none of the observations have a null value in it's list of features.
# -> operations from 1958 to 1969 have been noted.
# -> observations have been noted the occurance of cancer in patients of age 30 to 83 year.
# -> the no.of instances where patients survived for less than 5 years is 27% while 73 % of the       instances patients survived for more than 5 years.Thus this is unbalanced data set.
# -> there are more than 25% of instances where the patients with 0 +ve auxilary nodes are operated and 75%(230/306) with &lt;5 +ve nodes ,this    might be a sign of early diagnosis.
# -> the mean of age is 52 and std is 10 which signifies the higher chances of occurances between 40 and    60 years. 
# </pre>
# #### Visual representation of the data
# i)Scatter plots

# In[7]:


#sns.pairplot.__code__.co_varnames
sns.set()
style.use("ggplot")
sns.set_style("whitegrid");
sns.pairplot(data=df,hue='status',size=2.5)
plt.show()


# In[8]:


sns.FacetGrid(df,hue='status').map(sns.distplot,"age").add_legend()
plt.show()


# In[9]:


sns.FacetGrid(df,hue='status').map(sns.distplot,"year").add_legend()
plt.show()


# In[10]:


sns.FacetGrid(df,hue='status').map(sns.distplot,"nodes").add_legend()
plt.show()


# #### observation
# As the count of nodes increase we can notice the blue line to be positioned above red line thus the chances where patient survives for more than 5 years decreases with increase in no.of +ve auxilary nodes. 
# 
# Distribution plots are not sufficient to make strong points due to lack of separation.
# Consider following cummulative distribution plots to infer some statistical statements describing the visual observations done earlier.

# In[11]:


counts,bin_edges =np.histogram(df['age'],bins=10) #,density=True)
print("age counts {}\n".format(counts))
print("age bin_edges {}\n".format(bin_edges))
cumcounts=np.cumsum(counts)
print("age cumcounts {}\n".format(cumcounts))
densitycounts=counts/counts.sum()
print("age pdfcounts {}\n".format(densitycounts))
cumdensitycounts=np.cumsum(densitycounts)
print(cumdensitycounts)

countsyear,bin_years = np.histogram(df['year'],bins=10)
countsyear=countsyear/306
cumcountsyear=np.cumsum(countsyear)

countsnodes,bin_nodes = np.histogram(df['nodes'],bins=10)
countsnodes=countsnodes/306
cumcountsnodes=np.cumsum(countsnodes)


# In[12]:


plt.subplot(1, 3, 1)
plt.plot(bin_edges[1:],densitycounts)
plt.plot(bin_edges[1:],cumdensitycounts)
plt.xlabel('Age')
plt.ylabel('cdf')

plt.subplot(1,3,2)
plt.plot(bin_years[1:],countsyear)
plt.plot(bin_years[1:],cumcountsyear)
plt.xlabel('Year')
plt.ylabel('Cdf')

plt.subplot(1,3,3)
plt.plot(bin_nodes[1:],countsnodes)
plt.plot(bin_nodes[1:],cumcountsnodes)
plt.xlabel('Nodes')
plt.ylabel('Cdf')
plt.title("\nTrends in entire data\n")
plt.show()


# # Observation:
# -> ~70% of people have age between 41 and 65
# 
# 

# In[13]:


df=temp[temp['status']==1]
counts,bin_edges =np.histogram(df['age'],bins=10) #,density=True)
print("age counts {}\n".format(counts))
print("age bin_edges {}\n".format(bin_edges))

densitycounts=counts/counts.sum()
print("age pdfcounts {}\n".format(densitycounts))
cumdensitycounts=np.cumsum(densitycounts)
print(cumdensitycounts)


countsyear,bin_years = np.histogram(df['year'],bins=10)
countsyear=countsyear/countsyear.sum()
cumcountsyear=np.cumsum(countsyear)

countsnodes,bin_nodes = np.histogram(df['nodes'],bins=10)
countsnodes=countsnodes/countsnodes.sum()
cumcountsnodes=np.cumsum(countsnodes)


# In[14]:




plt.subplot(1, 3, 1)
plt.plot(bin_edges[1:],densitycounts)
plt.plot(bin_edges[1:],cumdensitycounts)
plt.xlabel('Age')
plt.ylabel('cdf')

plt.subplot(1,3,2)
plt.plot(bin_years[1:],countsyear)
plt.plot(bin_years[1:],cumcountsyear)
plt.xlabel('Year')
plt.ylabel('Cdf')

plt.subplot(1,3,3)
plt.plot(bin_nodes[1:],countsnodes)
plt.plot(bin_nodes[1:],cumcountsnodes)
plt.xlabel('Nodes')

plt.ylabel('Cdf')
plt.title("\ntrends in patients who survived more than 5 years\n")
plt.show()


# In[15]:



df=temp[temp['status']==2]
counts,bin_edges =np.histogram(df['age'],bins=10) #,density=True)
print("age counts {}\n".format(counts))
print("age bin_edges {}\n".format(bin_edges))

densitycounts=counts/counts.sum()
print("age pdfcounts {}\n".format(densitycounts))
cumdensitycounts=np.cumsum(densitycounts)
print(cumdensitycounts)

countsyear,bin_years = np.histogram(df['year'],bins=10)
countsyear=countsyear/countsyear.sum()
cumcountsyear=np.cumsum(countsyear)

countsnodes,bin_nodes = np.histogram(df['nodes'],bins=10)
countsnodes=countsnodes/countsnodes.sum()
cumcountsnodes=np.cumsum(countsnodes)


# In[16]:




plt.subplot(1, 3, 1)
plt.plot(bin_edges[1:],densitycounts)
plt.plot(bin_edges[1:],cumdensitycounts)
plt.xlabel('Age')
plt.ylabel('cdf')

plt.subplot(1,3,2)
plt.plot(bin_years[1:],countsyear)
plt.plot(bin_years[1:],cumcountsyear)
plt.xlabel('Year')
plt.ylabel('Cdf')

plt.subplot(1,3,3)
plt.plot(bin_nodes[1:],countsnodes)
plt.plot(bin_nodes[1:],cumcountsnodes)
plt.xlabel('Nodes')

plt.ylabel('Cdf')
plt.title("\ntrends in patients who couldn't survived more than 5 years\n")
plt.show()


# In[17]:


print("chart -1")

new =temp
new['bins'] = pd.cut(new['year'], 5)
print(new['bins'].value_counts())
print("survivals for>5 years")
new1=temp[temp['status']==1]
new1['bins'] = pd.cut(new1['year'], 5)
print(new1['bins'].value_counts())
print("survivals for <5 years")
new2=temp[temp['status']==2]
new2['bins'] = pd.cut(new2['year'], 5)
print(new2['bins'].value_counts())



# <pre>So the survival rate(>5years) is as follows:
# chart -2
# Years         -Rate
#     57.9-60.2  66/91 =>72%
#     60.2-62.4  39/49 =>79%
#     62.4-64.6  45/61 =>73%
#     64.6-66.8  37/56 =>66%
#     66.8-69    38/49 =>77%
# 
# </pre>

# In[18]:


print("chart -3")
new =temp
new['bins'] = pd.cut(new['age'], 5)
new[['bins', 'status']].groupby(['bins'], as_index=False).mean().sort_values(by='bins', ascending=True)


# In[19]:


print("chart -4")
new =temp
new['bins'] = pd.cut(new['year'], 5)
print(new[['bins', 'nodes']].groupby(['bins'], as_index=False).median().sort_values(by='bins', ascending=True))
print(new[['bins', 'nodes']].groupby(['bins'], as_index=False).mean().sort_values(by='bins', ascending=True))


#  
# #### observations:
# ************************
# form given data ~75% of total observations have &lt;10 nodes =>230 observations
# ~70%(~55) of status 2 observations and ~80%(~175) of status 1 observations => have &lt;10 nodes
# so if no.of nodes are &lt;10 then chances for survival for more than 5 years would be ~76% (175/230)
# similarly if no.of nodes are &gt;10 the chances would be ~67% ([225-175]/[225-175+81-55])
# ************************
# From the above table,it's evident that the survival rate is relatively low for patients of age >72.4 (since mean 1.33 is closer to 2 than any other bin)
# Similarly patients with age &lt;40.6 have relatively more chances of survival.
# ************************
# No.of operations were significantly high between 1957 and 1960
# from chart -1&2
# inspite of the smaller no.of observations after 1960 ,the survival rate hasn't significantly changed.
# from chart 4
# This might be due to fact
# 
# 

# In[ ]:




