#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# Importing Data from CSV using Pandas Read CSV Function and print the first five elements of it.

# In[3]:


haberman=pd.read_csv("../input/haberman.csv")
haberman.head()


# Observation:
#             1. Data contains values columns name are missing in the dataset
#             2. Need to add column data in dataset
#             3.Column name are found in column metadata tab in Kaggle

# In[4]:


#Adding column name to data set
columns=['Age','Operation_Year','AuxillaryNode','Survival']
haberman_data=pd.read_csv('../input/haberman.csv',names=columns)
haberman_data.head()


# Observation:
#             1. First column shows about age of the patient
#             2. Second column shows about operation year i.e people operated in 19XX
#             3. Third column shows about number of auxillary node i.e number of tumors found
#             4. Fourth column shows about Survival Status of person after operation
#                 1 - Shows about the person survived 5 years or longer
#                 2-  Shows about the person died in less than 5 years

# In[10]:


haberman_data.shape


# In[11]:


haberman_data.columns


# In[12]:


haberman_data.info()


# Observation:
#             1. Shape of data is 306 rows and 4 columns
#             2. Columns are age, operation_year,auxillaryNode,survival
#             3. From information of data all non-null elements and int values

# In[13]:


haberman_data.Survival.value_counts()


# Observation:
#             1. Following dataset is an unbalanced dataset 
#             2. Out of 306 operation performed
#             3. 225 people lived more than 5 years
#             4. 81 people died less than 5 years.
2-D Plots
# In[6]:


haberman_data.plot(kind='scatter',x='Age',y='AuxillaryNode')
plt.show()


# Observations:
#             1.Most of the people have zero auxillary nodes
Pair Plots
# In[7]:


sns.pairplot(haberman_data,hue="Survival")
plt.show()


# Relationship between Age,Survival,Operation_Year,AuxillaryNode

# In[8]:


sns.set_style("whitegrid")
sns.FacetGrid(haberman_data,hue='Survival',size=5).map(plt.scatter,"Age","AuxillaryNode").add_legend()
plt.show()


# Observation:
#             Age vs AuxillaryNode plot let us know most people survived with 0 auxillary node
#             Cannot distingush data easily most of them are overlapping

# Histogram, PDF

# In[11]:


sns.FacetGrid(haberman_data,hue='Survival',size=5).map(sns.distplot,"AuxillaryNode").add_legend()
plt.show()


# In[14]:


sns.FacetGrid(haberman_data,hue='Survival',size=5).map(sns.distplot,"Age").add_legend()
plt.show()


# In[15]:


sns.FacetGrid(haberman_data,hue='Survival',size=5).map(sns.distplot,"Operation_Year").add_legend()
plt.show()


# In[26]:


#In order find Age of people who survived we need to find mean age of people
print("Mean age of patients survived:", round(np.mean(haberman_data[haberman_data['Survival'] == 1]['Age'])))
print("Mean age of patients not survived:", round(np.mean(haberman_data[haberman_data['Survival'] == 2]['Age'])))


# Observation:
#             1. Auxillary node are used to identify the people who have survived
#             2. Average age of people who survived is 52.0
#             3. Average age of people who are not survived is 54.0
#             4.More number of people are not survived in year of operation of 1965

# CDF

# In[46]:


survived=haberman_data.loc[haberman_data["Survival"]==1]
notsurvived=haberman_data.loc[haberman_data["Survival"]==2]


# In[48]:


counts, bin_edges = np.histogram(survived['AuxillaryNode'], bins=20, 
                                 density = True)
pdf = counts/(sum(counts))
print(pdf);
print(bin_edges)
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf)
plt.plot(bin_edges[1:], cdf)
plt.legend(['Pdf  Survived',
            'Cdf  Survived'])
plt.show()


# In[49]:


counts, bin_edges = np.histogram(notsurvived['AuxillaryNode'], bins=20, 
                                 density = True)
pdf = counts/(sum(counts))
print(pdf);
print(bin_edges)
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf)
plt.plot(bin_edges[1:], cdf)
plt.legend(['Pdf Died',
            'Cdf Died'])
plt.show()

Summary 
# In[31]:


haberman_data[haberman_data['Survival']==1].describe()


# In[32]:


haberman_data[haberman_data['Survival']==2].describe()


# Observation:
#             1. People have less number of auxillary node have survived

# Box Plot and Whiskers

# In[33]:


sns.boxplot(x='Survival',y='AuxillaryNode', data=haberman_data)
plt.show()


# In[35]:


sns.boxplot(x='Survival',y='Age', data=haberman_data)
plt.show()


# In[37]:


sns.boxplot(x='Survival',y='Operation_Year', data=haberman_data)
plt.show()


# Violin Plot

# In[38]:


sns.violinplot(x='Survival',y='AuxillaryNode', data=haberman_data)
plt.show()


# In[42]:


sns.violinplot(x='Survival',y='Age', data=haberman_data)
plt.show()


# In[41]:


sns.violinplot(x='Survival',y='Operation_Year', data=haberman_data)
plt.show()


# Observation:
#             From box,violin plot number of people dead between age of 46-62 and 59-65
#             Number of people survived between age of 42-60 and 60-66
Contour Plot
# In[45]:


sns.jointplot(x="Age", y="Operation_Year", data=haberman_data, kind="kde");
plt.show();


# Observation:
#             There are more number of people undergone operation during the year 
#             1959 - 1964 period and between ages 42 - 60
