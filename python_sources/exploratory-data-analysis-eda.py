#!/usr/bin/env python
# coding: utf-8

# # Exploratory Data Analysis (EDA) Excercise - Data Visualization

# ## Haberman's Survival Data Set

# https://www.kaggle.com/gilsousa/habermans-survival-data-set <br />
# Number of observations: 306 <br />
# Total Features: 3 <br />
# Class Lable: 1 <br />
# 
# Features Attribute Details
# * Age of patient at the time of operation (numerical)
# * Patient's year of operation (numerical)
# * Number of positive axillary nodes detected (numerical)
# * Class Lable 
#         1 - patient has survived 5 or more years
#         2 - patient died within 5 year
#         

# ## Objective

# It is a Classification Problem.
# <br /><br />
# By given the age, year of operation and positive axillary nodes detected in the patient predict whether the patient will survive for next 5 years or not

# In[ ]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
print(os.listdir("../input"))


# In[ ]:


# provide lable to each column in dataset
col = ['Patient_age', 'Year_of_operation', 'pos_axillary_nodes', 'status']
#load dataset from csv file
dataset = pd.read_csv("../input/haberman.csv", names = col)


# In[ ]:


print(dataset.shape)


# In[ ]:


dataset['status'].value_counts()


# This is kind of <b>imbalance</b> dataset <br />
# 225 observations (patients) has survived and 81 observations (patients) died.

# In[ ]:


#sample dataset -- first 5 observations
dataset.head()


# ## 2D Scatter Plot

# In[ ]:


# 2-D Scatter plot with color-coding for each type/class.

sns.set_style("whitegrid");
sns.FacetGrid(dataset, hue="status", size=6)    .map(plt.scatter, "Patient_age", "pos_axillary_nodes")    .add_legend();
plt.show();


# Observation: <br />
# 1) Most of the patients have axillary nodes between 0 and 5. <br />
# 2) From the graph Axillary nodes versus Age, we can say that most people who survived have 0 to 5 Auxillary nodes detected. <br />
# 3) All patient age is between 30 to 78.

# In[ ]:


sns.set_style("whitegrid");
sns.FacetGrid(dataset, hue="status", size=6)    .map(plt.scatter, "Year_of_operation", "pos_axillary_nodes")    .add_legend();
plt.show();


# Observation: <br />
# 1) It is not possible to classify the observation based on Operation_year and pos_node.<br />
# 2) All the obervations's operation is done between 1958-1970

# ## Pair Plot

# In[ ]:


sns.set_style("whitegrid");
sns.pairplot(dataset, hue="status",
             vars=col[0:3])
plt.show()


# Observation: <br />
# 1) <b>We need nonlinear classifier to classify this dataset. </b><br />
# 2) Most of the patients have axillary nodes between 0 and 5. <br />
# 3) From the graph Auxillary nodes versus Age, we can say that most people who survived have 0 to 5 Auxillary nodes detected.

# ## Histogram, PDF, CDF

# In[ ]:


sns.FacetGrid(dataset, hue="status", size=5)    .map(sns.distplot, "pos_axillary_nodes")    .add_legend();
plt.show();


# Observation: <br />
# Most of the patients have axillary nodes between 0 and 5. <br />
# From the graph Auxillary nodes versus Age, we can say that most people who survived have 0 to 5 Axillary nodes detected.

# In[ ]:


sns.FacetGrid(dataset, hue="status", size=5)    .map(sns.distplot, "Patient_age")    .add_legend();
plt.show();


# Observation:<br/>
# As above graph is most likely normal distribution, the possibility of corrupted data is less and we can use mean, std-dev and variance as statistical measurement. <br /><br />
# Important Feature:<br/>
# Pos_Auxilary_Node > Patient_age = Operation_year

# In[ ]:


# alive means status=1 and dead means status =2
alive = dataset.loc[dataset['status'] == 1]
dead = dataset.loc[dataset['status'] == 2]


# In[ ]:


counts, bin_edges = np.histogram(alive['pos_axillary_nodes'], bins=20, 
                                 density = True,)
pdf = counts/(sum(counts))
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:], cdf)
plt.xlabel('pos_axillary_nodes')
plt.legend(['Cdf for the patients who survive more than 5 years'])
plt.show()


# 72% of the people who have survived has pos_axillary_node less than 3. <br />
# <b> The Patient who had pos_axillary_node greater than 46, didn't survive. </b><br />

# In[ ]:


counts, bin_edges = np.histogram(dead['pos_axillary_nodes'], bins=5, 
                                 density = True,)
pdf = counts/(sum(counts))
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:], cdf,)
plt.legend(['Cdf for the patients who have not survived more than 5 years'])
plt.xlabel('pos_axillary_nodes')
plt.show()


# 72% of the people who have not survived has pos_axillary_node less than 10. <br />

# ## Mean, Variance, Std-dev

# In[ ]:


print("Summary Statistics of Patients")
dataset.describe()


# In[ ]:


print("Summary Statistics of Patients, who have survived")
alive.describe()


# In[ ]:


print("Summary Statistics of Patients, who have not survived")
dead.describe()


# Observation: <br />
# 1) 72% of the people who have survived have pos_axillary_node less than 3. <br />
# 2)<b> The Patient who had pos_axillary_node greater than 46, didn't survive. </b><br />
# 3) All the values of the features patient_age and year_of_operation are almost same so pos_axillary_node is the more informative feature.<br/>
# 4) As mean age of survival and non-survived observation is almost same we can infer that survival status is not much dependent on the age of a patient.

# # Box Plot and Whiskers

# In[ ]:


sns.boxplot(x='status',y='pos_axillary_nodes', data=dataset)
plt.show()


# ## Violin plots

# In[ ]:


sns.violinplot(x='status',y='pos_axillary_nodes', data=dataset)
plt.show()


# In[ ]:


sns.violinplot(x='status',y='Patient_age', data=dataset)
plt.show()


# In[ ]:





# In[ ]:




