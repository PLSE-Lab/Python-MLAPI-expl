#!/usr/bin/env python
# coding: utf-8

# ## Haberman Dataset - Exploratory Data Analysis ##
# 
# **Relavant Information** : This Haberman dataset contains cases from a study that was conducted between 1958 and 1970 at the University of Chicago's Billings Hospital on the survival of patients who had undergone surgery for breast cancer.
# 
# **Number of Instances** : 306
# 
# **Number of Attributes** : 4 (including the class attribute)
# 
# **Attribute Information** :
# - Age of patient at time of operation (numerical)
# - Patient's year of operation (year - 1900, numerical)
# - Number of positive axillary nodes detected (numerical)
# - Survival status (class attribute) 1 = the patient survived 5 years or longer 2 = the patient died within 5 year
# 

# ### Objective
# 
# The objective is to understand the dataset with the help of univariate and bivariate data analysis and to check whether there is a presence of an attribute or a combination of attributes which would help determine the longetivity of a patient after her cancer treatment.

# In[ ]:


#importing the libraries

import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from statsmodels import robust


# In[ ]:


#inserting the dataset haberman.csv into a pandas dataframe

habermand=pd.read_csv("../input/haberman.csv")


# ## High Level Statistics of the Dataset

# In[ ]:


#Number of Points and Features in the Dataset

print(habermand.shape)


# In[ ]:


#The Column names in the Dataset

print(habermand.columns)


# In[ ]:


#We now try to look as to how the dataset is by looking at the head of the csv file
habermand.head()


# In[ ]:


#As the column names are not mentioned properly in the dataset given, we will try to rename them
#We try to store the column names in a list and integrate it into the our haberman dataframe

column_name = ["age","operation_year","positive_axilliary_nodes","survival_status"]

habermand=pd.read_csv("../input/haberman.csv", header=None, names=column_name)

habermand.head()


# In[ ]:


habermand["survival_status"].value_counts()
#so we get to know that this dataset is an imbalanced dataset because the number of datapoints for the class attribute is 
#not the same.


# In[ ]:


#checking the breif description of the dataset
habermand.info()


# In[ ]:


#Describing the dataset
habermand.describe()


# ## Univariate Analysis
# 
# ### Probablity Density Function

# In[ ]:


#plotting pdf for age
sns.FacetGrid(habermand, hue="survival_status", size=5)    .map(sns.distplot, "age")    .add_legend();
plt.show();


# In[ ]:


#plotting pdf for operation_year
sns.FacetGrid(habermand, hue="survival_status", size=5)    .map(sns.distplot, "operation_year")    .add_legend();
plt.show();


# In[ ]:


#plotting pdf for positive_axilliary_nodes
sns.FacetGrid(habermand, hue="survival_status", size=5)    .map(sns.distplot, "positive_axilliary_nodes")    .add_legend();
plt.show();


# ##### Observations
# 
# 1) From the above probablity density graphs we can see that we cannot linearly separate the 2 cases distinguishably.
# 
# 
# 2) No proper analysis can be stated from the above histograms.
# 
# 

# ### Cumulative Distribution Function
# 

# In[ ]:


#for cdf we divide the survived people and the people who have died
survived = habermand.loc[habermand["survival_status"] == 1];
died = habermand.loc[habermand["survival_status"] == 2];


# In[ ]:


#plotting cdf for positive axilliary nodes

#for people who survived for more than five year
counts, bin_edges = np.histogram(survived['positive_axilliary_nodes'], bins=10, 
                                 density = True)
pdf = counts/(sum(counts))
print(pdf)
print(bin_edges)
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf,label = "pdf more than 5")
plt.plot(bin_edges[1:], cdf, label = "cdf more than 5")

#for people who didnt survive for more than five year
counts, bin_edges = np.histogram(died['positive_axilliary_nodes'], bins=10, 
                                 density = True)
pdf = counts/(sum(counts))
print(pdf);
print(bin_edges)
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf, label = "pdf less than 5")
plt.plot(bin_edges[1:], cdf, label = "cdf less than 5")
plt.legend()
plt.xlabel("positive_axilliary_nodes")

plt.show();


# In[ ]:


#plotting cdf for  operation year

#for people who survived for more than five year
counts, bin_edges = np.histogram(survived['operation_year'], bins=10, 
                                 density = True)
pdf = counts/(sum(counts))
print(pdf)
print(bin_edges)
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf,label = "pdf more than 5")
plt.plot(bin_edges[1:], cdf, label = "cdf more than 5")

#for people who didnt survive for more than five year
counts, bin_edges = np.histogram(died['operation_year'], bins=10, 
                                 density = True)
pdf = counts/(sum(counts))
print(pdf);
print(bin_edges)
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf, label = "pdf less than 5")
plt.plot(bin_edges[1:], cdf, label = "cdf less than 5")
plt.legend()
plt.xlabel("operation_year")

plt.show();


# In[ ]:


#plotting cdf for age

#for people who survived for more than five year
counts, bin_edges = np.histogram(survived['age'], bins=10, 
                                 density = True)
pdf = counts/(sum(counts))
print(pdf)
print(bin_edges)
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf,label = "pdf more than 5")
plt.plot(bin_edges[1:], cdf, label = "cdf more than 5")

#for people who didnt survive for more than five year
counts, bin_edges = np.histogram(died['age'], bins=10, 
                                 density = True)
pdf = counts/(sum(counts))
print(pdf);
print(bin_edges)
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf, label = "pdf less than 5")
plt.plot(bin_edges[1:], cdf, label = "cdf less than 5")
plt.legend()
plt.xlabel("age")

plt.show();


# ##### Observations
# We can say that patients having axilliary nodes more than 47 have definetly not survived.
# 
# 
# 

# ### Box Plot

# In[ ]:


#boxplot for age v survival status
sns.boxplot(x='survival_status',y='age', data=habermand)
plt.show()


# In[ ]:


#boxplot for operation_year v survival status
sns.boxplot(x='survival_status',y='operation_year', data=habermand)
plt.show()


# In[ ]:


#boxplot for positive axilliary nodes v survival status
sns.boxplot(x='survival_status',y='positive_axilliary_nodes', data=habermand)
plt.show()


# ##### Observations
# 
# In the age v survival status box plot we could see that patient less than the age 35 as well survived the treatment and we can also say that the patient greater than the age 77 were not able to survive the treatment.

# ### Violin Plot
# 

# In[ ]:


#violin plot for survival status v year
sns.violinplot(x='survival_status',y='operation_year', data=habermand)
plt.show()


# In[ ]:


#violin plot for survival status v axiliary nodes
sns.violinplot(x='survival_status',y='positive_axilliary_nodes', data=habermand)
plt.show()


# In[ ]:


#violinplot for survival status and age
sns.violinplot(x='survival_status',y='age', data=habermand)
plt.show()


# ##### Observations
# 
# In the survival status v axilliary nodes violin plot, in the survivors we see a lot of denseness from the region of 0-6 nodes.
# 
# 

# ### Mean

# In[ ]:


#Finding the mean and standard deviation for age
print("Means:")
print(np.mean(survived["age"]))
print(np.mean(died["age"]))
print("\n")
print("Std-dev:");
print(np.std(survived["age"]))
print(np.std(died["age"]))


# ##### Observations
# 
# 1) So the mean age of patients who survived for longer than 5 years is approximately 52
# 
# 2) The mean age of patients who couldnt survive for longer than 5 years is approximately 53.5

# In[ ]:


#Finding the mean and standard deviation for positive axilliary nodes
print("Means:")
print(np.mean(survived["positive_axilliary_nodes"]))
print(np.mean(died["positive_axilliary_nodes"]))

print("\nStd-dev:");
print(np.std(survived["positive_axilliary_nodes"]))
print(np.std(died["positive_axilliary_nodes"]))


# ##### Observations
# 
# 1) The patients who survived for more than 5 years had an axilliary node count of approximately 3
# 
# 2) The patients who were not able to survive for more than 5 years had an axilliary node count of approximately 7

# In[ ]:


#Finding the mean and standard deviation for operating_year
print("Means:")
print(np.mean(survived["operation_year"]))
print(np.mean(died["operation_year"]))

print("\nStd-dev:");
print(np.std(survived["operation_year"]))
print(np.std(died["operation_year"]))


# ##### Observations
# 
# Unfortunately through the mean we are not able to deduce anything as the values are similar.

# ## Univariate Analysis
# 
# ### 2D Scatter Plots
# 

# In[ ]:


#scatter plot for age v node
sns.set_style("whitegrid");
sns.FacetGrid(habermand, hue="survival_status", size=6)    .map(plt.scatter, "age", "positive_axilliary_nodes")    .add_legend();
plt.show();


# In[ ]:


#scatter plot for operation year v node
sns.set_style("whitegrid");
sns.FacetGrid(habermand, hue="survival_status", size=6)    .map(plt.scatter, "operation_year", "positive_axilliary_nodes")    .add_legend();
plt.show();


# In[ ]:


#scatter plot for age v node
sns.set_style("whitegrid");
sns.FacetGrid(habermand, hue="survival_status", size=6)    .map(plt.scatter, "age", "operation_year")    .add_legend();
plt.show();


# ##### Observations
# 
# The three scatter plots are not linearly separable and cannot be distinguished in a correct manner.
# 

# ### Pair Plots

# In[ ]:


plt.close();
sns.set_style("whitegrid");
sns.pairplot(habermand, hue="survival_status", vars = ['age','operation_year','positive_axilliary_nodes'],size=5);
plt.show()


# ##### Observations
# 
# We are not able to dedue anything from the Pair Plot as they are too intermixed.
