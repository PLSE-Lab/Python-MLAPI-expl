#!/usr/bin/env python
# coding: utf-8

# In[184]:


import os
print(os.listdir('../input'))


# In[185]:


# Importing Packages

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[186]:


# Reading the data

habermans = pd.read_csv('../input/haberman.csv')


# In[187]:


# Renaming the column names for better understanding

habermans.columns = ['Age', 'Year', 'Nodes', 'SurvivalStatus']


# In[188]:


# Renaming the values in the survival status for pair plots

habermans["SurvivalStatus"] = habermans["SurvivalStatus"].map({1 : "Yes", 2 : "No"})


# In[189]:


# After renaming

habermans


# In[190]:


# No of datapoints and features

habermans.shape


# In[191]:


# Columns

habermans.columns


# In[192]:


# Datapoints per class

habermans['SurvivalStatus'].value_counts()


# # Observations:
# * Dataset is imbalanced as the datapoints for the survival status "Yes" and "No" differs.

# # Objective :
# * The Objective is to perform the Exploratory Data Analysis(EDA) on the Habermans dataset.

# In[193]:


# Performing Univariate Analysis

# PDF - Probability Density Funtion for Age

sns.FacetGrid(habermans, hue='SurvivalStatus',size=5)    .map(sns.distplot, "Age")    .add_legend()
plt.ylabel('PDF')
plt.title("PDF for Age")
plt.show()


# In[194]:


# PDF - Probability Density Funtion for Year

sns.FacetGrid(habermans, hue='SurvivalStatus',size=5)    .map(sns.distplot, "Year")    .add_legend()
plt.ylabel('PDF')
plt.title("PDF for Year")
plt.show()


# In[195]:


# PDF - Probability Density Funtion for Nodes

sns.FacetGrid(habermans, hue='SurvivalStatus',size=5)    .map(sns.distplot, "Nodes")    .add_legend()
plt.ylabel('PDF')
plt.title("PDF for Nodes")
plt.show()


# In[196]:


# CDF - Cumulative Density Function
# CDF for Age

fig, ax = plt.subplots()
pdf, bin_edges = np.histogram(habermans['Age'][habermans['SurvivalStatus'] == 'Yes'], bins = 10, density = True)
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:], pdf, label = 'Yes')
plt.plot(bin_edges[1:], cdf, label = 'Yes')

pdf, bin_edges = np.histogram(habermans['Age'][habermans['SurvivalStatus'] == 'No'], bins = 10, density = True)
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:], pdf, label = 'No')
plt.plot(bin_edges[1:], cdf, label = 'No')

plt.xlabel('Bin Edges')
plt.ylabel('PDF/CDF')
plt.title("CDF for Age")
plt.legend()
plt.show()


# In[197]:


# CDF for Year

fig,ax = plt.subplots()
pdf, bin_edges = np.histogram(habermans['Year'][habermans['SurvivalStatus'] == 'Yes'], bins = 10, density = True)
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:], pdf, label = 'Yes')
plt.plot(bin_edges[1:], cdf, label = 'Yes')

pdf, bin_edges = np.histogram(habermans['Year'][habermans['SurvivalStatus'] == 'No'], bins = 10, density = True)
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:], pdf, label = 'No')
plt.plot(bin_edges[1:], cdf, label = 'No')

plt.xlabel('Bin Edges')
plt.ylabel('PDF/CDF')
plt.title('CDF for Year')
plt.legend()
plt.show()


# In[198]:


# CDF for Nodes

fig, ax = plt.subplots()
pdf, bin_edges = np.histogram(habermans['Nodes'][habermans['SurvivalStatus'] == 'Yes'], bins = 10, density = True)
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:], pdf, label = "Yes")
plt.plot(bin_edges[1:], cdf, label = "Yes")

pdf, bin_edges = np.histogram(habermans['Nodes'][habermans['SurvivalStatus'] == 'No'], bins = 10, density = True)
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:], pdf, label = "No")
plt.plot(bin_edges[1:], cdf, label = "No")

plt.xlabel('Bin Edges')
plt.ylabel('PDF/CDF')
plt.title('CDF for Nodes')
plt.legend()
plt.show()


# In[199]:


# Box Plot
# BP for Age

sns.boxplot(x = habermans['SurvivalStatus'], y = habermans['Age'], data = habermans)
plt.title('Box Plot for Age')
plt.show()


# In[200]:


# BP for Year

sns.boxplot(x = habermans['SurvivalStatus'], y = habermans['Year'], data = habermans)
plt.title('Box Plot for Year')
plt.show()


# In[201]:


# BP for Nodes

sns.boxplot(x = habermans['SurvivalStatus'], y = habermans['Nodes'], data = habermans)
plt.title('Box Plot for Nodes')
plt.show()


# In[202]:


# Violin Plot for Age

sns.violinplot(x = habermans['SurvivalStatus'], y = habermans['Age'], data = habermans, size = 8)
plt.title('Violin Plot For Age')
plt.show()


# In[203]:


# Violin Plot for Year

sns.violinplot(x = habermans['SurvivalStatus'], y = habermans['Year'], data = habermans, size = 8)
plt.title('Violin Plot For Year')
plt.show()


# In[204]:


# Violin Plot for Nodes

sns.violinplot(x = habermans['SurvivalStatus'], y = habermans['Nodes'], data = habermans, size = 8)
plt.title('Violin Plot For Nodes')
plt.show()


# In[205]:


# Bivariate analysis - Pair plot

sns.set_style("whitegrid")
sns.pairplot(habermans, hue = "SurvivalStatus", size = 5)
plt.show()


# In[206]:


# Scatter Plot for Age and Year

sns.set_style('whitegrid')
sns.FacetGrid(habermans, hue = 'SurvivalStatus', size = 4)    .map(plt.scatter, "Age", "Year")    .add_legend()
plt.title('Scatter Plot for Age and Year')
plt.show()


# In[207]:


# Scatter Plot for Age and Nodes

sns.set_style('whitegrid')
sns.FacetGrid(habermans, hue = 'SurvivalStatus', size = 4)    .map(plt.scatter, "Age", "Nodes")    .add_legend()
plt.title('Scatter Plot for Age and Nodes')
plt.show()


# In[208]:


# Scatter Plot for Year and Nodes

sns.set_style('whitegrid')
sns.FacetGrid(habermans, hue = 'SurvivalStatus', size = 4)    .map(plt.scatter, "Year", "Nodes")    .add_legend()
plt.title('Scatter Plot for Year and Nodes')
plt.show()


#  # Observations :-
#  * From the univariate, bivarite analysis it is clear that there are overlapping occurs in all analysis.
#  * Through bi-variate analysis we can observe that no two specific combination of features are useful in classification.
#  * Since overlapping occurs in all the features age, year and the nodes it is clear that, one feature is dependent on all others and vice-versa. 
#  * So all the features are must needed for the classification of the survival status.
