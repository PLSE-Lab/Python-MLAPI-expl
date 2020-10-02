#!/usr/bin/env python
# coding: utf-8

# This is Haberman Dataset.
# The dataset contains cases from a study that was conducted between 1958 and 1970 at the University of Chicago's Billings Hospital on the survival of patients who had undergone surgery for breast cancer. The dataset has 4 features.
# 1. Age of Patiens
# 2. Year of Operation
# 3. Nunber of Nodes
# 4. Survival Status
# 
# The last feature here is the survival status.
# This is the class label.
# Here '1' means that the patient has survived the operation and '2' means the patient has not survived the operation .

# From the given data we can make sone Assumptions about the status 

# 1. Persons with higher number of nodes may have been diseased
# 2. Elderly persons may have been diseased due to the age factor
# 3. Persons who had their surgeries in earlier years may have died due to lack of medical facilities
# 4. Persons who had their surgeries lately may have survived due to advancements in medical facilities

# Let us draw our plots and find out

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")


# In[ ]:


columns = ['Age' , 'Year' ,'Nodes' , 'Status' ]
data = pd.read_csv("../input/haberman.csv", names = columns)


# Let us get an overview of the data first and then we will analize the data feature by feature

# In[ ]:


data.describe()


# In[ ]:


data.hist()
plt.plot()


# Now from this histogram we can conclude that 
# 1. A lot of patients are in the age of 40 to 60
# 2. Most of the patients have the cancer nodes in the range of 0 t0 10
# 3. Most of the patients have survived the operation 
# 4. Arround 60 patients were operated in the Year 1959 while 20-40 patients were operated in the later Years

# Now let us start with Pair plots for bivariate Analysis

# In[ ]:


sns.set_style("whitegrid");
sns.pairplot(data, hue = "Status", height = 5)
plt.show()


# Observation: Now from this scatter plot we are not getting any insight of the data so let us proceed with the PDFs and CDFs, Histograms and Violin Plots

# In[ ]:


sns.FacetGrid(data, hue='Status', height=5)     .map(sns.distplot, 'Age')     .add_legend()
plt.show()


# Observation: Older patients have less probablity of survival 

# In[ ]:


sns.FacetGrid(data, hue='Status', height=5)     .map(sns.distplot, 'Year')     .add_legend()
plt.show()


# Observation: Year of Operation cannot be used as a feature for classification

# In[ ]:


sns.FacetGrid(data, hue='Status', height=5)     .map(sns.distplot, 'Nodes')     .add_legend()
plt.show()


# Observation: People with number of Nodes < 5 have very high probablity for survival

# In[ ]:


#Segregating data on the basis of Age and Status  
data_Status_Survived = data[data.Status == 1]
data_Status_Diseased = data[data.Status == 2]


# In[ ]:


sns.boxplot(x = 'Status', y = 'Age', data = data)
plt.show()


# In[ ]:


sns.violinplot(x = 'Status', y = 'Age', data = data)
plt.show()


# In[ ]:


sns.boxplot(x = 'Status', y = 'Year', data = data)
plt.show()


# In[ ]:


sns.violinplot(x = 'Status', y = 'Year', data = data)
plt.show()


# In[ ]:


sns.boxplot(x = 'Status', y = 'Nodes', data = data)
plt.show()


# In[ ]:


sns.violinplot(x = 'Status', y = 'Nodes', data = data)
plt.show()


# In[ ]:


counts, bin_edges = np.histogram(data_Status_Survived['Age'], bins=10, density = True)

pdf = counts/(sum(counts))
print("PDF for survived patients are ",pdf);
print("Bin Edges for survived are", bin_edges)
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf)
plt.plot(bin_edges[1:],cdf)

counts, bin_edges = np.histogram(data_Status_Diseased['Age'], bins=10, density = True)

pdf = counts/(sum(counts))
print("PDF for diseased patients are ",pdf);
print("Bin Edges diseased patients are", bin_edges)
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf)
plt.plot(bin_edges[1:],cdf)

plt.legend(['Survived_PDF', 'Survived_CDF','Diseased_PDF', 'Diseased_CDF'])
plt.show()


# Observation: Patients with age < 45 have very high survival probablity

# In[ ]:


counts, bin_edges = np.histogram(data_Status_Survived['Year'], bins=10, density = True)

pdf = counts/(sum(counts))
print("PDF for survived patients are ",pdf);
print("Bin Edges for survived are", bin_edges)
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf)
plt.plot(bin_edges[1:],cdf)

counts, bin_edges = np.histogram(data_Status_Diseased['Year'], bins=10, density = True)

pdf = counts/(sum(counts))
print("PDF for diseased patients are ",pdf);
print("Bin Edges diseased patients are", bin_edges)
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf)
plt.plot(bin_edges[1:],cdf)

plt.legend(['Survived_PDF', 'Survived_CDF','Diseased_PDF', 'Diseased_CDF'])
plt.show()


# Observation: Year cannot be used as a feature for classification

# In[ ]:


counts, bin_edges = np.histogram(data_Status_Survived['Nodes'], bins=10, density = True)

pdf = counts/(sum(counts))
print("PDF for survived patients are ",pdf);
print("Bin Edges for survived are", bin_edges)
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf)
plt.plot(bin_edges[1:],cdf)

counts, bin_edges = np.histogram(data_Status_Diseased['Nodes'], bins=10, density = True)

pdf = counts/(sum(counts))
print("PDF for diseased patients are ",pdf);
print("Bin Edges diseased patients are", bin_edges)
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf)
plt.plot(bin_edges[1:],cdf)

plt.legend(['Survived_PDF', 'Survived_CDF','Diseased_PDF', 'Diseased_CDF'])
plt.show()


# Observation: People with number of nodes < 5 have very high surval chance  

# Conclusions: 
# 1. Patients with less number of Nodes have a very High chance of survival
# 2. Young patients have a very high chance of survival
# 3. Year of Operation has nothing to do with the rate of survival
