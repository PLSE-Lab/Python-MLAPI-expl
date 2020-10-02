#!/usr/bin/env python
# coding: utf-8

# # Exploratory data analysis (EDA)

# ## Haberman Cancer Patient Dataset

# In[ ]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")
#Load haberman.csv into a pandas dataFrame.
df = pd.read_csv("../input/haberman.csv")


# In[ ]:


# (Q) how many data-points and features?
print (df.shape)


# In[ ]:


#(Q) What are the column names in our dataset?
print (df.columns)


# In[ ]:


# naming the columns
df.columns = ['age','operation_year','axil_nodes','surv_status']
print(df.columns)
print(df.shape)


# In[ ]:


# How many data points for each class are present? 

df["surv_status"].value_counts()


# **Observation :**
# 
# 1. The dataset is having 304 rows and 4 columns.
# 2. The names of the columns are : 'age', 'operation_year', 'axil_nodes', 'surv_status'.
# 3. There are 2 ouptut classes viz., "1 = the patient survived 5 years or longer" and "2 = the patient died within 5 year"
# 4. The number of observations in each class 1 is 224 and in class 2 is 81.
# 5. Haberman dataset is a highly imbalanced dataset as the number of data points for every class is not equal.

# # Objective

# 1. To understand which feature or combination of features are useful for determining the life span of patient after the operation.

# ## Uni-variate analysis using Histograms

# In[ ]:


# Distribution of axil_nodes
sns.FacetGrid(df, hue="surv_status", size=5)    .map(sns.distplot, "axil_nodes")    .add_legend();
plt.show();


# In[ ]:


print(df['axil_nodes'].mean())


# **Observation : **
# 1. The distribution is right skewed and mass of distribution is centered on left of the figure.
# 2. The mean is centered at 4.036 and the patients who have axil nodes less than mean lived more the 5 years after operation.

# In[ ]:


# Distribution of axil_nodes
sns.FacetGrid(df, hue="surv_status", size=5)    .map(sns.distplot, "age")    .add_legend();
plt.show();


# **Obervations : **
# 1. The distributions are highly mixed up so nothing can be obtained.
# 2. But from the graph we can say that the patients who have age less than 40 survived for more then 5 years after operation.

# In[ ]:


# Distribution of axil_nodes
sns.set_style('whitegrid')
sns.FacetGrid(df, hue="surv_status", size=5) .map(sns.distplot, "operation_year").add_legend();
plt.show();


# **Obervations : **
# 1. The distributions are highly mixed up so nothing can be obtained.
# 2. But from the graph we can say that the patients who got operated between the years 58 to 62.5 survived for more then 5 years after operation.

# In[ ]:


# Dividing data on the basis of classes

survived = df.loc[df["surv_status"] == 1];
died = df.loc[df["surv_status"] == 2];


# In[ ]:


# Plots of CDF of axil_nodes for two classes.

# Survived 5 years or longer 
counts, bin_edges = np.histogram(survived['axil_nodes'], bins=20, 
                                 density = True)
pdf = counts/(sum(counts))
print(pdf);
print(bin_edges)
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf,label = "PDF of axil_nodes which Survived 5 years or longer ")
plt.plot(bin_edges[1:], cdf, label = "CDF of axil_nodes which Survived 5 years or longer ")


# Died within 5 years of operation
counts, bin_edges = np.histogram(died['axil_nodes'], bins=20, 
                                 density = True)
pdf = counts/(sum(counts))
print(pdf);
print(bin_edges)
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf, label = "PDF of axil_nodes which Died within 5 years of operation")
plt.plot(bin_edges[1:], cdf, label = "PDF of axil_nodes which Died within 5 years of operation")
plt.legend(loc = 'best')
plt.xlabel("axil_nodes")

plt.show();


# **Observation :**
# 1. Those who have axil_nodes less than 22 are more probable to be survived 5 years or longer .

# In[ ]:


# Plots of CDF of petal_length for various types of flowers.

# Misclassification error if you use 
counts, bin_edges = np.histogram(survived['age'], bins=10, 
                                 density = True)
pdf = counts/(sum(counts))
print(pdf);
print(bin_edges)
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf,label = "PDF of age which Survived 5 years or longer ")
plt.plot(bin_edges[1:], cdf,label = "CDF of age which Survived 5 years or longer ")


counts, bin_edges = np.histogram(died['age'], bins=10, 
                                 density = True)
pdf = counts/(sum(counts))
print(pdf);
print(bin_edges)
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf,label = "PDF of age which Died within 5 years of operation")
plt.plot(bin_edges[1:], cdf,label = "PDF of age which Died within 5 years of operation")
plt.legend(loc='best')
plt.xlabel("age")


plt.show();


# **Observation :**
# 1. Age less than 38 are definitely survived for more than 5 years after operation.

# In[ ]:


# Plots of CDF of operation_year for various types.

counts, bin_edges = np.histogram(survived['operation_year'], bins=10, 
                                 density = True)
pdf = counts/(sum(counts))
print(pdf);
print(bin_edges)
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf, label = "PDF of operation_year which Survived 5 years or longer ")
plt.plot(bin_edges[1:], cdf, label = "CDF of operation_year which Survived 5 years or longer ")



counts, bin_edges = np.histogram(died['operation_year'], bins=10, 
                                 density = True)
pdf = counts/(sum(counts))
print(pdf);
print(bin_edges)
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf, label = "PDF of operation_year which Died within 5 years of operation")
plt.plot(bin_edges[1:], cdf, label = "CDF of operation_year which Died within 5 years of operation")
plt.xlabel("operation_year")
plt.legend(loc = 'best')

plt.show();


# **Observation :**
# 1. Those who performed operation between year 61 - 65 are more probable to be survived 5 years or longer .

# # Mean

# In[ ]:


#Mean, Std-deviation of age
print("Means:")
print(np.mean(survived["age"]))
print(np.mean(died["age"]))

print("\nStd-dev:");
print(np.std(survived["age"]))
print(np.std(died["age"]))


# **Observation :**
# 1. The mean age of patients who Survived 5 years or longer is approximately 52 years.
# 2. The mean age of patients who Died within 5 years of operation is approximately 54 years.

# In[ ]:


#Mean, Std-deviation of axil_nodes
print("Means:")
print(np.mean(survived["axil_nodes"]))
print(np.mean(died["axil_nodes"]))

print("\nStd-dev:");
print(np.std(survived["axil_nodes"]))
print(np.std(died["axil_nodes"]))


# **Observation :**
# 1. The mean of axil_nodes of patients who Survived 5 years or longer is approximately 3.
# 2. The mean of axil_nodes of patients who Died within 5 years of operation is approximately 7.

# In[ ]:


#Mean, Std-deviation of operation_year
print("Means:")
print(np.mean(survived["operation_year"]))
print(np.mean(died["operation_year"]))

print("\nStd-dev:");
print(np.std(survived["operation_year"]))
print(np.std(died["operation_year"]))


# **Observation :**
# 1. The operation_year doesnot reveal any information as the mean year of both classes is same.

# # Box plot and Whiskers

# In[ ]:


# box plot for axil_nodes
sns.boxplot(x='surv_status',y='axil_nodes', data=df)
plt.show()


# In[ ]:


# box plot for age
sns.boxplot(x='surv_status',y='age', data=df)
plt.show()


# **Observation :**
# 1. The patients having age less than 34 years definitely Survived 5 years or longer.
# 2. The patients having age greater than 78 definitely Died within 5 years of operation.

# In[ ]:


# box plot for operation_year
sns.boxplot(x='surv_status',y='operation_year', data=df)
plt.show()


# # Violin plots

# In[ ]:


# Violin plot for Axil_nodes
sns.violinplot(x='surv_status',y='axil_nodes', data=df, size=8)
plt.show()


# In[ ]:


# Violin plot for age
sns.violinplot(x='surv_status',y='age', data=df, size=8)
plt.show()


# In[ ]:


# Violin plot for Axil_nodes
sns.violinplot(x='surv_status',y='operation_year', data=df, size=8)
plt.show()


# **Observation :**
#     
#     1. It is not useful.

# # 2-D Scatter Plot

# In[ ]:


# 2-D Scatter plot with color-coding for each class i.e.  
#    1 = the patient survived 5 years or longer 
#    2 = the patient died within 5 years

sns.set_style("whitegrid");
sns.FacetGrid(df, hue="surv_status", size=4)    .map(plt.scatter, "age", "axil_nodes")    .add_legend();
plt.show();


# In[ ]:


sns.set_style("whitegrid");
sns.FacetGrid(df, hue="surv_status", size=4)    .map(plt.scatter, "age", "operation_year")    .add_legend();
plt.show();


# In[ ]:


sns.set_style("whitegrid");
sns.FacetGrid(df, hue="surv_status", size=4)    .map(plt.scatter, "axil_nodes", "operation_year")    .add_legend();
plt.show();


# **Observation:**
# 1. Nothing can be told from 2-D scatter plot as the points are highly mixed up.

# ## 3D Scatter plot

# In[ ]:


from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

    
xs = df["age"]
ys = df["operation_year"]
zs = df["axil_nodes"]    

ax.scatter(xs, ys, zs)

ax.set_xlabel('age')
ax.set_ylabel('operation_year')
ax.set_zlabel('axil_nodes')

plt.show()


# **Observations :**
# 
# 1. Nothing can be obtained from 3-D scatter plot.
#     

# # Pair-plot

# In[ ]:


plt.close();
sns.set_style("whitegrid");
sns.pairplot(df, hue="surv_status", vars = ['age','operation_year','axil_nodes'],size=5);
plt.show()


# **Observations**
# 1. Nothing can be obtained from Pair-Plots as points are mixed up.

# In[ ]:


#1-D scatter plot of axil_nodes
plt.plot(survived["axil_nodes"], np.zeros_like(survived['axil_nodes']), 'o')
plt.plot(died["axil_nodes"], np.zeros_like(died['axil_nodes']), '^')

plt.show()


# # Histogram, PDF, CDF

# In[ ]:


#1-D scatter plot of operation_year
plt.plot(survived["operation_year"], np.zeros_like(survived['operation_year']), 'o')
plt.plot(died["operation_year"], np.zeros_like(died['operation_year']), '^')

plt.show()


# In[ ]:


#1-D scatter plot of age
plt.plot(survived["age"], np.zeros_like(survived['age']), 'o')
plt.plot(died["age"], np.zeros_like(died['age']), '^')

plt.show()


# **Observation:**
#     1. Nothing can be obtained from 1-D scatter plot as points are mixed up.

# # Conclusions

# After performing uni-variate analysis the features age and axil node are useful to build the model as
# 
# 1. if (age <= 34 years)
#         prediction("Survived 5 years or longer")
#    if (age >= 78 years)
#         prediction("Died within 5 years of operation")
#         
# 
# 2. if (axil_nodes <= 2)
#         prediction("Survived 5 years or longer")
#    if (axil_nodes >= 7)
#         prediction("Died within 5 years of operation")
#         
# 
# The multivariate analysis is not revealing any insides about the dataset.
# 
