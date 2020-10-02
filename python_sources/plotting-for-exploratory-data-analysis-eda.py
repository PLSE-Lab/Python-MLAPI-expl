#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import seaborn as sns
import matplotlib.pyplot as plt

import os
print(os.listdir("../input"))
haberman = pd.read_csv("../input/haberman.csv", header=None, names=['age', 'year', 'nodes', 'status'])
haberman.head()

# Any results you write to the current directory are saved as output.


# In[ ]:


print (haberman.shape)


# In[ ]:


#(Q) What are the column names in our dataset?
print (haberman.columns)


# In[ ]:


#(Q) How many data points for each class are present? 
haberman["status"].value_counts()


# We can see our dataset is not balanced, as out of 306 people 255 people suvived for 5 or more years and 81 of them died within 5 years

# In[ ]:


haberman.info()


# conclusuion: no null value found on the dataset

# **2-D Scatter Plot**

# In[ ]:


#2-D scatter plot:
haberman.plot(kind='scatter', x='age', y='nodes') ;
plt.show()

#cannot make much sense out it.We will include colour coding also.


# In[ ]:


# 2-D Scatter plot with color-coding.
sns.set_style("whitegrid");
sns.FacetGrid(haberman, hue="status", size=5)    .map(plt.scatter, "age", "nodes")    .add_legend();
plt.show();


# The points are not quite seperable.We can see that there is quite good concentration of data point When axil_node is 0.
# 
# Patients who are older than 50 and have axil nodes greater than 10 are more likely to die.
# 
# It is very much less likely to have patients with axil nodes more than 30.
# 
# From this Dataset we can say that the majority of operations are performed on people age range between 38 and 68

# **Pair-plot**

# In[ ]:


plt.close();
sns.set_style("whitegrid");
#sns.pairplot(haberman,hue="status", size=4)
sns.pairplot(haberman, size=4)
plt.show()


# In[ ]:


haberman_diedbefore5 = haberman.loc[haberman["status"] == 2];
haberman_aliveafter5 = haberman.loc[haberman["status"] == 1];


# In[ ]:


sns.FacetGrid(haberman, hue="status", size=5)    .map(sns.distplot, "nodes")    .add_legend();
plt.show();


# We can conclude that from this histogram (axil_node) that, Patients having 0 axil nodes are more likely to survive.

# In[ ]:


sns.FacetGrid(haberman, hue="status", size=5)    .map(sns.distplot, "age")    .add_legend();
plt.show();


# 1.This histogram is overlapping each other, but still we can say that people within range of 40-60 are more likely to die.
# 
# 2.People less than age 40 are more likely to survive.

# In[ ]:


sns.FacetGrid(haberman, hue="status", size=5)    .map(sns.distplot, "year")    .add_legend();
plt.show();


# In[ ]:


counts, bin_edges = np.histogram(haberman_aliveafter5['nodes'], bins=10, 
                                 density = True)
pdf = counts/(sum(counts))
print(pdf);
print(bin_edges);
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf);
plt.plot(bin_edges[1:], cdf)


counts, bin_edges = np.histogram(haberman_aliveafter5['nodes'], bins=20, 
                                 density = True)
pdf = counts/(sum(counts))
plt.plot(bin_edges[1:],pdf);

plt.show();


# In[ ]:


counts, bin_edges = np.histogram(haberman_aliveafter5['year'], bins=10, 
                                 density = True)
pdf = counts/(sum(counts))
print(pdf);
print(bin_edges);
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf);
plt.plot(bin_edges[1:], cdf)

counts, bin_edges = np.histogram(haberman_aliveafter5['nodes'], bins=10, 
                                 density = True)
pdf = counts/(sum(counts))
print(pdf);
print(bin_edges);
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf);
plt.plot(bin_edges[1:], cdf)


counts, bin_edges = np.histogram(haberman_aliveafter5['age'], bins=10, 
                                 density = True)
pdf = counts/(sum(counts))
print(pdf);
print(bin_edges);
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf);
plt.plot(bin_edges[1:], cdf)


plt.show();


# In[ ]:


#Mean, Variance, Std-deviation,  
print("Means:")
print(np.mean(haberman_aliveafter5["nodes"]))
print(np.mean(haberman_diedbefore5["nodes"]))

print(np.mean(haberman_aliveafter5["age"]))
print(np.mean(haberman_diedbefore5["age"]))

print(np.mean(haberman_aliveafter5["year"]))
print(np.mean(haberman_diedbefore5["year"]))


print("\nStd-dev:");
print(np.std(haberman_aliveafter5["nodes"]))
print(np.std(haberman_diedbefore5["nodes"]))

print(np.std(haberman_aliveafter5["age"]))
print(np.std(haberman_diedbefore5["age"]))


print(np.std(haberman_aliveafter5["year"]))
print(np.std(haberman_diedbefore5["year"]))


# **Median, Percentile, Quantile, IQR, MAD**

# In[ ]:


#Median, Quantiles, Percentiles, IQR.
print("\nMedians:")
print(np.median(haberman_aliveafter5["nodes"]))
print(np.median(haberman_diedbefore5["nodes"]))



print("\nQuantiles:")
print(np.percentile(haberman_aliveafter5["nodes"],np.arange(0, 100, 25)))
print(np.percentile(haberman_diedbefore5["nodes"],np.arange(0, 100, 25)))


print("\n90th Percentiles:")
print(np.percentile(haberman_aliveafter5["nodes"],90))
print(np.percentile(haberman_diedbefore5["nodes"],90))


from statsmodels import robust
print ("\nMedian Absolute Deviation")
print(robust.mad(haberman_aliveafter5["nodes"]))
print(robust.mad(haberman_diedbefore5["nodes"]))


# In[ ]:


#Median, Quantiles, Percentiles, IQR.
print("\nMedians:")
print(np.median(haberman_aliveafter5["age"]))
print(np.median(haberman_diedbefore5["age"]))



print("\nQuantiles:")
print(np.percentile(haberman_aliveafter5["age"],np.arange(0, 100, 25)))
print(np.percentile(haberman_diedbefore5["age"],np.arange(0, 100, 25)))


print("\n90th Percentiles:")
print(np.percentile(haberman_aliveafter5["age"],90))
print(np.percentile(haberman_diedbefore5["age"],90))


from statsmodels import robust
print ("\nMedian Absolute Deviation")
print(robust.mad(haberman_aliveafter5["age"]))
print(robust.mad(haberman_diedbefore5["age"]))


# In[ ]:


sns.boxplot(x='status',y='nodes', data=haberman)
plt.show()


sns.boxplot(x='status',y='age', data=haberman)
plt.show()


sns.boxplot(x='status',y='year', data=haberman)
plt.show()


# In[ ]:


# A violin plot combines the benefits of the previous two plots 
#and simplifies them

# Denser regions of the data are fatter, and sparser ones thinner 
#in a violin plot

sns.violinplot(x="status", y="nodes", data=haberman, size=8)
plt.show()

sns.violinplot(x="status", y="age", data=haberman, size=8)
plt.show()

sns.violinplot(x="status", y="year", data=haberman, size=8)
plt.show()


# **Multivariate probability density, contour plot.**

# In[ ]:


#2D Density plot, contors-plot
sns.jointplot(x="age", y="nodes", data=haberman_aliveafter5, kind="kde");
plt.show();


# **Final Observations**
# 
# 1.We can observe that given dataset is unbalanced. There are 255 people out of 306 people survived for 5 or more year and 81 died within 5 years. So, this is not balanced pair.
# 
# 2.There are no null value on the dataset.
# 
# 3.The points are not quite separable. We can see that there is quite good concentration of data point when axil_node is 0.
# 
# 4.Patients who are older than 50 and have axil nodes greater than 10 are more likely to die.
# 
# 5.It is very much less likely to have patients with axil nodes more than 30.
# 
# 6.From this Dataset we can say that the majority of operations are performed on people age range between 38 and 68.
# 
# 7.This histogram is overlapping each other, but still we can say that people within range of 40-60 are more likely to die.
# 
# 8.People less than age 40 are more likely to survive.
# 
# 9.from Node box plot, we have say that nodes close to zero and positive have more chances of survial for more than 5 years in comparision to bigger nodes.
# 
