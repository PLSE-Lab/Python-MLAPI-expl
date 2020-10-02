#!/usr/bin/env python
# coding: utf-8

# <h1> EDA for Habermans Dataset </h1>

# <h5> About the Dataset</h5>

# The dataset contains cases from a study that was conducted between 1958 and 1970 at the University of Chicago's Billings Hospital on the survival of patients who had undergone surgery for breast cancer.

# <h5> Attribute Information </h5>

# 1.Age of patient at time of operation (numerical)
# 
# 2.Patient's year of operation (year - 1900, numerical)
# 
# 3.Number of positive axillary nodes detected (numerical)
# 
# 4.Survival status (class attribute)  1 = the patient survived 5 years or longer   2 = the patient died within 5 year

# <h5> Objective </h5>

# To classify a new patient's <b>Survival Status</b> on the basis of other three attributes.

# In[ ]:


# importing the required python libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


import os
print(os.listdir("../input"))

# loading the dataset
df=pd.read_csv("../input/haberman.csv")
df.head()


# In[ ]:


df.columns=["age","year","nodes","status"]


# In[ ]:


# (Q) how many data-points/instances and attributes?
df.shape


# In[ ]:





# Number of Instances: 306
# 
# Number of Attributes: 4 , among which 3 are features and last one is label.

# In[ ]:


print(df.isnull().sum())
# No null values in the dataset


# From above we can infer that there are no null values present in any of the columns and hence the no. of counts of each attributes are equal to the no. of instances (306).
# 

# In[ ]:


# (Q) How many patient records of each survival status are present ?
df["status"].value_counts()


# So the given dataset is an imbalanced dataset as the no. of patients that survived 5 years or longer (1)  are more than the no. of patients that died within 5 year (2)

# <h2> Univariate Analysis</h2>

# Now as we have got basic idea about our dataset let's dive deeper and find which of the above fetures could be most vital in determining the srvival status of patients. Let's do some univariate analysis of the dataset.
# 
# Univariate involves the analysis of a single variable and hence we will look into each features to find inferences.

# In[ ]:


# dividing the dataset on the basis of survival status
status_1=df[df["status"]==1]
status_2=df[df["status"]==2]


# In[ ]:


status_1.describe()


# In[ ]:


status_2.describe()


# 1. The mean of ages and operation years of both status are nearly same.
# 2. The mean of nodes of status 2 is more than status 1 and hence we can say that more no. of positive axilary nodes are found in patients with staus 2.

# <h3>Histogram, PDF, CDF</h3>

# In[ ]:


# Plotting the hisograms

# Histogram for age
sns.FacetGrid(df, hue="status", size=5)    .map(sns.distplot, "age")    .add_legend();
plt.show();


# We plotted the histogram for 'age' but we are not getting a great output as the histograms are overlapping.
# 
# Hence we can infer that 'age' is not a great decisive factor in determining the survival status.

# In[ ]:


# Histogram for year

sns.FacetGrid(df, hue="status", size=5)    .map(sns.distplot, "year")    .add_legend();
plt.show();


# We plotted the histogram for 'year' but we are not getting a great output as the histograms are overlapping.
# 
# Hence we can infer that 'year' is also not a great decisive factor in determining the survival status.

# In[ ]:


# Histogram for nodes
sns.FacetGrid(df, hue="status", size=5)    .map(sns.distplot, "nodes")    .add_legend();
plt.show();


# The histogram for 'nodes' of both srvival status are overlapping too but from the plot it can be inferred that a large no. of patients that has survival status '1' i.e, the patients that survive for longer period of time has less no. of positive axillary nodes detected.
# 
# Hence according to me the attribute 'nodes' is more dominant than others in determining survival status.

# In[ ]:


# Plotting pdf and cdf plots

# for age

counts, bin_edges = np.histogram(status_1["age"], bins=10, 
                                 density = True)
pdf = counts/(sum(counts))

#compute CDF
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf,label='pdf (1)')
plt.plot(bin_edges[1:], cdf,label='cdf (1)')


counts, bin_edges = np.histogram(status_2["age"], bins=10, 
                                 density = True)
pdf = counts/(sum(counts))

#compute CDF
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf,label='pdf (2)')
plt.plot(bin_edges[1:], cdf,label='cdf (2)')
plt.legend()
plt.xlabel('age')
plt.show();


# The cdf plot for 'age' also does not depicts any valuable inference as it overlaps for both survival status.

# In[ ]:


# for year

counts, bin_edges = np.histogram(status_1["year"], bins=10, 
                                 density = True)
pdf = counts/(sum(counts))
#compute CDF
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf,label='pdf (1)')
plt.plot(bin_edges[1:], cdf,label='cdf (1)')


counts, bin_edges = np.histogram(status_2["year"], bins=10, 
                                 density = True)
pdf = counts/(sum(counts))


#compute CDF
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf,label='pdf (2)')
plt.plot(bin_edges[1:], cdf,label='cdf (2)')

plt.legend()
plt.xlabel('year')

plt.show();


# The cdf plot for 'year' also does not depicts any valuable inference as it overlaps for both survival status.

# In[ ]:


#for nodes

counts, bin_edges = np.histogram(status_1["nodes"], bins=10, 
                                 density = True)
pdf = counts/(sum(counts))


#compute CDF
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf,label='pdf (1)')
plt.plot(bin_edges[1:], cdf,label='cdf (1)')

counts, bin_edges = np.histogram(status_2["nodes"], bins=10, 
                                 density = True)
pdf = counts/(sum(counts))


#compute CDF
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf,label='pdf (2)')
plt.plot(bin_edges[1:], cdf,label='cdf (2)')

plt.legend()
plt.xlabel('nodes')

plt.show();


# From the above plot we can easily visualise that around 90% patients having nodes less than 10 have status '1' while around 60% patients having nodes less than 10 have status '2'.
# 
# Hence, the patients that survive for longer period of time has less no. of positive axillary nodes detected and thus the attribute 'nodes' is more dominant than others in determining survival status.

# <h3>Box Plot and Violin Plot

# In[ ]:


# box plot for age
sns.boxplot(x='status',y='age', data=df)
plt.grid(True, linestyle='-', which='major', color='lightgrey',
               alpha=0.5)
plt.show()


# From the box plot we can easily get to know that about 75% of the age of patients having status '2' overlaps with patients having status '1'. Hence considering 'age' as a dominant feature is not useful.

# In[ ]:


# boxplot for year

sns.boxplot(x='status',y='year', data=df)
plt.grid(True, linestyle='-', which='major', color='lightgrey',
               alpha=0.5)
plt.show()


# From the box plot we can easily get to know that about 70% of the year of operations of patients having status '1' overlaps with patients having status '2'.
# Hence considering 'year' as a dominant feature is not useful.

# In[ ]:


# box plot for nodes

sns.boxplot(x='status',y='nodes', data=df)
plt.grid(True, linestyle='-', which='major', color='lightgrey',
               alpha=0.5)
plt.show()


# 1. Above plot depicts that only about 25% nodes with status '2' has overlapping with status '1'. Hence chance of error decreses in chosing 'nodes' as a dominanat feature in achieving the main objective.
# 
# 2. We can also see that the box plot for status '1' is dense and skewed towards lower value as about 80% of the value is less than 5.
# 
# 3. The above plot contains a lot of outliers which need to be pre-processed before modelling.

# In[ ]:


# violin plots of features

sns.violinplot(x="status", y="age", data=df, size=8)
plt.grid(True, linestyle='-', which='major', color='lightgrey',
               alpha=0.5)
plt.show()
sns.violinplot(x="status", y="year", data=df, size=8)
plt.grid(True, linestyle='-', which='major', color='lightgrey',
               alpha=0.5)
plt.show()
sns.violinplot(x="status", y="nodes", data=df, size=8)
plt.grid(True, linestyle='-', which='major', color='lightgrey',
               alpha=0.5)
plt.show()


# <h4>Observation</h4>
# 
# After doing the above univariate analysis I can infer that 'nodes' can be the most useful feature among all others.

# <h2> Bivariate Analysis

# <h3> Pair Plot

# In[ ]:


# pair plot between combination of feature
sns.set_style("whitegrid");
sns.pairplot(df, hue="status", size=3);
plt.show()


# Above are the pair plots for the given combinations of features but they are not useful in classfication as no plot is linearly seperable .
# 

# <h3> Summary

# We plotted multiple plots above to get a inference about each feature's effect on the label and found that only the attribute 'nodes' gives us a clear idea that the patients having less no of positive axilary nodes live longer as about 80% of the patients having survival status '1' have no. of 'nodes' less than 5.
