#!/usr/bin/env python
# coding: utf-8

# # Exploratory data analysis (EDA) 

# # Haberman's Survival Data set
# 
# * __Dataset:__ [https://www.kaggle.com/gilsousa/habermans-survival-data-set]
# * __Description:__ The dataset contains cases from a study that was conducted between 1958 and 1970 at the University of   Chicago's  Billings Hospital on the survival of patients who had undergone surgery for breast cancer.
# 
# * __Objective:__ To analyze the survival of breast cancer patient for given 3 features.

# In[ ]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

hb = pd.read_csv('../input/habermans-survival-data-set/haberman.csv',names= ['age','op_year','nodes','survived'])
print(hb.head())


# In[ ]:


#Data points and freatures
print(hb.shape)


# In[ ]:


#Column names in the dataset
print(hb.columns)


# In[ ]:


#modifying 'survived' feature from 1,2 to categories Yes or no 
hb['survived'] = hb['survived'].map({1:"yes", 2:"no"})
hb['survived'] = hb['survived'].astype('category')
print(hb.head())


# In[ ]:


#No of prople survived
hb['survived'].value_counts()


# # Univariate Analysis
# 
# Univariate analysis is the analysis for the given dataset with respect to one feature (1-D Scatter plot) of a dataset,in this scenario patient age,operation performed year and auxiliary lymph nodes are the features.There are various method to visualize this analysis like PDF,CDF,Box plot and Whiskers and Violin plots. 

# # PDF (Probability Density Function)
# Probability density Function gives us count of numbers having a value from a given range of values of a feature.In X-axis there will values of feature and Y-axis gives a count of no having a value from x-axis. 
# 

# In[ ]:


#PDF analysis of all the features(fig 1.1)
sns.FacetGrid(hb, hue="survived", height=5)    .map(sns.distplot, "age")    .add_legend();
plt.show();


# In[ ]:


#fig(1.2)
sns.FacetGrid(hb, hue="survived", height=5)    .map(sns.distplot, "op_year")    .add_legend();
plt.show();


# In[ ]:


#fig(1.3)
sns.FacetGrid(hb, hue="survived", height=5)    .map(sns.distplot, "nodes")    .add_legend();
plt.show();


# # CDF (Cumulative distribution Function)
# CDF gives us a idea that how many percentage of data have value of a feature less than or equal to a particular value of a feature. For e.g. from the below graph we can se approx 70% of people had age less than or equal to 60.

# In[ ]:


#cdf analysis of all features(fig2.1)
counts, bin_edges = np.histogram(hb['age'], bins=10, 
                                 density = True)
pdf = counts/(sum(counts))
print(pdf);
print(bin_edges);
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf);
plt.plot(bin_edges[1:], cdf)


# In[ ]:


#fig(2.2)
counts, bin_edges = np.histogram(hb['op_year'], bins=10, 
                                 density = True)
pdf = counts/(sum(counts))
print(pdf);
print(bin_edges);
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf);
plt.plot(bin_edges[1:], cdf)


# In[ ]:


#fig(2.3)
counts, bin_edges = np.histogram(hb['nodes'], bins=10, 
                                 density = True)
pdf = counts/(sum(counts))
print(pdf);
print(bin_edges);
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf);
plt.plot(bin_edges[1:], cdf)


# # Box plot and whiskers
# This plot have a box from a value to anaother with a line in between. The lower line, line in between and upper line of the box is 25,50,75th percentile of the data.The whiskers which are above and below of data basically denotes the max and min data values   

# In[ ]:


#fig(3.1)
sns.boxplot(x ='survived',y = 'age',data = hb)
plt.show()


# In[ ]:


#fig(3.2)
sns.boxplot(x ='survived',y = 'op_year',data = hb)
plt.show()


# In[ ]:


#fig(3.3)
sns.boxplot(x ='survived',y = 'nodes',data = hb)
plt.show()


# # Violin plots
#    It follows the same concept of boxplot and whiskers.It has violin shape and black slot in center with a white point which is 50th percentile.The slot starts with 25th percentile and ends to 75th percentile.The lines above and below of the black slots are the same thing as whiskers in boxplot.  

# In[ ]:


#fig(4.1)
sns.violinplot(x="survived", y="age", data=hb, size=8)
plt.show()


# In[ ]:


#fig(4.2)
sns.violinplot(x="survived", y="op_year", data=hb, size=8)
plt.show()


# In[ ]:


#fig(4.3)
sns.violinplot(x="survived", y="nodes", data=hb, size=8)
plt.show()


# # Observations:
# * Almost 80% of people survived have positive auxiliary lymph node. (fig 1.3,2.3)
# * 50-75 percentile of people who couldn't survive had auxiliary lymph node around 5-11. (fig 3.3,4.3)
# * 50-75 percentile of people who could survive had auxiliary lymph node around 0-5. (fig 3.3,4.3) 
# 

# # Bi-variate analysis
# Bi-variate analysis is the analysis for the given dataset with respect to two feature (2-D Scatter plot,pair plot) of a dataset.In haberman dataset there are are 3 features, so 3C2 combinations are possible for 2-D plot
# 

# # Pair plot
# Pair plots are combined 2-D scatter plots of all combinations of features.

# In[ ]:


sns.set_style("whitegrid");
sns.pairplot(hb, hue="survived", vars = ["age", "op_year", "nodes"], size=3);
plt.show()


# # Observation
# * Classification of survival from the plots given above is difficult
# * Pair plot of features operation year and auxillary lymph node seems to be more informative among all the feature.
# * Majority of people who died, had their lymph node between 0-5 within the operation year from 1960-68.

# # Conclusion
# * As observed above the data points were too much overlapped within the features, therefore it is difficult to classify
# * if-else condition would not be sufficient for the prediction of the survival of patient.
# * Compared to all features, auxillary lymph node gives some intution for classification.
# * In the dataset, the data of survived people was much more than people who died after surgery. 

# In[ ]:




