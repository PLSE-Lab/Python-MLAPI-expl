#!/usr/bin/env python
# coding: utf-8

# # Plotting for Exploratory data analysis (EDA)
# 

#           

#       

#      

# ### 1. First we import all the libraries we use. and we import our data. As the data does not have headers, while importing our data we used attributesheader as none and also attributes nams to define our headers.

# In[ ]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import axes3d
import warnings 
warnings.filterwarnings("ignore")

"""Load haberman.csv into a pandas dataFrame.
Age->age,
Operation Year->operation_year,
Positive Lymph Nodes->posln,
Survival Status After years -> survival_status"""
haberman = pd.read_csv("../input/haberman.csv",header=None, names=['age', 'operation_year', 'positive_lymph_nodes', 'survival_status'])


# In[ ]:


print (haberman.shape)


# In[ ]:


print(haberman.columns)


# ### Observation:
# ##### There are 306 rows and 4 columns. 305 data ponts and 4 features in the data.

# In[ ]:


print(haberman.head())


# In[ ]:


print(haberman.describe())


# ### Observations:
# ##### The data with age column has max and min of 30 and 83 and also has a mean of 52.
# ##### The data is very small with number of records (306)

# In[ ]:


haberman["survival_status"].value_counts()


# ### Observation:
# ##### More than 73% of the data of survival status is of alive.

# # 2-D Scatter Plot

# In[ ]:


haberman.plot(kind='scatter', x='age', y='operation_year',title='Scatter Plot between Operation Year and Age')
plt.show()


# In[ ]:


sns.set_style("whitegrid");
sns.FacetGrid(haberman, hue="survival_status", size=10)    .map(plt.scatter, "operation_year", "positive_lymph_nodes")    .add_legend();
plt.show();


# ### Observarion: 
# ##### Can't classify between 1 and 2 because both are fully overlapped

# #  Pair-plot

# In[ ]:


plt.close();
sns.set_style("whitegrid");
sns.pairplot(haberman, hue="survival_status",vars = ["age", "operation_year", "positive_lymph_nodes"])     .add_legend()     .fig.suptitle("Pairplot for all the features");
plt.show()


# ### Observation: 
# ##### Not a single pair-plot can classify between 1 and 2 because everyone of them are overlapped

# # Histogram, PDF, CDF

# In[ ]:


import numpy as np
haberman_alive = haberman.loc[haberman["survival_status"] == 1];
haberman_dead = haberman.loc[haberman["survival_status"] == 2];

plt.title('1-D scatter plot for the number of positive lymph nodes for the dead and alive people')
plt.xlabel('No. of positive Lymph Nodes')
plt.ylabel('Y-Axis')
plt.plot(haberman_alive["positive_lymph_nodes"], np.zeros_like(haberman_alive['positive_lymph_nodes']), 'o')
plt.plot(haberman_dead["positive_lymph_nodes"], np.zeros_like(haberman_dead['positive_lymph_nodes']), 'o')

plt.show()


# In[ ]:


sns.FacetGrid(haberman, hue="survival_status", size=10)    .map(sns.distplot, "positive_lymph_nodes")    .add_legend().fig.suptitle('Histogram of positive axuxillary lymph nodes');
plt.ylabel("Counts")
plt.show();


# ### Observation:
# ##### Maximum no of people survived who have Number of positive lymph nodes less than five.

# In[ ]:


sns.FacetGrid(haberman, hue="survival_status", size=5)    .map(sns.distplot, "operation_year")    .add_legend().fig.suptitle('histogram of operation year')
plt.ylabel("Counts")
plt.show();


# Observation: Not classifiable and nothing can be inferred.

# In[ ]:


sns.FacetGrid(haberman, hue="survival_status", size=5)    .map(sns.distplot, "age")    .add_legend().fig.suptitle('Histogram of Age');
plt.ylabel("Counts")
plt.show();


# ### Observations: 
# ##### Not classifiable and nothing can be inferred.
# ##### Among all the variables the number of positive lymph nodes is the feature we would choose for further analysis.

# In[ ]:


counts, bin_edges = np.histogram(haberman_alive['positive_lymph_nodes'], bins=10, 
                                 density = True)
pdf = counts/(sum(counts))
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf,label='pdf of People Alive');
plt.plot(bin_edges[1:],cdf,label='cdf of People Alive')


counts, bin_edges = np.histogram(haberman_dead['positive_lymph_nodes'], bins=10, 
                                 density = True)
pdf = counts/(sum(counts))
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf,label='pdf of People Dead');
plt.plot(bin_edges[1:],cdf,label='cdf of People Dead')

plt.title('pdf and cdf of positive lymph nodes of alive and dead')
plt.xlabel('positive_lymph_nodes')
plt.ylabel('probabilities')
plt.legend()

plt.show();


# In[ ]:


counts, bin_edges = np.histogram(haberman_alive['positive_lymph_nodes'], bins=10, 
                                 density = True)
pdf = counts/(sum(counts))
print(pdf);
print(bin_edges)

#compute CDF
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf,label='pdf of People Alive')
plt.plot(bin_edges[1:],cdf,label='cdf of People Alive')


plt.title('pdf and cdf of positive lymph nodes of alive')
plt.xlabel('positive_lymph_nodes')
plt.ylabel('probabilities')
plt.legend()


# ### Observation:
# ##### About 83% of the people who survived had less than or equal to 3 positive lymph nodes. 

# In[ ]:


counts, bin_edges = np.histogram(haberman_dead['positive_lymph_nodes'], bins=10, 
                                 density = True)
pdf = counts/(sum(counts))
print(pdf);
print(bin_edges)

#compute CDF
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf,label='pdf of People Alive')
plt.plot(bin_edges[1:],cdf,label='cdf of People Alive')


plt.title('pdf and cdf of positive lymph nodes of dead')
plt.xlabel('positive_lymph_nodes')
plt.ylabel('probabilities')
plt.legend()


# ### Observations:
# ##### About 58% of people who had died had no. of positive lymph nodes less than 5
# ##### As both of the categories alive and dead have a significant amount of percentage of people who have less number of positive lymph nodes. It can be decided from this that the Number of positive lymph nodes is not the only factor.

# # Mean, Variance and Std-dev

# In[ ]:


#Mean, Variance, Std-deviation,  
print("Means:")
print(np.mean(haberman_alive["positive_lymph_nodes"]))
#Mean with an outlier.
print(np.mean(np.append(haberman_alive["positive_lymph_nodes"],50)));
print(np.mean(haberman_dead["positive_lymph_nodes"]))

print("\nStd-dev:");
print(np.std(haberman_alive["positive_lymph_nodes"]))
print(np.std(haberman_dead["positive_lymph_nodes"]))


# In[ ]:


print("\nMedians:")
print(np.median(haberman_alive["positive_lymph_nodes"]))
#Median with an outlier
print(np.median(np.append(haberman_alive["positive_lymph_nodes"],50)));
print(np.median(haberman_dead["positive_lymph_nodes"]))

print("\nQuantiles:")
print(np.percentile(haberman_alive["positive_lymph_nodes"],np.arange(0, 100, 25)))
print(np.percentile(haberman_dead["positive_lymph_nodes"],np.arange(0, 100, 25)))

#Determining number of people alive or dead having number of nodes 0
print(np.percentile(haberman_alive["positive_lymph_nodes"],np.arange(0, 100, 1)))
print(np.percentile(haberman_dead["positive_lymph_nodes"],np.arange(0, 100, 1)))

print("\n90th Percentiles:")
print(np.percentile(haberman_alive["positive_lymph_nodes"],90))
print(np.percentile(haberman_dead["positive_lymph_nodes"],90))


from statsmodels import robust
print ("\nMedian Absolute Deviation")
print(robust.mad(haberman_alive["positive_lymph_nodes"]))
print(robust.mad(haberman_dead["positive_lymph_nodes"]))


# ### Observations:
# ##### About 52% of the peope who are alive have number of nodes 0.
# ##### About 23% of the peope who are dead have number of nodes 0.

# # Box plot and Whiskers

# In[ ]:


sns.boxplot(x='survival_status',y='positive_lymph_nodes', data=haberman)
plt.show()


# ### Observations: 
# ###### People who survived had only 75th percentile value of positive axillary node is at 3. The 25th and 50th percentiles are overlapped.
# ###### People who did not survive had only 25th percentile value of positive axillary nodes at 3, 50thpercentile value of positive axillary node is at 3 and 75th percentile value of positive axillary node is at 11.

# # Violin plots

# In[ ]:


sns.violinplot(x='survival_status',y='positive_lymph_nodes', data=haberman, size=8)
plt.show()


# ### Observations:
# ###### 25th and 50th percentile of survivors coincide
# ###### 50th percentile of survivors have 0 positive nodes
# ###### 75th percentie of survivors have less than 3 positive axilary nodes
# ###### 25th percentile of dead have less than or equal to 1 positive axilary node
# ###### 50th percentile of dead have positive axilary nodes below 5
# ###### 75th percentile of dead have positive nodes below 11

# # Multivariate probability density, contour plot.
# 

# In[ ]:


sns.jointplot(x="survival_status", y="positive_lymph_nodes", data=haberman, kind="kde");
plt.show();


# ### Observation:
# ##### There are more number of points of values where the people have survived.

#         

#         

# # Summarizing plots

# ## Conclusion:
# #### Less number of positive axillary nodes always has greater chance of survival
# #### Cannot properly classify from any of the exploratory data analysis
# #### Very less number of dataset and imbalanced dataset
# #### Positive Axillary lymph node is a important in the dataset but it cannot be solely be accounted for all the data analysis.
# #### We need more features and more data to rather conclude or classify who will survive and who will not based on this data.
