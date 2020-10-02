#!/usr/bin/env python
# coding: utf-8

# 
#  
#  #  1.Exploratory Data Analysis : Haberman Data Set
#  
#  ## 1.1 Description 
#  
#   - EDA:Exploratory data analysis is the task of analysis of data using simple tools from statistics,plotting tools etc 
#   - Data set:Haberman Cancer Survival dataset
#   - EDA By:Berly Susan Babu
#   - EDA on:15th April 2019
#   
# ## 1.2 Haberman Cancer Survival Dataset 
# 
#   - Source:https://www.kaggle.com/gilsousa/habermans-survival-data-set
#   - Variables:age,year,nodes
#   - Class-label:status
#   - Haberman data set contains details of the patients who undergone breast cancer surgery from 1958-1970 in University of           Chicago's Billings Hospital
#   - Nodes in variables are the lymph nodes which are bean shaped,small clumps kind of filters in lymphatic system.The axillary       lymph node under the underarm is the first place where the cancel cells traps
#   - Age in variables,shows the age of patients who undergone surgery
#   - Year in variables, shows the year in which patient undergone surgery
#   - Status in class-label,represents 1 when patient survived 5 years or more after surgery else 2
#   - Objective:Predict the survival status of a new patient undergone surgery given 3 variables/features.
# 

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

#Load haberman.csv into pandas dataframe
haberman=pd.read_csv("../input/haberman.csv")
print(haberman)


# In[ ]:


#Shape of data,ie no: of datapoints in haberman datset

print(haberman.shape)


# __Observation__: 306 data points where imported which is having 4 attributes

# In[ ]:


#Labels of columns present in haberman dataset

print(haberman.columns)


# __Observation__: Able to see data type for each feature and class label

# In[ ]:


#Datapoints belongs to each status

print(haberman["status"].value_counts())


# __Observation__: There are 225 patients who survived for 5 years or more and 81 patients not survived atleast 5 years

# ## 1.3 Scatter plot:2D

# In[ ]:


#2-D scatter plot:


haberman.plot(kind='scatter', x='nodes', y='age') ;
plt.show()


# __Observation__: Not able to distinguish between datapoints since every point is of same color and overlapping

# In[ ]:


#2-D scatter plot with color coding for each type of datapoints

sns.set_style("darkgrid")
sns.FacetGrid(haberman,hue="status",size=4).map(plt.scatter,"nodes","age").add_legend()
plt.show()


# __Observation__:Able to classify between survival more than  5 years and less than 5 years.Blue dots represents survival status 1 and orange dots represents survival status 2.But datapoints are over lapping,so seperating from each other is harder.

# ## 1.4 Pair Plot
# 

# In[ ]:


#Pair plot is a smart way to visualise plots between all features of data set as pairs
plt.close();
sns.set_style("darkgrid");
sns.pairplot(haberman,hue="status",size=3,vars=['age','year','nodes'])
plt.show()


# __Observation__:
#  - Plot 1,5,9 are histograms so we can vomit as of now
#  - Plot 2 is between age of the patient and year of operation and it seems to me overlapping and not able to distinguish each      other.
#  - Plot 4 also same which is 90 degree rotated form of plot 1.
#  - Plot 3 is the plot between age of patient and no of axillary nodes.Overlapping is there but it is better than plot 2,plot 4
#  - Plot 7 is the 90 degree rotated form of plot 3
#  - Plot 6 is the plot between year of operation and nodes.Here overlapping points is almost similar like plot 2
#  - Plot 8 is same like plot 6 with a 90 degree rotation
#  - So for further analysis taking Plot 3

# ## 1.5 Histogram,PDF,CDF
# 
# ### (1.5.1) 1 D Scatter Plot

# In[ ]:


#1D Scatter plot-Plotting of each of the variables by seeting y axis as zero and x axis the intended feature for each class- label
#Below is the 1D scatter plot on nodes feature

haberman_one=haberman.loc[haberman["status"]==1]
haberman_two=haberman.loc[haberman["status"]==2]

#Plotting
plt.plot(haberman_one["nodes"], np.zeros_like(haberman_one['nodes']), 'o')
plt.plot(haberman_two["nodes"], np.zeros_like(haberman_two['nodes']), 'o')
plt.show()


# __Observation__:Survival status is overlapping while plotted 1d scatter plot using axillary nodes feature.So cant classify data based on this.

# ### (1.5.2)Probability Density Function(PDF)

# In[ ]:


#pdf is the number of data point situated at a particular region 

sns.FacetGrid(haberman,hue='status',size=5).map(sns.distplot,'nodes').add_legend()
plt.show()


# __Observation__:
# 
#  - For axillary node whose value less than 0 chance for survival is more
#  - For axillary node value between 0-3 chances of survival is more
#  - For axillary node value greater than 3 chances of survival of patient is less

# In[ ]:


sns.FacetGrid(haberman,hue='status',size=5).map(sns.distplot,'age').add_legend()
sns.FacetGrid(haberman,hue='status',size=5).map(sns.distplot,'year').add_legend()
plt.show()


# __Observation__:
# 
#  - Not able to distinguish between survival chances for the features age and year since it is showing almost same density at data points.So it is better to go with node numbers for further analysis

# ### (1.5.3)Cumilative distributive function(CDF)

# In[ ]:


# CDF helps in calculating the percentage of people survived after surgery

#Below code evaluates cdf of patients surviving more,that is survival status one

counts, bin_edges = np.histogram(haberman_one['nodes'], bins=10, 
                                 density = True)
pdf = counts/(sum(counts))
print(pdf);
print(bin_edges);
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf);
plt.plot(bin_edges[1:], cdf)

plt.show();


# __Observation__:
# 
#  - 85% of patients having survival chances if the nodes detected is less than 5
#  - 100 % of survived patients have nodes less than 40
# 
#      

# In[ ]:




#Below code evaluates cdf of patients surviving less,that is survival status two

counts, bin_edges = np.histogram(haberman_two['nodes'], bins=10, 
                                 density = True)
pdf = counts/(sum(counts))
print(pdf);
print(bin_edges);
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf);
plt.plot(bin_edges[1:], cdf)

plt.show();


# __Observation__:
# 
#  - 55% of patients having survival chances less if the nodes detected is less than 5
#  - 100 % patients having survival chances less if the nodes is greater than 35
# 

# In[ ]:


#CDF plotting together for more survival and less survival chances
counts, bin_edges = np.histogram(haberman_one['nodes'], bins=10, 
                                 density = True)
pdf = counts/(sum(counts))
print(pdf);
print(bin_edges);
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf)
plt.plot(bin_edges[1:], cdf)
counts, bin_edges = np.histogram(haberman_two['nodes'], bins=10, 
                                 density = True)
pdf = counts/(sum(counts))
print(pdf);
print(bin_edges);
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf)
plt.plot(bin_edges[1:], cdf)
plt.show();


# __Observation__:
# 
#  - For a node number less than 5 survival chances of patients are far more.  
#  - For node numbers from 5-25 survival chances is more
#  - For node numbers from 25-35 slightly more chances of survival
#  - For node numbers greater than 35,equal chances are for less survival and more survival
# 

# ## 1.6 Mean, Variance and Std-dev

# In[ ]:


# Mean is the average of all data points
# Variance is the summation of  all data points deviation from its mean,that is spread.if spread is less data point will be closer to the mean 
# Standard deviation is the square root of mean
print("Means:")
print(np.mean(haberman_one["nodes"]))
#Mean with an outlier.
print(np.mean(np.append(haberman_one["nodes"],50)));
print(np.mean(haberman_two["nodes"]))


print("\nStd-dev:");
print(np.std(haberman_one["nodes"]))
print(np.std(haberman_two["nodes"]))


# __Observation__:
# 
#  - For patients having less number of nodes survival chance is more even with outlier
#  - For patients having more number of nodes survivaval chances is less

# ## 1.7 Median, Percentile, Quantile, IQR, MAD

# In[ ]:


#Median, Quantiles, Percentiles, IQR.
print("\nMedians:")
print(np.median(haberman_one["nodes"]))
#Median with an outlier
print(np.median(np.append(haberman_one["nodes"],50)));
print(np.median(haberman_two["nodes"]))



print("\nQuantiles:")
print(np.percentile(haberman_one["nodes"],np.arange(0, 100, 25)))
print(np.percentile(haberman_two["nodes"],np.arange(0, 100, 25)))

print("\n90th Percentiles:")
print(np.percentile(haberman_one["nodes"],90))
print(np.percentile(haberman_two["nodes"],90))


from statsmodels import robust
print ("\nMedian Absolute Deviation")
print(robust.mad(haberman_one["nodes"]))
print(robust.mad(haberman_two["nodes"]))


# __Observation__
#    - Average nodes for the patients having more survival chances is 0 and while nodes are 4 for the patients having less              survival chances
#    - Nearly 50% of nodes are 0 for the most survival patients and 75% of nodes are 3 for the patients who survived,that is 
#      25% (75-50) people who survived having 3 nodes only.
#    - Average nodes for the patients who survived less is 4
#    - Nearly 25 % of nodes are 1 for those who didnt survive
#    - Nearly 50% of nodes are 4 those who didnt survive,that is 25%(50-25) nodes where 4 for those who didnt survive
#    - 75% of nodes are 11 for the patients who didnt survive,that is 25%(75-50) people who didnt survive having 11 nodes.
#    - At 90th percentile patients having 8 nodes or less survived while 11 or more not survived.
#    - Nodes of patients having most survival chance is 0 while calculating median absolute deviation whereas 
#      almost 6 for  patients having less survival chance
#    

# ## 1.8 Box plot and Whiskers(Univariate probability density)

# In[ ]:


#Box-plot with whiskers: another method of visualizing the  1-D scatter plot more intuitively.

sns.boxplot(x='status',y='nodes', data=haberman)
plt.show()


# __Observation__:
#    
#    - 50% and 25% of more survival patients are overlapping ,hence 50% of patients having more survival chance 
#      if there nodes is 0
#    - 75% having more survival chance if nodes are less than 4 
#    - 25%-50% having less survival chance for the nodes less than 3.So there is a chance of getting error 
#      while evaluating between 0-3 nodes
#    - 75% having less survival chance for the nodes 0-11
#    - Most of the points above 11 nodes lie in less survival chance patient
#    

# ## 1.9 Violin Plot(Univariate probability density)

# In[ ]:


# A violin plot combines the benefits of the previous two plots 
#and simplifies them

# Denser regions of the data are fatter, and sparser ones thinner 
#in a violin plot

sns.violinplot(x='status',y='nodes', data=haberman, size=8)
plt.show()


# __Observation__:
#    
#    - For the patients having more chances of survival most of the data points are located near 0 nodes and varies between 0-7
#    - For the patients having less chances of survival most of the data points are located near 3 nodes and varies between 0-11

# ## 1.10 Contour Plot(Multivariate probability density)

# In[ ]:


#Contour plot is a method of visualizing the  2-D scatter plot more intuitively
sns.jointplot(x="age", y="nodes", data=haberman_one, kind="kde");
plt.show();


# __Observation__:
#    
#    - Density of plot is more for the age between 45-60 and having nodes 0-3 for the patients survived

# ## 1.11 Inference
# 
#   Able to detect the survival chance of a patient having cancer through EDA technique with the help of python libraries.

# In[ ]:




