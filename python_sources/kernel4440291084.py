#!/usr/bin/env python
# coding: utf-8

# ### Exploratory Data analysis For Haberman Dataset

# #### Simple Definitions and Terminologies

# ##### What is EDA ?

# Exploratory Data Analysis (EDA) is the initial analysis of data using Statistics and Ploating techniques. It helps to get a deep insight of dataset, Extract important veriables, Spot outliers, Helps to Test Hypothesis. We can perform EDA by following methods
# 
# 1) Univariate Analysis
# 
# 2) Bivariate Analysis
# 
# 3) Multivariate Analysis
# 
# 4) Dimensionality Reduction

# #### Haberman Dataset Description:
#  Objective of analyzing Haberman dataset is to predict survival status based on the Age, Year Of Operation and Positive Lymph Nodes.
# #### Columns Description
# 1) Age - Age of Patient at the time of operation (Numaric)
# 
# 2) Year - Patient's Year of operation (Numaric)
# 
# 3) Nodes - Number of positive Axillary Nodes detected (Numaric)
# 
# 4) Status - Status contains two values 1 -> Patients survived 5 years or more and 2 -> Patients survived less than 5 years

# In[5]:


# Import packages and load the data

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the Haberman dataset from csv
df = pd.read_csv('../input/haberman.csv')
new_df = pd.read_csv('../input/haberman.csv')
print(df.head(5))

# Get Dataset info to see the does not have missing values
print('========== Dataset Info =========')
print(df.info())


# ##### High Level Statistics 

# In[6]:


# (Q) how many data-points and features?
# Haberman Dataset contains total 306 Datapoints, 3 features and 1 class
print(df.shape)


# In[7]:


#(Q) What are the column names in our dataset?
print(df.columns)


# In[8]:


#(Q) How many data points for each class are present?
print(df['status'].value_counts())


# In[9]:


# (Q) How many patient Survived 5 or more years, and How many died in less than 5 years

new_df['status'] = new_df['status'].map({1:'Yes', 2:'No'})
new_df['status'] = new_df['status'].astype('category')
print('Patients survived more than 5 years ?')
print(new_df['status'].value_counts())

# (Q) Balanced-dataset vs Imbalanced datasets
# Haberman is a imbalanced dataset because the number of points for each class are not same or not in 50/50 or even in 60/40


# ### Bi-variate Analysis
# 
# The purpose of Univariate analysis is to analyse the single feature, by finding pattern in single feature. we can perform Univariate analysis using Distribution, Scatter Plots, Pair Plot, Histogram, PDF, CDF

# #### Scatter Plot

# In[10]:


df.plot(kind='scatter', x = 'nodes', y='age')
plt.grid()
plt.show()


# #### Observation
# This scatter plot is showing overlaped data to we need to add color to distinguish data. so we can visually separate the 
# different variables
# 
# #### Use Seaborn
# 
# Below plot draw using the seaborn the blue dots reprasent the 1 -> Patients survived 5 years or more and Orange dot 2 -> Patients survived less than 5 years. Still we would not able to saperate the data.

# In[11]:


# plot data using seaborn library.
# Add color for each class

sns.set_style('whitegrid')
sns.FacetGrid(df, hue = 'status', height = 4)    .map(plt.scatter, 'nodes','age')    .add_legend()
plt.show()


# #### Observation
# Axile Node range 0-5 Have maximum concentration of data
# 

# In[12]:


# plot data using seaborn library.
# Add color for each class
# Age and year of operation
sns.set_style('whitegrid')
sns.FacetGrid(df, hue = 'status', height = 4)    .map(plt.scatter, 'age','year')    .add_legend()
plt.show()


# #### Observation
# Plot is highly mixed up, but we can see that the mesority of operations happend between the age range 35 to 70 year.  

# In[13]:


# plot data using seaborn library.
# Add color for each class
# Age and year of operation
sns.set_style('whitegrid')
sns.FacetGrid(df, hue = 'status', height = 4)    .map(plt.scatter, 'year','nodes')    .add_legend()
plt.show()


# #### Observation
# 
# Highly mixed and overlapped plot, but we can observe that most of the patient have axil node between 0 to 10 have operation between year 60 to 67

# ### Pair plot
# 
# Using pair plot we can have have more visibility over different features and class, so that we can clearly visualize the data. We use pair plot when number of features are high. We can plot nD features in 2D.

# In[14]:


plt.close()
sns.set_style('whitegrid')
sns.pairplot(df, hue = 'status', height = 4, vars = ['age','year','nodes'])
plt.show()


# #### Observations
# 
# As per above plot almost all plots are overlapping, plot 3 and 7 are having better separation than others.
# But still we can not make any conclusion,will draw 1D scatter plot
# 
# 1) Axile Node range 0-5 Have maximum concentration of data
# 
# 2) Plot is highly mixed up, but we can see that the mesority of operations happend between the age range 35 to 70 year.
# 
# 3) Highly mixed and overlapped plot, but we can observe that most of the patient have axil node between 0 to 10 have operation between year 60 to 67

# ### Uni-varaite analysis
# 
# In Uni-variate analysis we deal with one variable. It's exploration of one variable. in haberman dataset, i will do Univariate analysis by ploting PDF, CDF, Box Plot, Voilin Plot.

# In[15]:


# What about 1-D scatter plot using just one feature?

label = ['Survived More than 5 Years','Survived Less than 5 Years']
long_surv = df.loc[df['status']==1]
sort_surv = df.loc[df['status']==2]

plt.plot(long_surv['nodes'], np.zeros_like(long_surv['nodes']),'o')
plt.plot(sort_surv['nodes'], np.zeros_like(sort_surv['nodes']),'o')

plt.title('1D Scattered Plot of Axilliry Lymph Nodes')
plt.xlabel('Nodes')
plt.ylabel('Distribution')
plt.legend(label)
plt.show()


# in 1D scatter plot both are overlapping each other. we can visualize by ploting both separately, as below

# In[16]:


plt.plot(sort_surv['nodes'], np.zeros_like(sort_surv['nodes']),'o')

plt.title('1D Scattered Plot of Axilliry Lymph Nodes More Survived')
plt.xlabel('Nodes')
plt.ylabel('Distribution')
plt.show()


# In[17]:


plt.plot(long_surv['nodes'], np.zeros_like(long_surv['nodes']),'o')

plt.title('1D Scattered Plot of Axilliry Lymph Less survived')
plt.xlabel('Nodes')
plt.ylabel('Distribution')
plt.show()


# #### Observation
# Both plots are have almost same distribution over line so, 1D scatter plot is also not useful for haberman dataset. Now i will go with PDF,CDF Un-ivariate analysis.
# 
# #### Probability Density Function (PDF)
# 
# PDF plots between the data and density of the data at particuler point.PDF forms high peak if more data is prasent at the point.
# 
# Plot PDF of age,year,nodes and classify the output.

# In[18]:


# Histogram of variable 'age'

sns.FacetGrid(df, hue = 'status', height = 4)   .map(sns.distplot, 'age')   .add_legend()
plt.show()


# In[19]:


# Histogram of veriable 'year' Year of operation
sns.FacetGrid(df, hue = 'status', height = 5)   .map(sns.distplot, 'year')   .add_legend()
plt.show()


# In[20]:


# Histogram of variable 'nodes' Possitive lymph Nodes

sns.FacetGrid(df, hue = 'status', height = 5)   .map(sns.distplot, 'nodes')   .add_legend()
plt.show()


# #### Observation
# Plots of age and year of operation (year), are almost completely overlapped. There are not much differantiation found from both plots, so we reject plot of age and year.
# 
# In third plot of nodes we have bit seperation, we can observe that people having less axilliry node survived more, still it's hard to classify but it's better than others so, we can accept it.
# 
# if nodes < 0                              => long survival
# if nodes > 0 and nodes < 3 (Aproximate)   => Chances of survival are high
# if nodes > 3                              => Short Survival
# 
# But still we can not major the percentage, accurate calculation. So we go for CDF now
# 
# #### Cumulative Distribution Function (CDF)
# 
# It's a cumulative reprasentation of data of PDF. For a continuous distribution it gives the area under the Probability Density Function. From CDF we will have idea of %age of points are less than n.

# In[21]:


# counts - the no of counts, Y - axis reprasents number of counts of data lies in a bin range.
# bin_edges - The bin edges, X - axis reprasents the bin edges.

survived_more = df.loc[df["status"] == 1]
survived_less = df.loc[df["status"] == 2]

legend_label = ['PDF Survived >5 Year', 'CDF Survived >5 Years','PDF Survived <5 Year', 'CDF Survived <5 Years']

counts, bin_edges = np.histogram(survived_more['age'], bins = 10, density = True) 
pdf = counts/sum(counts)
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:], pdf)
plt.plot(bin_edges[1:], cdf)
plt.legend(legend_label)

counts, bin_edges = np.histogram(survived_less['age'], bins = 10, density = True) 
pdf = counts/sum(counts)
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:], pdf)
plt.plot(bin_edges[1:], cdf)
plt.legend(legend_label)

plt.xlabel('Age')
plt.title('PDF and CDF for variable Age')
plt.show()


# #### Observation
# 
# 15% of patients survived who have age less than 35. Patients have age between 35 - 45, few are survived.

# In[22]:


counts, bin_edges = np.histogram(survived_more['year'], bins = 10, density = True) 
pdf = counts/sum(counts)
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:], pdf)
plt.plot(bin_edges[1:], cdf)
plt.legend(legend_label)

counts, bin_edges = np.histogram(survived_less['year'], bins = 10, density = True) 
pdf = counts/sum(counts)
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:], pdf)
plt.plot(bin_edges[1:], cdf)
plt.legend(legend_label)

plt.xlabel('Year')
plt.title('PDF and CDF for variable Year of Operation')
plt.show()


# #### Observation
# 
# Plot is almost overlapping, we can not give any observation based on the variable Year of Operation

# In[23]:


counts, bin_edges = np.histogram(survived_more['nodes'], bins = 10, density = True) 
pdf = counts/sum(counts)
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:], pdf)
plt.plot(bin_edges[1:], cdf)
plt.legend(legend_label)

counts, bin_edges = np.histogram(survived_less['nodes'], bins = 10, density = True) 
pdf = counts/sum(counts)
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:], pdf)
plt.plot(bin_edges[1:], cdf)
plt.legend(legend_label)

plt.xlabel('Nodes')
plt.title('PDF and CDF for variable Axilliry Lymph Nodes')
plt.show()


# #### Observation
# 
# Count of axilliry lymph node < 5 have aprox 85% survival rate. There also people survived having count of axilliry lymph node between 5-25. Axiliry Lymph node more than 30 have very less Survival rate almost 0. Patient have Axiliry Lymph node more than 46 have survival rate 0.
# 
# Higher Axilliry Lympth Node have very less survival rate

# #### Mean,Median, Percentile, Quantile, IQR, MAD

# Dataset might have the milion of the row, The Estimate of Location(Median, Percentile, Quantile, IQR, MAD) basic step is to explore data is finding the typical values for each feature where most of the data is located ie called called central tendency.

# #### Estimate of Location ( Mean, Trimmed Mean, Median)   for each feature  of the Habernman Dataset

# In[24]:


# Mean for age, year, nodes
long_surved = df.loc[df['status'] == 1]
short_surved = df.loc[df['status'] == 2]

print('Mean of Long survived patient\'s Age:')
print(np.mean(long_surved['age']))
print(np.mean(short_surved['age']))
print('Mean sort servived patient\'s Year:')
print(np.mean(long_surved['year']))
print(np.mean(short_surved['year']))
print('Mean of both log survived and short survived patient\'s lymph nodes:')
print(np.mean(long_surved['nodes']))
print(np.mean(short_surved['nodes']))

# Trimmed mean to avoide outliers
# Trim 20 points from both ends
print('\nTrimmed Mean')
print(np.mean(long_surved['age'][20:-20]))
print(np.mean(short_surved['age'][20:-20]))

# Evaluate median for each feature in dataset
print('###################')
print('\nMedian of age')
print(np.median(long_surved['age']))
print(np.median(short_surved['age']))
print('Median of year')
print(np.median(long_surved['year']))
print(np.median(short_surved['year']))
print('Median of nodes')
print(np.median(long_surved['nodes']))
print(np.median(short_surved['nodes']))


# #### Estimate of Variability (std-dev, Variance, Percentile, Quantile, IQR, MAD ( Mean Absolute Deviation))
# 
# Estimate of location is summarises a feature. But second dimention called variability, also lies at the heart of statistice. Variability calculated where data is tightly clustered.
# 

# In[25]:


# std-dev it's a differance between the Estimate of location (Mean, Median) and observed Data
# Standerd deviation of the Features of haberman dataset
# Variance = std-dev * std-dev
print('Standerd Deviation')
print(np.std(df))
print('\nVariance of Features')
print(np.var(df))
print('\nPercentiles')
# 10th Percentile of age
print('10th Percentile of age:')
print(np.percentile(long_surved['age'],10))
print(np.percentile(short_surved['age'],10))
print('20th Percentile of Year:')
print(np.percentile(long_surved['year'], 20))
print(np.percentile(short_surved['year'], 20))
print('90th Percentile of nodes:')
print(np.percentile(long_surved['nodes'],90))
print(np.percentile(short_surved['nodes'],90))
print('\n Quantiles of Features')
print('Quantiles of age:')
print(np.percentile(long_surved['age'],np.arange(0,100,25)))
print(np.percentile(short_surved['age'],np.arange(0,100,25)))
print('Quantiles of year:')
print(np.quantile(long_surved, 0.5, axis = 0))
print(np.quantile(short_surved, 0.5, axis = 0))
print('Quantiles of nodes:')
print(np.percentile(long_surved['nodes'],np.arange(0,100,25)))
print(np.percentile(short_surved['nodes'],np.arange(0,100,25)))

# Compute IQR
from scipy.stats import iqr
print('Inter Quartile Range (IQR)')
print('IQR Age')
print(iqr(long_surved['age']))
print(iqr(short_surved['age']))
print('IQR Year')
print(iqr(long_surved['year']))
print(iqr(short_surved['year']))
print('IQR Nodes')
print(iqr(long_surved['nodes']))
print(iqr(short_surved['nodes']))

# Compute MAD
from statsmodels import robust
print('Median Absolute Deviation:')
print(robust.mad(long_surved['age']))
print(robust.mad(short_surved['age']))


# ### Box Plots and Whiskers
# 
# Box plot and whiskers are another way to reprasent the one dimentional scatter plot. It reprasents quantiles (25,50,75,100 Percentiles) eadges of box are the 25 and 75 percentiles and dark line deviding box is 50th percentile. 

# In[26]:


# Box plot for patient's age 
sns.set(style = 'whitegrid')
sns.boxplot(x = 'status', y='age', data=df)
plt.show()


# In[27]:


# Box plot for patient's age 
sns.set(style = 'whitegrid')
sns.boxplot(x = 'status', y='year', data=df)
plt.show()


# In[28]:


# Box plot for patient's age 
sns.set(style = 'whitegrid')
sns.boxplot(x = 'status', y='nodes', data=df)
plt.show()


# #### Observations
# from above box plots we can observe that
# 
# 1) From Age plot, age less than 45 have few survived
# 
# 2) from year of operation plot we can not make any observation so we can reject this plot.
# 
# 3) From nodes plot as we can see that more axil nodes more likely patient to die. we can observe that more people survived having 0 to 5 (Aprox) positive Axilliry Lymph Nodes. 5 to 24 are the mejority who died
# 
# #### Violin Plot
# 
# we will draw violin plot for nodes only

# In[29]:


sns.violinplot(x='status', y='nodes',data=df,size=9)


# #### Observation
# 1) From nodes plot as we can see that more axil nodes more likely patient to die. we can observe that more people survived having 0 to 5 (Aprox) positive Axilliry Lymph Nodes

# ### Final Conclusion
# 
# 1) Axile Node range 0-5 Have maximum concentration of data, most of the people had the Positive Axiliry Node count between 0 to 5.
# 
# 2) Mejority of operations happened between 35 to 70 year age.
# 
# 3) Most of the patient have axillary node between 0 to 10 have operation between year 60 to 67
# 
# 
# 4) Count of axillary lymph node < 5 have approx 85% survival rate. There also people survived having count of axillary lymph node between 5-25. Axillary Lymph node more than 30 have very less Survival rate almost 0. Patient have Axillary Lymph node more than 46 have survival rate 0.
# 
# 5) Higher Axillary Lymph Node have very less survival rate
# 
# 6) Axillary node is more important feature, Age also important feature, Patient have age less than 35 and Axillary node grater or equals 1have more likely to survive.
# 
# 7) Patient who have Axillary nodes 0 are more likely to more survived
# 
# 8) Majority of patient have Axillary node between 1 to 24 are died
# 
# 9) Patients having who have 0 axillary nodes are more likely to survive
# 
# 10) Patient have age grater than 45 and Axillary node grater than 10 have more likely to die
# 
# 
