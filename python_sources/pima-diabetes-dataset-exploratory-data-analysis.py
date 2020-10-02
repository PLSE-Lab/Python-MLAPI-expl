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

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set_style("whitegrid")


# **Import dataset**

# In[ ]:


data = pd.read_csv('../input/diabetes.csv') #importing dataset
data.info()


# **Dataset Desription**
# To get the central tendency of different fields of the dataset we can simply use describe() method. The central tendency includes mean, median, mode.
# 
# The describe function lists out: total count, mean, standard deviation, minimum value, First quartile(Q1), Median(Q2), Third Quartile(Q3), maximum value.
# 
# In the dataset we have 'Outcome' as the target variable. The value 1 or 0 which indicates whether or not the subject has diabetes.

# In[ ]:


data.describe()


# **Correlation between different fields**
# 
# To find the correlation of different fields we use corr() and plot it using heatmap() function in seaborn.
# 
# By looking at the corelation we can infer which fields may be used to predict the targe variable. We can even remove redundant fields if we find too much correlation amongst them, hence reducing dimensionality in the dataset which may make it easier for different prediction algorithms to process and provide effective results.
# 
# From the below heatmap we can see the correlation between the fields. Lighter areas suggest more correlation and similarly darker areas suggest very little or no correlation.

# In[ ]:


corr= data.corr()
sns.heatmap(corr,annot=True)


# From above heatmap we can infer that 'Glucose' and 'Outcome' have a correlation coefficient of 0.47. We also see a prominent correlation between 'Age' and 'Pregnancies' i.e. 0.54 which is self explanatory as the age of a woman increases the number of pregnancies she had would tend to increase.

# ****Visualizing data using different plots****
# 
# **Pairplot**
# 
# The pairplot is a method in seaborn that plots graphs of all the variables with respect to each other
# 
# The hue attribute can be used to differentiate between the classes of a particular feature. Here the hue used is the 'Outcome' of the dataset. Blue dots-> Outcome=0 Orange dots-> Outcome=1
# 
# The graph depicts increase in possibiity of Diabetes with increase in glucose levels.

# In[ ]:


sns.pairplot(data,hue='Outcome')


# **Distplot**
# 
# We can use distplot to see the ditribution of the data in form of histograms and we can even draw a line of probability distribution using kde attribute.

# In[ ]:


sns.distplot(data['Age'],kde=True,rug=True)


# In[ ]:


sns.distplot(data['Pregnancies'],kde=True,rug=True)


# In[ ]:


sns.distplot(data['Insulin'],kde=True,rug=True)


# **Jointplot**
# 
# Jointplots are used basically to see the relationship between the two field and it creates a simple bivariate graph.

# In[ ]:


sns.jointplot(data['Age'],data['Pregnancies'])


# In[ ]:


sns.jointplot(data['Glucose'],data['Outcome'],kind='reg')


# **Countplot**

# In[ ]:


sns.countplot(x='Pregnancies',hue='Outcome',data=data)


# In[ ]:



g = sns.FacetGrid(data,col='Outcome')
g.map(plt.hist,'Age')


# We now explore the data and look at the mean values of different fields based on whether the Outcome is 0 or 1. We need to disregard the entries for which the values of the fields(Insulin, Skin thickness, Glucose) is equal to zero as these entries may distort our observations.
# 
# Here insulin_0 represents the mean value of Insulin for the subjects for which Outcome is 0 and insulin_1 represents the mean value of Insulin for the subjects for which Outcome is 1.
# 
# We see considerable difference in both the values

# In[ ]:


insulin_0 = data[(data['Outcome']==0) & (data['Insulin']!=0)]['Insulin'].mean()
insulin_1 = data[(data['Outcome']==1) & (data['Insulin']!=0)]['Insulin'].mean()
print(insulin_0)
print(insulin_1)


# In[ ]:


glucose_0 = data[(data['Outcome']==0) & (data['Glucose']!=0)]['Glucose'].mean()
glucose_1 = data[(data['Outcome']==1) & (data['Glucose']!=0)]['Glucose'].mean()
print(glucose_0)
print(glucose_1)


# In[ ]:


skin_0 = data[(data['Outcome']==0) & (data['SkinThickness']!=0)]['SkinThickness'].mean()
skin_1 = data[(data['Outcome']==1) & (data['SkinThickness']!=0)]['SkinThickness'].mean()
print(skin_0)
print(skin_1)


# Interquartile Range and Boxplot
# From the above plots we see that there lies a considerable amount of outliers in the dataset. Many fields have values as zero such as BloodPressure, Insulin level, BMI etc. which cannot be zero as per normal human conditions are considered. Hence we need to deal with the outliers as they distort our observations for the dataset. For dealting with the outliers we can use BoxPlot which uses the concept of interquartile range(IQR).
# 
# IQR = Q3-Q1
# 
# Q3 = 75th Percentile Q1 = 25th Percentile

# In[ ]:


from scipy.stats import iqr


# The interquartile range for the Gluccose levels for Outcome 0 and 1 respectively are listed below. Again we disregard the 0 values for Glucose.
# 
# The interquartile value handles the outliers i.e. the extremely high values efficiently as compared to mean.

# In[ ]:


print(iqr(data[(data['Outcome']==0) & (data['Glucose']!=0)]['Glucose'],rng=(25,75)))

print(iqr(data[(data['Outcome']==1) & (data['Glucose']!=0)]['Glucose'],rng=(25,75)))


# In[ ]:


print(iqr(data[(data['Outcome']==0) & (data['Insulin']!=0)]['Insulin'],rng=(25,75)))

print(iqr(data[(data['Outcome']==1) & (data['Insulin']!=0)]['Insulin'],rng=(25,75)))


# **Boxplot**
# 
# We now look into the boxplots which is uses the idea of interquartile range.
# 
# The upper boundary of the boxplot represents the Q3 i.e. the 75th percentile, the lower boundary represents the Q1 i.e. 25th percentile.
# 
# The median is represented by a line inside the box.
# 
# We can also see some points outside the box. These values are either extremely low or high hence called outliers. The methods of calculating central tendency are affected by these outliers hence distorting the results. Unlike them, the IQR helps us to get a clearer picture with boxplots

# In[ ]:


plt.figure(figsize=(20,10))
sns.boxplot(data=data)


# In[ ]:


plt.figure(figsize=(20,10))
sns.boxplot(x='Outcome',y='Insulin',data=data)


# We now look into the boxplots with respects to the Outcome variable. This can be done by using the by= attribute. But first we will filter out the entries for which Insulin=0

# In[ ]:


data_new = data[data['Insulin']!=0]
plt.figure(figsize=(20,10))
sns.boxplot(x='Outcome',y='Insulin',data=data_new)


# In[ ]:


plt.figure(figsize=(20,10))
sns.boxplot(x='Outcome',y='Glucose',data=data)


# In[ ]:


data_new = data[data['Glucose']!=0]
plt.figure(figsize=(20,10))
sns.boxplot(x='Outcome',y='Glucose',data=data_new)


# So now we get a clearer picture. For Outcome=1 the median is close to 140 and for Outcome=0 the median is close to 105 though having some outliers.
# 
# **Filling the zero values**
# 
# Now in this data set we see a lot of zero values. What we can do is for better observations we can replace the zero values with the mean or median values of the columns. After that we look into the same results. So lets replace the zeroes with medians

# In[ ]:


data_new = pd.DataFrame()
data_new['Glucose'] = data['Glucose'].replace(0,data['Glucose'].median())
data_new['Insulin'] = data['Insulin'].replace(0,data['Insulin'].median())
data_new['BMI'] = data['BMI'].replace(0,data['BMI'].median())
data_new['SkinThickness'] = data['SkinThickness'].replace(0,data['SkinThickness'].median())
data_new['BloodPressure'] = data['BloodPressure'].replace(0,data['BloodPressure'].median())
data_new['Outcome'] = data['Outcome']


# In[ ]:


data_new.head()


# In[ ]:


plt.figure(figsize=(20,10))
sns.boxplot(x='Outcome',y='Glucose',data=data_new)


# In[ ]:


plt.figure(figsize=(20,10))
sns.boxplot(x='Outcome',y='Insulin',data=data_new)


# In[ ]:


plt.figure(figsize=(20,10))
sns.boxplot(x='Outcome',y='BMI',data=data_new)


# In[ ]:


print(iqr(data_new[(data_new['Outcome']==0)]['Glucose'],rng=(25,75)))

print(iqr(data_new[(data_new['Outcome']==1)]['Glucose'],rng=(25,75)))


# In[ ]:


print(iqr(data_new[(data_new['Outcome']==0)]['Insulin'],rng=(25,75)))

print(iqr(data_new[(data_new['Outcome']==1)]['Insulin'],rng=(25,75)))


# **Violin Plots**
# 
# A violin plot plays a similar role as a boxplot. It shows the distribution of data across several levels of one or more categorical variables such that those distributions can be compared. In box plot all of the plot components correspond to actual datapoints,whereas the violin plot features a kernel density estimation of the underlying distribution.

# In[ ]:


plt.figure(figsize=(20, 10))
sns.violinplot(x='Outcome',y='Glucose',data=data_new,inner='quartile')


# **The End**

# In[ ]:





# In[ ]:




