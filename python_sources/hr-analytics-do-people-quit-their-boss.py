#!/usr/bin/env python
# coding: utf-8

# We will try to analyse visually the trends in how and why are quitting their jobs. Is it because of monthly income level, or distance from home or performance ratings? As an initial step, we evaluate data using visual analytics. We will use seaborn (majorly) and matplot library. 

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns 
import warnings 
import os
import matplotlib.pyplot as plt


# In[ ]:


warnings.filterwarnings("ignore") 
hr = pd.read_csv("../input/WA_Fn-UseC_-HR-Employee-Attrition.csv", header=0) #Read the dataset
hr.head() #Display first few rows of the data


# ***Density Plots of Age, Working Years, Years at Present Organization and Years in Current Role***

# In[ ]:


# Reference for adding title for each sub plots: https://gist.github.com/dyerrington/dac39db54161dafc9359995924413a12
fig,ax = plt.subplots(2,2, figsize=(10,10))               # 'ax' has references to all the four axes
plt.suptitle("Understanding the distribution of various factors", fontsize=20)
sns.distplot(hr['Age'], ax = ax[0,0])  # Plot on 1st axes
ax[0][0].set_title('Distribution of Age')
sns.distplot(hr['TotalWorkingYears'], ax = ax[0,1])  # Plot on IInd axes
ax[0][1].set_title('Distribution of Total Working Years')
sns.distplot(hr['YearsAtCompany'], ax = ax[1,0])  # Plot on IIIrd axes
ax[1][0].set_title('Distribution of Years at company')
sns.distplot(hr['YearsInCurrentRole'], ax = ax[1,1])  # Plot on IV the axes
ax[1][1].set_title('Distribution of Years in Current Role')
plt.show()                                                # Show all of them now


# From the plots, we find that majority of the employees are in the age group of 35-40 years and have a total experience of around 8-10 years. They have typically spent around 2-4 years at the current organization and have spent 2-3 years in the current role they are in.

# ***Count Plot***

# In[ ]:


sns.countplot(hr['Attrition'])
plt.show()


# Above plot shows that around 250 people have left the organization while the organization has been 
# able to retain around 1200 people.

# ***Bar Plot***

# In[ ]:


#Bar plot of Job Satisfaction with respect to distance from home according to gender
#Estimator used is median
from numpy import median
sns.barplot(x='Attrition', y='MonthlyIncome', hue= 'Gender',data=hr, estimator=median)
plt.show()


# We can conclude that people with less monthlyincome (around 3000 units) are likely to leave the organization than those with better income (around 5000 units). 

# ***Box Plot***

# In[ ]:


f,ax = plt.subplots(figsize=(15,10))
sns.boxplot(x='JobSatisfaction', y='MonthlyIncome', data=hr, hue='JobLevel',palette='Set3')
plt.legend(loc='best')
plt.show()


# We can clearly observe the difference in monthly income across different job levels. The difference in the monthly income is observed by the median value for different job satisfaction level. 

# ***Violin Plot***

# In[ ]:


sns.violinplot(x="Attrition", y="YearsAtCompany", hue="Gender", data=hr, palette="muted", split=True,
               inner="quartile")
plt.show()


# Violin plots are similar to box plots but they have the capability to explain the data better. The distribution of data is measured by the width of the violin plot. 
# Here, we have plotted the number of years spent in an organization based on gender. The middle dashed line shows the median. The lines above and below the median show the interquartile range. The denser part shows the maximum population falls under that range and thinner part shows the lesser population. For details, please refer https://blog.modeanalytics.com/violin-plot-examples/ 

# ***Joint Plot (Scatter Plot)***

# In[ ]:


## Joint scatter plot
sns.jointplot(hr.Age,hr.MonthlyIncome, kind = "scatter")   
plt.show()


# Scatter plot shows the relationship between Age and Monthly Income. We can find a linear relationship. Further, the density plot above shows the distribution of age while density plot in the right shows the distribution of the monthly income. 

# ***Factor Plot and Facet Grid***

# In[ ]:


hr['age_group'] = pd.cut(hr['Age'], 3, labels=['Young', 'Middle', 'Senior']) #Slicing the continuous data into various groups
sns.factorplot(x =   'Attrition',     # Categorical
               y =   'MonthlyIncome',      # Continuous
               hue = 'JobLevel',    # Categorical
               col = 'age_group',
               col_wrap=2,           # Wrap facet after two axes
               kind = 'box',
               data = hr)
plt.show()


# Above factor plot shows that monthly income plays an important role in retaining the employees in an organization. It can be observed across job levels and different age groups.

# In[ ]:


g = sns.FacetGrid(hr, col="JobSatisfaction", row="Gender")
g.map(sns.kdeplot, "MonthlyIncome", "YearsInCurrentRole")
plt.show()


# We have created kernel density estimation plot. It displays the density distribution of two continuous variables (namely, Monthly income and years in current role). We have created facets according to different job satisfaction levels and gender.

# ***Pair Plot***

# In[ ]:


data1 = ['Attrition','Age','MonthlyIncome','DistanceFromHome']
sns.pairplot(hr[data1], kind="reg", diag_kind = "kde" , hue = 'Attrition' )
plt.show()


# Pairwise plots between continuous variables show the relationship between them. For example. observing the relationship between Age and Monthly Income, we can find that with age, monthly income has increased but the increase is not similar for both groups (Attrition and Retention). 

# In[ ]:


data2 = ['Gender', 'HourlyRate','DailyRate','MonthlyRate','PercentSalaryHike']
sns.pairplot(hr[data2], kind="reg", diag_kind = "kde" , hue = 'Gender' )
plt.show()


# The above plot does not convey much of any relationship between variables across gender. This shows that hourly rate, daily rate, monthly rate and percent salary hike is same for both female and male employees.

# ***Correlation (Heat) Plot***

# In[ ]:


#Plot a correlation map for all numeric variables
f,ax = plt.subplots(figsize=(18, 18))
sns.heatmap(hr.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
plt.show()


# Two variables are said to be highly correlation when they have a value of 0.7 or greater. The correlation plot between all continuous variables indicate that years at company and year with current manager, years in current role and years with current manager, monthly income and total working years, age and total working years, percent salary hike and performance rating are  highly correlated. 
# 
# For more details on advanced plotting features in Seaborn, refer: https://blog.insightdatascience.com/data-visualization-in-python-advanced-functionality-in-seaborn-20d217f1a9a6 
