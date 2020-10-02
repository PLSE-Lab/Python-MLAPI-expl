#!/usr/bin/env python
# coding: utf-8

# 
# ## The purpose of this analysis to insight into the reason why people are leaving an organization. This uses the IBM HR data available.
#  This analysis is done in below steps:
# 1. Look at the IBM HR data structure 
# 2. Analyze the data based on Numeric and Categorical variables
# 3. Correlations in data

# In[ ]:


## Call libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib.ticker import NullFormatter  # for plotting muilple distributions with NullFormatter()


# In[ ]:


## Read and explore data
# Read file and explore dataset
hr_attrition= pd.read_csv("../input/WA_Fn-UseC_-HR-Employee-Attrition.csv", header=0)
hr_attrition.columns.values  # Get the column names


# # Explore the data based on various numerical variables

# In[ ]:


#########################################################################################
#  Most of the variables are numerical in the dataset, some columns are actually categorical in nature
hr_attrition.info()


# # Plot various various distributions based on the numerical variables

# In[ ]:


###########################################################################################
# Define a common module for drawing subplots
# This module will draw subplot based on the parameters 
# There will be mutiple subplots within the main plotting window
#  Defination of the parameters are-
#  var_Name - this is the variable name from the data file
#  tittle_Name - this is the Tittle name give for the plot
#  nrow & ncol - this is the number of subplots within the main plotting window
#  idx - position of subplot in the main plotting window
#  fz - the font size of Tittle in the main plotting window
##########################################################################################
def draw_subplots(var_Name,tittle_Name,nrow=1,ncol=1,idx=1,fz=10):
    ax = plt.subplot(nrow,ncol,idx)
    ax.set_title('Distribution of '+var_Name)
    plt.suptitle(tittle_Name, fontsize=fz)

numeric_columns = ['Age', 'MonthlyIncome', 'TotalWorkingYears']

fig,ax = plt.subplots(1,1, figsize=(10,10))
j=0  # reset the counter to plot 
title_Str="Plotting the density distribution of various numeric Features"

for i in numeric_columns:
    j +=1
    draw_subplots(i,title_Str,3,1,j,20) # create a 1x3 subplots for plotting distribution plots
    sns.distplot(hr_attrition[i])
    plt.xlabel('')


# # Observation from Density plots:
# # 1. Employee's age is normally distributed , majority lies between early 20 till 50, mean is around 35
# # 2. Majority of people is earning less than 10000 monthly, distibution is sckewed to the left
# # 3. Work experiance of the people is densly populuated till 15 years in a company, later it diminishes rapidly, distibution is sckewed to the left

# In[ ]:


numeric_columns = ['Age', 'DistanceFromHome', 'TotalWorkingYears',
                   'YearsAtCompany', 'Education','StockOptionLevel']


fig,ax = plt.subplots(1,1, figsize=(15,15))

j=0 # reset the counter to plot 
title_Str="Plotting the count distributions of various numeric Features"

for i in numeric_columns:
    j +=1
    draw_subplots(i,title_Str,3,2,j,20) # create a 3x2 subplots for plotting distribution plots
    sns.countplot(hr_attrition[i],hue=hr_attrition["Attrition"])
    plt.xlabel('')


# # Observations from count plots:
# # 1. Attrition is high at age as 28,29 & 30
# # 2. People with 1 year of experience has quited most
# # 3. The people stays closer to office and attrition percentage is low in comparision
# # 4. There are more employees with a bachelors degree followed by masters degree.The attrition levels are not stated across education levels
# # 5. stock option as 0 has more attrition in comparision to others stock options

# In[ ]:


numeric_columns = ['Age', 'MonthlyIncome', 'TotalWorkingYears',
                   'YearsAtCompany', 'YearsInCurrentRole','YearsWithCurrManager']

fig,ax = plt.subplots(1,1, figsize=(10,10))
j=0 # reset the counter to plot 
title_Str="Plotting the Boxplot distribution of various numeric Features"

for i in numeric_columns:
    j +=1
    draw_subplots(i,title_Str,3,2,j,20) # create a 3x2 subplots for plotting distribution plots
    sns.boxplot(hr_attrition.Attrition, hr_attrition[i])  # Note the change in bottom level
    plt.xlabel('')


# # Observations from above box plots:
# ## 1. This distribution shows attrition is higher for age between 30 & 40
# ## 2. Attrition is higher for low Monthly income
# ## 3. Less work experience people are more attrited

# In[ ]:


numeric_columns = ['Age','DailyRate','DistanceFromHome','Education','EmployeeCount','EmployeeNumber',
                   'EnvironmentSatisfaction','HourlyRate','JobInvolvement','JobLevel','JobSatisfaction',
                   'MonthlyIncome','MonthlyRate','NumCompaniesWorked','PercentSalaryHike','PerformanceRating',
                   'RelationshipSatisfaction','StandardHours','StockOptionLevel','TotalWorkingYears',
                   'TrainingTimesLastYear','WorkLifeBalance','YearsAtCompany','YearsInCurrentRole',
                   'YearsSinceLastPromotion','YearsWithCurrManager']

# Site :: http://seaborn.pydata.org/examples/many_pairwise_correlations.html
# Compute the correlation matrix
corr=hr_attrition[numeric_columns].corr()

fig,ax = plt.subplots(1,1, figsize=(20,20))
# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, center=0,yticklabels="auto",xticklabels=False,
            square=True, linewidths=.5,annot=True, fmt= '.1f')


# # Conclusion from Correlation :
# # Correlation is very high (more than 0.7) for below variables 
#  1. Monthly Income 
# 2.Performance Rating 
# 3.Total working years
# 4.Years At company
# 5.Years in Current Role
# 6. Years in Last Promotion 
# 7.Years with Current Manager 

# In[ ]:


# Look for job satisfaction at various job levels
sns.kdeplot(hr_attrition.JobSatisfaction, hr_attrition.JobInvolvement)
## Job involvement is mostly high with high job satisfaction


# In[ ]:


# Look for satisfaction with environment with education levels of the employee
sns.kdeplot(hr_attrition.Education, hr_attrition.EnvironmentSatisfaction)


# ## Employees with Bachelor and Master degree are mostly likely statisfied with the work environment

# In[ ]:


#     Factorplots are plots between one continuous, one categorical
#     conditioned by another one or two categorical variables
sns.factorplot(x =   'Department',
               y =   'Education',
               hue = 'Attrition',
               col=  'BusinessTravel',
               row= 'OverTime',   
               kind = 'box',
               data = hr_attrition)


# ## Observation from above plots:
# ## 1. People working in Sales department is most likely quit the company when he/she has to travel frequently
# ## 2. HR people who has to travel rarely is mostly likely to quit irrespective of overtime work
# ## 3. R&D people who are frequent or rare travellers is mostly likely to quit
# ## 4. Mostly employee has Bachelor & Master degrees in all departments and this is not contributing factor to any attrition

# In[ ]:


sns.factorplot(x =   'JobSatisfaction',
               y =   'MonthlyIncome',
               hue = 'Attrition',
               col=  'Education',   
               col_wrap=2,           # Wrap facet after two axes
               kind = 'box',
               data = hr_attrition)


# ## Observations:
# ## 1. People with less Monthly income are highly attrited irrespetive of Education level & job satisfaction 
# ## 2. People with Below Collage degree has least Monthly income among all categories
# ## 3. People with Bachelor and Master degree are attrited more than other categories in terms of average income, more people quited when job satisfaction level is low.

# In[ ]:


# Distribution of Job roles in pie chart
fig,ax = plt.subplots(1,1, figsize=(10,10))

# The slices will be ordered and plotted counter-clockwise.
labels = hr_attrition['JobRole'].unique()
jr_array = []

for i in range(len(labels)):
    jr_array.append(hr_attrition['JobRole'].value_counts()[i])

plt.pie(jr_array, labels=labels,
                autopct='%1.1f%%', shadow=True, startangle=90)
                # The default startangle is 0, which would start
                # everything is rotated counter-clockwise by 90 degrees,
                # so the plotting starts on the positive y-axis.

plt.title('Job Role Pie Chart', fontsize=20)
plt.show()


# ## Observation:
# ## Jobs held by the employee is max in Sales Executive, then R&D , then Laboratory Technician. 

# In[ ]:


sns.factorplot(x =   'WorkLifeBalance',
               y =   'JobRole',
               hue = 'Attrition',
               col=  'PerformanceRating',   
               col_wrap=2,           # Wrap facet after two axes
               kind = 'box',
               data = hr_attrition)


# ## Observations:
# ## 1. Sale Executive and Sale Representative are mostly attited people****
# ## 2. Sales Executives even with outstanding performance rating , are attrited more
# ## 4. Laboratory Technicians with outstanding performance rating and low work life balance are all attrited.

# # Conclusions:
# Using some basic Exploratory Data Analysis , we came to know various factors that might have stated the attrition. 
