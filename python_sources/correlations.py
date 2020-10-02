#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


#reading excel files
import pandas as pd
Jan = pd.read_excel('../input/job-market-reports-sept19jan20/Jobs Sep 2019.xlsx')
Jan.head(2)


# In[ ]:


#get other files
Dec=pd.read_excel('../input/job-market-reports-sept19jan20/Jobs Dec 2019.xlsx')
Nov=pd.read_excel('../input/job-market-reports-sept19jan20/Jobs Nov 2019.xlsx')
Oct=pd.read_excel('../input/job-market-reports-sept19jan20/Jobs oct 2019.xlsx')
Sep=pd.read_excel('../input/job-market-reports-sept19jan20/Jobs Sep 2019.xlsx')
Aug=pd.read_excel('../input/job-market-reports-sept19jan20/Jobs August 2019.xlsx')


# In[ ]:


Aug.describe()


# In[ ]:


Aug.info()


# In[ ]:


Aug['Metro'].unique()


# In[ ]:


Aug['Dimension Type'].unique()


# In[ ]:


Aug['Dimension'].unique()


# In[ ]:


Aug['Measure'].unique()


# In[ ]:


Aug['Value'].unique()


# In[ ]:


Sep.head(2)


# In[ ]:


#Expand on October
Okt=Oct[['Metro', 'Dimension', 'Value']]
Okt.head(2)


# In[ ]:


import numpy as np
#label encode the metro column
from sklearn.preprocessing import LabelEncoder
Okt['Metro']= Okt['Metro'].astype('category')
Okt['Metro Catgs']=Okt['Metro'].cat.codes
Okt.head(3)


# In[ ]:


import seaborn as sns
sns.countplot(Okt['Metro Catgs'])


# In[ ]:


#group by categories to better understand the df data
G1= Okt.groupby('Metro Catgs')
G1.head().head()


# In[ ]:


Okt.astype(str)


# In[ ]:


Okt['Value'] = Okt['Value'].str.replace('$', '')


# In[ ]:


Okt['Value']=Okt['Value'].str.replace(',', '').astype(float)


# In[ ]:


Okt['Value']=Okt['Value'].fillna(0)


# In[ ]:


#Verify changes
Okt['Value'].head()


# In[ ]:


Okt['Metro Catgs'].astype(int)


# In[ ]:


#correlate
Okt.corr()

##there seem to be no correlation between these columns as they are.


# In[ ]:


#Visualize it
#Linear-graph for relationships
Okt.plot.scatter("Value", "Metro Catgs", color='green')


# In[ ]:


Okt.corr().plot.bar()


# In[ ]:


#The original columns have mixed information.
list(Nov['Dimension'].unique())


# In[ ]:


##extract job titles in the column
#First, find the location of the desired data
#Nov['Dimension'].head(50) ##job titles start at 42
Nov['Dimension'].tail(80) ##job titles end at 4837


# In[ ]:


N=Nov[42:4838] #to include the last row go 1 over
N.head(3)


# In[ ]:


#I want to examine the national data with job titles
National=N[(N['Metro']=='National')]
National =National[(National['Dimension Type']=='Job Title')]
National.head(3)


# In[ ]:


#Job opening titles with their base salary
Jobs=National[['Dimension','Value']]


# In[ ]:


#replace NaNs with 0s
Jobs['Value']=Jobs['Value'].fillna(0)


# In[ ]:


#clean data 
Jobs['Value'] = Jobs['Value'].str.replace('$', '')
Jobs['Value'] = Jobs['Value'].str.replace(',', '').astype(float)


# In[ ]:


Jobs.head(2)


# In[ ]:


#Label encode job titles
from sklearn.preprocessing import LabelEncoder
Jobs['Dimension']=Jobs['Dimension'].astype('category')
Jobs['Titles catgs']=Jobs['Dimension'].cat.codes
Jobs.head(2)


# There is no overlap among the Job Titles   
# 

# In[ ]:


Jobs.describe()


# In[ ]:


Jobs.median()


# According to the above, the **median salaries** offered is $56,194

# In[ ]:


Jobs.corr()
#There does not seem to be a correlation between the columns


# In[ ]:


#Job title-to-pay distribution
import matplotlib
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
plt.scatter(Jobs['Value'], Jobs['Titles catgs'], color='purple')
plt.show()


# In[ ]:


Jobs.corr().plot.bar()


# In[ ]:


#A df of monthly US Job Opening and Median pay across all dfs: Aug 2019 - Jan 2020
months = {'Jan20': Jan[0:2], 'Dec19': Dec[0:2], 'Nov19': Nov[0:2], 'Oct19': Oct[0:2], 'Sep19': Sep[0:2], 'Aug19': Aug[0:2]}

Monthly = pd.concat(months)
JobsNpay= Monthly.drop(['Metro', 'Dimension Type', 'Month', 'Dimension', 'YoY'], axis=1)
JobsNpay


# In[ ]:


#clean data 
JobsNpay['Value'] = JobsNpay['Value'].str.replace('$', '')
JobsNpay['Value'] = JobsNpay['Value'].str.replace(',', '').astype(float)


# In[ ]:


#Compare monthly US Job Opening and Median pay across all dfs: Aug 2019 - Jan 2020
Monthly1 = pd.concat([Jan[0:2], Dec[0:2], Nov[0:2], Oct[0:2], Sep[0:2], Aug[0:2]])
Monthly1


# In[ ]:


JobsNpay1= Monthly1.drop(['Metro', 'Dimension Type', 'Month', 'Dimension', 'YoY'], axis=1)
JobsNpay1


# In[ ]:


#clean data 
JobsNpay1['Value'] = JobsNpay1['Value'].str.replace('$', '')
JobsNpay1['Value'] = JobsNpay1['Value'].str.replace(',', '').astype(float)


# In[ ]:


sns.distplot(JobsNpay1['Value'], color='red', bins=3)
#Every other row represent something different.
#Also, the numbers could be normalize to 'even' the data


# In[ ]:


p=JobsNpay1.iloc[[0,2,4,6,8,10]]
p


# In[ ]:


USJobs=p.dropna()


# In[ ]:


USJobs.median()


# In[ ]:


USJobs.hist()


# In[ ]:


q=JobsNpay1.iloc[[1,3,5,7,9,11]]
Salary=q.dropna()
Salary


# In[ ]:


Salary.describe()


# In[ ]:


import matplotlib.pyplot as plotter 
# Months as label
pieLabels = 'Jan2020', 'Dec19', 'Nov19', 'Oct19', 'Sep19', 'Aug19' 

figureObject, axesObject = plotter.subplots()

# Draw the pie chart
axesObject.pie(USJobs['Value'],
        labels=pieLabels,
        autopct='%1.2f',
        startangle=90)

# Aspect ratio - equal means pie is a circle
axesObject.axis('equal') 

plotter.show()


# In[ ]:


Salary['Value'].corr


# In[ ]:


USJobs['Value'].corr


# **CORRELATE**  
# 
# Correlation coefficients evaluate how two variables are related to each other. The relationship could be linear, negatively linear, or monotonic.    
# In a monotonic relationship the variables may not change together at the same rate.   
# Corr() helps compute three different correlation coefficients between two variables using either the Pearson correlation method, Kendall Tau correlation method and Spearman correlation method.    
# 
# The correlation coefficients calculated using these methods vary from +1 to -1.   
# 
# Pearson correlation coefficient: the covariance of two variables divided by the product of their standard deviations. It evaluates the linear relationship between variables. Pearson correlation coefficient has a value between +1 and -1:   
# A result of 1 = a linear correlation between variable x and y. 0 = variables are not related. A -1 = there is an inverse (negative) correlation between variables.   
# Kendall Tau correlation coefficient: quantifies the discrepancy between the number of concordant and discordant pairs of two variables.      
# Spearman correlation coefficient:   
# a nonparametric evaluation that finds the strength and direction of the monotonic relationship between variables.   
# Best for when the data is not normally distributed or when the sample size is small (**under 30**).     
# 
# Source: https://pythontic.com/pandas/dataframe-computations/correlation

# **Relationships**   
# 1. define the relationships under evaluation   
# The data provides many opportunities for comparisons. Here, I analyze the relationship city-to-jobs relationship.   
# 2. define the data   
# Okt dataframe (the month in the six-month period with the most jobs.) and Dec (the month with the least jobs)

# In[ ]:


Okt['Dimension']= Okt['Dimension'].astype('category')


# In[ ]:


OctCityJobs=Okt[(Okt['Dimension']=='Metro Job Openings')]
OctCityJobs=OctCityJobs.drop(['Dimension','Metro Catgs'], axis=1)
OctCityJobs


# In[ ]:


Dec.head(2)


# In[ ]:


Dec1= Dec[['Metro','Dimension','Value']]
Dec1['Metro']=Dec1['Metro'].astype('category')
Dec1['Dimension']= Dec1['Dimension'].astype('category')
Dec1.head(2)


# In[ ]:


#clean data 
Dec1['Value'] = Dec1['Value'].str.replace('$', '')
Dec1['Value'] = Dec1['Value'].str.replace(',', '').astype(float)


# In[ ]:


DecCityJobs=Dec1[(Dec1['Dimension']=='Metro Job Openings')]
DecCityJobs=DecCityJobs.drop(['Dimension'], axis=1)
DecCityJobs.rename(columns={'Value': 'Dec Jobs'}, inplace=True)
DecCityJobs.head(2)


# In[ ]:


result = pd.concat([(DecCityJobs['Dec Jobs']), (OctCityJobs['Value'])], axis=1)
result


# In[ ]:


result.corr()


# In[ ]:


#Linear correlation 
x=result['Dec Jobs']
y=result['Value']
plt.scatter(x,y)
plt.show()


# In[ ]:


result.corr(method="spearman")


# In[ ]:


#heatmap
f, ax = plt.subplots(figsize =(7, 6)) 
sns.heatmap(result, ax = ax, cmap ="YlGnBu", linewidths = 0.1) 

