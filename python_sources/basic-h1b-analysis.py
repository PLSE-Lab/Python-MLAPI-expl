#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import pandas as pd


# In[ ]:


#Importing csv file 
d=pd.read_csv('../input/h1b_kaggle.csv')


# In[ ]:


#Doing some preliminary check on the data:


# In[ ]:


d.head()


# In[ ]:


#Dropping columns that wont be needed throughout the analysis:


# In[ ]:


d.columns


# In[ ]:


d=d.drop(['Unnamed: 0','lon','lat'],axis=1)


# In[ ]:


d[1:3]


# In[ ]:


#I will be renaming the columns to make it easier to call or perform analysis with:


# In[ ]:


d.columns=['case','employer','soc','jobTitle','fullTime','wage','year','location']


# In[ ]:


d.columns


# In[ ]:


#Checking dataframe for null values:


# In[ ]:


d.isna()


# In[ ]:


#There appears to be null values at the end of the dataframe/
#and we will be getting rid of all these rows where null is true
d=d.dropna(how='any')


# In[ ]:


d.isnull()


# In[ ]:


#Alright, now we can move on to see some summary statistics of our dataset:


# In[ ]:


d.describe()


# In[ ]:


#well something is wrong over here, let's try to fix this 


# In[ ]:


d.info()


# In[ ]:


#we need to convert year into a category dtype, which will solve what was wrong:
d.year=d['year'].astype('category')


# In[ ]:


d.info()


# In[ ]:


stat=d.describe().transpose()


# In[ ]:


stat.round(2)


# ---
# ---
# ---
# ---

# In[ ]:


#This is new section and this section will focus on Analysis and Insights:


# In[ ]:


import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
plt.show()
import warnings
warnings.filterwarnings('ignore')


# In[ ]:


#Let's start by looking at the dataframe:
d.head()


# ---
# ---

# In[ ]:


# Q. Which employers file the most petitions?
#we will be looking at the top 10 employers: 


# In[ ]:


top10=d.employer.value_counts().head(10)


# In[ ]:


#We now have a series and it goes on to show the top 10 employers that file the most petitions:
top10


# In[ ]:


top10.plot.barh(figsize=(15,5),title='Top 10 Employers Petitioning',xlim=[0,140000],xticks=np.arange(0,140000,10000))
plt.xlabel('Number of Petitions from 2011-2016')
plt.ylabel('Employers')
plt.style='dark'


# This shows the top 10 employers who filled for the most petitions from the year 2011 to 2016 with Microsoft being 7th. The largest petitioners are mostly Indian based companies.

# In[ ]:


# Q. What is the percentage of petitions filed by the top 10 compared to the total number of petitions?


# In[ ]:


#This calculates the total number of petitions:
d.employer.count()


# In[ ]:


#This will calculate the total number of petitions filed by the top 10 employers:
top10.sum()


# In[ ]:


#Now we can compute the statistics:
ans1=((top10.sum())/(d.employer.count()))*100


# In[ ]:



print('Percentage of petition filed by top 10 = ', ans1.round(2),'%')


# The top 10 petitioners formed about only 14% of the 3million petitions filled in the five year period.

# ---
# ---

# In[ ]:


# Q. What is the statistics behind the Prevailing wage of Financial Analyst over the years?


# In[ ]:


#Creating a new dataframe to make matters easier:
d1=d[['soc','wage','year']]


# In[ ]:


d1.head(4)


# In[ ]:


FinAn=d1[d1.soc=='Financial Analysts']


# In[ ]:


FinAn.count()


# In[ ]:


FinAn.boxplot(column='wage',by='year',figsize=(15,12),fontsize=10)
plt.ylim(0,250000)


# In[ ]:


#In the scatterplot above, it appears that there the years 2015-16 do not have data


# In[ ]:


#Previously going through the data, I found that there is a number of entries for financial analyst with the letter capitalized


# In[ ]:


FinAn1=d1[d1.soc=='FINANCIAL ANALYSTS']


# In[ ]:


FinAn1.boxplot(column='wage',by='year',figsize=(15,12),fontsize=10)
plt.ylim(0,250000)


# In[ ]:


#Hence we see the data for the two missing years, now we can connect both the dataframes 


# In[ ]:


frames=[FinAn,FinAn1]


# In[ ]:


FinAn2=pd.concat(frames)


# In[ ]:


FinAn2.boxplot(column='wage',by='year',figsize=(15,12),fontsize=10)
plt.ylim(0,250000)


# In[ ]:


FinAn2.head(2)


# In[ ]:


#This will compute the median wage for the years in the dataset
stat2=FinAn2.groupby('year')['wage']


# In[ ]:


stat2.median()


# In[ ]:


stat2.describe().round(2)


# From the boxplot analysis, we can see the mean wage in all those years were all over a 100,000, which is quite tempting
# till we look at the box plot which shows us quite a lot of outliers, and this is in fact has affected the mean over the years. To get a better picture, the median over the years are between 64,000 and 70,000, which gives us a better picture 
# of what the prevailing wage for Financial Analysts is for most companies. I computed this to have a better understanding of the role which I once wanted to pursue before changing my mind to moving for Data Science. I have computed the same statistics for Data Scientists roles which we will be looking over at the end of this project.

# In[ ]:





# ---
# ---
# ---
# ---

# In[ ]:


# Q. What positions were usually sought after by foreign workers in those years,combined? 
# for this question, I'll look into the top 10 roles/positions:


# In[ ]:


JobPos=d['jobTitle'].value_counts().head(20)


# In[ ]:


JobPos


# In[ ]:


JobPos.plot.barh(figsize=(20,10),title='Top 20 Positions Sought for Petition',xticks=np.arange(0,250000,10000),grid=True)
plt.xlabel('Number of Petitions')

plt.ylabel('Job Positions')
plt.style='dark'
plt.show()


# This analysis shows us the most sought after roles over the years and looking at it paints the picture that most foreign workers coming coming to the US are usually people in the technology sector as most of these roles belong there. 

# ---
# ---
# ---

# In[ ]:


# Q. What are the statistics for data scientists in all these years


# In[ ]:


#Since the data is large, I will be using this function to look for data scientists roles within the dataframe.
dsci=d[d.jobTitle.str.contains('DATA SCIENTIST')]


# In[ ]:


dsci.head()


# In[ ]:


dsci['case'].unique()


# In[ ]:





# In[ ]:


#I want to filter out CERTIFIED WITHDRAWN AND WITHDRAM because these were cases were either the petioner or the /
#worker opted out and I want to leave them out and only go with CERTIFIED AND DENIED.
dsci.case=dsci[(dsci['case']=='CERTIFIED')|(dsci['case']=='DENIED')]


# In[ ]:


dsci11=dsci.groupby('year')['case'].count()


# In[ ]:


dsci11.plot(figsize=(10,6),grid=True,title='Number of Petitions filled for Data Scientist Roles')
plt.xlabel('Years from 2011-2016')
plt.ylabel('Number of Petitions Filled')

plt.show()
#I'm not able to figure out why I cant insert the years in the x-axis.


# This is quite remarkable to see how the number of petitions for data scientist roles increased over these years from less than 200 to over a 1000 within just five years. This also goes on to show the emerging new role of data scientists

# ---

# In[ ]:


# Q.Let us look at the wage statistics for data scientists


# In[ ]:


dsci.boxplot(column='wage',by='year',figsize=(20,15),fontsize=15)
plt.ylim(0,250000)
plt.xlabel('Year')
plt.ylabel('Wage')


# In[ ]:





# In[ ]:


dsci.head(2)


# In[ ]:


statDS=dsci.groupby('year')['wage']


# In[ ]:


statDS.median()


# It seems like the median wage for most data scientists were inbwtween 85,000 and 97,000 which is quite higher than that of Financial Analysts which is quite cool. Not much outliers can be observed from this dataset which means the average prevailing wage can also be used.

# In[ ]:


statDS.describe()


# In[ ]:


statDS.mean().round(2)


# To conclude, this dataset has helped me in two ways:
#     1. I got to exercise what I learnt from the book 'Python for Data Analysis' written by Wes McKinney. 
#      2. I got to have basic insights into the H1B work visa petitions which I will be applying for once I graduate in the next two years.
# 
# I got to learn who the top 10 employers were, and the statistics behind Financial Analysts and Data Scientists. Much more things can be done through this dataset which I believe will be exercising over the course of this summer, to get more meaningful and organised insights.
# 

# In[ ]:




