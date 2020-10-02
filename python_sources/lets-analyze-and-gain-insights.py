#!/usr/bin/env python
# coding: utf-8

# # Tasks
# # 1) COMPARE EACH CAR BRAND AVERAGE PRICE WITH ITS EACH MODEL PRICE
# # 2) Frequency of colors sold in each year
# # 3) What is the frequency of colors sold in each state
# 
# # 4) Overall Distribution of PRICE(2015-2020) 
# # 5) Checking distribution of PRICE each year respectively 
# # 6) Analyzing Title_status feature

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from plotly.offline import iplot
import plotly as py
import plotly.tools as tls
import cufflinks as cf
py.offline.init_notebook_mode(connected=True)
cf.go_offline()


# In[ ]:




df1=pd.read_csv('../input/usa-cers-dataset/USA_cars_datasets.csv')


# # DATA:

# In[ ]:


df1.head()


# In[ ]:


df1.drop(columns=['Unnamed: 0'],inplace=True,axis=1)


# # Columns:

# In[ ]:


df1.columns


# # Null Values

# In[ ]:


df1.isnull().sum()


# # Data_types

# In[ ]:


df1.dtypes


# # Shape

# In[ ]:


df1.shape


# # Describing

# In[ ]:


df1.describe()


# # Price cannot be zero
# 

# # FIRST FIND MEDIAN PRICE BY EACH BRAND And creating new column Average_price

# In[ ]:


df1['Average_price']=df1.groupby('brand')['price'].transform('median')


# In[ ]:


df1.head()


# In[ ]:


df2=df1[df1['price']==0]


# In[ ]:


l1_delete=df1[df1['price']==0].index


# In[ ]:


df1.drop(l1_delete,inplace=True) 


# In[ ]:


df2['price']=df2['Average_price']


# In[ ]:


df3=pd.concat([df1,df2],ignore_index=True)


# # Replacing datapoints of price==0 with median pirce of the particular brand

# In[ ]:


df3.head() 


# # How many data points we have for each year

# In[ ]:


df3.groupby('year')['brand'].count()


# # WE SEE THAT WE HAVE VERY LESS DATA FOR MANY YEARS, THIS CAN LEAD TO BIAS BEHAVIOUR AND WE WOULD NOT BE ABLE TO MAKE AN INFERENCE PROPERLY

# # LOT OF DATA PRESENT IN BETWEEN 2015 AND 2020
# 

# In[ ]:


fig1=plt.figure(figsize=(12,5))
ax1=fig1.add_axes([0,0,1,1])
sns.countplot(x='year',data=df3,ax=ax1) 


# In[ ]:


#the mean data points that we have for each year(Highy influenced by outliers)


# # Average data points in each year

# In[ ]:


df3.groupby('year')['brand'].count().mean() 


# # Median data points in each year

# In[ ]:


df3.groupby('year')['brand'].count().median() 


# # From the given data we cannot analyze the data points which have very less data, and it is not possible to generate so many data points for a particular year.
# 

# In[ ]:


df3['Occurence']=df3.groupby('year')['brand'].transform('count')


# In[ ]:


df4=df3[np.logical_or.reduce([df3['year']==2015,df3['year']==2016,df3['year']==2017,df3['year']==2018,df3['year']==2019,df3['year']==2020])]


# # Taking years into account which have considerable amount of data

# In[ ]:


df4.groupby('year')['brand'].count()


# # Data points in the year 2015-2020

# In[ ]:


fig1=plt.figure(figsize=(12,5))
ax1=fig1.add_axes([0,0,1,1])
sns.countplot(x='year',data=df4,ax=ax1)


# # WORKING DATA
# 

# In[ ]:


df4.head()


# In[ ]:


df5=df4.drop(columns=['Average_price','Occurence'],axis=1)


# # LETS COMPARE EACH CAR BRAND AVERAGE PRICE WITH ITS EACH MODEL PRICE This will help us to know which model is priced over the average price of the brand .....Lets visualize this CONDITION: Taking into account for brands which have greater than atleast 3 models

# # BRANDS THAT HAVE MORE THAN 3 MODELS:

# In[ ]:


e1=df4.groupby(['brand'])['model'].count().reset_index()
a1=list(e1[e1['model']>3]['brand'])
a1


# # PLEASE SCROLL THROUGH THE GRAPHS , AS THERE ARE MANY GRAPHS CREATED

# In[ ]:


for i in a1:
    df4[df4['brand']==i].groupby('model')[['price','Average_price']].mean().iplot(kind='spread',xTitle='Average_price for BRAND',yTitle='Price of each model under the brand',size=5,title='FOR'+' '+i)


# **-----------------------------------------------------------------------------------------------------------------------------------------**

# # Frequency of colors sold in each year, THIS ALSO HAS NO_COLOR
# 

# In[ ]:


q2=df4.groupby(['year','color'])['brand'].count().reset_index()


# In[ ]:


l1=[2015,2016,2017,2018,2019,2020]

for i in l1:
    q2[q2['year']==i][['color','brand']].iplot(x='color',y='brand',color='red',kind='bar',title='Colors sold in year'+ ' '+str(i))


# # Frequency of each color sold in Each STATE, WE ALSO HAVE NO_COLOR IN THIS, ANSWER BELOW

# In[ ]:


q3=df4.groupby(['state','color'])['brand'].count().reset_index()


# In[ ]:


q31=q3['state'].unique()


# In[ ]:


for i in q31:
    q3[q3['state']==i][['color','brand']].iplot(x='color',y='brand',color='green',kind='barh',title='Colors sold in'+' '+i)


# # OVERALL DISTRIBUTION OF PRICE(2015-2020) 

# In[ ]:


df4['price'].iplot(kind='hist',title="OVERALL DISTRIBUTION OF PRICE")


# # NOW LETS CHECK DISTRIBUTION OF PRICE FOR EACH YEAR SEPERATELY
# 

# In[ ]:


yr=df4['year'].unique()

g1=df4.groupby('year')['price']
for i,j in g1:
    j.iplot(kind='hist',title='Distribution of price in year'+' '+str(i))


# # NOW LETS ANALYZE TITLE_STATUS FEATURE

# In[ ]:


df4['title_status'].value_counts()


# In[ ]:


df_salvage=df4[df4['title_status']=='salvage insurance']


# In[ ]:


df_salvage['brand'].value_counts().sort_values().iplot(kind='barh',xTitle='Count',yTitle='Car_Brands',title='CAR BRANDS HAVING SALVAGE INSURANCE')


# # If you like my work , please do comment or hit the like button,,more EDA on the way  :)

# In[ ]:




