#!/usr/bin/env python
# coding: utf-8

# # Project : Accidental Drug Related Deaths in Conneticut

# Data Set Description : 
# A listing of each accidental death associated with drug overdose in Connecticut from 2012 to 2018. A "Y" value under 
# the different substance columns indicates that particular substance was detected.
# Data are derived from an investigation by the Office of the Chief Medical Examiner which includes the toxicity report, death certificate, as well as a scene investigation.
# 
# Project proposal includes data source location
# 
# Data Source Loacation : 
# 
# During Project 1 I came to know about Data.Gov website. I am using asame website for project 2. 
# 1. Search for Data.gov in google
# 2. Click on Data Tab on the top 
# 3. From the results select 'Accidental Drug Related Deaths 2012-2018  177 recent views' which will navigate to this link       https://catalog.data.gov/dataset/accidental-drug-related-deaths-january-2012-sept-2015.
# 4. Then click on CSV download to download the data set. 
# 
# Note : This Data Set has mutiple formats but I have taken csv format.
# 
# 
# Describe how data source was discovered ? 
# I googled 'Data.Gov' and then I clicked on the Data tab. From the results I selected https://catalog.data.gov/dataset/accidental-drug-related-deaths-january-2012-sept-2015.
# I discovered this website during Project 1. 
# 
# 
# EDA and Accessabiltiy:
# Data is legally acessable and free to download but you might have to create account.
# Documented EDA is not available. There is no visualization present. Only, Description of the data set is given in the website and in Kaggle.
# 
# 
# Project proposal relevant to Data 601 objectives ? 
# This Data set is totally related to the objectives we learnt in DATA 601. 
# There is so much of scope for data cleaning, charecterization and visualization. 
# Also, lots of hypothesis can be concluded by see this data Set. 
# 
# Does student understand the stories in the data?
# I don't have any specific experience related to this csv file. But this csv delas with numbers and percentages of deaths. 
# I have similar kind of experience, where I used to work on policy premiums data sets. 
# 
# Data Size : 
# Columns: 41
# Rows : 5106
# Size : 1765 Kb or 1.7 Mb
# 
# Questions to be investigated :
# Which drug is causing the more deaths ? 
# Which gender has most of the deaths ? 
# 
# Hypothesis 1 : 
# Most of the students are prone to drugs. So, after visulaization : teengaers(14 to 22) might have more deaths related to drugs when compared to normal age( > 22) groups. 
# 
# Hypothesis 2 : 
# Since this data is related to Connecticut and from surveys I know that the white race is about 81.6 percent, Most of the deaths related to drugs will be from white race when 
# compared to other races. 

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


# importing all the libraries required.
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import lag_plot
import random


# In[ ]:


data = pd.read_csv('/kaggle/input/Accidental_Drug_Related_Deaths_2012-2018.csv')


# In[ ]:


# Finding out the total Nan value present in each column.
data.isnull().sum()


# In[ ]:


# The data frame is modified. 
# notnull() removes the Nan values present in Date column
data = data[pd.notnull(data['Date'])]


# In[ ]:


# The missing values are filled with random integers. 
data['Age'].fillna(random.randint(16,56), inplace = True)


# In[ ]:


# Making sure that Age is always an integer. 
data['Age'] = data['Age'].astype('int')


# In[ ]:


# Making sure that Age column has no null values
data['Age'].isnull().sum()


# In[ ]:


# drugNames is list which stores the drug column names of the data set. 
drugNames = list(data.columns[20:37])


# In[ ]:


# This for loop replaces the Nan values with N for all the drug columns. 
for drugColumn in drugNames:
    data[drugColumn].fillna('N', inplace=True)


# In[ ]:


data['OtherSignifican'].fillna('Not Specified', inplace = True)


# In[ ]:


# This list holds the names of columns which are to be dropped
dropColumns = list(data.columns[6:18])


# In[ ]:


# This lopp drops the columns 
for columnName in dropColumns:
    data = data.drop(columnName,axis = 1)


# In[ ]:


# Data Frame is group by Gender column
genderGroup = data.groupby('Sex')


# In[ ]:


# Groupby variable holds the data frame by grouping Sex Column. Getting the count of it will answer the above question. 
# A bar graph is plotted to show the results. 
_ = genderGroup['Sex'].count().plot(kind = 'bar')
_ = plt.xlabel('Gender')
_ = plt.ylabel('Number of Deaths')
_ = plt.title('Total number of deaths categorized by Gender')


# In[ ]:


# Declaring an empty list
totalValue = []


# In[ ]:


# This loop will pick the drug column name and iterates through dat frame. 
# The summation counter increments if the value of each cell is not N. 
for drug in drugNames:
    summation = 0
    for index , value in data.iterrows():
        if(data.at[index,drug] != 'N'):
            summation = summation + 1
    totalValue.append(summation)


# In[ ]:


# Plotting the results on a scatter plot. 
_=plt.figure(figsize=(30, 10))
_=plt.scatter(drugNames,totalValue)
_ = plt.xlabel('Type of Drug',fontsize = 20)
_ = plt.ylabel('Total Number of Deaths',fontsize = 20)
_ = plt.title('Total number of deaths categorized by type of Drug', fontsize = 30)
_ =plt.grid(True)


# 

# ### Hypothesis 1 : True
# 
# Since this data is related to Connecticut and from surveys I know that the white race is about 81.6 percent, Most of the deaths related to drugs will be from white race when 
# compared to other races. 
# 
# Programmatic Approach : 
# 1. Race column of the data set is grouped together. 
# 2. Count os Race column from the grouped variables is plotted on a bar graph.

# In[ ]:


# Grouping the Race column
raceGroup = data.groupby('Race')


# In[ ]:


# This code displays the total number of deaths caused by drugs categorized by Race in connecticut
raceGroup['Race'].count()


# In[ ]:


# Plotting the count of number of deaths from the grouped variable. 
_= plt.Figure(figsize = (50,30))
_= raceGroup['Race'].count().plot(kind = 'bar')
_ = plt.xlabel('Race', fontsize = 15)
_ = plt.ylabel('Total number of Deaths',fontsize = 15)
_ = plt.title('Total number of deaths categorized by Race',fontsize = 15)


# ### Hypothesis 2 : False
# 
# Most of the students are prone to drugs. So, after visulaization : teengaers(14 to 22) might have more deaths related to drugs when compared to normal age( > 22) groups. 
# 
# Programmatic Approach : 
# 1. A new column in the data set is created by the name Teenager. 
# 2. This column is either false or true. 
# 3. A lamda expression is applied on this column which checks for the Age and applies True or False if the X<= 22 condition is met.
# 3. A for loop is created to count the total number of deaths in Teenage Group and the  Other Groups 
# 4. The results are plotted on a pie chart.

# In[ ]:


# Creating a new column in the data set by the name Teenager. 
# lambda expression is applied which will declare the cell as True if the Age value is <= 22
data['Teenager'] = data['Age'].apply(lambda x: x <= 22)


# In[ ]:


# Displaying the data frame after adding Teenager Column
# Observe the last column name of the data set
data.head()


# In[ ]:


# This loop iterates through the data set and checks for Teenager Column. 
# Other age group Counter is incremented if Teenager is matcher with False and vice versa.
Teenager  = 0 #Teenage Counter
NotATeenager = 0 # Other Age Group Counter
for index , value in data.iterrows():
    if(data.at[index,'Teenager'] == False):
        NotATeenager = NotATeenager + 1
    else:
        Teenager = Teenager + 1


# In[ ]:


# PLotting a pie graph
labels = 'Teenage Group(<=22)', 'Other Age Groups(>22)'
sizes = [Teenager, NotATeenager]
colors = ['lightcoral', 'lightskyblue']
explode = (0.1, 0)  # explode 1st slice


plt.pie(sizes, explode=explode, labels=labels, colors=colors,
autopct='%1.1f%%', shadow=True, startangle=140)
plt.title('Percentage of Deaths by age groups')
plt.axis('equal')
plt.show()


# The above pie chart depicts that only 4.2 percent of the deaths are in teenage group. This means that most of the people are prone to drungs with age greater than 23.
