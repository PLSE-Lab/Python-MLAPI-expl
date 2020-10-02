#!/usr/bin/env python
# coding: utf-8

# Hi,
# This is my first kaggle try on a data set.
# I am a beginner in data science and I am trying to improve my skills.
# I have implemented some basic python analysis tools also I tried to use some structures such as lists, dictionaries and also loops to see how they work.
# This kernel just for beginners, I am also a beginner trying new things. :)
# Have fun!

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns # this is another visualization tool
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

# Any results you write to the current directory are saved as output.


# In[ ]:


data_ibm = pd.read_csv('../input/WA_Fn-UseC_-HR-Employee-Attrition.csv')


#  **First things we have to do:**
# First we must understand what we are going to analyze 

# In[ ]:


data_ibm.info() 


# In[ ]:


data_ibm.columns # to see which columns we have


# In[ ]:


data_ibm.head() # to take a look inside of the data frame


# In[ ]:


data_ibm.corr() # to see the correlations between parameters inside of the data set


# In[ ]:


# to understand the correlations better, we can use correlation map
f,ax = plt.subplots(figsize=(10,10))
# use sns seabon library for virtualization
sns.heatmap(data_ibm.corr(),annot=True,linewidth=5,fmt = '.1f', ax=ax)
plt.show()


# **NOW LET'S USE SOME PLOTS**

# In[ ]:


# LINE PLOT: we prefer to use line plot if the x axis must be time
# color = color, label = label, linewidth = width of the line, alpha = opacity, grid=grid, line style = style of the line
# here we took the first 102 rows from the data frame and we created a small data frame named "small_data". 
# And we plotted Monthly and Daily rates from the small data frame
small_data = data_ibm.head(102)
small_data.MonthlyRate.plot(kind='line',color='g',label='Monthly Rate',linewidth=1,alpha=1,grid=True,linestyle = ':')
small_data.DailyRate.plot(kind='line',color='r',label = 'Daily Rate', linewidth=1,alpha=1,grid=True,linestyle='-.')
plt.legend(loc='upper right')
plt.xlabel('x axis')
plt.ylabel('y axis')
plt.title('Line plot of Monthly and Daily Rates')
plt.show()


# In[ ]:


# SCATTER PLOT: we prefer to use scatter plot, when there is a correlation between the parameters inside of the data frame
# In this example, we analyzed teh correlation between Job Level and Monthly Income
# You can see the correlations easily by using ".corr()" method. If the correlation value is 1= it means we have correlation, 
# if the value is 0, we have no correlation. And if the value -1, it means we have negative correlation
data_ibm.plot(kind='scatter', x = 'MonthlyIncome', y = 'JobLevel',alpha=0.5,color='red')
plt.xlabel("Monthly Income")
plt.ylabel("Job Level")
plt.title("Correlation between Job Level and Monthly Income")
plt.show()


# In[ ]:


# Also the following code block can draw exactly the same scatter plot
data_ibm.columns
plt.scatter(data_ibm.MonthlyIncome,data_ibm.JobLevel,color="red",alpha=0.5)
plt.show()
# As we can see from the plots, there is a strong possitive correlation between Job level and Monthly Income 
# in the IBM HR Analytics Employee Attrition and Performance data set


# In[ ]:


# HISTOGRAM: We prefer to use histograms when we want to see distributions of numerical data in the data set
# bins = number of bars in the figure
data_ibm.Age.plot(kind='hist',bins = 50, figsize = (10,10))
plt.show()
# Here we can see the Age frequencies and its distributions in the data set


#  

# In[ ]:


# to clean a plot
# first create and then clean the plot,
data_ibm.JobLevel.plot(kind='hist',bins=50)
plt.clf()


# **NOW LET'S WORK WITH DICTIONARIES**
# You can think dictionaries as small databases. Each one has keys and values.
# Do not forget that, working with dictionaries is faster than working with lists.
# 

# In[ ]:


# In this part we do not use the data set
# create a dictionary and view its keys and values
dictionary = {'worker1': 'woman', 'worker2': 'man'}
print(dictionary.keys())
print(dictionary.values())


# In[ ]:


# If you want to update existing dictionary info:
dictionary['worker1'] = "man"
dictionary['worker2'] = "woman"
# If you want to add info to dictionary
dictionary['worker3'] = "woman"
print(dictionary)
print(dictionary.keys())
print(dictionary.values())


# In[ ]:


# to clear dictionary
dictionary.clear()
print(dictionary)
# when you clear the dictionary, it will remain there as an empty dictionary


# In[ ]:


# to delete dictionary
del dictionary
# NOTE: After you delete a dictionary, you cannot access that dictionary anymore. For example you cannot use 
# print function to display it because you do not have that dictionary anymore.


# **NOW LETS WORK ON PANDAS A LITTLE BIT**
# In this part we are going to use our data set again.

# In[ ]:


# import pandas library to be able to use 
# take data set and create a data frame named "data_ibm"
import pandas as pd
data_ibm = pd.read_csv('../input/WA_Fn-UseC_-HR-Employee-Attrition.csv')


# In[ ]:


# Difference between series and data frames
# to create a serie
series = data_ibm['JobLevel']
print(type(series)) # to see the type of "series"


# In[ ]:


# to create a data frame
data_frame = data_ibm[['JobLevel']]
print(type(data_frame)) # to see the type of "data_frame"


# When we create series , we do not use list structure [[ ]]
# When we crate data frames, we use the list structure [[ ]]
# 

# To work with pandas, the first things that we have to learn as follows:
# - Logic
# - Control
# - Filtering

# In[ ]:


# Comparison and boolean operators
print("5 is greater than 15: ",5>15) # it will give us true if 5 is greater than 15, otherwise false.
print("3 is not equal to 25: ",3!=25) # it will give us true if 3 is not equal to 25, otherwise false.
# boolean operators: we have 2 boolean operators; True and False. We can think them as True=1 and False=0
# to understand logically better.
print("Result of True & True: " ,True and True)
print("Result of True & False: ",True and False)
print("Result of False || True: ",False or True)
print("Result of False || False: ", False or False)


# In[ ]:


# Filtering: let's create a filter named x and use it to filter data from the data set
# In this case we filtered employees who is younger than 20 years old
x = data_ibm['Age']<20
data_ibm[x]


# In[ ]:


# Filtering part2: by using logical_and
# This filter gives us female workers who are younger than 30 
import numpy as np
data_ibm[np.logical_and(data_ibm['Age']<30,data_ibm['Gender']=='Female')]


# In[ ]:


# Following code block will do the same filtering with the above code block
data_ibm[(data_ibm['Age']<30) & (data_ibm['Gender']=='Female')]


# **BRIEF LOOK TO WHILE AND FOR LOOPS**

# In[ ]:


# a normal while loop-basic
i = 0
while i!= 5 :
    print('i is : ',i)
    i+=1
print('now i is equal to',i,' and we are outside of the while loop')


# In[ ]:


# for loop with list
list1 = [10,20,30,40,50,60]
for i in list1:
    print('i is : ',i)
print('we finished the list, we are outside of the for loop now')


# In[ ]:


# now let's use "enumerate" and use the same list1= 10,20,30,40,50,60
# index: value = 0:10, 1:20, 2:30, 3:40, 4:50, 5:60
for index, value in enumerate(list1):
    print("index: ",index,":","values: ",value)
print('we finished the list, we are outside of the for loop now')


# In[ ]:


# using loops with dictionaries
dictionary = {'worker1':'woman','worker2':'man','worker3':'man','worker4':'woman','worker5':'woman'}
for key, value in dictionary.items():
    print("key of the dictionary: ",key,":","value of the dictionary: ",value)
print('we finished the dictionary, we are outside of the for loop now')


# In[ ]:


# we can access the indexes and values also for pandas library
# In this example we took first 10 rows' ages and their indexes for pandas library
for index,value in data_ibm[['Age']][0:10].iterrows():
    print("index of the data in pandas: ",index,":","value of the data in pandas: ",value)


# 
# 
