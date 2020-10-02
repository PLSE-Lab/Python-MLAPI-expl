#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Step 1: Loading required libraries

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns  # visualization tool

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
   for filename in filenames:
       print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


# Step 2: Import and first look data, and columns' names

data = pd.read_csv('../input/novel-corona-virus-2019-dataset/covid_19_data.csv')
data.info() # basic data information
data.columns #data's columns names
data.head() # first five data information to quick look data


# In[ ]:


# basic statistic info:
data.describe()


# In[ ]:


#Step 3: Correlation check between numerical variables 

f,ax = plt.subplots(figsize=(18, 18))
sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
plt.show()

# as expected, confirmed cases, deaths and recovered cases are correlated


# In[ ]:


#Step 4: possible plots for data

#line plot

data.Confirmed.plot(kind = 'line', color = 'g', label = 'Confirmed', linewidth = 1, alpha = 0.9, grid = True, linestyle = ':')
data.Deaths.plot(kind = 'line', color = 'r', label = 'Deaths', linewidth = 1, alpha = 0.9, grid = True, linestyle = '--')
data.Recovered.plot(kind = 'line', color = 'y', label = 'Recovered', linewidth = 1, alpha = 0.9, grid = True, linestyle = '-.')
plt.legend(loc = 'upper right')
plt.xlabel('x axis')
plt.ylabel('y axis')
plt.title('line plot')


# In[ ]:


#scatter plot

data.plot(kind='scatter', x='Confirmed',y='Recovered',alpha=0.9,color='red')
plt.xlabel('Confirmed')
plt.ylabel('Recovered')
plt.title('Confirmed Recovered Scatter Plot')



# In[ ]:


#histogram

data.plot(kind='hist',x = 'Country/Region',y = 'Confirmed',bins = 50, figsize = (15,15))
plt.show()


# In[ ]:


#Step5: Dictionary
dictionary = {'usa':'newyork','spain':'madrid','italy':'Lombardy','germany':'berlin','france':'paris'}
print(dictionary.keys())
print(dictionary.values())

dictionary['spain'] = "madrid"
print(dictionary)

dictionary['france'] = "paris"
print(dictionary)
del dictionary['spain']
print(dictionary)
print('france' in dictionary)
dictionary.clear()
print(dictionary)
del dictionary

print(dictionary)


# In[ ]:


#PANDAS

series = data['Confirmed']
print(type(series))

data_frame = data[['Confirmed']]
print(type(data_frame))

# comparision operators:
print('Confirmed' > 'Deaths')
print('Deaths' != 'Recovered')

#Boolean operators:

print(True and False)
print(True or False)

x = data['Confirmed'] > 80000
data[x]

data[np.logical_and(data['Confirmed']>80000, data['Deaths']>8000)]

data[(data['Confirmed']>80000)&(data['Deaths']>8000)]


# In[ ]:


#Loop data Structures
#while and For Loops

#Stay in loop if condition(i is not equal 5) is true

i = 0
while i != 5:
    print('i is:',i)
    i +=1
print(i,'is equal to 5')

lis = [1,2,3,4,5]
for i in lis:
    print('i is :',i)
print('')

#Enumarate index and value of list
#index : value =0:1, 1:2, 2:3, 3:4,4:5

for index, value in enumerate(lis):
    print(index,":",value)
print('')

#for dictionaries
#we can use for loop to achive key and value of dictionary

dictionary = {'spain':'madrid','france':'paris'}
for key,value in dictionary.items():
    print(key,":",value)
print('')


#for pandas we can achieve index and value

for index,value in data[['Confirmed']][0:1].iterrows():
    print(index,":",value)

