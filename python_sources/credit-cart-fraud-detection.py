#!/usr/bin/env python
# coding: utf-8

# # ON CREDIT CARD FRAUD
# 
# **It's my first work for data science...**

# 

# I chose credit card fraud detection, for this work. Here we go!

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import os
print(os.listdir("../input"))
# Any results you write to the current directory are saved as output.
from subprocess import check_output
plt.show()


# * I need found the data of **Credit Card Fraud Detection**
# * Therefor i start by finding the data.
# * My data way (../input/creditcard.csv)
# * If i will copy and paste inside to *data = pd.read_csv()* python, will know where from this data.

# In[ ]:


data = pd.read_csv('../input/creditcard.csv')


# * I want to show data info. Therefor i used this code : *data.info()*
# * I can check how many float, string(object) or integer values in this data. And i can examine in this data have how many columns.
# * I see there how many memory usage in this data.

# In[ ]:


data.info()


# * If i need to see all of this data, i can write in console *data.corr()*
# * This code will show us correct of data.

# In[ ]:


data.corr()


# * In this line i work with correlation map.
# * Correlation map show us relationships between of values.
# * Using this code and i see that the values in the data are close to zero.

# In[ ]:


f,ax = plt.subplots(figsize=(20,20))
sns.heatmap(data.corr(), annot=True, linewidths=.3,fmt='.1f' , ax=ax)
plt.show()


# * If i want to check first five index from data, i can write in console **data.head()**

# In[ ]:


data.head()


# * If i want to learn is data columns i need to write **data.columns**

# In[ ]:


data.columns


# * I can check with line plot, in my data columns and values as line visualization. Therefor we need write to there **what is kind?**, **what is color?**, **what is label?**,  **which value of linewidth?**, **which value of alpha for clarifying?**, **gird have or not?** and **determine how will linestyle?** This work, i will detect this parameters and run code of line plot. Looks like this.
# * I use this function for three columns and his values.
# * And now, i know this datas values close to zero.
# 

# In[ ]:


#data.V4.plot(kind='line', color= 'red', label='V4', linewidth=1, alpha=1, grid=True, linestyle='-')
data.V5.plot(kind='line', color='blue', label='V5', linewidth=1, alpha=1, grid=True, linestyle='--')
#data.V6.plot(kind='line', color='green',label='V6', linewidth=1, alpha=0.7, grid=True, linestyle=':')
plt.legend(loc='upper right')
plt.xlabel('x axis')
plt.ylabel('y axis')
plt.title('Line Plot of Credit Carts Fraud Detections')
plt.show()


# In[ ]:


#data.V4.plot(kind='line', color= 'red', label='V4', linewidth=1, alpha=1, grid=True, linestyle='-')
#data.V5.plot(kind='line', color='blue', label='V5', linewidth=1, alpha=1, grid=True, linestyle='--')
data.V6.plot(kind='line', color='green',label='V6', linewidth=1, alpha=0.7, grid=True, linestyle=':')
plt.legend(loc='upper right')
plt.xlabel('x axis')
plt.ylabel('y axis')
plt.title('Line Plot of Credit Carts Fraud Detections')
plt.show()


# In[ ]:


data.Amount.plot(kind='line', color= 'red', label='Amount', linewidth=1, alpha=1, grid=True, linestyle='-')
#data.V5.plot(kind='line', color='blue', label='V5', linewidth=1, alpha=1, grid=True, linestyle='--')
#data.V6.plot(kind='line', color='green',label='V6', linewidth=1, alpha=0.7, grid=True, linestyle=':')
plt.legend(loc='upper right')
plt.xlabel('x axis')
plt.ylabel('y axis')
plt.title('Line Plot of Credit Carts Fraud Detections')
plt.show()


# * Scatter plot same as line plot. We need to determined to kind. If we want use scatter plot, then kind equal scatter. Looks like easy...
# * Which columns i will use i need determined. I chose in this data **Amount** and **V6**.

# In[ ]:


data.plot(kind='scatter', x='Amount', y='V6', alpha=0.6, color='orange')
plt.xlabel('Amount')
plt.ylabel('V6')
plt.title('Amount-V6 Scatter Plot')
plt.show()


# * Histogram is shown as hist. If i want to check histogramic visualization, then i will change of parameter kind to hist. Bins and figure size determines of how many big my visualization.

# In[ ]:


data.Amount.plot(kind='hist',bins=25, figsize=(10,10))
plt.show()


# * If i want to check, how many big my visiulation as pixels.

# In[ ]:


data.V8.plot(kind='hist', bins=25)
plt.clf()


# * If i want to know how many columns and values big then **10000** and small then **0**. 
# * I chose for columns **Amount** and **V2**, because it's credit card fraud detection.

# In[ ]:


data[np.logical_and(data['Amount']>10000, data['V2']<0)]


# * Now, i know how many registration more than **25,000** dollars defrauded.

# In[ ]:


data[(data['Amount']>25000) & (data['V17']<1)]


# * It's a my simple training about list and dictionary.

# In[ ]:


lis = ['cars','planes','trucks','ships','quadcopters']
for vehicles in lis:
    print('vehicles is: ',vehicles)
print("This from vehicle list...")
print("...........................................")
num = 50
while num != 110 :
    print('num is: ',num)
    num +=5 
print('last_index is:',num)
print("...........................................")
for index,value in data[['Amount']][0:3].iterrows():
    print(index," : ",value)


# * If we want change values of keys inside dictionaries we need some cods for this work.
# * keys and values in your dictionary. There for u need some dictionary or u can create your dictionary. Every time it varies like this keys1:values1 ,keys2:values2, keys3:values3, keys4:values4.
# * show dictionary : print(dictionary)
# * show keys : print(dictionary.keys())
# * show values : print(dictionary.values())
# * change value : dictionary[keys]=new values
# * add new keys : dictionary[new keys]=values
# * delete keys : del dictionary[keys]
# * check keys have or not : print ('keys' in dictionary)
# * delete dictionary : dictionary.clear()

# In[ ]:


dictionary={'monday':'day1', 'tuesday':'day2','wedn':'day3'}
print(dictionary.keys())
print(dictionary.values())
print("...........................................")
dictionary['monday']="first_dayOfweek"           #change value
for key,value in dictionary.items():
    print(key,":",value)
print("...........................................")
dictionary['thursday']="day4"                    #add
for key,value in dictionary.items():
    print(key,":",value)
print("...........................................")
del dictionary['tuesday']                        #del
for key,value in dictionary.items():
    print(key,":",value)
print("...........................................")
print('monday' in dictionary)                   #check
#last
dictionary.clear()
print("...........................................")
print(dictionary)

