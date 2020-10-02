#!/usr/bin/env python
# coding: utf-8

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


# **How to Read Data**
# - we will use pandas
# -method: read_csv('the path of the data folder')

# In[ ]:


# the of our data is 'data'
data = pd.read_csv('../input/pokemon.csv')


# In[ ]:


# information about our data
data.info()


# In[ ]:


# corelation between datas
data.corr()


# In[ ]:


#correlation map
f,ax = plt.subplots(figsize=(14, 14))
sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
plt.show()


# In[ ]:


# to see top 5 data
data.head()


# In[ ]:


data.columns


# In[ ]:


# Scatter Plot 
# x = attack, y = defense
data.plot(kind='scatter', x='Attack', y='Defense',alpha = 0.8,color = 'cyan')
plt.xlabel('Attack')              # label = name of label
plt.ylabel('Defence')
plt.title('Attack Defense Scatter Plot')            # title = title of plot


# In[ ]:


# Line Plot
# To seperate plots we have used plt.subplot()
plt.subplot(2,1,1)
data.Attack.plot(kind='line', color='red', label='Attack', grid=True, alpha=1, linestyle='-.')
plt.legend(loc='upper right')
plt.xlabel('No')
plt.ylabel('Attack')

plt.subplot(2,1,2)
data.HP.plot(kind='line', color='purple', label='HP', linestyle='-')
plt.legend(loc='upper right')
plt.ylabel('HP')


# In[ ]:


# Histogram plot
data.Generation.plot(kind='hist', color='green', label='Generation', bins=20, figsize=(9,9))
plt.show()


# In[ ]:


# to clean the plot use .clf()
data.Speed.plot(kind='line')
plt.clf()


# **Dictionary**
# *-Importance*: Similar to Datasets and therefore easy to analyse
# **Some functions that can be used for dictionaries**
# -keys()
# -values()
# -items()
# -del
# -clear()
# -update()
# 
# 

# In[ ]:


# Example
dictionary = {'spain' : 'madrid','usa' : 'vegas'}
print(dictionary.keys())
print(dictionary.values())


# In[ ]:


# Keys have to be immutable objects like string, boolean, float, integer or tubles
# List is not immutable
# Keys are unique

# to add elements
dictionary['turkey'] = 'istanbul'
dictionary['greece'] = 'athens'

# to see keys and values
dictionary.items()


# In[ ]:


#to delete item
del dictionary['spain']              # remove entry with key 'spain'
dictionary.items()


# In[ ]:


# to check item
print('something' in dictionary)     # False: it means that there is no 'something'


# In[ ]:


# to update 2 dictionaries
dictionary.items()
# 2nd dictionary
dict2 = {'Italy': 'Rome',
         'France': 'Paris',
        'usa': 'toronto'}
# update
dictionary.update(dict2)
dictionary.items()


# In[ ]:


# delete dictionary
del dictionary
# it gives error, becasue the dictionary has been deleted 
print(dictionary)


# **Comparison Operators**
# equal: ==
# greater than: >
# smaller than: <
# not equal: !=
# greater than or equal to: >=

# In[ ]:


# if we write a comparison statement into the print function, it will give us truth of that
# for example
print(2==3) # False
print(5==5) # True
print(6<8) # True
print(6<=8) # True
print(3!=1) # True
print(7>3) # True
print(6!=6) # False


# In[ ]:


# we can use this info also in the if - while statements
if 5<6:
    print('it is true')
else: 
    print('it is false')


# ***filtering data using pandas***

# In[ ]:


# 1) pokemons who have higher defense value than 190
strong_pokemons = data.Defense > 190
data[strong_pokemons]


# In[ ]:


# 2) To use two conditions: There are only 2 pokemons who have higher defence value than 199 and higher attack value than 120
# using numpy logical_and module
awesome_pokemons = np.logical_and(data.Defense > 199, data.Attack > 120)
data [awesome_pokemons]


# **WHILE and FOR LOOPS**
# some basic loops

# In[ ]:


# Stay in loop if condition( i is not equal 5) is true
i = 0
while i < 4 :
    print('i is: ',i)
    i +=1 
print('i is outside while loop and equal to {}'.format(i))


# In[ ]:


# Stay in loop if i is in listt
listt = [1,2,3,4,5]
for i in listt:
    print('i is: {}'.format(i))
print('*'*30)

# Enumerate index and value of list
# index : value = 0:1, 1:2, 2:3, 3:4, 4:5
for index,value in enumerate(listt):
    print('index: {} value: {}'.format(index, value))
print('*'*30)   

# For dictionaries
# We can use for loop to achive key and value of dictionary. We have learnt key and value at dictionary part.
dictionary = {'spain':'madrid','france':'paris'}
for key, value in dictionary.items():
    print('dictionary key is: {}'.format(key), end=' ')
    print('dictionary value for this key is: {}'.format(value))
print('*'*30)

# For pandas we can achieve index and value
for index, value in data.Defense[0].iterrows():
    print('index: {} and value: {} of the first defense value'.format(index,value))


# In[ ]:




