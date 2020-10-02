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


'''
PYTHON FOR DATA SCIENCE - LISTS
'''

square_list = [0, 1, 4, 9, 16, 25]
print('\n\nList = ',square_list)

'''Individual items can be accessed by index'''
#Indexing returns the items
print('\n\nItem at Index 0 = ',square_list[0])


'''slicing of list'''
# a range of items can be accessed by providing the first and last index
print('\n\n Items from index 2 to 4 = ', square_list[2:4])


'''Access elements from end with negative index'''
print('\n\nThe second last element of the list is :', square_list[-2])


# In[ ]:


'''
PYTHON FOR DATA SCIENCE - STRINGS
'''

# define a string Value
my_string = 'Hello'

# character at index 1
print('Character at index 1 = ', my_string[1])

# length of the string
print('\n\nLength of the string = ', len(my_string))

# string concatenation
new_string = my_string+' World'
print('\n\nnew String is = ',new_string)


# In[ ]:


'''
PYTHON FOR DATA SCIENCE - TUPLES AND DICTIONARY
'''

# define a tuple
my_tuple = (1, 2, 3, 4, 5)

# access single element using index
print('TUPLE = ',my_tuple)

print('\n\nelement at index 1 = ',my_tuple[1])

# uncomment the below code and you see the error as the tuples are immutable
#my_tuple[2] = 4

# define a dictionary
my_dict = {
  'key_1' : 4,
  'key_2' : 5,
  'key_3' : 6,
  'key_4' : 7
}

# access value of any key
print('\n\nThe value of key_1 in the dictionary is ',my_dict['key_1'])

# dictionaries are mutable
my_dict['key_1'] = 1000

print('\n\nThe value of key_1 after update in the dictionary is ',my_dict['key_1'])


#keys of dictionary
print('\n\nkeys = ',my_dict.keys())


# In[ ]:


import math as m


# In[ ]:


N=5
m.factorial(N)


# In[ ]:


df = pd.read_csv("../input/international-football-results-from-1872-to-2017/results.csv") #Reading the dataset in a dataframe using Pandas


# In[ ]:


df.head()


# In[ ]:


df.describe()


# In[ ]:


df.shape


# In[ ]:


df['home_score'].value_counts(ascending=False).head()


# In[ ]:


df['home_score'].hist(bins=50)


# In[ ]:


df.boxplot(column='home_score')


# In[ ]:


df.boxplot(column='home_score', by ='neutral')


# In[ ]:


df.apply(lambda x: sum(x.isnull()),axis=0) 


# In[ ]:


df['total_score']=df['home_score']+df['away_score']


# In[ ]:


df['total_score_log']=np.log(df['total_score'])


# In[ ]:


df['total_score_log'].head()

