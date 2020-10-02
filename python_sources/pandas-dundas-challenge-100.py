#!/usr/bin/env python
# coding: utf-8

# **Story line:**
# 
# Why Dundas?
# <br>Just a challenge between friends at Dundas Square about how quickly we can add basic exercises on pandas, hence "Pandas @ Dundas". 

# **Note:**
# 
# * Some tasks are created to showcase the bugs (like from_csv)
# * If you find any mistakes/improvements, please let me know. I am happy to fix them 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


import pandas
import pandas as pd
import numpy
import numpy as np
import random as rn
import functools
import re


# **Task 1: **
# 
# Check Pandas Version 

# In[ ]:


print('Task 1:')  
print(pd.__version__)


# **Task 2:**
#     
# Numpy Array 
# <br>
# Create three columns with Zero values

# In[ ]:


print('Task 2:')
dtype = [('Col1','int32'), ('Col2','float32'), ('Col3','float32')]
values = numpy.zeros(20, dtype=dtype)
index = ['Row'+str(i) for i in range(1, len(values)+1)]

df = pandas.DataFrame(values, index=index)

print(df)

df = pandas.DataFrame(values)
print(df)


# **Task 3:**
# 
# iLoc in Pandas
# <br>
# Print first five rows

# In[ ]:


print('Task 3:')
df = pandas.read_csv('../input/data1.csv', sep=';', header=None)
print(df.iloc[:4]) # 0 - 4 = 5 values


# **Task 4:**
# 
# Create Random integer between 2 to 10 with 4 items

# In[ ]:



print('Task 4:')
values = np.random.randint(2, 10, size=4)
print(values)


# **Task 5:**
# 
# Create Random integer between 0 to 100 

# In[ ]:


print('Task 5:')
df = pd.DataFrame(np.random.randint(0, 100, size=(3, 2)), columns=list('xy'))
print(df)


# **Task 6:**
# 
# Create Random integer between 2 to 10 with 4 columns

# In[ ]:


print('Task 6:')
df = pd.DataFrame(np.random.randint(0, 100, size=(2, 4)), columns=['A', 'B', 'C', 'D'])
print(df)


# **Task 7:**
# 
# 2D array with random between 0 and 5

# In[ ]:


print('Task 7:')
values = np.random.randint(5, size=(2, 4))
print(values)
print(type(values))


# **Task 8:        **
# 
# Create Random integer between 0 to 100 with 10 itmes (2 rows, 5 columns)

# In[ ]:


print('Task 8:')
df = pd.DataFrame(np.random.randint(0, 100, size=(3, 5)), columns=['Toronto', 'Ottawa', 'Calgary', 'Montreal', 'Quebec'])
print(df)


# **Task 9:**
# 
# 3 rows, 2 columns in pandas
# <br>
# 1st column = random between 10 to 20
# <br>
# 2nd column = random between 80 and 90
# <br>
# 3rd column = random between 40 and 50 

# In[ ]:


print('Task 9:')  
dtype = [('One','int32'), ('Two','int32')]
values = np.zeros(3, dtype=dtype)
index = ['Row'+str(i) for i in range(1, 4)]

df = pandas.DataFrame(values, index=index)
print(df)


# **Task 10:**
# 
# Fill Random Science and Math Marks
# (has some bugs in it)

# In[ ]:


print('Task 10:')  
dtype = [('Science','int32'), ('Maths','int32')]
values = np.zeros(3, dtype=dtype)

#print(type(dtype))
#values = np.random.randint(5, size=(3, 2))
#print(values)
#index = ['Row'+str(i) for i in range(1, 4)]

df = pandas.DataFrame(values, index=index)
print(df)


# **Task 11:**
# 
# CSV to DatRaframe (from_csv)
# <br>Note: from_csv is Deprecated since version 0.21.0: Use pandas.read_csv() instead.

# In[ ]:


print('Task 11:')  

csv = pd.DataFrame.from_csv('../input/uk-500.csv')
print(csv.head())


# **Task 12:**
# 
# CSV to Dataframe (from_csv)

# In[ ]:


print('Task 12:')  
#df = df.from_csv(path, header, sep, index_col, parse_dates, encoding, tupleize_cols, infer_datetime_format)
df = pd.DataFrame.from_csv('../input/uk-500.csv')
print(df.head())


# **Task 13:**
# 
# first 4 rows and 2 columns of CSV

# In[ ]:


print('Task 13:') 
df = pandas.read_csv('../input/data1.csv', sep=',')
print(df.shape) 
#print(df[2:14])
print(df.iloc[0:4,0:2])
#print(df[df.columns[0]])


# **Task 14:**
# 
# show even rows and first three columns

# In[ ]:


print('Task 14:') 
df = pandas.read_csv('../input/abc.csv', sep=',', encoding = "utf-8")
print(df.shape)

print(df.iloc[::2, 0:3])    


# **Task 15:**
# 
# New columns as sum of all

# In[ ]:


print('Task 15:') 
df = pandas.read_csv('../input/abc.csv', sep=',', encoding = "utf-8")
print(df.shape)
print(df) 
df['total'] = df.sum(axis=1)

print(df)


# **Task 16: ** 
# 
# Delete Rows of one column where the value is less than 50

# In[ ]:


print('Task 16:') 
df = pandas.read_csv('../input/abc.csv', sep=',', encoding = "utf-8")
print(df.shape)
print(df) 

df = df[df.science > 50]
print(df)


# **Task 17:**
# 
# Delete with Query
# <br>Note: Query doesn't work if your column has space in it

# In[ ]:


print('Task 17:') 
df = pandas.read_csv('../input/abc.csv', sep=',', encoding = "utf-8")
print(df.shape)
print(df) 

df = df.query('science > 45')
print(df)


# **Task 18:**
# 
# Skip single row

# In[ ]:


print('Task 18:') 
df = pandas.read_csv('../input/abc.csv', sep=',', encoding = "utf-8", skiprows=[5])
print(df.shape)
print(df)


# **Task 19:**
# 
# Skip multiple rows

# In[ ]:


print('Task 19:') 
df = pandas.read_csv('../input/abc.csv', sep=',', encoding = "utf-8", skiprows=[1, 5, 7])
print(df.shape)
#print(df) 

#df = df[df[[1]] > 45]
print(df)


# **Task 20:**
#         
# <br>Select Column by Index
# <br>Note:
#  <br>df[[1]] doesn't work in later Pandas

# In[ ]:


print('Task 20:') 
df = pandas.read_csv('../input/abc.csv', sep=',', encoding = "utf-8")
print(df.shape)
print(df) 

#df = df[int(df.columns[2]) > 45]
print(df)
print(type(df.columns[2]))


# **Task 21:**
#         
# Skip rows
# <br>Note:
# <br>df[[1]] doesn't work in later Pandas

# In[ ]:


print('Task 21:') 
df = pandas.read_csv('../input/abc.csv', sep=',', encoding = "utf-8", skiprows=[0])
print(df.shape)
print(df) 

#df = df[int(df.columns[2]) > 45]
#print(df)
print(df.columns[2])


# **Task 22:**
#  
# String to Dataframe
# <br>Note:
# <br>df[[1]] doesn't work in later Pandas

# In[ ]:


print('Task 22:')
from io import StringIO

s = """
        1, 2
        3, 4
        5, 6
    """

df = pd.read_csv(StringIO(s), header=None)

print(df.shape)
print(df)


# **Task 23:**
# 
# New columns as max of other columns
# <br>float to int used

# In[ ]:


print('Task 23:') 
df = pandas.read_csv('../input/abc.csv', sep=',', encoding = "utf-8")
print(df.shape)
df['sum'] = df.sum(axis=1)
df['max'] = df.max(axis=1)
df['min'] = df.min(axis=1)
df['average'] = df.mean(axis=1).astype(int)
print(df)


# **Task 24:**
# 
# New columns as max of other columns
# <br>float to int used
# <br>Math is considered more, so double the marks for maths

# In[ ]:


def apply_math_special(row):
    return (row.maths * 2 + row.language / 2 + row.history / 3 + row.science) / 4                

print('Task 24:') 
df = pandas.read_csv('../input/abc.csv', sep=',', encoding = "utf-8")
print(df.shape)
df['sum'] = df.sum(axis=1)
df['max'] = df.max(axis=1)
df['min'] = df.min(axis=1)
df['average'] = df.mean(axis=1).astype(int)
df['math_special'] = df.apply(apply_math_special, axis=1).astype(int)
print(df)


# **Task 25:**
#         
# New columns as max of other columns
# <br>35 marks considered as pass 
# <br>If the student fails in math, consider fail
# <br>If the student passes in language and science, consider as pass

# In[ ]:


def pass_one_subject(row):
    if(row.maths > 34):
        return 'Pass'
    if(row.language > 34 and row.science > 34):
        return 'Pass'
    
    return 'Fail'                

print('Task 25:') 
df = pandas.read_csv('../input/abc.csv', sep=',', encoding = "utf-8")
print(df.shape)   

df['pass_one'] = df.apply(pass_one_subject, axis=1)
print(df)


# **Task 26:**
#         
#  fill with average   
#        

# In[ ]:


print('Task 26:') 
df = pandas.read_csv('../input/abc2.csv', sep=',', encoding = "utf-8")
print(df.shape)   
print(df)
df.fillna(df.mean(), inplace=True)

#df['pass_one'] = df.apply(pass_one_subject, axis=1)
print(df)


# **Task 27:**
#         
# New columns as sum of all

# In[ ]:


print('Task 27:')
df = pd.DataFrame(np.random.rand(10, 5))
df.iloc[0:3, 0:4] = np.nan # throw in some na values
print(df)
df.loc[:, 'test'] = df.iloc[:, 2:].sum(axis=1)
print(df)


# **Task 28:**
# 
# Unicode issue and fix

# In[ ]:


print('Task 28:') 
df = pandas.read_csv('../input/score.csv', sep=',', encoding = "ISO-8859-1")
print(df.shape) 


# **Task 29:**
#         
# fill with average

# In[ ]:


print('Task 29:') 
df = pd.DataFrame(np.random.rand(3,4), columns=list("ABCD"))
print(df.shape)   
print(df)
df.fillna(df.mean(), inplace=True)

print(df)


# **Task 30:**
#  
#  Last 4 rows

# In[ ]:


print('Task 30:')  
df = pandas.read_csv('../input/data1.csv', sep=';') 
print(df[-4:])


# **Task 31:**
# 
# Expanding Apply

# In[ ]:


print('Task 31:')
series1 = pd.Series([i / 100.0 for i in range(1,6)])
print(series1)
def CumRet(x,y):
    return x * (1 + y)
def Red(x):
    return functools.reduce(CumRet,x,1.0)
s2 = series1.expanding().apply(Red)
# s2 = series1.expanding().apply(Red, raw=True) # is not working
print(s2)


# **Task 32:**
# 
# get 3 and 4th row

# In[ ]:


print('Task 32:')  
df = pandas.read_csv('../input/data1.csv', sep=';') 
print(df[2:4])


# **Task 33:**
# 
# Last 4th to 1st

# In[ ]:


print('Task 33:')  
df = pandas.read_csv('../input/data1.csv', sep=';') 
print(df[-4:-1])


# **Task 34:**
# 
# iloc position slice

# In[ ]:


print('Task 34:')  
df = pandas.read_csv('../input/data1.csv', sep=';') 
print(df.iloc[1:9])


# **Task 35:**
# 
# Loc - iloc - ix - at - iat

# In[ ]:


print('Task 35:')  
df = pandas.read_csv('../input/data1.csv', sep=';')


# **Task 36:**
# 
# Random data

# In[ ]:


print('Task 36:')  
def xrange(x):
    return iter(range(x))

rnd_1  =  [ rn.randrange ( 1 , 20 )  for  x  in  xrange ( 1000 )] 
rnd_2  =  [ rn.randrange ( 1 , 20 )  for  x  in  xrange ( 1000 )] 
rnd_3  =  [ rn.randrange ( 1 , 20 )  for  x in  xrange ( 1000 )] 
date  =  pd . date_range ( '2012-4-10' ,  '2015-1-4' )
print(len(date))
data  =  pd . DataFrame ({ 'date' : date ,  'rnd_1' :  rnd_1 ,  'rnd_2' :  rnd_2 ,  'rnd_3' :  rnd_3 })

data.head()


# **Task 37:**
# 
# filter with the value comparison

# In[ ]:


print('Task 37:')
below_20 = data[data['rnd_1'] < 20]    
print(below_20)


# **Task 38:**
# 
# Filter between 5 and 10 on col 1

# In[ ]:


print('Task 38:') 
def xrange(x):
    return iter(range(x))
rnd_1  =  [ rn.randrange ( 1 , 20 )  for  x  in  xrange ( 1000 )] 
rnd_2  =  [ rn.randrange ( 1 , 20 )  for  x  in  xrange ( 1000 )] 
rnd_3  =  [ rn.randrange ( 1 , 20 )  for  x in  xrange ( 1000 )] 
date  =  pd . date_range ( '2012-4-10' ,  '2015-1-4' )
print(len(date))
data  =  pd . DataFrame ({ 'date' : date ,  'rnd_1' :  rnd_1 ,  'rnd_2' :  rnd_2 ,  'rnd_3' :  rnd_3 })
below_20 = data[data['rnd_1'] < 20]
ten_to_20 = data[(data['rnd_1'] >= 5) & (data['rnd_1'] < 10)]
#print(ten_to_20)


# **Task 39:**
# 
# filter between 15 to 20

# In[ ]:


print('Task 39:')      
date  =  pd . date_range ( '2018-08-01' ,  '2018-08-15' )
date_count = len(date)

def fill_rand(start, end, count):
    return [rn.randrange(1, 20 ) for x in xrange( count )]

rnd_1 = fill_rand(1, 20, date_count) 
rnd_2 = fill_rand(1, 20, date_count) 
rnd_3 = fill_rand(1, 20, date_count)
#print(len(date))
data  =  pd . DataFrame ({ 'date' : date ,  'rnd_1' :  rnd_1 ,  'rnd_2' :  rnd_2 ,  'rnd_3' :  rnd_3 })
#print(len(date))
ten_to_20 = data[(data['rnd_1'] >= 15) & (data['rnd_1'] < 20)]
print(ten_to_20)


# **Task 40:**
# 
# 15 to 33

# In[ ]:


print('Task 40:')      
date  =  pd . date_range ( '2018-08-01' ,  '2018-08-15' )
date_count = len(date)

def fill_rand(start, end, count):
    return [rn.randrange(1, 20 ) for x in xrange( count )]

rnd_1 = fill_rand(1, 20, date_count) 
rnd_2 = fill_rand(1, 20, date_count) 
rnd_3 = fill_rand(1, 20, date_count)

data  =  pd . DataFrame ({ 'date' : date ,  'rnd_1' :  rnd_1 ,  'rnd_2' :  rnd_2 ,  'rnd_3' :  rnd_3 })

ten_to_20 = data[(data['rnd_1'] >= 15) & (data['rnd_1'] < 33)]
print(ten_to_20)


# **Task 41:**
# 
# custom method and xrnage on dataframe

# In[ ]:


print('Task 41:')  
date  =  pd . date_range ( '2018-08-01' ,  '2018-08-15' )
date_count = len(date)

def xrange(x):
    return iter(range(x))

def fill_rand(start, end, count):
    return [rn.randrange(1, 20 ) for x in xrange( count )]

rnd_1 = fill_rand(1, 20, date_count) 
rnd_2 = fill_rand(1, 20, date_count) 
rnd_3 = fill_rand(1, 20, date_count)

data  =  pd . DataFrame ({ 'date' : date ,  'rnd_1' :  rnd_1 ,  'rnd_2' :  rnd_2 ,  'rnd_3' :  rnd_3 })
filter_loc = data.loc[ 2 : 4 ,  [ 'rnd_2' ,  'date' ]]
print(filter_loc)


# **Task 42:**
# set index with date column

# In[ ]:


print('Task 42:')
date_date = data.set_index( 'date' ) 
print(date_date.head())


# **Task 43:**
# <br>Change columns based on other columns

# In[ ]:


print('Task 43:') 
df = pd.DataFrame({
    'a' : [1,2,3,4], 
    'b' : [9,8,7,6],
    'c' : [11,12,13,14]
});
print(df) 

print('changing on one column')
# Change columns
df.loc[df.a >= 2,'b'] = 9
print(df)


# **Task 44:**
# 
# Change multiple columns based on one column values

# In[ ]:


print('Task 44:')  
print('changing on multipe columns')
df.loc[df.a > 2,['b', 'c']] = 45
print(df)


# **Task 45:**
# 
# Pandas Mask

# In[ ]:


print('Task 45:')  
print(df)
df_mask = pd.DataFrame({
    'a' : [True] * 4, 
    'b' : [False] * 4,
    'c' : [True, False] * 2
})
print(df.where(df_mask,-1000))


# **Task 46:**
# 
# Check high or low comparing the column against 5

# In[ ]:


print('Task 46:')
print(df)  
df['logic'] = np.where(df['a'] > 5, 'high', 'low')
print(df)


# **Task 47:**
# 
# Student Marks (Pass or Fail)

# In[ ]:


print('Task 47:')
marks_df = pd.DataFrame({
    'Language' : [60, 45, 78, 4], 
    'Math' : [90, 80, 23, 60],
    'Science' : [45, 90, 95, 20]
});
print(marks_df)
marks_df['language_grade'] = np.where(marks_df['Language'] >= 50, 'Pass', 'Fail')
marks_df['math_grade'] = np.where(marks_df['Math'] >= 50, 'Pass', 'Fail')
marks_df['science_grade'] = np.where(marks_df['Science'] >= 50, 'Pass', 'Fail')
print(marks_df)


# **Task 48:**
# 
# Get passed grades

# In[ ]:


print('Task 48:')  
marks_df = pd.DataFrame({
    'Language' : [60, 45, 78, 4], 
    'Math' : [90, 80, 23, 60],
    'Science' : [45, 90, 95, 20]
});
print(marks_df)
marks_df_passed_in_language = marks_df[marks_df.Language >=50 ]
print(marks_df_passed_in_language)


# **Task 49:**
# 
# Students passed in Language and Math

# In[ ]:


print('Task 49:')  
marks_df_passed_in_lang_math = marks_df[(marks_df.Language >=50) & (marks_df.Math >= 50)]
print(marks_df_passed_in_lang_math)


# **Task 50:**
# 
# Students passed in Language and Science

# In[ ]:


print('Task 50:')  
marks_df_passed_in_lang_and_sc = marks_df.loc[(marks_df.Language >=50) & (marks_df.Science >= 50)]
print(marks_df_passed_in_lang_and_sc)


# **Task 51:**
# 
# Loc with Label oriented slicing
# <br>possible error:
# <br>pandas.errors.UnsortedIndexError

# In[ ]:


print('Task 51:')
stars = {
    'age' : [31, 23, 65, 50],
    'movies' : [51, 23, 87, 200],
    'awards' : [42, 12, 4, 78]
    }
star_names = ['dhanush', 'simbu', 'kamal', 'vikram']
stars_df = pd.DataFrame(data=stars, index=[star_names])
print(stars_df)


# **Task 52:**
# 
# iloc with positional slicing

# In[ ]:


print('Task 52:')  
print(stars_df.iloc[1:3])


# **Task 53:**
# 
# Label between numbers

# In[ ]:


print('Task 40:')  
numbers = pd.DataFrame({
        'one' : [10, 50, 80, 40],
        'two' : [2, 6, 56, 45]
    },
    index = [12, 14, 16, 18])
print(numbers)

print('label between 12 and 16')
print(numbers.loc[12:16])

print('index between 1 and 3')
print(numbers.iloc[1:3])


# **Task 54:**
# 
# stars with names

# In[ ]:


'''
    
'''
print('Task 54:') 
stars = {
    'age' : [31, 23, 65, 50],
    'movies' : [51, 23, 87, 200],
    'awards' : [42, 12, 4, 78]
    }
star_names = ['dhanush', 'simbu', 'kamal', 'vikram']
stars_df = pd.DataFrame(data=stars, index=[star_names])
numbers = pd.DataFrame({
        'one' : [10, 50, 80, 40],
        'two' : [2, 6, 56, 45]
    },
    index = [12, 14, 16, 18])
print(numbers)


# **Task 55:**
# 
# Row label selection
# Age is above 25 and movies above 25

# In[ ]:


print('Task 55:')

age_movies_25 = stars_df[(stars_df.movies > 25 ) & (stars_df.age > 25)]  
print(age_movies_25)


# **Task 56:**
# 
# stars in in certain ages

# In[ ]:


print('Task 56:')  
custom_stars = stars_df[stars_df.age.isin([31, 65])]
print(custom_stars)


# **Task 57:**
# 
# inverse opeartor
#    !( above one.45 and below two.50 )

# In[ ]:


print('Task 57:')  
print(numbers)
print(numbers[~( (numbers.one > 45) & (numbers.two < 50) )])


# **Task 58:**
# 
# apply custom function

# In[ ]:


print('Task 58:')
def GrowUp(x):
    avg_weight =  sum(x[x['size'] == 'series1'].weight * 1.5)
    avg_weight += sum(x[x['size'] == 'M'].weight * 1.25)
    avg_weight += sum(x[x['size'] == 'L'].weight)
    avg_weight /= len(x)
    return pd.Series(['L',avg_weight,True], index=['size', 'weight', 'adult'])

animals_df = pd.DataFrame({'animal': 'cat dog cat fish dog cat cat'.split(),
                   'size': list('SSMMMLL'),
                   'weight': [8, 10, 11, 1, 20, 12, 12],
                   'adult' : [False] * 5 + [True] * 2})

gb = animals_df.groupby(['animal'])

expected_df = gb.apply(GrowUp)
print(expected_df)


# **Task 59:**
# 
# Group by single column

# In[ ]:


print('Task 59:')
weights = animals_df.groupby(['weight']).get_group(20)  
print(weights)


# **Task 60:**
# 
# Creating new Columns using Applymap
# Sides & applymap

# In[ ]:


print('Task 60:')
sides_df = pd.DataFrame({
    'a' : [1, 1, 2, 4],
    'b' : [2, 1, 3, 4]
    })  
print(sides_df)
source_cols = sides_df.columns
print(source_cols)
new_cols = [str(x)+"_side" for x in source_cols]
side_category = {
    1 : 'North',
    2 : 'East',
    3 : 'South', 
    4 : 'West'
    }
sides_df[new_cols] = sides_df[source_cols].applymap(side_category.get)
print(sides_df)


# **Task 61:**
# 
# Replacing some values with mean of the rest of a group

# In[ ]:


print('Task 61:')  
df = pd.DataFrame({'A' : [1, 1, 2, 2], 'B' : [1, -1, 1, 2]})
print(df)

gb = df.groupby('A')

def replace(g):
    mask = g < 0
    g.loc[mask] = g[~mask].mean()
    return g

gbt = gb.transform(replace)

print(gbt)


# **Task 62:**
# 
# Students passed in Language or Science (any one subject)

# In[ ]:


print('Task 62:') 
marks_df = pd.DataFrame({
    'Language' : [60, 45, 78, 4], 
    'Math' : [90, 80, 23, 60],
    'Science' : [45, 90, 95, 20]
});
print(marks_df)
marks_df_passed_in_lang_or_sc = marks_df.loc[(marks_df.Language >=50) | (marks_df.Science >= 50)]
print(marks_df_passed_in_lang_or_sc)


# **Task 63:**
#    
# possible errors:
#             TypeError: 'Series' objects are mutable, thus they cannot be hashed

# In[ ]:


print('Task 63:')  
marks_df['passed_one_subject'] = 'Fail' 
marks_df.loc[(marks_df.Language >=50) , 'passed_one_subject'] = 'Pass'
print(marks_df)


# **Task 64:**
# 
# argsort
# Select rows with data closest to certain value using argsort

# In[ ]:


print('Task 64:')  
df = pd.DataFrame({
    "a": np.random.randint(0, 100, size=(5,)), 
    "b": np.random.randint(0, 70, size=(5,))
})
print(df)
par = 65
print('with argsort')
df1 = df.loc[(df.a-par).abs().argsort()]
print(df1)

print(df.loc[(df.b-2).abs().argsort()])


# **Task 65:**
# 
# argsort with stars        
# old stars (near by 50 age) argsort

# In[ ]:


print('Task 65:')  
stars = pd.DataFrame({
    "age": [17, 50, 24, 45, 65, 18], 
    "movies": [2, 3, 90, 45, 34, 2] 
})
print(stars.loc[(stars.age - 50).abs().argsort()])


# Task 66:
# Argsort with actors
# young stars (near by 17)

# In[ ]:


print('Task 66:')  
print(stars.loc[(stars.age - 17).abs().argsort()])


# **Task 67:**
# 
# Binary operators
# 
# Stars with
#     younger than 19 - very young
#     more movies acted

# In[ ]:


print('Task 67:')
stars = pd.DataFrame({
    "age": [17, 50, 24, 45, 65, 18], 
    "movies": [22, 33, 90, 75, 34, 2] 
})  
print(stars)
print('Young and more movies acted')
young = stars.age < 30    
more_movies = stars.movies > 30
young_more = [young, more_movies]
young_more_Criteria = functools.reduce(lambda x, y : x & y, young_more)
print(stars[young_more_Criteria])


# **Task 68:**
# 
# Young, Higher Salary, and Higher Position

# In[ ]:


print('Task 68:')  
employees = pd.DataFrame({
    "age": [17, 50, 24, 45, 65, 18], 
    "salary": [75, 33, 90, 175, 134, 78],
    "grade" : [7, 8, 9, 2, 7, 8] 
})  
print(employees)
print('Young, Higher Salary, and Higher Position')
young = employees.age < 30
high_salary = employees.salary > 60
high_position = employees.grade > 6
young_salary_position = [young, high_salary, high_position]
young_salary_position_Criteria = functools.reduce(lambda x, y : x & y, young_salary_position)
print(employees[young_salary_position_Criteria])


# **Task 69:**
# 
# Rename columns

# In[ ]:


print('Task 69:')  
employees = pd.DataFrame({
    "age": [17, 50, 24, 45, 65, 18], 
    "salary": [75, 33, 90, 175, 134, 78],
    "grade" : [7, 8, 9, 2, 7, 8] 
})  
print(employees)
employees.rename(columns={'age': 'User Age', 'salary': 'Salary 2018'}, inplace=True)
print(employees)


# **Task 70:**
# 
# Add a new column

# In[ ]:


print('Task 70:')  
employees = pd.DataFrame({
    "age": [17, 50, 24, 45, 65, 18], 
    "salary": [75, 33, 90, 175, 134, 78],
    "grade" : [7, 8, 9, 2, 7, 8] 
})  
print(employees)
employees['group'] = pd.Series(np.random.randn(len(employees)))
print(employees)


# **Task 71:**
# 
# Drop a column

# In[ ]:


print('Task 71:')  
employees = pd.DataFrame({
    "age": [17, 50, 24, 45, 65, 18], 
    "salary": [75, 33, 90, 175, 134, 78],
    "grade" : [7, 8, 9, 2, 7, 8] 
})  
print(employees)
employees['group'] = pd.Series(np.random.randn(len(employees)))
print(employees)
employees.drop(employees.columns[[0]], axis=1, inplace = True)
print(employees)


# **Task 72:**
# 
# Drop multiple columns

# In[ ]:


print('Task 72:')  
employees = pd.DataFrame({
    "age": [17, 50, 24, 45, 65, 18], 
    "salary": [75, 33, 90, 175, 134, 78],
    "grade" : [7, 8, 9, 2, 7, 8] 
})  
print(employees)
employees['group'] = pd.Series(np.random.randn(len(employees)))
print(employees)
employees.drop(employees.columns[[1, 2]], axis=1, inplace = True)
print(employees)


# **Task 73:**
# 
# Drop first and last column

# In[ ]:


print('Task 73:')  
employees = pd.DataFrame({
    "age": [17, 50, 24, 45, 65, 18], 
    "salary": [75, 33, 90, 175, 134, 78],
    "grade" : [7, 8, 9, 2, 7, 8],
    "group" : [1, 1, 2, 2, 2, 1] 
    
})  
print(employees)
employees.drop(employees.columns[[0, len(employees.columns)-1]], axis=1, inplace = True)
print(employees)


# **Task 74:**
# 
# Delete by pop function

# In[ ]:


print('Task 74:')  
employees = pd.DataFrame({
    "age": [17, 50, 24, 45, 65, 18], 
    "salary": [75, 33, 90, 175, 134, 78],
    "grade" : [7, 8, 9, 2, 7, 8],
    "group" : [1, 1, 2, 2, 2, 1] 
    
})  
print(employees)
group = employees.pop('group')
print(employees)
print(group)


# **Task 75:**
#     
# DataFrame.from_items

# In[ ]:


print('Task 75:')  
df = pd.DataFrame.from_items([('A', [1, 2, 3]), ('B', [4, 5, 6]), ('C', [7,8, 9])], orient='index', columns=['one', 'two', 'three'])
print(df)


# **Task 76:**
# 
# Pandas to list

# In[ ]:


print('Task 76:')
employees = pd.DataFrame({
    "age": [17, 50, 24, 45, 65, 18], 
    "salary": [75, 33, 90, 175, 134, 78],
    "grade" : [7, 8, 9, 2, 7, 8],
    "group" : [1, 1, 2, 2, 2, 1] 
    
})  
print(employees)  
employees_list1 = list(employees.columns.values) 
employees_list2 = employees.values.tolist()
#employees_list = list(employees)
print(employees_list1)
print(employees_list2)


# **Task 77:**
# 
# Pandas rows to list

# In[ ]:


print('Task 77:')
employees = pd.DataFrame({
    "age": [17, 50, 24, 45, 65, 18], 
    "salary": [75, 33, 90, 175, 134, 78],
    "grade" : [7, 8, 9, 2, 7, 8],
    "group" : [1, 1, 2, 2, 2, 1] 
    
})  
print(employees)  
employees_list2 = employees.values.tolist()
print(employees_list2)
print(type(employees_list2))
print(len(employees_list2))


# **Task 78:**
# 
# Pandas rows to array 
# 
# Note: as_matrix is deprecated

# In[ ]:


print('Task 78:')
employees = pd.DataFrame({
    "age": [17, 50, 24, 45, 65, 18], 
    "salary": [75, 33, 90, 175, 134, 78],
    "grade" : [7, 8, 9, 2, 7, 8],
    "group" : [1, 1, 2, 2, 2, 1] 
    
})  
print(employees)  
employees_list2 = employees.values
print(employees_list2)
print(type(employees_list2))
print(employees_list2.shape)


# **Task 79:**
# 
# Pandas rows to map

# In[ ]:


print('Task 79:')
employees = pd.DataFrame({
    "age": [17, 50, 24, 45, 65, 18], 
    "salary": [75, 33, 90, 175, 134, 78],
    "grade" : [7, 8, 9, 2, 7, 8],
    "group" : [1, 1, 2, 2, 2, 1] 
    
})  
print(employees)  
employees_list2 = map(list, employees.values)
print(employees_list2)
print(type(employees_list2))


# **Task 80:**
# 
# Pandas rows to map

# In[ ]:


print('Task 80:')
employees = pd.DataFrame({
    "age": [17, 50, 24, 45, 65, 18], 
    "salary": [75, 33, 90, 175, 134, 78],
    "grade" : [7, 8, 9, 2, 7, 8],
    "group" : [1, 1, 2, 2, 2, 1] 
    
})  
print(employees)  
employees_list2 = list(map(list, employees.values))
print(employees_list2)
print(type(employees_list2))


# **Task 81:**
#     
# Drop duplicates

# In[ ]:


print('Task 81:')  
users = pd.DataFrame({
    "id": [1, 1, 2, 2, 3, 3], 
    "city": ['Toronto', 'Montreal', 'Calgary', 'Montreal', 'Montreal', 'Ottawa'],
    "count" : [7, 8, 9, 2, 7, 8] 
    
}) 
print(users)
users.drop_duplicates('id', inplace=True, keep='last')
print(users)


# **Task 82:**
#     
# Selecting multiple columns

# In[ ]:


print('Task 82:')  
users = pd.DataFrame({
    "id": [1, 1, 2, 2, 3, 3], 
    "city": ['Toronto', 'Montreal', 'Calgary', 'Montreal', 'Montreal', 'Ottawa'],
    "count" : [7, 8, 9, 2, 7, 8] 
    
}) 
print(users)
users1 = users[['id', 'city']]
print(users1)


# **Task 83:**
#     
# Selecting multiple columns

# In[ ]:


print('Task 83:')  
users = pd.DataFrame({
    "id": [1, 1, 2, 2, 3, 3], 
    "city": ['Toronto', 'Montreal', 'Calgary', 'Montreal', 'Montreal', 'Ottawa'],
    "count" : [7, 8, 9, 2, 7, 8] 
    
}) 
print(users)
columns = ['id', 'count']
users1 = pd.DataFrame(users, columns=columns)
print(users1)


# **Task 84:**
#     
# Row and Column Slicing

# In[ ]:


print('Task 84:')  
users = pd.DataFrame({
    "id": [1, 1, 2, 2, 3, 3], 
    "city": ['Toronto', 'Montreal', 'Calgary', 'Montreal', 'Montreal', 'Ottawa'],
    "count" : [7, 8, 9, 2, 7, 8] 
    
}) 
print(users)    
users1 = users.iloc[0:2, 1:3]
print(users1)


# **Task 85:**
#     
# Iterating rows

# In[ ]:


print('Task 85:')  
users = pd.DataFrame({
    "id": [1, 1, 2, 2, 3, 3], 
    "city": ['Toronto', 'Montreal', 'Calgary', 'Montreal', 'Montreal', 'Ottawa'],
    "count" : [7, 8, 9, 2, 7, 8] 
    
}) 
print(users)    
for index, row in users.iterrows():
    print(row['city'], "==>", row['count'])


# **Task 86:**
#     
# Iterating tuples

# In[ ]:


print('Task 86:')  
users = pd.DataFrame({
    "id": [1, 1, 2, 2, 3, 3], 
    "city": ['Toronto', 'Montreal', 'Calgary', 'Montreal', 'Montreal', 'Ottawa'],
    "count" : [7, 8, 9, 2, 7, 8] 
    
}) 
print(users)    
for row in users.itertuples(index=True, name='Pandas'):
    print(getattr(row, 'city'))
    
for row in users.itertuples(index=True, name='Pandas'):
    print(row.count)


# **Task 87:**
#     
# Iterating rows and columns

# In[ ]:


print('Task 87:')  
users = pd.DataFrame({
    "id": [1, 1, 2, 2, 3, 3], 
    "city": ['Toronto', 'Montreal', 'Calgary', 'Montreal', 'Montreal', 'Ottawa'],
    "count" : [7, 8, 9, 2, 7, 8] 
    
}) 
print(users)    
for i, row in users.iterrows():
    for j, col in row.iteritems():    
        print(col)


# **Task 88:**
# 
# List of Dictionary to Dataframe

# In[ ]:


print('Task 88:')  
pointlist = [
                {'points': 50, 'time': '5:00', 'year': 2010}, 
                {'points': 25, 'time': '6:00', 'month': "february"}, 
                {'points':90, 'time': '9:00', 'month': 'january'}, 
                {'points_h1':20, 'month': 'june'}
            ]
print(pointlist)
pointDf = pd.DataFrame(pointlist)
print(pointDf)

pointDf1 = pd.DataFrame.from_dict(pointlist)
print(pointDf1)


# **Task 89:**

# In[ ]:


print('Task 89:')
df = pd.DataFrame(np.random.randn(10,6))
# Make a few areas have NaN values
df.iloc[1:3,1] = np.nan
df.iloc[5,3] = np.nan
df.iloc[7:9,5] = np.nan
print(df)
df1 = df.isnull()
print(df1)


# **Task 90:**
#     
# Sum of all nan

# In[ ]:


print('Task 90:')  
df = pd.DataFrame(np.random.randn(10,6))
# Make a few areas have NaN values
df.iloc[1:3,1] = np.nan
df.iloc[5,3] = np.nan
df.iloc[7:9,5] = np.nan
print(df)
print(df.isnull().sum())
print(df.isnull().sum(axis=1))
print(df.isnull().sum().tolist())


# **Task 91:**
#     
# Sum of all nan rowwise

# In[ ]:


print('Task 91:')  
df = pd.DataFrame(np.random.randn(10,6))
# Make a few areas have NaN values
df.iloc[1:3,1] = np.nan
df.iloc[5,3] = np.nan
df.iloc[7:9,5] = np.nan
print(df)
print(df.isnull().sum(axis=1))


# **Task 92:**
#     
# Sum of all nan as list

# In[ ]:


print('Task 92:')  
df = pd.DataFrame(np.random.randn(10,6))
# Make a few areas have NaN values
df.iloc[1:3,1] = np.nan
df.iloc[5,3] = np.nan
df.iloc[7:9,5] = np.nan
print(df)
print(df.isnull().sum().tolist())


# **Task 93:**
# 
# Change the order of columns
# 
# Note:
#     FutureWarning: '.reindex_axis' is deprecated and will be removed in a future version 

# In[ ]:


print('Task 93:')  
users = pd.DataFrame({
    "id": [1, 1, 2, 2, 3, 3], 
    "city": ['Toronto', 'Montreal', 'Calgary', 'Montreal', 'Montreal', 'Ottawa'],
    "count" : [7, 8, 9, 2, 7, 8] 
    
}) 
print(users)
users1 = users.reindex_axis(['city', 'count', 'id'], axis=1)
print(users1)

users2 = users.reindex(columns=['city', 'id', 'count'])
print(users2)


# **Task 94:**
# 
# Drop multiple rows

# In[ ]:


print('Task 94:')
numbers = pd.DataFrame({
    "id": [1, 2, 3, 4, 5, 6], 
    "number": [10, 20, 30, 30, 23, 12]
    
})  
print(numbers)
numbers.drop(numbers.index[[0, 3, 5]], inplace=True)
print(numbers)


# 

# **Task 95:**
#     
# Drop multiple rows by row name

# In[ ]:


print('Task 95:')  
numbers = pd.DataFrame({
    "id": [1, 2, 3, 4, 5, 6], 
    "number": [10, 20, 30, 30, 23, 12]
    
}, index=['one', 'two', 'three', 'four', 'five', 'six'])  
print(numbers)
numbers1 = numbers.drop(['two','six'])
print(numbers1)
numbers2 = numbers.drop('two')
print(numbers2)


# **Task 96:**
#     
# Get group

# In[ ]:


print('Task 96:')
cats = animals_df.groupby(['animal']).get_group('cat')
print(cats)


# **Task 97:**
# 
# Get the the odd row

# In[ ]:


print('Task 97:')  
x = numpy.array([
                    [ 1,  2,  3,  4,  5],
                    [ 6,  7,  8,  9, 10],
                    [11, 12, 13, 14, 15],
                    [16, 17, 18, 19, 20]
                ]
    )
print(x)
print(x[::2])


# **Task 98:**
# 
# Get the even columns

# In[ ]:


print('Task 98:')  
x = numpy.array([
                    [ 1,  2,  3,  4,  5],
                    [ 6,  7,  8,  9, 10],
                    [11, 12, 13, 14, 15],
                    [16, 17, 18, 19, 20]
                ]
    )
print(x)
print(x[:, 1::2])


# **Task 99:**
# 
# Odd rows and even columns

# In[ ]:


print('Task 99:')  

x = numpy.array([
                    [ 1,  2,  3,  4,  5],
                    [ 6,  7,  8,  9, 10],
                    [11, 12, 13, 14, 15],
                    [16, 17, 18, 19, 20]
                ]
    )
print(x)
print(x[::2, 1::2])


# **Task 100:**
#     
# Drop duplicates

# In[ ]:


print('Task 100:')  
users = pd.DataFrame({
    "id": [1, 1, 2, 2, 3, 3], 
    "city": ['Toronto', 'Montreal', 'Calgary', 'Montreal', 'Montreal', 'Ottawa'],
    "count" : [7, 8, 9, 2, 7, 8] 
    
}) 
print(users)
users.drop_duplicates('id', inplace=True)
print(users)


# **Task 101:**
#     
# Drop all duplicates

# In[ ]:


print('Task 101:')  
users = pd.DataFrame({
    "name": ['kevin', 'james', 'kumar', 'kevin', 'kevin', 'james'], 
    "city": ['Toronto', 'Montreal', 'Calgary', 'Montreal', 'Montreal', 'Ottawa'],
    "count" : [7, 8, 9, 2, 7, 8] 
    
}) 
print(users)
users.drop_duplicates('name', inplace=True, keep='last')
print(users)
users1 = users.drop_duplicates('name', keep=False)
print(users1)


# **Task 102:**
# 
# Basic group by

# In[ ]:


print('Task 102:')
animals_df1 = animals_df.groupby('animal').apply(lambda x: x['size'][x['weight'].idxmax()])
print(animals_df1)


# **Task 103:**
# 
# Missing Data:
# Make A'th 3rd coulmn Nan

# In[ ]:


print('Task 103:')  
df = pd.DataFrame(np.random.randn(6,1), index=pd.date_range('2013-08-01', periods=6, freq='B'), columns=list('A'))
print(df)
df.loc[df.index[3], 'A'] = np.nan
print(df)


# **Task 104:**
# 
# reindex

# In[ ]:


print('Task 104:')
df1 = df.reindex(df.index[::-1]).ffill()
print(df1)


# **Task 105:**
# 
# Column reset Nan

# In[ ]:


print('Task 105:')
animals_df = pd.DataFrame({'animal': 'cat dog cat fish dog cat cat'.split(),
                   'size': list('SSMMMLL'),
                   'weight': [8, 10, 11, 1, 20, 12, 12],
                   'adult' : [False] * 5 + [True] * 2})
print(animals_df)


# **Task 106:**

# In[ ]:


print('Task 106:')
df = pd.DataFrame([['http://wap.blah.com/xxx/id/11/someproduct_step2;jsessionid=....']],columns=['A'])
df1 = df['A'].str.findall("\\d\\d\\/(.*?)(;|\\?)",flags=re.IGNORECASE).apply(lambda x: pd.Series(x[0][0],index=['first']))
print(df1)


# **ps:**
# 
# * If you find it useful, please upvote this kernel.
# * Feel free to fork and do CRUD!

# In[ ]:




