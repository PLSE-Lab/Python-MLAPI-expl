#!/usr/bin/env python
# coding: utf-8

# **Content:**
# 
# **1. Introduction to Python:**
#     -Matplotlib
#     -Dictionaries
#     -Pandas
#     -Logic, control flow and filtering
#     -Loop data structures
#     
# **2. Python Data Science Toolbox:**
#     -User defined function
#     -Scope
#     -Nested function
#     -Default and flexible arguments
#     -Lambda function
#     -Anonymous function
#     -Iterators
#     -List comprehension
# 
# **3. Cleaning Data**
#     -Diagnose data for cleaning
#     -Exploratory data analysis
#     -Visual exploratory data analysis
#     -Tidy data
#     -Pivoting data
#     -Concatenating data
#     -Data types
#     -Missing data and testing with assert
# 
# **4. Pandas Foundation**
#     -Building data frames from scratch
#     -Visual exploratory data analysis
#     -Statistical explatory data analysis
#     -Indexing pandas time series
#     -Resampling pandas time series
# 
# **5. Manipulating Data Frames with Pandas**
#     -Indexing data frames
#     -Slicing data frames
#     -Filtering data frames
#     -Transforming data frames
#     -Index objects and labeled data
#     -Hierarchical indexing
#     -Pivoting data frames
#     -Stacking and unstacking data frames
#     -Melting data frames
#     -Categoricals and groupby

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

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[ ]:


data = pd.read_csv('../input/tmdb_5000_movies.csv')


# In[ ]:


data.info()


# In[ ]:


data.columns


# In[ ]:


data.head()


# In[ ]:


data.corr()


# In[ ]:


# correlation map
f,ax = plt.subplots(figsize = (10,10))
sns.heatmap(data.corr(), annot = True, linewidths=.5, fmt = '.1f', ax = ax)
plt.show()


# # 1. INTRODUCTION TO PYTHON

# **- MATPLOTLIB**

# In[ ]:


# Line plot
data.revenue.plot(kind='line', color='r', label='revenue', linewidth=.7, alpha=.5, grid=True, linestyle='-' )
data.budget.plot(color='g', label='budget', linewidth=.7, alpha=.8, grid=True, linestyle='-.')
plt.legend(loc='upper right')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Line Plot')
plt.show()


# In[ ]:


# Scatter Plot
data.plot(kind='scatter', x='vote_average', y='budget', alpha=.5, color='r')
plt.xlabel('vote_average')
plt.ylabel('budget')
plt.title('Scatter Plot')
plt.show()


# In[ ]:


# Histogram
data.budget.plot(kind='hist', bins = 20, figsize = (10,10))
plt.show()


# **- DICTIONARY**

# In[ ]:


dictionary = {'usa' : 'ford', 'japan' : 'toyota', 'france' : 'renault'}
print(dictionary.keys())
print(dictionary.values())


# In[ ]:


dictionary['usa'] = "chevrolet"
print(dictionary)
dictionary['german'] = "mercedes"
print(dictionary)
del dictionary['france']
print(dictionary)
print('france' in dictionary)
dictionary.clear()
print(dictionary)


# **- PANDAS**

# In[ ]:


data = pd.read_csv('../input/tmdb_5000_movies.csv')


# In[ ]:


series = data['budget']
print(type(series))
data_frame = data[['budget']]
print(type(data_frame))


# In[ ]:


x = data['budget']>260000000
data[x]


# In[ ]:


data[np.logical_and(data['budget']>260000000, data['vote_average']>7)]


# In[ ]:


data[(data['budget']>260000000) & (data['vote_average']>7)]


# **- WHILE AND FOR LOOPS**

# In[ ]:


i = 0
while i != 5 :
    print('i is: ', i)
    i += 1
print(i, 'is equal to 5')


# In[ ]:


lis = [1,2,3,4,5]
for i in lis:
    print('i is: ',i)
print('')

for index, value in enumerate(lis):
    print(index," : ",value)
print('') 

dictionary = {'usa' : 'ford', 'japan' : 'toyota', 'france' : 'renault'}
for key,value in dictionary.items():
    print(key," : ",value)
print('')

for index,value in data[['budget']][0:1].iterrows():
    print(index," : ",value)


# # 2. PYTHON DATA SCIENCE TOOLBOX

# **- USER DEFINED FUNCTION**

# In[ ]:


def tuble_ex():
    t = (1,2,3,4)
    return t
a,b,c,d = tuble_ex()
print(a,b,c,d)


# **- SCOPE**

# In[ ]:


# Scope - Global/Local Scope
x = 5
def f():
    x = 7
    return x
print(x)
print(f())    


# In[ ]:


# No local scope
x = 8
def f():
    y = 3*x
    return y
print(f())


# **- NESTED FUNCTION**

# In[ ]:


# Nested Function
def square():
    def add():
        x = 3
        y = 4
        z = x + y
        return z
    return add()**2
print(square())


# **- DEFAULT AND FLEXIBLE ARGUMENTS**

# In[ ]:


# Default and Flexible Arguments
# Default Argument
def f(a, b = 1, c = 4):     #if we can not write (a) firstly, there is an error!
    x = a + b + c
    return x
print(f(3))
print(f(1,4,7))


# In[ ]:


# Flexible Argument
def f(*args):
    for i in args:
        print(i)
f(1)
print("")
f(1,4,7,10)

def f(*kwargs):
    """ print key and value of dictionary"""
    for key, value in kwargs.items():
        print(key, " ", value)
# (!)f(country = 'usa', brand = 'chevrolet')


# **- LAMBDA FUNCTION**

# In[ ]:


# Lambda Function
square = lambda x: x**2
print(square(5))
total = lambda x,y,z: x+y+z
print(total(1,3,5))


# **- ANONYMOUS FUNCTION**

# In[ ]:


# Anonymous Function
number_list = [1,3,5]
y = map(lambda x:x**2, number_list)
print(list(y))


# **- ITERATION**

# In[ ]:


# Iteration
name = "gomez"
it = iter(name)
print(next(it))
print(*it)


# In[ ]:


# zip
list1 = [1,3,5,7]
list2 = [2,4,6,8]
z = zip(list1,list2)
print(z)
z_list = list(z)
print(z_list)


# In[ ]:


un_zip = zip(*z_list)
un_list1,un_list2 = list(un_zip)
print(un_list1)
print(un_list2)
print(type(un_list1))
print(type(un_list2))


# **- LIST COMPREHENSION**

# In[ ]:


# List Comprehension
num1 = [1,3,5]
num2 = [i+2 for i in num1]
print(num2)


# In[ ]:


# Conditionals on iterable
num1 = [8,10,12]
num2 = [i**2 if i == 10 else i-3 if i < 10 else i+8 for i in num1]
print(num2)


# In[ ]:


threshold = sum(data.vote_average)/len(data.vote_average)
print('threshold:',threshold)
data['vote_level'] = ['high' if i > threshold else 'low' for i in data.vote_average]
data.loc[:10, ['vote_level','vote_average']]


# # 3. CLEANING DATA

# **- DIAGNOSE DATA FOR CLEANING**

# In[ ]:


data = pd.read_csv('../input/tmdb_5000_movies.csv')
data.head()


# In[ ]:


data.tail()


# In[ ]:


data.columns


# In[ ]:


data.shape


# In[ ]:


data.info()


# **- EXPLORATORY DATA ANALYSIS**

# In[ ]:


print(data['status'].value_counts(dropna = False))


# In[ ]:


data.describe()


# **- VISUAL EXPLORATORY DATA ANALYSIS**

# In[ ]:


data.boxplot(column = 'vote_average', by = 'status')


# **- TIDY DATA**

# In[ ]:


data_new = data.head()
data_new


# In[ ]:


melted = pd.melt(frame = data_new, id_vars = 'original_title', value_vars = ['budget', 'revenue'])
melted


# **- PIVOTING DATA**

# In[ ]:


melted.pivot(index = 'original_title', columns= 'variable', values = 'value') 


# **- CONCATENATING DATA**

# In[ ]:


data1 = data.head()
data2 = data.tail()
conc_data_row = pd.concat([data1,data2], axis = 0, ignore_index = True)
conc_data_row


# In[ ]:


data1 = data['budget'].head()
data2 = data['revenue'].head()
conc_data_col = pd.concat([data1, data2], axis = 1)
conc_data_col


# **- DATA TYPES**

# In[ ]:


data.dtypes


# In[ ]:


data['status'] = data['status'].astype('category')
data['vote_count'] = data['vote_count'].astype('float')


# In[ ]:


data.dtypes


# **- MISSING DATA AND TESTING WITH ASSERT**

# In[ ]:


data.info()


# In[ ]:


data['status'].value_counts(dropna = False)


# In[ ]:


assert 1 == 1


# In[ ]:


#assert 1 == 2


# In[ ]:


data["homepage"].fillna('empty',inplace = True)


# In[ ]:


assert  data['homepage'].notnull().all()


# # 4. PANDAS FOUNDATION

# **- BUILDING DATA FRAMES FROM SCRATCH**

# In[ ]:


country = ['usa', 'japan']
population = ['35', '10']
list_label = ['country', 'population']
list_col = [country, population]
zipped = list(zip(list_label, list_col))
data_dict = dict(zipped)
df = pd.DataFrame(data_dict)
df


# In[ ]:


df ['capital'] = ['washington','tokyo']
df


# In[ ]:


df['income'] = 0
df


# **- VISUAL EXPLORATORY DATA ANALYSIS**

# In[ ]:


data1 = data.loc[:, ['revenue','budget','popularity']]
data1.plot()
plt.show()


# In[ ]:


data1.plot(subplots = True)
plt.show()


# In[ ]:


data1.plot(kind = 'scatter', x = 'revenue', y = 'budget')
plt.show()


# In[ ]:


data1.plot(kind = 'hist', y = 'popularity', bins = 25, range = (0,250), normed = True)
plt.show()


# In[ ]:


fig, axes = plt.subplots(nrows = 2, ncols =1)
data1.plot(kind = 'hist', y = 'popularity', bins = 25, range = (0,150), normed = True, ax = axes[0])
data1.plot(kind = 'hist', y = 'popularity', bins = 25, range = (0,150), normed = True, ax = axes[1], cumulative = True)
plt.savefig('graph.png')
plt.show()


# **- STATISTICAL EXPLORATORY DATA ANALYSIS**

# In[ ]:


data.describe()


# **- INDEXING PANDAS TIME SERIES**

# In[ ]:


time_list = ['2002-03-10','2002-04-27']
print(type(time_list[1]))

datetime_object = pd.to_datetime(time_list)
print(type(datetime_object))


# In[ ]:


data2 = data.head()
date_list = ['2002-01-10', '2002-02-10', '2002-03-10', '2003-03-20', '2003-03-30']
datetime_object = pd.to_datetime(date_list)
data2['date'] = datetime_object
data2 = data2.set_index('date')
data2


# In[ ]:


print(data2.loc['2002-01-10'])
print(data2.loc['2002-01-10':'2002-03-10'])


# **- RESAMPLING PANDAS TIME SERIES**

# In[ ]:


data2.resample('A').mean()


# In[ ]:


data2.resample('M').mean()


# In[ ]:


data2.resample('M').first().interpolate('linear')


# In[ ]:


data2.resample('M').first().interpolate('linear')


# # 5. MANIPULATING DATA FRAMES WITH PANDAS

# **- INDEXING DATA FRAMES**

# In[ ]:


data = pd.read_csv('../input/tmdb_5000_movies.csv')
data = data.set_index('original_title')
data.head()


# In[ ]:


data['popularity'][0]


# In[ ]:


data.popularity[0]


# In[ ]:


data.loc['Avatar',['popularity']]


# In[ ]:


data[['budget', 'revenue']]


# **- SLICING DATA FRAME**

# In[ ]:


print(type(data['popularity']))
print(type(data[['popularity']]))


# In[ ]:


data.loc['Avatar':'Spectre', 'title':'vote_count']


# In[ ]:


data.loc['Spectre':'Avatar':-1, 'title':'vote_count']


# In[ ]:


data.loc['Avatar':'Tangled', 'status':]


# **- FILTERING DATA FRAMES**

# In[ ]:


boolean = data.budget > 260000000
data[boolean]


# In[ ]:


first_filter = data.budget > 250000000
second_filter = data.vote_average > 7
data[first_filter & second_filter]


# In[ ]:


data.budget[data.vote_average > 8]


# **- TRANSFORMING DATA**

# In[ ]:


def div(n):
    return n/2
data.budget.apply(div)


# In[ ]:


data.budget.apply(lambda n : n/2)


# In[ ]:


data['profit_rate'] = data.revenue / data.budget
data.head()


# **- INDEX OBJECTS AND LABELED DATA**

# In[ ]:


print(data.index.name)

data.index.name = 'index_name'
data.head()


# In[ ]:


data.tail(50)


# In[ ]:


data.head()

data3 = data.copy()
data3.index = range(50, 4853, 1)
data3.head()


# **- HIERARCHICAL INDEXING**

# In[ ]:


data = pd.read_csv('../input/tmdb_5000_movies.csv')
data.head()


# In[ ]:


data1 = data.set_index(['status', 'original_title'])
data1.head(100)


# **- PIVOTING DATA FRAMES**

# In[ ]:


dic = {"treatment":["A","A","B","B"],"gender":["F","M","F","M"],"response":[10,45,5,9],"age":[15,4,72,65]}
df = pd.DataFrame(dic)
df


# In[ ]:


df.pivot(index = 'treatment', columns= 'gender', values = 'response')


# **- STACKING AND UNSTACKING DATAFRAME**

# In[ ]:


df1 = df.set_index(['treatment', 'gender'])
df1


# In[ ]:


df1.unstack(level=0)


# In[ ]:


df1.unstack(level=1)


# In[ ]:


df2 = df1.swaplevel(0,1)
df2


# **- MELTING DATA FRAMES**

# In[ ]:


df


# In[ ]:


pd.melt(df,id_vars="treatment",value_vars=["age","response"])


# **- CATEGORICALS AND GROUPBY**

# In[ ]:


df


# In[ ]:


df.groupby("treatment").mean()


# In[ ]:


df.groupby("treatment").age.max() 


# In[ ]:


df.groupby("treatment")[["age","response"]].min() 


# **#FINISH#**
# Thanks for your votes and comments.
