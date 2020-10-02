#!/usr/bin/env python
# coding: utf-8

# <h2>Content</h2>
# <hr>
# <h3>Pandas</h3><li>Read csv</li>
# <li>info()</li>
# <li>mean()</li>
# <li>min()</li>
# <li>max()</li>
# <li>nunique()</li>
# <li>min()</li>
# <li>max()</li>
# <li>nunique()</li>
# <li>value_counts()</li>
# <li>drop()</li>
# <li>Indexing Pandas Time Series</li>
# <li>Resampling Pandas Time Series</li>
# 
# <h3>Matplotlib</h3>
# <li>Create Figure</li>
# <li>Subplot</li>
# <li>Line Plot</li>
# <li>Scatter Plot</li>
# <li>Histogram Plot</li>
# 
# <h3>Python Data Science Toolbox</h3>
# <li>List Comprehension</li>
# <li>Lambda Function</li>
# <li>Default Argument</li>
# <li>Flexible Argument</li>
# 
# <h3>Cleaning Data</h3>
# <li>Diagnose Data for Cleaning</li>
# <li>Exploratory Data Analysis</li>
# <li>Dity Data</li>
# <li>Visual Exploratory Data Analysis</li>
# <li>Concatenating Data</li>
# <li>Data Types</li>
# <li>Missing Data and Testing With Assert</li>

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


data = pd.read_csv("../input/googleplaystore.csv")


# In[ ]:


data.info()


# <a id=2></a>

# In[ ]:


data.head() #view the top five data


# <a id=3></a>
# **I use two types;**
# * First type = Series
# * Second type = data_frame

# In[ ]:


series = data['Category'] #type series
print(type(series))
data_frame = data[['Category']] #type data frame
print(type(data_frame))


# <a id=4></a>

# In[ ]:


#Filter Pandas data frame
control_rating = data['Rating']>4.0  #list with rating greater than 4
data[control_rating]


# In[ ]:


len(data[control_rating]) #total number of ratings with a rating greater than 4


# In[ ]:


#Filtering Pandas with logical_and
data[np.logical_and(data['Rating']>4.0, data['Content Rating']=='Teen')]


# In[ ]:


#we can also use '&' for filtering.
data[(data['Rating']>4.0) & (data['Content Rating'] == 'Teen')]


# <a id=5></a>

# In[ ]:


data['Rating'].mean() #average of ratings


# <a id=6></a>

# In[ ]:


data[data['Rating'] == data['Rating'].max()] #application with the highest rating


# <a id=7></a>

# In[ ]:


data[data['Rating'] == data['Rating'].min()] #applications with the lowest rating


# <a id=8></a>

# In[ ]:


data["Category"].nunique() #total number of categories


# <a id=9></a>

# In[ ]:


data.groupby("Category").mean() #average number of ratings in categories


# <a id=10></a>

# In[ ]:


data['Category'].value_counts() #Number of applications in categories


# <a id=11></a>

# In[ ]:


data.drop(['Android Ver'],axis = 1,inplace = True) #drop 'Android Ver' column
data.info()


# <h3>INDEXING PANDAS TIME SERIES</h3>

# In[ ]:


import warnings
warnings.filterwarnings("ignore")
data2 = data.head()
date_list = ["1992-01-10","1992-02-10","1992-03-10","1993-03-15","1993-03-16"]
datetime_object = pd.to_datetime(date_list)
data2["date"] = datetime_object
data2 = data2.set_index("date")
data2


# <h3>RESAMPLING PANDAS TIME SERIES</h3>

# In[ ]:


#resample with year
data2.resample("A").mean()


# In[ ]:


#resample with month
data2.resample("M").mean()


# In[ ]:


data2.resample("M").first().interpolate("linear")


# In[ ]:


data2.resample("M").mean().interpolate("linear")


# <h2>MATPLOTLIB</h2>

# <a id=12></a>
# Create figure

# In[ ]:


x = np.arange(1,6)
y = np.arange(2,11,2)
plt.plot(x,y,"red")
plt.show()


# <a id=13></a>

# In[ ]:


plt.subplot(2,2,1)
plt.plot(x,y,"blue")

plt.subplot(2,2,2)
plt.plot(y,x,"red")

plt.subplot(2,2,3)
plt.plot(x**2,y,"black")


plt.subplot(2,2,4)
plt.plot(x,y**2,"green")
plt.show()


# In[ ]:


fig = plt.figure()
axes = fig.add_axes([0.1,0.2,0.4,0.6])
axes.plot(x,y)
axes.set_xlabel("x axis")
axes.set_ylabel("y axis")
axes.set_title("Plot")

plt.show()


# Reviews type = string 
# <br>Reviews type convert to int

# In[ ]:


int_reviews = []

for i in data['Reviews']:
    try:
        int_reviews.append(int(i))
    except ValueError:
        int_reviews.append(0)

data['IntReviews'] = int_reviews #add new column
data.info()


# In[ ]:


x = data['Rating']
y = data['IntReviews']
fig = plt.figure()
axes = fig.add_axes([0.1,0.1,0.8,0.8])
axes.plot(x,y,"red",linewidth=2,linestyle="--",marker="o",markersize=5,markerfacecolor="black",markeredgecolor="yellow",markeredgewidth=2)
plt.show()


# <a id=14></a>
# <h3>Line Plot</h3>

# In[ ]:


data.Rating.plot(kind="line",color="y",label="Rating",linewidth=1,alpha=0.5,grid=True,linestyle=":")
plt.xlabel('x axis')
plt.ylabel('y axis')
plt.title('Line Rating Plot')
plt.show()


# In[ ]:


data.IntReviews.plot(kind="line",color="b",label="Rating",linewidth=5,alpha=1,grid=True,linestyle="-",figsize=(12,12))
plt.xlabel('x axis')
plt.ylabel('y axis')
plt.title('Line Rating Plot')
plt.show()


# <a id=15></a>
# <h3>Scatter Plot</h3>

# In[ ]:


data.plot(kind='scatter',x='IntReviews', y='Rating', alpha = 0.5, color='green')
plt.xlabel('Reviews')
plt.ylabel('Rating')
plt.title('Scatter Plot')
plt.show()


# <a id=16></a>
# <h3>Histogram Plot</h3>

# In[ ]:


data.Rating.plot(kind='hist',bins=100,figsize=(12,12))
plt.show()


# <h2>PYTHON DATA SCIENCE TOOLBOX</h2>

# <h3>List Comprehension</h3>

# In[ ]:


data["rating_status"] = ["Popular" if i > 4.0 else "Not Popular" for i in data.Rating]

print(data.loc[:10,["rating_status","Rating"]])


# In[ ]:


numbers1 = np.arange(1,6)
print([i+10 for i in numbers1])


# In[ ]:


numbers2 = np.arange(1,25)

print(["Even Number" if i%2==0 else "Odd Number" for i in numbers2])


# <h3>LAMBDA FUNCTION</h3>

# In[ ]:


# classic function
def square_area_classic(x):
    return x**2

#lambda function
square_area_lambda = lambda x: x**2

#use functions
print(square_area_classic(10))
print(square_area_lambda(10))


# In[ ]:


#use multiple veriables
rectangle_area = lambda x,y: x*y
print(rectangle_area(3,5))


# In[ ]:


#exercise lambda function
reverse_str = lambda s : s[::-1]
print(reverse_str("Data Science"))


# <h3>DEFAULT ARGUMENT</h3>

# In[ ]:


def calculate(v_r,v_pi = 3.14):
    result = v_pi * v_r**2
    return result
print(calculate(3)) # v_r = 3 , v_pir = 3.14(default) 
print(calculate(4,3)) # change default arguments


# <h3>FLEXIBLE ARGUMENT</h3>

# In[ ]:


def flexible(*args):
    for i in args:
        print(i**2)
flexible(1,2,3)


# <h2>CLEANING DATA</h2>

# <h3>DIAGNOSE DATA for CLEANING</h3>
# We will use head, tail, columns, shape and info methods to diagnose data

# In[ ]:


data.head() #show first 5 rows


# In[ ]:


data.tail() #show last 5 rows


# In[ ]:


data.columns #gives column names of features


# In[ ]:


data.shape #gives number of rows and columns in a tuble


# In[ ]:


data.info() #gives data type like dataframe, number of sample or row, number of feature or column


# <h3>EXPLORATORY DATA ANALYSIS</h3>

# We will use describe() method. Describe method includes:
# * count: number of entries
# * mean: average of entries
# * std: standart deviation
# * min: minimum entry
# * 25%: first quantile
# * 50%: median or second quantile
# * 75%: third quantile
# * max: maximum entry
# 

# In[ ]:


data.describe()


# **value_counts(): Frequency counts**
# 

# In[ ]:


print(data['Rating'].value_counts(dropna=False))


# <h3>TIDY DATA</h3>
# We tidy data with melt().

# In[ ]:


first_five_data = data.head()
melted = pd.melt(frame=first_five_data,id_vars='App',value_vars=['Rating','Type'])
print(melted)


# <h3>VISUAL EXPLORATORY DATA ANALYSIS</h3>
# Box plots: visualize basic statistics like outliers, min/max or quantiles

# In[ ]:


data.boxplot(column="Rating",by ="rating_status")


# <h3>CONCATENATING DATA</h3>
# We can concatenate two dataframe 

# In[ ]:


data1 = data.head()
data2 = data.tail()
conc_data_row = pd.concat([data1,data2],axis=0,ignore_index=True)
conc_data_row


# <h3>DATA TYPES</h3>
# There are 5 basic data types:
# * object(string)
# * boolean
# * integer
# * float
# * categorial
# 

# In[ ]:


data.dtypes


# In[ ]:


#convert object(str) to categorical
data['Type'] = data['Type'].astype('category')
data.dtypes


# <h3>MISSING DATA and TESTING WITH ASSERT</h3>

# In[ ]:


#look at does data have nan value
data.info()


# In[ ]:


#check Rating
data["Rating"].value_counts(dropna=False)


# In[ ]:


#drop nan values
data1 = data
data1["Rating"].dropna(inplace=True)


# In[ ]:


#  Lets check with assert statement
# Assert statement:
assert 1==1 # return nothing because it is true


# In[ ]:


assert  data['Rating'].notnull().all() # returns nothing because we drop nan values


# In[ ]:


data["Rating"].fillna('empty',inplace = True)


# In[ ]:


assert  data['Rating'].notnull().all() # returns nothing because we do not have nan values

