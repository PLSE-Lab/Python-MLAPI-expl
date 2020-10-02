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
from pandas.core.reshape.reshape import pivot

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


data = pd.read_csv("../input/bitflyerJPY_1-min_data_2017-07-04_to_2018-06-27.csv")


# In[ ]:


data.info()


# In[ ]:


def tuble_ex():
    """ return defined t tuble"""
    t = (1,2,3)
    return t
x, y, z = tuble_ex()
print(x,y,z)


# In[ ]:


x = 2
def f():
    y = 5
    x = y *4
    return x
print(x)      
print(f())


# In[ ]:


x = 5
def f():
    y = 2 * x        
    return y
print(f()) 


# In[ ]:


import builtins
dir(builtins)


# In[ ]:


# Nested Function
def cube():
    """ return cube of value """
    def add():
        """ add three local variable """
        x = 2
        y = 3
        z = 5
        w = x + y + z
        return w
    return add()**3
print(cube()) 


# In[ ]:


# Default Arguments
def f(a, b = 10, c = 15):
    y = (c * b) / a
    return y
print(f(5))
print(f(3,5,6))


# In[ ]:


# flexible arguments *args
def f(*args):
    for i in args:
        print(i)
f(1)
print("")
f(1,2,3,4)
def f(**kwargs):
    """ print key and value of dictionary"""
    for key, value in kwargs.items():               
        print(key, " ", value)
f(country = 'turkey', capital = 'ankara', population = 123456)


# In[ ]:


# Lambda Function
cube = lambda x: x**3     
print(cube(5))
total = lambda x,y,z: x+y+z   
print(total(11,15,20))


# In[ ]:


number_list = [1,2,3]
y = map(lambda x:x**4,number_list)
print(list(y))


# In[ ]:


# Iteration Example
name = "Elif"
it = iter(name)
print(next(it))    
print(*it) 


# In[ ]:


# Zip Example
list_ = [1,2,3,4]
list__ = [5,6,7,8]
z = zip(list_,list__)
print(z)
z_list = list(z)
print(z_list)


# In[ ]:


un_zip = zip(*z_list)
un_list_,un_list__ = list(un_zip) # unzip returns tuble
print(un_list_)
print(un_list__)
print(type(un_list__))


# In[ ]:


# List Comprehension
num1 = [1,2,3]
num2 = [i * 5 for i in num1 ]
print(num2)


# In[ ]:


# Conditionals on Iterable
num1 = [10,20,50]
num2 = [i**3 if i == 10 else i-5 if i < 20 else i+5 for i in num1]
print(num2)


# In[ ]:


threshold = sum(data.Timestamp)/len(data.Timestamp)
data["timestamp_level"] = ["high" if i > threshold else "low" for i in data.Timestamp]
data.loc[:10,["timestamp_level","Timestamp"]]


# In[ ]:


SECTION 3


# In[ ]:


data.head()


# In[ ]:


data.tail()


# In[ ]:


data.shape


# In[ ]:


data.info()


# In[ ]:


print(data['timestamp_level'].value_counts(dropna =False))


# In[ ]:


data.describe()


# In[ ]:


# Tidy Data
new = data.head()   
new


# In[ ]:


melted = pd.melt(frame = new, id_vars = 'Name', value_vars= ['High','Low'])
melted


# In[ ]:


#Concetenating Data
data1 = data.head()
data2= data.tail()
conc_data_row = pd.concat([data1,data2],axis =0,ignore_index =True) 
conc_data_row


# In[ ]:


data1 = data['High'].head()
data2= data['Low'].head()
conc_data_col = pd.concat([data1,data2],axis =1) 
conc_data_col


# In[ ]:


data.dtypes


# In[ ]:


data['High'] = data['High'].astype('category')
data['Low'] = data['Low'].astype('float')


# In[ ]:


data.dtypes


# In[ ]:


data.info()


# In[ ]:


data["Close"].value_counts(dropna =False)


# In[ ]:


data1=data   
data1["Close"].dropna(inplace = True)


# In[ ]:


assert 10 == 10


# In[ ]:


assert 1==2


# In[ ]:


assert  data['Close'].notnull().all()


# In[ ]:


data["Close"].fillna('empty',inplace = True)


# In[ ]:


assert  data['Close'].notnull().all()


# In[ ]:


SECTION 4, 5


# In[ ]:


country = ["Turkey","Spain"]
population = ["150","200"]
list_label = ["country","population"]
list_colm = [country,population]
zipped = list(zip(list_label,list_colm))
data_dict = dict(zipped)
d_f = pd.DataFrame(data_dict)
d_f


# In[ ]:


d_f["capital"] = ["Japan","Tokyo"]
d_f


# In[ ]:


d_f["cities"] = 0 
d_f


# In[ ]:


# Plotting Data 
data1 = data.loc[:,["High","Low","Close"]]
data1.plot()
plt.show()


# In[ ]:


# Subplots
data1.plot(subplots = True)
plt.show()


# In[ ]:


# Scatter Plot  
data1.plot(kind = "scatter", x = "High", y = "Low")
plt.show()


# In[ ]:


# Hist Plot  
data1.plot(kind = "hist",y = "Close",bins = 50,range= (0,50),normed = True)


# In[ ]:


fig, axes = plt.subplots(nrows=2,ncols=1)
data1.plot(kind = "hist",y = "Close",bins = 50,range= (0,50),normed = True,ax = axes[0])
data1.plot(kind = "hist",y = "Close",bins = 50,range= (0,50),normed = True,ax = axes[1],cumulative = True)
plt.savefig('graph.png')
plt


# In[ ]:


data.describe()


# In[ ]:


time_list = ["1872-12-18","1792-09-12"]
print(type(time_list[1])) 
datetime_object = pd.to_datetime(time_list)
print(type(datetime_object))


# In[ ]:


import warnings
warnings.filterwarnings("ignore")
data2 = data.head()
date_list = ["1990-01-18","1991-02-10","1992-11-10","1993-10-15","1996-07-16"]
datetime_object = pd.to_datetime(date_list)
data2["date"] = datetime_object
data2= data2.set_index("date")
data2 


# In[ ]:


print(data2.loc["1990-01-18"])
print(data2.loc["1990-01-18":"1993-10-15"])


# In[ ]:


data2.resample("A").mean()


# In[ ]:


data2.resample("M").mean()


# In[ ]:


data2.resample("M").first().interpolate("linear")


# In[ ]:


data2.resample("M").mean().interpolate("linear")


# In[ ]:


data["Close"][1]


# In[ ]:


data.Close[1]


# In[ ]:


data.loc[1,["Close"]]


# In[ ]:


data[["High","Low"]]


# In[ ]:


# Series and Dataframes
print(type(data["Close"])) 
print(type(data[["Close"]])) 


# In[ ]:


data.loc[1:10,"High":"Low"]   


# In[ ]:


data.loc[10:1:-1,"High":"Low"] 


# In[ ]:


data.loc[1:10,"Close":] 


# In[ ]:


# Filtering Data Frames
boolean = data.Close > 10.352975
data[boolean]


# In[ ]:


# Combining Filters
first_filter = data.High > 1565767
second_filter = data.Low > 35334
data[first_filter & second_filter]


# In[ ]:


data.Low[data.Low<15]


# In[ ]:


# Transforming Data
def div(n):
    return n/2
data.High.apply(div)


# In[ ]:


data.High.apply(lambda n : n/5)


# In[ ]:


data["total_power"] = data.High * data.Low
data.head()


# In[ ]:


print(data.index.name)
data.index.name = "index_name"
data.head()


# In[ ]:


# Hierarchical Indexing
data = pd.read_csv('../input/bitflyerJPY_1-min_data_2017-07-04_to_2018-06-27.csv')
data.head()


# In[ ]:


data1 = data.set_index(["High","Low"]) 
data1.head(50)


# In[ ]:


# Pivoting Data Frames
dic = {"treatment":["K","L","M","N"],"gender":["E","L","I","F"],"response":[10,45,5,9],"age":[15,4,72,65]}
df = pd.DataFrame(dic)
df


# In[ ]:


df.pivot(index="treatment",columns = "gender",values="response")


# In[ ]:


df1 = df.set_index(["treatment","gender"])
df1


# In[ ]:


df1.unstack(level=0)


# In[ ]:


df1.unstack(level=1)


# In[ ]:


df2 = df1.swaplevel(0,1)
df2


# In[ ]:


# Melting
df


# In[ ]:


pd.melt(df,id_vars="treatment",value_vars=["age","response"])


# In[ ]:


df.groupby("treatment").mean() 


# In[ ]:


df.groupby("treatment").age.max()


# In[ ]:


df.groupby("treatment")[["age","response"]].min() 


# In[ ]:


df.info()


# In[ ]:




