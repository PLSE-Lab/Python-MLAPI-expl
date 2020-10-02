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
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[ ]:


data = pd.read_csv("../input/2017.csv")


# In[ ]:


data.info()


# In[ ]:


data.head(25)


# In[ ]:


data.corr()


# In[ ]:


f, ax = plt.subplots(figsize = (25,25))
sns.heatmap(data.corr(),annot = True, linewidth = .5, fmt = '.1f',ax=ax)
plt.show()


# In[ ]:


data.head()


# In[ ]:


data.columns


# **Matplot**

# **Line Plot**

# In[ ]:


data['Economy..GDP.per.Capita.'].plot(kind= 'line',color = 'r', label='GDP Per Capita', linewidth = 1,alpha = 0.9,grid=True, linestyle = ":" )
data["Health..Life.Expectancy."].plot(color = 'r',label = 'Life Expectancy',linewidth=1, alpha = 0.9,grid = True,linestyle = '-.')
plt.legend(loc='upper right')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Correlation Between GDP and HLE')
plt.show()


# **Scatter Plot**

# In[ ]:


data.plot(kind = 'scatter', x='Economy..GDP.per.Capita.', y='Health..Life.Expectancy.', alpha=0.9, color='black')
plt.xlabel('GDP')             
plt.ylabel('HLE')
plt.title('Attack Defense Scatter Plot')
plt.show()


# **Let's create a histogram**

# In[ ]:


data.Freedom.plot(kind='hist',bins = 50, figsize = (20,20))
plt.show()


# dictionary 

# In[ ]:


dictionary = {'cat' : 'tiger', 'reptile' : 'alligator'}
print(dictionary.keys())
print(dictionary.values())


# In[ ]:


dictionary['cat'] = 'lion'
print(dictionary)
dictionary['reptile'] = 'comodo dragon'
print(dictionary)
del dictionary['cat']
print(dictionary)
print('reptile' in dictionary)
print('lion' in dictionary)
dictionary.clear()
print(dictionary)


# **Pandas Exercise**

# In[ ]:


data = pd.read_csv("../input/2017.csv")


# In[ ]:


data.columns


# In[ ]:


series = data['Family']
print(type(series))
dataFrame = data[['Family']]
print(type(dataFrame))


# **logic, control flow and filtering**

# In[ ]:


print(3 > 2)
print(3!=2)
# Boolean operators
print(True and False)
print(True or False)


# In[ ]:


Filter = data['Economy..GDP.per.Capita.'] > 1.48
data[Filter]


# In[ ]:


data[np.logical_and(data['Economy..GDP.per.Capita.']>1.48, data['Generosity']>0.45)]


# In[ ]:


data[(data['Economy..GDP.per.Capita.']>1.48) & (data['Generosity']>0.45)]


# **while loop**

# In[ ]:


i = 0
while i != 7:
    
    print('i is', i)
    i+=1
print('i is equal to', i)    


# In[ ]:


lis = [1,2,3,4,5,6,7,8,9]
for i in lis:
    print('i is' ,i)
print('')
# enumaration
for index,value in enumerate(lis):
    print(index, ':' , value)
for index,value in data[['Generosity']][0:1].iterrows():
    print(index,':', value)


# **USER DEFINED FUNCTION**

# In[ ]:


def tuble_ex():
    """return defined t tuble"""
    t = (1,2,3)
    return t
a,b,c = tuble_ex()
print(a,b,c)


# **Scope**

# In[ ]:


x = 2 
def f():
    x = 3
    return x
print (x)
print (f())


# In[ ]:


x = 5
def f():
    y = 2*x
    return y
print(f())


# In[ ]:


import builtins
dir (builtins)


# **Nested Function**

# In[ ]:


def square():
    """ return square of value """
    def add():
        """ add two local variable """
        x = 2
        y = 3
        z = x + y
        return z
    return add()**2
print(square())   


# **DEFAULT and FLEXIBLE ARGUMENTS**

# In[ ]:


def f(*args):
    for i in args:
        print(i)
f(2)    
print("")
f(8,7,1,0,65)

def f(**kwargs):
    """print key and value of the dictionary"""
    for key,value in kwargs.items():
        print(key," ", value)
f( country = 'usa', capital = 'washington DC', population = 330000000)        


# **Lambda Fucntion**

# In[ ]:


kare = lambda x: x**2
print(kare(225))
abc = lambda x,y,z : x+y+z
print(abc(56,48,12))


# **Anonymous Function**

# In[ ]:


li = [1,2,3]
a = map(lambda x:x**2 , li)
print(list(a))


# **ITERATORS**

# In[ ]:


name = 'nicola tesla'
it = iter(name)
print(next(it))
print(*it)


# In[ ]:


list1 = [1,2,3,4]
list2 = [5,6,7,8]
z = zip(list1,list2)
print(z)
type(z) # we have to change it to list so
z_list = list(z)
print(z_list)


# In[ ]:


un_zip = zip(*z_list)
un_list1,un_list2 = list(un_zip)
print(un_list1)
print(un_list2)
print(type(un_list2))


# In[ ]:


num1 = [1,2,3]
num2 = [i + 1 for i in num1 ]
print(num2)


# In[ ]:


num1 = [5,10,15]
num2 = [i**2 if i == 10 else i-5 if i < 7 else i+5 for i in num1]
print(num2)


# In[ ]:


data.columns


# In[ ]:


d = sum(data["Economy..GDP.per.Capita."]) / len(data["Economy..GDP.per.Capita."])
data["GDP Level"] = ["high" if i> d else "low" for i in data["Economy..GDP.per.Capita."]]
data.loc[:30,["Economy..GDP.per.Capita.", "GDP Level"]]


# **CLEANING DATA** (here we go)

# In[ ]:


data = pd.read_csv('../input/2017.csv')
data.head()


# In[ ]:


data.tail()


# In[ ]:


data.columns


# In[ ]:


data.shape


# In[ ]:


data.info()


# **EXPLORATORY DATA ANALYSIS**

# In[ ]:


print(data['Freedom'].value_counts(dropna = False))


# In[ ]:


data.describe()


# In[ ]:


data.boxplot(column = 'Generosity' , by = 'Economy..GDP.per.Capita.')


# **Tidy Data**

# In[ ]:


data_new = data.head()    # I only take 5 rows into new data
data_new


# In[ ]:


melted = pd.melt(frame=data_new,id_vars = 'Country', value_vars= ['Freedom','Generosity'])
melted


# **PIVOTING DATA**

# In[ ]:


melted.pivot(index = 'Country' , columns = 'Freedom' , values = 'value')
malted.show()


# In[ ]:


data1 = data.head()
data2= data.tail()
conc_data_row = pd.concat([data1,data2],axis =0,ignore_index =True) # axis = 0 : adds dataframes in row
conc_data_row


# **DATA TYPES**

# In[ ]:


data.dtypes


# In[ ]:


data['Country'] = data['Country'].astype('category')
data['Family'] = data['Family'].astype('object')


# In[ ]:


data.dtypes


# **MISSING DATA and TESTING WITH ASSERT**

# In[ ]:


data.info()


# In[ ]:


data['Family'].value_counts(dropna = False)


# In[ ]:


data1 = data
data1['Generosity'].dropna(inplace = True)


# In[ ]:


assert 1==1


# In[ ]:


assert 1==2


# In[ ]:


assert data['Family'].notnull().all()


# In[ ]:


data['Family'].fillna('empty', inplace = True)


# In[ ]:


assert data['Family'].notnull().all()


# In[ ]:


# # With assert statement we can check a lot of thing. For example
# assert data.columns[1] == 'Name'
# assert data.Speed.dtypes == np.int


# **Pandas**

# In[ ]:


country = ["Spain","France"]
population = ["11","12"]
list_label = ["country","population"]
list_col = [country,population]
zipped = list(zip(list_label,list_col))
data_dict = dict(zipped)
df = pd.DataFrame(data_dict)
df


# In[ ]:


df["capital"] = ["madrid","paris"]
df


# In[ ]:


df["income"] = 0 #Broadcasting entire column
df


# **VISUAL EXPLORATORY DATA ANALYSIS**

# In[ ]:


data1 = data.loc[:,["Trust..Government.Corruption.","Freedom","Economy..GDP.per.Capita."]]
data1.reindex()
data1.plot()


# In[ ]:


data1.plot(subplots = True)
plt.show()


# In[ ]:


data1.plot(kind = "scatter",x="Trust..Government.Corruption.",y = "Economy..GDP.per.Capita.")
plt.show()


# In[ ]:


data1.plot(kind = "hist",y = "Economy..GDP.per.Capita.",bins = 50,range= (0,100),normed = True)


# In[ ]:


fig, axes = plt.subplots(nrows=2,ncols=1)
data1.plot(kind = "hist",y = "Freedom",bins = 50,range= (0,250),normed = True,ax = axes[0])
data1.plot(kind = "hist",y = "Freedom",bins = 50,range= (0,250),normed = True,ax = axes[1],cumulative = True)
plt.savefig('graph.png')
plt


# *i can't solve the problem above*

# **STATISTICAL EXPLORATORY DATA ANALYSIS**

# In[ ]:


data.describe()


# **INDEXING PANDAS TIME SERIES**

# In[ ]:


time_list = ["1992-03-08","1992-04-12"]
print(type(time_list[1])) # As you can see date is string
# however we want it to be datetime object
datetime_object = pd.to_datetime(time_list)
print(type(datetime_object))


# In[ ]:


import warnings
warnings.filterwarnings("ignore")
# In order to practice lets take head of pokemon data and add it a time list
data2 = data.head()
date_list = ["1992-01-10","1992-02-10","1992-03-10","1993-03-15","1993-03-16"]
datetime_object = pd.to_datetime(date_list)
data2["date"] = datetime_object
# lets make date as index
data2= data2.set_index("date")
data2 


# In[ ]:


print(data2.loc["1993-03-16"])
print(data2.loc["1992-03-10":"1993-03-16"])


# **RESAMPLING PANDAS TIME SERIES**

# In[ ]:


data2.resample("A").mean()


# In[ ]:


data2.resample("M").mean()


# In[ ]:


data2.resample("M").first().interpolate("linear")


# In[ ]:


data2.resample("M").mean().interpolate("linear")


# **MANIPULATING DATA FRAMES WITH PANDAS**

# In[ ]:


data = pd.read_csv('../input/2017.csv')

data.head()


# In[ ]:


data = data.set_index('#')


# In[ ]:


data['Freedom'][11]


# In[ ]:


data.loc[1, ['Freedom']]


# In[ ]:


data[['Freedom','Generosity']]


# **SLICING DATA FRAME**

# In[ ]:


print(type(data["Freedom"]))     # series
print(type(data[["Freedom"]]))   # data frames


# In[ ]:


data.loc[1:10,"Family":"Generosity"]


# In[ ]:


data.loc[10:1:-1,"Family":"Generosity"] 


# In[ ]:


data.loc[1:10 ,"Family":] 


# **FILTERING DATA FRAMES**

# In[ ]:


boolean = data.Family > 1.43
data[boolean]


# In[ ]:


first_filter = data.Family > 1.41
second_filter = data.Generosity > 0.20
data[first_filter & second_filter]


# In[ ]:


data.Family[data.Generosity<0.2]


# **TRANSFORMING DATA**

# In[ ]:


def div(n):
    return n/2
data.Family.apply(div)


# In[ ]:


data.Generosity.apply(lambda n : n/2)


# In[ ]:


data["total"] = data.Family + data.Generosity
data.head()


# **INDEX OBJECTS AND LABELED DATA**

# In[ ]:


# our index name is this:
print(data.index.name)
# lets change it
data.index.name = "index_name"
data.head()


# **PIVOTING DATA FRAMES**

# In[ ]:


dic = {"treatment":["A","A","B","B"],"gender":["F","M","F","M"],"response":[10,45,5,9],"age":[15,4,72,65]}
df = pd.DataFrame(dic)
df


# In[ ]:


df.pivot(index="treatment",columns = "gender",values="response")


# **STACKING and UNSTACKING DATAFRAME**

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


# **MELTING DATA FRAMES**

# In[ ]:


df


# In[ ]:


pd.melt(df,id_vars="treatment",value_vars=["age","response"])


# **CATEGORICALS AND GROUPBY**

# In[ ]:


df


# In[ ]:


df.groupby("treatment").mean()


# In[ ]:


df.groupby("treatment").age.max() 


# In[ ]:


df.groupby("treatment")[["age","response"]].min() 


# In[ ]:


df.info()


# I'm waiting for your support. This is my second exercise. Thank you.
