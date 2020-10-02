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
from scipy import stats
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


data=pd.read_csv('../input/fifa19/data.csv')


# In[ ]:


data.info()


# In[ ]:


data.corr()


# In[ ]:


data.head()


# In[ ]:



f,ax  = plt.subplots(figsize=(18,18))
sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax,cmap="cubehelix",cbar_kws={'label': 'My Colorbar', 'orientation': 'horizontal'})
plt.show()


# In[ ]:


data.columns


# In[ ]:


data.plot(kind='scatter', x='Age', y='Potential',alpha = 0.5,color = 'red')
plt.xlabel('Attack')              # label = name of label
plt.ylabel('Defence')
plt.title('Attack Defense Scatter Plot') 
plt.show()


# In[ ]:




data.Age.plot(kind = 'hist',bins = 75,figsize = (10,12),edgecolor='red')
plt.show()


# In[ ]:


data.Potential.plot(kind = 'hist',bins = 50)
plt.clf()


# In[ ]:


data.Age.plot(kind = 'line', color = 'g',label = 'Age',linewidth=1,alpha = 0.5,grid = True,linestyle = ':')
data.Potential.plot(color = 'r',label = 'Potential',linewidth=1, alpha = 0.5,grid = True,linestyle = '-.')
plt.legend(loc='upper right')     # legend = puts label into plot
plt.xlabel('x axis')              # label = name of label
plt.ylabel('y axis')
plt.title('Line Plot')            # title = title of plot
plt.show()


# ***what is DICTIONARY***
# > A dictionary is a collection which is  indexed. In Python dictionaries are written with curly brackets, and they have keys and values.
# 

# In[ ]:


dicts =	{
  "name": "Cristiano",
  "surname": "Ronaldo",
  "potential": 94
}
print(dicts)


# In[ ]:


x = dicts["surname"]
print(x)


# In[ ]:


dicts['potential']=99
print(dicts)


# In[ ]:


for x in dicts:
  print(dicts[x])


# In[ ]:


print(len(dicts))


# In[ ]:


for x in  dicts.values():
  print(x)


# In[ ]:


#Loop through both keys and values, by using the items() function:
for x, y in  dicts.items():
  print(x, y)


# In[ ]:


dicts.clear()                   # remove all entries in dict
print(dicts)


# In[ ]:


dicts["age"] = "31"
print(dicts)


# In[ ]:


dicts.pop("name")
print( dicts)


# **pandas**

# In[ ]:


series = data['Potential']        
print(type(series))
data_frame = data[['Special']]  
print(type(data_frame))


# In[ ]:


# Comparison operator
print(3 > 2)
print(3!=2)
# Boolean operators
print(True and False)
print(True or False)


# In[ ]:


x = data['Potential']>90
data[x]


# In[ ]:


data.describe()


# In[ ]:


print(data.Name)


# In[ ]:


data[(data['Age']<25) & (data['Potential']>90)]


# **WHILE and FOR LOOPS**

#  **The while Loop**
# > With the while loop we can execute a set of statements as long as a condition is true.

# In[ ]:


i = 1
while i < len(data):
  print(i)
  if i ==5:
    break
  i += 1


# In[ ]:


for index, value in enumerate(data):
    print(index," : ",value)
print('') 


# In[ ]:


data.columns


#  **second homework**

# In[ ]:


#tuble
def tuble():
    value=(29,82,70)
    return value
age,potantial,dribbling=tuble()
print(age,dribbling)


# In[ ]:


#SCOPE
x="player"
def scopf():
    x="game"
    return x
print(x)
print(scopf())


# In[ ]:


# What if there is no local scope
x = 3
def f():
    sonuc = 2*x+5      
    return sonuc
print(f())  


# In[ ]:


#built in scope
import builtins
dir(builtins)


# In[ ]:


# nested function
def circumference():
    def circ():
        def add():
            x = (3.14,6)
            return x
        pi,r=add()
        circle=pi*r
        return circle
    return circ()*2
print(circumference())


# **DEFAULT and FLEXIBLE ARGUMENTS**

# In[ ]:


# default arguments
def f(a, b = -4, c = 0.2):
    y = a + b + c
    return y
print(f(5))

print(f(5,4,3))


# In[ ]:


# flexible arguments *args
def f(*fuc):
    for i in fuc:
        print(i)
f(1)
print("")
f(5,4,3,2,1,0)
# flexible arguments **kwargs that is dictionary
def f(**kwargs):
   
    for key, value in kwargs.items():               
        print(key, "=", value)
f(country = 'spain', capital = 'madrid', age= 27)


# In[ ]:


# lambda function
square = lambda x: x**2    
print(square(4))
tot = lambda x,y,z: x*y/z   
print(tot(1,2,3))


# In[ ]:


number = [1,2,3]
y = map(lambda x:x**2,number)
print(list(y))


# In[ ]:


# iteration example
name = "ronaldo"
it = iter(name)
print(next(it))    
print(*it)


# In[ ]:


# zip example
list1 = [1,2,3,4]
list2 = [5,6,7,8]
z = zip(list1,list2)
print(z)
z_list = list(z)
print(z_list)


# In[ ]:


un_zip = zip(*z_list)
un_list1,un_list2 = list(un_zip) # unzip returns tuble
print(un_list1)
print(un_list2)
print(type(un_list2))


# In[ ]:


num1 = [1,2,3]
num2 = [i**2   for i in num1 ]
print(num2)


# In[ ]:


num1 = [5,10,15,20,25]
num2 = [i**3 if i == 15 else i-5 if i < 10 else i+5 for i in num1]
print(num2)


# In[ ]:


data.columns


# In[ ]:


threshold = sum(data.Potential)/len(data.Balance)
data["BallControl"] = ["high" if i > threshold else "low" for i in data.Balance]
data.loc[:10,["BallControl","Balance"]]


# **third homework**

# In[ ]:


data.tail()


# In[ ]:


data.shape


# In[ ]:


data.info()


# **EDA**

# In[ ]:


data.columns


# In[ ]:


print(data.Position.value_counts(dropna=False))


# In[ ]:


data.describe()


# **visual exploratory data analysis**

# In[ ]:


short=data.head(30)
short.boxplot(column="Potential", by="Position")


# In[ ]:


#melt
shorts=data.head()
melts=pd.melt(frame=shorts,id_vars="Name",value_vars=["Potential","Position"])
melts


# In[ ]:


melts.pivot(index="Name",columns="variable",values="value")


# In[ ]:


#concatenating data
data1 = data.head()
data2= data.tail()
conc_data_row = pd.concat([data1,data2],axis =0,ignore_index =True) 
conc_data_row


# In[ ]:


data1 = data['Age'].head()
data2= data['Potential'].head()
conc_data_col = pd.concat([data1,data2],axis =1) 
conc_data_col


# In[ ]:


data.dtypes


# In[ ]:


data['Name'] = data['Name'].astype('category')
data['Age'] = data['Age'].astype('float')


# In[ ]:


data.dtypes


# In[ ]:


data.info()


# In[ ]:


data.Club.value_counts(dropna=False)


# In[ ]:


data1=data   # also we will use data to fill missing value so I assign it to data1 variable
data1["Age"].dropna(inplace = True) 


# In[ ]:


assert 1==1


# In[ ]:


assert  data['Age'].notnull().all()


# In[ ]:


data["Age"].fillna('empty',inplace = True)


# In[ ]:


assert  data['Age'].notnull().all()


# In[ ]:


team=["fb","gs"]
prize=["26","12"]
label=["team","prize"]
col=[team,prize]
zips=list(zip(label,col))
dic=dict(zips)
df=pd.DataFrame(dic)
df


# In[ ]:


df["win"]=["38","21"]
df


# In[ ]:


df["null"]=0
df


# In[ ]:


data1=data.head()
data1= data.loc[:,["Age","Potential","Special"]]
data1.plot()


# In[ ]:


data1.plot(subplots = True)
plt.show()


# In[ ]:


data1.plot(kind = "scatter",x="Age",y = "Potential")
plt.show()


# In[ ]:


data1.plot(kind = "hist",y = "Potential",bins = 50,range= (0,250),normed = True)
plt.show()


# In[ ]:


fig, axes = plt.subplots(nrows=2,ncols=1)
data1.plot(kind = "hist",y = "Potential",bins = 50,range= (0,250),normed = True,ax = axes[0])
data1.plot(kind = "hist",y = "Potential",bins = 50,range= (0,250),normed = True,ax = axes[1],cumulative = True)
plt.savefig('graph.png')
plt


# In[ ]:


data.describe()


# In[ ]:


time_list = ["1992-03-08","1992-04-12"]
print(type(time_list[1])) # As you can see date is string
# however we want it to be datetime object
datetime_object = pd.to_datetime(time_list)
print(type(datetime_object))


# In[ ]:



data2 = data.head()
date_list = ["1992-01-10","1992-02-10","1992-03-10","1993-03-15","1993-03-16"]
datetime_object = pd.to_datetime(date_list)
data2["date"] = datetime_object
data2= data2.set_index("date")
data2 


# In[ ]:


print(data2.loc["1993-03-16"])
print(data2.loc["1992-03-10":"1993-03-16"])


# In[ ]:


data2.resample("A").mean()


# In[ ]:


data2.resample("M").mean()


# In[ ]:


data2.resample("M").first().interpolate("linear")


# **4th Homework**
# *indexing data frames*

# In[ ]:


data.head()


# In[ ]:


data = pd.read_csv('../input/fifa19/data.csv')
data=data.set_index('ID')
data.head()


# In[ ]:


data["Age"][20801]


# In[ ]:


data.Age[158023]


# In[ ]:


data.loc[158023,["Age"]]


# In[ ]:


data2=data[["Age","Name"]]
data2


# In[ ]:


data1=data2.head(10)
data1


# **Slicing**

# In[ ]:


data = pd.read_csv('../input/fifa19/data.csv')
print(type(data.Age))
print(type(data["Age"]))
print(type(data[["Age"]]))


# In[ ]:


data.head()


# In[ ]:


data.loc[1:10,["Name","Age"]]


# In[ ]:


data.loc[10:0:-1,["Name","Age"]]


# In[ ]:


data.loc[1:10,"Marking":]


# **filtering**

# In[ ]:


boolean=data.Age<17
#data[boolean]
data.loc[boolean,["Name","Age"]]


# In[ ]:


filt1=data.Age<17
filt2=data.Potential>80
data[filt1 & filt2]


# In[ ]:


data.Name[data.Potential>92]


# ****transforming data****

# In[ ]:


def age(n):
    return n/2
data.Age.apply(age)


# In[ ]:


data.Age.apply(lambda n:n/2)


# In[ ]:


data["expertise"]=data.Potential*(data.Age/100)
data.loc[0:10,["Name","expertise"]]


# **index object and labeled data**

# In[ ]:


print(data.index.name)
data.index.name="index_name"
data.head()


# In[ ]:


data.head()
data2=data.copy()
data2.index=range(2,18209,1)
data2.head()


# In[ ]:


#hierarchical indexing
data=pd.read_csv('../input/fifa19/data.csv')
data.head()


# In[ ]:


data1=data.set_index(["Age","Name"])
data1.head(100)


# **pivoting**

# In[ ]:


dic={"team":["a","b","c","d"],"player":["x","y","z","v"],"age":[20,25,32,21],"value":[5,10,6,11]}
df=pd.DataFrame(dic)
df


# In[ ]:


df.pivot(index="team",columns="player",values="age")


# **stacking**

# In[ ]:


df1=df.set_index(["team","player"])
df1


# In[ ]:


df1.unstack(level=0)


# In[ ]:


df2=df1.swaplevel(0,1)
df2


# In[ ]:


df
#melt


# In[ ]:



pd.melt(df,id_vars="team",value_vars=["age","value"])


# **categoricals and groupby**

# In[ ]:


df


# In[ ]:


df.groupby("team").mean()


# In[ ]:


df.groupby("team").age.max()


# In[ ]:



df.info()


# In[ ]:


df["team"] = df["team"].astype("category")
df


# In[ ]:


df["player"] = df["player"].astype("category")
df


# In[ ]:


df.info()


# **analysis has finished**
