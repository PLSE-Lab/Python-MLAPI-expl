#!/usr/bin/env python
# coding: utf-8

# DATA SCIENTIST
# 
# In this tutorial, I am trying to learn data scientist. I use examples of Data ScienceTutorial for Beginners. 

# In[ ]:



import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns  # visualization tool


# In[ ]:


data=pd.read_csv('../input/pokemon.csv')


# In[ ]:


data.info()


# In[ ]:


data.head() #if Type 2==''


# In[ ]:


data.corr() 


# In[ ]:


f,ax = plt.subplots(figsize=(9, 9))
sns.heatmap(data.corr(), annot=True, linewidths=.5,fmt= '.1f', ax=ax)
plt.show()


# In[ ]:


data.head(10)


# In[ ]:


data.columns


# INTRODUCTION TO PYTHON

# In[ ]:


# Line Plot
# color = color, label = label, linewidth = width of line, alpha = opacity, grid = grid, linestyle = sytle of line
data.Speed.plot(kind = 'line',color = 'g',label = 'Speed',linewidth=1,alpha = 0.5, grid = True)
data.Defense.plot(color = 'b',label = 'Defense',linewidth=1,grid = True, alpha = 1,linestyle = '-') #linestyle = '-.'
plt.legend(loc='upper right')     # legend = puts label into plot
plt.xlabel('it is x')             # label = name of label
plt.ylabel('it is y')
plt.title('It is Line Plot')            #title = title of plot
plt.show()


# In[ ]:


# Scatter Plot 
# x = attack, y = defense
data.plot(kind='scatter', x='Attack', y='Defense',alpha = 0.5,color = 'green',grid=True)
plt.xlabel('It is Attack')              # label = name of label
plt.ylabel('It is Defence')
plt.title('Attack Defense Scatter Plot')            # title = title of plot
plt.show()


# In[ ]:


# Histogram
# bins = number of bar in figure
data.Speed.plot(kind = 'hist',bins = 100,figsize = (15,15),grid=True)
plt.show()


# In[ ]:


# clf() = cleans it up again you can start a fresh
data.Speed.plot(kind = 'hist',bins = 50)
plt.clf()
# We cannot see plot due to clf()


# In[ ]:


data.head(10)


# DICTIONARY

# In[ ]:


#create dictionary and look its keys and values
dictionary = {'spain' : 'madrid','usa' : 'vegas'}
print(dictionary.keys())
print(dictionary.values())


# In[ ]:


# Keys have to be immutable objects like string, boolean, float, integer or tubles
# List is not immutable
# Keys are unique
dictionary['spain'] = "barcelona"    # update existing entry
print(dictionary)
dictionary['france'] = "paris"       # Add new entry
print(dictionary)
del dictionary['spain']              # remove entry with key 'spain'
print(dictionary)
print('france' in dictionary)        # check include or not
dictionary.clear()                   # remove all entries in dict
print(dictionary)


# PANDAS

# In[ ]:


data=pd.read_csv('../input/pokemon.csv')


# In[ ]:


series = data['Defense']        # data['Defense'] = series -> like one dimensional vector
print(type(series))
data_frame = data[['Defense']]  # data[['Defense']] = data frame -> like multidimensional vector
print(type(data_frame))
print (series)
print (data_frame)


# In[ ]:


# 1 - Filtering Pandas data frame
x = data['Defense']>200     # There are only 3 pokemons who have higher defense value than 200
data[x]

#data.head() #if Type 2==''


# In[ ]:


# 2 - Filtering pandas with logical_and
# There are only 2 pokemons who have higher defence value than 200 and higher attack value than 100
data[np.logical_and(data['Defense']>200, data['Attack']>100 )]


# 
# WHILE and FOR LOOPS

# In[ ]:


i = 0
while i != 5 :
    print('i is: ',i)
    i +=1 
print(i,' is equal to 5')


# In[ ]:


#Stay in loop if condition( i is not equal 5) is true
lis = [1,2,3,4,5]
for i in lis:
    print('i is: ',i)
print('')

# Enumerate index and value of list
# index : value = 0:1, 1:2, 2:3, 3:4, 4:5
for index, value in enumerate(lis):
    print(index," : ",value)
print('')   

# For dictionaries
# We can use for loop to achive key and value of dictionary. We learnt key and value at dictionary part.
dictionary = {'spain':'madrid','france':'paris'}
for key,value in dictionary.items():
    print(key," : ",value)
print('')

# For pandas we can achieve index and value
for index,value in data[['Attack']][0:1].iterrows():
    print(index," : ",value)


# In[ ]:


def tuble_ex():
    """ return defined t tuble"""
    t = (1,2,3)
    return t
a,b,c = tuble_ex()
print(a,b,c)


# In[ ]:


# What if there is no local scope
x = 5
def f():
    y = 2*x        # there is no local scope x
    return y
print(f())         # it uses global scope x
# First local scopesearched, then global scope searched, if two of them cannot be found lastly built in scope searched.


# In[ ]:


import builtins
dir(builtins)


# NESTED FUNCTION

# In[ ]:


#nested function
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


# In[ ]:


# lambda function
square = lambda x: x**2     # where x is name of argument
print(square(4))
tot = lambda x,y,z: x+y+z   # where x,y,z are names of arguments
print(tot(1,2,3))


# In[ ]:



number_list = [1,2,3] 
y = map(lambda x:x**2,number_list) #trying for each number
print(list(y))


# ITERATOR

# In[ ]:


# iteration example
name = "ronaldo"
it = iter(name)

print (it)
print(next(it))    # print next iteration
print(*it)         # print remaining iteration


# In[ ]:


# zip example
list1 = [1,2,3,4]
list2 = [5,6,7,8]
z = zip(list1,list2)
print(z)
z_list = list(z)
print(z_list)

un_zip = zip(*z_list)
un_list1,un_list2 = list(un_zip) # unzip returns tuble
print(un_list1)
print(un_list2)
print(type(un_list2))


# In[ ]:


# Example of list comprehension
num1 = [1,2,3]
num2 = [i + 1 for i in num1 ]
print(num2)


# In[ ]:


# Conditionals on iterable
num1 = [5,10,15]
num2 = [i**2 if i == 10 else i-5 if i < 7 else i+5 for i in num1]
print(num2)


# In[ ]:


# lets return pokemon csv and make one more list comprehension example
# lets classify pokemons whether they have high or low speed. Our threshold is average speed.
threshold = sum(data.Speed)/len(data.Speed)
data["speed_level"] = ["high" if i > threshold else "low" for i in data.Speed]
data.loc[:10,["speed_level","Speed"]] # we will learn loc more detailed later


# In[ ]:


data = pd.read_csv('../input/pokemon.csv')
data.head()  # head shows first 5 rows


# 

# In[ ]:


data.tail()


# In[ ]:


data.columns


# In[ ]:


data.shape


# In[ ]:


data.info


# In[ ]:


# For example lets look frequency of pokemom types
print(data['Type 1'].value_counts(dropna =False))  # if there are nan values that also be counted
# As it can be seen below there are 112 water pokemon or 70 grass pokemon


# In[ ]:


data.describe() #ignore null entries


# In[ ]:


# For example lets look frequency of pokemom types
print(data['Type 1'].value_counts(dropna =False))  # if there are nan values that also be counted
# As it can be seen below there are 112 water pokemon or 70 grass pokemon


# In[ ]:


data.describe() #ignore null entries


# In[ ]:


# For example: compare attack of pokemons that are legendary  or not
# Black line at top is max
# Blue line at top is 75%
# Red line is median (50%)
# Blue line at bottom is 25%
# Black line at bottom is min
# There are no outliers
data.boxplot(column='Attack',by = 'Legendary')


# TIDY DATA

# In[ ]:


# Firstly I create new data from pokemons data to explain melt nore easily.
data_new = data.head()    # I only take 5 rows into new data
data_new


# In[ ]:


# lets melt LIKE GROUP BY
# id_vars = what we do not wish to melt
# value_vars = what we want to melt
melted = pd.melt(frame=data_new,id_vars = 'Name', value_vars= ['Attack','Defense'])
melted


# In[ ]:


# Index is name
# I want to make that columns are variable
# Finally values in columns are value
melted.pivot(index = 'Name', columns = 'variable',values='value')


# In[ ]:


# Firstly lets create 2 data frame
data1= data.head()
data2= data.tail()
conc_data_row = pd.concat([data1,data2],axis =0,ignore_index =True) # axis = 0 : adds dataframes in row
conc_data_row


# In[ ]:


data1 = data['Attack'].head()
data2= data['Defense'].head()
conc_data_col = pd.concat([data1,data2],axis =1) # axis = 0 : adds dataframes in row
conc_data_col


# In[ ]:


data.dtypes


# In[ ]:


# lets convert object(str) to categorical and int to float.
data['Type 1'] = data['Type 1'].astype('category')
data['Speed'] = data['Speed'].astype('float')

print (type('Type 1'))
print (type('Speed'))


# In[ ]:


# As you can see Type 1 is converted from object to categorical
# And Speed ,s converted from int to float
data.dtypes


# In[ ]:


# Lets chech Type 2
data["Type 2"].value_counts(dropna =False)
# As you can see, there are 386 NAN value


# In[ ]:


# Lets drop nan values
data1=data   # also we will use data to fill missing value so I assign it to data1 variable
data1["Type 2"].dropna(inplace = False)  # inplace = True means we do not assign it to new variable. Changes automatically assigned to data
# So does it work ?


# In[ ]:


assert  data['Type 2'].notnull().all() # returns nothing because we drop nan values


# In[ ]:


data["Type 2"].fillna('empty',inplace = True)


# In[ ]:


assert  data['Type 2'].notnull().all() # returns nothing because we do not have nan values


# In[ ]:


# data frames from dictionary
country = ["Spain","France"]
population = ["11","12"]
list_label = ["country","population"]
list_col = [country,population]
zipped = list(zip(list_label,list_col))
data_dict = dict(zipped)
df = pd.DataFrame(data_dict)
df


# In[ ]:


# Add new columns
df["capital"] = ["madrid","paris"]
df


# In[ ]:


# Broadcasting
df["income"] = 0 #Broadcasting entire column
df


# In[ ]:


VISUAL EXPLORATORY DATA ANALYSIS


# In[ ]:


# Plotting all data 
data1 = data.loc[:,["Attack","Defense","Speed"]]
data1.plot()
# it is confusing


# In[ ]:


# subplots
data1.plot(subplots = True)
plt.show()


# In[ ]:


# scatter plot  
data1.plot(kind = "scatter",x="Attack",y = "Defense")
plt.show()


# In[ ]:


# hist plot  
data1.plot(kind = "hist",y = "Defense",bins = 50,range= (0,250),normed = True)


# In[ ]:


# histogram subplot with non cumulative and cumulative
fig, axes = plt.subplots(nrows=2,ncols=1)
data1.plot(kind = "hist",y = "Defense",bins = 50,range= (0,250),normed = True,ax = axes[0])
data1.plot(kind = "hist",y = "Defense",bins = 50,range= (0,250),normed = True,ax = axes[1],cumulative = True)
plt.savefig('graph.png')
plt


# In[ ]:


data.describe()


# INDEXING PANDAS TIME SERIES

# In[ ]:


time_list = ["1992-03-08","1992-04-12"]
print(type(time_list[1])) # As you can see date is string
# however we want it to be datetime object
datetime_object = pd.to_datetime(time_list)
print(type(datetime_object))


# In[ ]:


# close warning
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


# Now we can select according to our date index
print(data2.loc["1993-03-16"])
print(data2.loc["1992-03-10":"1993-03-16"])


# In[ ]:


data2.resample("A").mean()


# In[ ]:


# Lets resample with month
data2.resample("M").mean()


# In[ ]:



data2.resample("M").mean().interpolate("linear")


# In[ ]:


data = pd.read_csv('../input/pokemon.csv')
data= data.set_index("#")
data.head()


# MANIPULATING DATA FRAMES WITH PANDAS

# In[ ]:


# read data
data = pd.read_csv('../input/pokemon.csv')
data= data.set_index("#")
data.head()


# In[ ]:


data["HP"][1]


# In[ ]:


data.HP[1]


# In[ ]:


data.loc[1,["HP"]]


# In[ ]:


data[["HP","Attack"]]


# 
# SLICING DATA FRAME

# In[ ]:


# Difference between selecting columns: series and dataframes
print(type(data["HP"]))     # series
print(type(data[["HP"]]))   # data frames


# In[ ]:


# Reverse slicing 
data.loc[10:1:-1,"HP":"Defense"] 


# In[ ]:


# From something to end
data.loc[1:10,"Speed":] 


# FILTERING DATA FRAMES

# In[ ]:


# Creating boolean series
boolean = data.HP > 200
data[boolean]


# In[ ]:


# Combining filters IT IS VERY IMPORTANT
first_filter = data.HP > 150
second_filter = data.Speed > 35
data[first_filter & second_filter]


# In[ ]:


# Filtering column based others
data.HP[data.Speed<15]


# TRANSFORMING DATA

# In[ ]:


# Plain python functions
def div(n):
    return n/2
data.HP.apply(div)


# In[ ]:


# Or we can use lambda function
data.HP.apply(lambda n : n/2)


# In[ ]:


# Defining column using other columns
data["total_power"] = data.Attack + data.Defense
data.head()


# In[ ]:


INDEX OBJECTS AND LABELED DATA


# In[ ]:


# our index name is this:
print(data.index.name)
# lets change it
data.index.name = "index_name"
data.head()


# In[ ]:


# Overwrite index
# if we want to modify index we need to change all of them.
data.head()
# first copy of our data to data3 then change index 
data3 = data.copy()
# lets make index start from 100. It is not remarkable change but it is just example
data3.index = range(100,900,1)
data3.head()


# 
# HIERARCHICAL INDEXING

# In[ ]:


# lets read data frame one more time to start from beginning
data = pd.read_csv('../input/pokemon.csv')
data.head()


# In[ ]:


# Setting index : type 1 is outer type 2 is inner index
data1 = data.set_index(["Type 1","Type 2"]) 
data1.head(100)


# PIVOTING DATA FRAMES

# In[ ]:


dic = {"treatment":["A","A","B","B"],"gender":["F","M","F","M"],"response":[10,45,5,9],"age":[15,4,72,65]}
df = pd.DataFrame(dic)
df


# In[ ]:



df.pivot(index="treatment",columns = "gender",values="response")


# STACKING and UNSTACKING DATAFRAME

# In[ ]:


df1 = df.set_index(["treatment","gender"])
df1
# lets unstack it


# In[ ]:


# level determines indexes
df1.unstack(level=0)


# In[ ]:


# change inner and outer level index position
df2 = df1.swaplevel(0,1)
df2


# In[ ]:


df1.unstack(level=0)


# In[ ]:


df1.unstack(level=1)


# In[ ]:


# change inner and outer level index position
df2 = df1.swaplevel(0,1)
df2


# MELTING DATA FRAMES

# In[ ]:


df


# In[ ]:


pd.melt(df,id_vars="treatment",value_vars=["age","response"])


# CATEGORICALS AND GROUPBY

# In[ ]:


df.groupby("treatment").mean()   # mean is aggregation / reduction method


# In[ ]:


# we can only choose one of the feature
df.groupby("treatment").age.max() 


# In[ ]:


# Or we can choose multiple features
df.groupby("treatment")[["age","response"]].min() 


# In[ ]:


df.info()

