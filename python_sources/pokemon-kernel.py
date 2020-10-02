#!/usr/bin/env python
# coding: utf-8

# INTRODUCTION TO DATA SCIENCE

# A. Matplotlib

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


data = pd.read_csv('../input/pokemon-challenge/pokemon.csv')


# In[ ]:


data.info()


# In[ ]:


data.corr()


# In[ ]:


f, ax = plt.subplots(figsize=(15,15))
sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt= '.1f', ax=ax)
plt.show()


# In[ ]:


data.head()


# In[ ]:


#Line Plot
data.Speed.plot(kind = 'line', color = 'g', label = 'Speed', linewidth = 1, alpha = 0.5, grid = True, linestyle = ':')
data.Defense.plot(color = 'r', label = 'Defense', linewidth = 1, alpha = 0.5, grid = True, linestyle = '-.')
plt.legend(loc='upper right')
plt.xlabel('x axis')
plt.ylabel('y axis')
plt.title('Line Plot')
plt.show()


# In[ ]:


#Scatter Plot
#x=Attack ,y=Defense
data.plot(kind='scatter', x='Attack', y='Defense', alpha=0.5, color = 'red')
plt.xlabel('Attack')
plt.ylabel('Defense')
plt.title('Attack Defense Scatter Plot') #title of plot
plt.show()


# In[ ]:


#Histogram
#bins = number of bar in figure
data.Speed.plot(kind = 'hist', bins = 50, figsize = (8, 8))
plt.show()


# In[ ]:


# clf() = cleans it up again you can start a fresh 
data.Speed.plot(kind = 'hist', bins = 50)
plt.clf()


# B. DICTIONARIES
# #This section is left blank because the pokemon dataset is not suitable for dictionaries

# C.PANDAS

# In[ ]:


#There is two type libraries in Pandas..

#-Seriler = data[]
#-Dataframe = data[[]]
series = data['Defense'] 
print(type(series))
data_frame = data[['Defense']] 
print(type(data_frame))


# In[ ]:


#Filtering Pandas Data Frame
x = data['Defense']>200 #There are only 3 pokemons who have higher defense value than 200
data[x]


# In[ ]:


#Filtering Pandas with logical_and
#There are only 2 pokemons who have higher defense value than 200 and higher attack value than 100
data[np.logical_and(data['Defense']>200, data['Attack']>100)]


# In[ ]:


#For loop
for index, value in data[['Attack']][0:2].iterrows():
    print(index, " : ", value)


# In[ ]:


# List comprehension example with Pandas.
threshold = sum(data.Speed)/len(data.Speed)
print('threshold',threshold)
data["speed_level"] = ["high" if i > threshold else "low" for i in data.Speed]
data.loc[:10, ["speed_level", "Speed"]]


# 3.CLEANING DATA

# In[ ]:


data = pd.read_csv('../input/pokemon-challenge/pokemon.csv')
data.head()


# In[ ]:


data.columns


# In[ ]:


data.shape


# In[ ]:


data.info()


# EXPLORATARY DATA ANALYSIS (EDA)
# 

# value_counts(): counts the frequency
# outliers : the value that is considerably higher or lower from rest of the data
# * Lets say value at 75% is Q3 and value at 25% is Q1.
# * Outliers are smaller than Q1 - 1.5(Q3-Q1) and bigger than Q3 + 1.5(Q3-Q1).(Q3-Q1) = IQR
#   We will use describe() method. Describe method includes :
# * count: number of entries
# * mean: average of entries
# * std: standart deviation
# * min: minimum entry
# * 25%: first quantile
# * 50%: median or second quantile
# * 75%: third quantile
# * max: maximum entry
# 
# 

# In[ ]:


print(data['Type 1'].value_counts(dropna=False))


# In[ ]:


data.describe() # ignore null entries


# VISUAL EXPLORATORY DATA ANALYSIS
# 
# * Box plots : visualize basic statistics like outliers, min/max or quantiles

# In[ ]:


data.boxplot(column='Attack', by = 'Legendary')
plt.show()


# TIDY DATA
# 
# * We tidy data with melt() .

# In[ ]:


data_new = data.head()
data_new


# In[ ]:


#melting
melted = pd.melt(frame=data_new, id_vars = 'Name', value_vars = ['Attack', 'Defense'])
melted


# PIVOTING DATA
# * Reverse of melting

# In[ ]:


melted.pivot(index='Name', columns = 'variable', values='value')


# CONCATENATING DATA
# * We can concatenate two dataframes 

# In[ ]:


data1 = data.head()
data2 = data.tail()
conc_data_row = pd.concat([data1, data2], axis=0, ignore_index=True)
conc_data_row


# In[ ]:


data3 = data['Attack'].head()
data4 = data['Defense'].head()
conc_data_col = pd.concat([data3, data4], axis=1)
conc_data_col


# DATA TYPES
# * There are 5 basic data types:
#   * object(string)
#   * boolean
#   * integer
#   * float
#   * categorical

# In[ ]:


data.dtypes


# In[ ]:


#Lets convert object to categorical and int to float
data['Type 1'] = data['Type 1'].astype('category')
data['Speed'] = data['Speed'].astype('float')


# In[ ]:


data.dtypes


# MISSING DATA AND TESTING WITH ASSERT
# 

# In[ ]:


data.info()


# In[ ]:


#Lets check Type 2
data["Type 2"].value_counts(dropna=False)
#There are 386 NAN values


# In[ ]:


#Lets drop NAN values
data1 = data
data1["Type 2"].dropna(inplace = True)


# In[ ]:


assert 1==1


# In[ ]:


assert data['Type 2'].notnull().all() 
#return error because it is false


# In[ ]:


assert data["Type 2"].notnull().all()
#returns nothing bacause we drop all NAN values


# In[ ]:


data["Type 2"].fillna('empty', inplace = True)


# In[ ]:


assert data.columns[1] == 'Name'


# PANDAS FOUNDATION

# VISUAL EXPLORATORY DATA ANALYSIS
# * Plot
# * Subplot
# * Histogram

# In[ ]:


# Plotting all data 
data1 = data.loc[:,["Attack","Defense","Speed"]]
data1.plot()


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


fig, axes = plt.subplots(nrows=2,ncols=1)
data1.plot(kind = "hist",y = "Defense",bins = 50,range= (0,250),normed = True,ax = axes[0])
data1.plot(kind = "hist",y = "Defense",bins = 50,range= (0,250),normed = True,ax = axes[1],cumulative = True)
plt.savefig('graph.png')
plt


# STATISTICAL EXPLORATORY DATA ANALYSIS

# In[ ]:


data.describe()


# INDEXING PANDAS TIME SERIES
# * datetime = object
# * parse_dates(boolean): Transform date to ISO 8601 (yyyy-mm-dd hh:mm:ss ) format

# In[ ]:


time_list = ["1992-03-08","1992-04-12"]
print(type(time_list[1])) 
datetime_object = pd.to_datetime(time_list)
print(type(datetime_object))


# In[ ]:


import warnings
warnings.filterwarnings("ignore")
data2 = data.head()
date_list = ["1992-01-10","1992-02-10","1992-03-10","1993-03-15","1993-03-16"]
datetime_object = pd.to_datetime(date_list)
data2["date"] = datetime_object
data2= data2.set_index("date")
data2 


# In[ ]:


print(data2.loc["1993-03-16"])
print(data2.loc["1992-03-10":"1993-03-16"])


# RESAMPLING PANDAS TIME SERIES
# 

# In[ ]:


data2.resample("A").mean()


# In[ ]:


#Lets resample with month
data2.resample("M").mean()
# As you can see there are a lot of nan because data2 does not include all months


# In[ ]:


data2.resample("M").mean().interpolate("linear")


# MANIPULATING DATA FRAMES WITH PANDAS

# INDEXING DATA FRAMES
# * Indexing using square brackets
# * Using column attribute and row label
# * Using loc accessor
# * Selecting only some columns

# In[ ]:


# read data
data = pd.read_csv('../input/pokemon-challenge/pokemon.csv')
data= data.set_index("#")
data.head()


# In[ ]:


#indexing using square bracket
data["HP"][1]


# In[ ]:


#using column attribute and row label
data.HP[1]


# In[ ]:


#using loc accessor
data.loc[1,["HP"]]


# In[ ]:


#Selecting only some columns
data[["HP","Attack"]]


# SLICING DATA FRAME

# In[ ]:


print(type(data["HP"]))     # series
print(type(data[["HP"]]))   # data frames


# In[ ]:


#slicing and indexing series
data.loc[1:10,"HP":"Defense"]


# In[ ]:


# Reverse slicing 
data.loc[10:1:-1,"HP":"Defense"] 


# In[ ]:


#from something to end
data.loc[1:10,"Speed":]


# FILTERING DATA FRAMES

# In[ ]:


#creating boolean series
boolean = data.HP > 200
data[boolean]


# In[ ]:


#combining filters
first_filter = data.HP > 150
second_filter = data.Speed > 35
data[first_filter & second_filter]


# In[ ]:


#filtering column based others
data.HP[data.Speed<15]


# TRANSFORMING DATA

# In[ ]:


#plain python functions
def div(n):
    return n/2
data.HP.apply(div)


# In[ ]:


#we can also use lambda function
data.HP.apply(lambda n : n/2)


# In[ ]:


#defining column using other columns
data["total_power"] = data.Attack + data.Defense
data.head()


# INDEX OBJECTS AND LABELED DATA
# * index: sequence of label

# In[ ]:


print(data.index.name)
data.index.name = "index_name"
data.head()


# In[ ]:


# Overwrite index
data.head()
data3 = data.copy()
data3.index = range(100,900,1)
data3.head()


# HIERARCHICAL INDEXING

# In[ ]:


data = pd.read_csv('../input/pokemon-challenge/pokemon.csv')
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


# pivoting
df.pivot(index="treatment",columns = "gender",values="response")


# STACKING and UNSTACKING DATAFRAME

# In[ ]:


df1 = df.set_index(["treatment","gender"])
df1


# In[ ]:


# level determines indexes
df1.unstack(level=0)


# In[ ]:


df1.unstack(level=1)


# In[ ]:


#change inner and outer level index position
df2 = df1.swaplevel(0,1)
df2


# 
# MELTING DATA FRAMES
# * Reverse of pivoting

# In[ ]:


df


# In[ ]:


pd.melt(df,id_vars="treatment",value_vars=["age","response"])


# CATEGORICALS AND GROUPBY

# In[ ]:


#we will use df
df


# In[ ]:


df.groupby("treatment").mean()   # mean is aggregation / reduction method


# In[ ]:


#we can only choose one of the feature
df.groupby("treatment").age.max() 


# In[ ]:


#Or we can choose multiple features
df.groupby("treatment")[["age","response"]].min() 


# In[ ]:


df.info()


# CONCLUSION
