#!/usr/bin/env python
# coding: utf-8

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


# ## **Data Wrangling**     
# <br>
# > We will be using the relevant data only and removing the unused data
# </br>

# In[ ]:


df = pd.read_csv("../input/tmdb_movies_data.csv")
df.info()


# In[ ]:


df.head()


# In[ ]:


df.tail(2)


# ![](http://)**Let's Observe this DataSet**
# <br>
# by getiing summary of the dataset</br>

# In[ ]:


df.describe()


# In[ ]:


#this dataset contains null value also 
#lets find out the null value
df.isna().sum()


# In[ ]:


#fill the null values with 0
df.fillna(0)


# **Data Cleaning**

# In[ ]:


#finding duplicate rows
sum(df.duplicated())


# In[ ]:


#Dropping duplicate row
df.drop_duplicates(inplace = True)
#After Dropping
print("After Dropping Table(Rows, Col) :", df.shape)


# In[ ]:


#Changing Format Of Release Date Into Datetime Format
df['release_date'] = pd.to_datetime(df['release_date'])


# In[ ]:


df['release_date'].head()


# **Removing column which are not going to be used**

# In[ ]:


#lets remove these coloumn
df.drop(['budget_adj','revenue_adj','overview','imdb_id','homepage','tagline'],axis =1,inplace = True)


# In[ ]:


print("(rows,cols): ",df.shape)


#  **Dropping rows which contain incorrect or inappropriate values**

# In[ ]:


#checking with 0 values in budget and revenew columns
print("Budget col having 0 value :",df[(df['budget']==0)].shape[0])
print("Revenue col having 0 value :", df[(df['revenue']==0)].shape[0])


# **Exploratory Data Analysis**

# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt


# In[ ]:


# Counting the number of movies in each year 
data = df.groupby('release_year').count()['id']
print(data.tail())


# In[ ]:


#grouping the data a/q to release year and counting movies

sns.set(rc={'figure.figsize':(10,5)})
sns.set_style("darkgrid")
df.groupby('release_year').count()['id'].plot(xticks = np.arange(1960,2016,5))
plt.title("Year Vs Number Of Movies",fontsize = 14)
plt.xlabel('Release year',fontsize = 13)
plt.ylabel('Number Of Movies',fontsize = 13)


# **Let's Find out Which movies has highest or lowest profit**

# In[ ]:


df['Profit'] = df['revenue'] - df['budget']
def find_minmax(x):
    #use the function 'idmin' to find the index of lowest profit movie.
    min_index = df[x].idxmin()
    #use the function 'idmax' to find the index of Highest profit movie.
    high_index = df[x].idxmax()
    high = pd.DataFrame(df.loc[high_index,:])
    low = pd.DataFrame(df.loc[min_index,:])
    
    #print the movie with high and low profit
    print("Movie Which Has Highest "+ x + " : ",df['original_title'][high_index])
    print("Movie Which Has Lowest "+ x + "  : ",df['original_title'][min_index])
    return pd.concat([high,low],axis = 1)

#call the find_minmax function.
find_minmax('Profit')


# In[ ]:


#make a plot which contain top 10 movies which earn highest profit.
#sort the 'Profit' column in decending order and store it in the new dataframe,
info = pd.DataFrame(df['Profit'].sort_values(ascending = False))
info['original_title'] = df['original_title']
data = list(map(str,(info['original_title'])))
x = list(data[:10])
y = list(info['Profit'][:10])

#make a plot usinf pointplot for top 10 profitable movies.
ax = sns.pointplot(x=y,y=x)

#setup the figure size
sns.set(rc={'figure.figsize':(10,5)})
#setup the title and labels of the plot.
ax.set_title("Top 10 Profitable Movies",fontsize = 15)
ax.set_xlabel("Profit",fontsize = 13)
sns.set_style("darkgrid")


# **Movies with highest and lowest budget**

# In[ ]:


df['budget'] = df['budget'].replace(0,np.NAN)
find_minmax('budget')

