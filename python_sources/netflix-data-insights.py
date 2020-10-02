#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


data=pd.read_csv("/kaggle/input/netflix-shows/netflix_titles.csv")


# In[ ]:


print(data.columns)
data.head(15)


# In[ ]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns

data1=data[data['release_year']>=2000]
data1m=data1[data1['type']!="TV Show"]
data1t=data1[data1['type']!="Movie"]
data2m=pd.DataFrame(data1m['release_year'].value_counts()).reset_index()
data2m.rename(columns={'release_year':'count'},inplace=True)
movies=pd.DataFrame(['Movie']*data2m.shape[0],columns=['type'])
data2m=pd.concat([data2m,movies],axis=1)
data2t=pd.DataFrame(data1t['release_year'].value_counts()).reset_index()
data2t.rename(columns={'release_year':'count'},inplace=True)
Tv_shows=pd.DataFrame(['TV Shows']*data2t.shape[0],columns=['type'])
data2t=pd.concat([data2t,Tv_shows],axis=1)
data_final=pd.concat([data2m,data2t],ignore_index=True)
data_final.rename(columns={'index':'Release year'},inplace=True)
plt.figure(figsize=(16,6))

plt.title("Bargraph comparing the number of Movies and Tv shows from the year 2000 to 2020 ")

sns.barplot(x=data_final['Release year'],y=data_final['count'],hue=data_final['type'])


# From the above Bar Plot, it is clear that the number of movies increased until 2017 but decreased gradually after that. where as, the number of Tv shows in netflix increased over years. This might be due to the fact that people are showing more interest in Tv shows now-a-days and Netflix is able to get more subscriptions by producing more Tv shows when compared to the subscriptions they are getting through Movies.

# In[ ]:


plt.figure(figsize=(30,15))
datafiltered=data1[data1['type']=='Movie']
datafilteredtv=data1[data1['type']=='TV Show']
dataidk=datafiltered['country'].str.get_dummies(',')
dataidk1=datafilteredtv['country'].str.get_dummies(',')

count={}
count1={}

for col in dataidk.columns:
    col_= col.strip()
    if col_ in count.keys():
        count[col_]=count[col_]+dataidk[col].sum()
    else:
        
   
        count[col_]=dataidk[col].sum()
    
for col in dataidk1.columns:
    
    col_= col.strip()
    if col_ in count1.keys():
        count1[col_]=count1[col_]+dataidk1[col].sum()
    else:
        
   
        count1[col_]=dataidk1[col].sum()

datacon=list(count.items())


datacon=pd.DataFrame(datacon,columns=['country','count'])
datacon=datacon[datacon['count']>50].reset_index()

datacon1=list(count1.items())
datacon1=pd.DataFrame(datacon1,columns=['country','count'])
datacon1=datacon1[datacon1['count']>50].reset_index()

tvshowsapp=pd.DataFrame(['TV Show']*datacon1.shape[0],columns=['type'])
datacon1=pd.concat([datacon1,tvshowsapp],axis=1)
datacon1=datacon1.drop('index',axis=1)
movieapp=pd.DataFrame(['Movie']*datacon.shape[0],columns=['type'])
datacon=pd.concat([datacon,movieapp],axis=1)
datacon=datacon.drop('index',axis=1)
finaldatacon=pd.concat([datacon,datacon1],axis=0,ignore_index=True)
sns.set_context("paper", font_scale=1.4)
plt.title("BarPlot showing the number of Movies and Tv shows that are alteast greater than 50, produced by each country over 20 years, from 2000 to 2020")
sns.barplot(x=finaldatacon['country'],y=finaldatacon['count'],hue=finaldatacon['type'])


#  The above graph shows the number of Movies and Tv shows that each country has produced in Netflix that are atlest greater than 50, over 20 years span. From the graph, it is obvious that Movies and Tv shows based on United States are more in number compared to any other country.One of the sensible reason for this behavious could be that the Netflix was founded in United States. We can observe that Argentina,Australia,china,Germany,Hong Kong,mexico, spain, turkey and Belgium didn't produce any Tv show  or might produced shows less than 50 and South Korea,Taiwan didnt produce any Movie or might produced movies less than 50 till now from past 20 years. Even from India, Movies are dominating Tv shows.

# In[ ]:



   

plt.figure(figsize=(26,7))


plt.title("Directors with atleast seven movies on Netflix")

directorsbigcolumn=datafiltered['director'].str.get_dummies(','+'')
let={}
for col in directorsbigcolumn:
    let[col]=directorsbigcolumn[col].sum()
dataone=list(let.items())
finaldirect=pd.DataFrame(dataone,columns=['director','Number of Movies'])
finaldirect=finaldirect[finaldirect['Number of Movies']>=7]
sns.barplot(x=finaldirect['director'],y=finaldirect['Number of Movies'])


# From the above plot, it seems that Jan Sulter and Raul Campos have directed more movies than any other director that are added to netflix in the span of 20 years.

# 

# In[ ]:


plt.figure(figsize=(26,7))


plt.title("Directors with atleast two Tv shows on Netflix")

directorsbigcolumn1=datafilteredtv['director'].str.get_dummies(','+' ')
let1={}
for col in directorsbigcolumn1:
    let1[col]=directorsbigcolumn1[col].sum()
dataone1=list(let1.items())


finaldirect1=pd.DataFrame(dataone1,columns=['director','Number of Tv Series'])
finaldirect1=finaldirect1[finaldirect1['Number of Tv Series']>=2]

sns.barplot(x=finaldirect1['director'],y=finaldirect1['Number of Tv Series'])


# 

# 

# In[ ]:




