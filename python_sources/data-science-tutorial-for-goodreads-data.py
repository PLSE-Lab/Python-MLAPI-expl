#!/usr/bin/env python
# coding: utf-8

# **About this file**
# 
# This file contains the detailed information about the books, primarily. Detailed description for each column can be found alongside.

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

# Any results you write to the current directory are saved as output.


# Import dataset

# In[ ]:


data=pd.read_csv("../input/books.csv",error_bad_lines=False)
data.info()


# In[ ]:


data.describe()


# In[ ]:


data.head()


# In[ ]:


data.corr()


# In[ ]:


#correlation map
f, ax= plt.subplots(figsize=(10,10))
sns.heatmap(data.corr(),annot=True, linewidths=.5,fmt=".1f",ax=ax )
plt.show()


# In[ ]:


data.columns=[each.split()[0]+"_"+each.split()[1] if (len(each.split())>1) else each for each in data.columns]


# Replace the " " with "_" in column names

# In[ ]:


x= data["#_num_pages"] 


# Return the max_page_number book

# In[ ]:


print([data[data["#_num_pages"].max()==data["#_num_pages"]]])


# Return the max_ratings_count book

# In[ ]:


print(data[data.ratings_count.max()==data.ratings_count])


# In[ ]:


array=["ratings_count"]


# In[ ]:


data.groupby("language_code").mean().sort_values("average_rating",ascending = False)[array]


# In[ ]:


# line plot
data.plot(kind='scatter',x="#_num_pages",y="text_reviews_count",color='b',label='Number of pages', linewidth=1, alpha=0.9, 
grid=True, linestyle='-.')
#data.plot(kind='scatter',color='r',label='Average Rating', linewidth=1, alpha=0.9, 
#grid=True, linestyle='-')
plt.legend(loc='upper right')    
plt.xlabel('Number of Pages')
plt.ylabel('Text_Review')
plt.title('Number of Pages-Average Rating')
plt.show()


# In[ ]:


data.language_code.unique()


# In[ ]:


dictionary=(['eng', 'en-US', 'spa', 'fre', 'en-GB', 'mul', 'ger', 'ara', 'por',
       'grc', 'en-CA', 'enm', 'jpn', 'dan', 'zho', 'nl', 'ita', 'lat',
       'srp', 'rus', 'tur', 'msa', 'swe', 'glg', 'cat', 'wel', 'heb',
       'nor', 'gla', 'ale'])


# In[ ]:


x = data['average_rating']>4.9

data[x]
# In[ ]:


e=0
t=0
for i in data.language_code:
    if i=="eng":
        e+=1
    elif i=="tur":
        t+=1
lang_code={"English":e,"Turkish":t}
print(lang_code)


# In[ ]:


data.title[data.language_code=="eng"].count()


# **Retrieve multicolumns meet the conditions**

# In[ ]:


data[["authors","title"]][(data["language_code"]=="eng") & (data["average_rating"]>4.8) & (data["#_num_pages"]>500)]


# In[ ]:


data_groupby=data.groupby("language_code")
data2=data_groupby.mean().sort_values(by="average_rating",ascending=False).head()


# In[ ]:


type(data2)


# **Return average ratings of language codes**

# In[ ]:


data2.average_rating


# In[ ]:


data3=data[data.language_code=="eng"]


# In[ ]:


data3.count()


# In[ ]:


data3.plot(kind="scatter",x="average_rating",y="#_num_pages",color="blue",alpha=0.7)
plt.title("Num_pages - Average_ratings")


# In[ ]:


data3["#_num_pages"].plot(kind = 'hist',bins = 1000,figsize = (5,5))
plt.show()


# **Retrieve the max page number book name**

# In[ ]:


maximum = data["#_num_pages"].max()
data[(data["#_num_pages"] == maximum)].title


# In[ ]:


#retrieve the 5 best average rating book
maximum = data["average_rating"].max()
data[(data["average_rating"] == maximum)].title.head(5)


# In[ ]:


print(maximum)


# **Retrieve the page number of given book name**

# In[ ]:


data[data["title"]=="The Complete Aubrey/Maturin Novels (5 Volumes)"]["#_num_pages"]


# **Iterate the values of some specific range**

# In[ ]:


for index , value in data[data["average_rating"]>3.5][130:140].iterrows():
    print(index," : ", value)


# **retrive an info of specific language books**

# In[ ]:


print(data[data['language_code']=="tur"].info())


# Retrieve the number each values in column

# In[ ]:


data["language_code"].value_counts().sort_values(ascending=False)


# **Data analyse with Correleation heat Map**

# In[ ]:


corr = data.corr()
data_num = data._get_numeric_data()
sns.set(font_scale=1.2)
f,ax = plt.subplots(figsize=(10,10))
sns.heatmap(data.corr(), annot= True, linewidths= .5, fmt= '.1f', ax= ax,cmap="YlGnBu")
plt.show()


# **Column rename**

# In[ ]:


data.rename(columns={'#_num_pages':'number_of_pages'},inplace=True)


# In[ ]:


data.head()


# **Top 100 average_rating books**

# In[ ]:


top = data[data.average_rating>4.8].sort_values(by='average_rating',ascending=False).head(100)
top.plot('title','average_rating',kind='bar',figsize=(30,10))

plt.xlabel('Book Title')
plt.ylabel('Average Rating')
plt.title('100 Best Average Rating Books')


# In[ ]:


data.head(100).plot(kind='scatter', x='number_of_pages', y="average_rating",alpha = .8,color = 'blue',figsize= (6,6))
plt.legend()
plt.xlabel('Language Code')             
plt.ylabel("average Rating")
plt.title('Scatter Plot') 
plt.show()


# In[ ]:


threshold=int(sum(data.number_of_pages)/len(data.number_of_pages))
print("Average number of page: ",threshold)
data["thickness"]=["thick" if i>=threshold else "thin" for i in data.number_of_pages]
data.loc[:10, ["number_of_pages","thickness"]]


# # THAT'S ALL

# In[ ]:




