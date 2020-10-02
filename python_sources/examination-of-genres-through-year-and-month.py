#!/usr/bin/env python
# coding: utf-8

# # Examination about released genres through year in Movie Dataset
# 

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


import json
import pprint
pp = pprint.PrettyPrinter(indent=2,width=80)
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')


# 
# first of all we read our input datas in two meaningful variable

# In[ ]:


credits=pd.read_csv("../input/tmdb_5000_credits.csv")
movies=pd.read_csv("../input/tmdb_5000_movies.csv")


# i have a helper function for a banner like
# ```python
# ################################################################################
# #                                 CREDITS INFO                                 #
# ################################################################################
# ```

# In[ ]:


from termcolor import colored
def ht_title(title='',length=80, color='blue'):
    """
    helper
    hashtag title function
    this function is print title in center of print(length*"#")
    title:  Message what you say
    length: Column number
    
    """
    if length < 2:
        raise Exception("length must greater than 2")
    len_of_title = len(title)            
    if len_of_title == 0:
        left_space=length/2
        rigth_space=length/2
    elif len_of_title < 0 or len_of_title > length - 2:
        raise Exception("Out of range")
    elif len_of_title % 2 == 0:
        left_space=(length - len_of_title) / 2
        if length % 2==0:
            rigth_space=left_space
        else:
            rigth_space=left_space+1
    elif len_of_title % 2 == 1:
        left_space=(length - len_of_title) / 2
        if length % 2 == 0:
            rigth_space=left_space+1
        else:
            rigth_space=left_space
    print(colored(length*"#", color))
    print(colored("#", color),end='')
    print((int(left_space)-1)*" ",end='')
    print(colored(title.upper(), color),end='')
    print((int(rigth_space)-1)*" ",end='')
    print(colored("#",color))
    print(colored(length*"#",color))
        


# quick test for our banner

# In[ ]:


#ht_title(length=1)
ht_title('OK!',length=10)
ht_title('OK',length=9,color='yellow')
ht_title('OK',length=10,color='green')
ht_title('OK!',length=9,color='red')


# ## GENERAL INFORMATION
#   we can intuitively understand our datas with pandas info() , describe() functions and dtypes, columns features in a general manner
#   you can find output of those functions below
# 

# In[ ]:


ht_title("credits info")
pp.pprint(credits.info())


# In[ ]:


spp = pprint.PrettyPrinter(indent=2,width=30)
ht_title("credits features")
spp.pprint(list(credits.columns))


# In[ ]:


ht_title("credits dtypes")
pp.pprint(credits.dtypes)


# In[ ]:


ht_title("credits describe")
credits.describe()


# In[ ]:


ht_title("movies info")
pp.pprint(movies.info())


# In[ ]:


ht_title("movies features")
pp.pprint(list(movies.columns))


# In[ ]:


ht_title("movies dtypes")
pp.pprint(movies.dtypes)


# In[ ]:


ht_title("movies describe")
movies.describe()


# In[ ]:


movies["release_month"]=[str(each)[5:7] if str(each) != "nan" else "00" for each in movies.release_date]
movies["release_month_int"]=[int(each) for each in movies.release_month]

movies["release_year"]=[str(each)[0:4] if str(each) != "nan" else "1915" for each in movies.release_date]#1915 is nan value 
movies["release_year_int"]=[int(each) for each in movies.release_year]
movies.head()


# how many movies are released through year or month?
# we can learn that information with a histogram
# 
# we can use that information as a verification about "genres through year or month" graphic because sum of genre graphics must be equal the all relased movies at the end
# 

# In[ ]:


plt.figure(figsize=(24,6))
plt.title("Histogram of released movies through year")
plt.grid(True)
plt.xticks(range(1915,2021),rotation=90)
plt.yticks(range(0,251,10),rotation=25)
plt.xlabel('year')
plt.ylabel('movie count')
plt.hist(movies.release_year_int,bins=100)
plt.show()


# 

# In[ ]:


months=['January', 'Feburary', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
plt.figure(figsize=(24,6))
plt.title("Histogram of released movies through month")
plt.grid(True)
plt.xticks(range(1,13),months,rotation=90)
plt.yticks(rotation=25)
plt.xlabel('month')
plt.ylabel('movie count')
plt.hist(movies.release_month_int,bins=18)
plt.show()


# ## Brave New Table
# unluckily movies table field genres doesn't contain atomic data(it contains json) because of that  
# now we transform our movies table to genres table (denormalize table of genres) and we process genres table instead of movies table
# 

# In[ ]:


genre_tbl=pd.DataFrame({"genre":[],"original_title":[],"release_year":[],"release_month":[]})
index=1
for i,each in enumerate(movies.genres):
    genre_list=json.loads(each)
    for genre in genre_list:
        genre_tbl=pd.concat([genre_tbl,pd.DataFrame({
            "genre":genre["name"],
            "original_title":movies.original_title[i],
            "release_year":movies.release_year[i],
            "release_month":movies.release_month[i]
        },index=[index])],axis=0)
        index+=1


# In[ ]:


genre_tbl.head()


# In[ ]:


genre_tbl.tail()


# ## Query Parameters
# Before querying genre table with (year & genre) filter and (month & genre) filter we need to know what are the unique (year & genre) pairs and unique (month & genre) pairs

# In[ ]:


x=genre_tbl.genre.unique()
y=genre_tbl.release_year.sort_values().unique()
z=genre_tbl.release_month.sort_values().unique()
pairs_of_year=np.transpose([np.tile(x, len(y)), np.repeat(y, len(x))])
pairs_of_month=np.transpose([np.tile(x, len(z)), np.repeat(z, len(x))])



# In[ ]:


pairs_of_year


# In[ ]:


pairs_of_month


# ## Querying
# 
# Now we can query our genre_tbl table and i prefer results in a dictionary which is use genre as key and [(year, released_movie_count),] pairs list as value
# after that only one step behind

# In[ ]:


year_hist={}
for genre,year in pairs_of_year:
    if genre not in year_hist:
        year_hist[genre]=[]
    year_hist[genre].append((year,genre_tbl[(genre_tbl.genre==genre) & (genre_tbl.release_year==year)].count()[0]))
#year_hist


# In[ ]:


month_hist={}
for genre,month in pairs_of_month:
    if genre not in month_hist:
        month_hist[genre]=[]
    month_hist[genre].append((month,genre_tbl[(genre_tbl.genre==genre) & (genre_tbl.release_month==month)].count()[0]))
#month_hist


# ## Plotting
# Now we have histograms of genres we can use it and plotting our values

# In[ ]:


plt.figure(figsize=(24,6))
plt.title("released movie genres through year")
c=3
for genre,values in year_hist.items():
    x=[]
    y=[]
    c=+20
    for i in values:
        x.append(i[0])
        y.append(i[1])
    plt.plot(x,y,label=str(genre))

plt.legend()
plt.grid(True)
plt.yticks(range(0,131,10))
plt.xticks(rotation=90)
plt.ylabel("how many movies are released")
plt.xlabel("year")
plt.show()


# ## Genres Pie Chart
# According these datas all these years drama and comedy often has max release count you can see it in all years sums in below pie chart you can differ likely colors with pie charts values for example Romance> Adventure in general

# In[ ]:


plt.figure(figsize=(12,12))
plt.title("released movie genres pie chart in all years")
c=3
x=[]
y=[]
for genre,values in year_hist.items():
    x.append(genre)
    s=0
    for i in values:
        s+=i[1]
    y.append(s)
v=[(i/sum(y))*100 for i in y]
explode=(0,0,0,0,0,0.1,0,0,0,0,0.1,0,0,0,0,0,0,0,0,0)

plt.pie(v,explode=explode,labels=x,autopct='%1.1f%%',)
plt.show()


# In[ ]:


plt.figure(figsize=(24,6))
plt.title("released movie genres through month")
c=3
for genre,values in month_hist.items():
    x=[]
    y=[]
    c=+20
    for i in values:
        x.append(i[0])
        y.append(i[1])
    plt.plot(x,y,label=str(genre))
months=['January', 'Feburary', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']

plt.legend()
plt.grid(True)
plt.yticks()
plt.xticks(range(1,13),months,rotation=90)
plt.ylabel("how many movies are released")
plt.xlabel("month")
plt.show()


# ## Conclusion
# 
# After 1994 there is a quite increase in film industry. And drama, comedy, thriller, action films responsible nearly half of it. That information give us what are the most produceable topics according to film producers.
# 
#   seemingly we can always watch some drama & comedy no matter what the year is.  Ancient greek's tragedy&comedy or modern world's drama&comedy
# 
# 
# ### Interesting years
# * 2003 and 2012 in these years comedy movies and drama movies are equal which is possible with decrease in drama 
# * 2006 Begining of the Fall: weirdly after 2006 there is a quite noticeable fall in release graph [in this link](https://www.telegraph.co.uk/culture/film/10620201/What-has-happened-to-all-the-Hollywood-films.html) there is an explanation with budget costs if you interested in
# 
# 
