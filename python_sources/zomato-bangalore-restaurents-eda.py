#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import plotly.plotly as py
import plotly.graph_objs as go
import re
import sklearn
import ast
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords 
from collections import Counter 

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# ### Reading and cleaning the data

# In[ ]:


df= pd.read_csv("../input/zomato.csv")
df = df.dropna(subset=['rate'])  # Drop Nulls 


# ### Understanding and Cleaning Data

# In[ ]:


df.info()


# Removing Nulls

# In[ ]:


df.dropna(inplace=True)


# In[ ]:


df.rate.unique()


# - We can see Unwanted strings like "NEW","-" etc. And nulls too
# - We dont need the "/5" characters for our future visualizations so removing them

# In[ ]:


df=df[df.rate.apply(lambda x: len(str(x)))>=5]  #clean the rating remove unwanted symbols
def ext(strings):
    m=re.findall(r"\d\.\d",strings)
    m=float(m[0])
    return m
df.rate=df["rate"].apply(ext)
df.rate.unique()


# #### The approximate cost for two people.

# In[ ]:


df["approx_cost(for two people)"].unique()


# - We can realise that "," is used to measure thousands.
# - we need to remove "," as machine does not understand them as numeric.

# In[ ]:


def cltn(m):
    if len(m)>100000:
        return 0
    else:
        m.replace(",","")
        ns=''
        for each in m:
            if each!=",":
                ns+=each
        return(float(ns))
df["approx_cost(for two people)"]=df["approx_cost(for two people)"].apply(cltn)


# In[ ]:


df["approx_cost(for two people)"].unique()


# ### Are no. of Votes and Ratings dependent ?

# In[ ]:


plt.figure(figsize=(20,10))
df3=df[(df.votes>=4000)&(df.votes<12500)]
plt.scatter(df3.rate,df3.votes,color="green")
df2=df[(df.votes>=df.votes.mean())&(df.votes<4000)]
plt.scatter(df2.rate,df2.votes,color="yellow")
df1=df[df.votes<df.votes.mean()]
plt.scatter(df1.rate,df1.votes,color="red")
plt.xlabel('Rate')
plt.ylabel('Votes')
plt.title('rate Vs Votes')


# #### Realisations
# - Concidering the No. of votes as popularity of the restaurent.
# - Restaurennts with lesser votes are having lower Ratings.
# - Higher the no. of votes the higher is the potential probability of a company to get higher ratings.
# - Rating may depend on many other unexplored factors.

# ### Popularity Of North Indian Cuisines Vs South Indian Cuisines

# In[ ]:


N=df[df.cuisines.apply(lambda x: "North Indian" in str(x))]
N=N[N.cuisines.apply(lambda x: "South Indian" not in str(x))]
S=df[df.cuisines.apply(lambda x: "South Indian" in str(x))]
S=S[S.cuisines.apply(lambda x: "North Indian" not in str(x))]
plt.figure(figsize=(20,10))
height = [N.votes.mean(),S.votes.mean()]
bars = ('North Indian', 'South Indian')
y_pos = np.arange(len(bars))
plt.bar(y_pos, height,color=(0.2, 0.4, 0.6, 0.6))
plt.xticks(y_pos, bars)
plt.ylabel("Votes")
plt.xlabel("Basic cuisine")
plt.title("North Vs South indian Cuisines")
plt.show()


# #### Realizations
# >- Here we are comparing two basic and distinct cuisines North and South Indian for simplicity.
# >- North Indian cuisines tend to be more popular if compared with south indian cuisines(only).
# >- Maybe because of massive influx of North Indian Students or Employees North Indian cuisines are more populer in a South Indian City.

# In[ ]:


len(df.location.unique())


# ### Top 20 Popular Locations

# In[ ]:


x=df.groupby("location")["votes"].mean()
x=pd.DataFrame(x)
x.reset_index(inplace=True)
x.sort_values(by='votes', ascending=False,inplace=True)
x.reset_index(inplace=True)
plt.figure(figsize=(20,10))
height = x.votes[0:20]
bars = x.location[0:20]
y_pos = np.arange(len(bars))
plt.bar(y_pos, height,color="red")
plt.xticks(y_pos, bars)
plt.xticks(rotation=90)
plt.ylabel("Popularity")
plt.xlabel("Locations")
plt.title("Top 20 highest popular locations")
plt.show()


# #### Realizations 
# >- Not concidering Ratings of the restaurents at this point.
# >- No. of votes depict the popularity of the restaurent
# >- The above graph shows the top 20 locations with most popular restaurents

# In[ ]:


fig1, ax1 = plt.subplots()
ax1.set_title('Rating')
ax1.boxplot(df.rate)


# ### What did people like in the higher rated restaurents (Word Cloud)

# In[ ]:


#Fetching Stop Words
stop_words = set(stopwords.words('english'))

#Fetching the rewiews which had higher ratings into a string
rg3=''
for m in df.reviews_list:
    m = ast.literal_eval(m)
    for each in m:
        if each[0] == "Rated 4.0" or each[0] == "Rated 5.0":
            rg3+=" "+each[1]
#Tokenizing the collected reviews
token3=word_tokenize(rg3)
#Removing Stopwords
filt3=[]
for r in token3: 
    if not r.lower() in stop_words: 
        filt3.append(r.lower())
#Filtering the tokens by size
def by_size(words, size):
    return [word for word in words if len(word) >= size]
x=by_size(filt3,4)
#Finding the most common tokens
Counter = Counter(x)
most_occur = Counter.most_common(100)
##Plotting the word cloud
from wordcloud import WordCloud
text=''
for m in most_occur:
    if m[0] != "rated":
        text += " " + m[0]
wordcloud = WordCloud(width=1000, height=800, margin=0).generate(text)
plt.figure(figsize=(20,20))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.margins(x=0,y=0)
plt.show()    


# - The above word cloud states the most commonly used words in the HIgher rated reviews.
# - These are the words people talked about more frequently.
# - Completely filtering all the unnecessary words was not possible.
# - It provides a brief/ unstructured idea of what a restaurent should focus on for higher reviews.

# ### Top 20 most Liked Dishes/ Beverages

# Data for most liked dishes for each restaurent was provided in "dish_liked" column

# In[ ]:


from collections import Counter
#collecting the data into a list
st=[]
for m in df.dish_liked:
    if pd.notna(m):
        for each in m.split(","):
            st.append(each)
#Finding the most common dish/beverage
Counter1 = Counter(st)
most_occur1 = Counter1.most_common(20)
#plotting the bar graph for first 20 dishes
plt.figure(figsize=(20,10))
height = [m[1] for m in most_occur1]
bars = [m[0] for m in most_occur1]
y_pos = np.arange(len(bars))
plt.bar(y_pos, height,color="Blue")
plt.xticks(y_pos, bars)
plt.xticks(rotation=90)
plt.ylabel("Popularity")
plt.xlabel("Food")
plt.title("Foods/Beverages and their Popularity")
plt.show()


# ### Percentage of Restaurents Providing Online Order And Table Booking

# In[ ]:


x=df.groupby("online_order")["votes"].count()
labels = 'Yes', 'No'
sizes = [x.Yes, x.No]
colors = ['gold', 'yellowgreen']
explode = (0.1, 0,) 
plt.pie(sizes, explode=explode, labels=labels, colors=colors,
autopct='%1.1f%%', shadow=True, startangle=140)
plt.title("Restaurents Providing Online Orders")
plt.axis('equal')
plt.show()


# In[ ]:


x=df.groupby("book_table")["votes"].count()
labels = 'Yes', 'No'
sizes = [x.Yes, x.No]
colors = ['red', 'blue']
explode = (0.1, 0,) 
plt.pie(sizes, explode=explode, labels=labels, colors=colors,
autopct='%1.1f%%', shadow=True, startangle=140)
plt.title("Restaurents Providing Table Booking")
plt.axis('equal')
plt.show()


# ### Restaurents and their Availability

# In[ ]:


rt=[]
for m in df.rest_type:
    if pd.notna(m):
        for each in m.split(","):
            rt.append(each)
Counter2 = Counter(rt)
most_occur2 = Counter2.most_common(20)
plt.figure(figsize=(20,10))
height = [m[1] for m in most_occur2]
bars = [m[0] for m in most_occur2]
y_pos = np.arange(len(bars))
plt.bar(y_pos, height,color="orange")
plt.xticks(y_pos, bars)
plt.xticks(rotation=90)
plt.ylabel("Availability")
plt.xlabel("Rest_type")
plt.title("Restaurent types and their Availability")
plt.show()

