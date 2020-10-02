#!/usr/bin/env python
# coding: utf-8

# **Hello folks!******
# 
# I started with Machine Learning a couple of months back.In the college summer breaks I came to know about Kaggle and this is my first Kernel.Since this is my first kernel and first real world dataset , I certailny would have made number of mistakes.Point my mistakes so it helps me code better next time :)
# 
# **Thing Covered in Kernel:**
# 1. Histogram of Average Cost for 2 person
# 2. Best Rated Restraunts in Town
# 3. Restraunt Count area wise
# 4. Area with most Rated Restraunts
# 5. Average cost per Area
# 6. Number of Restraunts in rating range
# 7. Top chains Restraunt
# 8. Area with Best Rated Restraunts (Min 20)
# 9. Area with least cost for couple (Min 10)

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


# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


df=pd.read_csv('../input/zomato.csv')


# In[ ]:


df.isnull().sum()


# In[ ]:


#removing redundant data based on address and name
df.head()


# **There exists a large number of redundant values . Also it is to be noted that not all are different , they just differ in listed_in(type) hence it is necessary to remove duplicates based on name and address.**

# In[ ]:


df[df['name']=='Onesta']


# In[ ]:


#removing redundant data based on address and name
df=df.drop_duplicates(subset=['address','name'],keep='last')


# In[ ]:


#fill NaN in the rating column
df['Rate']=df['rate'].fillna('3.0/5',inplace=True)


# In[ ]:


#Rating is 4.1/5 we take 4.1 as string then convert it to float
df['Rating']=df['rate'].map(lambda x: str(x)[0:3])
df['Rating']=pd.to_numeric(df.Rating, errors='coerce')


# In[ ]:


#converting average price of 2 person to float
df=df.rename(columns={'approx_cost(for two people)':'cost'})
df['Ncost']=df['cost'].str.replace(',','')
df['Ncost']=pd.to_numeric(df.Ncost,errors='coerce')
df.isnull().sum()


# In[ ]:


#Dropping columns not necessary.
df=df.drop(['phone','cost','rate'],axis=1)
df=df.drop(['url','Rate'],axis=1)
df=df.drop(['menu_item'],axis=1)


# **Below we can see two columns with almost same data , we drop one with less data.**

# In[ ]:


print(df['listed_in(city)'].nunique())
print(df['location'].nunique())


# In[ ]:


df=df.drop('listed_in(city)',axis=1)


# In[ ]:


df['location'].value_counts()


# **Creating a histogram plot of cost of 2 person in Bangalore.**

# In[ ]:


df.hist(column='Ncost',bins=30)


# **Creating histogram with 5 areas with most Restraunts.**

# In[ ]:


plt.figure(figsize=(6,5))
sns.countplot(x='location',data=df,order=df.location.value_counts().iloc[:5].index,palette='rainbow')
plt.xticks(rotation=90)


# **Now we find the area with most Restraunts rated over 4.5 . No surprise as Kormangala comes out on top.**

# In[ ]:


df2=df[df['Rating']>4.4]


# In[ ]:


df2.shape
#195 restraunts hace 


# In[ ]:


plt.figure(figsize=(6,5))
sns.countplot(x='location',data=df2,order=df2.location.value_counts().iloc[:5].index)
plt.xticks(rotation=90)


# **Finding Best Restraunts**
# 
# In order to find the best restraunts we need to take care of 2 things :
# 1. Rating
# 2. Votes
# 
# We can't decide solely on basis of Rating factors. Since there are number of Restraunts with just small difference in rating despite having large difference in number of votes.
# Hence taking both factors into consideration , my Top 5 Restraunts are :
# 1. Byg Brewski Brewing Company
# 2. Toit
# 3. Truffles
# 4. AB's - Absolute Barbecues
# 5. The Black Pearl

# In[ ]:


#Top 7 rated restraunts
df=df.sort_values('votes',ascending=False)
fl=df[:5]
plt.figure(figsize=(10,10))

x=df['Rating']
y=df['votes']
plt.scatter(x,y,label='Best Restraunts In Bangalore',marker='o')

label=list(fl['name'])
x=list(fl['Rating'])
y=list(fl['votes'])
for i in range(len(label)):
    plt.annotate(label[i],(x[i],y[i]),ha='right')
plt.xlabel('Average Rating')
plt.ylabel('No of Votes')


# **Now we evaluate average cost for 2 persons in few popular areas.**

# In[ ]:


df['location'].value_counts()
loc=['Whitefield','BTM','Electronic City','HSR','Marathahalli','JP Nagar','Jayanagar','Banashankari','Koramangala 5th Block','Basavanagudi']


# In[ ]:


df3=df
areacost=pd.DataFrame(columns=['Area','Avg'])
for i in range(len(loc)):
    t3=df3[df3['location']==loc[i]]
    areacost.loc[i]=(loc[i],t3['Ncost'].mean())
areacost=areacost.sort_values('Avg')


# In[ ]:


areacost.head()


# In[ ]:


plt.figure(figsize=(10,5))
sns.barplot(x='Area',y='Avg',data=areacost)
plt.xticks(rotation=90)


# **Finding the costliest 5 Restraunts of Bangalore.**

# In[ ]:


df4=df
df4=df4.sort_values('Ncost',ascending=False)
print(df4[['name','Ncost']].iloc[:5])


# **Now we find the number of restraunts in the given rating range.**

# In[ ]:


li=np.arange(1.5,5.1,.5)
ratingres=pd.DataFrame(columns=['Rating','Number'])
df5=df
for i in range(len(li)-1):
    t5=df5[df5['Rating']>li[i]]
    t5=t5[t5['Rating']<=li[i+1]]
    ratingres.loc[i]=(str(li[i])+' - '+str(li[i+1]),len(t5))


# In[ ]:


ratingres.head()


# In[ ]:


plt.figure(figsize=(10,7))
sns.barplot(x='Rating',y='Number',data=ratingres)
#Most are rated between 3-4 which determines they are average.


# **Maximum Outlets**
# 
# Okay so this idea of finding the restraunt with most outlet came when I was looking at kernels.One thing which caught my eye was Onesta having 80+ outlets , which clearly is wrong. It possibly was since dataset had listed same place restraunt in different listed_in(type) despite having the same address . Hence I felt to remove those data and came up with different results hence.

# In[ ]:


df6=df


# In[ ]:


t6=df6.groupby('name').size().to_frame('count').reset_index().sort_values(['count'],ascending=False)


# In[ ]:


t6.head()


# Hence **Cafe Coffee Day** has the most outlets and probably correct as well.

# In[ ]:


plt.figure(figsize=(8,6))
g=sns.barplot(x='name',y='count',data=t6[:10])
plt.xticks(rotation=90)


# **Area with the best rated Restraunts (Min 20)**

# In[ ]:


df7=df
li=df7.groupby('location').size().to_frame('value_counts').reset_index().sort_values(['value_counts'],ascending=False)


# In[ ]:


avg_area_rating=pd.DataFrame(columns=['Area','AvgRating','Number'])


# In[ ]:


for i in range(len(li)):
    t7=df7[df7['location']==li.iloc[i][0]]
    if len(t7)>20:
        avg_area_rating.loc[i]=(li.iloc[i][0],t7['Rating'].mean(),len(t7))


# In[ ]:


avg_area_rating=avg_area_rating.sort_values('AvgRating',ascending=False)
avg_area_rating.head(7)


# In[ ]:


plt.figure(figsize=(10,6))
sns.barplot(x='Area',y='AvgRating',data=avg_area_rating.iloc[:7])
plt.xticks(rotation=90)


# **Area with least cost for 2 person min 10 restraunts.**

# In[ ]:


df8=df
min_cost_area=pd.DataFrame(columns=['Data','AvgCost','Number'])


# In[ ]:


li=df7.groupby('location').size().to_frame('value_counts').reset_index().sort_values(['value_counts'],ascending=False)


# In[ ]:


for i in range(len(li)):
    t7=df7[df7['location']==li.iloc[i][0]]
    if len(t7)>10:
        min_cost_area.loc[i]=(li.iloc[i][0],t7['Ncost'].mean(),len(t7))


# In[ ]:


min_cost_area=min_cost_area.sort_values('AvgCost',ascending=True)
min_cost_area.head()


# In[ ]:


label=[]
x=[]
y=[]
for i in range(10):
        label.append(min_cost_area.iloc[i][0])
        x.append(min_cost_area.iloc[i][2])
        y.append(min_cost_area.iloc[i][1])


# **Basavangudi seems a good place to go on a cheap date :p**

# In[ ]:


plt.figure(figsize=(10,6))
plt.scatter(x='Number',y='AvgCost',data=min_cost_area[:10])
plt.xlabel('Number of Restraunts')
plt.ylabel('Avg Cost')

for i in range(len(label)):
    plt.annotate(label[i],(x[i],y[i]),(x[i],y[i]),ha='left')


# Please provide your feedback in comments.
# **Thank You!**
