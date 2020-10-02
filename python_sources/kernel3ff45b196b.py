#!/usr/bin/env python
# coding: utf-8

# # IMPORTING DATA

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import os
df = pd.read_csv('../input/google-play-store-apps/googleplaystore.csv')
df.head()


# # DATA CLEANING

# Removing App column and 10472 index because these has irrelevent dat

# In[ ]:


df=df.drop(columns='App')
df=df.drop(index=10472)
df.head()


# Changing the datatype of Reviews column since it has str datatype

# In[ ]:


df.Reviews=df.Reviews.apply(lambda r: int(r))
type(df.Reviews[0])


# Converting the size into float and Convert the Mb and Kb values into respected 10^ values

# In[ ]:


df.Size=df.Size.apply(lambda x: float(x[:len(x)-1])*10**6 if 'M' in x else float(x[:len(x)-1])*10**3 if 'k' in x else np.nan)
type(df.Size[0])


# Converting datatype of Installs into Integer and removing '+' .

# In[ ]:


import re
df.Installs=df.Installs.apply(lambda x:int(re.sub('[+,,]','',x)))
type(df.Installs[0])


# Checking the unique values in the Type column and we can neglect the 'nan' because there will be no effect of nan in plotting

# In[ ]:


df.Type.unique()


# Removing unwanted symbols from the Price column and convert the Datatype from str to float or int

# In[ ]:


df.Price=df.Price.apply(lambda x: float(re.split('\$',x)[1]) if '$' in x else 0.0)


# Editing the column names which has space

# In[ ]:


column_name={"Content Rating":"Content_rating","Last Updated":"Last_update","Current Ver":"Current_ver","Android Ver":"Android_ver"}
df=df.rename(columns=column_name)
df.columns.values


# # VISUALISATION

# Plotting the count of Categories, From this we can know Family category apps are more than any other apps and Beauty apps are the least ones.

# In[ ]:


f,ax=plt.subplots(figsize=(10,8))
sns.countplot('Category',data=df,ax=ax)
plt.xticks(rotation='vertical')


# In[ ]:


df.Category.value_counts()


# Plotting of reviews and median of reviews with respect to Category to know which Category has hight rating and which has low. from this plot we can know that 'Books and reference','events' and 'health and fitness' apps are having higher rating than any other apps. 'Dating' apps are having lowest rating.

# In[ ]:


f,ax=plt.subplots(figsize=(10,8))
plt.scatter(x=df.Category,y=df.Rating)
plt.xticks(rotation='vertical')
plt.ylim(0,6)


# In[ ]:


rating=df.groupby(['Category']).median()
f,ax=plt.subplots(figsize=(10,8))
plt.scatter(x=rating.index,y=rating.Rating.values)
plt.xticks(rotation='vertical')
plt.ylim(3,6)


# In[ ]:


rating.Rating[rating.Rating==rating.Rating.min()]
rating.Rating[rating.Rating==rating.Rating.max()]


# Some of the apps are having no Ratings so we have to plot the Categories and Rating to know which apps are having most numbers of NaN in their Rating column

# In[ ]:


f,ax=plt.subplots(figsize=(10,8))
sns.countplot(df.Category[df.Rating.isnull()])
plt.xticks(rotation='vertical')


# Plotting Size with respect to Categories to know which Category apps are large size and which one is small. From this plot we can know "Libraries and demo" Category has lowest size apps and    'FINANCE','LIFESTYLE','GAME','MEDICAL','HEALTH_AND_FITNESS','FAMILY', 'SPORTS' these Category apps are having highest Size apps.

# In[ ]:


f,ax=plt.subplots(figsize=(10,8))
plt.scatter(df.Category,df.Size)
plt.xticks(rotation='vertical')


# In[ ]:


df.Category[df.Size==df.Size.max()]
df.Category[df.Size==df.Size.min()]


# Plotting installs with Categories to know which category apps are installed more. For this we take sum() of installs with respect to categories to know which category got more installation by users. From this plot we can know that the 'Events' apps are installed by users lesser than other apps and users love with gaming apps because those apps have high number of installation.

# In[ ]:


install=df.groupby('Category').sum()
f,ax=plt.subplots(figsize=(10,8))
plt.bar(x=install.index.values,height=install.Installs)
plt.xticks(rotation='vertical')


# In[ ]:


install[install.Installs==install.Installs.max()]
install[install.Installs==install.Installs.min()]


# We know our data set has two types of Data which is 'Free' and 'Paid'. Plotting 'type' is to know about which type has more number of Categories. For this plotting we are going to use 'Pie chart' and 'Bar chart'. From out plots we can know that 'Free' types apps are having more than 92% of categories with numbers of 10039 Apps and only 7.4% of the apps are 'Paid' apps which is only 800 apps.

# In[ ]:


df2=df.groupby('Type').count()
f,ax=plt.subplots(figsize=(5,5))
plt.pie(df2.Category,labels=df2.index.values,autopct='%1.1f%%')


# In[ ]:


f,ax=plt.subplots(figsize=(10,8))
plt.bar(x=df2.index.values,height=df2.Category)
for i,j in zip(df2.index.values,df2.Category): 
    plt.text(i, j, str(j))


# Plotting content rating is to know which has more number of apps and which has less number and we are going to relate that with category, price and installs. For this we are going to use countplot ,barplot, relplot. from the count plot we can know that 'everyone' content rating has more number of apps and 'Unrated' content rating has only 2 apps.
# 
# When plotting with Price we can know that which Content has high priced apps and which is low. we can relate price and category plot with respect to content rating to know which Category apps are have more number of priced apps under the certain content rating.like that we can plot Category and Rating with respect to Content rating.

# In[ ]:


f,ax=plt.subplots(figsize=(10,8))
sns.countplot('Content_rating',data=df,ax=ax)
plt.xticks(rotation='vertical')
content=df.groupby('Content_rating').count()
content[content.Category==content.Category.max()]
content[content.Category==content.Category.min()]


# In[ ]:


f,ax=plt.subplots(figsize=(10,8))
sns.countplot('Category',data=df,ax=ax,hue='Content_rating')
plt.xticks(rotation='vertical')


# In[ ]:


f,ax=plt.subplots(figsize=(10,8))
sns.catplot('Content_rating','Price',data=df,ax=ax)
#sns.catplot('Content_rating','Installs',data=df)
plt.xticks(rotation='vertical')


# In[ ]:


sns.relplot('Category','Price',data=df,col='Content_rating',col_wrap=3)


# In[ ]:


sns.relplot('Category','Rating',data=df,col='Content_rating',col_wrap=3)


# # PREDICTION

# In this dataset we can predict which kind of apps can be installed more. so we can split our number of installs into three types. we consider those installed types as our output. we have to assign some numerical values to our independant variables.
# we have to remove null values from our data set.then we can use classification prediction methods such as 'SVC' and 'Logistic regression'.

# In[ ]:


df['category_num']=df.Category.replace(list(df.Category.unique()),range(0,len(list(df.Category.unique()))))
df['type_num']=df.Type.replace(['Free','Paid'],range(0,2))
df['content_rating_num']=df.Content_rating.replace(list(df.Content_rating.unique()),range(0,len(list(df.Content_rating.unique()))))
df['installs_num']=df.Installs.apply(lambda x: 2 if x<=1000000000 and x>=500000000 else 1 if x<500000000 and x>=15464338 else 0)
df=df.dropna(axis='rows')
x=df.loc[:,['Rating','Size','category_num','type_num','content_rating_num']]
y=df.installs_num


# In[ ]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y)


# In[ ]:


from sklearn.svm import SVC
svc=SVC().fit(x_train,y_train)
print(svc.score(x_test,y_test))
print(svc.score(x_train,y_train))


# In[ ]:


from sklearn.linear_model import LogisticRegression
log=LogisticRegression().fit(x_train,y_train)
print(log.score(x_train,y_train),log.score(x_test,y_test))


# In[ ]:


from sklearn.neighbors import KNeighborsRegressor
knn=KNeighborsRegressor(n_neighbors=3).fit(x_train,y_train)
print(knn.score(x_train,y_train),knn.score(x_test,y_test))


# In[ ]:




