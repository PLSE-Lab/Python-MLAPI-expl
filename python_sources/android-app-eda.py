#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import mode


# In[ ]:


df = pd.read_csv("../input/google-play-store-apps/googleplaystore.csv")


# In[ ]:


df.shape


# In[ ]:


df.info()


# In[ ]:


df.head()


# #### We have attributes that have measurements but as data type object. We should convert them to into numeric for better usability.

# In[ ]:





# In[ ]:


def converter(x):
    if x[-1] == 'M' :
        x = float(x[:-1]) * 1000000
    else:
        x = float(x)
    return x
df['Reviews'] = df['Reviews'].apply(lambda x : converter(x))


# In[ ]:


def converter(x):
    if x[-1] == 'M' :
        x = float(x[:-1])
    elif x[-1] == 'k' :
        x = float(x[:-1]) /1024
    else :
        x = np.nan
    return x
df['Size'] = df['Size'].apply(lambda x : converter(x))


# In[ ]:


def converter(x):
    if x[-1] == '+':
        x = float(x[:-1].replace(',',''))
    elif x == 'Free':
        x = 0
    else :
        x = float(x.replace(',',''))
    return x
df['Installs'] =df['Installs'].apply(lambda x : converter(x))


# In[ ]:


def converter(x):
    if x[0] == '$':
        x = float(x[1:])
    elif x == 'Everyone' :
        x = 0
    else :
        x = float(x)
    return x
df['Price'] = df['Price'].apply(lambda x : converter(x))


# In[ ]:





# In[ ]:


df.isnull().sum()


# #### ALl these null value has to be taken care of.

# In[ ]:


mapper = pd.pivot_table(data=df, values='Size', columns='Category', aggfunc=mode)
temp  = df[df['Size'].isnull()]['Category'].apply(lambda x : mapper.loc[:,x][0][0][0])

df['Size'].fillna(value=temp,inplace=True)
del temp,mapper


# In[ ]:


mapper = pd.pivot_table(data=df, values='Rating', columns='Category', aggfunc=mode)

temp  = df[df['Rating'].isnull()]['Category'].apply(lambda x : mapper.loc[:,x][0][0][0])

df['Rating'].fillna(value=temp,inplace=True)
del temp,mapper


# In[ ]:


df.isnull().sum()


# #### Now that we have lesser null values we can simply drop them.

# In[ ]:


df.dropna(inplace=True)


# In[ ]:


df.info()


# In[ ]:


df.head()


# #### Attribute 'Last Updated' has data vlaues of date but in string format. We should converting it into datetime format will be more convinient.

# In[ ]:


df['Last Updated'] = pd.to_datetime(df['Last Updated'])


# #### Let's start some Exploratory Data Analysis and get some insights from data.

# In[ ]:


plt.figure(figsize=(15,4))
sns.countplot(df['Category'])
plt.xticks(rotation=90)


# #### Above diagram shows the distrbution of the app categories. Play store has more number of apps that fall in 'Family','game' or 'tools Category.

# In[ ]:


temp = df['Type'].value_counts()
plt.pie(temp, labels=temp.index, autopct='%1.1f%%', explode=[0,0.1], shadow=True)


# #### Above graph shows us that 92.6% of the apps on play store are free of cost. that sounds good.

# In[ ]:


sns.countplot(df['Content Rating'])
plt.xticks(rotation=90)


# #### Most apps are made for everyone where we have lesser number of apps that specifically focus on any group.

# In[ ]:





# In[ ]:


df.columns


# ### Let's see the top apps by every category.

# In[ ]:


top_rated = df.sort_values(by=['Rating','Reviews'],ascending=False).head(50)
top_rated[['App','Rating','Reviews']].head(10)


# ### Above are the top 10 most rated apps on playstore with their ratings and number of reviews. 

# In[ ]:


most_installs = df.sort_values(by='Installs',ascending=False).head(50)
most_installs[['App','Installs']].head(10)


# #### Above are the top 10 most intalled apps on playstore with their number of installs.

# In[ ]:


most_expensive = df.sort_values(by='Price',ascending=False).head(50)
most_expensive[['App','Price']].head(10)


# #### Above are the top 10 most expensive apps on playstore and their prices.

# In[ ]:


largest = df.sort_values(by='Size',ascending=False).head(50)
largest[['App','Size']].head(10)


# #### Above are the top 10 largest apps on playstore and their sizes.

# ## Now, we'll look for the distribution of these top apps of each segment.

# ### Top Rated :

# In[ ]:


fig,ax = plt.subplots(1,3,figsize=(20,5))

top_rated['Category'].value_counts().plot(kind='bar',ax=ax[0],color='y',title='Category')

top_rated['Content Rating'].value_counts().plot(kind='bar',ax=ax[1],color='y',title='Content Rating')

top_rated['Genres'].value_counts().plot(kind='bar',ax=ax[2],color='y',title='Genres')


# ### Most Installed :

# In[ ]:


fig,ax = plt.subplots(1,3,figsize=(20,5))

most_installs['Category'].value_counts().plot(kind='bar',ax=ax[0],color='y',title='Category')

most_installs['Content Rating'].value_counts().plot(kind='bar',ax=ax[1],color='y',title='Content Rating')

most_installs['Genres'].value_counts().plot(kind='bar',ax=ax[2],color='y',title = 'Genres')


# ### Most Expensive :

# In[ ]:


fig,ax = plt.subplots(1,3,figsize=(20,5))

most_expensive['Category'].value_counts().plot(kind='bar',ax=ax[0],color='y',title='Category')

most_expensive['Content Rating'].value_counts().plot(kind='bar',ax=ax[1],color='y',title='Content Rating')

most_expensive['Genres'].value_counts().plot(kind='bar',ax=ax[2],color='y',title='Genres')


# ### Largest :

# In[ ]:


fig,ax = plt.subplots(1,3,figsize=(20,5))

largest['Category'].value_counts().plot(kind='bar',ax=ax[0],color='y',title='Category')

largest['Content Rating'].value_counts().plot(kind='bar',ax=ax[1],color='y',title='Content Rating')

largest['Genres'].value_counts().plot(kind='bar', ax=ax[2],color='y',title='Genres')


# In[ ]:


del top_rated,most_expensive,most_installs,largest


# # Let's see the distribution of data by different attributes.

# ## Distribution By Category :

# In[ ]:


plt.figure(figsize=(15,4))
df.groupby(by='Category')['Price'].mean().sort_values(ascending=False).plot(kind='bar',title='Price')


# In[ ]:


plt.figure(figsize=(15,4))
df.groupby(by='Category')['Size'].median().sort_values(ascending=False).plot(kind='bar',title='Size')


# In[ ]:


plt.figure(figsize=(15,4))
df.groupby(by='Category')['Rating'].median().sort_values(ascending=False).plot(kind='bar',title='Rating')


# In[ ]:


plt.figure(figsize=(15,4))
df.groupby(by='Category')['Installs'].median().sort_values(ascending=False).plot(kind='bar',title='Number of Installs')


# ### from above graphs we can interpret that if:
# - App that fall into Finance, Lifestyle and medical category are expected to be expensive.
# - Apps that fall in tools or libraries category are expected to be smaller in size. Other than these, size vary by different categories except for games.
# - If someone is downloading a app that is of category Game, it is expected to be large in size.
# - All Categories have similar user ratings. This means that user ratings are not biased on categories.
# - If an app falls in Entertainment or Photography Category, it has chances of having more downloads.

# ## Distribution By Genres :

# In[ ]:


plt.figure(figsize=(20,4))
df.groupby(by='Genres')['Price'].mean().sort_values(ascending=False).plot(kind='bar',title='Price',color='g')


# In[ ]:


plt.figure(figsize=(20,4))
df.groupby(by='Genres')['Rating'].mean().sort_values(ascending=False).plot(title='Rating',color='g')


# In[ ]:


plt.figure(figsize=(20,4))
df.groupby(by='Genres')['Size'].mean().sort_values(ascending=False).plot(kind='bar',title='Size',color='g')


# In[ ]:


plt.figure(figsize=(20,4))
df.groupby(by='Genres')['Installs'].mean().sort_values(ascending=False).plot(kind='bar',title='Installs',color='g')


# ### From Above Graphs we can interpret :
# - An app that falls in finance or lifestyle genre is expectedly expensive.
# - User ratings are bias to genre, some of the genres generally better ratings than others.This can suggest that there are fewer good performing apps of those genres that have lesser general user ratings.
# - Size of the app varies in different genres. 
# - Apps that fall in Communication genre have generally more downloads.

# In[ ]:


fig,ax = plt.subplots(1,4,figsize=(15,3))
df.groupby(by='Content Rating')['Price'].mean().sort_values(ascending=False).plot(kind='bar',title='Price',ax=ax[0])
df.groupby(by='Content Rating')['Rating'].mean().sort_values(ascending=False).plot(kind='bar',title='Rating',ax=ax[1])
df.groupby(by='Content Rating')['Size'].mean().sort_values(ascending=False).plot(kind='bar',title='Size',ax=ax[2])
df.groupby(by='Content Rating')['Installs'].mean().sort_values(ascending=False).plot(kind='bar',title='Installs',ax=ax[3])


# In[ ]:





# In[ ]:





# ### Distribution of dataset.

# In[ ]:


sns.pairplot(df,hue='Type')


# ### How size affects raings :

# In[ ]:


sns.jointplot(data=df,x='Size',y='Rating')


# #### Most Top rated apps have 2mb < size < 40mb

# ### How Price affects ratings.

# In[ ]:


sns.jointplot(data=df,x='Price',y='Rating')


# In[ ]:


sns.scatterplot(data=df, x='Rating', y='Size',hue='Type')


# #### Paid apps have average high ratings.

# ### Ratings amongs different categories.

# In[ ]:


g = sns.FacetGrid(df,col='Category',col_wrap=5,sharey=False,margin_titles=True)
g.map(sns.countplot,'Rating',order=[1,2,3,4,5])


# ### How categories have paid and free apps count.

# In[ ]:


temp = df.groupby(by=['Category','Type'])['App'].count().unstack()
temp['Total'] = temp.sum(axis=1)
temp =temp.sort_values(by='Total',ascending=False).head(5)
temp.drop('Total',axis=1,inplace=True)

temp.plot(kind='bar')

del temp


# In[ ]:


import pandas_profiling


# In[ ]:


report = pandas_profiling.ProfileReport(df)
report


# In[ ]:


report.to_file('App_Data_Profile_Report.html')


# In[ ]:


df.columns


# ### Now, user reviews :

# In[ ]:


reviews = pd.read_csv("../input/google-play-store-apps/googleplaystore_user_reviews.csv")


# In[ ]:


reviews.head()


# In[ ]:


from wordcloud import WordCloud, STOPWORDS

stop = set(STOPWORDS)

sentiments = str('').join(reviews['Translated_Review'].dropna().to_list())


# #### Wordcloud below represents all sentimens.

# In[ ]:


cloud = WordCloud(width=800, height=500, stopwords=stop, background_color='white',max_words=50)

img = cloud.generate(sentiments)

plt.figure(figsize = (8, 5), facecolor = None) 
plt.imshow(img) 
plt.axis("off") 
plt.tight_layout(pad = 0) 
  
plt.show() 


# In[ ]:





# In[ ]:




