#!/usr/bin/env python
# coding: utf-8

# Hi, welcome to my kernel.

# I will explore this dataset the consists in 129970 reviews about some wine labels.
# 
# 

# 
# ## I will try answer a batch of questions, like:
# 
# - Have an Provinces the same number of wines? <br>
# - Whats the distribuition of Price and Points by Province? <br>
# - Whats the country distribuition<br>
# - The taster's have the same number of votings? <br>
# - What's the distribuition of Points and Prices by taster's name?   <br>
# - Taking a look on the word clouds.    <br>
#      

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


# In[2]:


df_wine1 = pd.read_csv('../input/winemag-data-130k-v2.csv', index_col=0)


# <h2>Let's take a first look on our data</h2>

# In[3]:


print(df_wine1.info())


# - 129971 rows and 13 columns. <br>
# 
# Now, let's look further the nulls
# 

# <h2>Null values</h2>

# In[4]:


df_wine1.isnull().sum()


# We have 4 variables with the a relative high number of nulls. Later I will explore this column  further

# <h2>Unique values</h2>

# In[5]:


print(df_wine1.nunique())


# <h2>Let's start looking the distribuition of Points and Prices </h2>

# In[6]:


print("Statistics of numerical data: ")
print(df_wine1.describe())


# Very interesting distribuition of Points and Price.
# - We can see that the values of points are distributed between 80 and 100
# - The price have a high difference between the values and a high standard deviation

# <h2>Lets see some graphs of this distribuitions</h2>

# In[7]:


plt.figure(figsize=(16,6))

g = plt.subplot(1,2,1)
g = sns.countplot(x='points', data=df_wine1)
g.set_title("Point Count distribuition ", fontsize=20)
g.set_xlabel("Points", fontsize=15)
g.set_ylabel("Count", fontsize=15)

g1 = plt.subplot(1,2,2)
g1 = sns.boxplot(df_wine1['points'], orient='v')
g1.set_title("Point Count distribuition ", fontsize=20)
g1.set_xlabel("", fontsize=15)
g1.set_ylabel("Count", fontsize=15)

plt.show()

print("% Percentile Values")
print(df_wine1['points'].quantile([.025,.25,.75,.975]))


# We can see clear the distribuitioon of our data.  Th IQR is 5 and our total 

# <h2>Now, let's take a first look at the Price column distribuition</h2>
# - We have some outliers on price column, because on our 3 quartile the value is 42 and the values goes until 3300
# '

# Let's look the outliers'

# In[8]:


plt.figure(figsize=(8,5))
plt.scatter(range(df_wine1.shape[0]), np.sort(df_wine1.price.values))
plt.xlabel('Index', fontsize=15)
plt.ylabel('Prices(US)', fontsize=15)
plt.title("Distribuition of prices", fontsize=20)
plt.show()


# Very interesting values. Below I will visualize the distribuition filtering the price value

# In[9]:


plt.figure(figsize=(20,6))

g = plt.subplot(121)
g = sns.distplot(df_wine1[df_wine1['price'] < 300]['price'])
g.set_title("Price Distribuition Filtered 300", fontsize=20)
g.set_xlabel("Prices(US)", fontsize=15)
g.set_ylabel("Frequency Distribuition", fontsize=15)

g1 = plt.subplot(122)
g1 = sns.distplot(np.log(df_wine1['price'].dropna()))
g1.set_title("Price Log distribuition  ", fontsize=20)
g1.set_xlabel("Price(Log)", fontsize=15)
g1.set_ylabel("Frequency LOG", fontsize=15)

plt.show()


# <h2>Let's take look the Points crossed by Prices'</h2>

# In[10]:


plt.figure(figsize=(10,4))

g = sns.regplot(x='points', y='price', data=df_wine1, x_jitter=True, fit_reg=False)
g.set_title("Points x Price Distribuition", fontsize=20)
g.set_xlabel("Points", fontsize= 15)
g.set_ylabel("Price", fontsize= 15)

plt.show()


# Very meaningful scatter plot. 
# - The highest values isn't of the wine with highest pontuation. 
# - The most expensive wine have ponctuation between 87 and 90

# <h2>Let's take a quick look at all wines with price highest than USD 2000 </h2>

# In[11]:


df_wine1.loc[(df_wine1['price'] > 2000)]


# We have just 7 wines with values highest than 1500 Us. Also, It's interesting to note that the most expensive is also the lowest ponctuation and that the almost all of them are from France. 

# <h2>Looking through the countries'</h2>

# In[12]:


plt.figure(figsize=(14,6))

country = df_wine1.country.value_counts()[:20]

g = sns.countplot(x='country', data=df_wine1[df_wine1.country.isin(country.index.values)])
g.set_title("Country Of Wine Origin Count", fontsize=20)
g.set_xlabel("Country's ", fontsize=15)
g.set_ylabel("Count", fontsize=15)
g.set_xticklabels(g.get_xticklabels(),rotation=45)

plt.show()


# Wow, very interesting distribuition. I was expecting to see Italy, Chile or Argentina as the biggest wine productor. <br>
#  If you want take a better look, look the print output below

# ## Now, I will take a look in the distribuition of this top 20 countrys by price and rating

# In[13]:


plt.figure(figsize=(16,12))

plt.subplot(2,1,1)
g = sns.boxplot(x='country', y='price',
                  data=df_wine1.loc[(df_wine1.country.isin(country.index.values))])
g.set_title("Price by Country Of Wine Origin", fontsize=20)
g.set_xlabel("Country's ", fontsize=15)
g.set_ylabel("Price Dist(US)", fontsize=15)
g.set_xticklabels(g.get_xticklabels(),rotation=45)

plt.subplot(2,1,2)
g1 = sns.boxplot(x='country', y='points',
                   data=df_wine1[df_wine1.country.isin(country.index.values)])
g1.set_title("Points by Country Of Wine Origin", fontsize=20)
g1.set_xlabel("Country's ", fontsize=15)
g1.set_ylabel("Points", fontsize=15)
g1.set_xticklabels(g1.get_xticklabels(),rotation=45)

plt.subplots_adjust(hspace = 0.6,top = 0.9)

plt.show()


# ## Taking a look on values lowest than 500

# In[14]:


plt.figure(figsize=(15,5))
g = sns.boxplot(x='country', y='price',
                  data=df_wine1.loc[(df_wine1.country.isin(country.index.values))  &
                                    (df_wine1.price < 500)])
g.set_title("Price by Country Of Wine Origin", fontsize=20)
g.set_xlabel("Country's ", fontsize=15)
g.set_ylabel("Price Dist(US)", fontsize=15)
g.set_xticklabels(g.get_xticklabels(),rotation=45)

plt.show()


# It's very interesting that all wines have  quartiles in a values lower than 100

# ## Province Exploration

# In[15]:


plt.figure(figsize=(14,15))

provinces = df_wine1['province'].value_counts()[:20]

plt.subplot(3,1,1)
g = sns.countplot(x='province', 
                  data=df_wine1.loc[(df_wine1.province.isin(provinces.index.values))],
                 palette='Set2')
g.set_title("Province Of Wine Origin ", fontsize=20)
g.set_xlabel("Provinces", fontsize=15)
g.set_ylabel("Count", fontsize=15)
g.set_xticklabels(g.get_xticklabels(),rotation=45)

plt.subplot(3,1,2)
g1 = sns.boxplot(y='price', x='province',
                  data=df_wine1.loc[(df_wine1.province.isin(provinces.index.values))],
                 palette='Set2')
g1.set_title("Province Of Wine Origin ", fontsize=20)
g1.set_xlabel("Province", fontsize=15)
g1.set_ylabel("Price", fontsize=15)
g1.set_xticklabels(g1.get_xticklabels(),rotation=45)

plt.subplot(3,1,3)
g2 = sns.boxplot(y='points', x='province',
                  data=df_wine1.loc[(df_wine1.province.isin(provinces.index.values))],
                 palette='Set2')
g2.set_title("Province Of Wine Origin", fontsize=20)
g2.set_xlabel("Provinces", fontsize=15)
g2.set_ylabel("Points", fontsize=15)
g2.set_xticklabels(g2.get_xticklabels(),rotation=45)

plt.subplots_adjust(hspace = 0.6,top = 0.9)

plt.show()


# In[16]:


df_wine1.nunique()


# In[17]:


plt.figure(figsize=(14,16))

provinces = df_wine1['province'].value_counts()[:20]

plt.subplot(3,1,1)
g = sns.countplot(x='taster_name', data=df_wine1, palette='hls')
g.set_title("Taster Name Count ", fontsize=20)
g.set_xlabel("Taster Name", fontsize=15)
g.set_ylabel("Count", fontsize=15)
g.set_xticklabels(g.get_xticklabels(),rotation=45)

plt.subplot(3,1,2)
g1 = sns.boxplot(y='price', x='taster_name', data=df_wine1,
                 palette='hls')
g1.set_title("Taster Name Wine Values Distribuition ", fontsize=20)
g1.set_xlabel("Taster Name", fontsize=15)
g1.set_ylabel("Price", fontsize=15)
g1.set_xticklabels(g1.get_xticklabels(),rotation=45)

plt.subplot(3,1,3)
g2 = sns.boxplot(y='points', x='taster_name',
                  data=df_wine1, palette='hls')
g2.set_title("Taster Name Points Distribuition", fontsize=20)
g2.set_xlabel("Taster Name", fontsize=15)
g2.set_ylabel("Points", fontsize=15)
g2.set_xticklabels(g2.get_xticklabels(),rotation=45)

plt.subplots_adjust(hspace = 0.6,top = 0.9)

plt.show()


# In[ ]:





# In[18]:


plt.figure(figsize=(14,16))

designation = df_wine1.designation.value_counts()[:20]

plt.subplot(3,1,1)
g = sns.countplot(x='designation', 
                  data=df_wine1.loc[(df_wine1.designation.isin(designation.index.values))],
                 palette='Set2')
g.set_title("Province Of Wine Origin ", fontsize=20)
g.set_xlabel("Country's ", fontsize=15)
g.set_ylabel("Count", fontsize=15)
g.set_xticklabels(g.get_xticklabels(),rotation=45)

plt.subplot(3,1,2)
g1 = sns.boxplot(y='price', x='designation',
                  data=df_wine1.loc[(df_wine1.designation.isin(designation.index.values))],
                 palette='Set2')
g1.set_title("Province Of Wine Origin ", fontsize=20)
g1.set_xlabel("Province", fontsize=15)
g1.set_ylabel("Price", fontsize=15)
g1.set_xticklabels(g1.get_xticklabels(),rotation=45)

plt.subplot(3,1,3)
g2 = sns.boxplot(y='points', x='designation',
                  data=df_wine1.loc[(df_wine1.designation.isin(designation.index.values))],
                 palette='Set2')
g2.set_title("Province Of Wine Origin", fontsize=20)
g2.set_xlabel("Provinces", fontsize=15)
g2.set_ylabel("Points", fontsize=15)
g2.set_xticklabels(g2.get_xticklabels(),rotation=45)

plt.subplots_adjust(hspace = 0.6,top = 0.9)

plt.show()


# In[ ]:





# In[19]:


plt.figure(figsize=(14,16))

variety = df_wine1.variety.value_counts()[:20]

plt.subplot(3,1,1)
g = sns.countplot(x='variety', 
                  data=df_wine1.loc[(df_wine1.variety.isin(variety.index.values))],
                 palette='Set2')
g.set_title("TOP 20 Variety ", fontsize=20)
g.set_xlabel(" ", fontsize=15)
g.set_ylabel("Count", fontsize=15)
g.set_xticklabels(g.get_xticklabels(),rotation=45)

plt.subplot(3,1,2)
g1 = sns.boxplot(y='price', x='variety',
                  data=df_wine1.loc[(df_wine1.variety.isin(variety.index.values))],
                 palette='Set2')
g1.set_title("Price by Variety's", fontsize=20)
g1.set_xlabel("", fontsize=15)
g1.set_ylabel("Price", fontsize=15)
g1.set_xticklabels(g1.get_xticklabels(),rotation=45)

plt.subplot(3,1,3)
g2 = sns.boxplot(y='points', x='variety',
                  data=df_wine1.loc[(df_wine1.variety.isin(variety.index.values))],
                 palette='Set2')
g2.set_title("Points by Variety's", fontsize=20)
g2.set_xlabel("Variety's", fontsize=15)
g2.set_ylabel("Points", fontsize=15)
g2.set_xticklabels(g2.get_xticklabels(),rotation=90)

plt.subplots_adjust(hspace = 0.7,top = 0.9)

plt.show()


# ## Let's take a look at Winery Distribuitions 

# In[20]:


plt.figure(figsize=(14,16))

winery = df_wine1.winery.value_counts()[:20]

plt.subplot(3,1,1)
g = sns.countplot(x='winery', 
                  data=df_wine1.loc[(df_wine1.winery.isin(winery.index.values))],
                 palette='Set2')
g.set_title("TOP 20 most frequent Winery's", fontsize=20)
g.set_xlabel(" ", fontsize=15)
g.set_ylabel("Count", fontsize=15)
g.set_xticklabels(g.get_xticklabels(),rotation=45)

plt.subplot(3,1,2)
g1 = sns.boxplot(y='price', x='winery',
                  data=df_wine1.loc[(df_wine1.winery.isin(winery.index.values))],
                 palette='Set2')
g1.set_title("Price by Winery's", fontsize=20)
g1.set_xlabel("", fontsize=15)
g1.set_ylabel("Price", fontsize=15)
g1.set_xticklabels(g1.get_xticklabels(),rotation=45)

plt.subplot(3,1,3)
g2 = sns.boxplot(y='points', x='winery',
                  data=df_wine1.loc[(df_wine1.winery.isin(winery.index.values))],
                 palette='Set2')
g2.set_title("Points by Winery's", fontsize=20)
g2.set_xlabel("Winery's", fontsize=15)
g2.set_ylabel("Points", fontsize=15)
g2.set_xticklabels(g2.get_xticklabels(),rotation=90)

plt.subplots_adjust(hspace = 0.7,top = 0.9)

plt.show()


# We can see that some winery's have +- 200 label in his portfolio's. 
# 
# Also, we can verify that Willians Selyem have highest points distribuition

# 

# In[ ]:


## WORDCLOUDS OF C


# In[ ]:



from wordcloud import WordCloud, STOPWORDS

stopwords = set(STOPWORDS)

newStopWords = ['fruit', "Drink", "black"]

stopwords.update(newStopWords)

wordcloud = WordCloud(
    background_color='white',
    stopwords=stopwords,
    max_words=1500,
    max_font_size=200, 
    width=1000, height=800,
    random_state=42,
).generate(" ".join(df_wine1['description'].astype(str)))

print(wordcloud)
fig = plt.figure(figsize = (12,14))
plt.imshow(wordcloud)
plt.title("WORD CLOUD - DESCRIPTION",fontsize=25)
plt.axis('off')
plt.show()


# ## WORDCLOUD OF WINE TITLES

# In[ ]:


stopwords = set(STOPWORDS)

newStopWords = ['']

stopwords.update(newStopWords)

wordcloud = WordCloud(
    background_color='white',
    stopwords=stopwords,
    max_words=1500,
    max_font_size=200, 
    width=1000, height=800,
    random_state=42,
).generate(" ".join(df_wine1['title'].astype(str)))

print(wordcloud)
fig = plt.figure(figsize = (12,14))
plt.imshow(wordcloud)
plt.title("WORD CLOUD - TITLES",fontsize=25)
plt.axis('off')
plt.show()


# In[ ]:




