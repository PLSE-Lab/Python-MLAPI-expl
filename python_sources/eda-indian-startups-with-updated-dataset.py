#!/usr/bin/env python
# coding: utf-8

# # Welcome to my EDA of Indian Startups !!
# 
# > Please, give your feedback and if you like this Kernel **votes up** :)
#  

# # **Context**
# 
# > *Interested in the Indian startup ecosystem just like me? Wanted to know what type of startups are getting funded in the last few years? Wanted to know who are the important investors? Wanted to know the hot fields that get a lot of funding these days? This dataset is a chance to explore the Indian start up scene. Deep dive into funding data and derive insights into the future!*

#  # **Objective**
# 
# > *In this project I will deal with **exploratory analysis**, where the objective is to understand how the data is distributed and generate insight for future decision-making, this analysis aims to explore as much as possible the data in a **simple, intuitive and informative** way. The data used in this project contains information only from **2015** to **2017**. Below is a sketch of all the stages made in these notebooks, following a logical and intuitive sequence to facilitate the understanding of the data.*
# > 

# # Importing the libraries

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS


# # Importing the Data

# In[ ]:


df=pd.read_csv('../input/indian-startup-funding/startup_funding.csv',index_col=0)
df.head()


# # Replacing misspelled Attributes

# In[ ]:


df['City  Location'] = df['City  Location'].replace(['Bangalore','Bengaluru'],'Bangalore')
df['City  Location'] = df['City  Location'].replace(['Gurgaon','Gurugram'],'Gurugram')

df['InvestmentnType'] = df['InvestmentnType'].replace(['Seed/ Angel Funding',
                                                       'Seed / Angel Funding',
                                                       'Seed/Angel Funding',
                                                       'Angel / Seed Funding',
                                                       'Seed Funding'],'Seed / Angel Funding')


# # Removing coma's from "**Amount in USD**" column

# In[ ]:


df["Amount in USD"] = df["Amount in USD"].apply(lambda x: str(x).replace(",",""))
df["Amount in USD"] = pd.to_numeric(df["Amount in USD"],errors='coerce')


# # Knowing our data

# In[ ]:


print('Shape of data',df.shape)
df.describe()


# # Top 10 Investment types

# In[ ]:


Investment=df.InvestmentnType.value_counts()
plt.figure(figsize=(15,12))
plt.subplot(221)
g = sns.barplot(x=Investment.index[:10],y=Investment.values[:10])
g.set_xticklabels(g.get_xticklabels(),rotation=70)
g.set_xlabel("Investment Types", fontsize=15)
g.set_ylabel("No of fundings made", fontsize=15)
plt.show()


# # Top 10 Cities for Startup's

# In[ ]:


city=df['City  Location'].value_counts()
plt.figure(figsize=(15,12))
plt.subplot(221)
g = sns.barplot(x=city.index[:10],y=city.values[:10])
g.set_xticklabels(g.get_xticklabels(),rotation=45)
g.set_xlabel("Cities", fontsize=15)
g.set_ylabel("No of fundings made", fontsize=15)
plt.show()


# # Top 5 Invester's (on basis of Max amount funded & frequency count)

# In[ ]:


maxfundingsdf=df.sort_values(by='Amount in USD',ascending=False,na_position='last')
top_fundings=maxfundingsdf['Amount in USD'].head(5)
invester_names= maxfundingsdf['Investors Name'].head(5)
plt.figure(figsize=(15,12))
plt.subplot(221)
g = sns.barplot(x=invester_names,y=top_fundings)
g.set_xticklabels(g.get_xticklabels(),rotation='vertical')
g.set_xlabel("\nInvesters group", fontsize=15)
g.set_ylabel("Max Amount", fontsize=15)
valuecount_investers=df['Investors Name'].value_counts()
plt.subplot(222)
g1 = sns.barplot(x=valuecount_investers.index[:5], y=valuecount_investers.values[:5]) 
g1.set_xticklabels(g.get_xticklabels(),rotation='vertical')
g1.set_xlabel("\nInvesters group", fontsize=15)
g1.set_ylabel("number of fundings made", fontsize=15)

plt.show()


# # WordCloud of Technology Investers

# In[ ]:


plt.figure(figsize = (15,15))

stopwords = set(STOPWORDS)
wordcloud = WordCloud(
                          background_color='white',
                          stopwords=stopwords,
                          max_words=150,
                          max_font_size=40, 
                          random_state=42
                         ).generate(str(df[df['Industry Vertical'] == 'Technology']['Investors Name']))

print(wordcloud)
fig = plt.figure(1)
plt.imshow(wordcloud)
plt.axis('off')
plt.show()


# # WordCloud of startup names

# In[ ]:


plt.figure(figsize = (15,15))

stopwords = set(STOPWORDS)
wordcloud = WordCloud(
                          background_color='white',
                          stopwords=stopwords,
                          max_words=150,
                          max_font_size=40, 
                          random_state=42
                         ).generate(str(df['Startup Name']))

print(wordcloud)
fig = plt.figure(1)
plt.imshow(wordcloud)
plt.axis('off')
plt.show()


# > **Thankyou** for reading this Notebook.I will continue this EDA.

# In[ ]:




