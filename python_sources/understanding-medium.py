#!/usr/bin/env python
# coding: utf-8

# In[20]:


import pandas as pd
import numpy as np
import nltk


# ## Reading Data

# In[21]:


df = pd.read_csv('../input/articles.csv')


# In[22]:


df.describe()


# In[23]:


df.info()


# ## Data cleaning and preprocessing

# In[24]:


df.head()


# In[25]:


df['len_text'] = df['text'].str.len()


# In[26]:


df['len_title'] = df['title'].str.len()


# ## EDA

# In[27]:


df.head()


# In[28]:


import matplotlib.pyplot as plt
import seaborn as sns


# ### Converting claps to int

# In[29]:


df['claps'] = df['claps'].apply(lambda s: int(float(s[:-1]) * 1000) if s[-1] == 'K' else int(s))


# In[30]:


df.drop('link', axis = 1, inplace=True)


# In[31]:


sns.distplot(df['len_text'], color="b")
plt.show()


# **It is easy to observe that most of the article are in range 5000 to 15000 characters**

# In[32]:


sns.distplot(df['len_title'], color="b")
plt.show()


# **Above curve contains two local maxima one at about 60 chars and other at 100 char which is quite fascinating as It implies that a specific group of writers prefer to write longer title **

# In[33]:


sns.distplot(df['claps'], color="b")
plt.show()


# **The distribution of claps entails a fact that it is highly skewed in the right side which clearly shows that a small class of authors get high amount of claps**

# In[34]:


df.head()


# ### Let's check relation between claps recieved and reading time of  the article

# In[35]:


sns.pointplot('reading_time', 'claps', data=df)
plt.show()


# In[36]:


sns.regplot('reading_time', 'claps', data=df, order=3)
plt.show()


# **As we can see It is beneficial to have reading time between 12 minutes and 18 minutes.It shows that readers usually inclined to read article which is not too short so that it does not contain any useful informations and not too long that it deaden their intreast **

# In[37]:


sns.regplot('len_text', 'claps', data=df, order=3)
plt.show()


# **It is easy to observe that it is smart to write articles 15000 chars and 25000 chars.**

# In[38]:


a4_dims = (25.7, 40.27)
fig, ax = plt.subplots(figsize=a4_dims)
sns.barplot('len_title', 'claps', data=df, orient='h')
plt.show()


# In[39]:


## Let's check relation between title length and article length
sns.lmplot('len_title', 'len_text', data=df,order=3)
plt.show()


# In[40]:


## Let's check relation between total claps recieved and article length
a4_dims = (12, 6)
fig, ax = plt.subplots(figsize=a4_dims)
sns.pointplot('len_title', 'claps', data=df)
plt.show()


# In[41]:


sns.heatmap(df[['claps', 'len_text', 'len_title', 'reading_time']].corr(),annot=True, cmap='BrBG')
plt.show()


# **Heatmap is quite obvious. Their has to be relation between reading time and text lenght. But their is also good correlationship between text lenght and claps which shows that readers also care about length of text along with its quality**

# ### Analyse Author

# In[42]:


a4_dims = (25.7, 35.27)
fig, ax = plt.subplots(figsize=a4_dims)
sns.barplot('claps', 'author', data = df, orient='h')
plt.show()


# **Above curve confirms our previous hypothesis that readers favour some authors than others. It is logical to assume that famous ones or the successful ones tend to getting more attention **

# In[43]:


a4_dims = (25.7, 35.27)
fig, ax = plt.subplots(figsize=a4_dims)
sns.boxplot('len_text', 'author', data = df, orient='h')
plt.show()


# **The above curve shows that most of the authors have very low variance which cleary shows that each author have a pecular style.**

# In[44]:


a4_dims = (25.7, 35.27)
fig, ax = plt.subplots(figsize=a4_dims)
sns.barplot('len_text', 'author', data = df, orient='h')
plt.show()


# In[45]:


a4_dims = (25.7, 35.27)
fig, ax = plt.subplots(figsize=a4_dims)
sns.barplot('len_title', 'author', data = df, orient='h')
plt.show()


# In[46]:


## Finding top articls
df[df['claps'] >= df['claps'].quantile(0.95)][['author', 'title', 'claps']]


# In[47]:


df.head()


# ### Let's Analyse Title

# #### Convert every word to lowercase

# In[48]:


df['title'] = df['title'].apply(lambda x: " ".join(x.lower() for x in x.split()))
df['title'].head()


# #### Removing punctuations

# In[49]:


df['title'] = df['title'].str.replace('[^\w\s]','')
df['title'].head()


# #### Removing stop words

# In[50]:


from nltk.corpus import stopwords
stop = stopwords.words('english')


# In[51]:


df['title'] = df['title'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))


# In[52]:


df.head()


# In[53]:


def get_words_count(df_series, col):
    words_count = {}
    m = df_series.shape[0]
    for i in range(m):
        words = df[col].iat[i].split()
        for word in words:
            if word.lower() in words_count:
                words_count[word.lower()] += 1
            else:
                words_count[word.lower()] = 1
    return words_count


# In[54]:


title_words = get_words_count(df, 'title')


# In[55]:


title_words = pd.DataFrame(list(title_words.items()), columns=['words', 'count'])


# In[56]:


sns.distplot(title_words['count'], color='b')
plt.show()


# In[57]:


## List of 15 most frequent words occurred in title
title_words.sort_values(by='count', ascending=False).head(15)


# In[58]:


from wordcloud import WordCloud


# In[59]:


fig = plt.figure(dpi=100)


# In[60]:


a4_dims = (6, 12)
fig, ax = plt.subplots(figsize=a4_dims)
wordcloud = WordCloud(background_color ='white', max_words=200,max_font_size=40,random_state=3).generate(str(title_words.sort_values(by='count', ascending=False)['words'].values[:20]))
plt.imshow(wordcloud)
plt.title = 'Top Word in the title of Medium Articles'
plt.show()


# In[61]:


title_words.head()


# In[62]:


## Let's get list of top ten words
topten_title_words = title_words.sort_values(by='count', ascending=False)['words'].values[:10]


# In[63]:


## Count occurence of top ten words in every title in dataframe
df['topten_title_count'] = df['title'].apply(lambda s: sum(s.count(topten_title_words[i]) for  i in range(10)))


# In[64]:


df.head()


# In[65]:


sns.regplot('topten_title_count', 'claps', data = df, order=3)
plt.show()


# In[66]:


sns.barplot('topten_title_count', 'claps', data = df)
plt.show()


# **It clearly shows that As occurence of top ten words increases, the claps recieved also increases. Well Title writing is an art and with enough data one can comprehend it **

# In[67]:


sns.distplot(df['topten_title_count'], color='b')
plt.show()


# In[68]:


sns.heatmap(df[['topten_title_count', 'len_text', 'len_title']].corr(), annot=True, cmap='BrBG')
plt.show()


# In[69]:


sns.jointplot('topten_title_count', 'claps', data=df, kind='hex')
plt.show()


# ### Let's Analyse text data

# In[70]:


df.head()


# #### Convert every word to lowercase

# In[71]:


df['text'] = df['text'].apply(lambda x: " ".join(x.lower() for x in x.split()))
df['text'].head()


# #### Removing punctuations

# In[72]:


df['text'] = df['text'].str.replace('[^\w\s]','')
df['text'].head()


# #### Removing stop words

# In[73]:


df['text'] = df['text'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))


# In[74]:


df.head()


# In[75]:


text_words = get_words_count(df, 'text')


# In[76]:


text_words = pd.DataFrame(list(text_words.items()), columns=['words', 'count'])


# In[77]:


sns.distplot(text_words['count'], color='b')
plt.show()


# In[78]:


## Most frequent 15 words in articles
text_words.sort_values(by='count', ascending=False).head(15)


# In[79]:


fig = plt.figure(dpi=100)
a4_dims = (6, 12)
fig, ax = plt.subplots(figsize=a4_dims)
wordcloud = WordCloud(background_color ='white', max_words=200,max_font_size=40,random_state=3).generate(str(text_words.sort_values(by='count', ascending=False)['words'].values[:20]))
plt.imshow(wordcloud)
plt.title = 'Top Word in the text of Medium Articles'
plt.show()


# In[80]:


## get list of most frequent 10 words in text
topten_text_words = text_words.sort_values(by='count', ascending=False)['words'].values[:10]


# In[81]:


df['topten_text_count'] = df['text'].apply(lambda s: sum(s.count(topten_text_words[i]) for  i in range(10)))


# In[82]:


df.head()


# In[83]:


sns.regplot('topten_text_count', 'claps', data = df, order=3)
plt.show()


# In[84]:


a4_dims = (6, 25)
fig, ax = plt.subplots(figsize=a4_dims)
sns.barplot('claps', 'topten_text_count', data = df, orient='h')
plt.show()


# **Unlike in the case of title, using top most frequent words in text data inordinately can actually become perilous.**

# In[85]:


sns.distplot(df['topten_text_count'], color='b')
plt.show()


# In[86]:


sns.heatmap(df[['topten_text_count', 'len_text', 'len_title']].corr(), annot=True, cmap='BrBG')
plt.show()


# In[87]:


sns.jointplot('topten_text_count', 'claps', data=df, kind='hex')
plt.show()


# ## Let's go deeper

# In[88]:


df.head()


# ## What highest clap getters do differently

# #### Let's get top 30 authors

# In[89]:


df_author = df.groupby(['author']).mean().reset_index()


# In[90]:


df_top30 = df_author.sort_values(ascending=False, by='claps')[:30]


# In[91]:


a4_dims = (10, 10)
fig, ax = plt.subplots(figsize=a4_dims)
sns.barplot('reading_time', 'claps', data=df_top30)
plt.show()


# **As we can see most of them have reading time between 10-15 minutes**

# In[92]:


a4_dims = (10, 10)
fig, ax = plt.subplots(figsize=a4_dims)
sns.distplot(df_top30['len_text'])
plt.show()


# **Oh! That peak is revealing itself. Most of them write between 5000 to 15000 characters of text**

# In[93]:


sns.kdeplot(df_top30['topten_text_count'], df_top30['topten_title_count'], shade=True, cbar=True)
plt.show()


# **This plot is clearly depicting density of both the attributes. It can deduced that most of the top clap-getters tend to use top ten frequent words on title about 1-3 times and top ten most frequent text words about 50-100 times**

# In[94]:


## Relationship  between length of text with lenght of title 
sns.kdeplot(df_top30['len_text'], df_top30['len_title'], shade=True, cbar=True)
plt.show()


# #### Let's checkout the big picture

# In[95]:


sns.clustermap(df_top30[['reading_time', 'claps', 'len_title', 'len_text', 'topten_text_count', 'topten_title_count']],cmap="mako", robust=True)
plt.show()

