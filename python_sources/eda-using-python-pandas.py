#!/usr/bin/env python
# coding: utf-8

# <p>&nbsp;</p>
# <img src="https://1000logos.net/wp-content/uploads/2017/05/Reddit-logo.png" width=400>
# <p>&nbsp;</p>

# ## Introduction
# 
# This is a brief exploratory data analysis using Pandas for a given public sample of random Reddit posts.
# We will get a feel of a dataset and try to answer the following questions: 
# * What are the most popular reddits? Which topics are viral?
# * Which posts have been removed and why? 
# * What % removed reddits are deleted by moderatos? 
# * Who are the most popular authors? 
# * Who are the biggest spammers at Reddit platform?
# 

# In[ ]:


#Getting all the packages we need: 

import numpy as np # linear algebra
import pandas as pd # data processing

import seaborn as sns #statist graph package
import matplotlib.pyplot as plt #plot package

import wordcloud #will use for the word cloud plot
from wordcloud import WordCloud, STOPWORDS # optional to filter out the stopwords

#Optional helpful plot stypes:
plt.style.use('bmh') #setting up 'bmh' as "Bayesian Methods for Hackers" style sheet
#plt.style.use('ggplot') #R ggplot stype


# ## <a name="read"></a>Reading the dataset
# Accessing Reddit dataset:

# In[ ]:


df = pd.read_csv('/kaggle/input/dataisbeautiful/r_dataisbeautiful_posts.csv')


# In[ ]:


df.sample(5)


# ## <a name="feel"></a>Getting a feel of the dataset
# Let's run basic dataframe exploratory commands

# In[ ]:


df.info()
df.describe()


# In[ ]:


print("Data shape :",df.shape)


# In[ ]:


#Empty values:

df.isnull().sum().sort_values(ascending = False)


# We note from the table above:
# - There are `173,611` entries in the dataset. Caveat, not all columns in the dataset are complete. 
# - The average reddit score `193`. The median value for the score is `1`, which means that a half of reddits in our dataset have the score `0` or `1` and only less than 75% reddits have the score more than `5`
# - The most popular reddit has `18,801` comments, while the average is `25` and the median is `1`. 

# ## <a name="corr"></a>Removed reddits deep dive

# Let's see who and why removes posts:

# In[ ]:


sns.countplot(x = 'removed_by', hue = 'removed_by', data = df)
#df.removed_by


# >As we can see, the most deleted posts (68%) were removed by moderator. Less than 1% are deleted by authors.
# 

# ## <a name="corr"></a>The most popular reddits

# ## <a name="corr"></a>The most common words in reddits:
# 
# Let's see the word map of the most commonly used words from reddit titles:

# In[ ]:


#To build a wordcloud, we have to remove NULL values first:
df["title"] = df["title"].fillna(value="")


# In[ ]:


#Now let's add a string value instead to make our Series clean:
word_string=" ".join(df['title'].str.lower())

#word_string


# In[ ]:


#And - plotting:

plt.figure(figsize=(15,15))
wc = WordCloud(background_color="purple", stopwords = STOPWORDS, max_words=2000, max_font_size= 300,  width=1600, height=800)
wc.generate(word_string)

plt.imshow(wc.recolor( colormap= 'viridis' , random_state=17), interpolation="bilinear")
plt.axis('off')


# ## <a name="corr"></a>Comments distribution
# 

# >The average reddit has less than 25 comments. Let's see the comment distribution for those reddits who have <25 comments:

# In[ ]:


#Comments distribution plot:

fig, ax = plt.subplots()
_ = sns.distplot(df[df["num_comments"] < 25]["num_comments"], kde=False, rug=False, hist_kws={'alpha': 1}, ax=ax)
_ = ax.set(xlabel="num_comments", ylabel="id")

plt.ylabel("Number of reddits")
plt.xlabel("Comments")

plt.show()


# >As we can see, the most reddits have less than 5 comments. 

# ## <a name="corr"></a>Correlation between dataset variables
# 
# Now let's see how the dataset variables are correlated with each other:
# * How score and comments are correlated? 
# * Do they increase and decrease together (positive correlation)? 
# * Does one of them increase when the other decrease and vice versa (negative correlation)? Or are they not correlated?
# 
# Correlation is represented as a value between -1 and +1 where +1 denotes the highest positive correlation, -1 denotes the highest negative correlation, and 0 denotes that there is no correlation.
# 
# * Let's see the correlation table between our dataset variables (numerical and boolean variables only)

# In[ ]:


df.corr()


# We see that score and number of comments are highly positively correlated with a correlation value of 0.6. 
# 
# There is some positive correlation of 0.2 between total awards received and score (0.2) and num_comments (0.1).
# 
# Now let's visualize the correlation table above using a heatmap
# 

# In[ ]:


h_labels = [x.replace('_', ' ').title() for x in 
            list(df.select_dtypes(include=['number', 'bool']).columns.values)]

fig, ax = plt.subplots(figsize=(10,6))
_ = sns.heatmap(df.corr(), annot=True, xticklabels=h_labels, yticklabels=h_labels, cmap=sns.cubehelix_palette(as_cmap=True), ax=ax)


# ## <a name="corr"></a>Score distribution
# 

# In[ ]:


df.score.describe()


# In[ ]:


df.score.median()


# In[ ]:


#Score distribution: 

fig, ax = plt.subplots()
_ = sns.distplot(df[df["score"] < 22]["score"], kde=False, hist_kws={'alpha': 1}, ax=ax)
_ = ax.set(xlabel="score", ylabel="No. of reddits")

