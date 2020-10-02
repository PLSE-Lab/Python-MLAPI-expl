#!/usr/bin/env python
# coding: utf-8

# This data contains tweets from major incubators and accelerators. The total number of tweets is more than 24K
# 
# Objectives 
# * Tweets Summary
# * Organization with most tweets
# * Tweet with most retweets
# * Most used words 
# * Tweets per day 
# * Distribution of tweets per day/month
# * Relationship btween number of retweets and the time of day/day itself

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt 
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS
import datetime
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import warnings
warnings.filterwarnings('ignore')
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[ ]:


df = pd.read_excel('../input/tweets.xlsx',sheet='tweets')


# In[ ]:


df.head()


# In[ ]:


df.info()


# In[ ]:


pd.isnull(df).any()


# Our dataset has no null values. So we are good to go, otherwise we would have to fill them or drop them depending on the situation

# In[ ]:


df.describe()


# we can tell that our file has 24933 entries. The mean of the tweets is 11.082 and the maximum number of retweets is 79537. Let's find out which tweet it is

# In[ ]:


df[df['retweets']==79537]


# The tweet with the most reweets is from ActiveSpaces and was posted in June at 7:19 PM. It would be good to find out later if there is any relationship between the time a tweet is posted and the number of retweets. The tweet is about Alexa being instructed to buy something from Whole Foods

# Now let's group the organizations according to the number of tweets

# In[ ]:


plt.figure(figsize=(12,8))
sns.countplot(data=df,y='username')


# Seedstars tweeted the most in 2017 followed by TonyElumeluFDN and then MESTAfrica.

# In[ ]:


toptweeps= df.groupby('username')[['tweet ']].count()
toptweeps.sort_values('tweet ',ascending=False)[:10]


# Which organization had the most retweets?

# In[ ]:


topretweets= df.groupby('username')[['retweets']].sum()
topretweets.sort_values('retweets',ascending=False)[:10]


# ActiveSpaces with 1949 had the most retweets, at 104581 retweets. TonyElumeluFDN with 2622 tweets came in second with 40705 retweets in total. I would like to kniw whether there is any correlation between the number of tweets and the the number of retweets

# Let's look at the most used words

# In[ ]:


corpus = ' '.join(df['tweet '])
corpus = corpus.replace('.', '. ')
wordcloud = WordCloud(stopwords=STOPWORDS, background_color='white',width=2400,height=2000).generate(corpus)
plt.figure(figsize=(12,15))
plt.imshow(wordcloud)
plt.axis('off')
plt.show()


# The word https appears so many times. I believe this is because most of these organizations share many links to either articles on their blog or applications. The organizations either fund startups or incubate them, so its not surprising that the word **startup** comes out prominently. Other words used that relate to the ecosystem include **entrepreneur**, **innovation**, **idea**, **market**, **founder**, **fintech** among others. The world **Apply** also appears many times suggesting that the organizations have been recruiting throught the year.

# I would like to see what MESTAfrica tweeted about in 2017

# In[ ]:


mest = df[df['username']=='MESTAfrica']
corpu = ' '.join(df['tweet '])
corpu = corpu.replace('.', '. ')
wordcloud = WordCloud(stopwords=STOPWORDS, background_color='white',width=2400,height=2000).generate(corpu)
plt.figure(figsize=(12,15))
plt.imshow(wordcloud)
plt.axis('off')
plt.show()


# Once again the same words relating to the ecosystem are prominent. You can also notice that there is mention of other organizations such as **BongoHive**, **Activespaces**, **Seedstars**, **TonyElumeluFDN**, **iHub**, **C4Dlab**, **sbootcamp**, **TheLaunchLab** etc suggesting that there is a lot collaboration in the space. This can be confirmed by looking at the words used by TonyElumeluFDN where there is mention of other organizations such as MESTAfrica, Afrilabs,Seedstars,BongoHive etc

# In[ ]:


tony = df[df['username']=='TonyElumeluFDN']
corp = ' '.join(df['tweet '])
corp = corp.replace('.', '. ')
wordcloud = WordCloud(stopwords=STOPWORDS, background_color='white',width=2400,height=2000).generate(corp)
plt.figure(figsize=(12,15))
plt.imshow(wordcloud)
plt.axis('off')
plt.show()


# Let's look at a summary of MESTAfrica's tweets

# In[ ]:


mest.describe()


# MESTAfrica had 2184 tweets and the most retweeted tweet had 2157 retweets. I would love to know what that tweet was about

# In[ ]:


mest[mest['retweets']==2157]


# The tweet is from July 2017 at 11:49 RT @sundarpichai: Hello from Lagos! #GoogleforNigeria https://t.co/m5OKp3QE40. This is the same day that Sundai Pichai Google's CEO was in Nigeria. The Tweet is a retweet from Sundai Pichai Twitter account

# I would like to see the distribution of the tweets throughout the year

# In[ ]:


df2 = df
df2['date'] = df2['created_at'].map(lambda x: x.split(' ')[0])
df2['time'] = df2['created_at'].map(lambda x: x.split(' ')[-1])
del df2['created_at']
df2.head()


# In[ ]:


month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
day_order = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
df2= df[['tweet_id','date','time','tweet ','retweets','username']]
df2.head()


# In[ ]:


df2['month'] = df2['date'].apply(lambda x: month_order[int(x.split('-')[1]) - 1])
month_df = pd.DataFrame(df2['month'].value_counts()).reset_index()
month_df.columns = ['month', 'tweets']


# In[ ]:


plt.figure(figsize=(12,6))
plt.title("All Tweets Per Month")
sns.barplot(x='month', y='tweets', data=month_df, order=month_order)


# The months with the most tweets are October, March and June. 

# In[ ]:


def getday(x):
    year, month, day = (int(i) for i in x.split('-'))    
    answer = datetime.date(year, month, day).weekday()
    return day_order[answer]
df['day'] = df['date'].apply(getday)
day_df = pd.DataFrame(df['day'].value_counts()).reset_index()
day_df.columns = ['day', 'tweets']
plt.figure(figsize=(12,6))
plt.title("All Tweets Per Day")
sns.barplot(x='day', y='tweets', data=day_df, order=day_order)


# Most of the tweets were tweeted on Thursday

# 
# Looking at MESTAfrica Tweets

# In[ ]:


mesting = df2[df2['username']=='MESTAfrica']
month_mest = pd.DataFrame(mesting['month'].value_counts()).reset_index()
month_mest.columns = ['month', 'tweets']


# In[ ]:


plt.figure(figsize=(12,6))
plt.title("MESTAfrica  Tweets Per Month")
sns.barplot(x='month', y='tweets', data=month_mest, order=month_order)


# In[ ]:


month_mest.head()


# Most of the tweets were tweeted in November. Summing all the tweets for all the months gives you the total number of tweets for the year

# In[ ]:


month_mest['tweets'].sum()


# In[ ]:


def getday(x):
    year, month, day = (int(i) for i in x.split('-'))    
    answer = datetime.date(year, month, day).weekday()
    return day_order[answer]
df['day'] = mesting['date'].apply(getday)
day_df = pd.DataFrame(df['day'].value_counts()).reset_index()
day_df.columns = ['day', 'tweets']
plt.figure(figsize=(12,6))
plt.title("MESTAfrica Tweets Per Day")
sns.barplot(x='day', y='tweets', data=day_df, order=day_order)


# It appears that most of MESTAfrica's Tweets are posted on Friday and Wednesday.

# In[ ]:


retweets_df = mesting.groupby('month')['retweets'].sum().reset_index()
plt.figure(figsize=(12,6))
plt.title("MEST Retweets per month")
sns.barplot(x="month",y="retweets",data=retweets_df,order=month_order)


# July is the month MESTAfrica got the most retweets as we had seen earlier followed closely by March. It would also be nice to know which months get the most retweets generally from other organizations

# In[ ]:


retweets_gen = df2.groupby('month')['retweets'].sum().reset_index()
plt.figure(figsize=(12,6))
plt.title("All Retweets per month")
sns.barplot(x="month",y="retweets",data=retweets_gen,order=month_order)


# Looking at all the organizations in question we can conclude that the month of June had the most retweets followed closely by January.

# Which days of the week get the most retweets?

# In[ ]:


def getday(x):
    year, month, day = (int(i) for i in x.split('-'))    
    answer = datetime.date(year, month, day).weekday()
    return day_order[answer]
df['day'] = df['date'].apply(getday)
day_df = df.groupby('day')['retweets'].sum().reset_index()

plt.figure(figsize=(12,6))
plt.title("All Reweets Per Day")
sns.barplot(x='day', y='retweets', data=day_df, order=day_order)


# **Friday** happens to be the day that be the day that tweets were reweeted most. Surprised? No you are not

# In[ ]:


mest_days = df[df['username']=='MESTAfrica']


# In[ ]:


df_mest = df.groupby('day')['retweets'].sum().reset_index()

plt.figure(figsize=(12,6))
plt.title("MESTAfrica Reweets Per Day")
sns.barplot(x='day', y='retweets', data=day_df, order=day_order)


# The story is the same at MESTAfrica. The number of retweets go increase steadily from Monday to Friday and drop on Saturday and Sunday for obvious reasons. If you want more retweets on your posts you know which day you have to post. Let's look at the number of retweets per organization

# In[ ]:


tweets_retweets = df.groupby('username')['retweets'].sum().reset_index()
plt.figure(figsize=(12,6))
plt.title('Retweets per organization')
sns.barplot(x='retweets',y='username',data=tweets_retweets)


# Activespaces had the most retweets followed by TonyElumeluFDN and then MESTAfrica. Is there any relationship between the number of tweets and the number of retweets? Let's find out

# In[ ]:


by_retweets = df.groupby('username')['retweets'].sum().reset_index()
by_retweets.head()


# In[ ]:


by_tweets = df.groupby('username')['tweet '].count().reset_index()
by_tweets.head()


# In[ ]:


merged_df = pd.merge(by_retweets,by_tweets,how='left',left_on='username',right_on='username')
merged_df.head()


# In[ ]:


sns.jointplot(data=merged_df,x='tweet ',y='retweets',color='g')
plt.title('Relationship between no of tweets and no of retweets')


# A pearson r value of 0.51 suggests a postive relationship between the number of tweets and the number of retweets

# If there is something you would like included leave it in the comments and I will do my best to include it. Cheers!

# In[ ]:




