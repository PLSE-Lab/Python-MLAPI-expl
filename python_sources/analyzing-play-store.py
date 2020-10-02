#!/usr/bin/env python
# coding: utf-8

# !!__**(editing)**__!!

# In[ ]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# In[ ]:


df = pd.read_csv("../input/googleplaystore_user_reviews.csv")
#print(os.listdir("../input/googleplaystore_user_reviews.csv"))


# Let's see a sample

# In[ ]:


df.sample()


# In[ ]:


df.isna().sum()


# There are many Nan values

# In[ ]:


df.sample(5)


# I would like to remove Nan values and analyze only the data that has values. 

# In[ ]:


df = df[~df.Sentiment.isna()]
df.isna().sum()


# In[ ]:


len(df)


# In[ ]:


df.Sentiment.value_counts()


# * Usually, if the rating is Positive it's ok, but if the rating is negative, it means the App is worse. We should penalize more if the APP has negative rating. 
# * So I thought of Assigning +1 to Positive, -2 to Negative

# In[ ]:


scores = df.Sentiment.map({"Positive":1,"Negative":-2,"Neutral":0})
df.Sentiment = scores


# In[ ]:


df.sample(3)


# I would like to analyze only "Sentiment" attribute

# In[ ]:


df.drop(columns=['Translated_Review','Sentiment_Polarity','Sentiment_Subjectivity'],inplace=True)


# In[ ]:


df.head()


# In[ ]:


after_editing = df.groupby('App').sum().reset_index()
after_editing.sample(5)


# Let's see top 10 apps with best Sentiment Scores

# In[ ]:


after_editing.sort_values(by='Sentiment',ascending=False)[:10]


# 10 apps with worst Sentiment Scores

# In[ ]:


after_editing.sort_values(by='Sentiment')[:10]


# Keep this DF aside and let's focus on other data now.

# In[ ]:


df = pd.read_csv("../input/googleplaystore.csv")


# In[ ]:


df.sample()


# In[ ]:


df.shape


# Shape of data : (10841, 13) . Let's see how many unique APPS exist

# In[ ]:


len(df.App.unique())


# Let's check the data on APP : "Angry Birds Classic"

# In[ ]:


df[df.App == "Angry Birds Classic"]


# Only Reviews column has different values

# Let's drop duplicates

# In[ ]:


df.drop_duplicates(inplace=True)


# In[ ]:


len(df)


# Drop columns that are not required for analysis

# In[ ]:


df.drop(columns=['Current Ver','Android Ver','Content Rating','Genres','Price'], inplace = True)


# In[ ]:


df.sample(5)


# In[ ]:


df.dropna(inplace= True)


# In[ ]:


df.isna().sum()


# In[ ]:


len(df.App.unique())


# In[ ]:


df.shape


# In[ ]:


len(after_editing)


# In[ ]:


df['Sentiment'] = np.NaN


# In[ ]:


x = list(after_editing.App)
for i in range(len(df)):
    check = df.iloc[i,0]
    if check in x:
        df.iloc[i,-1] = int(after_editing[after_editing.App == check].Sentiment)


# In[ ]:


df[df.App == '10 Best Foods for You']


# In[ ]:


after_editing[after_editing.App == '10 Best Foods for You']


# In[ ]:


h = int(after_editing[after_editing.App == '10 Best Foods for You'].Sentiment)
h


# In[ ]:


df.isna().sum()


# In[ ]:


df.head(5)


# In[ ]:


df[df.App=="ROBLOX"]


# In[ ]:


df.drop(columns='Category',inplace=True)


# In[ ]:


len(df)


# In[ ]:


df.drop_duplicates(inplace=True)


# In[ ]:


len(df)


# In[ ]:


df.drop_duplicates(subset=['App'],inplace=True)


# In[ ]:


df[df.App=="8 Ball Pool"]


# In[ ]:


len(df)


# In[ ]:


df.isna().sum()


# In[ ]:


df.info()


# In[ ]:


df.sort_values(by='Rating')


# Top 5 Apps with best user review sentiments are

# In[ ]:


df.sort_values(by='Sentiment',ascending=False)[:5]


# Bottom 5 Apps with worst user review sentiments are

# In[ ]:


df[~df.Sentiment.isna()].sort_values(by='Sentiment')[:5]


# There are many apps where Ratings are very less.....

# In[ ]:


df.sort_values(by='Rating')[:5]


# In[ ]:


df.sort_values(by='Rating')[:-6:-1]
#As we see rating is more than 5.0 which is an outlier. So delete that record


# In[ ]:


df = df[df.Rating != 19.0]


# In[ ]:


df.sort_values(by='Rating')[:-6:-1]


# In[ ]:


df['Reviews'].sample(10)


# Lets neglect those apps, which have less than 1000 Reviews.
# 

# In[ ]:


df = df[df.Reviews.astype('int') > 1000 ]


# In[ ]:


df.sample(5)

