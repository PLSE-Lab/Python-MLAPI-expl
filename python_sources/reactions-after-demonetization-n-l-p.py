#!/usr/bin/env python
# coding: utf-8

# **Namastey(hello)**
# This kernal is regarding Natural Language Processing of the twitter's tweet that were written during the demonetization phase in india. what was the reaction of the people ? where they happy ? where they sad ? or confused......... lets check it out
# 
# **WHAT WILL WE USE?**
# 1. PANDAS
# 
# 2.NLTK
# 
# 3.TEXTBLOB(FOR SENTIMENT ANALYSIS)
# 
# **ATTRIBUTES IN DATASET**
# 'Unnamed: 0','X', 'text', 'favorited', 'favoriteCount', 'replyToSN',
#        'created', 'truncated', 'replyToSID', 'id', 'replyToUID',
#        'statusSource', 'screenName', 'retweetCount', 'isRetweet', 'retweeted'
#        
#   **WHAT WE WILL DO IN THIS KERNEL ???????**
# 
# CHECK FOR THE MOST FREQUENT WORD
#  
#  GET THE USER WITH MOST NUMBER OF TWEETS
#  
#  CLEANING OF THE DATA
#  
# SENTIMENT ANALYSIS ;) :( ?

# **what the dataset look likes**

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
df = pd.read_csv("../input/demonetization-tweets.csv",encoding="ISO-8859-1")
df.head(5)


# In[ ]:


#remove the unwanted columns
df.drop(['Unnamed: 0', 'X', 'favorited', 'favoriteCount', 'replyToSN',
       'created', 'truncated', 'replyToSID', 'id', 'replyToUID',
       'statusSource'],axis=1,inplace=True)


# **THE TWEET WITH MAXIMUM  NUMBER OF RETWEETS**

# In[ ]:


df.iloc[df['retweetCount'].idxmax()]['text'] #IDXMAX : GIVES THE INDEX OF THE MAXIMUM VALUE


# **HMMM ..... MOST OF THE PEOPLE  ARE SUPPORTING MODI **

# **LETS FIND THE USER WITH MAXIMUM NUMBER OF TWEETS**

# In[ ]:


import nltk
from nltk.probability import FreqDist
user_list = df['screenName'].tolist()
max_user = FreqDist(user_list)
max_user.plot(10)


# **TO PROCEED FURTHER LETS HAVE SOME DATA CLEANING**

# **ITS RECOMMENDED THAT YOU SHOULD FOLLOW THE CLEANING FUNCTION STEP BY STEP YOURSELF TO KNOW WHAT HAPPENING INSIDE**

# In[ ]:


import string,re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('brown')
#THE CLEANING FUNCTION
def clean_text(tweets):
    tweets = word_tokenize(tweets)#SEPERATE EACH WORD
    tweets = tweets[4:] #to remove RT@
    tweets= " ".join(tweets)#JOIN WORDS
    tweets=re.sub('https','',tweets)#REMOVE HTTPS TEXT WITH BLANK
    tweets = [char for char in tweets if char not in string.punctuation]#REMOVE PUNCTUATIONS 
    tweets = ''.join(tweets)#JOIN THE LETTERS
    tweets = [word for word in tweets.split() if word.lower() not in stopwords.words('english')]#REMOVE COMMON ENGLISH WORDS(I,YOU,WE...)
    return " ".join(tweets)

df['cleaned_text']=df['text'].apply(clean_text) #adding clean text to dataframe


# **FIND THE MOST FREQUENT WORD IN TWEETS**

# In[ ]:


clean_term = []
for terms in df['cleaned_text']:
    clean_term += terms.split(" ")
cleaned = FreqDist(clean_term)
cleaned.plot(10)


# **SAMPLE SHOWS WORD SUCH AS [DEMONETIZATION, INDIA , MODI, PM] FREQUENTLY OCCURS**

# **LETS PERFORM SENTIMENT ANALYSIS BUT WE NEED TEXTBLOB LIBRARY FOR SENTIMENT ANALYSIS**
# 
# **TEXTBLOB.POLARITY GIVES THE MEASURE OF SENTIMENT, ITS VALUE IS AROUND [-1 TO 1]**
# 
# ** 1 MEANS POSITIVE FEEDBACK**
# 
# **-1 MEANS NEGETIVE FEEDBACK**

# In[ ]:


from textblob import TextBlob
#CREAING A FUNCTION TO GET THE POLARITY
def sentiments(tweets):
    tweets = TextBlob(tweets)
    pol = tweets.polarity #GIVES THE VALUE OF POLARITY
    return pol

df["polarity"] = df["cleaned_text"].apply(sentiments) #APPLY FUNCTION ON CLEAN TWEETS


# **THE FINAL DATASET WITH CLEAN TWEETS AND POLARITY OF EACH TWEET**

# In[ ]:


df.head(5)


# **CHECKING SENTIMENTS**

# In[ ]:


print("THE AVERAGE POLARITY",np.mean(df["polarity"])) #gives the average sentiments of people
print("THE MOST -VE TWEET :",df.iloc[df['polarity'].idxmin()]['text'])# most positive
print("THE MOST +VE TWEET :",df.iloc[df['polarity'].idxmax()]['text'])#most negetive


# **WE HAVE GOT THE AVERAGE TWEET POLARITY AS 0.05 THAT IS JUST ABOVE NEUTRAL**
# **...?THAT MEAN THE PEOPLE GOTS A MIX REACTION AFTER THE NEWS OF DEMONETIZATION **

# **WILL CONTINUE FURTHER TILL THEN GIVE A THUMBS UP IF YOU LIKED IT <3**

# In[ ]:




