#!/usr/bin/env python
# coding: utf-8

# # Here I will give the simple solution for tweet text extraction using SentimentIntensityAnalyzer.

# This method includes following 2 steps
# * First split the tweet into words.
# * Finding the word with highest intensity of sentiment using SentimentIntensityAnalyzer

# In[ ]:


import pandas as pd
import scipy.io
from array import *
import numpy as np
import re
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer


# In[ ]:


df_train = pd.read_csv ('../input/tweet-sentiment-extraction/train.csv')
df_train.head()


# In[ ]:


df_test = pd.read_csv ('../input/tweet-sentiment-extraction/test.csv')
df_test.head()


# In[ ]:


def unique_list(l):
    ulist = []
    [ulist.append(x) for x in l if x not in ulist]
    return ulist


# # Below code will get the words with highest polarity **

# In[ ]:


def choosing_selectedword(df_process):
    train_data = df_process['text']
    train_data_sentiment = df_process['sentiment']
    selected_text_processed = []
    analyser = SentimentIntensityAnalyzer()
    for j in range(0 , len(train_data)):
        text = re.sub(r'http\S+', '', str(train_data.iloc[j]))
        if(train_data_sentiment.iloc[j] == "neutral" or len(text.split()) < 2):
            selected_text_processed.append(str(text))
        if(train_data_sentiment.iloc[j] == "positive" and len(text.split()) >= 2):
            aa = re.split(' ', text)
        
            ss_arr = ""
            polar = 0
            for qa in range(0,len(aa)):
                score = analyser.polarity_scores(aa[qa])
                if score['compound'] >polar:
                    polar = score['compound']
                    ss_arr = aa[qa]
            if len(ss_arr) != 0:
                selected_text_processed.append(ss_arr)   
            if len(ss_arr) == 0:
                selected_text_processed.append(text)
        if(train_data_sentiment.iloc[j] == "negative"and len(text.split()) >= 2):
            aa = re.split(' ', text)
        
            ss_arr = ""
            polar = 0
            for qa in range(0,len(aa)):
                score = analyser.polarity_scores(aa[qa])
                if score['compound'] <polar:
                    polar = score['compound']
                    ss_arr = aa[qa]
            if len(ss_arr) != 0:
                selected_text_processed.append(ss_arr)   
            if len(ss_arr) == 0:
                selected_text_processed.append(text)  
    return selected_text_processed


# In[ ]:


selected_text_train = choosing_selectedword(df_train)


# In[ ]:


def jaccard(str1, str2): 
    a = set(str1.lower().split()) 
    b = set(str2.lower().split())
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))


# # Checking training data accuracey

# In[ ]:


train_selected_data = df_train['selected_text']
average = 0;
for i in range(0,len(train_selected_data)):
    ja_s = jaccard(str(selected_text_train[i]),str(train_selected_data[i]))
    average = ja_s+average
print('Training Data accuracey')
print(average/len(selected_text_train))


# In[ ]:


selected_text_test = choosing_selectedword(df_test)


# In[ ]:


df_textid = df_test['textID']
text_id_list = []
for kk in range(0,len(df_textid)):
    text_id_list.append(df_textid.iloc[kk])
df_sub = pd.DataFrame({'textID':text_id_list,'selected_text':selected_text_test})
df_sub.head()


# In[ ]:


df_sub.to_csv('submission.csv',index=False)

