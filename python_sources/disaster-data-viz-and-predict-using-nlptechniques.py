#!/usr/bin/env python
# coding: utf-8

# # NLP With Disaster Tweets

# There is a gorwing use of smart phones around the world. Along with this we have seen a surge in people connecting with social media, there is lots of information shared with social media platforms itself. There are many ways we can put that data for a good use and this method is one such way. With growing use of smart phones twitter has turned into a major channel for announcing emergency situations around people in real time. But not all tweets are about emergency or disaster so the work to classify whether a tweet is about an emergency has come up and NLP is goning to play a major part in identifying such tweets. This oppurtunity has brought many agencies to programatically monitoring Twitter.

# ![tweet info](https://si.wsj.net/public/resources/images/NA-CJ408_TWITTS_9U_20160311133006.jpg)

# This datset has tweets which are labelled with disaster or not a disaster. Goal is to created a model which will classify tweets with best accuracy possible. Lets load the data and have some visualization of what the data contains and head for prediction of tweets. We are provided with three CSv files one for training, other for testing and another a sample submission csv which gives format to submit predictions on test set. 

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import matplotlib.pyplot as plt
import seaborn as sb
import re
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        
train = pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')
test = pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')


# > ## Data understanding and visualiation

# Here we will look at basic information and perform single variable(column) analysis. first lest check the info on dataset.

# ## INFO

# In[ ]:


train.info()


# There is a total of 7613 rows(entries) with 4 columns(features) other than **id**. 
# 1. **keyword** tells about a particular keyword from text which might relate to disaster. It contains 61(0.8%) missing values.
# 2. **location** tells about where the tweet was sent from. It contains 2533(33.27%) missing values.
# 3. **text** is the tweet made.
# 4. **target** tells whether the tweet is real disaster(1) or not(0). It has a split of 0 with 4342(57%) and 1 with 3271(43%)

# ## Keyword Column

# In[ ]:


top = train.groupby('keyword')['id'].count()
top = pd.DataFrame({'keyword':top.index,'count':top.values}).sort_values(by=['count']).tail(20)

bottom = train.groupby('keyword')['id'].count()
bottom = pd.DataFrame({'keyword':bottom.index,'count':bottom.values}).sort_values(by=['count']).head(20)

plt.figure(figsize=(12,10))

plt.subplot(211)
barlist = plt.bar(data=top, x = 'keyword',height = 'count',color = 'cadetblue')
plt.xticks(rotation = 20);
plt.ylabel('count')
plt.title('Top20 unique keywords')
barlist[0].set_color('darkgoldenrod');
barlist[2].set_color('indianred');
barlist[3].set_color('indianred');
barlist[15].set_color('darkgoldenrod');
barlist[9].set_color('darkslategrey');
barlist[18].set_color('darkseagreen');

plt.subplot(212)
barlist = plt.bar(data=bottom, x = 'keyword',height = 'count', color = 'cadetblue');
plt.xticks(rotation = 45);
plt.ylabel('count');
plt.title('Bottom20 unique keywords')
barlist[14].set_color('darkslategrey');
barlist[10].set_color('darkseagreen');

sb.despine(left = True, bottom  = True)
plt.tight_layout()

print(str(train['keyword'].nunique())+ ' total unique keywords')


# **Observations**
# 1. fatalities was the highest keyword with around 42 tweets containing keyword. radiation emergency was the least with around 9 tweets containing it.
# 2. From the top20 we can see wrecked and wreckage as different keywords but both mean the same, just tense is different. There are other keywords also present in same format eg: dead, death, annihilation, annihilated, sunk, sinking etc.
# 3. same colors are given to repeated words with different tenses
# 
# **Text handling to be done**
# 1. Repalce missing with None
# 2. lemmatize and change values with same meaninig into a single value.
# 3. repalce the '%20' in text with some thing else

# ## location column

# In[ ]:


top = train.groupby('location')['id'].count()
top = pd.DataFrame({'location':top.index,'count':top.values}).sort_values(by=['count']).tail(20)


plt.figure(figsize=(16,6))

barlist = plt.bar(data=top, x = 'location',height = 'count', color = 'cadetblue')
plt.xticks(rotation = 90);
plt.ylabel('count')
plt.title('Top20 unique locations')

barlist[1].set_color('darksalmon')
barlist[4].set_color('darksalmon')
barlist[3].set_color('peru')
barlist[18].set_color('peru')
barlist[17].set_color('dimgrey')
barlist[19].set_color('dimgrey')

sb.despine(left = True, bottom  = True)

print(str(train['location'].nunique())+ ' total unique locations')


# **Observations**
# 1. 3341 unique values in 7631 which is 43.5% of the data.
# 2. Even unique there are few values which have repeated like "United States" and "USA" represent the same. "New York, NY" and "New York" both are same.
# 3. This column can be a country like INDIA or a city in that country Mumbai. Because the tweets can be specific to city or generic to a country.
# 
# **Text Handling to be done**
# 1. Replace NaN with None
# 2. change the same locations into one eg : 'New York NY', 'New York' both are same.
# 3. Dont not replce city to country eg: 'Mumbai' and 'INDIA' because tweet can be specifict to city or to a country.

# ## text column

# We will check few text columns to see what different values are present in the text column and what to handle.

# In[ ]:


train.loc[5,'text']


# From the above we can see some things to handle
# 1. #tags looks like many columns have hashtags representing cities, incidents etc.
# 2. Special characters like ' -, => , . ' are also present to handle.

# In[ ]:


train.loc[31,'text']


# Thigns to handle
# 1. @ notations to handle
# 2. http links to handle

# In[ ]:


train.loc[38,'text']


# Things to handle
# 1. #tags presnet representing town
# 2. Special symbols are present might be emotes need to be removed
# 3. http link to be handeled
# 4. special characters present ' : , . ' to be handeled.

# ### From all the three different texts what we see and what to do
# 
# **Obeservations**
# 1. None of the text columns are empty so null values are not there.
# 2. text column contains some non text contents like http links, emotes.
# 3. #tags and @noattions are present. Also contains special characters
# 
# **Texthandling**
# 1. Handle the special characters ' :, =>, ., - ' etc and symbols like '\x89UO' which is an emote. Mostly remove them
# 2. Handle #tags and @notations. Can be converted into new columns.
# 3. links are present remove or create a new column.

# > ## Predition With NLP Techiques

# ## Clean the Code

# From all the observations made it looks lot of cleaning is required. lets list them
# 1. Repalce missing in keyword and location column with None
# 2. In keyword column lemmatize and change values with same meaninig into a single value.
# 3. IN keyword repalce the '%20' in text with some thing else
# 4. change the same locations into one eg : 'New York, NY' and 'New York' both are same.
# 5. #tags representing cities, incidents @ notations and http links to handle
# 6. Special characters like ' -, => , .,: ' are also present to handle.
# 7. Special symbols like '\x89UO' are present might be emotes need to be removed

# In[ ]:


def find_URL(text):
    url = re.compile(r'https?://\S+|www\.\S+')
    return (url.search(text) != None)

def clean_text(text):
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
    text = emoji_pattern.sub(r'', text)
    
    url = re.compile(r'https?://\S+|www\.\S+')
    text = url.sub(r'',text)
    
    text = text.replace('#',' ')
    text = text.replace('@',' ')
    symbols = re.compile(r'[^A-Za-z0-9 ]')
    text = symbols.sub(r'',text)
    
    return text

def lemma(text):
    txt1 = wordnet_lemmatizer.lemmatize(text,pos=wordnet.NOUN)
    txt2 = wordnet_lemmatizer.lemmatize(text,pos=wordnet.VERB)
    txt3 = wordnet_lemmatizer.lemmatize(text,pos=wordnet.ADJ)
    if(len(txt1) < len(txt2) and len(txt1) < len(txt3)): 
        text = txt1
    elif(len(txt2) < len(txt1) and len(txt2) < len(txt3)):
        text = txt2
    elif(len(txt3) < len(txt1) and len(txt3) < len(txt2)):
        text = txt3
    else:
        text = txt1   
    
    return text


# In[ ]:


#1. Replace missing values
train['keyword'].fillna('None', inplace=True)
train['location'].fillna('None', inplace=True)

#3. Replace %20 in the keyword column
train['keyword'] = train['keyword'].str.replace('%20','')

#4. location column handling
for ind in range(train.shape[0]):
    train.loc[ind,'location'] = train.loc[ind,'location'].split(',')[0]

#5,6,7. Text column handling
for ind in range(train.shape[0]):
    train.loc[ind,'tags_count'] = len(train.loc[ind,'text']) -  len(train.loc[ind,'text'].replace('#',''))
    train.loc[ind,'@_count'] = len(train.loc[ind,'text']) -  len(train.loc[ind,'text'].replace('@',''))
    train.loc[ind,'http_link'] =  find_URL(train.loc[ind,'text'])
    
train['text'] = train['text'].apply(lambda x: clean_text(x))
train.head(70)

#2 lemmatize keyword
train['keyword'] = train['keyword'].apply(lambda x: lemma(x))


train.head(10)


#                     ***Project under development code will be added soon.***
