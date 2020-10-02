#!/usr/bin/env python
# coding: utf-8

# ## Regular Expression
# 
# A regular expression is a special sequence of characters that helps you match or find other strings or sets of strings, The Python module re provides full support for Perl-like regular expressions in Python.
# 
# Some of the common uses of regular expressions are:
# 
# - Search a string
# - Finding a string
# - Replace part of a string
# 

# # Applications of Regular Expressions
# 
# - Segmentation of words from Sentences
# 
# - Segmentation of sentence from Paragraph
# 
# - Text Cleaning - ( Noise Removal )
# 
# - Information retrival from text ( ex: -Chatot, News Dataset etc. )
# 

# # Types of Regular Expression
# 
# - \   Used to drop the special meaning of character
#     following it (discussed below)
# - []  Represent a character class
# - ^   Matches the beginning
# - $   Matches the end
# - .   Matches any character except newline
# - ?   Matches zero or one occurrence.
# - |   Means OR (Matches with any of the characters
#     separated by it.
# - *   Any number of occurrences (including 0 occurrences)
# - +   One or more occurrences
# - {}  Indicate number of occurrences of a preceding RE 
#     to match.
# - ()  Enclose a group of REs

# # Regular Expression Function
# 
# - Match : Find the first occurance of pattern in the string
# - Search : Locates the pattern in the string
# - Findall : Find all occurance of the string
# - sub : Search and replace of the string
# - split : Split the text by the given regular expression pattern

# # Implementation of Regular Expression 

# In[ ]:


import re
string = "tiger is the national animal of india "
pattern = "tiger"

# re.match function work on the only first function of the string
me = re.match(pattern, string)  
print(me)


# In[ ]:


import re
string = "tiger is the national animal of india "
pattern = "tiger"
pattern2 = "lion"

# re.match function work on the only first function of the string
me = re.match(pattern2, string)  
print(me)


# In[ ]:


string = "tiger is the national animal of india "
pattern = "national"

# re.search function works on the searc any where of the string.
me = re.search(pattern, string)
print(me)


# In[ ]:


string = "tiger is the national animal of india "
pattern = "national"
print(me.group(0))


# In[ ]:


string = "tiger is the national animal of india tiger is the national animal of india"
pattern = "national"

me = re.findall(pattern, string)
print(me)


# In[ ]:


me = re.finditer(pattern, string)   # ITER function returns  the indexes of the function present in the string.
for m in me:
    print(m.start())


# In[ ]:


string = "Ron was born on 12-09-1992 and he was addmited to school 15-12-1999"
pattern = "\d{2}-\d{2}-\d{4}"                          # we will use  while card specil character
me = re.findall(pattern, string)
print(me)


# In[ ]:


print(re.sub(pattern, "Monday", string))  # 


# # importing the Dataset

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


path = "/kaggle/input/tweets.csv"


# In[ ]:


dataset = pd.read_csv(path, encoding = "ISO-8859-1")

dataset.head()


# In[ ]:


for index, tweet in enumerate(dataset["text"][10:15]):
    print(index+1,".",tweet)


# # Regex for Cleaning Text Data
# 
# a) Removing RT
# 
# RT means that the given tweet is a retweet of another which is useful information, but fortunately it is already present in the isRetweet column of our dataset so we can get rid of it.

# In[ ]:


import re 

text = "RT @Joydas: Question in Narendra Modi App where PM is taking feedback if people support his #DeMonetization strategy https://t.co/pYgK8Rmg7r"
clean_text = re.sub(r"RT ", "", text)

print("Text before:\n", text)
print("Text after:\n", clean_text)


# b) Removing <U+..> like symbols
# 
# If you see the tweet 3 in the above example, there are strange symbols something of the sort <U+..> all over the place. We need to come up with a general Regex expression that will cover all such symbols. Let's break it down.

# In[ ]:


text = "@Jaggesh2 Bharat band on 28??<ed><U+00A0><U+00BD><ed><U+00B8><U+0082>Those who  are protesting #demonetization  are all different party leaders"
clean_text = re.sub(r"<U\+[A-Z0-9]+>", "", text)

print("Text before:\n", text)
print("Text after:\n", clean_text)


# Fixing the & and &
# 
# If you explore the tweets further, you'll see that there is & present in many tweets for example, RT @kanimozhi:

# In[ ]:


text = "RT @harshkkapoor: #DeMonetization survey results after 24 hours 5Lacs opinions Amazing response &amp; Commitment in fight against Blackmoney"
clean_text = re.sub(r"&amp;", "&", text)

print("Text before:\n", text)
print("Text after:\n", clean_text)


# # Regex for Text Data Extraction
# 
# a. Extracting platform type of tweets
# 
# Apart from cleaning text data, regex can be used effectively to extract information from given text data. For example, we extracted dates from text in the video module. But, Regex can be used creatively to make new features.
# 
# Take an example of the statusSource column in the dataset. If you look closely, you will find that you can find out more about the platform(android/iphone/web/windows phone) used for the given tweet. Information like this can be very useful for our machine learning model.

# In[ ]:


#List platforms that have more than 100 tweets
platform_count = dataset["statusSource"].value_counts()
top_platforms = platform_count.loc[platform_count>100]
top_platforms


# These are the platforms with atleast 100 tweets each. Now we can use our Regex to extract platform name from between .. HTML tags. Let's extract our platform names.

# In[ ]:


def platform_type(x):
    ser = re.search( r"android|iphone|web|windows|mobile|google|facebook|ipad|tweetdeck|onlywire", x, re.IGNORECASE)
    if ser:
        return ser.group()
    else:
        return None

#reset index of the series
top_platforms = top_platforms.reset_index()["index"]

#extract platform types
top_platforms.apply(lambda x: platform_type(x))


# **b. Extracting hashtags from the tweets**
# 
# Hashtags usually convey important information in social media related texts. Using regex, we can easily extract hashtags from each tweet

# In[ ]:


text = "RT @Atheist_Krishna: The effect of #Demonetization !!\r\n. https://t.co/A8of7zh2f5"
hashtag = re.search(r"#\w+", text)

print("Tweet:\n", text)
print("Hashtag:\n", hashtag.group())


# In[ ]:


text = """RT @kapil_kausik: #Doltiwal I mean #JaiChandKejriwal is "hurt" by #Demonetization as the same has rendered USELESS <ed><U+00A0><U+00BD><ed><U+00B1><U+0089> "acquired funds" No wo"""
hashtags = re.findall(r"#\w+", text)

print("Tweet:\n", text)
print("Hashtag:\n", hashtags)


# In[ ]:


text = """@Joydas: Question in Narendra Modi App where PM is taking feedback if people support his #DeMonetization strategy https://t.co/pYgK8Rmg7r"""
remove = re.sub(r'^https?:\/\/.*[\r\n]*', '', text, flags=re.MULTILINE)
print("Remove:\n", remove)


# # Consider an Upvote if you like it !
