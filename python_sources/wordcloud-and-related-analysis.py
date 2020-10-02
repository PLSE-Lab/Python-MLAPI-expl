#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
import matplotlib.pyplot as plt
from os import path
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

# Any results you write to the current directory are saved as output.


# **Now Load the data**

# In[ ]:


data = pd.read_csv('../input/mbti_1.csv')


# **Get the first 5 rows of data to see what it looks like**

# In[ ]:


data.head()


# **Use describe() to get counts and stats for data set**

# In[ ]:


data.describe()


# **Get the data from the first post**

# In[ ]:


text = data.posts[0]


# **Create a Wordcloud of the first post**

# In[ ]:


wordcloud = WordCloud().generate(text)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()


# **The text below will show the first 1000 posts**

# In[ ]:


fullText = data.posts[1000]


# **Display the generated image:**

# In[ ]:


wordcloud = WordCloud().generate(fullText)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()


# **Group by Briggs personality type**

# In[ ]:


descType = data.groupby("type")


# **Describe by type**

# In[ ]:


descType.describe()


# **Get unique list of values in type column**

# In[ ]:


names = data['type'].unique()


# # Example of getting posts by type

# In[ ]:


enfj = " ".join(review for review in data[data["type"]=="ENFJ"].posts)


# # Remove stopwords

# In[ ]:


stopWords = set(STOPWORDS)
#add each lower case personality type as a stopword
for word in names:
    word.lower()
    stopWords.add(word.lower())
#Remove all of the most common words
moreWords =['lot', 'time', 'love', 'actually', 'seem', 'need', 'infj', 'actually', 'pretty', 'sure', 'thought','type', 'one', 'even', 'someone', 'thing','make', 
            'now', 'see', 'things', 'feel', 'think', 'i', 'people', 'know', '-', "much", "something", "will", "find", "go", "going", "need", 'still', 'though', 
            'always', 'through', 'lot', 'time', 'love', 'really', 'want', 'way', 'never', 'find', 'say', 'it.', 'good', 'me.', 'many', 'first', 'wp', 'go', 
            'really', 'much', 'why', 'youtube', 'right', 'know', 'want', 'tumblr', 'great', 'say', 'well', 'people', 'will', 'something', 'way', 'sure', 
            'especially', 'thank', 'friend', 'good', 'ye', 'person', 'https', 'watch', 'yes', 'got', 'take', 'person', 'life', 'might', 'me', 'me,', 'around', 'best', 'try', 
            'maybe', 'probability', 'usually', 'sometimes', 'trying', 'read', 'us', 'may', 'use', 'work', ':)', 'said', 'two', 'makes', 'little', 'quite','fe', 'u', 'intps', 'probably', 'made', 'it', 'seems', 'look', 'yeah',
           'different', 'come', 'it,', 'friends', 'entps', 'different', 'esfjs', 'look', 'infjs', 'estps', 'kind', 'intjs', 'enfjs', 
            'entjs', 'infps', 'every', 'long', 'tell', 'new', 'jpg']

for i in moreWords:
    stopWords.add(i)


# # Now make a bar chart of how often each personality type is mentioned in a post

# In[ ]:


#get count of each personality type
data["type"].value_counts()
data["type"].value_counts().plot(kind="bar")


# # Make word clouds for each personality type - get each type - then filter by the personality type
# 

# In[ ]:


names = data['type'].unique()
i = 0
while i < len(names):
    for name in names:
        print(name)
        #filter by type
        specRows = data['type'] == name
        #combine the rows by type
        nameReturn = "".join(post for post in data[data["type"]== name].posts)
        nameReturn = nameReturn.lower()
        #make into a list to use comprehension to remove stopwords
        split = nameReturn.split()
        filtered_list = [word for word in split if word not in stopWords]
        filtered_words = "".join(filtered_list)
        #collocations = false to prevent duplicate words
        wordcloud = WordCloud(stopwords=stopWords, collocations=False).generate(filtered_words)
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        plt.show()
        i +=1                                      


# # Now go through all the personality type names to get data

# In[ ]:



from collections import Counter 
import nltk
from nltk.corpus import stopwords
names = data['type'].unique()
t=0     
while t < len(names):
    for name in names:
        #combine the rows by type
        nameReturn = "".join(post for post in data[data["type"]== name].posts)
        nameReturn = nameReturn.lower()
        print('The top 10 words for the ' + name + ' personality are')
        split = nameReturn.split() 
        filtered_words = [word for word in split if word not in stopWords]
        counter = Counter(filtered_words)
        most_occur = counter.most_common(10) 
        print(name)
        df = pd.DataFrame(most_occur, columns = ['Word', 'Count'])
        df.plot.bar(x='Word',y='Count', title=name)
        print(most_occur)
        t+=1


# Now make a chart to see the words visually

# # Create a graph to illustrate the count of top 10 most common words over all personality types

# In[ ]:




counter = Counter(filtered_words)
most_overall = counter.most_common(10) 
df = pd.DataFrame(most_overall, columns = ['Word', 'Count'])
df.plot.bar(x='Word',y='Count', title='Overall top words')

