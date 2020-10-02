#!/usr/bin/env python
# coding: utf-8

# **Sentiment Analysis of ITEM_DESCRIPTION in the mercari data using NLTK AND TEXTBLOB**
# 
# Sentiment Analysis provides another crucial insight to an item in the MERCARI PRICE CHALLENGE, using the item_description column provided, we can understand if the desc is positive , neutral or negative. Which is necessary to suggest the price of an item.  

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt

# Any results you write to the current directory are saved as output.


# In[ ]:


#reading in the test and training data
traindata=pd.read_csv( "../input/train.tsv",sep='\t')


# **LETS DO A BASIC EDA **

# In[ ]:


traindata.head(5)


# In[ ]:


traindata.shape


# In[ ]:


item_desc=traindata['item_description']


# In[ ]:


traindata['item_condition_id'].unique()


# In[ ]:


stopwords = set(STOPWORDS)


# In[ ]:


wordcloud = WordCloud(
                          background_color='black',
                          stopwords=stopwords,
                          max_words=100,
                          max_font_size=40, 
                          random_state=42
                         ).generate(str(item_desc))


# In[ ]:


print(wordcloud)
fig = plt.figure(1)
plt.imshow(wordcloud)
plt.axis('off')
plt.show()
fig.savefig("word1.png", dpi=900)


# WE SEE THAT THE WORDS LIKE SIZE , CONDITION, BRAND , EXCELLENT HAS THE HIGHEST FREQUENCY IN THE DESCRIPTION
# 

# **Looking into a few desc according to the item_condition provided against it see if there is any relevance between them**

# In[ ]:


ones=traindata[traindata['item_condition_id']==1]['item_description']
ones.head()


# In[ ]:


wordcloud = WordCloud(
                          background_color='black',
                          stopwords=stopwords,
                          max_words=100,
                          max_font_size=40, 
                          random_state=42
                         ).generate(str(ones))
print(wordcloud)
fig = plt.figure(1)
plt.imshow(wordcloud)
plt.axis('off')
plt.show()
fig.savefig("word1.png", dpi=900)


# In[ ]:


twos=traindata[traindata['item_condition_id']==2]['item_description']
twos.head()


# In[ ]:


wordcloud = WordCloud(
                          background_color='black',
                          stopwords=stopwords,
                          max_words=100,
                          max_font_size=40, 
                          random_state=42
                         ).generate(str(twos))
print(wordcloud)
fig = plt.figure(1)
plt.imshow(wordcloud)
plt.axis('off')
plt.show()
fig.savefig("word1.png", dpi=900)


# In[ ]:


threes=traindata[traindata['item_condition_id']==2]['item_description']
threes.head()


# In[ ]:


wordcloud = WordCloud(
                          background_color='black',
                          stopwords=stopwords,
                          max_words=100,
                          max_font_size=40, 
                          random_state=42
                         ).generate(str(threes))
print(wordcloud)
fig = plt.figure(1)
plt.imshow(wordcloud)
plt.axis('off')
plt.show()
fig.savefig("word1.png", dpi=900)


# In[ ]:


fours=traindata[traindata['item_condition_id']==4]['item_description']
fours.head()


# In[ ]:


wordcloud = WordCloud(
                          background_color='black',
                          stopwords=stopwords,
                          max_words=100,
                          max_font_size=40, 
                          random_state=42
                         ).generate(str(fours))
print(wordcloud)
fig = plt.figure(1)
plt.imshow(wordcloud)
plt.axis('off')
plt.show()
fig.savefig("word1.png", dpi=900)


# In[ ]:


fives=traindata[traindata['item_condition_id']==5]['item_description']
fives.head()


# In[ ]:


wordcloud = WordCloud(
                          background_color='black',
                          stopwords=stopwords,
                          max_words=100,
                          max_font_size=40, 
                          random_state=42
                         ).generate(str(fives))
print(wordcloud)
fig = plt.figure(1)
plt.imshow(wordcloud)
plt.axis('off')
plt.show()
fig.savefig("word1.png", dpi=900)


# **We find the desc seems to justify the item_condition . This is a good sign.**

# In[ ]:


len(item_desc)


# **Importing SentimentIntensityAnalyzer from nltk library as it provides valence along with the polarity of a
# paragraph**

# In[ ]:



from nltk.sentiment.vader import SentimentIntensityAnalyzer
analyser = SentimentIntensityAnalyzer()
#TESING THE ANALYSER FOR A SAMPLE OUTPUT
result = analyser.polarity_scores("hi i am good")
result


# **We find there are 4 parts to the result. The numbers shows the sentiment probabilty of the passed paragraph.
# Where compound the overall sentiment . A positive compound value tells it was a positive sentiment , while a negative for a negative sentiment. The intensity of the value gives the valence of the sentiment .  **
# 
# 

# In[ ]:


item_senti=[]


# **Itterating over all the descriptions to find their sentiment.**

# In[ ]:


#NOT GOING THROUGHT THE ENTIRE DATA TO FINISH EXECUTION FASTER
for desc in range(int(len(item_desc)/3)):
    senti = analyser.polarity_scores(str(item_desc[desc]))
    item_senti.append(senti['compound'])
    
    
    


# In[ ]:


item_senti[:5]


# In[ ]:


len(item_senti)


# In[ ]:


df=pd.DataFrame({'item_description':item_desc[:len(item_senti)],'sentiment':item_senti})


# In[ ]:


df.head()


# **Performing sentiment analysis using TEXTBLOB **

# In[ ]:


from textblob import TextBlob


# In[ ]:


text="tHIS IS the best SHIRT"
blob = TextBlob(text)
blob.sentiment.polarity


# In[ ]:


textblob_senti=[]


# In[ ]:


for desc in range(int(len(item_desc)/3)):
    senti = TextBlob(str(item_desc[desc]))
    textblob_senti.append(senti.sentiment.polarity)


# In[ ]:


textblob_senti[:5]


# In[ ]:


df2=pd.DataFrame({'item_description':item_desc[:len(textblob_senti)],'sentiment':textblob_senti})


# In[ ]:


df2.head()


# In[ ]:


df2[df2.sentiment<-0.5].head()


# **THE POLARITY PROVIDED BY BOTH NLTK AND TEXTBLOB SEEM TO VARY QUITE A LOT . FOR THIS PARTICULAR DESCRIPTION I FOUND TEXTBLOB TO BE A BETTER FIT. **

# **Since I am mostly interested in knowing if a description is positive , neutral or negative . I am going to segment the values in 3 parts.**

# In[ ]:


NEG_SEG=-0.3   #anything less than this value is negative
POS_SEG=0.4    #anyting greater than this value is positive
#here the values in the range of  -0.3  - 0.4 will be neutral 


# In[ ]:


for i in range(int(len(item_desc)/3)):
    if item_senti[i]<=POS_SEG and item_senti[i]>=NEG_SEG:
        item_senti[i]=0
    elif item_senti[i]<NEG_SEG:
        item_senti[i]=-1
    elif item_senti[i]>POS_SEG:
        item_senti[i]=1
        
        
        
    

