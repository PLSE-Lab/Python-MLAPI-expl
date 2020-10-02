#!/usr/bin/env python
# coding: utf-8

# # Abstract
# This data provides about 1.1 million news headlines and their publish dates from 2003 to 2017. 
# Different big events happened every year,  such as Iraq War in 2003 and Finacial Crisis started from 2008.  These kinds of events are hot topics which should be  published in huge numbers. This would be reflected in the numbers of headlines that contain them. Different headlines that reported different  events may also contain different sentiments, So, there might be some correlations between events and sentiment.
#  
# goal
# 1.  Finding each year's 20 most frequent  words. 
# 2. Using Vader Sentiment Analysis Tool in NLTK to do sentiment analysis in each year's headlines.
# 3. Plotting graphs to show how the hot topics change with time and how they impacts the sentiment. 

# # Step1.Input Data
# 
# Since publish data contains month and day that are unnecessary and slow down our calculation, we use "text['publish_date']/10000" and "text['publish_date'] = text['publish_date'].astype(int)" to cut down them. 
# 
# 

# In[2]:


from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import pandas as pd
import random 
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt

text = pd.read_csv('../input/abcnews-date-text.csv')
text['publish_date'] = text['publish_date']/10000
text['publish_date'] = text['publish_date'].astype(int)
text.head()


# # Step2. Seekinging 20 Most Frequent Word Each Year
# 
# Spliting each year's headlines into words and filtering them with stopwords. Then lemmatizing and appending  them into lists.
# 

# In[3]:


stop_words = stopwords.words("english")
lemmatizer = WordNetLemmatizer()
result = []
start = 0
end = 0
for i in range(2003,2018):
    word_list = {}
    temp = text.loc[text['publish_date']==i]
    temp = temp['headline_text']
    start = end
    lenth = len(temp)
    end = end + lenth
    for j in range(start,end):
        token = temp[j].split()
        for w in token:
            if w not in stop_words:
                w = lemmatizer.lemmatize(w)
                if w not in word_list:
                    word_list[w]=1
                else:
                    word_list[w]+=1
    count_list = sorted(word_list.items(),key = lambda x:x[1],reverse = True)
    temp_list = list(zip(*count_list[0:20]))
    result.append(list(temp_list[0]))
    print(i)
    print(count_list[0:20])


# # Step3. Sentiment Analysis Each Year
# Using vader sentiment tool in nltk to analyze each headline's sentimental components and counted the annual average values. 

# In[5]:


se = SentimentIntensityAnalyzer()
neg_change = []
neu_change = []
pos_change = []
compound_change = []
start = 0
end = 0
for i in range(2003,2018):
    temp = text.loc[text['publish_date']==i]
    temp = temp['headline_text']
    start = end
    lenth = len(temp)
    end = end + lenth
    neg = 0.0
    pos = 0.0
    neu = 0.0
    compound = 0.0
    for j in range(start,end):
        Sentiment = se.polarity_scores(temp[j])
        neg = neg + Sentiment['neg']
        neu = neu + Sentiment['neu']
        pos = pos + Sentiment['pos']
        compound = compound + Sentiment['compound']
    neg_change.append(neg/lenth)
    pos_change.append(pos/lenth)
    neu_change.append(neu/lenth)
    compound_change.append(compound/lenth)
    print(i)
    print('neg:%-6.3f,neu:%-6.3f,pos:%-6.3f,compound:%-6.3f'%(neg/lenth,neu/lenth,pos/lenth,compound/lenth))


# # Step4. Visualization and Analysis
# 

# In[6]:


year = [i for i in range(2003,2018)]

stack_bottom = []
for i in range(0,len(neg_change)):
    stack_bottom.append(neg_change[i] + neu_change[i])
b1 = plt.bar(year, neg_change)
b2 = plt.bar(year, neu_change, bottom = neg_change)
b3 = plt.bar(year, pos_change, bottom = stack_bottom)

for i in year:
    k = i-2003
    for j in range(0,20):
        plt.text(i-0.3,0.85-0.03*(j+1) ,result[k][j])
plt.title('Sentiment Change Bars')
plt.xlabel('years')
plt.ylabel('sentiment rate')
plt.legend([b1,b2,b3],['neg','neu','pos'])
plt.gcf().set_size_inches(18,10)
plt.show()


# In[7]:


year = [i for i in range(2003,2018)]
l1 = plt.plot(year,neg_change,label='neg')
l2 = plt.plot(year,neu_change,label='neu')
l3 = plt.plot(year,pos_change,label='pos')
for i in year:
    k = i-2003
    for j in range(0,20):
        plt.text(i-0.2,0.85-0.03*(j+1) ,result[k][j])
plt.title('Sentiment Change Curves')
plt.xlabel('years')
plt.ylabel('sentiment rate')
plt.legend([b1,b2,b3],['neg','neu','pos'],loc='lower left')
plt.gcf().set_size_inches(18,10)
plt.show()


# As we can see, there are some topics are constantly published every year, such as police and govt
# . 
# Some topics only appear in certain years, like:
# 1. 'iraq' in 2003 and 2004 when iraq war erupted
# 2. After the Finacial crisis in 2008, 'interview' is always in a high place until 2014.
# 3. 'election' in 2016 and 'trump' in 2017.
# 
# Sentiment analysis;
# The negative sentiment tends to increase from 2003 and reaches its peak in 2009. Then it starts to decrease until 2014. 
