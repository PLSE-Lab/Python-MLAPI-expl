#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns; 
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
df = pd.read_csv('../input/Hotel_Reviews.csv')
df.head()
# Any results you write to the current directory are saved as output.


# In[ ]:


df.shape # check data size


# ### Wtih  a same dataset, different people care different questions. To fulfill most of demands, the report will be in 3 parts:
# 1. How to find the best cozy or popular hotel in a specific place? 
# 2. Comments sentiment analysis
# 3. How to help hotels improve customer satisfaction ?

# Firstly, I will check the score distribution. If you want to evalulate a thing is good or bad, you need a baseline to define or divide good and bad.
# 1. To know how many hotel in the dataset (some popular hotels may have several reviews).
# 2. To know the distribution of the score

# In[ ]:


print(df.Hotel_Name.nunique(), 'hotels in the dataset')
# ok, we have 1492 hotels in the dataset are be reviewed


# In[ ]:


df_uni = df[['Hotel_Name','Average_Score']].drop_duplicates()
plt.figure(figsize = (14,6))
sns.countplot(x = 'Average_Score',data = df_uni,color = 'orange')


# In[ ]:


df.Average_Score.describe()


# We can see very clearly from the upper image and the table, most of reviews stay the rightside of 8.1.
# If a hotel's Avg_score is under 8.1, that means it only wins 25% competitors.

# In[ ]:


# take a glance of address, and separate city and nation.
df.Hotel_Address[10]


# In[ ]:


df.Hotel_Address = df.Hotel_Address.str.replace('United Kingdom','UK')
df['State'] = df.Hotel_Address.apply(lambda x: x.split(' ')[-1])
# Now we can check distribution in  and in state
plt.figure(figsize = (12,5))
plt.title('Hotel distribution in States')
df.State.value_counts().plot.barh(color = 'orange')


# Let's define the popular hotels are above 8.8 Avg_score and with the most reviews.
# 
# And then, we find the best and worst hotels in specific city, how about Amsterdam and Paris?

# In[ ]:


df[df.Average_Score >= 8.8][['Hotel_Name','Average_Score','Total_Number_of_Reviews']].drop_duplicates().sort_values(by ='Total_Number_of_Reviews',ascending = False)[:15]
# The most popular 15 hotels 


# In[ ]:


#split city
df['City']= df.Hotel_Address.apply(lambda x: x.split(' ')[-2])
# find the best 10 hotels in Amsterdam
df[(df.Average_Score >= 9.0) & (df.City == 'Amsterdam')][['Hotel_Name','Average_Score']].drop_duplicates().sort_values(by ='Average_Score',ascending = False)[:10]


# In[ ]:


# find the worst 10 hotels in Paris
df[(df.Average_Score <= 8.1) & (df.City == 'Paris')][['Hotel_Name','Average_Score']].drop_duplicates().sort_values(by ='Average_Score',ascending = True)[:10]


# See! Clean and easy! 
# A data-driven way to find your dream vacation hotel.

# ###  In this part, we will process the word comments and find some interesting features of that.

# In[ ]:


# Create a subside dataset.
df_com = df[['Hotel_Name','Reviewer_Score','Negative_Review','Review_Total_Negative_Word_Counts','Positive_Review','Review_Total_Positive_Word_Counts','Total_Number_of_Reviews','Total_Number_of_Reviews_Reviewer_Has_Given']]


# In[ ]:


# Create NLP and machine learning environment
import re, nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
wordnet_lemmatizer = WordNetLemmatizer()
from sklearn.feature_extraction.text import CountVectorizer


# In[ ]:


#NLP prepross neg and pos reviews
neg = []
for i in df_com['Negative_Review']:
    letters = re.sub('[^a-zA-Z]',' ',i)
    tokens = nltk.word_tokenize(letters)
    lowercase = [l.lower() for l in tokens]
    filtered_result = list(filter(lambda l: l not in stop_words, lowercase))
    lemmas = [wordnet_lemmatizer.lemmatize(t) for t in filtered_result] 
    neg.append(' '.join(lemmas))
    
pos = []
for i in df_com['Positive_Review']:
    pletters = re.sub('[^a-zA-Z]',' ',i)
    ptokens = nltk.word_tokenize(pletters)
    plowercase = [l.lower() for l in ptokens]
    filtered_presult = list(filter(lambda l: l not in stop_words, plowercase))
    plemmas = [wordnet_lemmatizer.lemmatize(t) for t in filtered_presult] 
    pos.append(' '.join(plemmas))


# In[ ]:


#Finding most important words in Negative Reviews and in Postive Reviews
cv = CountVectorizer(analyzer = 'word',stop_words = 'english',max_features = 20,ngram_range=(2,2))
most_negative_words = cv.fit_transform(neg)
temp_counts = most_negative_words.sum(axis=0)
temp_words = cv.vocabulary_
print('the most important words in Negative Reviews:')
print('--------------------------------------------')
display(temp_words)

print('                                          ')
cv = CountVectorizer(analyzer = 'word',stop_words = 'english',max_features = 20,ngram_range=(2,2))
most_positive_words = cv.fit_transform(pos)
temp1_counts = most_positive_words.sum(axis=0)
temp1_words = cv.vocabulary_
print('the most important words in Positive Reviews:')
print('--------------------------------------------')
temp1_words


# Now, we roughly know reasons of positive review and negative review. Some words appear many times. We can dig deeper. 
# 
# The reason we profile sentiement is not only to find out how customers like us or not, that is an existed result, but also to find out the flawness of our work or service, improving our performance in the next days.
# 
# Let's strat from  'small room' and 'tea coffee'

# In[ ]:


small_room = np.zeros(len(df_com))
for i in range(len(df_com)):
    if ('small room' in neg[i]) or ('room small' in neg[i]):
        small_room[i] = 1
display(np.sum(small_room))

teacoffee = np.zeros(len(df_com))
for i in range(len(df_com)):
    if ('tea' in neg[i]) or ('coffee' in neg[i]):
        teacoffee[i] = 1
display(np.sum(teacoffee))


# In[ ]:


# that's a little specific,let's try broader.
room_problem = np.zeros(len(df_com))
for i in range(len(df_com)):
    if ('mini bar' in neg[i]) or ('room service' in neg[i]) or ('double bed' in neg[i]) or ('double room' in neg[i]): 
        room_problem[i] = 1
np.sum(room_problem)


# So far, it's pretty clear that small room is one of the key reasons of nagative reviews. 
# 
# If the size of room can not be changed in short time, maybe hotel can make more efforts on upgradting quality of beverage and improving room service.
# 
# 'Helpful staff' and 'comfort bed' are the easiest ways to earn good review.

# ###  Do some simple quantitive process.
# 1. clean and count pos and neg reviews
# 2. define and make a neg-rate
# 3. Use the neg-rate to evaluate a hotel's management
# 

# In[ ]:


df_com['+'] = 1
df_com['-'] = 1
df_com['+'] = df_com.apply(lambda x: 0 if x["Positive_Review"] == 'No Positive' else x['+'],axis =1)
df_com['-'] = df_com.apply(lambda x: 0 if x["Negative_Review"] == 'No Negative' else x['-'],axis =1)
counted_reviews = pd.DataFrame(df_com.groupby(['Hotel_Name'])['+','-','Total_Number_of_Reviews_Reviewer_Has_Given'].sum())
counted_reviews['Total'] = counted_reviews['+'] +counted_reviews['-']
counted_reviews['Neg_rate'] = round(counted_reviews['-'] / counted_reviews['Total'],2)
counted_reviews['Neg_rate'].describe()


# In[ ]:


counted_reviews[counted_reviews.Neg_rate > 0.5]


# In[ ]:


#I really want to know what happened to the most neg-rate hotel: Hotel Liberty
df[df.Hotel_Name == 'Hotel Liberty'][['Positive_Review','Negative_Review','Average_Score']]


# It should be an old 4-star hotel, good location and free breakfast earn a lot of good reviews. However, it still has many places need to be improved.
# 
# Aa s hotel owner or  manager, it is a good way to learn good and bad from customes' direct feedback. 
# 
# Harsh truth is still a truth.

# 
