#!/usr/bin/env python
# coding: utf-8

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


# read data
df = pd.read_csv('../input/515k-hotel-reviews-data-in-europe/Hotel_Reviews.csv')
df


# In[ ]:


df.iloc[0].Hotel_Address


# In[ ]:


city = df.iloc[0].Hotel_Address.split()[-2]
country = df.iloc[0].Hotel_Address.split()[-1]
city, country


# In[ ]:


city = df.iloc[1].Hotel_Address.split()[-2]
country = df.iloc[1].Hotel_Address.split()[-1]
city, country


# In[ ]:


city = df.iloc[1034].Hotel_Address.split()[-2]
country = df.iloc[1034].Hotel_Address.split()[-1]
city, country


# In[ ]:


city = df.iloc[134].Hotel_Address.split()[-2]
country = df.iloc[134].Hotel_Address.split()[-1]
city, country


# In[ ]:


df.columns


# In[ ]:


df.dtypes


# In[ ]:


df.Reviewer_Score.value_counts()


# In[ ]:


df['neg_flag'] = df.Review_Total_Negative_Word_Counts <= df.Review_Total_Positive_Word_Counts


# In[ ]:


df


# In[ ]:


import seaborn as sns
sns.scatterplot(x=df.lat, y=df.lng, hue=df['neg_flag'])


# In[ ]:


cmap = sns.cubehelix_palette(dark=.3, light=.8, as_cmap=True)
sns.scatterplot(x=df.lat, y=df.lng, size=df['neg_flag'],
                sizes=(20, 200), palette=cmap)


# In[ ]:


df.shape


# In[ ]:


df.lat.value_counts()


# In[ ]:


# visualization library
import seaborn as sns
#  pair plot
sns.pairplot(df)


#  **Output description:  Most of the pairplot has linear (vertical or horizental) swarmp, means that many of the attributes do not affect the Reviewers_Score**.     Check Later...

# # **General overview of not-important attributes**

# In[ ]:


sns.regplot(x=df['lat'], y=df['Reviewer_Score'])


# Output description: The line is slightly changes (inversely), means there is no coefficient between the lables.

# In[ ]:


sns.regplot(x=df['lng'], y=df['Reviewer_Score'])


# Output description: The line is slightly changes (positively), means there is no coefficient between the lables.

# # Reviewer_Score based on nationality

# In[ ]:


# Reviewe_Score counts
sns.distplot(df["Reviewer_Score"],kde=False,bins=15)


# In[ ]:


df.shape


# Vast majority of the Reviewer_Score (33%) and the others are also considered high, which is an obvius indicator that most of Reviews are pretty positive.
# I will check the positivity/negativity in the end of the notebook (text-cleaning, NLP)

# # Highest and Lowest Scoring Countries

# In[ ]:


df['Reviewer_Score'].min() , df['Reviewer_Score'].max(), df['Reviewer_Score'].mean()


# >*Top_Reviewers Nationality

# In[ ]:


countries = df["Reviewer_Nationality"].value_counts()[df["Reviewer_Nationality"].value_counts() > 100]
g = df.groupby("Reviewer_Nationality").mean()
g.loc[countries.index.tolist()]["Reviewer_Score"].sort_values(ascending=False)[:10].plot(kind="bar",ylim=(8.395076569886239,9),title="Top Reviewing Countries")


# Least_Reviewers Nationality

# In[ ]:


g.loc[countries.index.tolist()]["Reviewer_Score"].sort_values()[:10].plot(kind="bar",ylim=(2.5,8.395076569886239),title="least Reviewing Countries")


# The question now is:   It seems that most of the least reveiews basicully from "Middle East"

# # Best Hotels

# based on Region

# In[ ]:


def country_ident(st):
    last = st.split()[-1]
    if last == "Kingdom": return "United Kingdom"
    else: return last
    
df["Hotel_Country"] = df["Hotel_Address"].apply(country_ident)
df.groupby("Hotel_Country").mean()["Reviewer_Score"].sort_values(ascending=False)


# In[ ]:


sns.swarmplot(x=df.Hotel_Country, y=df.Reviewer_Score)


# Best Hotels

# In[ ]:


best_hotels = df.groupby('Hotel_Name')['Reviewer_Score'].mean().sort_values(ascending=False).head(10)
best_hotels.plot(kind="bar",color = "Green")


# The mean are slightly different. which draw the same conclusion that is until now there is no a descrimantal attribute (Hotel-coutry and the previous checked ones)

# # Review Date (searching about a trend, pattern,...)

# In[ ]:


from datetime import datetime
df["Review_Date_Month"] = df["Review_Date"].apply(lambda x: x[5:7])
df[["Review_Date","Reviewer_Score"]].groupby("Review_Date").mean().plot(figsize=(15,5))


# There is no specific pattern, means there are other different features that affect the Reviewers_Score

# # Word Cloud

# In[ ]:


from wordcloud import WordCloud
import matplotlib.pyplot as plt
def show_wordcloud(data, title = None):
    wordcloud = WordCloud(
        background_color = 'white',
        max_words = 200,
        max_font_size = 40, 
        scale = 3,
        random_state = 42
    ).generate(str(data))

    fig = plt.figure(1, figsize = (20, 20))
    plt.axis('off')
    if title: 
        fig.suptitle(title, fontsize = 20)
        fig.subplots_adjust(top = 2.3)

    plt.imshow(wordcloud)
    plt.show()
    
# print wordcloud
show_wordcloud(df['Positive_Review'])


# In[ ]:


# most positive
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(analyzer = "word",stop_words = 'english',max_features = 20,ngram_range=(2,2))
most_positive_words = cv.fit_transform(df['Positive_Review'])
temp1_counts = most_positive_words.sum(axis=0)
temp1_words = cv.vocabulary_
temp1_words


# In[ ]:


show_wordcloud(df['Negative_Review'])


# In[ ]:


cv = CountVectorizer(analyzer = "word",stop_words = 'english',max_features = 20,ngram_range=(2,2))
most_negative_words = cv.fit_transform(df['Negative_Review'])
temp1_counts = most_negative_words.sum(axis=0)
temp1_words = cv.vocabulary_
temp1_words


# Conclusion: until now, the only features affect the scores are the hotels themesleves (positive review) due to many aspects such as location,....
# 
# Need to extract all the hotel_positive_modes and then find the weight
# 
# Also, there are other suggestions that are related to the reviewers:
# 
# Extracting data 'from Tags' such as (trip type, social status, room, stayed nights)
# Noticing the reviewers' nationalities: The less satisifed ones are somehow come from Asia, specifically Westren Union countris; which needs modeling if it is not by chance

# # Extracting from Tags, and positive/negative most words

# In[ ]:


# extrating nights from tag
def splitString(string):
    array = string.split(" ', ' ")
    array[0] = array[0][3:]
    array[-1] = array[-1][:-3]
    if not 'trip' in array[0]:
        array.insert(0,None)
    try:
        return float(array[3].split()[1])
    except:
        return None

df["Nights"] = df["Tags"].apply(splitString)
sns.jointplot(data=df,y="Reviewer_Score",x="Nights",kind="reg")


# The more the reviewer stay at hotel the lower the score is (but also slightly)

# Extracting Trip_type

# In[ ]:


df['Leisure'] = df['Tags'].map(lambda x: 1 if ' Leisure trip ' in x else 0)
df['Business'] = df['Tags'].map(lambda x: 2 if ' Business trip ' in x else 0)
df['Trip_type'] = df['Leisure'] + df['Business']


# ....  will checked in building model
# conclusion
# x = Reviewer_Score
# y = positive and negative reviews word vector or/and some features from tag

# In[ ]:




