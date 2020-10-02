#!/usr/bin/env python
# coding: utf-8

# Goal
# 
# Given the large amount of literature and rapidly spreading COVID-19, it is difficult to differ an understanding between the research presented by the academic community and the information being presented for mass consumption, as COVID-19 is approached with aggressive timelines. The pretext is to gain a basic level understanding of the sentiment regarding COVID-19 presented by reputable sources.
# 
# Approach:
# 
#     Exploratory data analysis

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
for dirname, _, filenames in os.walk('/kaggle/input/CORD-19-research-challenge/2020-03-13/all_sources_metadata_2020-03-13.csv'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


All_Sources = pd.read_csv('/kaggle/input/CORD-19-research-challenge/2020-03-13/all_sources_metadata_2020-03-13.csv')


# **Upon file import it is of utmost importance to completely understand the data in the context of healthcare. 
# Python is helpful in data exploration, similar to an open world video game simulation. **

# In[ ]:


All_Sources.columns
##Begin with understanding our dataset columns and list them out


# In[ ]:


All_Sources.head(10)
#Sample the data, are there any columns that we would like to focus on first?


# In[ ]:


##After the sample I would like to get an good idea of the size of this dataset therefore count the rows
count_row = All_Sources.shape[0]  
count_row


# In[ ]:


print(All_Sources['journal'].describe())


# **This is for the most part, a small dataset therefore let us find any trends to focus our analysis. 
# As we would expect most of the columns have varied data let's see who the most active journals are on COVID-19.
# We may afterwards find the most active authors. **

# In[ ]:


print(All_Sources['journal'].value_counts())


# In[ ]:


import csv
f= open("/kaggle/input/CORD-19-research-challenge/2020-03-13/all_sources_metadata_2020-03-13.csv")
csvreader = csv.reader(f)
All_Data = list(csvreader)


# In[ ]:


Journals = [row[10] for row in All_Data]

Journals_Counts = {}
for Journals in Journals:
    if Journals not in Journals_Counts:
        Journals_Counts[Journals] = 1
    else:
        Journals_Counts[Journals] += 1
        
Journals_Counts
pd.DataFrame.from_dict(Journals_Counts, orient='index')


# In[ ]:


Journal_Counts_CSV = pd.DataFrame.from_dict(Journals_Counts, orient='index')
Journal_Counts_CSV.to_csv('Jounal_Counts.csv')


# **From our analysis we can quickly sort in on the most active journals, and even focus in on those that have full text.  
# For now lets continue understanding the most active journals. **

# In[ ]:


import matplotlib as plt
All_Sources['journal'].value_counts().iloc[[0,1,2,3,4]].plot.barh(
    title = 'Top Five Active Journals')


# **Here we get an great basic visual of where the next most active journals are in relation to the number one most active. **

# In[ ]:


All_Sources['journal'].value_counts().iloc[[0,1,2,3,4,5,6,7,8,9]].plot.pie(figsize = (10,10), autopct = '%.2f%%',
                                                                           title = 'Top Ten Active Journals')


# **Afterwards we may then switch over and begin NLP as an subsequent method of analysis in an seperate project.**

# **PLoS One is by far the most active journal. Let us see if we can identify any trends that might provide greater insight. **

# In[ ]:


##Lets focus on the abstract column as a starting point for our sentiment-analysis, and convert it to string. 
All_Sources['abstract'] = All_Sources['abstract'].astype(str)
##Lowercase the abstract column
All_Sources['abstract'] = All_Sources['abstract'].apply(lambda x: " ".join(x.lower() for x in x.split()))
##Remove punctuation
All_Sources['abstract'] = All_Sources['abstract'].str.replace('[^\w\s]','')
##Remove stop words
import nltk
from nltk.corpus import stopwords
stop = stopwords.words('english')
All_Sources['abstract'] = All_Sources['abstract'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))
All_Sources = All_Sources.drop_duplicates()


# In[ ]:


##Removing empty abstract values 
All_Sources['abstract'] = All_Sources['abstract'].fillna('nan')
My_List = ['nan']
All_Sources=All_Sources[All_Sources["abstract"].isin(My_List) == False]


# In[ ]:


##Now let's run our sentiment 
from textblob import TextBlob
def senti(x):
    return TextBlob(x).sentiment  
 
All_Sources['senti_score'] = All_Sources['abstract'].apply(senti)
 
All_Sources.senti_score.head()


# **The first score is sentiment polarity which tells if the sentiment is positive or negative and the second score is subjectivity score to tell how subjective is the text included in the abstract.** 

# In[ ]:


##In order to prepare visuals we will need to clean our sentiment scores and sort them into two new columns
All_Sources['senti_score'] = All_Sources['senti_score'].astype(str)
All_Sources['senti_pos_neg'], All_Sources['senti_subjectivity'] = All_Sources['senti_score'].str.split(',', 1).str
All_Sources['senti_pos_neg'] = All_Sources.senti_pos_neg.str.replace('Sentiment\(polarity=,?' , '')
All_Sources['senti_subjectivity'] = All_Sources.senti_subjectivity.str.replace('subjectivity=,?' , '')
All_Sources['senti_subjectivity'] = All_Sources.senti_subjectivity.str.replace('\)' , '')
All_Sources[['senti_subjectivity', 'senti_pos_neg']] = All_Sources[['senti_subjectivity', 'senti_pos_neg']].apply(pd.to_numeric)


# In[ ]:


All_Sources['senti_pos_neg'].plot.hist()


# **Here we have now an visual representation of the frequency of the sentiment associated with all abstracts in our dataset. 
# Positive words in sentiment obtain an + score, and - for a negative word. Let's continue in our understanding of the sentiment associated with the abstracts associated with the Journals. **

# In[ ]:


All_Sources['senti_pos_neg'].value_counts().iloc[[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17]].plot.pie(figsize = (12,12), autopct = '%.2f%%',
                                    title = 'Top Percentages of Sentiment Reflected in the Data')


# In[ ]:


Senti_Counts = All_Sources['senti_pos_neg'].value_counts()
Senti_Counts


# ** The top percentages of sentiment reflected in the data show us the data predominantly supports neutral to positive sentiment in our journal abstracts. **
