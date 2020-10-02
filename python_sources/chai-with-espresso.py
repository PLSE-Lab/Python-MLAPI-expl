#!/usr/bin/env python
# coding: utf-8

# # Dirty chai
# According to [this website](https://www.thespruceeats.com/dirty-chai-definition-765697#:~:text=Dirty%20chai%20is%20a%20popular,and%20a%20chai%20tea%20latte.) Chai with Espresso (aka Dirty chai) "packs a double whammy of caffeine from the black tea and shot or two of espresso. The average caffeine level of a 12-ounce dirty chai latte is 160 milligrams (versus 50 to 70 milligrams for a chai latte)."
# 
# That sounds like my kind of Chai! Lets explore this awesome dataset while hyped up on caffene!

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import seaborn as sns
import os
from tqdm.notebook import tqdm
import datetime as dt


# In[ ]:


get_ipython().system('ls -GFlash --color ../input/chai-time-data-science/')


# In[ ]:


thumb = pd.read_csv('../input/chai-time-data-science/Anchor Thumbnail Types.csv')
eps = pd.read_csv('../input/chai-time-data-science/Episodes.csv')


# ## Read in all Cleaned Subtitles and Parse

# In[ ]:


cleaned_st_files = os.listdir('../input/chai-time-data-science/Cleaned Subtitles/')

def add_duration(df):
    df['colon_count'] = df['Time'].str.count(':')
    df.loc[df['colon_count'] == 1, 'Time'] = '0:' + df.loc[df['colon_count'] == 1]['Time']
    df['Time_dt'] = df['Time'].apply(lambda x: dt.datetime.strptime(x, "%H:%M:%S"))
    df['Duration'] = (df['Time_dt'] - dt.datetime(1900, 1, 1))         .apply(lambda x: x.total_seconds()).astype('int')
    return df

c_fs = []
for f in tqdm(cleaned_st_files):
    df = pd.read_csv(f'../input/chai-time-data-science/Cleaned Subtitles/{f}')
    df = add_duration(df)
    df['E'] = f.replace('.csv','')
    c_fs.append(df)
all_subs = pd.concat(c_fs)


# In[ ]:


all_subs['Duration'].plot(kind='hist', figsize=(15, 5), bins=20,
                          title='Duration of Single Subtitle')
plt.show()


# ## Longest subtitle?

# In[ ]:


all_subs.sort_values('Duration')['Text'].values[-1]


# # First word someone says in subtitle section

# In[ ]:


all_subs['first_word'] = all_subs['Text'].str.split(' ', expand=True)[0]
all_subs['first_word_clean'] = all_subs['first_word'].str.lower().str.replace(',','').str.replace('.','')


# In[ ]:


all_subs['first_word_clean'].value_counts().head(20)     .sort_values().plot(kind='barh', figsize=(15, 8),
                        title='First word spoken in subtitle section')
plt.show()


# # Simple Sentiment Analysis
# Reference: https://medium.com/analytics-vidhya/simplifying-social-media-sentiment-analysis-using-vader-in-python-f9e6ec6fc52f

# In[ ]:


get_ipython().system('pip install vaderSentiment > /dev/null')
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyser = SentimentIntensityAnalyzer()
all_subs['Polarity_Scores'] = all_subs['Text'].apply(lambda x: analyser.polarity_scores(x))


# In[ ]:


all_subs['neg_score'] = all_subs['Polarity_Scores'].apply(lambda x: x['neg'])
all_subs['neu_score'] = all_subs['Polarity_Scores'].apply(lambda x: x['neu'])
all_subs['pos_score'] = all_subs['Polarity_Scores'].apply(lambda x: x['pos'])
all_subs['compound_score'] = all_subs['Polarity_Scores'].apply(lambda x: x['compound'])


# In[ ]:


fig, ax = plt.subplots(figsize=(15, 5))
all_subs['neg_score'].plot(kind='hist',
                           bins=50,
                           alpha=0.5,
                           title='Distribution of Sentiment Scores')
all_subs['pos_score'].plot(kind='hist',
                           bins=50,
                           alpha=0.5)
all_subs['neu_score'].plot(kind='hist',
                           bins=50,
                           alpha=0.5)
plt.legend()
plt.show()

