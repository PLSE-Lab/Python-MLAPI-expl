#!/usr/bin/env python
# coding: utf-8

# ## Importing The Libaries

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re

get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings("ignore")

import os
print(os.listdir("../input"))


# In[ ]:


pd.set_option('display.max_columns',None)


# ## Loading The Data

# In[ ]:


US_comments = pd.read_csv('../input/youtube/UScomments.csv', error_bad_lines=False)


# In[ ]:


US_videos = pd.read_csv('../input/youtube/USvideos.csv', error_bad_lines=False)


# In[ ]:


US_videos.head()


# ## Let's do some analysis and Data Cleaning on both the datasets.

# In[ ]:


US_videos.shape


# In[ ]:


US_videos.nunique()


# In[ ]:


US_videos.info()


# In[ ]:


US_videos.head()


# In[ ]:


US_comments.head()


# In[ ]:


US_comments.shape


# In[ ]:


US_comments.isnull().sum()


# In[ ]:


US_comments.dropna(inplace=True)


# In[ ]:


US_comments.isnull().sum()


# In[ ]:


US_comments.shape


# In[ ]:


US_comments.nunique()


# In[ ]:


US_comments.info()


# In[ ]:


US_comments.drop(41587, inplace=True)


# In[ ]:


US_comments = US_comments.reset_index().drop('index',axis=1)


# In[ ]:


US_comments.likes = US_comments.likes.astype(int)
US_comments.replies = US_comments.replies.astype(int)


# In[ ]:


US_comments.head()


# ## Removing Punctuations, Numbers and Special Characters.

# In[ ]:


US_comments['comment_text'] = US_comments['comment_text'].str.replace("[^a-zA-Z#]", " ")


# ## Removing Short Words.

# In[ ]:


US_comments['comment_text'] = US_comments['comment_text'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>3]))


# ## Changing the text to lower case.

# In[ ]:


US_comments['comment_text'] = US_comments['comment_text'].apply(lambda x:x.lower())


# ## Tokenization

# In[ ]:


tokenized_tweet = US_comments['comment_text'].apply(lambda x: x.split())
tokenized_tweet.head()


# ## Lemmatization

# In[ ]:


from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords


# In[ ]:


wnl = WordNetLemmatizer()


# In[ ]:


tokenized_tweet.apply(lambda x: [wnl.lemmatize(i) for i in x if i not in set(stopwords.words('english'))]) 
tokenized_tweet.head()


# In[ ]:


for i in range(len(tokenized_tweet)):
    tokenized_tweet[i] = ' '.join(tokenized_tweet[i])


# In[ ]:


US_comments['comment_text'] = tokenized_tweet


# ## Let's do the Sentiment Analysis on the US Comments Dataset

# In[ ]:


import nltk
nltk.download('vader_lexicon')


# In[ ]:


from nltk.sentiment.vader import SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()


# ## Setting The Sentiment Scores

# In[ ]:


US_comments['Sentiment Scores'] = US_comments['comment_text'].apply(lambda x:sia.polarity_scores(x)['compound'])


# In[ ]:


US_comments.head()


# ## Classifying the Sentiment scores as Positive, Negative and Neutral

# In[ ]:


US_comments['Sentiment'] = US_comments['Sentiment Scores'].apply(lambda s : 'Positive' if s > 0 else ('Neutral' if s == 0 else 'Negative'))


# In[ ]:


US_comments.head()


# In[ ]:


US_comments.Sentiment.value_counts()


# ## Now we will calculate the percentage of comments which are positive in all the videos.

# In[ ]:


videos = []
for i in range(0,US_comments.video_id.nunique()):
    a = US_comments[(US_comments.video_id == US_comments.video_id.unique()[i]) & (US_comments.Sentiment == 'Positive')].count()[0]
    b = US_comments[US_comments.video_id == US_comments.video_id.unique()[i]]['Sentiment'].value_counts().sum()
    Percentage = (a/b)*100
    videos.append(round(Percentage,2))


# ## Making a dataframe of the videos with their Positive Percentages.

# In[ ]:


Positivity = pd.DataFrame(videos,US_comments.video_id.unique()).reset_index()


# In[ ]:


Positivity.columns = ['video_id','Positive Percentage']


# In[ ]:


Positivity.head()


# ## Now we will add the channel name of the videos which are their in our new dataset.

# In[ ]:


channels = []
for i in range(0,Positivity.video_id.nunique()):
    channels.append(US_videos[US_videos.video_id == Positivity.video_id.unique()[i]]['channel_title'].unique()[0])


# In[ ]:


Positivity['Channel'] = channels


# In[ ]:


Positivity.head()


# In[ ]:


Positivity[Positivity['Positive Percentage'] == Positivity['Positive Percentage'].max()]


# ## So these are the videos and their channels whose comments are 100% Positive (Well, this might be less likely because NLTK is poor with sarcasmic comments but i can say that most of the comments are positive).

# In[ ]:


Positivity[Positivity['Positive Percentage'] == Positivity['Positive Percentage'].min()]


# ## So these are the videos and their channels whose comments are 0% Positive (Means the comments are either Negative or Neutral. This stat is also less likely but as i said earlier, MOSTLY).

# ## Let's Contstruct a wordcloud of all the comments to see the most frequent comments.

# In[ ]:


all_words = ' '.join([text for text in US_comments['comment_text']])
from wordcloud import WordCloud
wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(all_words)

plt.figure(figsize=(10, 7))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.show()


# ## Let's Construct a Wordcloud of Positive Comments

# In[ ]:


all_words_posi = ' '.join([text for text in US_comments['comment_text'][US_comments.Sentiment == 'Positive']])


# In[ ]:


wordcloud_posi = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(all_words_posi)

plt.figure(figsize=(10, 7))
plt.imshow(wordcloud_posi, interpolation="bilinear")
plt.axis('off')
plt.show()


# ## Let's Construct a Wordcloud of Negative Comments

# In[ ]:


all_words_nega = ' '.join([text for text in US_comments['comment_text'][US_comments.Sentiment == 'Negative']])


# In[ ]:


wordcloud_nega = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(all_words_nega)

plt.figure(figsize=(10, 7))
plt.imshow(wordcloud_nega, interpolation="bilinear")
plt.axis('off')
plt.show()


# ## Let's Contsruct a Wordcloud of Neutral Comments.

# In[ ]:


all_words_neu = ' '.join([text for text in US_comments['comment_text'][US_comments.Sentiment == 'Neutral']])


# In[ ]:


wordcloud_neu = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(all_words_neu)

plt.figure(figsize=(10, 7))
plt.imshow(wordcloud_neu, interpolation="bilinear")
plt.axis('off')
plt.show()

