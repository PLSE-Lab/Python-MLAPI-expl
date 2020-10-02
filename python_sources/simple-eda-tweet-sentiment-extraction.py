#!/usr/bin/env python
# coding: utf-8

# # Simple EDA - Tweet Sentiment Extraction
# 
# ### Description
# > In this competition we've extracted support phrases from Figure Eight's Data for Everyone platform. The dataset is titled Sentiment Analysis: Emotion in Text tweets with existing sentiment labels, used here under creative commons attribution 4.0. international licence. Your objective in this competition is to construct a model that can do the same - look at the labeled sentiment for a given tweet and figure out what word or phrase best supports it.
# 
# ### Data description
# > Each row contains the text of a tweet and a sentiment label. In the training set you are provided with a word or phrase drawn from the tweet (selected_text) that encapsulates the provided sentiment.
# >
# > Make sure, when parsing the CSV, to remove the beginning / ending quotes from the text field, to ensure that you don't include them in your training.
# 
# ### What we should predicting?
# > You're attempting to predict the word or phrase from the tweet that exemplifies the provided sentiment. The word or phrase should include all characters within that span (i.e. including commas, spaces, etc.). The format is as follows:
# >
# > `<id>,"<word or phrase that supports the sentiment>"`
# >
# > For example:
# > ```
# > 2,"very good"
# > 5,"I am neutral about this"
# > 6,"bad"
# > 8,"if you say so!"
# > ```
# 
# ### Columns
# - **textID** - unique ID for each piece of text
# - **text** - the text of the tweet
# - **sentiment** - the general sentiment of the tweet
# - **selected_text** - [train only] the text that supports the tweet's sentiment
# 
# ------------------------
# **I'll update this EDA notebook in the next days/weeks, stay tuned!**

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import plotly.graph_objects as go

from wordcloud import WordCloud, STOPWORDS

DIR_INPUT = '/kaggle/input/tweet-sentiment-extraction'


# ## Train dataset

# In[ ]:


train_df = pd.read_csv(DIR_INPUT + '/train.csv')
train_df.head()


# In[ ]:


train_df['sentiment'].value_counts(normalize=True)


# In[ ]:


dist = train_df['sentiment'].value_counts()

fig = go.Figure([go.Bar(x=dist.index, y=dist.values)])
fig.update_layout(
    title='Sentiment distribution in train dataset'
)
fig.show()


# ### Examples

# In[ ]:


text = 'TEXT: \n{}\n\nSELECTED_TEXT: \n{}\n\nSENTIMENT: \n{}'

for i in range(3):
    print("============")
    print(text.format(train_df.iloc[i, 1],
                      train_df.iloc[i, 2],
                      train_df.iloc[i, 3]))
    print("============\n\n")


# ### Wordclouds - Frequent words:

# In[ ]:


rnd_comments = train_df[train_df['sentiment'] == 'neutral'].sample(n=2000)['text'].values
wc = WordCloud(background_color="black", max_words=2000, stopwords=STOPWORDS)
wc.generate(" ".join(rnd_comments))

plt.figure(figsize=(20,10))
plt.axis("off")
plt.title("Frequent words in neutral comments", fontsize=20)
plt.imshow(wc.recolor(colormap= 'viridis' , random_state=17), alpha=0.98)
plt.show()


# In[ ]:


rnd_comments = train_df[train_df['sentiment'] == 'negative'].sample(n=2000)['text'].values
wc = WordCloud(background_color="black", max_words=2000, stopwords=STOPWORDS)
wc.generate(" ".join(rnd_comments))

plt.figure(figsize=(20,10))
plt.axis("off")
plt.title("Frequent words in negative comments", fontsize=20)
plt.imshow(wc.recolor(colormap= 'viridis' , random_state=17), alpha=0.98)
plt.show()


# In[ ]:


rnd_comments = train_df[train_df['sentiment'] == 'positive'].sample(n=2000)['text'].values
wc = WordCloud(background_color="black", max_words=2000, stopwords=STOPWORDS)
wc.generate(" ".join(rnd_comments))

plt.figure(figsize=(20,10))
plt.axis("off")
plt.title("Frequent words in positive comments", fontsize=20)
plt.imshow(wc.recolor(colormap= 'viridis' , random_state=17), alpha=0.98)
plt.show()


# ## Test dataset

# In[ ]:


test_df = pd.read_csv(DIR_INPUT + '/test.csv')
test_df.head()


# In[ ]:


test_df['sentiment'].value_counts(normalize=True)


# In[ ]:


dist = test_df['sentiment'].value_counts()

fig = go.Figure([go.Bar(x=dist.index, y=dist.values)])
fig.update_layout(
    title='Sentiment distribution in test dataset'
)
fig.show()


# ### Examples

# In[ ]:


text = 'TEXT: \n{}\n\nSENTIMENT: \n{}'

for i in range(3):
    print("============")
    print(text.format(test_df.iloc[i, 1],
                      test_df.iloc[i, 2]))
    print("============\n\n")


# ### Wordclouds - frequent words

# In[ ]:


rnd_comments = test_df[test_df['sentiment'] == 'neutral'].sample(n=1000)['text'].values
wc = WordCloud(background_color="black", max_words=2000, stopwords=STOPWORDS)
wc.generate(" ".join(rnd_comments))

plt.figure(figsize=(20,10))
plt.axis("off")
plt.title("Frequent words in neutral comments", fontsize=20)
plt.imshow(wc.recolor(colormap= 'viridis' , random_state=17), alpha=0.98)
plt.show()


# In[ ]:


rnd_comments = test_df[test_df['sentiment'] == 'negative'].sample(n=1000)['text'].values
wc = WordCloud(background_color="black", max_words=2000, stopwords=STOPWORDS)
wc.generate(" ".join(rnd_comments))

plt.figure(figsize=(20,10))
plt.axis("off")
plt.title("Frequent words in negative comments", fontsize=20)
plt.imshow(wc.recolor(colormap= 'viridis' , random_state=17), alpha=0.98)
plt.show()


# In[ ]:


rnd_comments = test_df[test_df['sentiment'] == 'positive'].sample(n=1000)['text'].values
wc = WordCloud(background_color="black", max_words=2000, stopwords=STOPWORDS)
wc.generate(" ".join(rnd_comments))

plt.figure(figsize=(20,10))
plt.axis("off")
plt.title("Frequent words in positive comments", fontsize=20)
plt.imshow(wc.recolor(colormap= 'viridis' , random_state=17), alpha=0.98)
plt.show()


# In[ ]:




