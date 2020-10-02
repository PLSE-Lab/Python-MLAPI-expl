#!/usr/bin/env python
# coding: utf-8

# I've never done word clouds before. Let me try it!

# In[ ]:


import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

from wordcloud import WordCloud, STOPWORDS


# In[ ]:


df = pd.read_csv("../input/comeytestimony/qa.csv") # updated to match uploaded set
df.head()


# In[ ]:


questions = df["Full Question"]
answers = df["Comey Response"]


# In[ ]:


wordcloud_q = WordCloud(
                          background_color='white',
                          stopwords=set(STOPWORDS),
                          max_words=250,
                          max_font_size=40, 
                          random_state=1705
                         ).generate(str(questions))
wordcloud_a = WordCloud(
                          background_color='white',
                          stopwords=set(STOPWORDS),
                          max_words=250,
                          max_font_size=40, 
                          random_state=1705
                         ).generate(str(answers))


# In[ ]:


def cloud_plot(wordcloud):
    fig = plt.figure(1, figsize=(20,15))
    plt.imshow(wordcloud)
    plt.axis('off')
    plt.show()


# ## Questions word cloud

# In[ ]:


cloud_plot(wordcloud_q)


# ## Answers word cloud

# In[ ]:


cloud_plot(wordcloud_a)


# .... that wasn't bad for being my first ever word cloud, was it? Please upvote if you like it!
