#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

stopwords = set(STOPWORDS)

# Filtering out some words that are too common, and we want to focus on the
# words that are diffrent between varieties
stopwords.add("wine")
stopwords.add("flavor")
stopwords.add("flavors")
stopwords.add("finish")
stopwords.add("drink")
stopwords.add("now")
stopwords.add("note")
stopwords.add("notes")

df = pd.read_csv('../input/wine-reviews/winemag-data_first150k.csv')
df = df[['variety', 'points', 'description']]
df = df[~df.variety.str.contains('Blend|,')]


# # Determining Most Popular Wine Varieties
# Examining the review data set, I wanted to take a look at which varieties were the most popular (based on review count).
# 
# You can also see the addition of the points (the average review score), which is displayed as a % of the reviews. The interesting thing is that the less popular variety has a higher average rating.

# In[ ]:


top_reviewed = df[['variety']].groupby('variety').size().reset_index(name='count')

top_scored = df[['variety', 'points']].groupby('variety').mean()
top_reviewed = top_reviewed.join(top_scored, on='variety')

top_reviewed['points'] = top_reviewed['count'] * (top_reviewed['points'] / 100)

top_reviewed.sort_values('count', ascending=False)[:20].sort_values('count').set_index("variety", drop=True).plot(kind='barh', title='Top 20 Varieties by Review Count')
plt.show()


# ## Finding Defining Properties
# My hope was that if I could look at the most word occurance in reviews for each variety, that we can see some of the defining characteristics for each. Word clouds are masked and coloured using wine bottle images of the same variety (for fun and learning).

# In[ ]:


for number, variety_name in enumerate(top_reviewed.sort_values('count', ascending=False)[:10]['variety']):
    variety = df[df['variety'] == variety_name]
    reviews = ' '.join(' '.join(variety['description']).split())

    image_path = os.path.join('../input/wine-bottle-images/', variety_name + '.jpg')
    bottle_coloring = np.array(Image.open(image_path))
    image_colors = ImageColorGenerator(bottle_coloring)
    
    wc = WordCloud(background_color="white", max_words=2000, stopwords=stopwords, mask=bottle_coloring)
    wc.generate(text=reviews)

    try:
        plt.imshow(wc.recolor(color_func=image_colors), interpolation="bilinear")
        plt.axis("off")
        plt.title(str(number + 1) + '. ' + variety_name)
        plt.show()
    except:
        pass


# If anyone has any tips on how to increase the size/resolution of the wordclouds, it would be much appreciated.
