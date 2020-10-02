#!/usr/bin/env python
# coding: utf-8

# **Loading the Libraries**

# In[ ]:


import os
print(os.listdir("../input"))
import pandas as pd
import numpy as np
import re #Regular expression for deleting characters which are not letters
import nltk #natural language tool kit
import PIL
from nltk import punkt
from nltk.corpus import stopwords 
from os import path #creating word cloud
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# We will look to analyse both the datasets using python visualisation libraries. 

# In[ ]:


GPSA = pd.read_csv('../input/googleplaystore.csv')
GPSA.head()


# *The number of unique values within the category variable*

# In[ ]:


GPSA['Category'].value_counts()


# *In the categories variable, '1.9' value seems like an anamoly within the dataset. We are going to find its index value and then remove it altogether*

# In[ ]:


GPSA.index[GPSA['Category'] == "1.9"].tolist()


# In[ ]:


GPSA = GPSA.drop(10472, axis = 0)


# In[ ]:


GPSA.rename(columns={'Content Rating':'Content'},inplace=True)
GPSA.head()


# In[ ]:


print("There are {} observations and {} dimensions in this dataset. \n".format(GPSA.shape[0],GPSA.shape[1]))

print("There are {} categories in this dataset such as {}... \n".format(len(GPSA.Category.unique()),
                                                                           ", ".join(GPSA.Category.unique()[0:5])))

print("There are {} genres under different categories in this dataset such as {}... \n".format(len(GPSA.Genres.unique()),
                                                                                      ", ".join(GPSA.Genres.unique()[0:5])))


# *Explore the grouped category variable using measures of central tendenceis such as mean, median, quartiles, SD etc.*

# In[ ]:


Category = GPSA.groupby("Category")
Category.describe().head()


# *The average ratings per category*

# In[ ]:


Category.mean().sort_values(by="Rating",ascending=False).head()


# **Number of Android Applications grouped by Category**

# In[ ]:


plt.figure(figsize=(15,10))
Category.size().sort_values(ascending=False).plot.bar()
plt.xticks(rotation=50)
plt.xlabel("Application Category")
plt.ylabel("Number of Android Applications")
plt.show()


# **Average Ratings per Category**

# In[ ]:


plt.figure(figsize=(15,10))
Category.max().sort_values(by="Rating",ascending=False)["Rating"].plot.bar()
plt.xticks(rotation=50)
plt.xlabel("Application Category")
plt.ylabel("Highest Rating")
plt.show()


# **Creating bokeh chart to plot category and content by mean ratings**

# In[ ]:


from bokeh.io import show, output_file
from bokeh.models import ColumnDataSource
from bokeh.plotting import figure
from bokeh.palettes import RdPu6
from bokeh.transform import factor_cmap

output_file("Google_Play.html")

GPSA.Content = GPSA.Content.astype(str)
GPSA.Category = GPSA.Category.astype(str)

group = GPSA.groupby(by=['Content', 'Category'])

source = ColumnDataSource(group)

index_cmap = factor_cmap('Content_Category', palette = RdPu6, factors = sorted(GPSA.Content.unique()), end = 1)

p = figure(plot_width=1200, plot_height=500, title= "Mean Ratings by Category and Content", x_range=group, toolbar_location=None, tooltips=[("Rating", "@Rating_mean"), ("Content, Category", "@Content_Category")])

p.vbar(x='Content_Category', top = 'Rating_mean' , width=1, source=source, line_color="white", fill_color=index_cmap, )

p.y_range.start = 0
p.x_range.range_padding = 0.025
p.xgrid.grid_line_color = None
p.xaxis.axis_label = "Categories grouped by Content"
p.xaxis.major_label_orientation = 1.0
p.outline_line_color = None

show(p)


# ![Screen%20Shot%202019-05-28%20at%205.26.19%20PM%20%282%29.png](attachment:Screen%20Shot%202019-05-28%20at%205.26.19%20PM%20%282%29.png)
# The file outputs as html and I have attached .png file so that you can view the output. Please provide any suggestions if you know how I can output it in the kaggle notebook. 
# 

# **Word Cloud using the Individual Reviews Dataset**

# In[ ]:


GPSAr = pd.read_csv('../input/googleplaystore_user_reviews.csv', encoding = "latin1")
GPSAr.head()


# *Since we are not going to do any sentiment analysis, we are going to concatenate Translated_Review and Sentiment*

# In[ ]:


GPSAr = pd.concat([GPSAr.Translated_Review,GPSAr.Sentiment],axis=1)
GPSAr.dropna(axis=0,inplace=True) #drop NaN values
GPSAr.head()


# *The counts of sentiments per group*

# In[ ]:


GPSAr['Sentiment'].value_counts()


# *Cleaning the data set for characted values such as !,. etc.*

# In[ ]:


text_list = []
for i in GPSAr.Translated_Review:
    text = re.sub("[^a-zA-Z]"," ",i)
    text = text.lower()
    text = nltk.word_tokenize(text)
    lemma = nltk.WordNetLemmatizer()
    text = [lemma.lemmatize(word) for word in text]
    text = " ".join(text)
    text_list.append(text)


# In[ ]:


text_list[5:10]


# In[ ]:


text1 = " ".join(review for review in GPSAr.Translated_Review)
print ("There are {} words in the combination of all reviews.".format(len(text1)))


# *Editing the wordcloud; changing font size, background colour and the interpolation*

# In[ ]:


wordcloud = WordCloud(max_font_size=150, max_words=100, background_color="grey").generate(text)
plt.figure()
plt.imshow(wordcloud, interpolation="sinc")
plt.axis("off")
plt.show()


# *Create stopword list. Added stop words by multiple iterations of the code to remove selected words from each iteration of the word cloud*

# In[ ]:


stopwords = set(STOPWORDS)
stopwords.update(["app", "game", "thank", "you", "think", "even", "make", "still", "really", "find", "much",
                  "now", "go", "thing", "say", "got", "lot", "open", "day", "one", "back", "please", "sometime",
                 "way", "first", "though"]) 

# Generate a word cloud image
wordcloud = WordCloud(stopwords=stopwords, background_color="white").generate(text1)

# Display the generated image:
# the matplotlib way:
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()


# I hope I did justice to the dataset. Do comment if you have any suggestion for this notebook. 
# Cheers!
