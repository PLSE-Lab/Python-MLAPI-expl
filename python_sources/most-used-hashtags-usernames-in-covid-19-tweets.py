#!/usr/bin/env python
# coding: utf-8

# # Most Used Hashtags and Usernames in Covid-19 Tweets
# 
# <table>
#   <tr>
#     <td><img src="https://upload.wikimedia.org/wikipedia/commons/8/82/SARS-CoV-2_without_background.png" width=270 height=480></td>
#     <td><img src="https://help.twitter.com/content/dam/help-twitter/brand/logo.png" width=270 height=480></td>
#   </tr>
#  </table>
# <br>
# <br>
# 
# **Hello,**
# 
# **In this kernel we will extract hashtags and user nick names from tweets and visualize most used ones.**

# In[ ]:


# We will use this libraries
import numpy as np
import pandas as pd
import nltk

from PIL import Image
import urllib.request
from collections import Counter
from wordcloud import WordCloud, ImageColorGenerator

import matplotlib.pyplot as plt
import plotly.graph_objects as go


# In[ ]:


# Reading data from csv
dataFrame = pd.read_csv("/kaggle/input/covid19-250000-tweets/covid19_en.csv")


# In[ ]:


# This functions gets a tweet as string and finds hashtags or usernames

def extractHashTags(tweet):
    # We use nltk TweetTokenizer to tokenize twitter special keywords like #abc or @abc
    tokenized_tweet = nltk.tokenize.TweetTokenizer().tokenize(tweet)
    # We check every token/word first char
    hashTags = [word for word in tokenized_tweet if word[0] == "#" and len(word) > 1]
    return hashTags

def extractUserNames(tweet):
    tokenized_tweet = nltk.tokenize.TweetTokenizer().tokenize(tweet)
    userNames = [word for word in tokenized_tweet if word[0] == "@" and len(word) > 1]
    return userNames


# In[ ]:


# We apply functions to our dataframe. Results will write as new columns
dataFrame["userNames"] = dataFrame["tweet"].apply(extractUserNames)
dataFrame["hashTags"] = dataFrame["tweet"].apply(extractHashTags)


# In[ ]:


# Filter if there is no hashtag or username in a tweet
userNames = dataFrame[dataFrame["userNames"].str.len() != 0]['userNames']
hashTags = dataFrame[dataFrame["hashTags"].str.len() != 0]['hashTags']


# In[ ]:


# Merge lists as a one new list
uN = [i for u in userNames.values for i in u]
hT = [i for h in hashTags.values for i in h]


# In[ ]:


# To find most frequent elements in a list we need this function
def mostFrequentelemnts(List, n): 
    occurence_count = Counter(List) 
    return occurence_count.most_common(n)


# ## Most Used Usernames in Tweets

# In[ ]:


# We use plotly bar graph to visualize most used 10 usernames

fig = go.Figure()
fig.add_trace(go.Bar(
    x=[i[0] for i in mostFrequentelemnts(uN, 10)],
    y=[i[1] for i in mostFrequentelemnts(uN, 10)],
    marker_color='aqua',
    opacity=0.5
))

# Here we modify the tickangle of the xaxis, resulting in rotated labels.
fig.update_layout(title = "Most Used 10 Usernames", barmode='group', xaxis_tickangle=-45)
fig.show()


# In[ ]:


# Create our images and show
mask = np.array(Image.open(urllib.request.urlopen("https://dslv9ilpbe7p1.cloudfront.net/_FsA5Yg9iTCQnkQFlfyrxw_store_header_image")))

wc = WordCloud(background_color="black", width = 1920, height = 1080)

wc.generate(" ".join(uN))

image_colors = ImageColorGenerator(mask)
wc.recolor(color_func=image_colors)

fig = plt.figure(figsize = (40, 30))
plt.imshow(wc, interpolation = 'bilinear')
plt.axis('off')
plt.tight_layout(pad=0)
plt.show()


# ## Most Used Hashtags in Tweets

# In[ ]:


wc = WordCloud(background_color="black", width = 1920, height = 1080)

wc.generate(" ".join(hT))

image_colors = ImageColorGenerator(mask)
wc.recolor(color_func=image_colors)

fig = plt.figure(figsize = (40, 30))
plt.imshow(wc, interpolation = 'bilinear')
plt.axis('off')
plt.tight_layout(pad=0)
plt.show()


# ### There are lot of covid strings here as expected. So lets filter from covid or corona keywords for hashtags.

# In[ ]:


filtered_hT = [hashtag for hashtag in hT if "covid" not in hashtag.lower() and "corona" not in hashtag.lower()]


# In[ ]:


fig = go.Figure()
fig.add_trace(go.Bar(
    x=[i[0] for i in mostFrequentelemnts(filtered_hT, 10)],
    y=[i[1] for i in mostFrequentelemnts(filtered_hT, 10)],
    marker_color='magenta',
    opacity=0.5
))

# Here we modify the tickangle of the xaxis, resulting in rotated labels.
fig.update_layout(title = "Most Used 10 Hashtags", barmode='group', xaxis_tickangle=-45)
fig.show()


# In[ ]:


wc = WordCloud(background_color="black", width = 1920, height = 1080)

wc.generate(" ".join(filtered_hT))

image_colors = ImageColorGenerator(mask)
wc.recolor(color_func=image_colors)

fig = plt.figure(figsize = (40, 30))
plt.imshow(wc, interpolation = 'bilinear')
plt.axis('off')
plt.tight_layout(pad=0)
plt.show()


# ### Let's check other languages.

# # Turkish

# In[ ]:


dataFrame = pd.read_csv("/kaggle/input/covid19-250000-tweets/covid19_tr.csv", encoding='utf-8')

# We apply functions to our dataframe. Results will write as new columns
dataFrame["userNames"] = dataFrame["tweet"].apply(extractUserNames)
dataFrame["hashTags"] = dataFrame["tweet"].apply(extractHashTags)

# Filter if there is no hashtag or username in a tweet
userNames = dataFrame[dataFrame["userNames"].str.len() != 0]['userNames']
hashTags = dataFrame[dataFrame["hashTags"].str.len() != 0]['hashTags']

# Merge lists as a one new list
uN = [i for u in userNames.values for i in u]
hT = [i for h in hashTags.values for i in h]


# ### Usernames

# In[ ]:


fig = go.Figure()
fig.add_trace(go.Bar(
    x=[i[0] for i in mostFrequentelemnts(uN, 10)],
    y=[i[1] for i in mostFrequentelemnts(uN, 10)],
    marker_color='aqua',
    opacity=0.5
))

# Here we modify the tickangle of the xaxis, resulting in rotated labels.
fig.update_layout(title = "Most Used 10 Usernames", barmode='group', xaxis_tickangle=-45)
fig.show()


# In[ ]:


wc = WordCloud(background_color="black", width = 1920, height = 1080)

wc.generate(" ".join(uN))

image_colors = ImageColorGenerator(mask)
wc.recolor(color_func=image_colors)

fig = plt.figure(figsize = (40, 30))
plt.imshow(wc, interpolation = 'bilinear')
plt.axis('off')
plt.tight_layout(pad=0)
plt.show()


# In[ ]:


filtered_hT = [hashtag for hashtag in hT if "covid" not in hashtag.lower() and "corona" not in hashtag.lower()]


# ### Hashtags

# In[ ]:


fig = go.Figure()
fig.add_trace(go.Bar(
    x=[i[0] for i in mostFrequentelemnts(filtered_hT, 10)],
    y=[i[1] for i in mostFrequentelemnts(filtered_hT, 10)],
    marker_color='magenta',
    opacity=0.5
))

# Here we modify the tickangle of the xaxis, resulting in rotated labels.
fig.update_layout(title = "Most Used 10 Hashtags", barmode='group', xaxis_tickangle=-45)
fig.show()


# In[ ]:


wc = WordCloud(background_color="black", width = 1920, height = 1080)

wc.generate(" ".join(filtered_hT))

image_colors = ImageColorGenerator(mask)
wc.recolor(color_func=image_colors)

fig = plt.figure(figsize = (40, 30))
plt.imshow(wc, interpolation = 'bilinear')
plt.axis('off')
plt.tight_layout(pad=0)
plt.show()


# # German

# In[ ]:


dataFrame = pd.read_csv("/kaggle/input/covid19-250000-tweets/covid19_de.csv", encoding='utf-8')

# We apply functions to our dataframe. Results will write as new columns
dataFrame["userNames"] = dataFrame["tweet"].apply(extractUserNames)
dataFrame["hashTags"] = dataFrame["tweet"].apply(extractHashTags)

# Filter if there is no hashtag or username in a tweet
userNames = dataFrame[dataFrame["userNames"].str.len() != 0]['userNames']
hashTags = dataFrame[dataFrame["hashTags"].str.len() != 0]['hashTags']

# Merge lists as a one new list
uN = [i for u in userNames.values for i in u]
hT = [i for h in hashTags.values for i in h]


# ### Usernames

# In[ ]:


fig = go.Figure()
fig.add_trace(go.Bar(
    x=[i[0] for i in mostFrequentelemnts(uN, 10)],
    y=[i[1] for i in mostFrequentelemnts(uN, 10)],
    marker_color='aqua',
    opacity=0.5
))

# Here we modify the tickangle of the xaxis, resulting in rotated labels.
fig.update_layout(title = "Most Used 10 Usernames", barmode='group', xaxis_tickangle=-45)
fig.show()


# In[ ]:


wc = WordCloud(background_color="black", width = 1920, height = 1080)

wc.generate(" ".join(uN))

image_colors = ImageColorGenerator(mask)
wc.recolor(color_func=image_colors)

fig = plt.figure(figsize = (40, 30))
plt.imshow(wc, interpolation = 'bilinear')
plt.axis('off')
plt.tight_layout(pad=0)
plt.show()


# In[ ]:


filtered_hT = [hashtag for hashtag in hT if "covid" not in hashtag.lower() and "corona" not in hashtag.lower()]


# ### Hashtags

# In[ ]:


fig = go.Figure()
fig.add_trace(go.Bar(
    x=[i[0] for i in mostFrequentelemnts(filtered_hT, 10)],
    y=[i[1] for i in mostFrequentelemnts(filtered_hT, 10)],
    marker_color='magenta',
    opacity=0.5
))

# Here we modify the tickangle of the xaxis, resulting in rotated labels.
fig.update_layout(title = "Most Used 10 Hashtags", barmode='group', xaxis_tickangle=-45)
fig.show()


# In[ ]:


wc = WordCloud(background_color="black", width = 1920, height = 1080)

wc.generate(" ".join(filtered_hT))

image_colors = ImageColorGenerator(mask)
wc.recolor(color_func=image_colors)

fig = plt.figure(figsize = (40, 30))
plt.imshow(wc, interpolation = 'bilinear')
plt.axis('off')
plt.tight_layout(pad=0)
plt.show()

