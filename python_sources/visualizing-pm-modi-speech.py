#!/usr/bin/env python
# coding: utf-8

# # In this notebook i explained how we can visualize any speech text.I used PM Modi 15th August 2019 speech text and i scrapped this speech text from Business Today site. We will create a visual representaion of frequently occured words in PM Modi speech.
# 
# Link for transcript of PM Modi speech: https://www.businesstoday.in/current/economy-politics/independence-day-pm-modi-address-nation-full-text-speech-15-august-red-fort/story/372903.html

# In[ ]:


import numpy as np
import pandas as pd


# ***I used requests and beautifulsoup package for web scrapping***

# In[ ]:


import requests
from bs4 import BeautifulSoup as bs


# In[ ]:


url="https://www.businesstoday.in/current/economy-politics/independence-day-pm-modi-address-nation-full-text-speech-15-august-red-fort/story/372903.html"


# In[ ]:


html=requests.get(url,"text")


# ***I used html5lib to parse HTML data***

# In[ ]:


text=bs(html.text,"html5lib")


# ***Now for step 10 you should know some basics of html. In html texts are wrapped under different different tags. We can extract particular part of text by its tags. This can be done by inspecting html code of respective site.***

# In[ ]:


text_p=text.find_all('div',{'class':'story-right relatedstory paywall'})
speech=""
for final in text_p[0].find_all('p')[5:]:
    speech=speech+final.text


# ***After inspecting given page link you will understand why i initialized list from 5th index***

# ***Now we will do some pre-processing of our text. We will remove some unwanted things like punctuations,stopwords and other things from speech.***

# In[ ]:


speech


# ***You can see that there are words like "\xao" which is not a part of speech. We are going to remove such things.***

# In[ ]:


import string
from string import punctuation


# In[ ]:


speli=[word for word in speech.split() if word != r'\xao']
speech=" ".join(speli)
spech=[ch if ch not in punctuation else " " for ch in speech]
speech="".join(spech)


# In[ ]:


import nltk
nltk.download("stopwords")
from nltk.corpus import stopwords
stopword=stopwords.words('english')


# In[ ]:


speli=[word for word in speech.split() if word not in stopword]
speech=" ".join(speli)


# In[ ]:


speech


# ***In the next LOC i make count of unique words occured in speech.***

# In[ ]:


from collections import Counter


# In[ ]:


word_count=Counter(speech.split())


# ***In the next Line i made a list of top 50 words with their count. And plotted a bar plot for the same using plotly***

# In[ ]:


key=[]
value=[]
for tup in word_count.most_common()[:50]:
    key.append(tup[0])
    value.append(tup[1])


# In[ ]:


import plotly.express as px


# In[ ]:


px.bar


# In[ ]:


px.bar(x=key,y=value,labels={"x":"Most Frequent Words","y":"Frequency"},hover_name=key)


# ***There are some other methods also which can be used to plot most frequent words. wordcloud is the one of them which is used for visual representation of text data.***

# In[ ]:


from wordcloud import WordCloud
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


#use this command to see the parameters that we can pass in WordCloud.
get_ipython().run_line_magic('pinfo', 'WordCloud')


# In[ ]:


wordcloud = WordCloud(max_words=50,include_numbers=True,min_word_length=2).generate(speech)
plt.figure(figsize=(12,12))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()


# # Thank You
