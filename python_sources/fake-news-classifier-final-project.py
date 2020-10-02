#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
print(os.listdir("../input"))


# This Kaggle notebook serves to hold all our data, code,  explanations, machine learning models, and explanations for our Fake News Classifier final project for our data science class.
# 
# Here's our multi-part approach for the project.
# 
# 1. data collection:
#     - have two datasets (fakes news and real news)
#     - explain rationale
# 
# 2. data preprocessing:
#     - filter and manipulate columns, add labels (fake or real), and merge on selected columns
# 
# 3. preprocessing the text:
# 
# 4. text to features conversion:
# 
# 5. classification + model selection
# 
# 6. topic modeling
# 
# 7. conclusions
# 
# 

# In[ ]:


fake_news = pd.read_csv("../input/fake-news/fake.csv")
real_news = pd.read_csv("../input/gathering-real-news-for-oct-dec-2016/real_news.csv")


# In[ ]:


#Here are the size of our datasets:
print(fake_news.shape)
print(real_news.shape)


# In[ ]:


# Let's see what columns we have
print(list(fake_news.columns))
print(list(real_news.columns))


# Now let's obtain similar features for both datasets before we combine them.
# 
# title - title of article  
# content - article text  
# publication - which company published this article  
# label - real or fake
# 
# Regarding publication for both of the datasets, the real news has the publication company in the 'publication' column while the fake news has the publication company embedded in the url in the 'site_url' column.
# 
# We will have to parse the site_urls into publication names before we merge.

# In[ ]:


# now let's obtain similar features for both datasets before we combine them
# Let's add our label to the dataset "real" for real news and "fake" for fake news

real_news2 = real_news[['title', 'content', 'publication']]
real_news2['label'] = 'real'
real_news2.head(15)


# In[ ]:


fake_news2 = fake_news[['title', 'text','site_url']]
fake_news2['label'] = 'fake'
fake_news2.head(15)


# In[ ]:


# let's obtain all the unique site_urls
site_urls = fake_news2['site_url']

# let's remove the domain extensions
site_urls2 = [x.split('.',1)[0] for x in site_urls]

# now let's replace the old site_url column
fake_news2['site_url'] = site_urls2
fake_news2.head()


# In[ ]:


# let's rename the features in our datasets to be the same
newlabels = ['title', 'content', 'publication', 'label']
real_news2.columns = newlabels
fake_news2.columns = newlabels

# let's concatenate the dataframes
frames = [fake_news2, real_news2]
news_dataset = pd.concat(frames)
news_dataset


# Let's save our data frame as a new csv file called "news_dataset.csv".

# In[ ]:


news_dataset.to_csv('news_dataset.csv', encoding='utf-8')

