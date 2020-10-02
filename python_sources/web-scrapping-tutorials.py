#!/usr/bin/env python
# coding: utf-8

# # Web Scraping Tutorials
# 
# * In this Notebook, I will cover some of the most important libraries used in the space of web scraping used in the field of Data Science.
# 
# * **Contents of the Tutorials**
# 
#     *  ```Web Scraping to Scrap Twitter```
#     *  ```Web Scraping to Fetch News from Websites```
#     *  ```Web Scraping to Fetch News from Newspapers and Articles```
#     *  ```Web Scraping to Fetch Financial Data```
#     *  ```Web Scraping to Scrap Reddit```
# 

# ## Installing Required Libraries

# In[ ]:


get_ipython().system('pip install twitterscraper')


# In[ ]:


get_ipython().system('pip install selenium')


# In[ ]:


get_ipython().system('pip install beautifulsoup4')


# In[ ]:


get_ipython().system('pip install newspaper3k')


# In[ ]:


get_ipython().system('pip install scrapy')


# In[ ]:


get_ipython().system('pip install praw')


# ## Installing Libraries

# In[ ]:


# importing basic libraries
import datetime as dt
import pandas as pd
import numpy as np

# importing libraries for scraping
from twitterscraper import query_tweets
from urllib.request import urlopen
from selenium import webdriver
from bs4 import BeautifulSoup
from newspaper import Article
import scrapy
import praw


# ## Scraping Twitter 

# In[ ]:


from twitterscraper import query_tweets
begin_date = dt.date(2019 , 9 , 1)
end_date = dt.date(2019 , 9 , 2)

limit = 100
lang = 'english'

def tweet(emp):
    tweets = query_tweets(emp , begindate = begin_date , enddate = end_date , limit = limit , lang = lang)
    Df = pd.DataFrame(t.__dict__ for t in tweets)

    
    print(Df.columns)
    Texts = list(Df.text)

    for i in Texts :
        print("  -----------------------------------------------------------------------------------------")
        print(i)
        print("  -----------------------------------------------------------------------------------------")
    


# In[ ]:


tweet('taylor swift')


# ## Scraping Google News[](http://)

# In[ ]:


def scrap(url):
    Client = urlopen(url)
    xml_page = Client.read()
    Client.close()
    soup_page = BeautifulSoup(xml_page, "xml")
    news_list = soup_page.findAll("item")

    # print news title, url and publish date
    
    for news in news_list:
        news_list = set()
        print(news.title.text), print(news.link.text), print(news.pubDate.text)


# In[ ]:


scrap("http://news.google.com/news/rss")


# ## Scraping Newspapers/Articles

# In[ ]:


#A new article from TOI 
url = "https://timesofindia.indiatimes.com/"
  


# In[ ]:


#For different language newspaper refer above table 
toi_article = Article(url, language="en") # en for English 


# In[ ]:


import nltk
nltk.download('punkt')


# In[ ]:


#To download the article 
toi_article.download() 
  
#To parse the article 
toi_article.parse() 
  
#To perform natural language processing ie..nlp 
toi_article.nlp()


# In[ ]:


#To extract title 
print("Article's Title:") 
print(toi_article.title) 
print("n") 
  
#To extract text 
print("Article's Text:") 
print(toi_article.text) 
print("n") 
  
#To extract summary 
print("Article's Summary:") 
print(toi_article.summary) 
print("n") 
  
#To extract keywords 
print("Article's Keywords:") 
print(toi_article.keywords)


#  ## Scraping Reddit

# In[ ]:


def headline(company):
    headlines = set()
    reddit = praw.Reddit(client_id='F2uCF0oF0q46GA',
                     client_secret='5Vo58uJsQLDfoswrzOflyWWfNdQ',
                     user_agent='roshan220206')
    for submission in reddit.subreddit('google').new(limit=None):
        headlines.add(submission.title)
       
    return headlines


# In[ ]:


headline('google')


# In[ ]:




