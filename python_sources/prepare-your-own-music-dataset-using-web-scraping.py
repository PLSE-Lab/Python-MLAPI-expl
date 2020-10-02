#!/usr/bin/env python
# coding: utf-8

# # I am going to explain you how you can create your own music dataset using web scraping

# > Being a data scientist one of the most important skill you need to have is the ability to prepare your own dataset.
# > I am going to teach you how to scrape data to make a simple music and genre dataset.
# > Later I will make more tutorials based on user requests, if more than **5 to 10 people** asks me to explain more about data scraping I will do so.
# > Please comment your feedback in notebook's comment section 
# > 

# I am using Free Music Archieves platform for demo.
# This is out data source.
# https://freemusicarchive.org

# In[ ]:


from os import system
import requests 
from lxml import html
import pandas as pd


# 1. Here Requests package handles all the requests which we are going to send and recieve 
# 2. lxml package used to parse details
# 3. pandas for dataframe
# 

# In[ ]:


Links = []
Genres= []
for k in range(1, 10):
    bu = "https://freemusicarchive.org/genre/Folk?sort=track_date_published&d=1&page=" #base url
    key = k #page number
    print (key)    
    final_link = bu + str(key)
    print(final_link)
    page = requests.get(final_link)
    pData = html.fromstring(page.content)
    headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36'}
    newP = requests.get(final_link,headers=headers)
    ntree = html.fromstring(newP.content)
    for j in range (1, 21):
      pL = ntree.xpath('//*[@id="content"]/div[3]/div[1]/div[2]/div[{}]/span[2]/a/@href'.format(j))
      if len(pL)!=0:
        Links.append(pL[0])
      gnrs = []
      o = 5
      while o>=0:
        add = ntree.xpath('//*[@id="content"]/div[3]/div[1]/div[2]/div[{}]/div/span[4]/a[{}]/@href'.format(j,o))
        if len(add)!=0:
          gnrs.append(add[0])
          print("Now working at ", o)
        o-=1
      Genres.append(gnrs)
    

print(len(Links))
print(len(Genres))


# **Now I am adding these to a dataframe**

# In[ ]:


for item in Genres:
    for i in range(0, len(item)):
        item[i] = item[i].replace("/genre/","")
        item[i] = item[i].replace("/","")


# In[ ]:


scraped = pd.DataFrame()


# In[ ]:


scraped["Links"] = Links
scraped["Genres"] = Genres


# In[ ]:


scraped.head(5)


# > You can download songs directly to any folder by using below code snippet

# In[ ]:


song = scraped["Links"][0]
song


# In[ ]:


get_ipython().system('wget -x --load-cookies cookies.txt -nH https://files.freemusicarchive.org/storage-freemusicarchive-org/music/ccCommunity/Aglow_Hollow/Proximate_Laws_Baba_Yaga_Booty_Calls/Aglow_Hollow_-_04_-_Dog_Soldier___Stand_Down.mp3')


# In[ ]:


ls storage-freemusicarchive-org/music/ccCommunity/Aglow_Hollow/Proximate_Laws_Baba_Yaga_Booty_Calls/


# # Woooh Woohhhh!! With this demo you might have learnt something new I guess, Please upvote this notebook if you find it clear/useful to you or to anyone. Share if you can.I believe knowledge sharing as Caring

# P.S: Please ignore typos & grammatical...

# 

# 
