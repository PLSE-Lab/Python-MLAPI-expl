#!/usr/bin/env python
# coding: utf-8

# # **Scraping Pitchfork's review on Eminem's albums**

# # **Will the real SLIM SHADY please stand up?**
# In this notebook we will be scraping , cleaning and visualizing pitchfork's review on Eminem's albums.

# ![EMINEM](https://static.independent.co.uk/s3fs-public/thumbnails/image/2020/05/21/12/3318f3cf181ad2673fab74719f3eed40.1000x1000x1-2.png?w968h681)

# # Part 1 : Scraping Data
# We will be scraping Pitchfork's Eminem's artist page.\
# Will be using Scrapy for creating a crawler.\
# Crawler(in case you don't know about it):A program that systematically browses the World Wide Web in order to create an index of data.
# 

# In[ ]:


get_ipython().system('pip install scrapy #installing scrapy')


# ### Importing required libraries

# In[ ]:


import numpy as np
import pandas as pd
import re # for cleaning data
import scrapy 
from scrapy.crawler import CrawlerProcess
from scrapy import Selector
import requests


# ## Naive Method
# I will suggest you to build crawlers rather than this method as crawlers are more automated and easier to modify.
# Look for the crawler I created below this naive method.

# In[ ]:


#This is a normal scraping method without building a crawler class 
'''url="https://pitchfork.com/artists/1339-eminem"
html=requests.get(url).content
sel=Selector(text=html)
album_reviews={}
links_to_the_reviews=sel.xpath('//div[@class="review"]/a/@href').extract()
for link in links_to_the_reviews:
  album_page=requests.get("https://pitchfork.com/"+link).content
  sel_album=Selector(text=album_page)
  review=sel_album.xpath("//div[@class='contents dropcap']//text()").extract()
  name=sel_album.xpath('//h1[@class="single-album-tombstone__review-title"]/text()').extract()[0]
  #print(name)
  for i in range(review.count('\n')):
    review.remove('\n')
  review="".join(review)

  album_reviews[name]=review'''  


# ## Building the crawler(**A cute one tho**)
# ![alt text](https://images.news18.com/ibnlive/uploads/2017/11/lucas.jpg?impolicy=website&width=536&height=356)

# ## The art of war
# 1. The crawler will firstly load up the links to the reviews from the Pitchfork's page of Eminem.  
# 2. The next step is to retrieve the reviews from the extracted links of the albums.
# 3. On the review page the name of the album and the review will be stored in different lists.\
# [Eminem's Pitchfork artist page](https://pitchfork.com/artists/1339-eminem)

# In[ ]:


class scraper(scrapy.Spider):
  name="scraper"
  def start_requests(self):
    url="https://pitchfork.com/artists/1339-eminem"
    yield scrapy.Request(url=url,callback=self.parse)
  def parse(self,response):
    links=response.css('div.review>a::attr(href)').extract()
    for link in links:
      yield response.follow(url=link,callback=self.parse2)
  def parse2(self,response):
    name=response.css('h1.single-album-tombstone__review-title::text').extract()[0]
    names.append(name)
    review=response.css('div.contents.dropcap ::text').extract()
    reviews.append(review)     


# ### If you have any problem in understanding how the spider works , I highly recommend you to go through DataCamp's [Web scraping with python](https://learn.datacamp.com/courses/web-scraping-with-python). 

# In[ ]:


#Starting the crawler to crawl through web pages and extract the data of our need.
reviews=[]
names=[]
process = CrawlerProcess()
process.crawl(scraper)
process.start()


# # Part 2 : Data Munging and Cleaning 
# So we have the name of the albums in **names** and the review of the respective album in **reviews**. 
# 
# *   names
# *   reviews
# 
# 

# In[ ]:


#let us check how our review looks like
reviews[0]


# ***Confused Eminem noise***
# 
# <img src="https://i.gifer.com/PmtJ.gif"> 

# In[ ]:


#Joining the data
for i in range(len(reviews)):
  reviews[i]="".join(reviews[i])
  reviews[i]=re.sub("\n"," ",reviews[i])


# In[ ]:


data=pd.DataFrame(list(zip(names,reviews)),columns=["Album_name","Review"])
data.head()


# **HUFFF RELIEF.......**

# # Part 3 : So we are done with the cleaning part. Let's get into the analyzing part
# <img src="https://media.giphy.com/media/NS7gPxeumewkWDOIxi/giphy.gif">

# In[ ]:


data["Album_name"] #Albums 


# ## We will create a graph of the most frequent words the critics used in the review to get a quick look over the review and will build a WordCloud to dig a bit deeper 

# In[ ]:


#Importing required libraries
from collections import Counter
import matplotlib.pyplot as plt
from wordcloud import WordCloud,STOPWORDS 
import spacy
import seaborn as sns
from PIL import Image
nlp=spacy.load('en_core_web_sm')
import logging


# ## Let us see what the pitchfork's critics have to say about Eminem's Music to be murdered by 

# In[ ]:


Album_name=data.iloc[0,0] #Music to be murdered by
Album_review=data.iloc[0,1] #review of the album


# In[ ]:


Common_words=[] #This will contain the most frequent words used by reviewers
doc=nlp(Album_review) #Tokenizing the review
tokens=[token.lower_ for token in doc if not token.is_punct and not token.is_stop and token.lower_  not in "eminem"] #Removing stop words 
#and punctuations
count=Counter(tokens) 
count=count.most_common(40)


# In[ ]:


logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
labels=[tup[0] for tup in count]
value=[tup[1] for tup in count]
plt.figure(figsize=(10,5))
plot=sns.barplot(x=labels,y=value)
plot.set_xticklabels(labels=labels,rotation=90)
plot.set_title("Review:Music To Be Murdered By(most frequent words)")


# ## We can get more information by making a wordcloud.
# 
# 
# ---
# But a normal wordcloud is basic and we don't do basics, Right?
# Let's make it a bit unique so that it resembles to Eminem.
# 
# 
# 

# In[ ]:


#from PIL import Image
#from google.colab import files
#image=files.upload() # I made this notebook on colab , you might be on a different platform so look out when doing this cell.


# In[ ]:


my_mask=np.array(Image.open("../input/6930355.png"))
cloud=WordCloud(background_color="white",mask=my_mask,stopwords=STOPWORDS)
cloud.generate(data.iloc[0,-1])
#cloud.to_file("Eminem.png")
plt.figure(figsize=(10,10))
plt.imshow(cloud,interpolation='bilinear')
plt.tight_layout()
plt.axis("off")


# ### We can now look upto the keywords we are interested in and it looks beautiful isn't it?

# # **THE END**
# ##        I hope you enjoyed the notebook and got some new things to learn.
# ###       Upvotes are always appreciated
# ![alt text](https://1zl13gzmcsu3l9yq032yyf51-wpengine.netdna-ssl.com/wp-content/uploads/2017/12/eminem-lyrics-to-motivate-you-1068x561.jpg)
