import requests   # Importing requests to extract content from a url
from bs4 import BeautifulSoup as bs # Beautifulsoup is for web scrapping...used to scrap specific content 
import re 
import os

import urllib.request
import urllib.parse
import urllib.error
from bs4 import BeautifulSoup
import ssl
import json


# For ignoring SSL certificate errors
ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

import nltk
from nltk.corpus import stopwords

import matplotlib.pyplot as plt
from wordcloud import WordCloud

# creating empty reviews list 
iphone_reviews=[]
#forest = ["the","king","of","jungle"]


#url="https://www.amazon.in/All-New-Kindle-Display-Built-Light/product-reviews/B07FRJTZ4T/ref=dpx_acr_txt?showViewpoints=1"
  #url = "https://www.amazon.in/All-New-Kindle-reader-Glare-Free-Touchscreen/product-reviews/B0186FF45G/ref=cm_cr_getr_d_paging_btm_3?showViewpoints=1&pageNumber="
 # response = requests.get(url)
  #soup = bs(response.content,"html.parser")# creating soup object to iterate over the extracted content 
  #reviews = soup.findAll("div",attrs={"class","a-fixed-right-grid view-point"})# Extracting the content under specific tags  
  #reviews



for i in range(1,4):
  ip=[]  
  url="https://www.amazon.in/All-New-Kindle-Display-Built-Light/product-reviews/B07FRJTZ4T/ref=dpx_acr_txt?showViewpoints="+str(i)
  #url = "https://www.amazon.in/All-New-Kindle-reader-Glare-Free-Touchscreen/product-reviews/B0186FF45G/ref=cm_cr_getr_d_paging_btm_3?showViewpoints=1&pageNumber="
  response = requests.get(url)
  soup = bs(response.content,"html.parser")# creating soup object to iterate over the extracted content 
  reviews = soup.findAll("div",attrs={"class","a-fixed-right-grid view-point"})# Extracting the content under specific tags  
  reviews
  for i in range(len(reviews)):
    ip.append(reviews[i].text)  
    iphone_reviews=iphone_reviews+ip  # adding the reviews of one page to empty list which in future contains all the reviews
iphone_reviews
#os.chdir('D:\Python codes\Text mining NLP')
# writng reviews in a text file 
with open("iphone.txt","w",encoding='utf8') as output:
    output.write(str(iphone_reviews))
    
    
    
# Joinining all the reviews into single paragraph 
ip_rev_string = " ".join(iphone_reviews)



# Removing unwanted symbols incase if exists
ip_rev_string = re.sub("[^A-Za-z" "]+"," ",ip_rev_string).lower()
ip_rev_string = re.sub("[0-9" "]+"," ",ip_rev_string)
ip_rev_string


# words that contained in iphone 7 reviews
ip_reviews_words = ip_rev_string.split(" ")

stop_words = stopwords.words('english')

with open("D:\\Python codes\\Text mining NLP\\sw.txt","r") as sw:
    stopwords = sw.read()

stopwords = stopwords.split("\n")


temp = ["this","is","awsome","Data","Science"]
[i for i in temp if i not in "is"]

ip_reviews_words = [w for w in ip_reviews_words if not w in stopwords]



# Joinining all the reviews into single paragraph 
ip_rev_string = " ".join(ip_reviews_words)

# WordCloud can be performed on the string inputs. That is the reason we have combined 
# entire reviews into single paragraph
# Simple word cloud


wordcloud_ip = WordCloud(
                      background_color='black',
                      width=1800,
                      height=1400
                     ).generate(ip_rev_string)

plt.imshow(wordcloud_ip)

# positive words # Choose the path for +ve words stored in system
with open("D:\\Python codes\\Text mining NLP\\positive-words.txt","r") as pos:
  poswords = pos.read().split("\n")

poswords  
poswords = poswords[36:]



# negative words  Choose path for -ve words stored in system
with open("D:\\Python codes\\Text mining NLP\\negative-words.txt","r") as neg:
  negwords = neg.read().split("\n")

negwords = negwords[37:]

# negative word cloud
# Choosing the only words which are present in negwords
ip_neg_in_neg = " ".join ([w for w in ip_reviews_words if w in negwords])

wordcloud_neg_in_neg = WordCloud(
                      background_color='black',
                      width=1800,
                      height=1400
                     ).generate(ip_neg_in_neg)

plt.imshow(wordcloud_neg_in_neg)

# Positive word cloud
# Choosing the only words which are present in positive words
ip_pos_in_pos = " ".join ([w for w in ip_reviews_words if w in poswords])
wordcloud_pos_in_pos = WordCloud(
                      background_color='black',
                      width=1800,
                      height=1400
                     ).generate(ip_pos_in_pos)

plt.imshow(wordcloud_pos_in_pos)
