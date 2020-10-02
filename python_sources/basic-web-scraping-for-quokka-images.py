#!/usr/bin/env python
# coding: utf-8

# * <h1>Basic web scraping for Quokka Images</h1>
# 
# In this kernel I will be looking for Quokka Images on a static website (wikipedia in this case).
# 
# url = https://en.wikipedia.org/wiki/Quokka
# 
# below is a Quokka for reference
# 

# <img src="https://images.unsplash.com/photo-1513333420772-7b64ad15ca96?ixlib=rb-1.2.1&ixid=eyJhcHBfaWQiOjEyMDd9&auto=format&fit=crop&w=934&q=80" style="float:left;width:300px" alt="Drawing" />
# 
# <br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br>
#  Quokka by Natalie Su on unsplash
#  <br>
#  

# In[ ]:


#Imports
import os
import requests
from bs4 import BeautifulSoup
import urllib.request
from IPython.display import Image


# In[ ]:


##Scrapping static pages

#Static website
url = "https://en.wikipedia.org/wiki/Quokka"
try:
    webpage = requests.get(url)
except requests.exceptions.RequestsExceptions as e:
    print(e)

#Show content of webpage
#Parse downloaded content into a BeautifulSoup object
soup = BeautifulSoup(webpage.content, "lxml")
#print(soup.prettify())


# In[ ]:


#List containing all url src from html with img tag
images_url = []
#Filter out the url with Quokka keyword
quokka_img_url = []
for link in soup.find_all('img'):
    images_url.append(link.get('src'))

#Completing the url
for img in images_url:
    if "Quokka" in img:
        quokka_img_url.append("https:" + img)
    if "Qokka" in img:
        quokka_img_url.append("https:" + img)
        
quokka_img_url


# In[ ]:


#Displaying image from url
for url in quokka_img_url:
    display(Image(url,width=300, height=300))


# **Unable to add a folder to this workspace so the code below will be in markdown**
# 
# **Required to make a new dir "quokka_images"

# **Downloading the image into a local directory**
# <Code>def dl_img(url, path, fileName):
#     full_path = path + fileName + ".jpg"
#     urllib.request.urlretrieve(url,full_path)
# 
# image_number = 0
# for url in quokka_img_url:
#     fileName = "Quokka_" + str(image_number)
#     image_number+=1
#     print(url)
#     dl_img(url, "quokka_images/", fileName)<Code>

# **Displaying image from local directory**
# <code>for file in os.listdir('quokka_images'):
#     if file.endswith(".jpg"):
#         file_name = "quokka_images/" + file
#         print(file_name)
#         display(Image(filename = file_name, width=100, height=100))<code>
