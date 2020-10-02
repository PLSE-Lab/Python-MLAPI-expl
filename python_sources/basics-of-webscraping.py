#!/usr/bin/env python
# coding: utf-8

# ## Basics of Webscraping
#      
#    Web Scraping is one of the important concepts that every Data Scientist must learn ,In this tutorial I will be showing you the bare basics of Web     Scraping like how to fetch a WebPage and scrape details from it and store it as CSV file.
#                
#                
#   ![Web Scraping](https://p0.pxfuel.com/preview/1022/769/167/social-media-twitter-googleplus-royalty-free-thumbnail.jpg)                 

# The important modules for Webscraping are 
# - requests
# - BeautifulSoup4
# 
# **requests**: requests module is an awesome Http Client for Python used to make Http requests to a website.
# **BeautifulSoup4** : BeautifulSoup4 is a library to parse the response 

# # Importing the Necessary Modules

# In[ ]:


from bs4 import BeautifulSoup
import requests 


# # Note
# 
# Before starting to Scrape a Website we must see their policies like which routes they allow us to scrape  Every major site has a route **robots.txt**. We will be  using my website to scrape

# # Making Http Request

# In[ ]:


res = requests.get('https://vishnurao.tech/')


# ## We are fetching the websites content

# In[ ]:


res.text


# ## We are passing the HTML Content to the BeautifulSoup Parser

# In[ ]:


soup =BeautifulSoup(res.text)


# ## Getting the Title Element of the Website

# In[ ]:


soup.title


# ## To extract the Text from the HTML Element use the **text** property of the  website

# In[ ]:


title = soup.title.text


# In[ ]:


title


# ## Finding all the **links** in the Website

# In[ ]:


list_of_a_tags = soup.find_all('a')


# In[ ]:


list_of_a_tags


# ## To find all the the occurences of the given tag  find_all method is used

# In[ ]:


for link in list_of_a_tags:
    print(link['href'])


# ## **href** is known as the attribute  of the Html tag when the HTML is passed to the Parser a tag is converted as dictionary its corresponding attributes are keys in the dictionary

# ## Finding all the dates

# In[ ]:


dates_list= soup.find_all('small')


# In[ ]:


for date in dates_list:
    print(date.text)


# ## Finding image in the Website , To find the first occurence of the element find is used

# In[ ]:


image_tag= soup.find('img')
image_tag


# In[ ]:


image_tag['src']


# ## To select a element with particular Class or id we can pass a dict with the required arguments to the find method

# In[ ]:


div_with_id = soup.find('div',{'id':"gatsby-focus-wrapper"})


# In[ ]:


div_with_id


# ## If we try to select a element that doesn't exist None will be returned

# In[ ]:


non_existent_element = soup.find('div',{"id":"foo"})


# In[ ]:


print(non_existent_element)


# ## Further Resources
# 
#  - https://www.crummy.com/software/BeautifulSoup/bs4/doc/
#  - https://www.youtube.com/watch?v=ng2o98k983k&t=5s
#  - https://www.youtube.com/watch?v=XQgXKtPSzUI
#  - https://vishnurao.tech/day-2-100-days-of-code/

# In[ ]:




