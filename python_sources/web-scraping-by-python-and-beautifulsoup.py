#!/usr/bin/env python
# coding: utf-8

# In this kaggle kernel, I'm going to do web scraping of Website "https://www.holidify.com/explore/" Page.
# This Webpage Contain 60 Places to Visit in India and I'm going to scrap Those 60 Places information like Name of the Place, then the State it is in, then Average Rating given by people who went there, then short Paragraph describing the Place, then the Total Estimated Price, then finally there numbers of major attraction points.
# 
# **Gangtok, Sikkim**
# ![](http://www.holidify.com/images/bgImages/GANGTOK.jpg)
# 

# First we Import all the Important Packages that are required to do Web Scraping.
# 
# **Numpy** and **Pandas** are standard and I always import it, who knows when it come handy.
# 
# **re** is the package required to do Regular Expression, with the help of this package, we can easily search for particular type of words or digits in a string, and re is one of the most important string manipulation package.
# 
# **requests** is a Python module that you can use to send all kinds of HTTP requests. It is an easy-to-use library with a lot of features ranging from passing parameters in URLs to sending custom headers and SSL Verification.
# 
# **BeautifulSoup** is a Python library for pulling data out of HTML and XML files. It works with your favorite parser to provide idiomatic ways of navigating, searching, and modifying the parse tree. It saves hours or days of work.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import re
import requests
from bs4 import BeautifulSoup


# And here comes the link, first we copy and paste the url and assigned it to the variable link, then we send a request to the web page to reture the information.
# 
# After requesting, we will print out the value like I did here
# print(P_link) , which gives the output of "<Response [200]>" or "200" which means you are allowed to do web scraping on such websites.
# 
# After Getting the Response we will convert the HTML page into readable form of text and assigned it to the variable P_html. 
# 
# Then we use Beatiful Soup to Convert the HTML page in to readable form by passing "html.parser".
# 

# In[ ]:


link = "https://www.holidify.com/explore/"
P_link = requests.get(link)
print(P_link)

P_html = P_link.text
P_soup = BeautifulSoup(P_html, "html.parser")


# Here we first go to the First Table of the Website Holidify and open the HTML page by **Right Clicking**, then go to **Inspect Element**.
# 
# Where we find that the all the values of the first table is in the "div" "Class" of "col-12 col-md-6 pr-md-3 result". 
# 
# Thus we type **findAll** value in our **P_soup**  inside the div class of "col-12 col-md-6 pr-md-3 result".
# 
# At last we will print the len of the total numbers of table on one webpage. Here in this case there are 60 values on first WebPage.

# In[ ]:


containers = P_soup.findAll("div", {"class" : "col-12 col-md-6 pr-md-3 result"})
print(len(containers))


# Then here we assigned the first value in the table with containers[0] and assigned it a new variable container.
# We will do all the scraping work on the first value of table first then we move on create a loop and do it for the rest of table.
# 
# After that we create a new **DataFrame** which became our actual file, which contain all the data, which we want to create from the start.
# 
# We create a list of values and assigned a name **column**,this will become our columns name of our Data **Places**.

# In[ ]:


container = containers[0]
column = ['Place','State', 'Ratings', 'About','Price', 'Attraction']
Places = pd.DataFrame(columns = column)


# Here we create a loop and print all the Values which we want to extract from the Webpage, here I want to import values like **Places Name, City Name, Ratings, About, Prices, Attraction**. 
# 
# Remember it's not always necessery that each value you want to extract are always available separately or will be in the structured form, we have to **string manipulation** to clean the data and also **split()** the values to create two or more values.

# In[ ]:


for container in containers:
    p_name = container.findAll("h2", {"class":"card-heading"})
    p_nameN = p_name[0].text[4:].strip().split()
    if len(p_nameN) == 2:        
          p_nameP = p_nameN[0]
          p_nameP = p_nameP.replace(',','')
          p_nameC = p_nameN[1]
    elif len(p_nameN) == 3:
          p_nameP = p_nameN[0]
          p_nameP = p_nameP.replace(',','')
          p_nameC = p_nameN[1] + " " + p_nameN[2]
    elif len(p_nameN) == 4:
          p_nameP = p_nameN[0]
          p_nameP = p_nameP.replace(',','')
          p_nameC = p_nameN[1] + " " + p_nameN[2] + " " + p_nameN[3]      
    else:
          p_nameP = p_nameN[0]
          p_nameC = "NaN"
    p_rating = container.findAll("span", {"class" : "rating-badge"})
    p_rating = p_rating[0].text[1:4]
    p_about = container.findAll("p",{"class": "card-text"})
    p_about = p_about[0].text
    p_price = container.findAll("p",{"class": "collection-cta"})
    if len(p_price) == 1:
        p_num = p_price[0].text.replace(',','')
        p_numb = re.findall(r'\d+', p_num)
        num = ""
        for i in p_numb:
            num += i
    else:
        num = "NaN"
    p_attraction = container.findAll("div", {"class":"content-card-footer"})
    p_attraction = p_attraction[0].text[:-12].strip()
    
    Data = pd.DataFrame([[p_nameP ,p_nameC, p_rating , p_about, num , p_attraction]])
    Data.columns = column
    Places = Places.append(Data, ignore_index = True)


# Then Finally after extracting all the Data and adding it to our DataFrame which we created earlier. We will print out the value by typing **print(Places.head())** to print the first five values.

# In[ ]:


print(Places.head())


# And at the end we will create a CSV file and export our newly created Data.

# In[ ]:


Places.to_csv("Places.csv", index = None)

