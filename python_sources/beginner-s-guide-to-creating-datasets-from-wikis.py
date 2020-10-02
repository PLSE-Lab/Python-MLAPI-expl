#!/usr/bin/env python
# coding: utf-8

# # A Beginner's Guide to Creating Datasets from Wikis
# 
# In an effort to increase the amount of interesting datasets that are uploaded to this site, I thought I'd give a basic example of taking a dataset from a wikipage and uploading it. I'll use a small, non-interesting dataset as an introduction; the [base stats of Skyrim races](https://elderscrolls.fandom.com/wiki/Races_(Skyrim)). I'll use the 'BeautifulSoup' package but there are many alternative ways of doing this.

# In[ ]:


import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
import requests
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from matplotlib.pyplot import figure

img=mpimg.imread('/kaggle/input/skyrim/skyrimtable.bmp')
figure(num=None, figsize=(16, 12), dpi=80, facecolor='w', edgecolor='k')

imgplot = plt.imshow(img)


# # Setup
# 
# First we need to take the actual webpage and turn it into a html file that we can search through. This will work for most wiki webpages, from my experience

# In[ ]:


page = requests.get("https://elderscrolls.fandom.com/wiki/Races_(Skyrim)").text
soup = BeautifulSoup(page, 'html.parser')


# Next step is to right-clicking a part of our table and hitting 'inspect'. This will show us the html of the page and we can figure out what we're looking for

# In[ ]:


img=mpimg.imread('/kaggle/input/skyrim/skyrimrightclick.bmp')
figure(num=None, figsize=(16, 12), dpi=80, facecolor='w', edgecolor='k')

imgplot = plt.imshow(img)


# In[ ]:


img=mpimg.imread('/kaggle/input/skyrim/skyrimhtmltable.bmp')
figure(num=None, figsize=(16, 12), dpi=80, facecolor='w', edgecolor='k')

imgplot = plt.imshow(img)


# So as expected, we're looking for a table. It seems to be the first table on the page so we'll choose that one
# 
# ***
# # Identifying the Layout of the Data
# 
# Now we know what we're looking for, let's select it

# In[ ]:


skills_table = soup.find_all('table')[0]


# We're looking for the rows of this table, so we'll find all the rows in our table.

# In[ ]:


skills_table.find_all('tr')[0:3]


# Let's focus on the first row, our choices of race

# In[ ]:


races_row = skills_table.find_all('tr')[0]
races_row


# This will be the first column in our dataframe, so we want to take the text values from each

# In[ ]:


races_row.text


# This is all one string, which is no good to us, we should split it up to get each race

# In[ ]:


races = races_row.text.split('\n')
races


# Next we want to take out the name of our column (Race)

# In[ ]:


col = races[1][0:4]
col


# And remove the unnecessary entries

# In[ ]:


races = races[2:-1]
races


# Now we've figured out how to take our data from the table, we can work on turning it into a useable dataset
# 
# ***
# # Creating the Dataset
# 
# The easiest way to do this is using a dictionary (hashmap) for each of our columns. The keys will be our column names and the values will be our entries. This will make it very easy for pandas to make the dataframe later on

# In[ ]:


dict2 = {}
dict2[col] = races
dict2


# Now all we have to do is apply this to the rest of the rows in the table

# In[ ]:


for i in range(1,19):
    skill = skills_table.find_all('tr')[i].text.split('\n')
    col = skill[1]
    skill = skill[2:-1]
    dict2[col] = skill


# Pandas can turn our dictionary directly into a dataframe

# In[ ]:


skyrimstats = pd.DataFrame(dict2)
skyrimstats


# And there we go! Easy transition from wiki table to useable dataset. Pandas will also let us cast it to a csv

# In[ ]:


skyrimstats.to_csv("skyrimstats.csv")


# Hope you enjoyed!
