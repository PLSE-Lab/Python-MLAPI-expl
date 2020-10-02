#!/usr/bin/env python
# coding: utf-8

# # Web scraping with python to .csv file
# 
# Original Question: https://www.kaggle.com/questions-and-answers/147422
# 
# 
# > I couldn't run this code, however I could run it in Jupyter Notebook. It gave me an error, and a csv file with only the headers.
# 
# ```
# filename = 'Cities.csv'
# headers = 'City,Latitude,Longitude\n'
# f = open(filename, "w")
# f.write(headers)
# for j in range(1,9):    
#    page_url = f'https://www.latlong.net/category/cities-102-15-{j}.html'
#    uClient = uReq(page_url)
#    page_soup = soup(uClient.read(), "html.parser")
#    uClient.close()
#    rows = page_soup.findAll('tr')
#    rows = rows[1:]
# 
#    for row in rows:
#        cell = row.findAll('td')
#        City = cell[0].text
#        Latitude = cell[1].text
#        Longitude = cell[2].text
#        f.write(City.replace(',', '|') + ',' + Latitude + ',' + Longitude + '\n')
# f.close()
# ```
# 
# > Thanks for any advice!

# Your code works for me. 
# - I think you were just missing a couple of import statements. 
# - I also tweaked a `City.replace(',', '|')` with `re.sub(r',\s*', '|',City)` to remove the whitespace between |'s in the output

# In[ ]:


from urllib.request import urlopen as uReq
from bs4 import BeautifulSoup as soup
import re

filename = 'Cities.csv'
headers  = 'City,Latitude,Longitude\n'
f = open(filename, "w")
f.write(headers)
for j in range(1,9):    
    page_url  = f'https://www.latlong.net/category/cities-102-15-{j}.html'
    uClient   = uReq(page_url)
    page_soup = soup(uClient.read(), "html.parser")
    uClient.close()
    rows = page_soup.findAll('tr')
    rows = rows[1:]
    for row in rows:
        cell      = row.findAll('td')
        City      = cell[0].text
        Latitude  = cell[1].text
        Longitude = cell[2].text
        f.write(re.sub(r',\s*', '|',City) + ',' + Latitude + ',' + Longitude + '\n')
f.close()


# We can check that this outputs correctly with a cat

# In[ ]:


get_ipython().system('cat Cities.csv')


# Here is the raw text of the data being scraped

# In[ ]:


get_ipython().system('curl https://www.latlong.net/category/cities-102-15-1.html')

