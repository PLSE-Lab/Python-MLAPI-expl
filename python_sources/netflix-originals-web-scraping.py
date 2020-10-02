#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# import the library we use to open URLs
import urllib.request


# In[ ]:


# specify which URL/web page we are going to be scraping
url = "https://en.wikipedia.org/wiki/List_of_original_programs_distributed_by_Netflix"


# In[ ]:


# open the url using urllib.request and put the HTML into the page variable
page = urllib.request.urlopen(url)


# In[ ]:


# import the BeautifulSoup library so we can parse HTML and XML documents
from bs4 import BeautifulSoup


# In[ ]:


# parse the HTML from our URL into the BeautifulSoup parse tree format
soup = BeautifulSoup(page, "lxml")


# In[ ]:


# use the 'find_all' function to bring back all instances of the 'table' tag in the HTML and store in 'all_tables' variable
all_tables=soup.find_all("table",class_='wikitable sortable')
all_tables


# In[ ]:


len(all_tables)


# In[ ]:


Title=[]
Genre=[]
Premiere=[]
Seasons=[]
Length=[]
Status=[]


# In[ ]:


for table in range(len(all_tables)):
    for row in all_tables[table].find_all("tr"):
        cells=row.findAll('td')
        if len(cells)==6:
            Title.append(cells[0].find(text=True))
            Genre.append(cells[1].find(text=True))
            Premiere.append(cells[2].find(text=True))
            Seasons.append(cells[3].find(text=True))
            Length.append(cells[4].find(text=True))
            Status.append(cells[5].find(text=True))
			


# In[ ]:


import pandas as pd

df=pd.DataFrame(Title,columns=['Title'])
df['Genre']=Genre
df['Original Network'] ='Netflix'
df['Premiere']=Premiere
df['Seasons']=Seasons
df['Length']=Length
df['Netflix Exclusive Regions']='Worldwide'
df['Status']=Status


# In[ ]:


df


# In[ ]:


Title=[]
Genre=[]
Original_Network=[]
Premiere=[]
Seasons=[]
Length=[]
Netflix_Exclusive_Regions=[]
Status=[]


# In[ ]:


for table in range(len(all_tables)):
    for row in all_tables[table].find_all("tr"):
        cells=row.findAll('td')
        if len(cells)==8:
            Title.append(cells[0].find(text=True))
            Genre.append(cells[1].find(text=True))
            Original_Network.append(cells[2].find(text=True))
            Premiere.append(cells[3].find(text=True))
            Seasons.append(cells[4].find(text=True))
            Length.append(cells[5].find(text=True))
            Netflix_Exclusive_Regions.append(cells[6].find(text=True))
            Status.append(cells[7].find(text=True))


# In[ ]:



df2=pd.DataFrame(Title,columns=['Title'])
df2['Genre']=Genre
df2['Original Network']=Original_Network
df2['Premiere']=Premiere
df2['Seasons']=Seasons
df2['Length']=Length
df2['Netflix Exclusive Regions']=Netflix_Exclusive_Regions
df2['Status']=Status


# In[ ]:


df2.head(27)


# In[ ]:


netflix_originals = pd.concat([df,df2.head(27)])
netflix_originals


# In[ ]:


netflix_originals.to_csv ('netflix_originals.csv', index = False, header=True)

