#!/usr/bin/env python
# coding: utf-8

# **Webscraping and Analyzing data
# For that purpose we are going to use SLC result data**
# 
# **Steps we are going to follow**
# 
# * Query the website
# * Extract the required data
# * Preprocess the data
# * Data Visualization

# **(1) Query the website**

# In[ ]:


#Import pandas to convert list to data frame

import pandas as pd
import numpy as np
import requests
from pandas_profiling import ProfileReport

import urllib.request

#import the beatiful soup functions to parse the data
from bs4 import BeautifulSoup


# In[ ]:


URL = "http://gbsnote.com/slc-result-history-nepal/"
#query the website
page = requests.get(url = URL)

#parse the html and store in Beautiful soup format
soup = BeautifulSoup(page.text)


# **See HTML page nested structure
# print(soup.prettify())**

# In[ ]:


soup.title.string


# **(2) Extract the required data**

# In[ ]:


#find all links

all_links = soup.find_all("a")
for link in all_links:
    pass
    #print(link.get("href"))


# In[ ]:


#find all tables
all_tables = soup.find('table')
#print(all_tables)


# In[ ]:


#Generate lists

A = []
B = []
C = []
D = []

for row in all_tables.findAll("tr"):
    cells = row.findAll('td')
    
    #Only extract table body
    if(len(cells) == 4):
        A.append(cells[0].find(text = True))
        B.append(cells[1].find(text = True))
        C.append(cells[2].find(text = True))
        D.append(cells[3].find(text = True))


# In[ ]:


df = pd.DataFrame()

df['Year(BS)'] = A
df['Total Appeared'] = B
df['Total Passed'] = C
df['Pass Percentage'] = D


# In[ ]:


truedf = df[1:]
truedf['Pass Percentage'] = truedf['Pass Percentage'].str.replace('%', '')
truedf['Year(BS)'] = truedf['Year(BS)'].str.replace('\n', '0')


# In[ ]:


truedf.to_csv('slc.csv', encoding='utf-8', index=False)


# **(3) Preprocess the data**
# 
# **Data Analyzing starts**

# In[ ]:


#Read in data
features = pd.read_csv('slc.csv')


# In[ ]:


type(features['Year(BS)'])


# In[ ]:


features.mean()


# In[ ]:


#Replacing zero value of year
past = 0
value = 0
for value in (features['Year(BS)']):
    #print(past)
    if(value != 0):
        past = value
    else:
        nextval = past+1
        features.replace(value,nextval)
        break
print(nextval)


# In[ ]:


finalfea = features.replace(value,nextval)
finalfea.to_csv('slcfinal.csv', encoding='utf-8', index=False)


# **(4) Data Visualization**

# In[ ]:


finalfea.head()


# In[ ]:


profile = ProfileReport(finalfea, title='Pandas Profiling Report', html={'style':{'full_width':True}})


# In[ ]:


profile.to_notebook_iframe()


# In[ ]:


finalfea.describe()


# In[ ]:


from plotly.offline import iplot, init_notebook_mode
import cufflinks as cf
cf.go_offline()
print( cf.__version__)


# **BarPlot Year(BS), Total Appeared and TotalPassed**

# In[ ]:


finalfea_process = finalfea[['Year(BS)','Total Appeared', 'Total Passed']].set_index('Year(BS)')
finalfea.iplot(x='Year(BS)', y=['Total Appeared', 'Total Passed'], kind='bar',xTitle='Year(BS)')


# In[ ]:


import matplotlib.pyplot as plt


# **Plot Year(BS) and Pass Percentage**

# In[ ]:


finalfea.iplot(x= 'Year(BS)',y = 'Pass Percentage',xTitle='Year(BS)', yTitle='Pass Percentage')


# **Plot Year(BS) and Total Appeared**

# In[ ]:


finalfea.iplot(x= 'Year(BS)',y = 'Total Appeared',xTitle='Year(BS)', yTitle='Total Appeared')


# **Plot Year(BS) and Total Passed**

# In[ ]:


finalfea.iplot(x= 'Year(BS)',y = 'Total Passed',xTitle='Year(BS)', yTitle='Total Passed')


# **Year VS Total Appeared and Total Passed**

# In[ ]:


finalfea.iplot(x= 'Year(BS)',y = ['Total Appeared','Total Passed'],xTitle='Year(BS)')

