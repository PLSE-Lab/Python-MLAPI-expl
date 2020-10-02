#!/usr/bin/env python
# coding: utf-8

# # Data extraction
#  - It is one of the most important task data scientist has to perform. In this notebook I will be extracting data from API. 
#  - Data extraction through API is not working in the kaggle kernal however working absolutely fine in the local jupyter notebook. I could not find the root cause yet. I will update the kernal further once I will figure it out. 
#  - Please share your comments and other codes in the comment for other types of data extraction. 

# ## Data Extraction from API
# Following are the steps to extract data from APIs
#   - Import package --- import requests
#   - Use get function to make GET request --- result = requests.get(url)
#   - Status Code --- result.status_code
#   - Header information --- result.headers
#   - Response text --- result.text
#   - Response encoding --- result.encoding

# **Test data:**
# Following is the step to get the api details data.gov. You can use any other website for the source.
#  - Step 1: Go to the https://collegescorecard.ed.gov/data/documentation/
#  - Step 2: Click on the link on page https://api.data.gov/signup.
#  - Step 3: Signup to get your key. Check emails to get the key. API Key: 7g1Yt0xxEVGxrd8hovbQunznrPE4oxyJivQsoL4h
#  - Step 4. prepare the url  https://api.data.gov/ed/collegescorecard/v1/schools?api_key=7g1Yt0xxEVGxrd8hovbQunznrPE4oxyJivQsoL4h
# 

# In[ ]:


#import requests module
import requests


# In[ ]:


# store the URI in the url variable.
url = 'https://api.data.gov/ed/collegescorecard/v1/schools?api_key=7g1Yt0xxEVGxrd8hovbQunznrPE4oxyJivQsoL4h'


# *** Below code is not working in Kaggle however it is working absolutely fine in my local machine. Please refer the screen shots. Also see the subsequent commands for extracting the data in text and json format. You can run in your local.***
# 
# ![image.png](attachment:image.png)

# In[ ]:


#using get command, returns response object
result = requests.get(url)


# In[ ]:


# Get the status code
result.status_code


# In[ ]:


#Get the complete results in the text format.
result.text


# ## Web Scraping
# Here we will creat an html page and extract data from headers, paragraphs and table. This is just example. 
# In the real world we have to take necessury permission to perform webscrapping for any website. 

# In[ ]:


# Improt the module to read html code and capture the data.
from bs4 import BeautifulSoup


# In[ ]:


#HTML Created. Please refer the result in the next command for exact page. 
html_string = """
<!doctyp html>
<html lang="en">
<head>
    <title>Doing Data Science With Python</title>
</head>
<body>
    <h1 style="color:#F15B2A;">Data Extraction: WebScrapping</h1>
    <p id="author">Author : Neeraj Sharma</p>
    <p id="description">This notebook will help you to learn webscraping.</p>
    
    <h3 style="color:#404040;">Where does Data Scientist Spends their time?</h3>
    <table id="workdistribution" style="width:100%">
        <tr>
            <th>Work</th>
            <th>% of time</th>
        </tr>
        <tr>
            <td>Data Extraction</td>
            <td>20</td>
        </tr>
          <tr>
            <td>Data Organize</td>
            <td>60</td>
        </tr>
          <tr>
            <td>Building Model and Evaluation</td>
            <td>10</td>
        </tr>
          <tr>
            <td>Presentation and Other tasks</td>
            <td>10</td>
        </tr>
    </table>
    </body>
    </html>
"""


# In[ ]:


#Display HTML page in the juyper notedbook
from IPython.core.display import display, HTML
display(HTML(html_string))


# In[ ]:


#use beautiful soup to read html string and create an object
ps = BeautifulSoup(html_string)


# In[ ]:


# print the value of ps
print(ps)


# In[ ]:


# print the body from html
body = ps.find(name="body")
print(body)


# In[ ]:


#use text attribute to get the content of the tag
print(body.find(name="h1").text)


# In[ ]:


#print the value of <p> tag
print(body.find(name='p'))


# In[ ]:


#print the value of all paragraphs tag
print(body.findAll(name='p'))


# In[ ]:


#loop through  each element in <p> tag and print them one by one
for p in body.findAll(name='p'):
    print(p.text)


# In[ ]:


# add attributes author also in selection process
print(body.findAll(name='p', attrs={"id":"author"}))


# In[ ]:


#print attributes description along with paragraph
print(body.findAll(name='p', attrs={"id":"description"}))


# In[ ]:


# Read and print columns of the table. this can be later stored in the variable and create df to work further. 
#body
body = ps.find(name='body')
#module table 
module_table = body.find(name='table', attrs={"id":"workdistribution"})
#iterate through each row in the table (skipping the first row)
for row in module_table.findAll(name='tr')[1:]:
    #module title
    title = row.findAll(name='td')[0].text
    #module duration
    duration = int(row.findAll(name='td')[1].text)
    print(title, duration)

