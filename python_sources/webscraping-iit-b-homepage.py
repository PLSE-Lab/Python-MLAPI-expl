#!/usr/bin/env python
# coding: utf-8

# # **Import packages**

# In[ ]:


from urllib.request import urlopen
from bs4 import BeautifulSoup
import re


# # Open the URL of the website to be scraped

# In[ ]:


html = urlopen('http://www.iitb.ac.in')
#html.read()


# # Beautify the contents of website and display

# In[ ]:


bs = BeautifulSoup(html.read(), 'html.parser')
#print(bs)


# # Display the headers

# In[ ]:


print(bs.h1)
print(bs.h2)
print(bs.h3)
print(bs.h4)
print(bs.h5)


# # Display titles News & Events section

# In[ ]:


nameList = bs.findAll('div', {'class': 'views-field views-field-title'})
for name in nameList:
    print(name.get_text())


# # Display details in the News section

# In[ ]:


nameList = bs.findAll('div', {'class': 'views-field views-field-body'})
for name in nameList:
    print(name.get_text())


# # Display "Research Highlights" section

# In[ ]:


nameList = bs.findAll('div', {'class': 'work-item-content'})
for name in nameList:
    print(name.get_text())


# # Display "Footer" section

# In[ ]:


nameList = bs.findAll('li', {'class': 'leaf'})
for name in nameList:
    print(name.get_text())


# # Display Menu items and Menu listing

# In[ ]:


nameList = bs.findAll('ul', {'class': 'menu'})
for name in nameList:
    print(name.get_text())


# # Display all the URL links on the homepage

# In[ ]:


for link in bs.findAll('a', attrs={'href': re.compile("^http://")}):
    print(link.get('href'))

