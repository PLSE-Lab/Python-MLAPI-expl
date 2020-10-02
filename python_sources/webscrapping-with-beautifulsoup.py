#!/usr/bin/env python
# coding: utf-8

# In[ ]:


Web Scrapping Using Beautifulsoup & requests


# In[ ]:


mport requests
from bs4 import BeautifulSoup
import pandas as pd


# In[ ]:


address = requests.get("http://www.ipaidabribe.com/reports/all#gsc.tab=0")
soup = BeautifulSoup(address.content, "html.parser")


# In[ ]:


bribes = soup.find_all('section', {'class': 'ref-module-paid-bribe'})


# In[ ]:


title_list = []
address_list = []
category_list = []
price_list = []
date_list = []


# In[ ]:


for bribe in bribes:
    title = bribe.find('h3').find('a')['title']
    title_list.append(title)


# In[ ]:


address = bribe.find('div').find('a')['title']
address_list.append(address)


# In[ ]:


category = bribe.find('ul', {'class': 'department clearfix'}).find('li', {'class': 'transaction'}).find('a')['title']
category_list.append(category)


# In[ ]:


price = bribe.find('li', {'class': 'paid-amount'}).find('span')
price = price.text
price = price.strip('')
price_list.append(price)


# In[ ]:


date = bribe.find('div', {'class': 'key'}).find('span', {'class': 'date'})
date = date.text
date_list.append(date)


# In[ ]:


dictionary = {
    'Title': title_list,
    'Address': address_list,
    'Category': category_list,
    'Price': price_list,
    'Date': date_list
}


# In[ ]:


data = pd.DataFrame(dictionary)
data.to_csv('Bribe.csv')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




