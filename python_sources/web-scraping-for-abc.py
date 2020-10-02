#!/usr/bin/env python
# coding: utf-8

# # Import & Install

# In[1]:


get_ipython().system('pip install requests beautifulsoup4 fake-useragent --quiet')

import os
import time
from bs4 import BeautifulSoup
import requests
from urllib.request import Request, urlopen
from fake_useragent import UserAgent
import random

print(os.listdir('./'))


# # Fake Proxies

# In[2]:


ua = UserAgent() 

def get_proxies():
    req = Request('https://www.sslproxies.org/')
    req.add_header('User-Agent', ua.random)
    doc = urlopen(req).read().decode('utf8')
    soup = BeautifulSoup(doc, 'html.parser')
    table = soup.find(id='proxylisttable')
    proxies = []
    for row in table.tbody.find_all('tr'):
        proxies.append({
            'protocol': 'https' if row.find_all('td')[6].string == 'yes' else 'http',
            'ip': row.find_all('td')[0].string, 
            'port': row.find_all('td')[1].string 
        })
    return proxies


# # Detail of website
# Number of all pages is 1380 and each page has 200 files. <br />
# This site can't request under <b>1</b> second. (Fake proxy and UA are not worked.)

# ## <span style="color:red">To get all data due to limit of resources. You need to edit following variables.</span>
# - <b>start_page</b> - first page is 0
# - <b>last_page</b> - not included (last page that gets request is <b>last_pages - 1</b>)

# In[3]:


base_url = 'http://abcnotation.com'

# Edit these values
start_page = 200
last_page = 300


# In[ ]:


for i in range(start_page, last_page):
    cur_page = str(i)
    while len(cur_page) != 4:
        cur_page = '0' + cur_page
    page_link = base_url + '/browseTunes?n=' + cur_page
    page_response = requests.get(page_link)
    page_soup = BeautifulSoup(page_response.text, 'html.parser')
    pre_tag = page_soup.find('pre')
    a_tag = pre_tag.find_all('a', href=True)
    
    print('Page:', i)
    file_number = 0
    proxies = get_proxies()
    prev_proxy_index = 0
    
    for link in a_tag:
        file_link = base_url + link['href']
        
#         Fake Proxy & UA => Can't break web security.
        cur_proxy_index = random.randint(0, len(proxies) - 1)
        while cur_proxy_index == prev_proxy_index:
            cur_proxy_index = random.randint(0, len(proxies) - 1)
        proxy = {proxies[cur_proxy_index]['protocol']: 'http://' + proxies[cur_proxy_index]['ip'] + ':' + proxies[cur_proxy_index]['port']}
        prev_proxy_index = cur_proxy_index
        headers = requests.utils.default_headers()
        headers.update({'User-Agent': ua.random})
        file_response = requests.get(file_link, proxies=proxy, headers=headers)

        if file_response.status_code == 200:
            file_soup = BeautifulSoup(file_response.text, 'html.parser')
    #         print(file_soup.prettify())
            text_area = file_soup.find('textarea')
    #         print(text_area.contents[0])
            with open('./scraping_' + str(i) + '_' + str(file_number) + '.abc', 'a') as file:
                file.write(text_area.contents[0])
#                 print('scraping_' + str(i) + '_' + str(file_number) + '.abc created') 
        else:
            print('Failed at page ' + str(i) + ' file ' + str(file_number))
        file_number+=1
        time.sleep(1)


# # Avoid limit of output files

# In[ ]:


get_ipython().system('tar -zcf scraping.tar.gz ./')
get_ipython().system('rm -rf ./scraping_*')
# !tar -zxf scraping.tar.gz # For extraction
get_ipython().system('ls ./')


# In[ ]:




