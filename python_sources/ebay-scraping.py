# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


from requests import get
from bs4 import BeautifulSoup
import re
import pandas as pd
from time import sleep, time
from random import randint
from IPython.core.display import  clear_output
from warnings import warn

title =[]
time_sold = []
condition = []
#platform =[]
price = []
bid = []
shipping =[]


#prepare for change pages
pages = [str(i) for i in range(1,16)]
url = 'https://www.ebay.com/sch/i.html?_from=R40&_nkw=red+dead+redemption+2&_sacat=0&LH_TitleDesc=0%7C0&LH_Auction=1&LH_Complete=1&LH_Sold=1&_ipg=200&_pgn=1'

# prepare monitoring of the loop
start_time = time()
requests = 0
# for every page in the interval 1-15
for page in pages:
    response = get(url)

    #pause the loop
    sleep(randint(8,15))

    # monitor the requests
    requests +=1
    elapsed_time = time() - start_time
    print('Request:{}; Frequency: {} requests/s'.format(requests, requests/elapsed_time))
    clear_output(wait = True)
        
# Throw a warning for non-200 status codes
    if response.status_code != 200: 
        warn('Request: {}; Status code: {}'.format(requests, response.status_code))

# Break the loop if the number of requests is greater than expected
    if requests > 16: 
        warn('Number of requests was greater than expected.')
        break

#create beautifulSoup project
    html_soup = BeautifulSoup(response.text, 'html.parser')

#get containers

    item_containers = html_soup.find_all('li', class_ = 's-item')
    items=html_soup.find('li', class_ = 's-item')




    for items in item_containers:
    
        item_title=items.h3.div.next_sibling
        title.append(item_title)

        item_time = items.div.find(class_="s-item__title-tag").text
        item_time=re.split('\\bSold\\b',item_time)[-1]
        item_time=item_time.strip()
        time_sold.append(item_time)

        item_condition=items.find("span",{"class":"SECONDARY_INFO"}).get_text()
        condition.append(item_condition)
    
#    item_platform=items.find("span",{"class":"SECONDARY_INFO"}).next_sibling.next_sibling
#    platform.append(item_platform)
    
        item_price=items.find('span', class_='s-item__price').text
        item_price =re.findall("\d+\.\d+", item_price)
        item_price="".join(item_price)
        price.append(item_price)

        item_bid=items.find('span', class_='s-item__bidCount').text.split()[0]
        #item_bid=re.split('\s',item_bid)[0]
        bid.append(item_bid)
        
        
        item_shipping = items.find("span", {"class": "s-item__shipping"}).get_text()
        item_shipping =re.findall("\d+\.\d+", item_shipping)
#item_shipping = re.findall("\d.*?(?=\s)", item_shipping)
        item_shipping="".join(item_shipping)
#item_shipping = re.split('\s',item_shipping)[0]
        if item_shipping is '':
            item_shipping='0'
        shipping.append(item_shipping)


df=pd.DataFrame({'title':title, 'time':time_sold, 'condition':condition, 'price':price, 'bid':bid, 'shipping':shipping})

df.to_csv('ebay_scraping.csv')