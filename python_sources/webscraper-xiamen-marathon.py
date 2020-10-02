# A webscraper I built to extract Marathon data from a quite tricky webpage structure.
# Includes additional data cleaning and handling of missing 'rank' values in the 2019 data.
# I just let it run localy because it takes quite some time. I had to build in the time delays because the internet
# connection and loading speed of the wepage has been horrible.


import requests
import lxml.html as lh
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import re 
import time

from selenium import webdriver
from selenium.webdriver.common.keys import Keys

#pull substrings
#isolate strings
#put them in array
#add array to DF
#repeat for whole page
#move to next page
#repeat


df = pd.DataFrame()

#define driver, click buttons
driver = webdriver.Chrome()
driver.get("http://www.xmim.org/homes/searchableresult.html")

year = driver.find_element_by_xpath("//select[@id='year']/option[text()='2017']").click()
elem = driver.find_element_by_xpath("//button[@type='submit']")
elem.send_keys(Keys.ENTER)

n=1
t=1
#<1374
while t<1055:
    
    if (n>1) & ((n-1)%10==0):
        pageskip = driver.find_element_by_xpath("//span[@id='page_entry_next']")
        pageskip.click()
        n=n-10
        time.sleep(8)
    #click on page number
    pageelem = driver.find_element_by_xpath("//span[@id='page_entry_{}']".format(n))
    pageelem.click()
    time.sleep(5)

    assert "No results found." not in driver.page_source
    #define soup target URL
    url = driver.page_source
    #driver.close()

    soup = BeautifulSoup(url,'html.parser')

    #run soup on current page
    results = str(soup.find("div",{'class':"rms-grid-item",'id':"regist_list"}))

    #identify data and clean strings
    found = re.findall(r'>.*?<',results,flags=0)

    for i in found:
        try:
            found.remove('><')
        except:
            break

    m=0
    while m<len(found):
        found[m]=found[m][1:-1]
        m=m+1

        
    #eliminate ''
    found[:] = [x for x in found if x]

    #Seperate by subject and add to DF    
    k=0
    j=0
    while k<(len(found)/10):
        try:
            if found[(j+10)].isdigit()==False:
                found.insert(j+10, 30000)
        except IndexError:
            pass
        item = pd.DataFrame(found[j:(j+11)]).T
        df=df.append(item,ignore_index=True)
        k=k+1
        j=j+11
    print('Turn #{}'.format(t))
    n=n+1
    t=t+1
    time.sleep(3)

driver.close()    
df.columns=['Name','StartNum','Nationality','Sex','Age','NT','Net Time','GT','Gun Time','R','Rank']

df.to_excel("Xiamen_Marathon_2017.xlsx")