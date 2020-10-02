#!/usr/bin/env python
# coding: utf-8

# # <font color='red'> Sample Pre-processing of Web Content for Dataset Preparation </font>

# ## Initialisation Code

# In[ ]:


# Basic Libraries to be installed before moving ahead
get_ipython().system('pip install pysafebrowsing')
get_ipython().system('pip install tld')
get_ipython().system('pip install whois')
get_ipython().system('pip install geoip2')


# In[ ]:


# Basic Initialisation
import time
import pandas as pd
import numpy as np
pd.set_option('mode.chained_assignment', None) #Switch off warning


# In[ ]:


#Verifying pathname of dataset before loading
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename));
        print(os.listdir("../input"))


# ## Loading the Sample Web Content Crawled & Collected by MalCrawler

# In[ ]:


# Loading Dataset containing Raw Web Content, URL and IP Address (Output of MalCrawler)
def loadDataset():
    df = pd.read_csv("/kaggle/input/preprocessingsampledata/PreprocessingSampleData.csv")
    return df

df = loadDataset()
df = df[['url','ip_add', 'content']] # The three Columns of the initial data
df


# In[ ]:


#Adding new blank columns to the dataframe df
df['geo_loc']=""
df['url_len']=""
df['js_len']=""
df['js_obf_len']=""
df['tld']=""
df['who_is']=""
df['https']=""
df['label']=""
df = df[['url','ip_add','geo_loc','url_len','js_len','js_obf_len','tld','who_is','https','content','label']]
#df


# ## Computing the 'geo_loc' Attribute from IP Address

# In[ ]:


# Filling the 'geo_loc' column of dataframe 
import os
import geoip2.database
import socket
import time

reader = geoip2.database.Reader('/kaggle/input/geoipdatabase/GeoLite2-Country.mmdb')

for x in df.index:
    try:
        ip_add = str(df['ip_add'][x])
        response = reader.country(ip_add)
        df['geo_loc'][x] = response.country.name
        #print(x, "Finished,value is:",response.country.name)   
    except Exception as msg:
        df['geo_loc'][x] = ""
        #print(x," Finished with Error Msg:",msg)

reader.close()
#df


# ## Computing 'url_len

# In[ ]:


#Generating 'url_len' from 'url'
df['url_len'] = df['url'].str.len()
#df


# ## Computing 'js_len'

# In[ ]:


import re       #importing regex for string selection and parsing

def get_js_len_inKB(content): #Function for computing 'js_len from Web Content
    js=re.findall(r'<script>(.*?)</script>',content)
    complete_js=''.join(js)
    js_len = len(content.encode('utf-8'))/1000
    return js_len
for x in df.index: #Computing and Putting 'js_len' in Pandas Dataframe
    df['js_len'][x] = get_js_len_inKB(df['content'][x])

#df


# ## Computing 'js_obf_len'

# In[ ]:


# Computed using Selenium Emulator, thus will have to be run separately and then added
# Code given in https://github.com/lucianogiuseppe/JS-Auto-DeObfuscator/blob/master/jsado.py


# ## Computing 'tld' Attribute

# In[ ]:


#Filling up TLD column
from tld import get_tld

for x in df.index:       
    try:
        u = df.url[x]
        s = get_tld(str(u), fix_protocol=True)
        df['tld'][x] = s
    except:
        pass
#df


# ## Computing 'who_is' Attribute

# In[ ]:


#Whois processing
import whois
start_time = time.time()

for x in df.index:  
    try:    
        domain = whois.query(df['url'][x])
        #print(domain.registrar)
        if len(str(domain.registrar)) >1 :
            df['who_is'][x]= 'complete'
        else:
            df['who_is'][x]= 'incomplete'
    except Exception as msg:
        #print(x,", Error: ",msg)
        df['who_is'][x]= 'incomplete'
    #print(x,df['who_is'][x])

print("***Total Time taken --- %s seconds ---***" % (time.time() - start_time))
#df


# In[ ]:


# Alternate Code for Computing using WHOIS API
from urllib.request  import  urlopen       # Importing url library
import  json                               # Importing the JSON Module

url =  'https://www.bits-pilani.ac.in'  #A sample URL
apiKey = 'at_YC7W9LM2w1lQOCMmN0KUe3OU7B8Jc'
url = 'https://www.whoisxmlapi.com/whoisserver/WhoisService?'    + 'domainName=' + url + '&apiKey=' + apiKey + "&outputFormat=JSON"

whois_data= urlopen(url).read().decode('utf8') #WHO IS info returned by API
data=json.loads(whois_data) # Converting it from JSON to a Python Dict Object 
#if data['registrarName']=="":
    #who_is = 'incomplete'
#else:
    #who_is = 'complete'
  
# Sample of one URL is shown here
# Similarly, who_is data is checked for all URLs in the dataset


# ## Computing the 'https' Attribute

# In[ ]:


# Filling the column https_status
import http.client

start_time = time.time()

for x in df.index:
    https_status= False
    try:
        conn = http.client.HTTPSConnection(df['url'][x])
        conn.request("HEAD", "/")
        res = conn.getresponse()
        if res.status == 200 or res.status==301 or res.status==302:
            https_status= True   
        #print(x,res.status,res.reason,https_status)
    except Exception as msg:
        df['https'][x]= 'no'
        #print(x,"Error: ",msg)
    finally:
        df['https'][x]= https_status
        #conn.close

print("***Total Time taken --- %s seconds ---***" % (time.time() - start_time))
#df


# ## Allocation of Class Label 

# In[ ]:


# Filling the label of training set from Google Safe Browising API
from pysafebrowsing import SafeBrowsing
KEY= "AIzaSyABO6DPGmHpCs8U5ii1Efkp1dUPJHQfGpo"

start_time = time.time()
s = SafeBrowsing(KEY)

for x in df.index:
    
    try:
        url = df['url'][x]
        r = s.lookup_urls([url])
        label=r[url]['malicious']    
        df['label']=label
        #print(x, label)
    except Exception as msg:
        df['label']=""
        #print(x,"Error: ",msg)

print("***Total Time taken --- %s seconds ---***" % (time.time() - start_time))

#df


# ## Saving of Processed Data

# In[ ]:


# Saving the file
#df.to_csv("Datasets/processed_webdata_sample.csv")

