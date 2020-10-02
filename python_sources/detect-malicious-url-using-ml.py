#!/usr/bin/env python
# coding: utf-8

# # Detecting Maclicious URLs using Machine Learning<br>
# The malicious urls can be detected using the lexical features along with tokenization of the url strings. I aim to build a basic binary classifier which would help classify the URLs as malicious or benign.

# Steps followed in building the machine learning classifier<br>
# 1. Data Preprocessing / Feature Engineering
# 2. Data Visualization
# 3. Building Machine Learning Models using Lexical Features.
# 4. Building Machine Learning Models using Lexical Features and Tokenization. (Will Update this part)

# Importing The Dependencies

# In[ ]:



import numpy as np 
import pandas as pd 

import matplotlib.pyplot as plt
import seaborn as sns

import os
print(os.listdir("../input"))


# In[ ]:


urldata = pd.read_csv("../input/urldata.csv")


# In[ ]:


urldata.head()


# In[ ]:


#Removing the unnamed columns as it is not necesary.
urldata = urldata.drop('Unnamed: 0',axis=1)


# In[ ]:


urldata.head()


# In[ ]:


urldata.shape


# In[ ]:


urldata.info()


# Checking Missing Values

# In[ ]:


urldata.isnull().sum()


# No missing values in any column.

# ## 1. DATA PREPROCESSING

# The following features will be extracted from the URL for classification. <br>
# <ol>
#     <li>Length Features
#     <ul>
#         <li>Length Of Url</li>
#         <li>Length of Hostname</li>
#         <li>Length Of Path</li>
#         <li>Length Of First Directory</li>
#         <li>Length Of Top Level Domain</li>
#     </ul>
#     </li>
#     <br>
#    <li>Count Features
#     <ul>
#     <li>Count Of  '-'</li>
#     <li>Count Of '@'</li>
#     <li>Count Of '?'</li>
#     <li>Count Of '%'</li>
#     <li>Count Of '.'</li>
#     <li>Count Of '='</li>
#     <li>Count Of 'http'</li>
#     <li>Count Of 'www'</li>
#     <li>Count Of Digits</li>
#     <li>Count Of Letters</li>
#     <li>Count Of Number Of Directories</li>
#     </ul>
#     </li>
#     <br>
#     <li>Binary Features
#     <ul>
#         <li>Use of IP or not</li>
#         <li>Use of Shortening URL or not</li>
#     </ul>
#     </li>
#     
# </ol>
# 
# Apart from the lexical features, we will use TFID - Term Frequency Inverse Document as well.

# ### 1.1 Length Features

# In[ ]:


get_ipython().system('pip install tld')


# In[ ]:


#Importing dependencies
from urllib.parse import urlparse
from tld import get_tld
import os.path


# In[ ]:


#Length of URL
urldata['url_length'] = urldata['url'].apply(lambda i: len(str(i)))


# In[ ]:


#Hostname Length
urldata['hostname_length'] = urldata['url'].apply(lambda i: len(urlparse(i).netloc))


# In[ ]:


#Path Length
urldata['path_length'] = urldata['url'].apply(lambda i: len(urlparse(i).path))


# In[ ]:


#First Directory Length
def fd_length(url):
    urlpath= urlparse(url).path
    try:
        return len(urlpath.split('/')[1])
    except:
        return 0

urldata['fd_length'] = urldata['url'].apply(lambda i: fd_length(i))


# In[ ]:


#Length of Top Level Domain
urldata['tld'] = urldata['url'].apply(lambda i: get_tld(i,fail_silently=True))
def tld_length(tld):
    try:
        return len(tld)
    except:
        return -1

urldata['tld_length'] = urldata['tld'].apply(lambda i: tld_length(i))


# In[ ]:


urldata.head()


# In[ ]:


urldata = urldata.drop("tld",1)


# Dataset after extracting length features

# In[ ]:


urldata.head()


# ### 1.2 Count Features

# In[ ]:


urldata['count-'] = urldata['url'].apply(lambda i: i.count('-'))


# In[ ]:


urldata['count@'] = urldata['url'].apply(lambda i: i.count('@'))


# In[ ]:


urldata['count?'] = urldata['url'].apply(lambda i: i.count('?'))


# In[ ]:


urldata['count%'] = urldata['url'].apply(lambda i: i.count('%'))


# In[ ]:


urldata['count.'] = urldata['url'].apply(lambda i: i.count('.'))


# In[ ]:


urldata['count='] = urldata['url'].apply(lambda i: i.count('='))


# In[ ]:


urldata['count-http'] = urldata['url'].apply(lambda i : i.count('http'))


# In[ ]:


urldata['count-https'] = urldata['url'].apply(lambda i : i.count('https'))


# In[ ]:


urldata['count-www'] = urldata['url'].apply(lambda i: i.count('www'))


# In[ ]:


def digit_count(url):
    digits = 0
    for i in url:
        if i.isnumeric():
            digits = digits + 1
    return digits
urldata['count-digits']= urldata['url'].apply(lambda i: digit_count(i))


# In[ ]:


def letter_count(url):
    letters = 0
    for i in url:
        if i.isalpha():
            letters = letters + 1
    return letters
urldata['count-letters']= urldata['url'].apply(lambda i: letter_count(i))


# In[ ]:


def no_of_dir(url):
    urldir = urlparse(url).path
    return urldir.count('/')
urldata['count_dir'] = urldata['url'].apply(lambda i: no_of_dir(i))


# Data after extracting Count Features

# In[ ]:


urldata.head()


# ### 1.3 Binary Features

# In[ ]:


import re


# In[ ]:


#Use of IP or not in domain
def having_ip_address(url):
    match = re.search(
        '(([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\.([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\.([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\.'
        '([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\/)|'  # IPv4
        '((0x[0-9a-fA-F]{1,2})\\.(0x[0-9a-fA-F]{1,2})\\.(0x[0-9a-fA-F]{1,2})\\.(0x[0-9a-fA-F]{1,2})\\/)' # IPv4 in hexadecimal
        '(?:[a-fA-F0-9]{1,4}:){7}[a-fA-F0-9]{1,4}', url)  # Ipv6
    if match:
        # print match.group()
        return -1
    else:
        # print 'No matching pattern found'
        return 1
urldata['use_of_ip'] = urldata['url'].apply(lambda i: having_ip_address(i))


# In[ ]:


def shortening_service(url):
    match = re.search('bit\.ly|goo\.gl|shorte\.st|go2l\.ink|x\.co|ow\.ly|t\.co|tinyurl|tr\.im|is\.gd|cli\.gs|'
                      'yfrog\.com|migre\.me|ff\.im|tiny\.cc|url4\.eu|twit\.ac|su\.pr|twurl\.nl|snipurl\.com|'
                      'short\.to|BudURL\.com|ping\.fm|post\.ly|Just\.as|bkite\.com|snipr\.com|fic\.kr|loopt\.us|'
                      'doiop\.com|short\.ie|kl\.am|wp\.me|rubyurl\.com|om\.ly|to\.ly|bit\.do|t\.co|lnkd\.in|'
                      'db\.tt|qr\.ae|adf\.ly|goo\.gl|bitly\.com|cur\.lv|tinyurl\.com|ow\.ly|bit\.ly|ity\.im|'
                      'q\.gs|is\.gd|po\.st|bc\.vc|twitthis\.com|u\.to|j\.mp|buzurl\.com|cutt\.us|u\.bb|yourls\.org|'
                      'x\.co|prettylinkpro\.com|scrnch\.me|filoops\.info|vzturl\.com|qr\.net|1url\.com|tweez\.me|v\.gd|'
                      'tr\.im|link\.zip\.net',
                      url)
    if match:
        return -1
    else:
        return 1
urldata['short_url'] = urldata['url'].apply(lambda i: shortening_service(i))


# Data after extracting Binary Features

# In[ ]:


urldata.head()


# # 2. Data Visualization

# In[ ]:


#Heatmap
corrmat = urldata.corr()
f, ax = plt.subplots(figsize=(25,19))
sns.heatmap(corrmat, square=True, annot = True, annot_kws={'size':10})


# In[ ]:


plt.figure(figsize=(15,5))
sns.countplot(x='label',data=urldata)
plt.title("Count Of URLs",fontsize=20)
plt.xlabel("Type Of URLs",fontsize=18)
plt.ylabel("Number Of URLs",fontsize=18)


# In[ ]:


print("Percent Of Malicious URLs:{:.2f} %".format(len(urldata[urldata['label']=='malicious'])/len(urldata['label'])*100))
print("Percent Of Benign URLs:{:.2f} %".format(len(urldata[urldata['label']=='benign'])/len(urldata['label'])*100))


# The data shows a class imbalance to some extent.

# In[ ]:


plt.figure(figsize=(20,5))
plt.hist(urldata['url_length'],bins=50,color='LightBlue')
plt.title("URL-Length",fontsize=20)
plt.xlabel("Url-Length",fontsize=18)
plt.ylabel("Number Of Urls",fontsize=18)
plt.ylim(0,1000)


# In[ ]:


plt.figure(figsize=(20,5))
plt.hist(urldata['hostname_length'],bins=50,color='Lightgreen')
plt.title("Hostname-Length",fontsize=20)
plt.xlabel("Length Of Hostname",fontsize=18)
plt.ylabel("Number Of Urls",fontsize=18)
plt.ylim(0,1000)


# In[ ]:


plt.figure(figsize=(20,5))
plt.hist(urldata['tld_length'],bins=50,color='Lightgreen')
plt.title("TLD-Length",fontsize=20)
plt.xlabel("Length Of TLD",fontsize=18)
plt.ylabel("Number Of Urls",fontsize=18)
plt.ylim(0,1000)


# In[ ]:


plt.figure(figsize=(15,5))
plt.title("Number Of Directories In Url",fontsize=20)
sns.countplot(x='count_dir',data=urldata)
plt.xlabel("Number Of Directories",fontsize=18)
plt.ylabel("Number Of URLs",fontsize=18)


# In[ ]:


plt.figure(figsize=(15,5))
plt.title("Number Of Directories In Url",fontsize=20)
sns.countplot(x='count_dir',data=urldata,hue='label')
plt.xlabel("Number Of Directories",fontsize=18)
plt.ylabel("Number Of URLs",fontsize=18)


# In[ ]:


plt.figure(figsize=(15,5))
plt.title("Use Of IP In Url",fontsize=20)
plt.xlabel("Use Of IP",fontsize=18)

sns.countplot(urldata['use_of_ip'])
plt.ylabel("Number of URLs",fontsize=18)


# In[ ]:


plt.figure(figsize=(15,5))
plt.title("Use Of IP In Url",fontsize=20)
plt.xlabel("Use Of IP",fontsize=18)
plt.ylabel("Number of URLs",fontsize=18)
sns.countplot(urldata['use_of_ip'],hue='label',data=urldata)
plt.ylabel("Number of URLs",fontsize=18)


# In[ ]:


plt.figure(figsize=(15,5))
plt.title("Use Of http In Url",fontsize=20)
plt.xlabel("Use Of IP",fontsize=18)
plt.ylim((0,1000))
sns.countplot(urldata['count-http'])
plt.ylabel("Number of URLs",fontsize=18)


# In[ ]:


plt.figure(figsize=(15,5))
plt.title("Use Of http In Url",fontsize=20)
plt.xlabel("Count Of http",fontsize=18)
plt.ylabel("Number of URLs",fontsize=18)
plt.ylim((0,1000))
sns.countplot(urldata['count-http'],hue='label',data=urldata)
plt.ylabel("Number of URLs",fontsize=18)


# In[ ]:


plt.figure(figsize=(15,5))
plt.title("Use Of http In Url",fontsize=20)
plt.xlabel("Count Of http",fontsize=18)

sns.countplot(urldata['count-http'],hue='label',data=urldata)

plt.ylabel("Number of URLs",fontsize=18)


# In[ ]:


plt.figure(figsize=(15,5))
plt.title("Use Of WWW In URL",fontsize=20)
plt.xlabel("Count Of WWW",fontsize=18)
sns.countplot(urldata['count-www'])
plt.ylim(0,1000)
plt.ylabel("Number Of URLs",fontsize=18)


# In[ ]:


plt.figure(figsize=(15,5))
plt.title("Use Of WWW In URL",fontsize=20)
plt.xlabel("Count Of WWW",fontsize=18)

sns.countplot(urldata['count-www'],hue='label',data=urldata)
plt.ylim(0,1000)
plt.ylabel("Number Of URLs",fontsize=18)


# ## 3. Building Models Using Lexical Features Only

# I will be using three models for my classification.
# <br>1. Logistic Regression
# <br>2. Decision Trees
# <br>3. Random Forest

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix,classification_report,accuracy_score

from sklearn.tree import DecisionTreeClassifier

from sklearn.linear_model import LogisticRegression


# In[ ]:


#Predictor Variables
x = urldata[['hostname_length',
       'path_length', 'fd_length', 'tld_length', 'count-', 'count@', 'count?',
       'count%', 'count.', 'count=', 'count-http','count-https', 'count-www', 'count-digits',
       'count-letters', 'count_dir', 'use_of_ip']]

#Target Variable
y = urldata['result']


# In[ ]:


x.shape


# In[ ]:


y.shape


# In[ ]:


#Splitting the data into Training and Testing
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.3, random_state=42)


# In[ ]:


#Decision Tree
dt_model = DecisionTreeClassifier()
dt_model.fit(x_train,y_train)

dt_predictions = dt_model.predict(x_test)
accuracy_score(y_test,dt_predictions)


# In[ ]:


print(confusion_matrix(y_test,dt_predictions))


# In[ ]:


#Random Forest
rfc = RandomForestClassifier()
rfc.fit(x_train, y_train)

rfc_predictions = rfc.predict(x_test)
accuracy_score(y_test, rfc_predictions)


# In[ ]:


print(confusion_matrix(y_test,rfc_predictions))


# In[ ]:


#Logistic Regression
log_model = LogisticRegression()
log_model.fit(x_train,y_train)

log_predictions = log_model.predict(x_test)
accuracy_score(y_test,log_predictions)


# In[ ]:


print(confusion_matrix(y_test,log_predictions))


# Overall all the models showed great results with decent accuracy and low error rate.<br>
# The high accuracy can be due to the class imbalance situation which is not fixed yet.

# Further Improvements<br>
# 1. Analyse the code and tags used in the webpages.
# 2. Reduce the class imbalance problem.

# In[ ]:




