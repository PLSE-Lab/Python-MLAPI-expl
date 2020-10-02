#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# ## Web Scrapping

# # Today we will learn how to do web scrapping. There are many ways of doing it,I have done it using the library bs4.

# In[ ]:


get_ipython().system('pip install requests')


# In[ ]:


get_ipython().system('pip install bs4')


# In[ ]:


import requests
from bs4 import BeautifulSoup


# In[ ]:


# we will analyze the whitehouse url and get the header
result= requests.get("https://www.whitehouse.gov/briefings-statements/") # getting the response for this URL i.e. status code 200
print(result)


# In[ ]:


src = result.content   # To see the html content
print(src)


# In[ ]:


# Parsing the content

soup = BeautifulSoup(src,"html5")  # html5 is a type of parsing technique
print(soup)


# In[ ]:


print(soup.prettify())  # prettify function will show the html content in better way and is more readable


# In[ ]:


# Now we have to get the header and store it in some variable, Below code will do the operation
urls=[]
for h2_tag in soup.find_all("h2"):
    a_tag = h2_tag.find("a")
    urls.append(a_tag.attrs["href"])

print(urls)


# # In the above we can see all the header are extracted and shown line by line. We can store this in some file and can do many operations.

# **Please check my other kernels:**
# 
# **Dealing with Imbalanced Dataset** -- https://www.kaggle.com/pravatkm/dealing-with-imbalanced-dataset
# 
# **Logistic regression on Qualitative_Bankruptcy Data** -- https://www.kaggle.com/pravatkm/logistic-regression-on-qualitative-bankruptcy-data
# 
# **Fashion MNIST - Image Classification - Tensorflow** --  https://www.kaggle.com/pravatkm/fashion-mnist-image-classification-tensorflow
# 
