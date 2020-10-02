#!/usr/bin/env python
# coding: utf-8

# # Introduction
# 
# This notebook shows how to use the [Google's Translation API](https://cloud.google.com/translate) to translate review texts from a popular dataset from English to German. To do so, we're
# 
# - Using a user-defined secret to store a service account credential within Kaggle (and provide it to the API from within Kernels). If you're forking this Kernel, you have to provide your own
# - Install the necessary packages
# - Initialize the API client and wrap the call in a function
# - Read the input CSV
# - Row by row, call the Translation API and populate a new column to our data
# - Write down the results data
# 
# (this naively calls the Translation API row by row. For better performance, use it in [batch mode](https://cloud.google.com/translate/docs/advanced/batch-translation))
# 

# In[ ]:


# Install packages
get_ipython().system('pip install --upgrade google-cloud-translate==2.0.0')
get_ipython().system('pip install --upgrade google-auth')


# In[ ]:


# Handle credentials
import json
from google.oauth2 import service_account
from kaggle_secrets import UserSecretsClient

user_secrets = UserSecretsClient()
secret_value = user_secrets.get_secret("translation-playground")

service_account_info = json.loads(secret_value)
credentials = service_account.Credentials.from_service_account_info(
    service_account_info)


# In[ ]:


# Setup client & translation function
from google.cloud import translate_v2 as translate

translate_client = translate.Client(credentials=credentials)

def translate(text, target_lang, source_lang="en"):
    try:    
        result = translate_client.translate(text, target_language=target_lang, source_language=source_lang)
        return result['translatedText']
    except:
        return ""


# In[ ]:


# Test it
print(translate("This is a very nice text to translate", "de"))


# In[ ]:


import pandas as pd

data = pd.read_csv("../input/womens-ecommerce-clothing-reviews/Womens Clothing E-Commerce Reviews.csv")
data.head()

#data = data[:10]


# In[ ]:


data['Review Text DE'] = data.apply(lambda row: translate(row['Review Text'], "de"), axis = 1)

data.head()


# In[ ]:


data.to_csv("/kaggle/working/Reviews_DE.csv");

