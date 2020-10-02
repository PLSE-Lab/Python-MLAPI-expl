#!/usr/bin/env python
# coding: utf-8

# # Develop "data intuition" by translating names using Yandex Translate
# This notebooks shows how the names of categories can be translated. This helps developing an intuition about the data and coming up with new features. The Yandex service is easy to set-up and offers a free trier. A first try using the Google translate with the unofficial API hack via the python package https://github.com/ssut/py-googletrans quickly stopped working using the kaggle notebook service.

# ## Getting an API key
# You can get an API key for Yandex for free. https://translate.yandex.com/developers.
# You can save the key in the notebook using the menu Add-ons > Secrets

# In[ ]:


from kaggle_secrets import UserSecretsClient
YANDEX_API_KEY = UserSecretsClient().get_secret("YANDEX_API_KEY")


# ## Translating the names
# Just some basic code to show how the service can be called. It is missing error handling and it does not handle large requests (>10k characters) that should be split into multiple requests.

# In[ ]:


import requests
def translate(x, key, src='ru', dest='en'):
    original = x.unique()
    url = 'https://translate.yandex.net/api/v1.5/tr.json/translate'
    params = dict(
        key=key,
        lang=src+'-'+dest
    )
    payload = {'text': original}
    response = requests.post(url=url, params=params, data=payload)
    translated_text = response.json()['text']
    dictionary = dict(zip(original, translated_text))
    return([dictionary.get(item, item) for item in x])


# In[ ]:


import pandas as pd
categories = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/item_categories.csv')


# In[ ]:


categories['item_category_name_en'] = translate(categories['item_category_name'], YANDEX_API_KEY)


# In[ ]:


categories.sample(10)


# In[ ]:




