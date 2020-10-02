#!/usr/bin/env python
# coding: utf-8

# Hi guys!
# 
# This kernel is about how I find almost all devices from the datasets.
# 
# I have used yandex-search for this task.
# // Yandex allows 10k searches per day when registered with a validated (international) mobile number.
# 
# To repeat this code you should read https://pypi.org/project/yandex-search/ and have a valid account on yandex.
# 
# But if you don't I have just leave my scraped data below.
# 
# Have a fun!
# 

# In[ ]:


# !pip install yandex_search


# In[ ]:


# import yandex_search
import requests
from bs4 import BeautifulSoup

import datetime
from joblib import Parallel, delayed
import joblib

import getpass
import pandas as pd
import numpy as np
import threading

import matplotlib.pylab as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# Start with using yandex-search:

# In[ ]:


# api_user = getpass.getpass()


# In[ ]:


# api_key = getpass.getpass()


# In[ ]:


# yandex = yandex_search.Yandex(api_user=api_user, api_key=api_key)


# In[ ]:


# yandex.search('"ale-l23" android').items


# In[ ]:


def search_first_url_on_site(yandex, query, site):
    try:
        search_results = yandex.search("site:{} {}".format(site, query))
        if search_results is None or len(search_results.items) == 0:
            return link
    
        return search_results.items[0]['url']
    except KeyboardInterrupt:
        raise
    except:
        pass
    return None

# search_first_url_on_site(yandex, 'p5006a', 'www.handsetdetection.com')


# Let's use it for device-info:

# In[ ]:


def read_data():
    train_identity = pd.read_csv('/kaggle/input/ieee-fraud-detection/train_identity.csv')
    test_identity = pd.read_csv('/kaggle/input/ieee-fraud-detection/test_identity.csv')
    train_transaction = pd.read_csv('/kaggle/input/ieee-fraud-detection/train_transaction.csv')
    test_transaction = pd.read_csv('/kaggle/input/ieee-fraud-detection/test_transaction.csv')

    train = train_transaction.merge(train_identity, how='left', left_index=True, right_index=True)
    del train_transaction, train_identity

    y_train = train['isFraud'].astype("uint8").copy()
    train = train.drop('isFraud', axis=1)
    
    test = test_transaction.merge(test_identity, how='left', left_index=True, right_index=True)
    del test_transaction, test_identity

    return train, test, y_train


# In[ ]:


train, test, y_train = read_data()
common_df = pd.concat([train, test])

target = np.concatenate([y_train, np.full(test.shape[0], -1)])
common_df['target'] = target


# In[ ]:


common_df['DeviceInfo'] = common_df['DeviceInfo'].astype('str')
common_df['ClearDeviceInfo'] = np.vectorize(lambda x : x.lower().split('build')[0].strip())(common_df['DeviceInfo'])
common_df['ClearDeviceInfo'] = common_df['ClearDeviceInfo'].astype('str')


# In[ ]:


common_df['ClearDeviceInfo'].value_counts().nlargest(30)


# In[ ]:


incomplete_devices = ['Windows', 'iOS Device', 'MacOS', 'SAMSUNG', 'Trident/7.0', 'nan']
incomplete_devices = [t.lower() for t in incomplete_devices]


# In[ ]:


rest = common_df[(~common_df['ClearDeviceInfo'].str.lower().isin(incomplete_devices))                 & (~common_df['ClearDeviceInfo'].str.contains('rv:'))]
print(common_df.shape, rest.shape, len(rest['ClearDeviceInfo'].value_counts()))


# Device info often consist of several parts - model, short name of manufactorer and others.
# And on sites these parts mb separated - lets find each of parts.

# In[ ]:


def remove_non_alpha(text):
    return "".join(c for c in text if c.isalpha() or c.isdigit())

def clear_text(text):
    return remove_non_alpha(text).strip()


# In[ ]:


clear_text('azumi_doshi_a5 5_ql')


# In[ ]:


def is_each_word_contained(text_to_find_for_it, text_to_find_in_it):
    chars = [' ', '.', '-', '_', '(', ')']
    words = [text_to_find_for_it.strip()]
    
    for char in chars:
        next_iter_words = []
        for word in words:
            next_iter_words.extend(word.split(char))
        words = next_iter_words
    
    for word in words:
        if clear_text(word) not in clear_text(text_to_find_in_it):
            return False
    
    return True


# In[ ]:


is_each_word_contained("blade v8", 'https:///zte/blade-v8-v0800')


# We use yandex-search to finding specific page for each device and after it check existanse of each part of device-info in link. 

# In[ ]:


# phones_sites = ['www.handsetdetection.com']

# phones_sites_path = ['https://www.handsetdetection.com/device-detection-database/devices/']
# phone_specs_links = {}

# not_found_phones = {}

# for model_i, model in enumerate(rest['ClearDeviceInfo'].unique()):
#     if model_i % 50 == 0:
#         print(model_i, len(phone_specs_links))
    
#     urls = []

#     for site, site_path in zip(phones_sites, phones_sites_path):
#         url = search_first_url_on_site(yandex, model, site)
#         if url is not None and site_path in url and is_each_word_contained(model, url):
#             phone_specs_links[model] = url
#             break
#         urls.append(url)
    
#     if model not in phone_specs_links:
#         not_found_phones[model] = urls


# I saved this dict in pkl, you can just load it.

# In[ ]:


phone_specs_links = joblib.load('/kaggle/input/phone-to-spec-link/phone_specs_links.joblib')
# joblib.dump(phone_specs_links, 'phone_specs_links.joblib')


# And we can extract manufactor from url:

# In[ ]:


def get_manufacturer(url):
    if url is not None and 'handsetdetection' in url:
        return url.split('/devices/')[1].split('/', 1)[0]
    else:
        return ""
    
get_manufacturer('https://www.handsetdetection.com/device-detection-database/devices/samsung/sm-g892a/')


# In[ ]:


rest['manufacturer'] = rest['ClearDeviceInfo'].apply(lambda model                                 : get_manufacturer(phone_specs_links.get(model, "")))


# In[ ]:


samsung_df = common_df[common_df['ClearDeviceInfo'].str.lower().isin(['samsung'])]


# In[ ]:


samsung_df['manufacturer'] = 'samsung'


# In[ ]:


rest = pd.concat([rest, samsung_df])


# In[ ]:


rest['manufacturer'].value_counts().nlargest(20)


# In[ ]:


dic = {}

key = 'manufacturer'

train_rest = rest[rest['target'] >= 0]

top_values = rest[key].value_counts().nlargest(20).keys()

for value in train_rest[key].unique():
    if value in top_values:
        dic[value] = train_rest[train_rest[key] == value]['target'].mean()
    
plt.figure(figsize=(15, 5))
plt.bar(dic.keys(), dic.values())


# Let's dig deeper! - Parse all specs from handsetdetection.com

# In[ ]:


def get_non_empty_children(children):
    for child in children:
        if len(str(child).strip()) != 0:
            yield child


# In[ ]:


def get_specs(link):
    values = {}
    
    if link is None or 'handsetdetection.com/device-detection-database/devices' not in link:
        return values
    
#     try:
    if True:
        page = requests.get(link)
        soup = BeautifulSoup(page.content, "html.parser")

        for table in soup.find_all('table'):
            for t_table in get_non_empty_children(table.children):
                if t_table.name != 'tbody':
                    continue
                
                for row in get_non_empty_children(t_table.children):
                    for cell_i, cell in enumerate(get_non_empty_children(row.children)):
                        for field_i, field in enumerate(get_non_empty_children(cell.children)):
                            if field_i == 0:
#                                 print(cell_i, cell.text.split('\n', 1)[0].strip())
                                if cell_i == 0:
                                    spec_name = cell.text.split('\n', 1)[0].strip()
                                elif cell_i == 1:
                                    spec_value = cell.text.split('\n', 1)[0].strip()

                    values[spec_name] = spec_value
                      
#     except:
#         pass
       
    return values
    


# In[ ]:


get_specs('https://www.handsetdetection.com/device-detection-database/devices/samsung/sm-g892a/')


# In[ ]:


# start_time = datetime.datetime.now()

# backend = 'threading'
# model_and_specs = Parallel(n_jobs=-1, backend=backend)\
#     (delayed(lambda model_and_url : (model_and_url[0], get_specs(model_and_url[1])))(model_and_url) \
#          for model_and_url in phone_specs_links.items())

# end_time =  datetime.datetime.now()
# print(end_time - start_time)


# In[ ]:


# model_and_specs = { k: v for k, v in model_and_specs}
# joblib.dump(model_and_specs, 'model_and_specs_info.joblib')

model_and_specs = joblib.load('/kaggle/input/phone-specs/model_and_specs_info.joblib')


# In[ ]:


def try_get_ram(text):
    parts = text.split(',')
    if len(parts) != 2:
        return ""
    else:
        return parts[1].strip()
    
try_get_ram('128GB, 4GB RAM')


# For example, add RAM to dataframe:

# In[ ]:


device_params_keys = ['memory internal']

for key in device_params_keys:
    rest[key] = rest['ClearDeviceInfo'].apply(lambda x :                        try_get_ram(model_and_specs.get(x, {key:","}).get(key, ",")))


# In[ ]:


rest[device_params_keys[0]].value_counts().nlargest(30)


# In[ ]:


fig, ax = plt.subplots(1,1, figsize=(16, 6))

dic = {}

key = device_params_keys[0]

rest_and_samsung_df=rest
train_rest = rest_and_samsung_df[rest_and_samsung_df['target'] >= 0]

top10_manufacturer = rest_and_samsung_df[key].value_counts().nlargest(10).to_dict().keys()

for manufacturer in train_rest[key].unique():
    if manufacturer in top10_manufacturer:
        dic[manufacturer] = train_rest[train_rest[key] == manufacturer]['target'].mean()

plt.figure(figsize=(15, 5))
ax.bar(dic.keys(), dic.values())
ax.set_label(key)
# plt.show()
# ax.legend()

ax.set_ylabel('Fraud')
ax.set_title(key)
ax.legend()

plt.show()


# We can use all of specs from handsetdetection as features.
# 
# Thanks!
