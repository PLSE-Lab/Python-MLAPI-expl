#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


play_store_file = '/kaggle/input/googleplaystore.csv'
user_reviews = '/kaggle/input/googleplaystore_user_reviews.csv'


# In[ ]:


store = pd.read_csv(play_store_file, na_values=['','None','NaN', 'NA', 'nan'])
store.head(10)


# In[ ]:


reviews = pd.read_csv(user_reviews, na_values=['','None','NaN', 'NA', 'nan'])
reviews.head(10)


# In[ ]:


store.info()


# In[ ]:


reviews.info()


# In[ ]:


store.nunique()


# <h1>App</h1>
# Many entries have same application name. Most of them seems redundant entries. We need to remove redundant data, But first, I will clean all other columns to get clear picture about each record.

# In[ ]:


store[store['App'].duplicated(keep=False)]['App'].sort_values(ascending=False)[:10]


# In[ ]:


store[store['App']=='wetter.com - Weather and Radar']


# In[ ]:


store[store['App']=='8 Ball Pool']


# <h1>Category</h1>
# Category column has one unexpected value <i>'1.9'</i>. When it is envestigated, It looks that perticular record has wrongly entered as it is shifted to right by one column. I have fixed it by shifting it left side by one column. I have filled category value as <code>None</code> for that record.

# In[ ]:


store['Category'].unique()


# In[ ]:


store[store['Category']=='1.9']


# In[ ]:


col = store.columns[1:]
col_b = store.columns[1:-1]
temp = store.loc[10472, col_b]
store.loc[10472, 'Category'] = None
for i in range(0, len(col_b)):
    store.loc[10472, col[i+1]] = temp[col_b[i]]
    
store['Category'].unique()


# <h1>Rating</h1>

# In[ ]:


store['Rating'] = pd.to_numeric(store['Rating'], downcast='unsigned', errors='coerce')
store['Rating'].unique()


# <h1>Reviews</h1>

# In[ ]:


store['Reviews'] = pd.to_numeric(store['Reviews'], downcast='unsigned', errors='coerce')
store.head()


# <h1>Size</h1>
# Size was given in MB and kB units. Many records have have value <code>'Varies with device'</code>, <code>'Varies with device'</code> does not help in analysis hence it is made <code>None</code>. And column is converted to numeric type. To keep value in same unit for all records, all values in kB are converted to MB unit. 

# In[ ]:


last_ch = []
val = store['Size']
for v in val:
    if v:
        c = v[-1]
        if c not in last_ch:
            last_ch.append(c)
last_ch


# In[ ]:


store[store['Size']=='Varies with device'][:5]


# In[ ]:


def convert_size(v):
    if v:
        if v == 'Varies with device':
            return None
        else:
            c = v[-1]
            if c == 'M':
                return v[:-1]
            elif c == 'k':
                n = float(v[:-1])
                n = n / 1024.0
                return str(n)
            else:
                return v
    else:
        return None

store['Size'] = pd.to_numeric(store['Size'].apply(convert_size), downcast='unsigned', errors='coerce')
store.head(5)


# <h1>Installs</h1>

# In[ ]:


store['Installs'].unique()


# In[ ]:


def remove_plus(v):
    if '+' in v:
        return v[:-1].replace(',','')
    else:
        return v.replace(',','')

store['Installs'] = pd.to_numeric(store['Installs'].apply(remove_plus), errors='coerce')
store['Installs'].unique()


# <h1>Type</h1>

# In[ ]:


store['Type'].unique()


# <h1>Price</h1>

# In[ ]:


store['Price'].unique()


# In[ ]:


def get_first(v):
    if v:
        return v[0]
    else:
        return v

store['Price'].apply(get_first).unique()


# In[ ]:


def remove_dollor(v):
    if '$' in v:
        return v[1:]
    else:
        return v

store['Price'] = pd.to_numeric(store['Price'].apply(remove_dollor), errors='coerce')
store['Price'].unique()


# <h1>Content Rating</h1>

# In[ ]:


store['Content Rating'].unique()


# <h1>Genres</h1>

# In[ ]:


store['Genres'].unique()


# <h1>Android Ver</h1>
# Android version is generally written in x.x.x format.
# Android versions for application are provided in range (e.g. 5.0 - 8.0) or only earliest supported version (e.g. 5.0 and up).
# To make it uniform for all records, I have used format x.x.x - y.y.y.
# I have used 99.99.99 for highest possible android version.
# Value <code>'Varies with device'</code> is replced with complete version range 1.0.0 - 99.99.99

# In[ ]:


store['Android Ver'].unique()


# In[ ]:


import re
# 4.4W where W stands for wearable
store['Android Ver'] = store['Android Ver'].apply(lambda x: str(x).replace('W',''))
def replace_nan(v):
    if v=='nan':
        return None
    else:
        return v

per_form = '^[0-9]+\.[0-9]+\.[0-9]+ - [0-9]+\.[0-9]+\.[0-9]+$'

def make_uniform(v):
    if v:
        if 'and' in v:
            n = v.count('.')
            if n == 0:
                chat = v.find('and')
                return v[:chat-1]+'.0.0 - 99.99.99'
            elif n == 1:
                chat = v.find('and')
                return v[:chat-1]+'.0 - 99.99.99'
            elif n == 2:
                chat = v.find('and')
                return v[:chat-1]+' - 99.99.99'
            else:
                return v
        elif '-' in v:
            parts = v.split(' - ')
            final = []
            for vp in parts:
                n = vp.count('.')
                if n == 0:
                    vp = vp +'.0.0'
                elif n == 1:
                    vp = vp +'.0'
                final.append(vp)
            return final[0] + ' - ' + final[1]
        else:
            return v
    else:
        return v

def fill_default(v):
    if v == 'Varies with device':
        return '1.0.0 - 99.99.99'
    else:
        return v
store['Android Ver'] = store['Android Ver'].apply(replace_nan)
store['Android Ver'] = store['Android Ver'].apply(make_uniform)
store['Android Ver'] = store['Android Ver'].apply(fill_default)
store['Android Ver'].unique()


# <h1>Last Updated</h1>

# In[ ]:


sorted(store['Last Updated'].unique())[:10]


# In[ ]:


import dateutil
store['Last Updated'] = pd.to_datetime(store['Last Updated'], errors='coerce')
store['Last Updated'] = store['Last Updated'].dt.normalize()
store['Last Updated'].unique()


# <h1>Current Ver</h1>
# Values of current version are made uniform by converting it in format x.x.x Any other value which are not possible to convert to that format are made None.

# In[ ]:


store['Current Ver'].unique()


# In[ ]:


valid_pat = '^[0-9]+\.[0-9]+\.[0-9]+$'
def find_dif(v):
    if not re.match(valid_pat, str(v)) and v != 'Varies with device':
        return True
    else:
        return False
store[store['Current Ver'].apply(find_dif)]['Current Ver'].unique()


# In[ ]:


store[store['Current Ver'].apply(find_dif)]


# In[ ]:


only_text = '^[a-zA-z\s-]*$'
start_v = r'[a-zA-Z \(\)\-]+?'
only_num = '^[0-9]+$'
def remove_string(v):
    if re.match(only_text, str(v)):
        return None
    else:
        return v
def remove_v(v):
    return re.sub(start_v, '', v)
def remove_large(v):
    if ('.' in str(v)) or (v == 'Varies with device') or (len(str(v)) <= 4 and re.match(only_num, str(v))):
        return v
    else:
        return None
def add_dot(v):
    if v != None and '.' not in v and v != 'Varies with device':
        return v+'.0.0'
    elif v != None and str(v).count('.')==1 and v != 'Varies with device':
        return v+'.0'
    else:
        return v
    
def fetch_valid(v):
    if v:
        valid = '[0-9]+\.[0-9]+\.[0-9]+|$'
        find = re.findall(valid, v)[0]
        if find != "":
            return find
        else:
            return v
store['Current Ver'] = store['Current Ver'].apply(lambda x: str(x).replace('_','.'))
store['Current Ver'] = store['Current Ver'].apply(lambda x: x.replace(',','.'))
store['Current Ver'] = store['Current Ver'].apply(remove_v)
store['Current Ver'] = store['Current Ver'].apply(fetch_valid)
store['Current Ver'] = store['Current Ver'].apply(remove_large)
store['Current Ver'] = store['Current Ver'].apply(remove_string)
store['Current Ver'] = store['Current Ver'].apply(add_dot)
store.loc[store['Current Ver'].apply(find_dif), 'Current Ver'] = None
store.loc[store['Current Ver'].apply(find_dif), 'Current Ver'].unique()


# In[ ]:


store.head(5)


# In[ ]:


store.info()

