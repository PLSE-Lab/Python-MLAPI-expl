#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from unidecode import unidecode # fast strip accents
import string

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


pages = pd.read_csv("/kaggle/input/kensho-derived-wikimedia-data/page.csv",dtype = {"title":"str"})
print(pages.shape)
pages.describe()


# In[ ]:


pages


# In[ ]:


pages.sort_values("views")


# ### clean text data
# * lowercase later. leave in "-" for now
# 
# * To get a Set/dict/list of the words afterwards : 
#     * https://stackoverflow.com/questions/18936957/count-distinct-words-from-a-pandas-data-frame  

# In[ ]:


pages["title"] = pages["title"].astype(str)


# In[ ]:


pages.head(10)["title"].apply(unidecode)


# In[ ]:


display(pages["title"].head())
pages["title"] = pages["title"].apply(unidecode)


# In[ ]:


display(pages["title"])


# In[ ]:


print(pages.shape[0])
pages.drop_duplicates(["title"]).shape[0]


# In[ ]:


pages.sort_values("views")["title"]


# In[ ]:


print(string.punctuation)


# In[ ]:


## take a subset of punctuations to remove , or keep all of them? (what about "new-york" ?)
puncts = "!#$%&'()*+,-./:;<=>?@[\]^_`{|}~"


# In[ ]:


a = pages.sort_values("views")["title"].replace("[']", "",regex=True).replace("[,.]", " ",regex=True)
a


# In[ ]:


# pages["title"].apply(unidecode).to_csv("wikipedia_pages.csv.gz",index=False,compression="gzip")

a.to_csv("wikipedia_pages.csv.gz",index=False,compression="gzip")

