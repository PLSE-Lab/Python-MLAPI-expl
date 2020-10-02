#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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


# In[ ]:


import pandas as pd

df = pd.read_csv('../input/room_type.csv')
df.count()


# In[ ]:


df.head(10)


# In[ ]:


from fuzzywuzzy import fuzz


# **RATIO** - Compares the entire string similarity

# In[ ]:


fuzz.ratio('Deluxe Room, 1 King Bed', 'Deluxe King Room')


# In[ ]:


fuzz.ratio('Traditional Double Room, 2 Double Beds', 'Double Room with Two Double Beds')


# In[ ]:


fuzz.ratio('Room, 2 Double Beds (19th to 25th Floors)', 'Two Double Beds - Location Room (19th to 25th Floors)')


# **PARTIAL RATIO** - Compares partial string similarity

# In[ ]:


fuzz.partial_ratio('Deluxe Room, 1 King Bed', 'Deluxe King Room')


# In[ ]:


fuzz.partial_ratio('Traditional Double Room, 2 Double Beds', 'Double Room with Two Double Beds')


# In[ ]:


fuzz.partial_ratio('Room, 2 Double Beds (19th to 25th Floors)', 'Two Double Beds - Location Room (19th to 25th Floors)')


# **TOKEN SORT RATIO** - Ignores word order

# In[ ]:


fuzz.token_sort_ratio('Deluxe Room, 1 King Bed', 'Deluxe King Room')


# In[ ]:


fuzz.token_sort_ratio('Traditional Double Room, 2 Double Beds', 'Double Room with Two Double Beds')


# In[ ]:


fuzz.token_sort_ratio('Room, 2 Double Beds (19th to 25th Floors)', 'Two Double Beds - Location Room (19th to 25th Floors)')


# **TOKEN SET RATIO** - Ignore duplicate words similarly to token sort ratio

# In[ ]:


fuzz.token_set_ratio('Deluxe Room, 1 King Bed', 'Deluxe King Room')


# In[ ]:


fuzz.token_set_ratio('Traditional Double Room, 2 Double Beds', 'Double Room with Two Double Beds')


# In[ ]:


fuzz.token_set_ratio('Room, 2 Double Beds (19th to 25th Floors)', 'Two Double Beds - Location Room (19th to 25th Floors)')


# As **TOKEN SET RATIO** is the best for this dataset, let's explore it a bit more.

# In[ ]:


def get_ratio(row):
    name1 = row['Expedia']
    name2 = row['Booking.com']
    return fuzz.token_set_ratio(name1, name2)

rated = df.apply(get_ratio, axis=1)
rated.head(10)


# Which ones got a set ratio greater than 70%?

# In[ ]:


greater_than_70_percent = df[rated > 70]
greater_than_70_percent.count()


# In[ ]:


greater_than_70_percent.head(10)


# In[ ]:


len(greater_than_70_percent) / len(df)


# More than 90% of the records have a score greater than 70%.

# In[ ]:


greater_than_70_percent = df[rated > 60]
greater_than_70_percent.count()


# In[ ]:


len(greater_than_70_percent) / len(df)


# And more than 98% of the records have a score greater than 60%.

# Thanks to [Susan Li](https://github.com/susanli2016).
