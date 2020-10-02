#!/usr/bin/env python
# coding: utf-8

# Go to this URL in a browser, and take from where your authorization code

# In[ ]:


https://accounts.google.com/o/oauth2/auth?client_id=947318989803-6bn6qk8qdgf4n4g3pfee6491hc0brc4i.apps.googleusercontent.com&redirect_uri=urn%3aietf%3awg%3aoauth%3a2.0%3aoob&response_type=code&scope=email%20https%3a%2f%2fwww.googleapis.com%2fauth%2fdocs.test%20https%3a%2f%2fwww.googleapis.com%2fauth%2fdrive%20https%3a%2f%2fwww.googleapis.com%2fauth%2fdrive.photos.readonly%20https%3a%2f%2fwww.googleapis.com%2fauth%2fpeopleapi.readonly


# In[ ]:


from google.colab import drive
drive.mount('/content/gdrive')


# In[ ]:


ls


# In[ ]:


url_1 = 'gdrive/My Drive/data/capstone_user_identification/3users/user0001.csv'


# In[ ]:


import pandas as pd
df = pd.read_csv(url_1)
df.head()

