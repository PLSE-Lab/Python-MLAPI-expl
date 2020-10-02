#!/usr/bin/env python
# coding: utf-8

# # What's in the data?

# In[ ]:


import pandas as pd
import numpy as np


# In[ ]:


# Yeah, I could have made a for loop, but no F*(#s) were given.

#df = pd.read_csv('../input/documents_categories.csv') #['document_id, category_id, confidence_level']
#df1 = pd.read_csv('../input/clicks_test.csv') # ['Display_id, ad_id']
#df2 = pd.read_csv('../input/documents_meta.csv') #['document_id, source_id, publisher_id, publish_time']
#df3 = pd.read_csv('../input/documents_entities.csv') #['document_id, entity_id, confidence_level']
#df4 = pd.read_csv('../input/promoted_content.csv') #['ad_id, document_id, campaign_id, advertister_id']
#df5 = pd.read_csv('../input/sample_submission.csv') #['display_id, ad_id']
#df6 = pd.read_csv('../input/documents_topics.csv') #['document_id, topic_id, confidence_level']
#df7 = pd.read_csv('../input/clicks_train.csv') # ['display_id, ad_id, clicked']
#df8 = pd.read_csv('../input/events.csv')# ['Display_id, uuid, document_id, timestamp, platform, geo_location']
#df9 = pd.read_csv('../input/page_views.csv') 
#df10 = pd.read_csv('../input/page_views_sample.csv') #['uuid, document_id, timestamp, platform, geo_location, traffic_source' ]


# Hmm weird, I can't read in the page_views.csv... Mehhh

# In[ ]:


df = pd.read_csv('../input/documents_categories.csv') #['document_id, category_id, confidence_level']
df.count() # 5481475  (int64)


# In[ ]:


df1 = pd.read_csv('../input/clicks_test.csv') # ['Display_id, ad_id']

df1.count() # 32225162


# In[ ]:


df2 = pd.read_csv('../input/documents_meta.csv') 
#['document_id, source_id, publisher_id, publish_time']
df2.count() # Each id has a differing amount of data as a whole 
#dunno why but it's prob insignificant.


# In[ ]:


df3 = pd.read_csv('../input/documents_entities.csv') #['document_id, entity_id, confidence_level']
df3.count() #5537552


# In[ ]:


df4 = pd.read_csv('../input/promoted_content.csv') #['ad_id, document_id, campaign_id, advertister_id']
df.count() #5481475 
# This is the same as documents_categories.csv


# In[ ]:


df5 = pd.read_csv('../input/sample_submission.csv') #['display_id, ad_id']
df5.count() # 6245533


# In[ ]:


df6 = pd.read_csv('../input/documents_topics.csv') #['document_id, topic_id, confidence_level']
df6.count() # 11325960


# In[ ]:


df7 = pd.read_csv('../input/clicks_train.csv') # ['display_id, ad_id, clicked']
df7.count() #87141731


# In[ ]:


df8 = pd.read_csv('../input/events.csv')# ['Display_id, uuid, document_id, timestamp, platform, geo_location']


# In[ ]:





# In[ ]:




