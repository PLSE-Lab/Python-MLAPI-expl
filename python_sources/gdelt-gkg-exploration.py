#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import math
import gdelt as gd

import matplotlib.pyplot as plt


# Documentation -http://data.gdeltproject.org/documentation/GDELT-Global_Knowledge_Graph_Codebook-V2.1.pdf 

# In[2]:


gs = gd.gdelt(version=2)


# In[3]:


df=gs.Search(date=['2019 04 01','2019 04 03'],table='gkg',output='df',normcols=True)


# In[4]:


def checkFloatNan(x):   
    if type(x).__name__ == 'float':
        return math.isnan(x)
    else:
        return False        
    


# In[5]:


df['IsThemeNaN'] = df['themes'].apply(checkFloatNan)

df = df[df['IsThemeNaN']==False][['gkgrecordid', 'date', 'sourcecollectionidentifier', 'sourcecommonname',
       'documentidentifier', 'counts', 'v2counts', 'themes', 'v2themes',
       'locations', 'v2locations', 'persons', 'v2persons', 'organizations',
       'v2organizations', 'v2tone', 'dates', 'gcam', 'sharingimage',
       'relatedimages', 'socialimageembeds', 'socialvideoembeds', 'quotations',
       'allnames', 'amounts', 'translationinfo', 'extras']]


# In[6]:


df.head()


# In[7]:


df.shape


# In[8]:


print(df['sourcecollectionidentifier'].value_counts())

print('1 = WEB')


# In[9]:


df['sourcecommonname'].value_counts()[:5]#Web Domain names


# In[10]:


df['documentidentifier'][:5]#Complete Urls


# In[11]:


df['documentidentifier'][:1][0]


# In[24]:


print(df[~df['counts'].isnull()]['counts'][2:5].reset_index(drop=True)[0])
print('-------------')
print(df[~df['v2counts'].isnull()]['v2counts'][2:5].reset_index(drop=True)[0])


# In[21]:


df['themes'][0].split(';')


# In[22]:


df['v2themes'][0].split(';')#CharOffset


# In[26]:


print(df[~df['locations'].isnull()]['locations'].reset_index(drop=True)[0])
print('-----------------')
print(df[~df['v2locations'].isnull()]['v2locations'].reset_index(drop=True)[0])


# In[27]:


print(df[~df['persons'].isnull()]['persons'][:5].reset_index(drop=True)[0])
print('--------------')
print(df[~df['v2persons'].isnull()]['v2persons'][:5].reset_index(drop=True)[0])#Char Offset


# In[28]:


print(df[~df['organizations'].isnull()]['organizations'].reset_index(drop=True)[0])
print('----------------')
print(df[~df['v2organizations'].isnull()]['v2organizations'].reset_index(drop=True)[0])#Char Offset


# In[31]:


df['v2tone'].reset_index(drop=True)[0]#between -10 and +10,Positive Score 0-100,Negative Score 0- 100,Polarity score,Ref Den,Ref Den,Word Count.


# In[33]:


df['sharingimage'].reset_index(drop=True)[1]#image


# In[34]:


df[~df['relatedimages'].isnull()]['relatedimages'].reset_index(drop=True)[0]


# In[35]:


df[~df['socialimageembeds'].isnull()]['socialimageembeds'].reset_index(drop=True)[0]


# In[36]:


df[~df['socialvideoembeds'].isnull()]['socialvideoembeds'].reset_index(drop=True)[0]


# In[37]:


df[~df['quotations'].isnull()]['quotations'].reset_index(drop=True)[0]#OFfset|length|verb|Actual Quote


# In[38]:


df[~df['allnames'].isnull()]['allnames'].reset_index(drop=True)[0]#names#offset


# In[39]:


df[~df['amounts'].isnull()]['amounts'].reset_index(drop=True)[0]#Amount,object,offset


# In[41]:


df[~df['translationinfo'].isnull()]['translationinfo']#blank for documents originally in English
#SRCLC. This is the Source Language Code,ENG. This is a textual citation string that indicates the engine(


# In[43]:


df['extras'].reset_index(drop=True)[1]#XML


# In[ ]:





# In[ ]:




