#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from textblob import TextBlob
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
import csv
    
data = pd.read_csv('../input/temp_room.csv', sep=';') 
print('done')


# In[ ]:


df = pd.DataFrame(data)
df.columns = ['date', 'time', 'temp', '']
df['index'] = df.index
df.head()


# In[ ]:


sns.lineplot('index', 'temp', data=df)

