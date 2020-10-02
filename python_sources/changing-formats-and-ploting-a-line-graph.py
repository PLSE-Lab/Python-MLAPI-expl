#!/usr/bin/env python
# coding: utf-8

# Import essential modules:

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Directory of the dataset :

# In[ ]:


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# Import dataset:

# In[ ]:


df = pd.read_csv('/kaggle/input/hang-seng-index-futures-daily-2000-to-2019/Hang_Seng_Futures_Historical_Data_2009_to_2019.csv')
print(df.dtypes)


# You can see that all of the data is in object/string format. We need to fix it before use. Handling Date:

# In[ ]:


df['Month'] = df['Date'].str.slice(start=0,stop=3) # Get Day from Date
df['Day'] = df['Date'].str.slice(start=4,stop=6).astype(int) # Get Month from Date
df['Year'] = df['Date'].str.slice(start=8,stop=12).astype(int) # Get Year from Date
df = df.rename({'Price': 'Close'}, axis=1) # I prefer to call it 'Close'
df['Date'] = pd.to_datetime(df['Date'])


# Changing formats of the data:

# In[ ]:


df['Vol.'] = df['Vol.'].map(lambda x: np.nan if x == '-' else x)
for i in ['Close','Open','High','Low','Vol.','Change %']:
    df[i] = df[i].str.replace(',','') # Remove ','
    df[i] = df[i].str.replace('K','') # Remove 'K' and multiply the values by 1000 later
    df[i] = df[i].str.replace('%','')
    df[i] = df[i].astype(float)
df['Vol.'] = df['Vol.']*1000
print(df.dtypes)
print(df.head().to_string())


# Let's plot the close price graph from 2018 to 2019:

# In[ ]:


df = df[(df['Year'] == 2018) | (df['Year'] == 2019)]


# In[ ]:


df.plot(x='Date',y='Close',kind='line')
plt.show()

