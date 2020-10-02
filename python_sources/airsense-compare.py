#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


# In[2]:


ES = pd.read_excel('../input/airsense_compare/ESP_00983246_2018-08-01_2018-08-21.xlsx')


# In[3]:


ES.head()


# In[4]:


xls = pd.ExcelFile('../input/compare_device/airvisual162.xlsx')


# In[5]:


li = []
for sheet_name in xls.sheet_names:
    df = pd.read_excel('../input/compare_device/airvisual162.xlsx', sheetname=sheet_name)
    li.append(df)


# In[6]:


data_mt = pd.concat(li, axis=0, ignore_index=True).drop_duplicates()


# In[7]:


data_mt = pd.concat(li, axis=0).drop_duplicates().dropna()


# In[8]:


data_mt.head()


# In[9]:


data_mt = data_mt.drop(columns=['Date','Time'])


# In[10]:


data_mt['Datetime'] =  pd.to_datetime(data_mt['Datetime'], format='%m/%d/%y %H:%M:%S')


# In[11]:


data_mt.head()


# In[12]:


data_mt = data_mt.groupby(pd.Grouper(key='Datetime', freq='5T')).mean().dropna()


# In[13]:


data_mt.head()


# In[14]:


ES = ES.groupby(pd.Grouper(key='Time Stamp', freq='5T')).mean().dropna()


# In[15]:


ES.head()


# In[16]:


data_mt = data_mt.rename(index=str, columns={"PM2_5(ug/m3)": "PM2p5", "Temperature(C)": "Temperature", "Humidity(%RH)" : 'Humidity',"PM10(ug/m3)": 'PM10'})


# In[17]:


ES = ES.rename(index=str, columns={" Temperature": "Temperature", " Humidity" : 'Humidity'})


# In[18]:


ES.head()


# In[19]:


data_mt.head()


# In[20]:


ES.shape


# In[21]:


data_mt.shape


# In[22]:


df_inner = pd.merge(ES, data_mt, right_index=True, left_index=True)


# In[23]:


df_inner.head()


# In[24]:


df_inner.shape


# In[25]:


print('PM2.5: %s PM10: %s Temperature: %s Humidity: %s' %(df_inner.PM2p5_x.mean(), df_inner.PM10_x.mean(), df_inner.Temperature_x.mean(), df_inner.Humidity_x.mean()))


# In[26]:


print('PM2.5: %s PM10: %s Temperature: %s Humidity: %s' %(df_inner.PM2p5_y.mean(),  df_inner.PM10_y.mean(), df_inner.Temperature_y.mean(), df_inner.Humidity_y.mean()))


# In[27]:


from scipy.stats import ttest_ind


# In[28]:


ttest_ind(df_inner.Humidity_x, df_inner.Humidity_y)


# In[29]:


ttest_ind(df_inner.Temperature_x, df_inner.Temperature_y)


# In[30]:


ttest_ind(df_inner.PM10_x, df_inner.PM10_y)


# In[31]:


ttest_ind(df_inner.PM2p5_x, df_inner.PM2p5_y)


# In[32]:


import matplotlib.pyplot as plt


# In[33]:


df_inner.Humidity_x.plot(label="ESP_00983246", figsize=(20,10))
df_inner.Humidity_y.plot(label='AirVisual 162')
plt.legend()
plt.title("Humidity")
plt.show()


# In[34]:



df_inner.Temperature_x.plot(label="ESP_00983246", figsize=(20,10))
df_inner.Temperature_y.plot(label='AirVisual 162')
plt.legend()
plt.title("Temperature")
plt.show()


# In[35]:


df_inner.PM10_x.plot(label="ESP_00983246", figsize=(20,10))
df_inner.PM10_y.plot(label='AirVisual 162')
plt.legend()
plt.title("PM10")
plt.show()


# In[36]:


df_inner.PM2p5_x.plot(label="ESP_00983246", figsize=(20,10))
df_inner.PM2p5_y.plot(label='AirVisual 162')
plt.legend()
plt.title("PM2p5")
plt.show()


# In[38]:


from scipy.stats.stats import pearsonr   


# In[39]:


pearsonr(df_inner.Humidity_x,df_inner.Humidity_y)


# In[40]:


pearsonr(df_inner.Temperature_x,df_inner.Temperature_y)


# In[41]:


pearsonr(df_inner.PM10_x,df_inner.PM10_y)


# In[42]:


pearsonr(df_inner.PM2p5_x,df_inner.PM2p5_y)


# In[ ]:




