#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


raindata = pd.read_csv("../input/rainfall-in-india/rainfall in india 1901-2015.csv")
print(raindata.head())


# In[ ]:


print(raindata.columns)
print(set(raindata['SUBDIVISION']))
# ap_rain= raindata[raindata['SUBDIVISION']=='Andhra Pradesh']
# print(set(raindata['Year']))
# ap_rain_2000 = ap_rain[ap_rain['Year'] == 2000]
# print(ap_rain_2000)


# In[ ]:


valid_states_rain = raindata[(raindata['SUBDIVISION']=='BIHAR') | (raindata['SUBDIVISION']=='KERALA') | (raindata['SUBDIVISION']=='ARUNACHAL PRADESH')|(raindata['SUBDIVISION']=='TAMIL NADU')|(raindata['SUBDIVISION']=='JAMMU & KASHMIR')|(raindata['SUBDIVISION']=='UTTARAKHAND')|(raindata['SUBDIVISION']=='ORISSA')|(raindata['SUBDIVISION']=='HIMACHAL PRADESH')|(raindata['SUBDIVISION']=='LAKSHADWEEP')|(raindata['SUBDIVISION']=='PUNJAB')|(raindata['SUBDIVISION']=='CHHATTISGARH')|(raindata['SUBDIVISION']=='ANDAMAN & NICOBAR ISLANDS')|(raindata['SUBDIVISION']=='JHARKHAND')]


# In[ ]:


print(set(valid_states_rain['SUBDIVISION']))
print(valid_states_rain.head())
valid_states_rain = valid_states_rain[['SUBDIVISION','YEAR','ANNUAL']]
print(valid_states_rain.head())
print(valid_states_rain.describe())


# In[ ]:


valid_states_rain = valid_states_rain[valid_states_rain['YEAR']>1996]


# In[ ]:


valid_states_rain.columns = ['State','Year','Rainfall']


# In[ ]:


valid_states_rain


# In[ ]:


cropdata = pd.read_csv("../input/crop-dataset/my_dataset.csv")


# In[ ]:


print(set(valid_states_rain['State']))
print(set(cropdata['State']))


# In[ ]:


valid_states_crop = cropdata[(cropdata['State']=='Bihar') | (cropdata['State']=='Kerala') | (cropdata['State']=='Arunachal Pradesh')|(cropdata['State']=='Tamil Nadu')|(cropdata['State']=='Jammu and Kashmir ')|(cropdata['State']=='Uttarakhand')|(cropdata['State']=='Odisha')|(cropdata['State']=='Himachal Pradesh')|(cropdata['State']=='Punjab')|(cropdata['State']=='Chhattisgarh')|(cropdata['State']=='Andaman and Nicobar Islands')|(cropdata['State']=='Jharkhand')]


# In[ ]:


valid_states_crop


# In[ ]:


print(set(valid_states_crop['State']))
print(set(valid_states_rain['State']))


# In[ ]:


valid_states_rain = valid_states_rain[valid_states_rain.State != 'LAKSHADWEEP']
valid_states_crop = valid_states_crop.replace('Jammu and Kashmir ','Jammu and Kashmir')
valid_states_rain = valid_states_rain.replace('UTTARAKHAND','Uttarakhand')
valid_states_rain = valid_states_rain.replace('ORISSA','Odisha')
valid_states_rain = valid_states_rain.replace('HIMACHAL PRADESH','Himachal Pradesh')
valid_states_rain = valid_states_rain.replace('JHARKHAND','Jharkhand')
valid_states_rain = valid_states_rain.replace('ARUNACHAL PRADESH','Arunachal Pradesh')
valid_states_rain = valid_states_rain.replace('TAMIL NADU','Tamil Nadu')
valid_states_rain = valid_states_rain.replace('CHHATTISGARH','Chhattisgarh')
valid_states_rain = valid_states_rain.replace('JAMMU & KASHMIR','Jammu and Kashmir')
valid_states_rain = valid_states_rain.replace('ANDAMAN & NICOBAR ISLANDS','Andaman and Nicobar Islands')
valid_states_rain = valid_states_rain.replace('BIHAR','Bihar')
valid_states_rain = valid_states_rain.replace('PUNJAB','Punjab')
valid_states_rain = valid_states_rain.replace('KERALA','Kerala')


# In[ ]:


print(set(valid_states_crop['State']))
print(set(valid_states_rain['State']))


# In[ ]:


#valid_states_crop.columns = ['State','Year','Crop','Area','Production','Rainfall']
Rainfall_list = [0]*6997
valid_states_crop['Rainfall'] = Rainfall_list


# In[ ]:


print(valid_states_crop.head())
print(valid_states_rain.head())


# In[ ]:


states_set = set(valid_states_crop['State'])
year_set = set(valid_states_crop['Year'])
print(states_set)
print(year_set)


# In[ ]:


for state in states_set:
    for year in year_set:
        #print(valid_states_crop[valid_states_crop['State']==state]['Year'])
        if(year in list(valid_states_crop[valid_states_crop['State']==state]['Year'])):
            #print(state,year)
            valid_states_crop.loc[(valid_states_crop['State']==state) &                               (valid_states_crop['Year']==year),'Rainfall']=             list(valid_states_rain[(valid_states_rain['State']==state) &                                    (valid_states_rain['Year']==year)]['Rainfall'])[0]


# In[ ]:


valid_states_crop


# In[ ]:


crop_data_alpha = valid_states_crop.dropna()
crop_data_alpha = crop_data_alpha[crop_data_alpha.Production !=0.0]
print(crop_data_alpha.head())
print(set(crop_data_alpha['Crop']))


# In[ ]:


crop_data_alpha = crop_data_alpha.drop('State',axis = 1)
crop_data_alpha = crop_data_alpha.drop('Year',axis = 1)
crop_data_alpha = crop_data_alpha.sort_values(by='Crop')
crop_data_alpha = crop_data_alpha.reset_index(drop=True)
print(crop_data_alpha.head())


# In[ ]:


for index,row in crop_data_alpha.iterrows():
    crop_data_alpha.loc[index,'Production'] = row['Production']/row['Area']
crop_data_alpha.head()


# In[ ]:


wheat_data = crop_data_alpha[crop_data_alpha['Crop']=='Wheat']
print(len(wheat_data))
print(wheat_data.head())
print(wheat_data[wheat_data['Production']==3.0])


# In[ ]:


plt.scatter(wheat_data['Production'],wheat_data['Rainfall'])


# In[ ]:


potato_data = crop_data_alpha[crop_data_alpha['Crop']=='Potato']
plt.scatter(potato_data['Production'],potato_data['Rainfall'])

