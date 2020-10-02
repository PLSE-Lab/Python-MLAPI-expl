#!/usr/bin/env python
# coding: utf-8

# ## This Kernel is inspired by EDA kernel from Allunia [here](https://www.kaggle.com/allunia/ds4g-eda)

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

#import os

from sklearn.linear_model import LinearRegression

# Any results you write to the current directory are saved as output.


# In[ ]:


from matplotlib.image import imread
import matplotlib.pyplot as plt
from skimage.io import imread
import seaborn as sns
sns.set()
import os


# In[ ]:


my_data_dir = '/kaggle/input/ds4g-environmental-insights-explorer/eie_data/'


# In[ ]:


os.listdir(my_data_dir)


# We can see several subfolders. By looking into the data description, we can find several sources:
# 
# -Global power plant database (gppd)
# 
# -Sentinal 5P OFFL NO2 (s5p_no2)
# 
# -Global Forecast System 384-Hour Predicted Atmosphere Data (gfs)
# 
# -Global Land Data Assimilation System (gldas)

# ### Lets see the GFS Data

# In[ ]:


gfs = my_data_dir+'gfs/'
sample = os.listdir(gfs)[0:5]


# In[ ]:


sample


# The names look like some recent dates but theres a two digit code at the end which I cannot comprehend

# In[ ]:


for n in range(5):
    image = imread(gfs + sample[n])
    print(image.shape)


# 6 channels :o

# In[ ]:


fig, ax = plt.subplots(1,6,figsize=(20,3))

for n in range(6):
    ax[n].imshow(image[:,:,n], cmap="coolwarm")
    ax[n].grid(False)


# ### I cant derive any insights from it

# Lets see the s5p data

# In[ ]:


s5p = "../input/ds4g-environmental-insights-explorer/eie_data/s5p_no2/"


# In[ ]:


sample = os.listdir(s5p)[0:5]
sample


# In[ ]:


for n in range(5):
    image = imread(s5p + sample[n])
    print(image.shape)


# In[ ]:


fig, ax = plt.subplots(1,6,figsize=(20,3))

for n in range(6):
    ax[n].imshow(image[:,:,n], cmap="Blues")
    ax[n].grid(False)


# In[ ]:


fig, ax = plt.subplots(1,6,figsize=(20,3))

for n in range(6):
    ax[n].imshow(image[:,:,n], cmap="coolwarm")
    ax[n].grid(False)


# ### something's going on here but I still cant comprehend

# In[ ]:


fig, ax = plt.subplots(1,6,figsize=(20,3))

for n in range(6):
    ax[n].imshow(image[:,:,n], cmap="Greens")
    ax[n].grid(False)


# In[ ]:


fig, ax = plt.subplots(1,6,figsize=(20,3))

for n in range(6):
    ax[n].imshow(image[:,:,n], cmap="Reds")
    ax[n].grid(False)


# ## Okay, I cant deduct much here. Lets see the csv file

# In[ ]:


data = pd.read_csv('../input/ds4g-environmental-insights-explorer/eie_data/gppd/gppd_120_pr.csv')


# In[ ]:


data.head()


# In[ ]:


data.shape


# In[ ]:


data.info()


# In[ ]:


data.isna().sum()


# In[ ]:


fig, ax = plt.subplots(2,2,figsize=(20,12))
sns.distplot(data.capacity_mw, kde=True, ax=ax[0,0], color="g")
ax[0,0].set_ylabel("counts")
sns.distplot(data.estimated_generation_gwh.apply(np.log), kde=True, ax=ax[0,1], color="k")
ax[0,1].set_xlabel("Log estimated generation gwh")
ax[0,1].set_ylabel("counts");
sns.countplot(data.primary_fuel, ax=ax[1,0], palette="coolwarm");
sns.countplot(data.source, ax=ax[1,1], palette="Blues");
labels = ax[1,1].get_xticklabels()
ax[1,1].set_xticklabels(labels, rotation=90);


# ### Lets see if we can go beyond from here.

# In[ ]:


data.columns


# In[ ]:


data['total_generation'] = data['generation_gwh_2013'] + data['generation_gwh_2014'] + data['generation_gwh_2015'] + data['generation_gwh_2016'] + data['generation_gwh_2017'] 


# In[ ]:


data = data.drop(['generation_gwh_2013','generation_gwh_2014','generation_gwh_2015','generation_gwh_2016','generation_gwh_2017'],axis = 1)


# In[ ]:


data.head()


# In[ ]:


ax = sns.scatterplot(x="capacity_mw", y="estimated_generation_gwh", data=data, color = 'm')


# Looks like there a strong relation here between capacity and estimated generation

# In[ ]:



ax = sns.barplot(x="primary_fuel", y="estimated_generation_gwh", data=data, color = 'k')


# In[ ]:


ax = sns.barplot(x="estimated_generation_gwh", y="source", data=data, color = 'g')


# In[ ]:


ax = sns.scatterplot(x="commissioning_year", y="estimated_generation_gwh", data=data, color = 'r')


# In[ ]:


fuel_mean = data.groupby('primary_fuel').mean()


# In[ ]:


fuel_mean


# In[ ]:


X = fuel_mean['capacity_mw']
y = fuel_mean['estimated_generation_gwh']


# In[ ]:


X = X.values.reshape(-1,1)


# In[ ]:


reg = LinearRegression()
reg = reg.fit(X, y)


# In[ ]:


predictions = reg.predict(X)


# In[ ]:


predictions


# In[ ]:


plt.scatter(X, y, color = 'k')
plt.plot(X, predictions, color = 'm')
plt.title('capacity vs estimated gwh')
plt.xlabel('capacity')
plt.ylabel('estimated gwh')


# ## Lets evaluate our model

# In[ ]:


from sklearn.metrics import r2_score, mean_squared_error


# In[ ]:


r2_score(y,predictions)


# In[ ]:


mean_squared_error(y,predictions)


# Lets see if can find a way to calculate coefficients

# In[ ]:


fuel_mean = fuel_mean.sort_values('capacity_mw')
fuel_mean['weights'] = list(fuel_mean['estimated_generation_gwh'] / fuel_mean['capacity_mw'])
fuel_mean['weights'] = fuel_mean['weights'] / fuel_mean['weights'].sum() * 1000
fuel_mean['the_adjusted_ratio'] = fuel_mean['weights'] * fuel_mean['capacity_mw']


# In[ ]:


fuel_mean


# In[ ]:


X = fuel_mean['the_adjusted_ratio']
y = fuel_mean['estimated_generation_gwh']


# In[ ]:


reg = LinearRegression()
reg = reg.fit(X.values.reshape(-1,1),y)
predictions = reg.predict(X.values.reshape(-1,1))


# In[ ]:


r2_score(y,predictions), mean_squared_error(y,predictions)


# In[ ]:


coefficients = dict(zip(fuel_mean.index, fuel_mean['the_adjusted_ratio']))


# In[ ]:


coefficients


# #### Follow the link [here](http://open_jicareport.jica.go.jp/pdf/12339586_07.pdf)
# 
# Research by Record of JCC and TWG Workshop

# ## exp_gwh = coefficient * capacity

# In[ ]:


for index,row in data.iterrows():
    for name in data.primary_fuel.unique():
        if row['primary_fuel'] == name:
            data.loc[index,'predicted_gwh'] = (coefficients[name] * row['capacity_mw'])
            
mean_squared_error(data['estimated_generation_gwh'], data['predicted_gwh'])


# In[ ]:


ax = sns.scatterplot(x="estimated_generation_gwh", y="predicted_gwh", data=data, color= 'k')


# ### Lets calculate the mean of primary fuel

# In[ ]:


fuel_mean = data.groupby('primary_fuel').mean()
fuel_mean['new_ratios'] = fuel_mean['predicted_gwh'] / fuel_mean['estimated_generation_gwh']


# In[ ]:


for name in fuel_mean.index:
    coefficients[name] =  coefficients[name] / fuel_mean.loc[name,'new_ratios']


# In[ ]:


for index,row in data.iterrows():
    for name in data.primary_fuel.unique():
        if row['primary_fuel'] == name:
            data.loc[index,'predicted_gwh'] = (coefficients[name] * row['capacity_mw'])
            

mean_squared_error(data['estimated_generation_gwh'], data['predicted_gwh'])

