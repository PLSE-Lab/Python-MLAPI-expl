#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import warnings
warnings.filterwarnings("ignore")

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# **Data Loading**

# In[ ]:


df_automobile = pd.read_csv("../input/Automobile_data.csv")


# **Data Cleaning**
# *  Data contains "?" replace it with NAN
# 

# In[ ]:


df_data = df_automobile.replace('?',np.NAN) 
df_data.isnull().sum()


# **Missing Data**
# *  fill missing data of  *normalised-losses, price, horsepower, peak-rpm, bore, stroke* with the respective column mean
# *  Fill missing data category Number of doors with the mode of the column i.e. Four

# In[ ]:


df_temp = df_automobile[df_automobile['normalized-losses']!='?']
normalised_mean = df_temp['normalized-losses'].astype(int).mean()
df_automobile['normalized-losses'] = df_automobile['normalized-losses'].replace('?',normalised_mean).astype(int)

df_temp = df_automobile[df_automobile['price']!='?']
normalised_mean = df_temp['price'].astype(int).mean()
df_automobile['price'] = df_automobile['price'].replace('?',normalised_mean).astype(int)

df_temp = df_automobile[df_automobile['horsepower']!='?']
normalised_mean = df_temp['horsepower'].astype(int).mean()
df_automobile['horsepower'] = df_automobile['horsepower'].replace('?',normalised_mean).astype(int)

df_temp = df_automobile[df_automobile['peak-rpm']!='?']
normalised_mean = df_temp['peak-rpm'].astype(int).mean()
df_automobile['peak-rpm'] = df_automobile['peak-rpm'].replace('?',normalised_mean).astype(int)

df_temp = df_automobile[df_automobile['bore']!='?']
normalised_mean = df_temp['bore'].astype(float).mean()
df_automobile['bore'] = df_automobile['bore'].replace('?',normalised_mean).astype(float)

df_temp = df_automobile[df_automobile['stroke']!='?']
normalised_mean = df_temp['stroke'].astype(float).mean()
df_automobile['stroke'] = df_automobile['stroke'].replace('?',normalised_mean).astype(float)


df_automobile['num-of-doors'] = df_automobile['num-of-doors'].replace('?','four')
df_automobile.head()


# **Summary statistics of variable**

# In[ ]:


df_automobile.describe()


# **Univariate Analysis**

# In[ ]:


# 1 plt.figure(figsize=(10,8))
df_automobile[['engine-size','peak-rpm','curb-weight','horsepower','price']].hist(figsize=(10,8),bins=6,color='Y')
# 2 plt.figure(figsize=(10,8))
plt.tight_layout()
plt.show()


# **Findings**
# *  Most of the car has a Curb Weight is in range 1900 to 3100
# * The Engine Size is inrange 60 to 190
# *  Most vehicle  has horsepower 50 to 125
# *  Most Vehicle are in price range 5000 to 18000
# *  peak rpm is  mostly distributed between 4600 to 5700
# 

# In[ ]:


plt.figure(1)
plt.subplot(221)
df_automobile['engine-type'].value_counts(normalize=True).plot(figsize=(10,8),kind='bar',color='red')
plt.title("Number of Engine Type frequency diagram")
plt.ylabel('Number of Engine Type')
plt.xlabel('engine-type');


plt.subplot(222)
df_automobile['num-of-doors'].value_counts(normalize=True).plot(figsize=(10,8),kind='bar',color='green')
plt.title("Number of Door frequency diagram")
plt.ylabel('Number of Doors')
plt.xlabel('num-of-doors');

plt.subplot(223)
df_automobile['fuel-type'].value_counts(normalize= True).plot(figsize=(10,8),kind='bar',color='purple')
plt.title("Number of Fuel Type frequency diagram")
plt.ylabel('Number of vehicles')
plt.xlabel('fuel-type');

plt.subplot(224)
df_automobile['body-style'].value_counts(normalize=True).plot(figsize=(10,8),kind='bar',color='orange')
plt.title("Number of Body Style frequency diagram")
plt.ylabel('Number of vehicles')
plt.xlabel('body-style');
plt.tight_layout()
plt.show()


# **Findings**
# *  More than 70 % of the vehicle has Ohc type of Engine
# * 57% of the cars has 4 doors
# * Gas is preferred by 85 % of the vehicles
# *  Most produced vehicle are of body style sedan around 48% followed by hatchback 32%

# In[ ]:


import seaborn as sns
corr = df_automobile.corr()
plt.figure(figsize=(20,9))
a = sns.heatmap(corr, annot=True, fmt='.2f')


# **Findings**
# 
# * curb-size, engine-size, horsepower  are positively corelated
# * city-mpg,highway-mpg are negatively corelated

# **Bivariate Analysis**
# 
# >    Price Analysis

# In[ ]:


plt.rcParams['figure.figsize']=(23,10)
ax = sns.boxplot(x="make", y="price", data=df_automobile)


# In[ ]:


plt.rcParams['figure.figsize']=(19,7)
ax = sns.boxplot(x="body-style", y="price", data=df_automobile)


# In[ ]:


sns.catplot(data=df_automobile, x="body-style", y="price", hue="aspiration" ,kind="point")


# In[ ]:


plt.rcParams['figure.figsize']=(10,5)
ax = sns.boxplot(x="drive-wheels", y="price", data=df_automobile)


# **Findings**
# 
# *   Mercedez-Benz ,BMW, Jaguar, Porshe produces expensive cars more than 25000
# *   cheverolet,dodge, honda,mitbushi, nissan,plymouth subaru,toyata produces budget models with lower prices
# *   most of the cars comapany produces car in range below 25000
# *  Hardtop model are expensive in prices followed by convertible and sedan body style
# *  Turbo models have higher prices than for the standard model
# *  Convertible has only standard edition with expensive cars
# *  hatchback and sedan turbo models are available below 20000
# *  rwd wheel drive vehicle have expensive prices
# 
# 
# 

# In[ ]:


sns.factorplot(data=df_automobile, x="engine-type", y="engine-size", col="body-style",row="fuel-type")


# In[ ]:


sns.catplot(data=df_automobile, x="num-of-cylinders", y="horsepower",kind="violin")


# **Findings**
# *  ohc is the most used Engine Type both for diesel and gas
# *  Diesel vehicle have Engine type "ohc" and "I" and engine size ranges between 100 to 190
# * Engine type ohcv has the bigger Engine size ranging from 155 to 300  
# * Body-style  Hatchback uses max variety of Engine Type followed by sedan
# * Body-style Convertible  is not available with Diesel Engine type
# *  Vehicle with above 200 horsepower  has Eight Twelve Six cyclinders

# In[ ]:


sns.catplot(data=df_automobile, y="normalized-losses", x="symboling" , hue="body-style" ,kind="point")


# **Losses Findings**
# 
# Note :- here +3 means risky vehicle and -2 means safe vehicle 
# *  Increased in risk rating linearly increases in normalised losses in vehicle
# *  covertible car and hardtop car has mostly losses with risk rating  above 0
# *  hatchback  cars has highest  losses at risk rating 3
# *  sedan  and Wagon car has  losses even in less risk (safe)rating  

# In[ ]:


g = sns.pairplot(df_automobile[["city-mpg", "horsepower", "engine-size", "curb-weight","price", "fuel-type"]], hue="fuel-type", diag_kind="hist")


# **Findings**
# * Vehicle  Mileage decrease as increase in Horsepower , engine-size, Curb Weight
# * As horsepower increase  the engine size increases
# *  Curbweight increases with the increase in Engine Size
# 
# **Price Analysis**
# *  engine size and curb-weight  is positively co realted with price
# * city-mpg is negatively corelated with price as  increase horsepower reduces the mileage
