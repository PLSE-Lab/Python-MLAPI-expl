#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


df  = pd.read_csv(r'../input/german-house-prices/germany_housing_data.csv')


# In[ ]:


df.head()


# In[ ]:


df.drop('Unnamed: 0', axis=1, inplace=True)


# In[ ]:


df.dtypes


# In[ ]:


df.drop_duplicates(inplace=True)


# In[ ]:


df.drop_duplicates(inplace=True)


# In[ ]:


df.shape


# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt


# In[ ]:


sns.barplot(x=df.isnull().sum()/11973, y=df.columns)


# In[ ]:


sns.violinplot(df.Price)


# In[ ]:


sns.violinplot(df.Living_space)


# In[ ]:


sns.violinplot(df.Lot)


# In[ ]:


sns.violinplot(df.Usable_area)


# In[ ]:


sns.violinplot(df.Energy_consumption)


# In[ ]:


plt.xticks(rotation=90)
sns.countplot(df.Rooms)


# In[ ]:


sns.boxplot(df.Rooms)


# In[ ]:


plt.xticks(rotation=90)
sns.countplot(df.Bedrooms)


# In[ ]:


sns.boxplot(df.Rooms)


# In[ ]:


plt.xticks(rotation=90)
sns.countplot(df.Bathrooms)


# In[ ]:


sns.boxplot(df.Bathrooms)


# In[ ]:


sns.countplot(df.Floors)


# In[ ]:


sns.violinplot(df.Year_built)


# In[ ]:


sns.violinplot(df.Year_renovated)


# In[ ]:


plt.xticks(rotation=90)
sns.countplot(df.Garages)


# In[ ]:


plt.xticks(rotation=90)
sns.countplot(df.Type)


# In[ ]:


plt.xticks(rotation=90)
sns.countplot(df.Furnishing_quality)


# In[ ]:


plt.xticks(rotation=90)
sns.countplot(df.Condition)


# In[ ]:


plt.xticks(rotation=90)
sns.countplot(df.Heating)


# In[ ]:


plt.xticks(rotation=90)
sns.countplot(df.Energy_certificate)


# In[ ]:


plt.xticks(rotation=90)
sns.countplot(df.Energy_certificate_type)


# In[ ]:


plt.xticks(rotation=90)
sns.countplot(df.Energy_efficiency_class)


# In[ ]:


plt.xticks(rotation=90)
sns.countplot(df.State)


# In[ ]:


plt.xticks(rotation=90)
sns.countplot(df.Garagetype)


# In[ ]:


def correlation_heatmap(train):
    correlations = train.corr()

    fig, ax = plt.subplots(figsize=(12,12))
    sns.heatmap(correlations, vmax=1.0, center=0, fmt='.2f',
            square=True, linewidths=.5, annot=True, cbar_kws={"shrink": .70})
    plt.show();
correlation_heatmap(df)


# In[ ]:


sns.scatterplot(x='Living_space', y='Price', data=df)


# In[ ]:


sns.scatterplot(x='Lot', y='Price', data=df)


# In[ ]:


sns.scatterplot(x='Usable_area', y='Price', data=df)


# In[ ]:


sns.scatterplot(x='Floors', y='Price', data=df)


# In[ ]:


sns.scatterplot(x='Garages', y='Price', data=df)


# In[ ]:


sns.scatterplot(x='Rooms', y='Price', data=df)


# In[ ]:


sns.scatterplot(x='Energy_consumption', y='Price', data=df)


# In[ ]:


sns.scatterplot(x='Bathrooms', y='Price', data=df)


# In[ ]:


sns.scatterplot(x='Bedrooms', y='Price', data=df)


# In[ ]:


plt.xticks(rotation=90)
sns.stripplot(x='State', y='Price', data=df)


# In[ ]:


sns.barplot(x='Price', y='State', data=df)


# In[ ]:


sns.barplot(x='Energy_certificate', y='Price', data=df)


# In[ ]:


sns.stripplot(x='Energy_certificate', y='Price', data=df)


# In[ ]:


sns.barplot(x='Energy_certificate_type', y='Price', data=df)


# In[ ]:


sns.stripplot(x='Energy_certificate_type', y='Price', data=df)


# In[ ]:


plt.xticks(rotation=90)
sns.barplot(x='Type', y='Price', data=df)


# In[ ]:


plt.xticks(rotation=90)
sns.stripplot(x='Type', y='Price', data=df)


# In[ ]:


plt.xticks(rotation=90)
sns.barplot(x='Condition', y='Price', data=df)


# In[ ]:


plt.xticks(rotation=90)
sns.stripplot(x='Condition', y='Price', data=df)


# In[ ]:


plt.xticks(rotation=90)
sns.barplot(x='Garagetype', y='Price', data=df)


# In[ ]:


plt.xticks(rotation=90)
sns.stripplot(x='Garagetype', y='Price', data=df)


# In[ ]:


plt.xticks(rotation=90)
sns.stripplot(x='Energy_efficiency_class', y='Price', data=df)


# In[ ]:


plt.xticks(rotation=90)
sns.barplot(x='Energy_efficiency_class', y='Price', data=df)


# In[ ]:


plt.xticks(rotation=90)
sns.stripplot(x='Energy_efficiency_class', y='Price', data=df)


# In[ ]:


plt.xticks(rotation=90)
sns.barplot(x='Furnishing_quality', y='Price', data=df)


# In[ ]:


plt.xticks(rotation=90)
sns.stripplot(x='Furnishing_quality', y='Price', data=df)


# In[ ]:





# In[ ]:


plt.xticks(rotation=90)
sns.barplot(x='Heating', y='Price', data=df)


# In[ ]:


plt.xticks(rotation=90)
sns.stripplot(x='Heating', y='Price', data=df)


# In[ ]:


sns.stripplot(x='Energy_efficiency_class', y='Energy_consumption', data=df)


# In[ ]:


plt.xticks(rotation=90)
sns.stripplot(x='Heating', y='Energy_consumption', data=df)


# In[ ]:


sns.stripplot(x='Furnishing_quality', y='Living_space', data=df)


# In[ ]:


plt.xticks(rotation=90)
sns.stripplot(x='State', y='Living_space', data=df)


# In[ ]:


sns.lmplot(x='Living_space', y='Price', hue='State', data=df)


# In[ ]:


sns.lmplot(x='Living_space', y='Price', hue='Furnishing_quality', data=df)


# In[ ]:


sns.lmplot(x='Living_space', y='Price', hue='Type', data=df)


# In[ ]:


sns.lmplot(x='Living_space', y='Price', hue='Condition', data=df)


# In[ ]:


sns.lmplot(x='Living_space', y='Price', hue='Energy_efficiency_class', data=df)


# In[ ]:




