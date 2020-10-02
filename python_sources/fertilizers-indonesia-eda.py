#!/usr/bin/env python
# coding: utf-8

# # **Fertilizer Export and import in indonesia**

# # 1. Content from the dataset

# the Fertilizers by Product dataset contains information on product amounts for the Production, Trade, Agriculture Use and Other Uses of chemical and mineral fertilizers products, over the time series 2002-present. The fertilizer statistics data are validated separately for a set of over thirty individual products. Both straight and compound fertilizers are included.

# # 2. Overview Data

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


data=pd.read_csv('/kaggle/input/fertilizers-by-product-fao/FertilizersProduct.csv', encoding='latin-1')
data.head()


# Let's look into more details to the data.

# In[ ]:


data.describe()


# looking to the data, we can confirm that the data contains 164,468 with an average of 2,009+/year

# # 3. Check Missing Data

# In[ ]:


data.isna().sum()


# There is no missing data in the entire dataset.

# # 4. Data Exploration

# here I will analyze with the data, How is the import and export relationship of each type of fertilizer in indonesia over the years?

# In[ ]:


indonesia=data.drop(['Area Code','Item Code', 'Element Code', 'Year Code', 'Flag'],axis=1)
indonesia=indonesia.loc[indonesia.Area=='Indonesia']


# In[ ]:


indonesia


# first, i wanna see the development of fertilizer in Indonesia from year to year

# In[ ]:


plt.figure(figsize=(20,8))
total = float(len(indonesia["Area"]))

ax = sns.countplot(x="Year", data=indonesia)
for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x()+p.get_width()/2.,
            height + 3,
            '{:1.2f}'.format((height/total)*100),
            ha="center") 
plt.show()


# seen from the bar plot above that the increase in export imports in Indonesia with the highest percentage in 2017 while the lowest percentage in 2002. second, I wanna see what Elements are exported or imported in Indonesia

# In[ ]:


indonesia.Element.dropna(inplace=True)
labels=indonesia.Element.value_counts().index
colors=['grey','blue','red','yellow','green','pink']
explode=[0,0,0,0,0,0]
sizes=indonesia.Element.value_counts().values

plt.figure(figsize=(10,10))
plt.pie(sizes,explode=explode,colors=colors,labels=labels,autopct='%1.1f%%')
plt.title('export or import elements in Indonesia',color='blue',fontsize=15)


# seen from the bar chart that the most export or import element with a percentage of 25.3% is import value, while the lowest percentage with 5.4% is agricultural use

# In[ ]:


plt.figure(figsize=(10,20))
total = float(len(indonesia))

ax = sns.countplot(y="Item", data=indonesia)
for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x()+p.get_width()/2.,
            height + 3,
            '{:1.2f}'.format((height/total)*120),
            ha="center") 
plt.show()


# seen from the bar that the most export or import production with a percentage of 7.47 is Urea and Superphosphate above 35%, while the lowest percentage with 0.31 is Superphosphates,other.

# In[ ]:


#rata-rata bunuh diri berdasarkan dekade dan gender
plt.figure(figsize=(20,5))
sns.barplot(x='Year',y='Value', hue='Unit',data=indonesia)
plt.xlabel('Year')
plt.ylabel('Value')
plt.title('units based on value and year')
plt.show()


# based on the above bar that tonnes has a far more development than 1000 US$

# In[ ]:




