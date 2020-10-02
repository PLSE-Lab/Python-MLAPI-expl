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
get_ipython().run_line_magic('matplotlib', 'inline')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


data=pd.read_csv('../input/countries-of-the-world/countries of the world.csv')


# ### Assessing data

# In[ ]:


data.shape


# In[ ]:


data.info()


# In[ ]:


data.isnull().sum()


# In[ ]:


data.head()


# ### Data Wrangling

# ### Making a copy of the dataset

# In[ ]:


df=data.copy()


# In[ ]:


df=df.fillna(0)


# In[ ]:


df.isnull().sum()


# ### Replacing all decimal values with decimal point instead of commas and changing data type from str to float

# ### Code

# In[ ]:


cols = df[['Pop. Density (per sq. mi.)' , 'Coastline (coast/area ratio)' , 'Net migration' , 'Infant mortality (per 1000 births)' , 
                   'Literacy (%)' , 'Phones (per 1000)' , 'Arable (%)' , 'Crops (%)' , 'Other (%)' , 'Climate' , 'Birthrate' , 'Deathrate' , 'Agriculture' ,
                   'Industry' , 'Service']]
def rectify(cols):
    for i in cols:
        df[i] = df[i].astype(str)
        new_col = []
        for val in df[i]:
            val = val.replace(',','.')
            val = float(val)
            new_col.append(val)

        
        df[i] = new_col


rectify(cols)


# In[ ]:


df.info()


# In[ ]:


df.head()


# ### Test

# In[ ]:


df


# ### Dropping the 'Other (%)' Column as no inferences can be drawn

# In[ ]:


df=df.drop(columns=['Other (%)'])


# ### Exploratory Data Analysis

# In[ ]:


plt.subplots(figsize=(10,5))
sns.heatmap(df.corr(),linewidth=0.5)
plt.show()


# The follwing have high negative correlation:
# 1. GDP and Infant mortality
# 2. Phones and infant mortality
# 3. Birthrate and GDP
# 4. Agriculture and GDP
# 5. Other and Arable
# 6. Phones and Birthrate
# 7. Agriculture and Phones

# ### The follwing have high positive correlation:
# 1. Birthrate and Infant mortality
# 2. Phones and GDP
# 3. Agriculture and Birthrate
# 4. Infant mortality and agriculture
# 5. Infant mortality and death rate
# 

# In[ ]:


plt.subplots(figsize=(8,8))
df1=df.sort_values('Population',ascending=False).head(10)
plt.pie('Population', labels='Country', autopct="%0.2f%%",data=df1)
plt.show()


# ### Inferences drawn:
# The most populated Country is China followed by India and United States

# In[ ]:


plt.subplots(figsize=(12,5))
df4=df.sort_values('Birthrate',ascending=False).head(10)
sns.barplot(x='Country',y='Birthrate',hue='Region',data=df4)
plt.show()


# ### The birthrate is maximum for Niger followed by Mail and Uganda

# In[ ]:


plt.subplots(figsize=(12,5))
df5=df.sort_values('Deathrate',ascending=False).head(10)
sns.barplot(x='Country',y='Deathrate',hue='Region',data=df5)
plt.show()


# ### The Deathrate is maximum for Swaziland followed by Botswana and Lesotho

# In[ ]:


plt.subplots(figsize=(12,5))
df6=df.sort_values('Literacy (%)',ascending=False).head(10)
sns.barplot(x='Country',y='Literacy (%)',hue='Region',data=df6)
plt.show()


# ### The Literacy percentage is maximum for Austraila followed by Liechtenstein and Andorra
# 
# 
# 
# 
# 
# 

# In[ ]:


plt.subplots(figsize=(10,8))
df_new=df.sort_values('Population',ascending=False).head(10)
sns.barplot(x='Country', y='Literacy (%)', hue='Population',data=df_new)
plt.show()


# ### Inspite of being among the top 10 populated countries, Pakistan and Bangladesh have a literacy percentage of about 45%.

# In[ ]:


plt.subplots(figsize=(12,5))
sns.distplot(df['Birthrate'],hist=False,label='Birthrate')
sns.distplot(df['Deathrate'],hist=False, label='Deathrate')
sns.distplot(df['Infant mortality (per 1000 births)'],hist=False, label='Infant Mortality')
plt.xlabel('Rate')
plt.show()


# ### The probability of 0-20 deaths per 1000 people of population is maximum whereas the probability of 20-60 births per 1000 people of population is maximum.

# In[ ]:


plt.subplots(figsize=(5,5))
df7=df[df['Deathrate']>df['Birthrate']].shape[0]
plt.pie([df7,(df.shape[0]-df7)],labels=['Death Rate > Birth Rate','Birth Rate > Death Rate'],autopct="%0.2f%%")
plt.show()


# ### About 11% of the population have Death Rate>Birth Rate

# In[ ]:


plt.subplots(figsize=(12,5))
df3=df.sort_values('Industry',ascending=False).head(10)
sns.barplot(x='Country',y='Industry',hue='Region',data=df3)
plt.show()


# ### Equatorial Guinea has maximum Industry followed by Qatar and Iraq

# In[ ]:


plt.subplots(figsize=(15,5))
df2=df.sort_values('GDP ($ per capita)',ascending=False).head(10)
sns.barplot(x='Country',y='GDP ($ per capita)',hue='Region',data=df2)
plt.show()


# ### Luxembourg has the highest GDP followed by Norway and United States

# In[ ]:


sns.jointplot(x='Industry',y='GDP ($ per capita)',kind='reg',data=df)
plt.show()


# ### Shockingly,GDP has poor correlation with Industry. The GDP seems to be almost constant with the increase in Industry.

# In[ ]:


sns.jointplot(x='Service',y='GDP ($ per capita)',kind='reg',color='r',data=df)
plt.show()


# ### Here we see that the growth of GDP is very slow with the increase in services.

# In[ ]:


sns.jointplot(x='Agriculture',y='GDP ($ per capita)',kind='reg',data=df,color='g')
plt.show()


# ### Astonishingly, Agriculture shows high negative correlation with GDP.

# In[ ]:


sns.stripplot(x='Climate',y='Crops (%)',data=df,color='maroon')
plt.show()


# ### Climate 2 is favourable for crops

# In[ ]:


plt.subplots(figsize=(12,5))
df9=df.sort_values('Arable (%)',ascending=False).head(10)
sns.barplot(x='Country',y='Arable (%)',hue='Region',data=df9)
plt.show()


# ### Bangladesh has maximum arable land followed by Ukraine and Moldova

# ### Conclusion:
# 1. The most populated Country is China followed by India and United States.
# 2. The birthrate is maximum for Niger followed by Mail and Uganda.
# 3. The Deathrate is maximum for Swaziland followed by Botswana and Lesotho.
# 4. The Literacy percentage is maximum for Austraila followed by Liechtenstein and Andorra.However, inspite of being 
#    among the top 10 populated countries, Pakistan and Bangladesh have a literacy percentage of about 45%.
# 5. About 11% of the population have Death Rate>Birth Rate.The probability of 0-20 deaths per 1000 people of population 
#    is maximum whereas the probability of 20-60 births per 1000 people of population is maximum.
# 6. Equatorial Guinea has maximum Industry followed by Qatar and Iraq.
# 7. Luxembourg has the highest GDP followed by Norway and United States.
# 8. Shockingly,GDP has poor correlation with Industry. The GDP seems to be almost constant with the increase in
#    Industry.
# 9. The growth of GDP is very slow with the increase in Services.
# 10. Astonishingly, Agriculture shows high negative correlation with GDP.
# 11. Climate 2 is favourable for crops.
# 12. Bangladesh has maximum arable land followed by Ukraine and Moldova.
# 
# 
# 
# 
# 
# 
# 
# 

# In[ ]:




