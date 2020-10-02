#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O(e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
import missingno as mis
get_ipython().run_line_magic('matplotlib', 'inline')
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


data=pd.read_csv('/kaggle/input/farmers-markets-in-the-united-states/wiki_county_info.csv')
info=pd.read_csv('/kaggle/input/farmers-markets-in-the-united-states/farmers_markets_from_usda.csv')


# In[ ]:


data.head()


# In[ ]:


info.head()


# **Well from the above data we conclude that**
# 1. data :- file contain's information about the citizen's of the U.S
# 2. info :- contains information about the U.S. market 
# 
# Well let's start analyzing both the data files one by and by one

# In[ ]:


data.info()


# In[ ]:


data.describe()


# Let's check if it conatin's some missing data or not ?

# In[ ]:


sns.heatmap(data.isnull(),cmap='Blues')


# In[ ]:


data.isnull().sum()


# Well if our data contain's more than values then we will calculate the % and printing the bargraph's of it but as it contain's limited missiing values so we have printed the limited values of it :)

# As the missing values are very less so we will drop it all :)

# In[ ]:


data.dropna(inplace=True)


# #  **Analyzing Part**

# 1. Let's work on county part

# In[ ]:


data['county'].value_counts()


# In[ ]:


plt.figure(figsize=(8,6))
sns.barplot(x=data['county'].value_counts()[0:30],y=data['county'].value_counts()[0:30].index)


# 2. State

# In[ ]:


data['State'].nunique()


# Well we have data of 56 states

# In[ ]:


plt.figure(figsize=(8,6))
sns.barplot(x=data['State'].value_counts()[0:30],y=data['State'].value_counts()[0:30].index,palette="ch:.25")


#  # **Converting the columns into integer so as to get more insight on our visualisation**

# In[ ]:


def remove_char(df):
    bad_var = ['per capita income', 'median household income', 'median family income', 'population', 'number of households']
    bad_tokens = ['$', r',']
    
    for var in bad_var:
        df[var] = df[var].replace('[\$,]', '', regex=True).astype(int)
    return df


# In[ ]:


data=remove_char(data)


# 3. **Population**

# In[ ]:


plt.figure(figsize=(20,6))
plt.plot(data['population'])
plt.title('Variation of Population')


# 4. **No of households**

# In[ ]:


plt.figure(figsize=(20,6))
plt.plot(data['number of households'])
plt.title('Variation of No of Households')


# 5. Let's see the relation between **Population vs No of households**

# In[ ]:


plt.figure(figsize=(10,4))
axes=sns.lmplot(x='population',y='number of households',data=data,palette='coolwarm')


# Well we can conclude from the above that there is a liner relation between population and no of households across all the county

# Let's calculate the avg person present in a particular household

# In[ ]:


data['Avg_person']=data['population']/data['number of households']
avg=data['Avg_person'].sum()/len(data)
print(avg)


# In[ ]:


data['Avg_person']=data['Avg_person'].round(0).astype(int)


# In[ ]:


sns.countplot(x=data['Avg_person'])


# **So we can say that 3 person per household is more common in many homes :)**

# Let's check the **State vs Avg person per household**

# In[ ]:


plt.figure(figsize=(6,13))
sns.stripplot(x='Avg_person',y='State',data=data,jitter=True) 


# Well from above we can conclude that only **Alaska** contain's (2,3,4,5,7) different avg person's

# **Now let's work on the income part**

# 1. Let's look at the trend of per capita income

# In[ ]:


plt.figure(figsize=(20,6))
plt.title('Variation of Per Capita Income')
plt.plot(data['per capita income'])


# 2. Median household income

# In[ ]:


plt.figure(figsize=(20,6))
plt.title('Variation of Median Household Income')
plt.plot(data['median household income'])


# 3. Median family income

# In[ ]:


plt.figure(figsize=(20,6))
plt.title('Variation of Median Family Income')
plt.plot(data['median family income'])


# **Let's see the relation between PCC vs MHI vs MFI**

# In[ ]:


plt.figure(figsize=(20,8))
sns.lineplot(data=data.iloc[:,[3,4,5]])


# **Well median household income is always greater than median household income**

# Let' see how the COUNNTY and STATE realtion among the other dataset's

# Let's convert the data in the required form so that the particular county and state are having a mean values of their corresponding data :)

# In[ ]:


county=data.groupby(['county']).mean()
state=data.groupby(['State']).mean()


# In[ ]:


county.head()


# In[ ]:


state.head()


# #  **State**

# 1. With PCI

# In[ ]:


state_PCI=state.sort_values(by=['per capita income'],ascending=False)
state_PCI=state_PCI.head(10)
state_PCI=state_PCI.reset_index()
plt.figure(figsize=(16,5))
sns.barplot(x='State',y='per capita income',data=state_PCI)
plt.title('Top 10 states having higest Per Capita Income')


# 2. **Median Household Income**

# In[ ]:


state_MHI=state.sort_values(by=['median household income'],ascending=False)
state_MHI=state_MHI.head(10)
state_MHI=state_MHI.reset_index()
plt.figure(figsize=(16,5))
sns.barplot(y='State',x='median household income',data=state_MHI)
plt.title('Top 10 states having higest Median Household Income')


# 3. **Median family income**

# In[ ]:


state_MFI=state.sort_values(by=['median family income'],ascending=False)
state_MFI=state_MFI.head(10)
state_MFI=state_MFI.reset_index()
plt.figure(figsize=(16,5))
sns.barplot(x='State',y='median family income',data=state_MFI)
plt.title('Top 10 states having higest Median Family Income')


# 4. Population

# In[ ]:


state_pop=state.sort_values(by=['population'],ascending=False)
state_pop=state_pop.head(10)
state_pop=state_pop.reset_index()
plt.figure(figsize=(16,5))
sns.barplot(x='population',y='State',data=state_pop)
plt.title('Top 10 states having higest Population')


# # **County**

# 1. **PCI**

# In[ ]:


county_PCI=county.sort_values(by=['per capita income'],ascending=False)
county_PCI=county_PCI.head(10)
county_PCI=county_PCI.reset_index()
plt.figure(figsize=(22,5))
sns.scatterplot(x='county',y='per capita income',data=county_PCI)
plt.title('Top 10 county having higest Per Capita Income')


# 2. **Median Household Income**

# In[ ]:


county_MHI=county.sort_values(by=['median household income'],ascending=False)
county_MHI=county_MHI.head(10)
county_MHI=county_MHI.reset_index()
plt.figure(figsize=(22,5))
sns.barplot(x='county',y='median household income',data=county_MHI)
plt.title('Top 10 county having higest Median Household Income')


# 3. Median family income

# In[ ]:


county_MFI=county.sort_values(by=['median family income'],ascending=False)
county_MFI=county_MFI.head(10)
county_MFI=county_MFI.reset_index()
plt.figure(figsize=(22,5))
sns.barplot(x='county',y='median family income',data=county_MFI)
plt.title('Top 10 county having higest Median Family Income')


# 4. **Population**

# In[ ]:


county_pop=county.sort_values(by=['population'],ascending=False)
county_pop=county_pop.head(10)
county_pop=county_pop.reset_index()
plt.figure(figsize=(22,5))
sns.barplot(x='population',y='county',data=county_pop)
plt.title('Top 10 county having higest Median Family Income')


# Let's get a overview of how everything is correlated to each other :)

# In[ ]:


sns.pairplot(data,palette='coolwarm')


# In[ ]:


plt.figure(figsize=(15,6))
sns.heatmap(data.corr(),annot=True,cmap='coolwarm')


# From the above correlation we get the following conclusion's

# 1. **Per Capita Income is directly propotional to the Median Family Income and Median Household income**
# 
# Now let's visualize them :)

# In[ ]:


plt.figure(figsize=(24,8))
sns.regplot(x='per capita income',y='median household income',data=data,color='orange')
plt.title('Per Capita Income vs Median Household Income')


# In[ ]:


plt.figure(figsize=(24,8))
sns.regplot(x='per capita income',y='median family income',data=data,color='blue')
plt.title('Per Capita Income vs Median Family Income')


# 2. Population is amost showing 1 to one relation with the No of households 

# In[ ]:


plt.figure(figsize=(24,8))
sns.regplot(x='number of households',y='population',data=data,color='green')
plt.title('Population vs No of Households')

