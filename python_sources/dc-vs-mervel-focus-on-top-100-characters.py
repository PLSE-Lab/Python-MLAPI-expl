#!/usr/bin/env python
# coding: utf-8

# # Introduction
# This kernel explored the data set about comic charecters.  
# **If you like it, please upvote.  **  
# Your one click makes me very happy ;)

# # Notebook Outline
# 1.  [**Data Load**](#Data-Load)   
# 2.  [**Data Visualization**](#Data-Visualization)   
# 3.  [**Data Visualization focus on Top 100 characters**](#Data-Visualization-focus-on-Top-100-characters)   

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
import warnings
warnings.filterwarnings('ignore')
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# # Comic Characters
# 
# This folder contains data behind the story [Comic Books Are Still Made By Men, For Men And About Men](http://fivethirtyeight.com/features/women-in-comic-books/).
# 
# The data comes from [Marvel Wikia](http://marvel.wikia.com/Main_Page) and [DC Wikia](http://dc.wikia.com/wiki/Main_Page). Characters were scraped on August 24. Appearance counts were scraped on September 2. The month and year of the first issue each character appeared in was pulled on October 6.
# 
# The data is split into two files, for DC and Marvel, respectively: `dc-wikia-data.csv` and `marvel-wikia-data.csv`. Each file has the following variables:
# 
# Variable | Definition
# ---|---------
# `page_id` | The unique identifier for that characters page within the wikia
# `name` | The name of the character
# `urlslug` | The unique url within the wikia that takes you to the character
# `ID` | The identity status of the character (Secret Identity, Public identity, [on marvel only: No Dual Identity])
# `ALIGN` | If the character is Good, Bad or Neutral
# `EYE` | Eye color of the character
# `HAIR` | Hair color of the character
# `SEX` | Sex of the character (e.g. Male, Female, etc.)
# `GSM` | If the character is a gender or sexual minority (e.g. Homosexual characters, bisexual characters)
# `ALIVE` | If the character is alive or deceased
# `APPEARANCES` | The number of appareances of the character in comic books (as of Sep. 2, 2014. Number will become increasingly out of date as time goes on.)
# `FIRST APPEARANCE` | The month and year of the character's first appearance in a comic book, if available
# `YEAR` | The year of the character's first appearance in a comic book, if available

# # Data Load

# In[ ]:


data_dc = pd.read_csv('../input/dc-wikia-data.csv')


# In[ ]:


data_dc.head()


# In[ ]:


data_dc.tail()


# In[ ]:


data_dc.sample(5)


# In[ ]:


data_dc.describe()


# In[ ]:


data_dc.info()


# In[ ]:


data_dc.columns


# In[ ]:


data_dc = data_dc.rename(columns={'page_id':'Page_id',
                                  'name':'Name',
                                  'urlslug':'Urlslug',
                                  'ID':'ID',
                                  'ALIGN':'Align',
                                  'EYE':'Eye',
                                  'HAIR':'Hair',
                                  'SEX':'Gender',
                                  'GSM':'GSM',
                                  'ALIVE':'Alive',
                                  'APPEARANCES':'Appearances',
                                  'FIRST APPEARANCE':'FirstAppearances',
                                  'YEAR':'Year'})


# In[ ]:


data_dc['Inc'] = 'DC'


# In[ ]:


data_dc.head()


# In[ ]:


data_marvel = pd.read_csv('../input/marvel-wikia-data.csv')


# In[ ]:


data_marvel.head()


# In[ ]:


data_marvel.tail()


# In[ ]:


data_marvel.sample(5)


# In[ ]:


data_marvel.describe()


# In[ ]:


data_marvel.info()


# In[ ]:


data_marvel.columns


# In[ ]:


data_marvel = data_marvel.rename(columns={'page_id':'Page_id',
                                          'name':'Name',
                                          'urlslug':'Urlslug',
                                          'ID':'ID',
                                          'ALIGN':'Align',
                                          'EYE':'Eye',
                                          'HAIR':'Hair',
                                          'SEX':'Gender',
                                          'GSM':'GSM',
                                          'ALIVE':'Alive',
                                          'APPEARANCES':'Appearances',
                                          'FIRST APPEARANCE':'FirstAppearances',
                                          'Year':'Year'})


# In[ ]:


data_marvel['Inc'] = 'Mervel'


# In[ ]:


data = pd.concat([data_dc,data_marvel])


# In[ ]:


data.describe()


# In[ ]:


data.info()


# In[ ]:


data.isnull().sum()


# # Data Visualization

# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
def drow_pie(dataset,column):
    f,ax=plt.subplots(1,2,figsize=(18,9))
    explode_list = [0.1] * (dataset[dataset['Inc'] == 'DC'][column].unique().size-1)
    dataset[dataset['Inc'] == 'DC'][column].value_counts().plot.pie(explode=explode_list,autopct='%1.1f%%',ax=ax[0],shadow=True)
    ax[0].set_title('DC {} Count'.format(column))
    ax[0].set_ylabel('Count')
    explode_list = [0.1] * (dataset[dataset['Inc'] == 'Mervel'][column].unique().size-1)
    dataset[dataset['Inc'] == 'Mervel'][column].value_counts().plot.pie(explode=explode_list,autopct='%1.1f%%',ax=ax[1],shadow=True)
    ax[1].set_title('Mervel {} Count'.format(column))
    ax[1].set_ylabel('Count')
    plt.show()


# In[ ]:


drow_pie(data,'ID')


# In[ ]:


plt.figure(figsize=(20,5))
sns.set_context("paper", 2.0, {"lines.linewidth": 4})
sns.countplot(x='ID',hue='Inc',data=data)


# In[ ]:


drow_pie(data,'Align')


# In[ ]:


plt.figure(figsize=(20,5))
sns.set_context("paper", 2.0, {"lines.linewidth": 4})
sns.countplot(x='Align',hue='Inc',data=data)


# In[ ]:


drow_pie(data,'Eye')


# In[ ]:


plt.figure(figsize=(10,20))
sns.set_context("paper", 2.0, {"lines.linewidth": 4})
sns.countplot(y='Eye',hue='Inc',data=data)


# In[ ]:


drow_pie(data,'Hair')


# In[ ]:


plt.figure(figsize=(10,20))
sns.set_context("paper", 2.0, {"lines.linewidth": 4})
sns.countplot(y='Hair',hue='Inc',data=data)


# In[ ]:


drow_pie(data,'Gender')


# In[ ]:


plt.figure(figsize=(25,5))
sns.set_context("paper", 2.0, {"lines.linewidth": 4})
sns.countplot(x='Gender',hue='Inc',data=data)


# In[ ]:


drow_pie(data,'GSM')


# In[ ]:


plt.figure(figsize=(25,5))
sns.set_context("paper", 2.0, {"lines.linewidth": 4})
sns.countplot(x='GSM',hue='Inc',data=data)


# In[ ]:


drow_pie(data,'Alive')


# In[ ]:


plt.figure(figsize=(20,5))
sns.set_context("paper", 2.0, {"lines.linewidth": 4})
sns.countplot(x='Alive',hue='Inc',data=data)


# In[ ]:


import numpy as np
plt.figure(figsize=(20,5))
sns.distplot(data['Appearances'],hist=False,bins=1)


# In[ ]:


sns.FacetGrid(data, hue="Inc", size=8).map(sns.kdeplot, "Appearances").add_legend()
plt.ioff() 
plt.show()


# # Data Visialization focus on Top 100 characters

# In[ ]:


# Select the top 100 appearances
data_top_100_dc = data[data['Inc'] == 'DC'].nlargest(100,'Appearances')  
data_top_100_dc.shape


# In[ ]:


data_top_100_dc.head()


# In[ ]:


# Select the top 100 appearances
data_top_100_mervel = data[data['Inc'] == 'Mervel'].nlargest(100,'Appearances')  
data_top_100_mervel.shape


# In[ ]:


data_top_100_mervel.head()


# In[ ]:


data_top_100 = pd.concat([data_top_100_dc,data_top_100_mervel])
data_top_100.shape


# In[ ]:


sns.FacetGrid(data_top_100, hue="Inc", size=8).map(sns.kdeplot, "Appearances").add_legend()
plt.ioff() 
plt.show()


# In[ ]:


f,ax=plt.subplots(1,2,figsize=(18,9))
explode_list = [0.1] * (data_top_100[data_top_100['Inc'] == 'DC']['ID'].unique().size)
data_top_100[data_top_100['Inc'] == 'DC']['ID'].value_counts().plot.pie(explode=explode_list,autopct='%1.1f%%',ax=ax[0],shadow=True)
ax[0].set_title('DC {} Count'.format('ID'))
ax[0].set_ylabel('Count')
explode_list = [0.1] * (data_top_100[data_top_100['Inc'] == 'Mervel']['ID'].unique().size)
data_top_100[data_top_100['Inc'] == 'Mervel']['ID'].value_counts().plot.pie(explode=explode_list,autopct='%1.1f%%',ax=ax[1],shadow=True)
ax[1].set_title('Mervel {} Count'.format('ID'))
ax[1].set_ylabel('Count')
plt.show()


# In[ ]:


plt.figure(figsize=(20,5))
sns.set_context("paper", 2.0, {"lines.linewidth": 4})
sns.countplot(x='ID',hue='Inc',data=data_top_100)


# In[ ]:


plt.figure(figsize=(15,10))
sns.swarmplot(x="Inc", y="Appearances",hue='ID', data=data_top_100)


# In[ ]:


plt.figure(figsize=(15,10))
sns.boxplot(x="ID", y="Appearances",
            hue="Inc", palette=["m", "g"],
            data=data_top_100)


# In[ ]:


f,ax=plt.subplots(1,2,figsize=(18,9))
explode_list = [0.1] * (data_top_100[data_top_100['Inc'] == 'DC']['Align'].unique().size-1)
data_top_100[data_top_100['Inc'] == 'DC']['Align'].value_counts().plot.pie(explode=explode_list,autopct='%1.1f%%',ax=ax[0],shadow=True)
ax[0].set_title('DC {} Count'.format('Align'))
ax[0].set_ylabel('Count')
explode_list = [0.1] * (data_top_100[data_top_100['Inc'] == 'Mervel']['Align'].unique().size-1)
data_top_100[data_top_100['Inc'] == 'Mervel']['Align'].value_counts().plot.pie(explode=explode_list,autopct='%1.1f%%',ax=ax[1],shadow=True)
ax[1].set_title('Mervel {} Count'.format('Align'))
ax[1].set_ylabel('Count')
plt.show()


# In[ ]:


plt.figure(figsize=(20,5))
sns.set_context("paper", 2.0, {"lines.linewidth": 4})
sns.countplot(x='Align',hue='Inc',data=data_top_100)


# In[ ]:


plt.figure(figsize=(15,10))
sns.swarmplot(x="Inc", y="Appearances",hue='Align', data=data_top_100)


# In[ ]:


plt.figure(figsize=(20,10))
sns.boxplot(x="Align", y="Appearances",
            hue="Inc", palette=["m", "g"],
            data=data_top_100)


# In[ ]:


f,ax=plt.subplots(1,2,figsize=(18,9))
explode_list = [0.1] * (data_top_100[data_top_100['Inc'] == 'DC']['Eye'].unique().size-1)
data_top_100[data_top_100['Inc'] == 'DC']['Eye'].value_counts().plot.pie(explode=explode_list,autopct='%1.1f%%',ax=ax[0],shadow=True)
ax[0].set_title('DC {} Count'.format('Eye'))
ax[0].set_ylabel('Count')
explode_list = [0.1] * (data_top_100[data_top_100['Inc'] == 'Mervel']['Eye'].unique().size)
data_top_100[data_top_100['Inc'] == 'Mervel']['Eye'].value_counts().plot.pie(explode=explode_list,autopct='%1.1f%%',ax=ax[1],shadow=True)
ax[1].set_title('Mervel {} Count'.format('Eye'))
ax[1].set_ylabel('Count')
plt.show()


# In[ ]:


plt.figure(figsize=(10,10))
sns.set_context("paper", 2.0, {"lines.linewidth": 4})
sns.countplot(y='Eye',hue='Inc',data=data_top_100)


# In[ ]:


plt.figure(figsize=(15,10))
sns.swarmplot(x="Inc", y="Appearances",hue='Eye', data=data_top_100)


# In[ ]:


plt.figure(figsize=(15,25))
sns.boxplot(y="Eye", x="Appearances",
            hue="Inc", palette=["m", "g"],
            data=data_top_100)


# In[ ]:


f,ax=plt.subplots(1,2,figsize=(18,9))
explode_list = [0.1] * (data_top_100[data_top_100['Inc'] == 'DC']['Hair'].unique().size-1)
data_top_100[data_top_100['Inc'] == 'DC']['Hair'].value_counts().plot.pie(explode=explode_list,autopct='%1.1f%%',ax=ax[0],shadow=True)
ax[0].set_title('DC {} Count'.format('Hair'))
ax[0].set_ylabel('Count')
explode_list = [0.1] * (data_top_100[data_top_100['Inc'] == 'Mervel']['Hair'].unique().size)
data_top_100[data_top_100['Inc'] == 'Mervel']['Hair'].value_counts().plot.pie(explode=explode_list,autopct='%1.1f%%',ax=ax[1],shadow=True)
ax[1].set_title('Mervel {} Count'.format('Hair'))
ax[1].set_ylabel('Count')
plt.show()


# In[ ]:


plt.figure(figsize=(10,10))
sns.set_context("paper", 2.0, {"lines.linewidth": 4})
sns.countplot(y='Hair',hue='Inc',data=data_top_100)


# In[ ]:


plt.figure(figsize=(15,10))
sns.swarmplot(x="Inc", y="Appearances",hue='Hair', data=data_top_100)


# In[ ]:


plt.figure(figsize=(15,25))
sns.boxplot(y="Hair", x="Appearances",
            hue="Inc", palette=["m", "g"],
            data=data_top_100)


# In[ ]:


f,ax=plt.subplots(1,2,figsize=(18,9))
explode_list = [0.1] * (data_top_100[data_top_100['Inc'] == 'DC']['Gender'].unique().size)
data_top_100[data_top_100['Inc'] == 'DC']['Gender'].value_counts().plot.pie(explode=explode_list,autopct='%1.1f%%',ax=ax[0],shadow=True)
ax[0].set_title('DC {} Count'.format('Gender'))
ax[0].set_ylabel('Count')
explode_list = [0.1] * (data_top_100[data_top_100['Inc'] == 'Mervel']['Gender'].unique().size)
data_top_100[data_top_100['Inc'] == 'Mervel']['Gender'].value_counts().plot.pie(explode=explode_list,autopct='%1.1f%%',ax=ax[1],shadow=True)
ax[1].set_title('Mervel {} Count'.format('Gender'))
ax[1].set_ylabel('Count')
plt.show()


# In[ ]:


plt.figure(figsize=(20,5))
sns.set_context("paper", 2.0, {"lines.linewidth": 4})
sns.countplot(x='Gender',hue='Inc',data=data_top_100)


# In[ ]:


data_top_100[data_top_100['Gender']=='Genderfluid Characters']


# That's right?

# In[ ]:


plt.figure(figsize=(15,10))
sns.swarmplot(x="Inc", y="Appearances",hue='Gender', data=data_top_100)


# In[ ]:


plt.figure(figsize=(20,10))
sns.boxplot(x="Gender", y="Appearances",
            hue="Inc", palette=["m", "g"],
            data=data_top_100)


# In[ ]:


f,ax=plt.subplots(1,2,figsize=(18,9))
explode_list = [0.1] * (data_top_100[data_top_100['Inc'] == 'DC']['GSM'].unique().size-1)
data_top_100[data_top_100['Inc'] == 'DC']['GSM'].value_counts().plot.pie(explode=explode_list,autopct='%1.1f%%',ax=ax[0],shadow=True)
ax[0].set_title('DC {} Count'.format('GSM'))
ax[0].set_ylabel('Count')
explode_list = [0.1] * (data_top_100[data_top_100['Inc'] == 'Mervel']['GSM'].unique().size-1)
data_top_100[data_top_100['Inc'] == 'Mervel']['GSM'].value_counts().plot.pie(explode=explode_list,autopct='%1.1f%%',ax=ax[1],shadow=True)
ax[1].set_title('Mervel {} Count'.format('GSM'))
ax[1].set_ylabel('Count')
plt.show()


# In[ ]:


plt.figure(figsize=(20,5))
sns.set_context("paper", 2.0, {"lines.linewidth": 4})
sns.countplot(x='GSM',hue='Inc',data=data_top_100)


# In[ ]:


plt.figure(figsize=(15,10))
sns.swarmplot(x="Inc", y="Appearances",hue='GSM', data=data_top_100)


# In[ ]:


plt.figure(figsize=(20,10))
sns.boxplot(x="GSM", y="Appearances",
            hue="Inc", palette=["m", "g"],
            data=data_top_100)


# In[ ]:


f,ax=plt.subplots(1,2,figsize=(18,9))
explode_list = [0.1] * (data_top_100[data_top_100['Inc'] == 'DC']['Alive'].unique().size)
data_top_100[data_top_100['Inc'] == 'DC']['Alive'].value_counts().plot.pie(explode=explode_list,autopct='%1.1f%%',ax=ax[0],shadow=True)
ax[0].set_title('DC {} Count'.format('Alive'))
ax[0].set_ylabel('Count')
explode_list = [0.1] * (data_top_100[data_top_100['Inc'] == 'Mervel']['Alive'].unique().size)
data_top_100[data_top_100['Inc'] == 'Mervel']['Alive'].value_counts().plot.pie(explode=explode_list,autopct='%1.1f%%',ax=ax[1],shadow=True)
ax[1].set_title('Mervel {} Count'.format('Alive'))
ax[1].set_ylabel('Count')
plt.show()


# In[ ]:


plt.figure(figsize=(20,5))
sns.set_context("paper", 2.0, {"lines.linewidth": 4})
sns.countplot(x='Alive',hue='Inc',data=data_top_100)


# In[ ]:


plt.figure(figsize=(15,10))
sns.swarmplot(x="Inc", y="Appearances",hue='Alive', data=data_top_100)


# In[ ]:


plt.figure(figsize=(20,10))
sns.boxplot(x="Alive", y="Appearances",
            hue="Inc", palette=["m", "g"],
            data=data_top_100)

