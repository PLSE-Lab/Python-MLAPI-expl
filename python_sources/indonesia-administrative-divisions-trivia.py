#!/usr/bin/env python
# coding: utf-8

# Indonesia is a country located in Southeast Asia with more than seventeen thousand islands and with over 261 million people, it is the world's 4th most populous country.
# 
# 
# 
# 
# 
# Indonesia is divided into provinces **(Indonesian: Provinsi)**. Provinces are made up of regencies **(Indonesian: Kabupaten)** and cities (Indonesian: Kota). Regencies and cities are made up of districts **(Indonesian: Kecamatan)**. And finally districts consist of villages **(Indonesian: Desa)**.
# 
# [https://en.wikipedia.org/wiki/Subdivisions_of_Indonesia](http://)
# 
# In this kernel, we will do simple data analysis:
# 0. Data exploration
# 1. Which word is the most common word used for village name?
# 2. Distribution of "Kabupaten" and "Kota" across all provinces (In Progress).
# 3. Implementation of Geography Map in Python (TBA). 

# 
# # **0. Data exploration**

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from os import path
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
desa=pd.read_csv("../input/desa.csv")
kecamatan=pd.read_csv("../input/kecamatan.csv")
kabupaten=pd.read_csv("../input/kabupaten.csv")
provinsi=pd.read_csv("../input/provinsi.csv")
# Any results you write to the current directory are saved as output.


# From the dataset, we can find that there are 4 csv files, which are respectively for provinsi, kabupaten, kecamatan, and desa. Let's have a quick look on each file.

# In[ ]:


print(desa.info())
desa.columns=['desa_id','kecamatan_id','desa_name']
desa.head()


# From desa.info(), we know that there are 3 columns, which are 'code', 'parent_code' and 'name'. We will notice later that the remaining files will have the same columns name.

# In[ ]:


print(kecamatan.info())
kecamatan.columns=['kecamatan_id','kabupaten_id','kecamatan_name']
kecamatan.head()


# In[ ]:


print(kabupaten.info())
kabupaten.columns=['kabupaten_id','provinsi_id','kabupaten_name']
kabupaten.head()


# In[ ]:


print(provinsi.info())
provinsi.columns=['provinsi_id','parent_code','provinsi_name']
provinsi


# We can find out that each file 'parent_code' and 'code' has relationship as follows:
# - 'code' in provinsi.csv = 'parent_code' in kabupaten.csv
# - 'code' in kabupaten.csv = 'parent_code' in kecamatan.csv
# - 'code' in kecamatan.csv = 'parent_code' in desa.csv 
# 
# In order to consolidate all information in single dataframe, we need to merge all these dataframe in each files.
# Columns names have been renamed for easier merging.

# In[ ]:


df=pd.merge(desa, kecamatan, left_on='kecamatan_id', right_on='kecamatan_id')
df.head()


# In[ ]:


df=pd.merge(df, kabupaten, left_on='kabupaten_id', right_on='kabupaten_id')
df.head()


# In[ ]:


df=pd.merge(df, provinsi, left_on='provinsi_id', right_on='provinsi_id')
df.head()


# # **1. Which word is the most common word used for village name?**

# In[ ]:


common_word=pd.Series(' '.join(df['desa_name']).split()).value_counts()[:100]
common_word


# We can find among the list of common words, there are several one character (one of them is not even letter, "/"). Let's have a further look in our data.

# In[ ]:


df[df['desa_name'].str.contains(" I ")]


# We can find the following
# - Some of the names are not properly recorded and is shown as letters with space. (i.e "B I N J A I" instead of "BINJAI")
# - Some of the names have alias and separated with character "/"
# 
# Currently I don't have enough knowledge to filter/rectify these irregular data and we will ignore this error for now.

# Another interesting finding to take note is that some of the common words are "II" and "III"

# In[ ]:


df[df['desa_name'].str.contains(" II ")]


# In[ ]:


df[df['desa_name'].str.contains(" III ")]


# It seems that desa naming convention is also based on numbers, displayed in Roman numeral (I, II, III, IV, V, etc)

# Let us go back to our main focus, which is finding most common word in village name. 
# ![](http://)We will exclude 'I' and 'II' because they are not words and we will make word cloud and plot of 30 most common words against word count.

# In[ ]:


common_word=pd.Series(' '.join(df['desa_name']).split()).value_counts()[:32]
common_word=common_word.drop(labels=['I','II'])
common_word.plot(kind='barh',figsize=(30,20), fontsize=20)


# In[ ]:


# Start with one review:
text = " ".join(review for review in df.desa_name)

# Create and generate a word cloud image:
wordcloud = WordCloud().generate(text)

# Display the generated image:
plt.figure(figsize=(15,10))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()


# Now let us list up the table with the English meaning as well.

# In[ ]:


meaning=['Glorious','New','Cape','River','Rock','West','East','Essence','Mountain','River Mouth'
         ,'Majestic','Field','Coral','Field (Acehnese)','South','Like','North','Great','Central','Source'
         ,'Prosperous','Hole','Water','Island','Hill','Bay','Lively (Javanese)','Village','City','Intersection']
pd.DataFrame({'name':common_word.index, 'count':common_word.values, 'meaning':meaning})


# From above, we can conclude common words are categorized as follows:
# 1. Popular words with good meaning such as 'Jaya', 'Baru', 'Sari', etc.
# 2. Another common words are North, South, East, West and Central, indicating village name is part of district name.
# 3. Words describing nature such as 'Tanjung', 'Batu', 'Gunung', etc.
# 4. Words describing the village itself such as 'Kampung', 'Kota', 'Simpang'

#  # **2. Distribution of "Kabupaten" and "Kota" across all provinces.**

# In[ ]:


kabupaten.info()


# In[ ]:


kabupaten[kabupaten['kabupaten_name'].str.startswith('KOTA')]


# In[ ]:


kabupaten[kabupaten['kabupaten_name'].str.startswith('KABUPATEN')]


# In[ ]:


pd.Series(' '.join(kabupaten['kabupaten_name']).split()).value_counts()[:100]

