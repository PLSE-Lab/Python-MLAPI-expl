#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import PIL.Image as Image, PIL.ImageDraw as ImageDraw, PIL.ImageFont as ImageFont
import plotly.graph_objects as go
from scipy.stats import norm


# Thanks to Peter for a great notebook on EDA  - Check it out here:
# 
# https://www.kaggle.com/pestipeti/bengali-quick-eda

# In[ ]:


HEIGHT = 137
WIDTH = 236


# In[ ]:


def load_as_npa(file):
    df = pd.read_parquet(file)
    return df.iloc[:, 1:].values.reshape(-1 , HEIGHT , WIDTH)


# In[ ]:


def image_from_char(char):
    image = Image.new('RGB', (WIDTH, HEIGHT))
    draw = ImageDraw.Draw(image)
    myfont = ImageFont.truetype('/kaggle/input/bengalifont/HindSiliguri.ttf', 120)
    w, h = draw.textsize(char, font=myfont)
    draw.text(((WIDTH - w) / 2,(HEIGHT - h) / 2), char, font=myfont)

    return image


# In[ ]:


get_ipython().run_cell_magic('time', '', "images0 = load_as_npa('/kaggle/input/bengaliai-cv19/train_image_data_0.parquet')")


# In[ ]:


fig , ax = plt.subplots(5,5,figsize = (15,10))
ax = ax.flatten()

for i in range(25):
    ax[i].imshow(images0[i], cmap='Greys')


# In[ ]:


df_train = pd.read_csv('/kaggle/input/bengaliai-cv19/train.csv')
df_train.head()


# In[ ]:


df_train.shape


# In[ ]:


df_classmap = pd.read_csv('/kaggle/input/bengaliai-cv19/class_map.csv')
df_classmap.head()


# In[ ]:


df_classmap.shape


# 1. Grapheme Roots-

# In[ ]:


print("Unique grapheme roots:" ,df_train['grapheme_root'].nunique())


# In[ ]:


sns.set(rc={'figure.figsize':(10,10)})


# Histogram with maximum likelihood gaussian distribution fit :

# In[ ]:


sns.distplot(df_train['grapheme_root'] ,fit=norm , kde=False)


# Histogram with frequencies  -

# In[ ]:


sns.distplot(df_train['grapheme_root'] ,kde=False)


# KDE function -

# In[ ]:


sns.kdeplot(df_train['grapheme_root'] , shade=True)


# 2. Vowel Diacritics :

# In[ ]:


print("Unique vowel diacritcs : ",df_train['vowel_diacritic'].nunique())


# Histogram with frequencies :

# In[ ]:


sns.distplot(df_train['vowel_diacritic'] , kde=False )


# KDE function :

# In[ ]:


sns.kdeplot(df_train['vowel_diacritic'] , shade=True)


# Take a look at the vowel diacritics -

# In[ ]:


x = df_train['vowel_diacritic'].value_counts().sort_values().index
vowels = df_classmap[(df_classmap['component_type'] == 'vowel_diacritic') & (df_classmap['label'].isin(x))]['component']


# In[ ]:


fig, ax = plt.subplots(3, 5, figsize=(15, 10))
ax = ax.flatten()

for i in range(15):
    if i < len(vowels):
        ax[i].imshow(image_from_char(vowels.values[i]), cmap='Greys' )
        ax[i].grid(None)
        


# 3. Consonant Diacritics 

# In[ ]:


print("Unique consonant diacritcs : ",df_train['consonant_diacritic'].nunique())


# Histogram with frequencies :

# In[ ]:


sns.distplot(df_train['consonant_diacritic'] , kde=False )


# KDE function :

# In[ ]:


sns.kdeplot(df_train['consonant_diacritic'] , shade=True )


# Take a look at the consonant diacritics :

# In[ ]:


y = df_train['consonant_diacritic'].value_counts().sort_values().index
consonants = df_classmap[(df_classmap['component_type'] == 'consonant_diacritic') & (df_classmap['label'].isin(y))]['component']


# In[ ]:


fig, ax = plt.subplots(2, 5, figsize=(16, 10))
ax = ax.flatten()


for i in range(15):
    if i < len(consonants):
        ax[i].imshow(image_from_char(consonants.values[i]), cmap='Greys' )
        ax[i].grid(None)


# 

# In[ ]:


df_submission = pd.read_csv('/kaggle/input/bengaliai-cv19/sample_submission.csv')
df_submission['target'] = 0
df_submission.to_csv("submission.csv", index=False)

