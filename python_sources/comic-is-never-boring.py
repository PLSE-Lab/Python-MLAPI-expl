#!/usr/bin/env python
# coding: utf-8

# ![Comics](https://www.dccomics.com/sites/default/files/DCVol2Marquee_57466713405381.60938022.jpg)

# **Content:**
# * [Clean Data](#1)
# * [Heroes Gender Evolvement by Decade](#2)
# * [Align and Alive Distribution](#3)
# * [Is always bad hero dead?](#4)
# * [Transgender and Fluidgender Heroes?](#5)
# * [Who are the most appearance heroes in Marvels and DC?](#6)
# * [What good and bad heroes look like?](#7)
# 
# 
# 
# 

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='white',palette = 'Set3',context = 'talk')


# In[ ]:


dc = pd.read_csv('../input/dc-wikia-data.csv')
mar = pd.read_csv('../input/marvel-wikia-data.csv')
print(dc.shape,mar.shape)


# ## Clean Data
# 

# In[ ]:


def clean(x):
    x.name = x.name.apply(lambda x: x.split('(')[0])
    cols = ('ID','ALIGN','EYE','HAIR','SEX','ALIVE')
    for c in cols:
       x[c]=  x[c].fillna('Unknown')
       x[c]=  x[c].apply(lambda x: x.split(' ')[0])

clean(dc)


# In[ ]:


mar.columns
mar['YEAR'] = mar['Year']
clean(mar)


# In[ ]:


dc.head(2)


# ## Heroes Gender Evolvement by Decade

# **Shortcut:**
# * 1980 is a turning point, the number of comic heros is boosted after 1980
# * In 2010, no matter DC or Marvels, the number of comic heros dropped drastically.

# In[ ]:


dc.groupby('YEAR')['SEX'].value_counts().unstack().plot(figsize = (16,6))
plt.title('DC - Character Gender Evolvement')

mar.groupby('YEAR')['SEX'].value_counts().unstack().plot(figsize = (16,6))
plt.title('Marvels - Character Gender Evolvement')


# ## Align and Alive Distribution

# **Shortcut:**
# * Good Heroes are always more than Bad heroes. Don't lose faith in the life!
# * In hero's world, men are always more than women
# * No matter good or bad, man or woman, one day we all gonna die. 

# In[ ]:


plt.subplots(1,2,figsize = (18,6))
plt.subplot(121)
sns.countplot(x= 'ALIGN',hue = 'SEX',data = dc)
plt.legend(loc='upper right')
plt.subplot(122)
sns.countplot(x= 'ALIVE',hue = 'SEX',data = dc)
plt.legend(loc='upper right')


# In[ ]:


plt.subplots(1,2,figsize = (18,6))
plt.subplot(121)
sns.countplot(x= 'ALIGN',hue = 'SEX',data = mar)
plt.legend(loc='upper right')
plt.subplot(122)
sns.countplot(x= 'ALIVE',hue = 'SEX',data = mar)
plt.legend(loc='upper right')


# ## Is always bad hero dead?
# 

# In[ ]:


dead_mar = mar[mar.ALIVE == 'Deceased']
dead_dc = dc[dc.ALIVE == 'Deceased']

plt.subplots(1,2,figsize = (18,6))
plt.subplot(121)
sns.countplot(x='ALIGN',data = dead_mar)
plt.title ('Marvels - dead hero distribution')
plt.subplot(122)
sns.countplot(x='ALIGN',data = dead_dc)
plt.title ('DC - dead hero distribution')


# ## Transgender and Fluidgender Heroes?
# That is probably [the reason that comic books are attracting increasingly numbers of LGBT people](https://fivethirtyeight.com/features/women-in-comic-books/).
# 

# In[ ]:


display(dc[dc.SEX =='Transgender'][['name','YEAR']])
display(mar[mar.SEX =='Genderfluid'][['name','YEAR']])


# ![Daystar 2009](https://vignette.wikia.nocookie.net/marvel_dc/images/7/79/Daystar_001.jpg/revision/latest?cb=20130204123456)
# ![Xavin_2005](https://i.pinimg.com/originals/92/5b/02/925b02d30f35fbae2b85a3196a45e170.gif)

# ## Who are the most appearance heroes in Marvels and DC?

# In[ ]:


tmp =mar.sort_values(by = 'APPEARANCES',ascending = False)[:10][['name','SEX','APPEARANCES']]
tmp


# In[ ]:


tmp =dc.sort_values(by = 'APPEARANCES',ascending = False)[:10][['name','SEX','APPEARANCES']]
tmp


# **Finally I found two women on the list, what a surprise! **
# ![Wonder Woman and Black Canary](https://i.pinimg.com/originals/5c/f1/dd/5cf1ddc83ac59113663378fb8ee10f7a.png)

# ## What good and bad heroes look like?
#      

# In[ ]:


good_mar = mar[mar.ALIGN == 'Good'].sort_values(by = 'APPEARANCES',ascending = False)[:10]
good_dc = dc[dc.ALIGN == 'Good'].sort_values(by = 'APPEARANCES',ascending = False)[:10]
bad_mar = mar[mar.ALIGN == 'Bad'].sort_values(by = 'APPEARANCES',ascending = False)[:10]
bad_dc = dc[dc.ALIGN == 'Bad'].sort_values(by = 'APPEARANCES',ascending = False)[:10]


# In[ ]:


plt.subplots(1,2,figsize=(18,6))
plt.subplots_adjust(wspace =0.3)
plt.subplot(121)
sns.boxenplot(x='APPEARANCES', y='HAIR',data=good_dc,hue='EYE').set_title('DC-Top Appearance good hero looking')
plt.subplot(122)
sns.boxenplot(x='APPEARANCES', y='HAIR',data=bad_dc,hue='EYE').set_title('DC-Top Appearance bad hero looking')


# **Shortcut:**
# * In DC comic, good hero always has black hair and blue eyes, or brown hair.
# * And Bad hero could be in any hair color but with red eyes,or green hair.
# 
# **Let's try one !**

# In[ ]:


#bad_dc[bad_dc.HAIR == 'Green'] # JOKER 1940
good_dc[(good_dc.HAIR == 'Black') &(good_dc.EYE == 'Blue')]


# **Joker**
# ![Joker](https://images-na.ssl-images-amazon.com/images/I/71eNXTFfszL._SY679_.jpg)
# **SuperMan**
# ![SuperMan](https://img.purch.com/o/aHR0cDovL3d3dy5uZXdzYXJhbWEuY29tL2ltYWdlcy9pLzAwMC8xNTEvOTQ2L2kwMi9HYXJ5RnJhbmtfU3VwZXJtYW4uanBn)
# 

# In[ ]:


plt.subplots(1,2,figsize=(18,6))
plt.subplots_adjust(wspace =0.3)
plt.subplot(121)
sns.boxplot(x='APPEARANCES', y='HAIR',data=good_mar,hue='EYE').set_title('DC-Top Appearance good hero looking')
plt.subplot(122)
sns.boxplot(x='APPEARANCES', y='HAIR',data=bad_mar,hue='EYE').set_title('DC-Top Appearance bad hero looking')


# **Shortcut:**
# * In marvels, good hero and bad hero may both have brown hair, which is a bit ironic. 
# * But,in order to lower the difficulty of recognising bad heroes, they sometimes may in Auburn(dark red) hair or no hair at all.

# In[ ]:


bad_mar[bad_mar.HAIR == 'Auburn'] #Norman Osborn 1964
bad_mar[bad_mar.HAIR == 'Bald'] #Wilson Fisk 1967


# **Norman Osborn 1964**
# ![Norman Osborn](https://superbromovies.files.wordpress.com/2017/08/023b68d55c07fa9028dc35cfd41953b9c4ca60e1792382eb2a53957288bf2423.png?w=640)
# 
# **Wilson Fisk 1967**
# ![Wilson Fisk](https://media.comicbook.com/2016/07/kingping-romita-189138.jpg)
# 
# 

# **If you like the kernel, please do hesitate to give me a upvote!**
# 
# **Have a nice weekend!**
