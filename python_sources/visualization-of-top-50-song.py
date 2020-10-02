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


import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings("ignore")
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns 


# In[ ]:


import numpy as np


# In[ ]:


shu =pd.read_csv('../input/top50spotify2019/top50.csv',encoding='ISO-8859-1',index_col=0)


# In[ ]:


shu.head(5)


# In[ ]:


from matplotlib import cm


# In[ ]:


sns.FacetGrid(shu,height=10).map(sns.distplot,'Beats.Per.Minute').add_legend()
plt.title("pdf of Bpm")
plt.ylabel("probablity")
plt.grid()
plt.show()


# In[ ]:


sns.FacetGrid(shu,height=10).map(sns.distplot,'Energy',bins=np.linspace(0,100,50)).add_legend()
plt.title("pdf of energy")
plt.ylabel("probablity")
plt.grid()
plt.show()


# In[ ]:


shu['Artist.Name'].value_counts().plot(kind='bar',figsize=(10,8),colormap=cm.get_cmap('ocean'))


# In[ ]:


sns.FacetGrid(shu,height=10).map(sns.distplot,'Danceability',bins=np.linspace(0,100,50)).add_legend()
plt.title("pdf of danceabilty")
plt.ylabel("probablity")
plt.grid()
plt.show()


# In[ ]:


wordcloud=WordCloud(width=1000,height=600,max_font_size=200,max_words=150,background_color='black').generate("".join(shu['Artist.Name']))
plt.figure(figsize=[10,10])
plt.imshow(wordcloud,interpolation='bilinear')
plt.axis("off")
plt.show()


# In[ ]:


wordcloud=WordCloud(width=1000,height=600,max_font_size=200,max_words=150,background_color='black').generate("".join(shu['Track.Name']))
plt.figure(figsize=[10,10])
plt.imshow(wordcloud,interpolation='bilinear')
plt.axis("off")
plt.show()


# In[ ]:


shu.info()


# In[ ]:


#bit-variante
sns.set_style("whitegrid")
sns.FacetGrid(shu,height=6).map(plt.scatter,'Beats.Per.Minute','Popularity').add_legend()
plt.show()


# In[ ]:


sns.set_style("whitegrid")
sns.FacetGrid(shu,height=6).map(plt.scatter,'Energy','Popularity').add_legend()
plt.show()


# In[ ]:


sns.set_style("whitegrid")
sns.FacetGrid(shu,height=5).map(plt.scatter,'Danceability','Popularity').add_legend()
plt.show()


# In[ ]:


shu.isnull().sum()


# In[ ]:


shu.rename(columns={'Track.Name':'Track_Name','Artist.Name':'Artist_Name','Beats.Per.Minute':'Beats_Per_Minute','Loudness..dB..':'Loudness_dB','Valence.':'Valence','Length.':'Length','Acousticness..':'Acousticness','Speechiness.':'Speechiness'},inplace=True)


# In[ ]:


shu.head()


# In[ ]:


data= shu.groupby('Artist_Name')


# In[ ]:


data.first()


# In[ ]:


data.get_group('Ed Sheeran')


# In[ ]:


data_1 = shu.groupby('Popularity')


# In[ ]:


data_1.first().max()


# In[ ]:


data_2 = shu.groupby(['Genre','Popularity'])


# In[ ]:


data_2.first()


# In[ ]:


tag="Shawn Mendes"
shu['relevent']=shu["Artist_Name"].fillna("").apply(lambda x:1 if tag.lower() in x.lower() else 0)
small = shu[shu['relevent']==1]
small[["Track_Name","Genre","Popularity"]]


# In[ ]:


tag="Ed Sheeran"
shu['relevent']=shu["Artist_Name"].fillna("").apply(lambda x:1 if tag.lower() in x.lower() else 0)
small = shu[shu['relevent']==1]
small[["Track_Name","Genre","Popularity"]]


# In[ ]:


small =shu.sort_values("Beats_Per_Minute",ascending=True)
small =small[small['Energy']!=""]
small[["Track_Name","Beats_Per_Minute"]][:20]


# In[ ]:


small =shu.sort_values("Energy",ascending=True)
small =small[small['Beats_Per_Minute']!=""]
small[["Track_Name","Energy"]][:20]


# In[ ]:




