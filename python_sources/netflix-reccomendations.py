#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


netflix_data=pd.read_csv('../input/netflix-shows/netflix_titles.csv')
netflix_data.head()


# In[ ]:


netflix_data.isnull().sum()


# In[ ]:


netflix_data.fillna('Unknown', inplace=True)


# In[ ]:


netflix_data.isnull().sum()


# In[ ]:


pie_df=netflix_data.groupby('type', axis=0).count()
pie_df['title'].plot(kind='pie',
                     figsize=(7,8),
                     autopct='%1.1f%%',
                    pctdistance=1.12,
                    explode=(0.1,0),
                    colors=['lightcoral', 'darkblue'],
                    labels=None)
plt.legend(labels=pie_df.index, loc='upper left')
plt.title('Distribution of TV shows and Movies')
plt.show()


# In[ ]:


bar_conti=netflix_data.groupby('country').count()
bar=bar_conti.nlargest(10, 'show_id')
bar['show_id'].plot(kind='bar', figsize=(11,15))
plt.xlabel('Countries')
plt.ylabel('Number of Movies/TV shows')
plt.show()


# ### This shows highest shows being telecasted in US

# In[ ]:


genere=netflix_data.groupby('listed_in').count()
genere.sort_values(by='show_id', inplace=True, ascending=False)
genere_top=genere.head(20)


# In[ ]:


genere_top['show_id'].plot(kind='barh', figsize=(11,15))
plt.xlabel('Number of movies/tv shows')
plt.ylabel('Genere')
plt.show


# In[ ]:



netflix_data.set_index('title', inplace=True)


# In[ ]:


def get_reccomendation(liked):
        type=netflix_data.loc[liked,'type']
        country=netflix_data.loc[liked,'country']
        genere=netflix_data.loc[liked,'listed_in']

        req=netflix_data[netflix_data['country']==country]
        required=req[req['listed_in']==genere]
        req1=required[required['type']==type]
        return(req1.index.tolist())


# ### For example, you like Friends. The reccomended shows for you will be:

# In[ ]:


liked='Friends'
get_reccomendation(liked)


# ### Say you like PK movie, then the reccomendation will be:

# In[ ]:


like='PK'
get_reccomendation(like)


# In[ ]:




