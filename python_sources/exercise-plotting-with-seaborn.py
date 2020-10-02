#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import seaborn as sns

reviews = pd.read_csv('../input/wine-reviews/winemag-data_first150k.csv', index_col=0)
reviews.head()


# In[ ]:


sns.countplot(reviews['points'])


# In[ ]:


sns.kdeplot(reviews.query('price < 200').price)


# In[ ]:


reviews[reviews['price']<200].price.value_counts().sort_index().plot.line()


# In[ ]:


sns.kdeplot(reviews[reviews['price'] < 200].loc[:, ['price', 'points']].dropna().sample(5000))


# In[ ]:


sns.distplot(reviews['points'], bins=10, kde=False)


# In[ ]:


sns.jointplot(x='price', y='points', data=reviews[reviews['price'] < 100])


# In[ ]:


sns.jointplot(x='price', y='points', data=reviews[reviews['price'] < 100], kind='hex', 
              gridsize=20)


# In[ ]:


df = reviews[reviews.variety.isin(reviews.variety.value_counts().head(5).index)]

sns.boxplot(
    x='variety',
    y='points',
    data=df
)


# In[ ]:


sns.violinplot(
    x='variety',
    y='points',
    data=reviews[reviews.variety.isin(reviews.variety.value_counts()[:5].index)]
)


# In[ ]:


reviews.head()


# In[ ]:


pokemon = pd.read_csv("../input/pokemon/Pokemon.csv", index_col=0)
pokemon.head()


# In[ ]:


sns.countplot(pokemon['Generation'])


# In[ ]:


sns.distplot(pokemon['HP'])


# In[ ]:


sns.jointplot(x='Attack', y='Defense', data=pokemon)


# In[ ]:


sns.jointplot(x='Attack', y='Defense', data=pokemon, kind='hex')


# In[ ]:


sns.kdeplot(pokemon['HP'], pokemon['Attack'])


# In[ ]:


sns.boxplot(x='Legendary', y='Attack', data=pokemon)


# In[ ]:


sns.violinplot(x='Legendary', y='Attack', data=pokemon)

