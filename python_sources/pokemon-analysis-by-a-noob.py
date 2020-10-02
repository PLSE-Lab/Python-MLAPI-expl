#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[ ]:


pokemon=pd.read_csv('../input/pokemon.csv')
pokemon.drop_duplicates('#',keep='first',inplace=True)
pokemon = pokemon.replace(np.nan, 'None', regex=True)


# In[ ]:


pokemon.head()


# In[ ]:


pokemon.describe()


# In[ ]:


import seaborn as sns
sns.set(rc={'figure.figsize':(11.7,8.27)})


# In[ ]:


sns.countplot(pokemon['Generation'])


# In[ ]:


pokemon.Generation.value_counts()


# Pokemon Generation overviwe

# In[ ]:


(pokemon['Generation'].value_counts().sort_index()).plot.bar()
plt.show()


# Legendary and non Legendary overview

# In[ ]:


color=['gold','silver']
leg=pokemon[pokemon['Legendary']==True]
nonleg=pokemon[pokemon['Legendary']==False]
plt.pie([leg["Name"].count(),nonleg['Name'].count()],colors=color,labels=['Legendary','Non Legendary'])
plt.show()


# In[ ]:


sns.catplot(
y='Type 1',
data=pokemon,
kind='count',
)


# In[ ]:


sns.catplot(
y='Type 2',
data=pokemon,
kind='count',
)


# Relation between type1 and type 2

# In[ ]:


sns.heatmap(
    pokemon.groupby(['Type 1', 'Type 2']).size().unstack(),
    linewidths=1,
    annot=True,
)


# relation between difernt properties with generation and legendary

# In[ ]:


sns.heatmap(
    pokemon[['Attack','Defense','HP','Sp. Atk','Sp. Def','Speed','Generation','Legendary']].corr(),
    linewidths=1,
    annot=True,
)


# In[ ]:




