#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(color_codes = True)
get_ipython().run_line_magic('matplotlib', 'inline')
import os
print(os.listdir("../input"))


# In[ ]:


battle_dataset = pd.read_csv('../input/battles.csv')


# In[ ]:


char_death = pd.read_csv("../input/character-deaths.csv")


# In[ ]:


char_pred = pd.read_csv("../input/character-predictions.csv")


# In[ ]:


battle_dataset


# In[ ]:


battle_dataset.info()


# In[ ]:


battle_dataset.isnull().sum()


# In[ ]:


battle_dataset['attacker_size'].mean()


# In[ ]:


battle_dataset['defender_size'].mean()


# In[ ]:


new_battle = battle_dataset[['defender_size','attacker_size','attacker_outcome']].dropna()


# In[ ]:


new_battle.reset_index(inplace=True)


# In[ ]:


new_battle = new_battle.iloc[:,1:]


# In[ ]:


new_battle


# In[ ]:


sns.pairplot(new_battle, hue='attacker_outcome')


# In[ ]:


battle_dataset[battle_dataset['attacker_outcome'] == 'loss']


# In[ ]:


battle_dataset.info()


# In[ ]:


sns.countplot(battle_dataset['year'])


# In[ ]:


battle_dataset.groupby('year')['attacker_outcome'].value_counts().plot(kind = 'bar')


# In[ ]:


battle_dataset.groupby('attacker_outcome')['year'].value_counts().plot(kind = 'bar')


# In[ ]:


plt.figure(figsize=(12,6))
sns.countplot(battle_dataset['attacker_king'])


# In[ ]:


plt.figure(figsize=(15,6))
sns.countplot(battle_dataset['defender_king'])


# In[ ]:


battle_dataset.groupby('attacker_king')['attacker_outcome'].value_counts().plot(kind = 'bar')


# In[ ]:


battle_dataset.groupby('attacker_king')['attacker_outcome'].value_counts()


# In[ ]:


battle_dataset['attacker_king'].value_counts()


# In[ ]:


battle_dataset.groupby('attacker_king')['defender_king'].value_counts().plot(kind = 'bar')


# In[ ]:


battle_dataset.groupby(['attacker_king','defender_king'])['attacker_outcome'].value_counts().plot(kind = 'bar')


# In[ ]:


battle_dataset.groupby(['attacker_king','defender_king'])['attacker_outcome'].value_counts()


# In[ ]:


battle_dataset['battle_type'].value_counts().plot(kind = 'bar')


# In[ ]:


battle_dataset.groupby('battle_type')['attacker_outcome'].value_counts().plot(kind = 'bar')


# In[ ]:


battle_dataset.groupby(['attacker_king','battle_type'])['attacker_outcome'].value_counts()


# In[ ]:


battle_dataset.groupby('summer')['attacker_outcome'].value_counts()


# In[ ]:


battle_dataset['region'].value_counts().plot(kind = 'bar')


# In[ ]:


battle_dataset['location'].value_counts().plot(kind = 'bar')


# In[ ]:


battle_dataset.groupby('attacker_outcome')['major_capture'].sum().plot(kind = 'bar')


# In[ ]:


battle_dataset.groupby('attacker_outcome')['major_death'].sum().plot(kind = 'bar')


# In[ ]:


battle_dataset.groupby('attacker_outcome')['major_death'].sum()


# In[ ]:


char_death.head()


# In[ ]:


char_death.info()


# In[ ]:


char_death.isnull().sum()


# In[ ]:


battle_dataset.groupby('year')[['major_death','major_capture']].sum().plot(kind = 'bar')


# In[ ]:


battle_dataset.groupby('attacker_outcome')[['major_death','major_capture']].sum().plot(kind = 'bar')


# In[ ]:


battle_dataset.groupby(['attacker_king','year'])[['attacker_1','attacker_2','attacker_3','attacker_4']].count()


# In[ ]:


a = battle_dataset.groupby(['defender_king','year'])[['defender_1','defender_2','defender_3','defender_4']].count()


# In[ ]:


a.index


# In[ ]:


sns.heatmap(a)


# In[ ]:


sns.heatmap(battle_dataset.groupby(['attacker_king','year'])[['attacker_1','attacker_2','attacker_3','attacker_4']].count())


# In[ ]:


char_death.head()


# In[ ]:


### Use below code and add similar commands in lambda to replace duplicate Allegiances with House name
char_death['Allegiances'] = char_death['Allegiances'].apply(lambda x : 'House Martell' if(x == 'Martell') else 'House Stark' if(x=='Stark') else 'House Targaryen' if(x=='Targaryen') else 'House Tully' if(x=='Tully') else 'House Tyrell' if(x=='Tyrell') else x)


# In[ ]:


char_death['Gender'].value_counts().plot(kind = 'bar')


# In[ ]:


char_death['Allegiances'].value_counts().plot(kind = 'bar')


# In[ ]:


char_death[char_death['Death Year'].notnull()]['Allegiances'].value_counts().plot(kind = 'bar')


# In[ ]:


plt.figure(figsize=(12,5))
char_death[char_death['Death Year'].notnull()].groupby('Death Year')['Allegiances'].value_counts().plot(kind = 'bar')


# In[ ]:


char_death[char_death['Death Year'].notnull()]['Death Year'].value_counts()


# In[ ]:


char_death.groupby('Gender')['Allegiances'].value_counts().plot(kind = 'bar')


# In[ ]:


char_death.groupby('Allegiances')['Gender'].value_counts().plot(kind = 'bar')


# In[ ]:




