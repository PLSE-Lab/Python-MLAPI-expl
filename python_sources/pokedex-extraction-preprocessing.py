#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[ ]:


import requests
from bs4 import BeautifulSoup
import re
import pandas as pd
import numpy as no
import lxml


# In[ ]:


Number = []
Name = []
Type = []
Weakness = []
HP = []
Attack = []
Defense = []
Special_Attack = []
Special_Defense = []
Speed = []


# In[ ]:


title = '/us/pokedex/bulbasaur'


# In[ ]:


while True:
    data = requests.get(f'https://www.pokemon.com{title}')
    data_bs = BeautifulSoup(data.text)

    data_name_tag =data_bs.find_all('div',class_='pokedex-pokemon-pagination-title')
    for i in data_name_tag:
        name = re.findall(r'\b[A-Z]\w+',i.text)
        number = re.findall(r'[#](\d+)',i.text)
    Name.append(''.join(name))
    Number.append(''.join(number))
    data_type_tag =data_bs.find('div',class_='dtm-type')
    each_type=[]
    for j in data_type_tag.find_all('a'):
        each_type.append(j.text)
    Type.append(each_type)
    data_weak_tag =data_bs.find('div',class_='dtm-weaknesses')
    weak=[]
    for k in data_weak_tag.find_all('a'):
        weakness = re.findall(r'[A-Z]\w+',k.text)
        weak.append(''.join(weakness))
    Weakness.append(weak)
    data_stats_tag =data_bs.find_all('li',class_='meter')
    stats =[]
    for l in data_stats_tag:
        stats.append(l['data-value'])
    HP.append(stats[0])
    Attack.append(stats[1])
    Defense.append(stats[2])
    Special_Attack.append(stats[3])
    Special_Defense.append(stats[4])
    Speed.append(stats[5])
    next_pok = data_bs.find_all('a',class_='next')
    for d in next_pok:
        title = d['href']
    print(title)
    if title =='/us/pokedex/bulbasaur':
        break
    
    


# In[ ]:


pokedex = pd.DataFrame({'Number':Number,'Name':Name,'Type':Type,'Weakness':Weakness,'HP':HP,'Attack':Attack,'Defense'
                       :Defense,'Special_Attack':Special_Attack,'Special_Defense':Special_Defense,'Speed':Speed})
pokedex.head()


# In[ ]:


pokedex.info()


# In[ ]:


pokedex['Number'] = pokedex['Number'].astype(int)


# In[ ]:


pokedex['HP'].unique()


# In[ ]:


pokedex['HP'] = pokedex['HP'].astype(int)


# In[ ]:


pokedex['Attack'].unique()


# In[ ]:


pokedex['Attack'] = pokedex['Attack'].astype(int)


# In[ ]:


pokedex['Defense'].unique()


# In[ ]:


pokedex['Defense'] = pokedex['Defense'].astype(int)


# In[ ]:


pokedex['Special_Attack'].unique()


# In[ ]:


pokedex['Special_Attack'] = pokedex['Special_Attack'].astype(int)


# In[ ]:


pokedex['Special_Defense'].unique()


# In[ ]:


pokedex['Special_Defense'] = pokedex['Special_Defense'].astype(int)


# In[ ]:


pokedex['Speed'].unique()


# In[ ]:


pokedex['Speed'] = pokedex['Speed'].astype(int)


# In[ ]:


pokedex.info()


# In[ ]:


pokedex.to_csv('./Pokedex.csv',index=False)

