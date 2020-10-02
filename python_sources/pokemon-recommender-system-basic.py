#!/usr/bin/env python
# coding: utf-8

# Today I am going to try to build a basic recommendation system for a Pokemon. I am still learning how to make my notebook clean and well-structed. If you have any suggestion, please comment below. Thank you so much. 

# In[ ]:


import numpy as np 
import pandas as pd 

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
# Any results you write to the current directory are saved as output.


# # 1. Import and clean our data

# In[ ]:


pokemon = pd.read_csv('../input/pokemon/Pokemon.csv')


# In[ ]:


pokemon.head()


# In[ ]:


pokemon.info()


# In[ ]:


Legend = pd.get_dummies(pokemon['Legendary'],drop_first=True)


# In[ ]:


pokemon.drop(['#','Legendary'],axis=1,inplace=True)
pokemon = pd.concat([pokemon,Legend],axis=1)
pokemon.rename({True:'Legend'},axis=1, inplace=True)


# In[ ]:


pokemon.head()


# In[ ]:


def type_numbering(string) : 
    if string == 'Normal' :
        return 1
    elif string== 'Fire' :
        return 2
    elif string == 'Fighting' :
        return 3
    elif string == 'Water' :
        return 4
    elif string == 'Flying' :
        return 5
    elif string == 'Grass' :
        return 6
    elif string == 'Poison' :
        return 7
    elif string == 'Electric' :
        return 8
    elif string == 'Ground' :
        return 9
    elif string == 'Psychic' :
        return 10
    elif string == 'Rock' :
        return 11
    elif string == 'Ice' :
        return 12
    elif string == 'Bug' :
        return 13
    elif string == 'Dragon' :
        return 14
    elif string == 'Ghost' :
        return 15
    elif string == 'Dark' :
        return 16
    elif string == 'Steel' :
        return 17
    elif string == 'Fairy' :
        return 18
    else :
        return 0


# In[ ]:


pokemon['Type 1'] = pokemon['Type 1'].apply(type_numbering)
pokemon['Type 2'] = pokemon['Type 2'].apply(type_numbering)


# In[ ]:


pokemon.head()


# # 2. Building our recommender system

# <p>At the very first, I am thinking about consider the pokemons as vectors. And to find the similar pokemons, we are going to calculate the Euclidean distance between them and sort to get top 10</p>
# <p>Here is the link if you do not know: [Wiki for Euclidean distance](https://en.wikipedia.org/wiki/Euclidean_distance)</p>
# <p>In numpy we can use: numpy.linalg.norm(a,b)</p>

# In[ ]:


indices = pd.Series(pokemon.index, index=pokemon['Name'])


# In[ ]:


pokematrix = pokemon.drop('Name',axis=1)


# In[ ]:


def recommendation(pkm):
    idx = indices[pkm]
    sim_scores = []
    for i in range(pokemon.shape[0]):
        sim_scores.append(np.linalg.norm(pokematrix.loc[idx]-pokematrix.loc[i]))
    sim_scores = list(enumerate(sim_scores))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=False)
    sim_scores = sim_scores[1:31]
    pkm_indices = [i[0] for i in sim_scores]
    sim_pkm = pokemon.iloc[pkm_indices].head(10)
    return sim_pkm


# In[ ]:


recommendation('Pikachu')


# <p>The result seems not as my expectation as only Voltorb is an electric pokemon.</p>
# <p>So I decide to use **Cosine Similarity** to calculate the similarity between two pokemons:</p>
# <p>Mathematically, it is defined as follows:</p>
# <p>$cosine(x,y) = \frac{x. y^\intercal}{||x||.||y||} $</p>
# 

# In[ ]:


from sklearn.metrics.pairwise import cosine_similarity


# In[ ]:


cosine_sim = cosine_similarity(pokematrix,pokematrix)


# In[ ]:


def recommendation_2(pkm):
    idx = indices[pkm]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:31]
    pkm_indices = [i[0] for i in sim_scores]    
    sim_pkm = pokemon.iloc[pkm_indices].head(10)
    return sim_pkm


# In[ ]:


recommendation_2('Pikachu')


# <p>The result is getting better, I can see more electric pokemon. But it would be even better if the system can show pokemons having same type with Pikachu on top.</p>
# <p>Let build a score for types of pokemon like below:</p>

# In[ ]:


def check_type(x,a,b):
    pkm_type_1 = x['Type 1']
    pkm_type_2 = x['Type 2']
    if (pkm_type_1 == a) and (pkm_type_2 == b):
        return 1
    elif (pkm_type_1 == a) or (pkm_type_2 == b):
        return 0.5
    else:
        return 0


# In[ ]:


def enhanced_recommendation(pkm):
    idx = indices[pkm]
    pkm_type1= pokematrix.loc[idx]['Type 1']
    pkm_type2= pokematrix.loc[idx]['Type 2']
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:31]
    pkm_indices = [i[0] for i in sim_scores]
    
    sim_pkm = pokemon.iloc[pkm_indices].copy()
    sim_pkm['sim_type'] = sim_pkm.apply(lambda x: check_type(x,pkm_type1,pkm_type2), axis=1)
    sim_pkm = sim_pkm.sort_values('sim_type', ascending=False).head(10)
    return sim_pkm


# In[ ]:


enhanced_recommendation('Pikachu')


# <p>Yes!!! The result seems quite good. 
# Elekid, Tynamo, Pichu, Raichu are quite similar to Pikachu</p>

# In[ ]:


enhanced_recommendation('Charizard')


# In[ ]:


enhanced_recommendation('Bulbasaur')


# I think it is enough for today. I will enhance the system later.
# <p>Thank you.</p>
