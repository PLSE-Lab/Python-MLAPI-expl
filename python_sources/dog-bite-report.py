#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[ ]:


dataset = pd.read_csv('../input/animal-bites/Health_AnimalBites.csv')


# In[ ]:


dataset.tail()


# In[ ]:


species = dataset.SpeciesIDDesc
species = species.dropna() 
speciesOfAnimal = species.unique()
print(speciesOfAnimal)


# In[ ]:


animal_list = []
for  i in speciesOfAnimal:
    animal_list.append(len(species[species==i]))
print(animal_list)


# In[ ]:


import seaborn as sns
count = dataset.BreedIDDesc.value_counts()
plt.figure(figsize=(15,8))
ax = sns.barplot(x=count[0:10].index,y=count[0:10])
plt.xticks(rotation=20)
plt.ylabel("Number of Bite")
plt.savefig('graph.png')
print(count[0:10].index)


# In[ ]:




