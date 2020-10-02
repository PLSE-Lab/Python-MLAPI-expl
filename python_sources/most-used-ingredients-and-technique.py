#!/usr/bin/env python
# coding: utf-8

# # What mostly used in Indonesian Recipes
# 
# In this set of code i'll show how to seperate each words in ingredients column and steps column, the objective is to see what is the most used ingredient and technique in Indonesian recipes.

# In[ ]:


import numpy as np
import pandas as pd

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# 1. Read All the data from dataset using read_csv from Panda

# In[ ]:


ay = pd.read_csv('/kaggle/input/indonesian-food-recipes/dataset-ayam.csv')
ik = pd.read_csv('/kaggle/input/indonesian-food-recipes/dataset-ikan.csv')
ka = pd.read_csv('/kaggle/input/indonesian-food-recipes/dataset-kambing.csv')
sa = pd.read_csv('/kaggle/input/indonesian-food-recipes/dataset-sapi.csv')
ta = pd.read_csv('/kaggle/input/indonesian-food-recipes/dataset-tahu.csv')
te = pd.read_csv('/kaggle/input/indonesian-food-recipes/dataset-telur.csv')
tem = pd.read_csv('/kaggle/input/indonesian-food-recipes/dataset-tempe.csv')
ud = pd.read_csv('/kaggle/input/indonesian-food-recipes/dataset-udang.csv')


# 2. Make all CSVs as dataframes 

# In[ ]:


ayam = pd.DataFrame(ay)
ikan = pd.DataFrame(ik)
kambing = pd.DataFrame(ka)
sapi = pd.DataFrame(sa)
tahu = pd.DataFrame(ta)
telur = pd.DataFrame(te)
tempe = pd.DataFrame(tem)
udang = pd.DataFrame(ud)


# 3. Concat all of dataframes to make it easier to analyze all of them.

# In[ ]:


indofood = pd.concat([ayam, ikan, kambing, sapi, tahu, telur, tempe, udang])
indofood = pd.DataFrame(indofood)


# 4. To make sure all characters inside the columns are string we have to conver it to string using astype(str), after that we have to eliminate all symbols inside the columns using str.replace so we can count all words inside of the columns.

# In[ ]:


indofood['Ingredients'] = indofood['Ingredients'].astype(str)
indofood['Ingredients'] = indofood['Ingredients'].str.replace('[^a-zA-Z]', ' ')
indofood['Steps'] = indofood['Steps'].astype(str)
indofood['Steps'] = indofood['Steps'].str.replace('[^a-zA-z]',' ')


# 5. Next step is to count all of the words inside the columns using value_counts()

# In[ ]:


countingredients = indofood['Ingredients'].str.split(expand=True).stack().value_counts()
ingredients = pd.DataFrame({'ingredients':countingredients.index, 'value':countingredients.values})
countsteps = indofood['Steps'].str.split(expand=True).stack().value_counts()
steps = pd.DataFrame({'steps':countsteps.index,'value':countsteps.values})


# 6. Finally, we can see the most used ingridients and technique in Indonesian Recipe

# In[ ]:


ingredients


# In[ ]:


steps


# My method is not perfect, i still have to extract the data manually from the ingredients and steps table, but we can see the that the most used ingridients is bawang and most used technique is goreng.
# 
# If i have list of Indonesian food ingredients probably the right method is to search the ingredients from the list to the column.
# 
# Im open to critique and suggestion feel free to message me or comment down below.
