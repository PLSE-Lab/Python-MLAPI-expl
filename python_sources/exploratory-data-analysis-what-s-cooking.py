#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
from pprint import pprint
from time import time
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_context('talk')


# In[ ]:


train = pd.read_json('../input/train.json', orient='columns')
test = pd.read_json('../input/test.json', orient='columns')


# ## Distribution of cuisines in the data set

# In[ ]:


f, ax = plt.subplots(figsize=(5,6))
sns.countplot(y = 'cuisine', 
                   data = train,
                  order = train.cuisine.value_counts(ascending=False).index)


# ## Analysis of ingredients

# In[ ]:


ingredients_individual = Counter([ingredient for ingredient_list in train.ingredients for ingredient in ingredient_list])
ingredients_individual = pd.DataFrame.from_dict(ingredients_individual,orient='index').reset_index()


# In[ ]:


ingredients_individual = ingredients_individual.rename(columns={'index':'Ingredient', 0:'Count'})


# ### Frequency of Ingredients 

# ### Frequency of each ingredient

# In[ ]:


ingredients_individual.sort_values('Count', ascending = False)['Count'].describe()


# ### Most Common Ingredients

# In[ ]:


f, ax = plt.subplots(figsize=(15,10))
sns.barplot(x = 'Count', 
            y = 'Ingredient',
            data = ingredients_individual.sort_values('Count', ascending=False).head(20))


# ### Least Common Ingredients

# In[ ]:


ingredients_individual.sort_values('Count', ascending=True).head(20)


# ### Distribution of number of ingredients in meals

# In[ ]:


f, ax = plt.subplots(figsize=(15,10))
sns.barplot(x='number_ingredients_meal',
            y='number_meals',
            data= (train.ingredients.map(lambda l: len(l))
                    .value_counts()
                    .sort_index()
                    .reset_index()
                    .rename(columns={'index':'number_ingredients_meal', 'ingredients':'number_meals'}))
            )


# ### Boxplots for numer of ingredients per cuisine
# There seems to be no cuisines that use far less or more ingredients per meal.

# In[ ]:


f, ax = plt.subplots(figsize=(32,15))
sns.boxplot(x='cuisine',
            y='number_ingredients',
            data= (pd.concat([train.cuisine,train.ingredients.map(lambda l: len(l))], axis=1)
                    .rename(columns={'ingredients':'number_ingredients'}))
            )

