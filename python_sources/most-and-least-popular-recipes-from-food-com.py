#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
pd.set_option('max_colwidth',100)


# # Load Data & Merge Data

# In[ ]:


df_interactions = pd.read_csv('../input/food-com-recipes-and-user-interactions/interactions_train.csv')
gb_interactions = df_interactions.groupby('recipe_id')['rating']
df_rating = pd.concat([gb_interactions.count(),gb_interactions.mean()],axis=1)
df_rating = pd.concat([df_rating,gb_interactions.std()],axis=1)
df_rating.columns = ['Count','Rating','Stdev']

df_rating = df_rating.sort_values(by=['Rating','Count'],ascending=[False,False])
df_recipe_details = pd.read_csv('../input/food-com-recipes-and-user-interactions/RAW_recipes.csv')
df_rating.merge(df_recipe_details[['id','name']],how='inner',left_index=True,right_on='id')[:10]
df_recipe_details = pd.read_csv('../input/food-com-recipes-and-user-interactions/RAW_recipes.csv')


# # Most popular recipes
# Ordered by rating followed by number of ratigs

# In[ ]:


df_rating = df_rating.sort_values(by=['Rating','Count'],ascending=[False,False])
df_rating.merge(df_recipe_details[['id','name']],how='inner',left_index=True,right_on='id')[:10]


# # Least popular recipes
# Ordered by rating followed by number of ratings. Only looks at receipes with 5 or more ratings.

# In[ ]:


df_rating_mt5 = df_rating[df_rating['Count'] >=2]
df_rating_mt5 = df_rating_mt5.sort_values(by=['Rating','Count'],ascending=[True,False])
df_rating_mt5.merge(df_recipe_details[['id','name']],how='inner',left_index=True,right_on='id')[:10]


# # Most divisive recipes
# Ordered by standard deviation of the rating followed by number of ratings. Only looks at receipes with 5 or more ratings.

# In[ ]:


df_rating_mt5 = df_rating[df_rating['Count'] >=5]
df_rating_mt5 = df_rating_mt5.sort_values(by=['Stdev','Count'],ascending=[False,False])
df_rating_mt5.merge(df_recipe_details[['id','name']],how='inner',left_index=True,right_on='id')[:10]


# In[ ]:




