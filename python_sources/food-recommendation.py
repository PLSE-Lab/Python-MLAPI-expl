#!/usr/bin/env python
# coding: utf-8

# # Food Recommendation

# I have an idea that we can give food recommendation to the users based on their real time locations because  food choices are different from country to courty and food which is easy to get in country A may be hard to get in country B. As a result, I come up with the idea to give food recommendation to users based on their countries.
# For example, we can have different ranking criterias for food such as high-protein, high-fiber, low-calorie and etc. Next, I give an example on choosing out the food with highest-protein. 

# ## High Protein
# 

# In[62]:


get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

world_food_facts = pd.read_csv('../input/en.openfoodfacts.org.products.tsv','\t')
world_food_facts.countries = world_food_facts.countries.str.lower()

world_protein = world_food_facts[world_food_facts.proteins_100g.notnull()]

def return_protein(country):
    return world_protein[world_protein.countries == country][['product_name','proteins_100g']]

# Say if we want to get highest protein food in Canada
cn_protein_food = return_protein('canada')
cn_protein_food = cn_protein_food.sort_values(['proteins_100g'], ascending=[0])
top_cn_protein_food=cn_protein_food[0:10]
# top_cn_protein_food is what we want :


# In[ ]:





# 

# 
