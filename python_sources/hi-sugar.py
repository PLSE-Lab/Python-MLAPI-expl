#!/usr/bin/env python
# coding: utf-8

# # Hello Sugar!
# 
# #### Finding mean Sugar consumption for all or a selected group of Countries
# 
# Ok guys, this is my attempt to show you guys a much easier way of finding the mean sugar consumption. I am a beginner and a novice, so bear with me. Hope its helpful!

# In[ ]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import pandas as pd
import csv as csv


# In[ ]:


foodFacts = pd.read_csv('../input/FoodFacts.csv', header=0) 


# In[ ]:


#Pre processing and exploring the data
sugar = foodFacts
sugar = sugar[['product_name','quantity','countries','sugars_100g','additives_n','nutrition_score_uk_100g']]
sugar = sugar[sugar.sugars_100g.notnull()]
sugar.countries = sugar.countries.str.lower() 
sugar = sugar.sort_values(by='sugars_100g', ascending = False) 
#Notice we have sugar as a product; on the other extreme we have some products with 0 sugar. Let's leave them for now!  
sugar.head()


# In[ ]:


#This will give you sugar mean for all countries. We have some repeats, but you get the trick.
sugarMean = sugar.groupby(['countries']).sugars_100g.mean().order(ascending=False)
sugarMean.head()


# In[ ]:


#What if you want this only for a select group of Countries. Let's do that too!
sugarSelect = sugar[sugar.countries.isin(['united states', 'france', 'united kingdom','south africa'])]
sugarSelect.groupby(['countries']).sugars_100g.mean().order(ascending=False)


# In[ ]:




