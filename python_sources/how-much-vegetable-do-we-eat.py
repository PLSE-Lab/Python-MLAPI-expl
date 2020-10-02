#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


# Learn form other people: How Much Sugar Do We Eat?

After watching [That Sugar Film](http://www.imdb.com/title/tt3892434/) and getting more into cooking and food in general, I thought it would be interesting to see how much of particular ingredients the people of certain countries eat in their food.

## Sugar
The first check was how much sugar a number of countries take in. Companies have been putting more and more sugar into the products we eat for a number of decades now, particularly in products like soft drinks/sodas, which isn't great for our bodies. There are some stereotypical guesses one could make about the countries that consume the most sugar, but doing some data analysis is generally more informative. 


# ## Which countries eat the most vegetable?
# 
# Interesting results, although for a number of countries the amount of data is a lot less (particularly countries like South Africa), and thus the data can be skewed. Another interesting note is the lack of any data on total sugars for Asian countries such as Japan and China. There are not many data entries for these countries either, but there are enough to make me wonder why there is no data on their sugar intake.
# 
# 

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

world_food_facts = pd.read_csv('../input/FoodFacts.csv')
world_food_facts.countries = world_food_facts.countries.str.lower()
    
def mean(l):
    return float(sum(l)) / len(l)

world_fvn = world_food_facts[world_food_facts.fruits_vegetables_nuts_100g.notnull()]
type(world_fvn)

def return_fvn(country):
    return world_fvn[world_fvn.countries == country].fruits_vegetables_nuts_100g.tolist()
    
# Get list of sodium per 100g for some countries
fr_fvn = return_fvn('france')
uk_fvn = return_fvn('united kingdom')
nd_fvn = return_fvn('netherlands') 
jp_fvn = return_fvn('japan')
print ('finish')
print (uk_fvn[0:])
print ('above ch +++++++++++++++++++++++++++++++')
print (fr_fvn[0:])
print ('above fr ++++++++++++++++++++++++++++++++')
print (nd_fvn[0:])
print ('above nd +++++++++++++++++++++++')
print (jp_fvn[0:])
print ('above jp +++++++++++++++++++++++')
countries = ['FR', 'CH']
fvn_list=[mean(fr_fvn), mean(uk_fvn)]
y_pos = np.arange(len(countries))
    
plt.bar(y_pos, fvn_list, align='center', alpha=0.5)
plt.title('Average amount of fruid veg. and nuts')
plt.xticks(y_pos, countries)
plt.ylabel('Amount of fvn')
    
plt.show()

