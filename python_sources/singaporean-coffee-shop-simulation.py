#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# ## Intro
# I am a coffee lover. Any kind of coffee, from American, Japanese, Italian, Southeast Asian, and others. Graduated from MIT, I will open a coffee shop in Singapore beside my work as ML engineer. The menus was listed in the data below.
# 
# The society likes coffee and tea very much. I was wondering why they are much smile. Everyone like my coffee, but we have limited stock.
# 
# People order our menu with Singlish or English, so we need to adjust.

# In[ ]:


menu = pd.read_csv('/kaggle/input/sg-kopi/kopitiam.csv')
menu


# ## 1. Extracting ingredients
# ###### Forget NLP, is hard, just use regex split 

# In[ ]:


import re
subs = []
leng = []
for i in range(menu['English'].shape[0]):
    a = menu['English'][i]
    sep1 = 'without'
    ingredient = a.split(sep1, 1)[0]
    ingredient = re.split('with | but | and|,',ingredient)
    print(i)
    print(menu['Singlish'][i], ingredient, len(ingredient))
    subs.append(ingredient)
    leng.append(len(ingredient))
    


# In[ ]:


contain = pd.DataFrame(subs, columns = ['Ingredient 1','Ingredient 2','Ingredient 3','Ingredient 4','Ingredient 5'])
length  = pd.DataFrame(leng, columns = ['Ingredient count'])
contain['Ingredient 1'] = contain['Ingredient 1'].str.strip()
contain['Ingredient 2'] = contain['Ingredient 2'].str.strip()
contain['Ingredient 3'] = contain['Ingredient 3'].str.strip()
contain['Ingredient 4'] = contain['Ingredient 4'].str.strip()
contain['Ingredient 5'] = contain['Ingredient 5'].str.strip()


# In[ ]:


menu = menu.join(contain)
menu = menu.join(length)


# In[ ]:


menu = menu.drop('Source',axis = 1)


# In[ ]:


print(menu['Ingredient 1'].value_counts(), '\n\n')
print(menu['Ingredient 2'].value_counts(), '\n\n')
print(menu['Ingredient 3'].value_counts(), '\n\n')
print(menu['Ingredient 4'].value_counts(), '\n\n')
print(menu['Ingredient 5'].value_counts(), '\n\n')


# In[ ]:


menu.fillna(value=pd.np.nan, inplace=True)


# In[ ]:


menu.head(5)


# ## 2. Set Price
# 
# We will set price based on how much ingredients used in beverages. Because I am generous, I will set each ingredient for 1 SGD. 

# In[ ]:


menu['price (SGD)'] = menu['Ingredient count']


# In[ ]:


menu.head(5)


# ## 3. Buy Ingredient Stock
# 
# To operate, we need to buy the ingredient, and it will be used for selling. I assume there will be direct restock if the inventory was empty for each ingredient, because we won't disappoint the customer. 
# 
#     For 1st ingredient, 100 stock
#     For 2nd ingredient, 50 stock
#     For 3rd ingredient, 30 stock
#     For 4th ingredient, 20 stock
#     For 5th ingredient, 10 stock
# 

# In[ ]:


menu['Ingredient 1 stock'] =100
menu['Ingredient 2 stock'] =50
menu['Ingredient 3 stock'] =30
menu['Ingredient 4 stock'] =20
menu['Ingredient 5 stock'] =10


# In[ ]:


menu.head(5)


# ## 4. Operational hour!
# 
# Customer were coming, they order random items, using Singlish or English name, at random quantity. I assume they pay with e-money so no changes. 
# 
# There will be 100 Customer!

# In[ ]:


order_singlish = menu['Singlish'].tolist()
order_english = menu['English'].tolist()
order = np.concatenate((order_singlish, order_english))


# In[ ]:


print('Menus: ',order)


# The customer specify how many kind of beverages, then the number for each beverages.

# In[ ]:


buy = []
many = []
import random
random.seed(42)
for i in range(100):
    kind = random.randint(1,3)
    for k in range(kind):
        pick = random.choice(order)
        buy.append(pick)
        q = random.randint(1,4)
        many.append(q)
        
    print(pick,q)


# In[ ]:


buy = pd.DataFrame(buy, columns = ['Ordered beverage'])
many = pd.DataFrame(many, columns = ['Counts'])


# In[ ]:


buy = buy.join(many)


# In[ ]:


buy.head(10)


# ## 5. Calculate Revenue
# 
# Time to close. We need to calculate how much we got. (To be continued)
