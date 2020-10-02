#!/usr/bin/env python
# coding: utf-8

# #About
# I was intrigued by the "pricey shoes" from reading previous posts as well as my own analysis. I started doing some data cleaning.

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


data = pd.read_csv('../input/7003_1.csv')


# It turns out that some 'prices.amountMin' values are not actual price values (string or numeric). See below. I first use write a function to find out which values are not related to price.

# In[ ]:


def convertable_to_numeric(s, row):
    try:
        pd.to_numeric(s)
    except ValueError:
        print('Row: %s, Content: %s' % (row, data.loc[row, 'prices.amountMin']))
        return row


# From the above, I found out that those cells in those rows are actually dislocated or shifted on the right hand side of 'prices.amountMin' column. More on this later.

# Since there are only a few of those items having a missing 'prices.amountMin', I decide to remove those items for now.

# In[ ]:


rows_to_drop = []
for i in range(len(data)):
    rows_to_drop.append(convertable_to_numeric(data.loc[i, 'prices.amountMin'], i))
rows_to_drop = [i for i in rows_to_drop if i != None]


# In[ ]:


data_new = data.drop(data.index[rows_to_drop])


# In[ ]:


data_new['prices.amountMin'] = pd.to_numeric(data_new['prices.amountMin'])


# Find pricey stuff (>$1000)

# In[ ]:


pricey_items = list(data_new.loc[data_new['prices.amountMin'] > 1000, :].index)


# In[ ]:


for row in pricey_items:
    print('ID: %s \nName: %s \nprices.amountMin: %s\n' % (data_new.loc[row, 'id'], data_new.loc[row, 'name'], data_new.loc[row, 'prices.amountMin']))


# As printed out above, a number of those pricey are probably jewelry items (wedding bands or diamonds) rather than shoes.

# A summary of non-shoe items (keywords)

# In[ ]:


non_shoe_items = ['Ring', 'Watch', 'Watches', 'Jacket', 'Earrings', 'Earring', '14kt', '14k', 'Diamond', 'Diamonds', 'Band', 'Bands', 'Bra', 'Cap', 'Analog', 'Cttw', 'Necklace', '18k', 'Coat', 'Glove', 'Underwear', 'Pants', 'Panties', 'Lingerie', 'Socks', 'Clip', 'Scarf', 'Tights', 'Skirt', 'Sunglasses', 'Shorts', 'Hoodie', 'Bolero', 'Robe']

