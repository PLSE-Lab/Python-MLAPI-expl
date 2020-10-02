#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Step 1. Import the necessary libraries

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


# In[ ]:


#Step 2. Import the dataset from this address.
#Step 3. Assign it to a variable called chipo.

url = 'https://raw.githubusercontent.com/justmarkham/DAT8/master/data/chipotle.tsv'
chipo = pd.read_csv(url, sep = '\t')

#Step 4. See the first 10 entries
chipo.head(10)


# In[ ]:


#Step 5. What is the number of observations in the dataset?

# Solution 1
chipo.info() #the number entries = observations


# In[ ]:


#solution 2

chipo.shape[0]


# In[ ]:


#Step 6. What is the number of columns in the dataset?

chipo.shape[1]


# In[ ]:


#Step 7. Print the name of all the columns.

chipo.columns


# In[ ]:


#Step 8. How is the dataset indexed?

chipo.index


# In[ ]:


a = chipo.head()
a


# In[ ]:


#Step 9. Which was the most-ordered item?

c = chipo.groupby('item_name').sum()
c = c.sort_values(['quantity'], ascending = False)
c.head(2)


# In[ ]:


#Step 10. For the most-ordered item, how many items were ordered?

print('For the most-ordered item, ordered were:',str(713926))


# In[ ]:


#Step 11. What was the most ordered item in the choice_description column?

d = chipo.groupby('choice_description').sum()
d = d.sort_values(['quantity'], ascending = False)
d.head(2)


# In[ ]:


#Step 12. How many items were orderd in total?

total_items_ordered = chipo.quantity.sum()
total_items_ordered


# In[ ]:


#Step 13. Turn the item price into a float
#Step 13.a. Check the item price type

chipo.item_price.dtype


# In[ ]:


#Step 13.b. Create a lambda function and change the type of item price
try:
    dollarizer = lambda x: float(x[1:-1])
    chipo.item_price = chipo.item_price.apply(dollarizer)
    
except:TypeError 
    


# In[ ]:


chipo.item_price.dtype


# In[ ]:


a


# In[ ]:


#Step 14. How much was the revenue for the period in the dataset?

revenue = (chipo['quantity'] * chipo['item_price']).sum()

print('Revenue was: $' + str(np.round(revenue,2)))


# In[ ]:


# How many orders were made in the period?

orders = chipo.order_id.value_counts().count()
orders



# In[ ]:


#Step 16. What is the average revenue amount per order?

# Solution 1
chipo['revenue'] = chipo['quantity'] * chipo['item_price']
d = order_grouped = chipo.groupby(by=['order_id']).sum()
order_grouped.mean()['revenue']


# In[ ]:


chipo.groupby('order_id').sum().mean()['revenue']


# In[ ]:


#Step 17. How many different items are sold?

chipo.item_name.value_counts().count()


# In[ ]:




