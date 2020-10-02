#!/usr/bin/env python
# coding: utf-8

# This exercise is copied from for learning purpose: https://github.com/guipsamora/pandas_exercises
# # Getting and Knowing your Data

# This time we are going to pull data directly from the internet.
# Special thanks to: https://github.com/justmarkham for sharing the dataset and materials.
# 
# ### Step 1. Import the necessary libraries

# In[ ]:


import numpy as np
import pandas as pd


# ### Step 2. Import the dataset from this [address](https://raw.githubusercontent.com/justmarkham/DAT8/master/data/chipotle.tsv). 

# ### Step 3. Assign it to a variable called chipo.

# In[ ]:


url = 'https://raw.githubusercontent.com/justmarkham/DAT8/master/data/chipotle.tsv'
chipo = pd.read_csv(url, sep = '\t')


# ### Step 4. See the first 10 entries

# In[ ]:


chipo.head(10)


# ### Step 5. What is the number of observations in the dataset?

# In[ ]:





# ### Step 6. What is the number of columns in the dataset?

# In[ ]:





# ### Step 7. Print the name of all the columns.

# In[ ]:





# ### Step 8. How is the dataset indexed?

# In[ ]:





# ### Step 9. Which was the most ordered item?

# In[ ]:





# ### Step 10. How many items were ordered?

# In[ ]:





# ### Step 11. What was the most ordered item in the choice_description column?

# In[ ]:





# ### Step 12. How many items were orderd in total?

# In[ ]:





# ### Step 13. Turn the item price into a float

# In[ ]:





# ### Step 14. How much was the revenue for the period in the dataset?

# In[ ]:





# ### Step 15. How many orders were made in the period?

# In[ ]:





# ### Step 16. What is the average amount per order?

# In[ ]:





# ### Step 17. How many different items are sold?

# In[ ]:




