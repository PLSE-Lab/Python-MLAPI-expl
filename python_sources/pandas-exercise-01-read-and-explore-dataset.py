#!/usr/bin/env python
# coding: utf-8

# # Exercise 1

# ### Step 1. Go to [Kaggle]( https://www.kaggle.com/openfoodfacts/world-food-facts)

# ### Step 2. Download the dataset to your computer and unzip it.

# ### Step 3. Use the csv file and assign it to a dataframe called food

# In[ ]:


import pandas as pd
food = pd.read_csv('../input/en.openfoodfacts.org.products.tsv',  sep='\t')


# ### Step 4. See the first 5 entries

# In[ ]:


food.head(5)


# ### Step 5. What is the number of observations in the dataset?

# In[ ]:


food.shape[0]


# ### Step 6. What is the number of columns in the dataset?

# In[ ]:


food.shape[1]


# ### Step 7. Print the name of all the columns.

# In[ ]:


food.columns


# ### Step 8. What is the name of 105th column?

# In[ ]:


food.columns[105]


# ### Step 9. What is the type of the observations of the 105th column?

# In[ ]:


food.dtypes[105]


# ### Step 10. How is the dataset indexed?

# In[ ]:


food.index


# ### Step 11. What is the product name of the 19th observation?

# In[ ]:


food["product_name"][105]

