#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from collections import Counter

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


train_df = pd.read_json('../input/train.json')
# train_df['ingredients'] = train_df['ingredients'].apply(np.array) # Convert list of ingredients to numpy arrays


# In[ ]:


train_df.head()


# In[ ]:


cusine_list = list(train_df['cuisine'].unique())


# In[ ]:


len(cusine_list)


# In[ ]:


train_df.index = train_df['cuisine']


# In[ ]:


ingredients = train_df['ingredients'].values.tolist()
flatten = lambda l: [item for sublist in l for item in sublist]
ingredients = flatten(ingredients)


# ## Top 70 most common ingredients

# In[ ]:


ingredient_counts = Counter(ingredients).most_common(70)
df = pd.DataFrame.from_dict(ingredient_counts)
df.set_index(0, drop=True, inplace=True)
df.plot(kind='bar', figsize=(15, 5))


# ## 50 Least common ingredients

# In[ ]:


ingredient_counts = Counter(ingredients).most_common()[-50:]
df = pd.DataFrame.from_dict(ingredient_counts)
df.set_index(0, drop=True, inplace=True)
df.plot(kind='bar', figsize=(15, 5))


# In[ ]:




