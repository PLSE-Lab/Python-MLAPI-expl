#!/usr/bin/env python
# coding: utf-8

# ## Introduction
# ** This is my first Kaggle Kernel and please let me know if you have any suggestions/feedback and comment about improvements.
# 
# Avito.ru is a Russian classified advertisements website with sections devoted to general good for sale, jobs, real estate, personals, cars for sale, and services.
# 
# Sellers sometimes feel frustrated with both too little demand (indicating something is wrong with the product or the product listing) or too much demand (indicating a hot item with a good description was underpriced).
# 
# The aim of this challenge is to predict demand for an online advertisement based on its full description (title, description, images, etc.), its context (geographically where it was posted, similar ads already posted) and historical demand for similar ads in similar contexts. With this information, Avito can inform sellers on how to best optimize their listing and provide some indication of how much interest they should realistically expect to receive.

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn import preprocessing, model_selection, metrics
import lightgbm as lgb

color = sns.color_palette()
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# ### Load data

# In[ ]:


train_df = pd.read_csv("../input/train.csv")
test_df = pd.read_csv("../input/test.csv")


# In[ ]:


train_df.head(4)


# The train dataset description is as follows:
# 
# * item_id - Ad id.
# * user_id - User id.
# * region - Ad region.
# * city - Ad city.
# * parent_category_name - Top level ad category as classified by Avito's ad model.
# * category_name - Fine grain ad category as classified by Avito's ad model.
# * param_1 - Optional parameter from Avito's ad model.
# * param_2 - Optional parameter from Avito's ad model.
# * param_3 - Optional parameter from Avito's ad model.
# * title - Ad title.
# * description - Ad description.
# * price - Ad price.
# * item_seq_number - Ad sequential number for user.
# * activation_date- Date ad was placed.
# * user_type - User type.
# * image - Id code of image. Ties to a jpg file in train_jpg. Not every ad has an image.
# * image_top_1 - Avito's classification code for the image.
# * deal_probability - The target variable. This is the likelihood that an ad actually sold something. It's not possible to verify every transaction with certainty, so this column's value can be any float from zero to one.

# In[ ]:


test_df.head(4)


# In[ ]:


plt.scatter(range(train_df.shape[0]), np.sort(train_df['deal_probability'].values))
plt.xlabel('index', fontsize=12)
plt.ylabel('deal probability', fontsize=12)
plt.title("Deal Probability Distribution", fontsize=14)
plt.show()

