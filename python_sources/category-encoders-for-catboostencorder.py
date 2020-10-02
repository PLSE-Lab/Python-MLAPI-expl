#!/usr/bin/env python
# coding: utf-8

# ### References
# - [Categorical-Encoding Doc - Catboost Encorder](https://contrib.scikit-learn.org/categorical-encoding/catboost.html)
# - [Categorical-Encoding Doc - Catboost Encorder (Code)](https://contrib.scikit-learn.org/categorical-encoding/_modules/category_encoders/cat_boost.html#CatBoostEncoder)
# - [Transforming categorical features to numerical features](https://catboost.ai/docs/concepts/algorithm-main-stages_cat-to-numberic.html)

# In[ ]:


import numpy as np
import pandas as pd
import category_encoders as ce
from sklearn.datasets import load_boston, load_wine


# In[ ]:


ce.__version__


# # Regression

# In[ ]:


# load data
bunch = load_boston()
df = pd.DataFrame(bunch.data, columns=bunch.feature_names)


# In[ ]:


target_cols = ['CHAS', 'RAD']


# In[ ]:


df.head()


# In[ ]:


df[target_cols].head()


# In[ ]:


enc = ce.CatBoostEncoder(cols=['CHAS', 'RAD']).fit(df, bunch.target)
numeric_dataset = enc.transform(df)


# In[ ]:


numeric_dataset.head()


# In[ ]:


numeric_dataset[target_cols].head()


# # Classifier

# In[ ]:


# load data
bunch = load_wine()
df = pd.DataFrame(bunch.data, columns=bunch.feature_names)


# In[ ]:


df.head()


# In[ ]:


enc = ce.CatBoostEncoder().fit(df, bunch.target)
numeric_dataset = enc.transform(df)


# In[ ]:


numeric_dataset.head()


# In[ ]:




