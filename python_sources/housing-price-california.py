#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd

import os


# In[ ]:


HOUSING_PATH = '../input/housing.csv'
def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = HOUSING_PATH
    return pd.read_csv(csv_path)


# In[ ]:


housing = load_housing_data()
housing.head(5)


# In[ ]:


housing.info()


# > Now we'll try to find out what categories exist and how mant districts belong to each category

# In[ ]:


housing['ocean_proximity'].value_counts()


# In[ ]:


housing.describe()


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
housing.hist(bins=50,figsize=(20,15))
plt.show()


# **Create Test set**

# In[ ]:


import numpy as np
def split_train_test(data,test_ratio):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]


# In[ ]:


train_set, test_set = split_train_test(housing, 0.2)
print(len(train_set),"train +", len(test_set),"test")


# In[ ]:


import hashlib
def test_set_check(identifier, test_ratio,hash):
    return hash(np.int64(identifier)).digest()[-1] < 256*test_ratio

def split_train_test_by_id(data, test_ratio, id_column, hash = hashlib.md5):
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: test_set_check(id_,test_ratio,hash))
    return data.loc[-in_test_set], data.loc[in_test_set]


# In[ ]:


housing_with_id = housing.reset_index()
train_set,test_set = split_train_test_by_id(housing_with_id,0.2,"index")


# In[ ]:


housing_with_id["id"] = housing["longitude"] *1000 +housing["latitude"]
train_set, test_set = split_train_test_by_id(housing_with_id,0.2,"id")


# In[ ]:


from sklearn.model_selection import train_test_split
train_set,test_set = train_test_split(housing, test_size = 0.2, random_state = 42)


# In[ ]:


housing["income_cat"] = np.ceil(housing["median_income"]/1.5)
housing["income_cat"].where(housing["income_cat"] < 5, 5.0, inplace=True)


# In[ ]:


from sklearn.model_selection import StratifiedShuffleSplit

split = StratifiedShuffleSplit(n_splits = 1, test_size = 0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]


# In[ ]:


housing["income_cat"].value_counts() / len(housing)


# **Remove the income_cat attribute so the data is back to its original state**

# In[ ]:


for set in (strat_train_set, strat_test_set):
    set.drop(["income_cat"], axis = 1, inplace=True)


# **Copy of training set**

# In[ ]:


housing = strat_train_set.copy()


# In[ ]:


housing.plot(kind="scatter", x="longitude", y="latitude")


# In[ ]:


housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)


# In[ ]:


housing.plot(kind="scatter", x="longitude",y="latitude", alpha=0.4,s=housing["population"]/100 , label="population",
            c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True)
plt.legend()
#plt.show()


# In[ ]:




