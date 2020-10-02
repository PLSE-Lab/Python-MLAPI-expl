#!/usr/bin/env python
# coding: utf-8

# # Mercari price prediction
# 
# ## Background:
# 
# Mercari is one of the biggest mobile social commerce apps based out of Japan. 
# Users are able to upload items to sell via pictures, descriptions and an estimated price that they would like to sell for.
# Other users can then purchase that item if they are interested or offer a different sale price.
# 
# One of the business problems that this challenge attempts to solve is the item sale price.
# If relatively accurate item prices can be predicted based on item category/name/descriptions, then app will have a better user experience. Users will be able to sell more items and spend less time seting the correct price.

# In[ ]:



import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import xgboost as xgb

import scipy

from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import LabelBinarizer


get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


train = pd.read_csv('../input/train.tsv', sep='\t')
test = pd.read_csv('../input/test.tsv', sep='\t')


# ## Description of data columns
# 
# * train_id or test_id - the id of the listing
# 
# * name - the title of the listing. Note that we have cleaned the data to remove text that look like prices to avoid leakage. These removed prices are represented as [rm]
# 
# * item_condition_id - the condition of the items provided by the seller
# 
# * category_name - category of the listing
# 
# * brand_name
# 
# * price - the price that the item was sold for. This is the target variable that you will predict. The unit is USD. This column doesn't exist in test.tsv since that is what you will predict.
# 
# * shipping - 1 if shipping fee is paid by seller and 0 by buyer
# 
# * item_description - the full description of the item. Note that we have cleaned the data to remove text that look like prices  to avoid leakage. These removed prices are represented as [rm]

# In[ ]:


train.head()


# After taking a look at the first 5 items of the training dataset, it looks like we have text, categorical, and also numerical data. It also looks like we will have to clean the data throughly for missing data(NaN).

# In[ ]:


test.head()


# The first 5 items of the test data looks more or less the same as the training data except that it is missing the price column.

# In[ ]:


train.info()


# After taking a look at the data types, it looks like we havce ints, floats, and string objects.

# In[ ]:


train.corr()['price']


# Above shows the correlation between columns and the price column.

# In[ ]:


train['price'].describe()


# Lets take a look at the distribution of the price column. 
# Average price is around 27 dollars. There are 1.5 million items. 
# There are free items and the amx is 2000 dollars.

# In[ ]:


train['price'].plot.hist(title="Histogram of item prices in Dataset")


# It looks like most of the items are priced between 0 and 200 dollars

# In[ ]:


train.shape


# In[ ]:


test.shape


# ## Category name

# In[ ]:


len(train['category_name'].unique())


# There are 1.2K + unique category names

# In[ ]:


train['category_name'].value_counts()


# It looks like each category is actually a combination of 3 different category types. We can try to break them into their own categories to take a look at their individual representations.

# ## Brand Name

# In[ ]:


train['brand_name'].describe()


# It looks like there are a lot of missing values

# In[ ]:


train['brand_name'].value_counts()


# In[ ]:


train['brand_name'] = train['brand_name'].fillna("missing")


# Lets fill the na values with missing and then take a look at the distribution.

# In[ ]:


train['brand_name'].value_counts()


# In[ ]:


train['has_brand'] = np.where(train['brand_name'] == 'missing', 0, 1)


# Lets take a look at the difference in price between items with a brand vs. the ones which don't.'

# In[ ]:


train.head()


# In[ ]:


train.corr()


# It looks like there are some correlations between price and if the item has brand.

# In[ ]:


len(train['brand_name'].unique())


# ## Shipping
# 

# In[ ]:


train.groupby('shipping').mean()


# It looks like there is a difference between items that are shipped by buyer vs. seller. Average price for item shipped 0, is $30 vs. $22 for when shipping is 1. There also seems that there is a slight difference between whether the item has a brand or not.

# In[ ]:


train.groupby('shipping').mean()['price'].plot.bar(title="Average price of item vs. shipping")


# Above plot shows the difference between average price of items wiht different shipping.

# ## Item description

# Lets take a look at the different item descriptions for the items.

# In[ ]:


train['item_description'].head()


# In[ ]:


train['item_description'].str.len().plot.hist(title="Histogram of length of item descriptions")


# It looks like the item description lengths are left skewed.

# In[ ]:


train['item_description'].str.len().describe()


# On average item descriptions are 145 characters.

# ## Name
# 

# In[ ]:


train['name'].describe()


# Taking a look at the name column shows that there are text data with most of them unique.

# In[ ]:


# checks null values, 
# a few in item_description
# a lot in brand_name 
# 6k + in category_name
train.isnull().sum()


# ## Condition

# In[ ]:


train.groupby('item_condition_id').count()['price'].plot.bar(title="Distribution of items by item condition")


# It looks like we mostly have items of condition 1-3 and conditions of 4 and 5 are rather rare.

# ## Preprocess Data

# In[ ]:


train['first_category_name'] = train['category_name'].str.split('/').str.get(0)
train['second_category_name'] = train['category_name'].str.split('/').str.get(1)
train['third_category_name'] = train['category_name'].str.split('/').str.get(2)


# Lets split the category name into its category and subcategories

# In[ ]:


train.head()


# In[ ]:


train = train.drop('category_name', axis=1)


# In[ ]:


train.head()


# Fill missing values in our training dataset

# In[ ]:


train['first_category_name'] = train['first_category_name'].fillna('missing')
train['second_category_name'] = train['second_category_name'].fillna('missing')
train['third_category_name'] = train['third_category_name'].fillna('missing')


# In[ ]:


train['item_description'] = train['item_description'].fillna('missing')


# In[ ]:


train.isnull().sum()


# In[ ]:


train.head()


# Next, lets encode our categorical and text data columns.

# In[ ]:


y = np.log1p(train["price"])

NUM_BRANDS = 2500
NAME_MIN_DF = 10
MAX_FEAT_DESCP = 50000

print("Encodings")
count = CountVectorizer(min_df=NAME_MIN_DF)
X_name = count.fit_transform(train["name"])

print("Category Encoders")
count_category_one = CountVectorizer()
X_category_one = count_category_one.fit_transform(train["first_category_name"])

count_category_two = CountVectorizer()
X_category_two = count_category_two.fit_transform(train["second_category_name"])

count_category_three = CountVectorizer()
X_category_three = count_category_three.fit_transform(train["third_category_name"])

print("Descp encoders")
count_descp = TfidfVectorizer(max_features = MAX_FEAT_DESCP, 
                              ngram_range = (1,3),
                              stop_words = "english")
X_descp = count_descp.fit_transform(train["item_description"])

print("Brand encoders")
vect_brand = LabelBinarizer(sparse_output=True)
X_brand = vect_brand.fit_transform(train["brand_name"])

print("Dummy Encoders")
X_dummies = scipy.sparse.csr_matrix(pd.get_dummies(train[[
    "item_condition_id", "shipping"]], sparse = True).values)


# In[ ]:


X = scipy.sparse.hstack((X_dummies, 
                         X_descp,
                         X_brand,
                         X_category_one,
                         X_category_two,
                         X_category_three,
                         X_name)).tocsr()


# In[ ]:


X_train, X_test, y_val_train, y_val_test = train_test_split( X, y, test_size=0.3, random_state=42)
model = xgb.XGBRegressor()
model.fit(X_train,y_val_train)
print(model)


# In[ ]:


from sklearn.metrics import mean_squared_error
predictions = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_val_test, predictions))
print(rmse)


# In[ ]:



xgb.plot_importance(model)


# In[ ]:


"""
to try gridsearching xgboost with gridsearchcv:

from sklearn.model_selection import GridSearchCV

model = xgb.XGBRegressor()

parameters = {
        'n_estimators': [100, 250, 500],
        'learning_rate': [0.1, 0.15, 0.],
        'max_depth': [6, 9, 12],
        'subsample': [0.9, 1.0],
    }

clf = GridSearchCV(model, parameters)
clf.fit(X_train, y_val_train)
print(clf)
"""


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




