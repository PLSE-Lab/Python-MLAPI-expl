#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory


from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[ ]:


# Function for filling the NaN values in the dataframe
def transform_values(data):
    data.fillna(value='missing', inplace = True)
    return data


# In[ ]:


# Read the datafiles
df_train = pd.read_csv('../input/train.tsv', sep='\t') 
df_test = pd.read_csv('../input/test.tsv', sep='\t')

# Fill the NaN values with "missing"
df_train = transform_values(df_train)
df_test = transform_values(df_test)


# In[ ]:


# Split the categories into a maincategory, subcategory 1 and subcategory 2
def transform_category_name(category_name):
    # If there is a category split it
    try:
        main, sub1, sub2= category_name.split('/')
        return main, sub1, sub2
    # Else return three times "missing"
    except:
        return 'missing', 'missing', 'missing'

# Splitting the catorgy name, removing the 0 price items and fill in the missing descriptions with "missing"
df_train['category_main'], df_train['category_sub1'], df_train['category_sub2'] = zip(*df_train['category_name'].apply(transform_category_name))
df_train[df_train.price != 0.0]
df_train[df_train.item_description != 'missing']

df_test['category_main'], df_test['category_sub1'], df_test['category_sub2'] = zip(*df_test['category_name'].apply(transform_category_name))
df_test[df_test.item_description != 'missing']


# In[ ]:


# Name vectorizing with count vectorizer
from sklearn.feature_extraction.text import CountVectorizer

#Initizializing the countVectorizer
cv = CountVectorizer(ngram_range=(1,2), min_df=7, stop_words='english')

# Making X-vectors for the name, train and test
X_name_train = cv.fit_transform(df_train['name'])
X_name_test = cv.transform(df_test['name'])


# In[ ]:


# Categories vectorization
cat_cv1 = CountVectorizer()
cat_cv2 = CountVectorizer()
cat_cv3 = CountVectorizer()

# Vectorize the trainingscategories to count vectors
X_catmain_train = cat_cv1.fit_transform(df_train['category_main'])
X_catsub1_train = cat_cv2.fit_transform(df_train['category_sub1'])
X_catsub2_train = cat_cv3.fit_transform(df_train['category_sub2'])

# Vectorize the testcategories to count vectors
X_catmain_test = cat_cv1.transform(df_test['category_main'])
X_catsub1_test = cat_cv2.transform(df_test['category_sub1'])
X_catsub2_test = cat_cv3.transform(df_test['category_sub2'])


# In[ ]:


from sklearn.feature_extraction.text import TfidfVectorizer

# create the vectorizer
vectorizer = TfidfVectorizer(ngram_range=(1,2), max_features=90000, strip_accents="ascii",
                             stop_words='english')
# fit and transform the vectorizer for train
X_description_train = vectorizer.fit_transform(df_train['item_description'])
# fit and transform the vectorizer for test
X_description_test = vectorizer.transform(df_test['item_description'])


# In[ ]:


from sklearn.preprocessing import LabelBinarizer
#Brand name binary vectorization train and test set
lb = LabelBinarizer()
X_brand_train = lb.fit_transform(df_train['brand_name'])
X_brand_test = lb.transform(df_test['brand_name'])


# In[ ]:


# Shipping and item state converge to dummies trainingsset
X_shipping_train = pd.get_dummies(df_train['shipping'])
X_condition_train = pd.get_dummies(df_train['item_condition_id'])

# Shipping and item state converge to dummies testset
X_shipping_test = pd.get_dummies(df_test['shipping'])
X_condition_test = pd.get_dummies(df_test['item_condition_id'])


# In[ ]:


from sklearn.model_selection import train_test_split
from scipy.sparse import hstack

# Stack all the vectorized data together, trainingsset
vectorized_data_train = hstack((X_name_train, X_catmain_train, X_catsub1_train, X_catsub2_train, 
                            X_description_train, X_shipping_train, X_condition_train))

# Stack all the vectorized data together, testset
vectorized_data_test = hstack((X_name_test, X_catmain_test, X_catsub1_test, X_catsub2_test, 
                            X_description_test, X_shipping_test, X_condition_test))


# In[ ]:


from sklearn.linear_model import Ridge

Y_train = np.log1p(df_train['price']) #log1p because is nice

# Training the ridge algorithm on the trainingsset
clf = Ridge(alpha=5.3, fit_intercept=True, normalize=False, 
      copy_X=True, max_iter=None, tol=0.01, solver='auto', random_state=None)
clf.fit(vectorized_data_train, Y_train)

Y_pred_test = clf.predict(vectorized_data_test)


# In[ ]:


submission = pd.DataFrame(data=df_test[['test_id']])
submission['price'] = np.expm1(Y_pred_test)
submission.to_csv("submission_ridge.csv", index=False)


# In[ ]:


import math
#A function to calculate Root Mean Squared Logarithmic Error (RMSLE)
def rmsle(y, y_pred):
    assert len(y) == len(y_pred)
    terms_to_sum = [(math.log(y_pred[i] + 1) - math.log(y[i] + 1)) ** 2.0 for i,pred in enumerate(y_pred)]
    return (sum(terms_to_sum) * (1.0/len(y))) ** 0.5


#print(rmsle(np.expm1(Y_train_v), np.expm1(Y_pred)))


# * alpha 5 = 0.45512
# * alpha 5.1 = 0.45513
# * alpha 5.2 = 0.45511
# * alpha 5.3 = 0.45510
# * alpha 5.4 = 0.45512
# * alpha 5.5 = 0.45511
# * alpha 0.5 = 0.461
# * alpha 0.6 = 0.460
# * alpha 15 = 0.458
# * alpha 100 = 0.481
