#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import warnings
warnings.filterwarnings('ignore')

import os
print(os.listdir("../input"))


# In[ ]:


df = pd.read_csv('../input/train.tsv',sep="\t")


# In[ ]:


df.head()


# In[ ]:


df.set_index('train_id',inplace=True)


# In[ ]:


# quick check on missing data
df_na = (df.isnull().sum() / len(df)) * 100
df_na = df_na.drop(df_na[df_na == 0].index).sort_values(ascending=False)[:30]
missing_data = pd.DataFrame({'Missing Ratio' :df_na})
missing_data.head(20)


# ## Target variable (Item price)
# Drop samples with price = 0 and fix the skewness of the target variable by log transformation

# In[ ]:


len(df[df['price'] == 0])


# In[ ]:


df = df[df['price']>0]


# In[ ]:


# split target and features
y = df['price']
df = df.drop('price',axis=1)


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="whitegrid")
from scipy.stats import norm

sns.distplot(y , fit=norm);

# Get the fitted parameters used by the function
(mu, sigma) = norm.fit(y)

#Now plot the distribution
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
            loc='best')
plt.ylabel('Frequency')
plt.title('Price distribution')
plt.show()


# In[ ]:


import numpy as np
y_log = np.log1p(y)


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="whitegrid")
from scipy.stats import norm

sns.distplot(y_log , fit=norm);

# Get the fitted parameters used by the function
(mu, sigma) = norm.fit(y_log)

#Now plot the distribution
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
            loc='best')
plt.ylabel('Frequency')
plt.title('Price distribution')
plt.show()


# ## Features

# In[ ]:


df.dtypes


# In[ ]:


# replace missing values with 'missing'
df['brand_name'] = df['brand_name'].fillna('missing')
df['category_name'] = df['category_name'].fillna('missing')
df['item_description'] = df['item_description'].fillna('missing')


# In[ ]:


df.head()


# In[ ]:


# changing the column types for categorical features
df['category_name'] = df['category_name'].astype('category')
df['brand_name'] = df['brand_name'].astype('category')
df['item_condition_id'] = df['item_condition_id'].astype('category')


# In[ ]:


# clean up text based features before tf-idf 
def clean_text(col):
    # remove non alpha characters
    col = col.str.replace("[\W]", " ") #a-zA-Z1234567890
    # all lowercase
    col = col.apply(lambda x: x.lower())
    return col

df['name']=clean_text(df['name'])
df['category_name']=clean_text(df['category_name'])
df['item_description']=clean_text(df['item_description'])


# In[ ]:


df.head()


# In[ ]:


# create feature matrix for name and category_name
from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(min_df=10,max_df=0.1, stop_words='english')
X_name = cv.fit_transform(df['name'])
cv = CountVectorizer()
X_category = cv.fit_transform(df['category_name'])


# In[ ]:


X_name.shape


# In[ ]:


X_category.shape


# In[ ]:


# Feature matrix for item description
cv = CountVectorizer(min_df=10,max_df=0.1, stop_words='english')
X_item_description = cv.fit_transform(df['item_description'])


# In[ ]:


X_item_description.shape


# In[ ]:


# feature matrix for brand
from sklearn.preprocessing import LabelBinarizer
lb = LabelBinarizer(sparse_output=True)
X_brand = lb.fit_transform(df['brand_name'])


# In[ ]:


X_brand.shape


# In[ ]:


# feature matrix for item condition and shipping
from scipy.sparse import csr_matrix
X_condition_shipping = csr_matrix(pd.get_dummies(df[['item_condition_id', 'shipping']], sparse=True).values)


# In[ ]:


X_condition_shipping.shape


# In[ ]:


# create the complete feature matrix
from scipy.sparse import hstack

X_all = hstack((X_brand, X_category, X_name, X_item_description, X_condition_shipping)).tocsr()


# In[ ]:


X_all.shape


# In[ ]:


# reduce the feature columns by removing all features with a document frequency smaller than 1
mask = np.array(np.clip(X_all.getnnz(axis=0) - 1, 0, 1), dtype=bool)
X_all = X_all[:, mask]


# In[ ]:


X_all.shape


# In[ ]:


# split into test and train samples
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_all, y_log, random_state=42, train_size=0.1, test_size=0.02)


# In[ ]:


X_train.shape


# In[ ]:


X_test.shape


# ## Training

# In[ ]:


from sklearn.model_selection import KFold, cross_val_score
def score_model(model):
    kf = KFold(3, shuffle=True, random_state=42).get_n_splits(X_train)
    model_score = np.mean(cross_val_score(model, X_train, y_train, scoring="neg_mean_squared_error", cv = kf, n_jobs=-1))
    return((type(model).__name__,model_score))


# In[ ]:


# get a baseline for a few regression models

import time
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor

model_scores = pd.DataFrame(columns=['model','NMSE'])
reg_model = [Ridge(),Lasso(), GradientBoostingRegressor(),XGBRegressor()]
for model in reg_model:
    start = time.time()
    sc = score_model(model)
    total = time.time() - start
    print("done with {}, ({}s)".format(sc[0],total))
    model_scores = model_scores.append({'model':sc[0],'NMSE':sc[1]},ignore_index=True)    

# print results
model_scores.sort_values('NMSE',ascending=False)


# let's continue with Ridge and XGB as both train relatively fast and had the highest scores.

# In[ ]:




