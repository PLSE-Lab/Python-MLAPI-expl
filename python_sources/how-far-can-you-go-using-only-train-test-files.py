#!/usr/bin/env python
# coding: utf-8

# # How far can you go using only train.csv and test.csv ?
# ### In this kernel, I will tackle the problem the easiest way : 
# - Using only the features available in the train.csv and test.csv files
# - Using a Linear Regression as model
# 
# I'll also tackle the outliers issue a bit.
# 
# ### Any feedback is always appreciated ! 

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.svm import  OneClassSVM
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.linear_model import LinearRegression
import lightgbm as lgb

sns.set_style('whitegrid')


# ### Loading Data

# In[ ]:


train_df = pd.read_csv("../input/train.csv")
test_df = pd.read_csv("../input/test.csv")


# In[ ]:


train_df.info()


# Nothing is missing

# In[ ]:


test_df.info()


# We have one data missing, let us fill it.

# In[ ]:


test_df['first_active_month'] = test_df['first_active_month'].fillna('2018-01-01')


# In[ ]:


train_df.head()


# In[ ]:


print('Size of train dataset :', len(train_df))
print('Size of test dataset :', len(test_df))


# ### Let us check the repartition of the 3 features

# In[ ]:


plt.figure(figsize=(10, 7))
sns.countplot(train_df['feature_1'])
plt.show()


# In[ ]:


plt.figure(figsize=(10, 7))
sns.countplot(train_df['feature_2'])
plt.show()


# In[ ]:


plt.figure(figsize=(10, 7))
sns.countplot(train_df['feature_3'])
plt.show()


# Features seem pretty balanced.
# 

# ### First active month

# In[ ]:


train_df['first_active_month'] = pd.to_datetime(train_df['first_active_month'])
test_df['first_active_month'] = pd.to_datetime(test_df['first_active_month'])


# In[ ]:


print("Newest client :", max( max(train_df['first_active_month']), max(test_df['first_active_month'])))


# In[ ]:


first_active_day = pd.to_timedelta(train_df['first_active_month']).apply(lambda x: x.days)
first_active_day_test = pd.to_timedelta(test_df['first_active_month']).apply(lambda x: x.days)


# In[ ]:


newest = max(max(first_active_day), max(first_active_day_test))


# In[ ]:


ancienety = (newest - first_active_day).astype(int) // 30
ancienety_test = (newest - first_active_day_test).astype(int) // 30


# In[ ]:


plt.figure(figsize=(12, 8))
sns.distplot(ancienety)
plt.xlabel('Ancienety (in month)')
plt.show()


# In[ ]:


train_df['ancienety'] = ancienety
test_df['ancienety'] = ancienety_test


# ### Now to the distribution of the target

# In[ ]:


plt.figure(figsize=(10, 7))
sns.distplot(train_df['target'])
plt.show()


# It's basically a Gaussian with some outliers at on the left. Let's take a closer look at those outliers.

# In[ ]:


df_outlier = train_df[train_df['target'] < -30]


# In[ ]:


print('Number of outliers :', len(df_outlier))


# In[ ]:


df_outlier.head()


# In[ ]:


df_inlier = train_df[train_df['target'] > -30]


# Nothing can be said so far, except that they'll play an important role in the future.

# ## First prediction :
# > Using a linear regression

# ### Linear Regression

# In[ ]:


X_train = np.array(train_df[['feature_1', 'feature_2', 'feature_3', 'ancienety']])
X_test = np.array(test_df[['feature_1', 'feature_2', 'feature_3', 'ancienety']])
y_train = np.array(train_df['target'])


# In[ ]:


lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)
y_pred = lin_reg.predict(X_train)


# In[ ]:


print('RMSE on train data :', np.sqrt(mean_squared_error(y_train, y_pred)))


# In[ ]:


y_test = lin_reg.predict(X_test)


# ### Linear Regression without Outliers

# In[ ]:


X_inlier = np.array(df_inlier[['feature_1', 'feature_2', 'feature_3', 'ancienety']])
y_inlier = np.array(df_inlier['target'])


# In[ ]:


lin_reg = LinearRegression()
lin_reg.fit(X_inlier, y_inlier)
y_pred = lin_reg.predict(X_inlier)


# In[ ]:


print('RMSE on inliers :', np.sqrt(mean_squared_error(y_inlier, y_pred)))


# #### Outliers seem to play quite a big role, it could be useful to detect them before applying a classifier.

# ## Can we detect the Outliers ?
# 
# I'm using the One Class SVM from sklearn.
# > https://scikit-learn.org/stable/modules/generated/sklearn.svm.OneClassSVM.html

# In[ ]:


X_outlier = np.array(df_outlier[['feature_1', 'feature_2', 'feature_3', 'ancienety']])
y_outlier = np.ones(X_outlier.shape[0])


# In[ ]:


clf = OneClassSVM(nu=0.01, kernel="rbf", gamma=0.1)
clf.fit(X_inlier[:50000])  #It takes a while so I don't take all the inliers


# In[ ]:


y_pred_inlier = clf.predict(X_inlier)
y_pred_outlier = clf.predict(X_outlier)


# In[ ]:


print("Accuracy on inliers :", accuracy_score(y_pred_inlier, np.ones(y_inlier.size)))
print("Accuracy on outliers :", accuracy_score(y_pred_outlier, - np.ones(y_outlier.size)))


# #### No surprise, the answer is no. But it could probably be done with more features !

# ## Submission

# In[ ]:


submission = pd.DataFrame({"card_id": test_df["card_id"].values})
submission["target"] = y_test


# In[ ]:


submission.to_csv("submission.csv", index=False)


# This simple model scored 3.925, which is way better than I expected.

# ## Following work :
# - Taking the other stuff into account !
# 
# #### Thanks for reading !
