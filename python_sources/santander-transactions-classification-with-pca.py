#!/usr/bin/env python
# coding: utf-8

# <h1 align=center><font size = 4>Santander Customer Transaction Prediction</font></h1>
# <h1 align=center><font size = 5>Model Binary Classifier with Dimensionality Reduction</font></h1>

# # Table of Contents
# * [Introduction/Business Problem](#introduction)
# * [Setup](#setup)
# * [Get the Data](#get_data)
# * [Take a Quick Look at the Data Structure](#data_structure)
# * [Create a Valuation Set](#create_val_set)
# * [Explore and Visualize the Data to Gain Insights](#explore_visualize)
# * [Prepare Data for Machine Learning](#preparation)
# * [Machine Learning Models](#modeling)
# * [Model Evaluation using Valuation Set](#val_set_score)
# * [Predictions using Test Set](#predictions)

# <a id = "introduction"></a>
# # Introduction/Business Problem

# At [Santander](https://www.santanderbank.com/us/personal) our mission is to help people and businesses prosper. We are always looking for ways to help our customers understand their financial health and identify which products and services might help them achieve their monetary goals.
# 
# Our data science team is continually challenging our machine learning algorithms, working with the global data science community to make sure we can more accurately identify new ways to solve our most common challenge, binary classification problems such as: is a customer satisfied? Will a customer buy this product? Can a customer pay this loan?
# 
# In this challenge, we invite Kagglers to help us identify which customers will make a specific transaction in the future, irrespective of the amount of money transacted. The data provided for this competition has the same structure as the real data we have available to solve this problem.

# <a id="setup"></a>
# # Setup

# Import a few common modules and ensure MatplotLib plots figures inline

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


import seaborn as sns
sns.set(style="darkgrid")

# to make this notebook's output stable across runs
np.random.seed(42)

# To plot figures
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib
import matplotlib.pyplot as plt
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

print('Libraries imported.')


# <a id="get_data"></a>
# # Get the Data

# We are provided with an anonymized dataset containing numeric feature variables, the binary target column, and a string ID_code column.
# 
# The task is to predict the value of target column in the test set.
# 
# **File descriptions**
# * **train.csv** - the training set.
# * **test.csv** - the test set. The test set contains some rows which are not included in scoring.
# * **sample_submission.csv** - a sample submission file in the correct format.

# In[ ]:


transactions = pd.read_csv('../input/train.csv')
print('train data imported.')


# <a id="data_structure"></a>
# # Take a Quick Look at the Data Structure

# In[ ]:


transactions.info(verbose=True, null_counts=True)


# The data set contains numeric features type `float64`, besides the binary target column type `int64` and the ID_code column type `object`

# In[ ]:


len(transactions.ID_code.unique())


# In[ ]:


print('number of rows {}'.format(len(transactions)))
print('number of columns {}'.format(len(transactions.columns)))


# Each row represents one transaction. The `info()` method shows the total number of rows and each attribute's type and number of the non-null values. There are 200000 instances in the dataset and the id per transaction is unique as expected.
# 
# The data has different type of attributes. The majority of the attributes is type numerical. In addition, the dataset contains the target and the ID_code attributes.

# In[ ]:


transactions.isna().any().any()


# There are no `NaN` in the transaction dataset as the `info` method has shown (all attributes have 200000 non-null)

# In[ ]:


transactions.head()


# In[ ]:


transactions.target.describe()


# In[ ]:


sns.countplot(transactions.target)
plt.show()


# From the total number of transactions 10% belongs to 1 (the mean in the `target` attribute), so the classes are imbalanced and accuracy won't be a reasonable metric to evaluate our model

# In[ ]:


transactions.drop(['target','ID_code'], axis=1).describe()


# In[ ]:


transactions.drop(['target','ID_code'],axis=1).describe().loc['mean'].sort_values(ascending=False)[:10]


# In[ ]:


plt.figure(figsize=(8, 4))
sns.distplot(transactions.drop(['target','ID_code'],axis=1).describe().loc['mean'])
plt.title('Distribution of numeric features mean values')
plt.show()


# In[ ]:


transactions.drop(['target','ID_code'], axis=1).describe().loc['std'].sort_values(ascending=False)[:10]


# In[ ]:


plt.figure(figsize=(8, 4))
sns.distplot(transactions.drop(['target','ID_code'],axis=1).describe().loc['std'])
plt.title('Distribution of numeric features std values')
plt.show()


# In[ ]:


plt.figure(figsize=(8, 4))
sns.distplot(transactions.drop(['target','ID_code'],axis=1).describe().loc['max'])
plt.title('Distribution of numeric features max values')
plt.show()


# In[ ]:


plt.figure(figsize=(8, 4))
sns.distplot(transactions.drop(['target','ID_code'],axis=1).describe().loc['min'])
plt.title('Distribution of numeric features min values')
plt.show()


# The above analysis indicates that the numerical features have different scales, thus feature scaling should be considered in ML algorithms.

# <a id="create_val_set"></a>
# # Create a Valuation Set

# Before we look at the data any further, we need to create a valuation set, put it aside and use it later to evaluate the model.

# In[ ]:


from sklearn.model_selection import train_test_split

train_set, val_set = train_test_split(transactions, test_size=0.9, random_state=42, stratify=transactions.target)
print(train_set.shape, val_set.shape)


# In[ ]:


train_set.target.describe()


# In[ ]:


val_set.target.describe()


# <a id="explore_visualize"></a>
# # Explore and Visualize the Data to Gain Insights

# Remove the target and ID_code columns from the dataset

# In[ ]:


X_train = train_set.drop(['target', 'ID_code'], axis=1)
print(X_train.shape)
X_train.head()


# 
# ### Kurtosis Analysis

# In[ ]:


kurtosis = X_train.kurtosis().sort_values(ascending=False)
kurtosis[:9]


# In[ ]:


X_train[kurtosis[:9].index.values.tolist()].hist(bins=50, figsize=(12,8))
plt.show()


# In[ ]:


X_train[kurtosis[-9:].index.values.tolist()].hist(bins=50, figsize=(12,8))
plt.show()


# In[ ]:


X_train_transform = X_train[kurtosis[-9:].index.values.tolist()].apply(lambda x: np.sign(x) * np.log(1 + np.abs(x)))
X_train_transform.hist(bins=50, figsize=(12,8))
plt.show()


# In[ ]:


X_train_transform.kurtosis()


# In[ ]:


kurtosis[-9:]


# The kurtosis analysis has shown that we don't need to improve the distribution of the dataset. We don't apply any transformation.

# 
# ### Scaling

# As we have shown before, the attributes in the dataset have different scales and therefore we scale the data by removing the mean and scaling to unit variance

# In[ ]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)


# ### Dimensionality Reduction

# In[ ]:


from sklearn.decomposition import PCA
pca = PCA(n_components=0.95)
pca.fit(X_train_scaled)
pca.n_components_


# In[ ]:


X_train_scaled_pca = pca.transform(X_train_scaled)
X_train_scaled_pca.shape


# ### Visualization

# In[ ]:


np.random.seed(42)

m = 5000
idx = np.random.permutation(len(X_train_scaled_pca))[:m]

X_plot = X_train_scaled_pca[idx]
y_plot = train_set['target'].values[idx]
print(X_plot.shape, y_plot.shape)


# In[ ]:


def plot_2D_dataset(X_2D, title=None):
    plt.figure(figsize=(8,8))
    plt.scatter(X_2D[:,0], X_2D[:, 1], c=y_plot, cmap='jet')
    plt.xlabel('1st component')
    plt.ylabel('2nd component')
    if title is not None:
        plt.title(title)


# Use t-SNE to reduce the dataset down to two dimensions and plot the result using Matplotlib. I use a scatterplot using 2 different colors to represent each transaction's target class.

# In[ ]:


from sklearn.manifold import TSNE
import time

tsne = TSNE(n_components=2, random_state=42)

t0 = time.time()
X_2D_tsne = tsne.fit_transform(X_plot)
t1 = time.time()

print("t-SNE took {:.1f}s.".format(t1 - t0))
plot_2D_dataset(X_2D_tsne, title='t-SNE')
plt.show()


# In[ ]:


from sklearn.decomposition import PCA
import time

pca = PCA(n_components=2, random_state=42)

t0 = time.time()
X_2D_pca = pca.fit_transform(X_plot)
t1 = time.time()

print("PCA took {:.1f}s.".format(t1 - t0))
plot_2D_dataset(X_2D_pca, title='PCA')
plt.show()


# In[ ]:


from sklearn.manifold import LocallyLinearEmbedding

lle = LocallyLinearEmbedding(n_components=2, random_state=42)

t0 = time.time()
X_2D_lle = lle.fit_transform(X_plot)
t1 = time.time()

print("LLE took {:.1f}s.".format(t1 - t0))
plot_2D_dataset(X_2D_lle, title='LLE')
plt.show()


# <a id="preparation"></a>
# # Prepare Data for Machine Learning

# In[ ]:


X_train = train_set.drop(['target', 'ID_code'], axis=1)


# In[ ]:


from sklearn.pipeline import Pipeline
preparation_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=0.95))
])


# In[ ]:


X_train_prepared = preparation_pipeline.fit_transform(X_train)
print(X_train_prepared.shape)
X_train_prepared


# In[ ]:


y_train = train_set['target']
print(y_train.shape)
y_train.head()


# <a id="modeling"></a>
# # Machine Learning Models

# ### Logistic Regression

# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import reciprocal

param_distribs = {
        'solver': ['lbfgs', 'liblinear', 'sag'],
        'C': reciprocal(0.01, 10)
    }

lr_clf = LogisticRegression()

lr_rnd_search = RandomizedSearchCV(lr_clf, param_distributions=param_distribs,
                                    n_iter=10, cv=3, random_state=42, scoring='f1')

lr_rnd_search.fit(X_train_prepared, y_train)

print("best parameter: {}".format(lr_rnd_search.best_params_))
print("best score: {}".format(lr_rnd_search.best_score_))
print("best model: {}".format(lr_rnd_search.best_estimator_))


# In[ ]:


cvres = lr_rnd_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(mean_score, params)


# ### Stochastic Gradient

# In[ ]:


from sklearn.linear_model import SGDClassifier

param_distribs = {
        'penalty' : ['l2', 'l1', 'elasticnet'],
        'alpha': reciprocal(0.0001, 0.1)
    }

sgd_clf = SGDClassifier()

sgd_rnd_search = RandomizedSearchCV(sgd_clf, param_distributions=param_distribs,
                                    n_iter=10, cv=3, random_state=42, scoring='f1')

sgd_rnd_search.fit(X_train_prepared, y_train)

print("best parameter: {}".format(sgd_rnd_search.best_params_))
print("best score: {}".format(sgd_rnd_search.best_score_))
print("best model: {}".format(sgd_rnd_search.best_estimator_))


# In[ ]:


cvres = sgd_rnd_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(mean_score, params)


# ### Random Forest

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
from scipy.stats import randint

param_distribs = {
    'bootstrap':[True, False],
    'n_estimators': randint(low=10, high=15),
    'max_depth': randint(low=40, high=80)
    }

rf_clf = RandomForestClassifier(random_state=42)
rf_rnd_search = RandomizedSearchCV(rf_clf, param_distributions=param_distribs,
                                   n_iter=10, cv=3, random_state=42, scoring='f1')

rf_rnd_search.fit(X_train_prepared, y_train)

print("best parameter: {}".format(rf_rnd_search.best_params_))
print("best score: {}".format(rf_rnd_search.best_score_))
print("best model: {}".format(rf_rnd_search.best_estimator_))


# In[ ]:


cvres = rf_rnd_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(mean_score, params)


# ### Evaluation on the Train Set

# In[ ]:


clf_models = {
    name: search.best_estimator_ for name, search in zip(
    ('LR','SGD','RF'),
    (lr_rnd_search, sgd_rnd_search, rf_rnd_search))}
clf_models


# In[ ]:


print('Train Set Scores')
print('----------------')
for key, clf in clf_models.items():
    print("{}: {:.4f}".format(key, clf.score(X_train_prepared, y_train)))


# <a id="val_set_score"></a>
# # Model Evaluation using Valuation Set

# In[ ]:


from sklearn.metrics import jaccard_score
from sklearn.metrics import f1_score


# In[ ]:


print(val_set.shape)
val_set.head()


# In[ ]:


X_val = val_set.drop(['target','ID_code'], axis=1)

X_val_prepared = preparation_pipeline.transform(X_val)
y_val = val_set['target']

print(X_val_prepared.shape)
print(y_val.shape)

X_val_prepared


# In[ ]:


jaccard_scores = {key:jaccard_score(y_val, clf.predict(X_val_prepared)) for key, clf in clf_models.items()}
jaccard_scores


# In[ ]:


f1_scores = {key:f1_score(y_val, clf.predict(X_val_prepared)) for key, clf in clf_models.items()}
f1_scores


# In[ ]:


print('Valuation Set Scores')
print('----------------')
for key, clf in clf_models.items():
    print("{}: {:.4f}".format(key, clf.score(X_val_prepared, y_val)))


# <a id="predictions"></a>
# # Predictions using Test Set

# In[ ]:


test_set = pd.read_csv('../input/test.csv')
print('test data imported.')


# In[ ]:


print(test_set.shape)
test_set.head()


# In[ ]:


X_test = test_set.drop(['ID_code'], axis=1)

X_test_prepared = preparation_pipeline.transform(X_test)

print(X_test_prepared.shape)
X_test_prepared


# In[ ]:


y_pred = clf_models['SGD'].predict(X_test_prepared)


# In[ ]:


predictions = test_set = pd.read_csv('../input/sample_submission.csv')
print(predictions.shape)
predictions.head()


# In[ ]:


predictions.describe()


# In[ ]:


predictions['target'] = y_pred.ravel()
predictions.describe()


# In[ ]:


predictions.to_csv('submission.csv', index=False)

