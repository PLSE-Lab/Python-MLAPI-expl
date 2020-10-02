#!/usr/bin/env python
# coding: utf-8

# # Categorical Feature Encoding:
# 
# ## Introduction:
# 
# In most data science problems, our datasets will contain categorical features. Categorical features contain a finite number of discrete values. How we represent these features will have an impact on the performance of our model. Like in other aspects of machine learning, there are no silver bullets. Determining the correct approach, specific to our model and data is part of the challenge.
# 
# This tutorial aims to cover a few of these methods. We begin by covering a straight-forward technique before tackling more complicated lesser-known approaches.
# 
# **List of methods covered**:
# 1. One-Hot Encoding
# 2. Feature Hashing
# 3. Binary Encoding
# 4. Target Encoding
# 5. Weight of Evidence

# In[ ]:


# Import required libraries:
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import os
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Set our random seed:
SEED = 17
PATH_TO_DIR = '../input/amazoncom-employee-access-challenge/'

print(os.listdir(PATH_TO_DIR))


# For this tutorial, we will be using the '[Amazon.com Employee Access Challenge](https://www.kaggle.com/c/amazon-employee-access-challenge)' dataset. This binary classification dataset is made up of strictly categorical features, which are already converted into numerals, making it a particularly suitable choice to explore various encoding techniques. To simplify things we will only be using a subset of the features for this demonstration.

# In[ ]:


# Import data:
train = pd.read_csv(PATH_TO_DIR + 'train.csv')


# In[ ]:


y = train['ACTION']
train = train[['RESOURCE', 'MGR_ID', 'ROLE_FAMILY_DESC', 'ROLE_FAMILY', 'ROLE_CODE']]


# We will compare differences of these encoding methods on both a linear model and tree-based model. These represent two families of models which have contrasting behaviours when it comes to different feature representations.

# In[ ]:


logit = LogisticRegression(random_state=SEED)
rf = RandomForestClassifier(random_state=SEED)


# In[ ]:


# Split dataset into train and validation subsets:
X_train, X_val, y_train, y_val = train_test_split(train, y, test_size=0.2, random_state=SEED)


# In[ ]:


# We create a helper function to get the scores for each encoding method:
def get_score(model, X, y, X_val, y_val):
    model.fit(X, y)
    y_pred = model.predict_proba(X_val)[:,1]
    score = roc_auc_score(y_val, y_pred)
    return score


# In[ ]:


# Lets have a quick look at our data:
X_train.head(5)


# In[ ]:


X_train.info()


# In[ ]:


# Discover the number of categories within each categorical feature:
len(X_train.RESOURCE.unique()), len(X_train.MGR_ID.unique()), len(X_train.ROLE_FAMILY_DESC.unique()), len(X_train.ROLE_FAMILY.unique()),len(X_train.ROLE_CODE.unique())


# In[ ]:


# Create a list of each categorical column name:
columns = [i for i in X_train.columns]


# Before getting started, lets have a look at the speed and performance of training these models without any feature encoding.

# In[ ]:


get_ipython().run_cell_magic('time', '', "baseline_logit_score = get_score(logit, X_train, y_train, X_val, y_val)\nprint('Logistic Regression score without feature engineering:', baseline_logit_score)")


# In[ ]:


get_ipython().run_cell_magic('time', '', "baseline_rf_score = get_score(rf, X_train, y_train, X_val, y_val)\nprint('Random Forest score without feature engineering:', baseline_rf_score)")


# ## One-Hot Encoding:

# The first method we will be covering is one that no doubt will be familiar to you. One-hot encoding expands a categorical feature made up of m categories into m* distinct features with values of either 0 or 1.
# 
# There are two ways of implementing one-hot encoding, either with pandas or scikit-learn. In this tutorial we have chosen to use the latter.
# 
# *Actually, it is seen as more correct to expand m categories into (m - 1) distinct features. The reason for this is twofold. Firstly, if the values of (m - 1) features are known, the m-th feature can be inferred and secondly because including the m-th feature can cause certain linear models to become unstable. More on that can be found [here](https://www.algosome.com/articles/dummy-variable-trap-regression.html). In practice I think this depends on your model. Some non-linear models actually do better with m features.

# In[ ]:


from sklearn.preprocessing import OneHotEncoder

one_hot_enc = OneHotEncoder(sparse=False)


# In[ ]:


print('Original number of features: \n', X_train.shape[1], "\n")
data_ohe_train = (one_hot_enc.fit_transform(X_train))
data_ohe_val = (one_hot_enc.transform(X_val))
print('Features after OHE: \n', data_ohe_train.shape[1])


# In[ ]:


get_ipython().run_cell_magic('time', '', "ohe_logit_score = get_score(logit, data_ohe_train, y_train, data_ohe_val, y_val)\nprint('Logistic Regression score with one-hot encoding:', ohe_logit_score)")


# In[ ]:


get_ipython().run_cell_magic('time', '', "ohe_rf_score = get_score(rf, data_ohe_train, y_train, data_ohe_val, y_val)\nprint('Random Forest score with one-hot encoding:', ohe_rf_score)")


# As we can see, while the performance of the model has improved, training took longer as well. This is due to the increase in the number of features. Computational costs are not the only problem associated with the increase in dimensions. A dataset with more features will require a model with more parameters which in turn will require more data to train these parameters. In many cases, such as kaggle competitions, the size of our data is fixed and as a result the dimensionality of our data should always be a concern.
# 
# One way of dealing with high dimensionality is by compressing the features. Feature hashing, which we will be covering next, is an example of this.

# ## Feature Hashing:

# Feature hashing maps each category in a categorical feature to an integer within a pre-determined range. This output range is smaller than the input range so multiple categories may be mapped to the same integer. Feature hashing is very similar to one-hot encoding but with a control over the output dimensions.
# 
# To implement feature hashing in python we can use category_encoder, a library containing sklearn compabitable category encoders.

# In[ ]:


# Install category_encoders:
# pip install category_encoders


# In[ ]:


from category_encoders import HashingEncoder


# The size of the output dimensions is controlled by the variable n_components. This can be treated as a hyperparameter.

# In[ ]:


n_components_list = [100, 500, 1000, 5000, 10000]
n_components_list_str = [str(i) for i in n_components_list]


# In[ ]:


fh_logit_scores = []

# Iterate over different n_components:
for n_components in n_components_list:
    
    hashing_enc = HashingEncoder(cols=columns, n_components=n_components).fit(X_train, y_train)
    
    X_train_hashing = hashing_enc.transform(X_train.reset_index(drop=True))
    X_val_hashing = hashing_enc.transform(X_val.reset_index(drop=True))
    
    fe_logit_score = get_score(logit, X_train_hashing, y_train, X_val_hashing, y_val)
    fh_logit_scores.append(fe_logit_score)


# In[ ]:


plt.figure(figsize=(8, 5))
plt.plot(n_components_list_str, fh_logit_scores, linewidth=3)
plt.title('n_compontents vs roc_auc for feature hashing with logistic regression')
plt.xlabel('n_components')
plt.ylabel('score')
plt.show;


# As we can see, performance on the Logistic Regression model improves as the number of components increase. But let us have a look at the effect of reducing the dimensions has on a Random Forest model.

# In[ ]:


hashing_enc = HashingEncoder(cols=columns, n_components=10000).fit(X_train, y_train)

X_train_hashing = hashing_enc.transform(X_train.reset_index(drop=True))
X_val_hashing = hashing_enc.transform(X_val.reset_index(drop=True))


# In[ ]:


X_train_hashing.head()


# In[ ]:


get_ipython().run_cell_magic('time', '', "hashing_logit_score = get_score(logit, X_train_hashing, y_train, X_val_hashing, y_val)\nprint('Logistic Regression score with feature hashing:', hashing_logit_score)")


# In[ ]:


get_ipython().run_cell_magic('time', '', "hashing_rf_score = get_score(rf, X_train_hashing, y_train, X_val_hashing, y_val)\nprint('Random Forest score with feature hashing:', hashing_rf_score)")


# It improves! As we may have guessed, reducing the number of features improves the performance of tree-based models. 

# ## Binary Encoding:

# Binary encoding involves converting each category into a binary code, for example 2 becomes 11 and 3 becomes 100, and then splitting the resulting binary string into columns. 
# 
# This may be easier to understand with an example:

# In[ ]:


# Create example dataframe with numbers ranging from 1 to 5:
example_df = pd.DataFrame([1,2,3,4,5], columns=['example'])

from category_encoders import BinaryEncoder

example_binary = BinaryEncoder(cols=['example']).fit_transform(example_df)

example_binary


# Binary encoding is clearly very similar to feature hashing however much more restricted. In practice using feature hashing is often advised over binary encoding due to the control you have over the output dimensions.

# In[ ]:


binary_enc = BinaryEncoder(cols=columns).fit(X_train, y_train)


# In[ ]:


X_train_binary = binary_enc.transform(X_train.reset_index(drop=True))
X_val_binary = binary_enc.transform(X_val.reset_index(drop=True))
# note: category_encoders implementations can't handle shuffled datasets. 


# In[ ]:


print('Features after Binary Encoding: \n', X_train_binary.shape[1])


# In[ ]:


get_ipython().run_cell_magic('time', '', "be_logit_score = get_score(logit, X_train_binary, y_train, X_val_binary, y_val)\nprint('Logistic Regression score with binary encoding:', be_logit_score)")


# In[ ]:


get_ipython().run_cell_magic('time', '', "binary_rf_score = get_score(rf, X_train_binary, y_train, X_val_binary, y_val)\nprint('Random Forest score with binary encoding:', binary_rf_score)")


# ## Target Encoding:

# Target encoding is the first of our Bayesian encoders. These are a family of encoders which take information about the target variable into account. Target encoding may refer to an encoder which considers the statistical correlation between the individual categories of a categorical feature. In this tutorial we will only look at target encoders which focus on the relationship between each category and the mean of the target as this is the most commonly used variation of target encoding.

# In[ ]:


from category_encoders import TargetEncoder

targ_enc = TargetEncoder(cols=columns, smoothing=8, min_samples_leaf=5).fit(X_train, y_train)


# In[ ]:


X_train_te = targ_enc.transform(X_train.reset_index(drop=True))
X_val_te = targ_enc.transform(X_val.reset_index(drop=True))


# In[ ]:


X_train_te.head()


# In[ ]:


get_ipython().run_cell_magic('time', '', "te_logit_score = get_score(logit, X_train_te, y_train, X_val_te, y_val)\nprint('Logistic Regression score with target encoding:', te_logit_score)")


# In[ ]:


get_ipython().run_cell_magic('time', '', "te_rf_score = get_score(rf, X_train_te, y_train, X_val_te, y_val)\nprint('Random Forest score with target encoding:', te_rf_score)")


# As a result of using the target variable, data-leakage and overfitting is a huge concern. The category_encoders implementation has two out of the box ways of regularizing the encodings, 'smoothing' and 'min_samples_leaf'. These parameters may be treated as hyperparameters.
# 
# 'smoothing' determines the weighting of the individual category's mean with the mean of the entire categorical variable. This is to prevent the influence of unreliable means from categories with low sample sizes.
# 
# 'min_samples_leaf' is the minimum number of samples within a category to take it's mean into account.

# In[ ]:


targ_enc = TargetEncoder(cols=columns, smoothing=8, min_samples_leaf=5).fit(X_train, y_train)

X_train_te = targ_enc.transform(X_train.reset_index(drop=True))
X_val_te = targ_enc.transform(X_val.reset_index(drop=True))


# In[ ]:


get_ipython().run_cell_magic('time', '', "me_logit_score = get_score(logit, X_train_te, y_train, X_val_te, y_val)\nprint('Logistic Regression score with target encoding with regularization:', me_logit_score)")


# In[ ]:


get_ipython().run_cell_magic('time', '', "me_rf_score = get_score(rf, X_train_te, y_train, X_val_te, y_val)\nprint('Random Forest score with target encoding with regularization:', me_rf_score)")


# Another approach to regularizing the target encoder is to calculate the statistic relationship between each category and the target variable via a kfold split. This method is currently not available in category_encoders implementation and needs to be written from scratch.

# In[ ]:


from sklearn.model_selection import KFold

# Create 5 kfold splits:
kf = KFold(random_state=17, n_splits=5, shuffle=False)


# In[ ]:


# Create copy of data:
X_train_te = X_train.copy()
X_train_te['target'] = y_train


# In[ ]:


all_set = []

for train_index, val_index in kf.split(X_train_te):
    # Create splits:
    train, val = X_train_te.iloc[train_index], X_train_te.iloc[val_index]
    val=val.copy()
    
    # Calculate the mean of each column:
    means_list = []
    for col in columns:
        means_list.append(train.groupby(str(col)).target.mean())
    
    # Calculate the mean of each category in each column:
    col_means = []
    for means_series in means_list:
        col_means.append(means_series.mean())
    
    # Encode the data:
    for column, means_series, means in zip(columns, means_list, col_means):
        val[str(column) + '_target_enc'] = val[str(column)].map(means_series).fillna(means) 
    
    list_of_mean_enc = [str(column) + '_target_enc' for column in columns]
    list_of_mean_enc.extend(columns)
    
    all_set.append(val[list_of_mean_enc].copy())

X_train_te=pd.concat(all_set, axis=0)


# In[ ]:


# Apply encodings to validation set:
X_val_te = pd.DataFrame(index=X_val.index)
for column, means in zip(columns, col_means):
    enc_dict = X_train_te.groupby(column).mean().to_dict()[str(column) + '_target_enc']
    X_val_te[column] = X_val[column].map(enc_dict).fillna(means)


# In[ ]:


# Create list of target encoded columns:
list_of_target_enc = [str(column) + '_target_enc' for column in columns]


# In[ ]:


get_ipython().run_cell_magic('time', '', "kf_reg_logit_score = get_score(logit, X_train_te[list_of_target_enc], y_train, X_val_te, y_val)\nprint('Logistic Regression score with kfold-regularized target encoding:', kf_reg_logit_score)")


# In[ ]:


get_ipython().run_cell_magic('time', '', "kf_reg_rf_score = get_score(rf, X_train_te[list_of_target_enc], y_train, X_val_te, y_val)\nprint('Random Forest score with kfold-regularized target encoding:', kf_reg_rf_score)")


# ## Weight Of Evidence (WOE):

# Weight of evidence (WOE) encoder calculates the natural log of the % of non-events divided by the % of events for each category within a categotical feature. For clarification, the events are referring to the target variable. 

# In[ ]:


from category_encoders import WOEEncoder

woe_enc = WOEEncoder(cols=columns, random_state=17).fit(X_train, y_train)


# In[ ]:


X_train_woe = woe_enc.transform(X_train.reset_index(drop=True))
X_val_woe = woe_enc.transform(X_val.reset_index(drop=True))


# In[ ]:


X_train_woe.head()


# In[ ]:


get_ipython().run_cell_magic('time', '', "woe_logit_score = get_score(logit, X_train_woe, y_train, X_val_woe, y_val)\nprint('Logistic Regression score with woe encoding:', woe_logit_score)")


# In[ ]:


get_ipython().run_cell_magic('time', '', "woe_rf_score = get_score(rf, X_train_woe, y_train, X_val_woe, y_val)\nprint('Random Forest score with woe encoding:', woe_rf_score)")


# In summary, categorical features may be represented in more ways than the traditional one-hot encoding. These representations have different effects on our models and the choice of representation is task specific. Feature hashing and binary encoding offer us ways of encoding the data with lower dimensions which is cheaper computationally as well as being better suited for tree-based models. Target encoding and weight of evidence encoding seem to be much more task specific. 
# 
# Feedback would be appreciated, as well as upvotes! Thank you.

# ### Further Reading:
# 
# * [category_encoder documentation](http://contrib.scikit-learn.org/categorical-encoding/)
# * [weight of evidence](https://www.listendata.com/2015/03/weight-of-evidence-woe-and-information.html)
# * [smarter ways of encoding categorical data for machine learning](https://towardsdatascience.com/smarter-ways-to-encode-categorical-data-for-machine-learning-part-1-of-3-6dca2f71b159)
# * [an exploration of categorical variables](http://www.willmcginnis.com/2015/11/29/beyond-one-hot-an-exploration-of-categorical-variables/)
