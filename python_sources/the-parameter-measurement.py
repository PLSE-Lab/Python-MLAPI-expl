#!/usr/bin/env python
# coding: utf-8

# **INTRODUCTION**
# 
# The concept of overfitting can be equalized to a porpular phenomenon of our domestic Tailors or cloth designers. One day as it may, a customer who initially asked his tailor for a special suit design visited the said Tailor for his ceremonial suit. In pursuit of customer certisfaction in perfecting the sizing, the Tailor in the cause of measuring the customer, he added 2inches each to the size of the customer. The customer in his curiosity ask the reason for the strange action. The Tailor replied in a relaxed voice; if it is your exact size, the suit will too fit(Overfit) and if shorter the suit will not fit (Underfit). If at a certian range, the suit will be moderate (Fit).
# 
# The margin between Overfit and Underfit is a narrow path through the needles hole. Only the sage and the experienced lives their. Overfiting is when model learns and memories both details and noise during training. It is the perfectly fit illustrated in our story. The main effect of this is that it has a negative effect on a new data.
# 
# Some of the few methods in solving this methods are:
# 
# 1. Training with more data
# 
# 2. Cross-Validation
# 
# 3. Removal of irrelevant features
# 
# 4. Early stopping
# 
# 5. Ensembling
# 
# 6. Regularization
# 
# In this kernel, I will be treating the data provided some of this solving skills. Four models will be practiced and prediction will be made. I will also be during my first NN model here.
# 
# I just hope this kernel serves you well and solve some questions in your mind.
# 
# Now lets get the ball rolling..............

# **Preparing Data for Analysis**
# 
# Loading Packages

# In[1]:


import numpy as np
import pandas as pd
pd.set_option('max_columns', None)
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from matplotlib.colors import ListedColormap
import datetime
import lightgbm as lgb
from scipy import stats
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold, cross_val_score, GridSearchCV, RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score
import os
from scipy.stats import zscore
from sklearn import metrics
from sklearn.model_selection import KFold
from keras.layers.core import Dense, Activation
import tensorflow as tf
from tensorflow import keras
import keras
from sklearn.datasets import make_moons
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense
from keras import models
from keras import layers
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.datasets import make_classification
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
import xgboost as xgb
import lightgbm as lgb
from sklearn import model_selection
from sklearn.metrics import accuracy_score, roc_auc_score
import json
import ast
import time
from sklearn import linear_model
import eli5
from eli5.sklearn import PermutationImportance
import shap
import gc
import itertools
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_selection import GenericUnivariateSelect, SelectPercentile, SelectKBest, f_classif, mutual_info_classif, RFE
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')
from catboost import CatBoostClassifier


# **Loading Data**
# 
# We will check the data files that are available and also print its formats. This helps us to know the real data we are working with. The nature of the data and how the data was collated. This is very crucial in for data cleansing and visualisation. The first code below is to access the location of the file and to check for the available file.
# 
# The data contains 20000 rows of continues variable and mere handful of training sample (250). This is a problem as it sounds. We will be faced with small sample data that can already cause overfit.

# In[2]:


IS_LOCAL = False
if (IS_LOCAL):
    location = "../input/dont-overfit/"
else:
    location = "../input/"
os.listdir(location)


# In[3]:


get_ipython().run_cell_magic('time', '', "train = pd.read_csv(os.path.join(location, 'train.csv'))\ntest = pd.read_csv(os.path.join(location, 'test.csv'))\nsample_submission = pd.read_csv(os.path.join(location, 'sample_submission.csv'))")


# I will like to print the shape of the files to know the topography of my data. The train data has 250 rows with 302 features while test data has 19750 rows with 301 features. Their is unbalance and the train data is too small. Let us dig deep into the data.

# In[4]:


print("train: {}\ntest: {}".format(train.shape, test.shape))


# **Data Exploration**
# 
# Now we can see that the train set has id and target while the test set has the id row. If this two are removed the the data will be equal in features(columns). Also the data are all in variables with no alphabets. We need to go further in checking the missing values and seperating the target as label.

# In[5]:


def show_head(data):
    return(data.head())


# In[6]:


show_head(train)


# In[7]:


show_head(test)


# I will create a function to check missing variables and also to show the type of variables. I noticed that theirs is no missing values and the variables are all floats.

# In[8]:


def missing_data(data):
    total = data.isnull().sum()
    percent = (total/data.isnull().count()*100)
    miss_column = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    types = []
    for col in data.columns:
        dtype = str(data[col].dtype)
        types.append(dtype)
    miss_column['Types'] = types
    return(np.transpose(miss_column)) 


# In[9]:


missing_data(train)


# In[10]:


missing_data(test)


#  I will now creat a function called describe_data, this will describe our data. We will know waht exactly the problem is in this function. this function will show us the mean, std, counts and max and min of each measurement. it seems the measurement are categorised into series.

# In[11]:


def describe_data(data):
    return(data.describe())


# In[12]:


describe_data(train)


# In[13]:


describe_data(test)


# I will now seperate the Label data and find the type of classification we are dealing with.
# 
# From the code below, we noticed that the target is a binary target with some level of unbalance. The columns are more similar

# In[15]:


label = train['target']
train = train.drop(['id', 'target'], axis=1)
test = test.drop(['id'], axis=1)


# In[16]:


label_count = label.value_counts().reset_index().rename(columns = {'index' : 'Labels'})
label_count


# This shows that we are dealing with an unbalance data set. The data for training is samll and unbalance.

# In[17]:


show_head(train)


# In[18]:


show_head(label)


# I will briefly show the density plot of the columns to have a better understanding on how the columns are distributed.
# 
# All the columns are normally distributed with a shape peak.

# I will like to reduce the dimmension of my data to ease visualization. I will be using principal component anaalysis to achieve this. PCA is essentially a method that reduces the dimention of the feature space in such a way that new variables are orthogonal to each other.
# 
# To use PCA, we need to apply scale standard to unitarilize our units of dimention.
# 
# I will first plot a PCA graph to determine the number of components we need in apply PCA. This plot tells us that selecting 205 components we will preserve something around 96 percent of the total variance of the data. Not using all the component  makes us to use only the principle ones. I will now go forward by using the components suggested for me by PCA. 

# In[19]:


sc = StandardScaler()
xtrain = sc.fit_transform(train)
xtest = sc.transform(test)

pca = PCA().fit(xtrain)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance');


# In[20]:


print("xtrain: {}\nxtest: {}\nlabel: {}".format(xtrain.shape, xtest.shape, label.shape))


# In[21]:


missing_data(train)


# Now we will use Kfold cross validation to compare with our leave one out cross validation

# **RandomForestClassifier with KFold cross validation**

# In[23]:


from sklearn.metrics import classification_report as c_report
from sklearn.metrics import confusion_matrix as c_matrix
folds = StratifiedKFold(n_splits=10, shuffle=True, random_state=4422)
forcast_rfc2 = np.zeros(len(xtest))
validation_pred_rfc2 = np.zeros(len(xtrain))
scores_rfc2 = []
valid_rfc2 = np.zeros(len(label))
for fold_, (xtrn_indx, xvad_indx) in enumerate(folds.split(xtrain, label)):
    print('Fold {}, started at {}'.format(fold_, time.ctime()))
    x_train, x_valid = xtrain[xtrn_indx], xtrain[xvad_indx]
    y_train, y_valid = label[xtrn_indx], label[xvad_indx]
    
    clf_rfc2 = RandomForestClassifier(n_estimators=1000, n_jobs=-1, random_state=0)
    clf_rfc2.fit(x_train, y_train)
    pred_valid = clf_rfc2.predict(x_valid).reshape(-1,)
    score = accuracy_score(y_valid, pred_valid)
    Pred_Real = clf_rfc2.predict_proba(xtest)[:, 1]
    
    validation_pred_rfc2[xvad_indx] += (pred_valid).reshape(-1,)
    scores_rfc2.append(accuracy_score(y_valid, pred_valid))
    forcast_rfc2 += Pred_Real
    valid_rfc2[xvad_indx] += (y_valid)

print(c_report(valid_rfc2[xvad_indx], validation_pred_rfc2[xvad_indx]))
print(c_matrix(valid_rfc2[xvad_indx], validation_pred_rfc2[xvad_indx]))
print('accuracy is: {}, std: {}.'.format(np.mean(scores_rfc2), np.std(scores_rfc2)))


# **DecisionTreeClassifier**

# In[24]:


from sklearn.tree import DecisionTreeClassifier
forcast_dtc2 = np.zeros(len(xtest))
validation_pred_dtc2 = np.zeros(len(xtrain))
scores_dtc2 = []
valid_dtc2 = np.zeros(len(label))
for fold_, (xtrn_indx, xvad_indx) in enumerate(folds.split(xtrain, label)):
    print('Fold {}, started at {}'.format(fold_, time.ctime()))
    x_train, x_valid = xtrain[xtrn_indx], xtrain[xvad_indx]
    y_train, y_valid = label[xtrn_indx], label[xvad_indx]
    
    clf_dtc2 = DecisionTreeClassifier()
    clf_dtc2.fit(x_train, y_train)
    pred_valid = clf_dtc2.predict(x_valid).reshape(-1,)
    score = accuracy_score(y_valid, pred_valid)
    Pred_Real = clf_dtc2.predict_proba(xtest)[:, 1]
    
    validation_pred_dtc2[xvad_indx] += (pred_valid).reshape(-1,)
    scores_dtc2.append(accuracy_score(y_valid, pred_valid))
    forcast_dtc2 += Pred_Real
    valid_dtc2[xvad_indx] += (y_valid)

print(c_report(valid_dtc2[xvad_indx], validation_pred_dtc2[xvad_indx]))
print(c_matrix(valid_dtc2[xvad_indx], validation_pred_dtc2[xvad_indx]))
print('accuracy is: {}, std: {}.'.format(np.mean(scores_dtc2), np.std(scores_dtc2)))


# **svm**

# In[25]:


from sklearn import svm
forcast_svm2 = np.zeros(len(xtest))
validation_pred_svm2 = np.zeros(len(xtrain))
scores_svm2 = []
valid_svm2 = np.zeros(len(label))
for fold_, (xtrn_indx, xvad_indx) in enumerate(folds.split(xtrain, label)):
    print('Fold {}, started at {}'.format(fold_, time.ctime()))
    x_train, x_valid = xtrain[xtrn_indx], xtrain[xvad_indx]
    y_train, y_valid = label[xtrn_indx], label[xvad_indx]
    
    clf_svm2 = svm.SVC(kernel='linear', gamma=1)
    clf_svm2.fit(x_train, y_train)
    pred_valid = clf_svm2.predict(x_valid).reshape(-1,)
    score = accuracy_score(y_valid, pred_valid)
    Pred_Real = clf_svm2.predict(xtest)
    
    validation_pred_svm2[xvad_indx] += (pred_valid).reshape(-1,)
    scores_svm2.append(accuracy_score(y_valid, pred_valid))
    forcast_svm2 += Pred_Real
    valid_svm2[xvad_indx] += (y_valid)

print(c_report(valid_svm2[xvad_indx], validation_pred_svm2[xvad_indx]))
print(c_matrix(valid_svm2[xvad_indx], validation_pred_svm2[xvad_indx]))
print('accuracy is: {}, std: {}.'.format(np.mean(scores_svm2), np.std(scores_svm2)))


# **LogisticRegression**

# In[26]:


forcast_lr2 = np.zeros(len(xtest))
validation_pred_lr2 = np.zeros(len(xtrain))
scores_lr2 = []
valid_lr2 = np.zeros(len(label))
for fold_, (xtrn_indx, xvad_indx) in enumerate(folds.split(xtrain, label)):
    print('Fold {}, started at {}'.format(fold_, time.ctime()))
    x_train, x_valid = xtrain[xtrn_indx], xtrain[xvad_indx]
    y_train, y_valid = label[xtrn_indx], label[xvad_indx]
    
    clf_lr2 = linear_model.LogisticRegression(class_weight='balanced', penalty='l1', C=0.1, solver='liblinear')
    clf_lr2.fit(x_train, y_train)
    pred_valid = clf_lr2.predict(x_valid).reshape(-1,)
    score = accuracy_score(y_valid, pred_valid)
    Pred_Real = clf_lr2.predict_proba(xtest)[:, 1]
    
    validation_pred_lr2[xvad_indx] += (pred_valid).reshape(-1,)
    scores_lr2.append(accuracy_score(y_valid, pred_valid))
    forcast_lr2 += Pred_Real
    valid_lr2[xvad_indx] += (y_valid)

print(c_report(valid_lr2[xvad_indx], validation_pred_lr2[xvad_indx]))
print(c_matrix(valid_lr2[xvad_indx], validation_pred_lr2[xvad_indx]))
print('accuracy is: {}, std: {}.'.format(np.mean(scores_lr2), np.std(scores_lr2)))


# Using ELI5 and permutation importance helps us to know how weights are been ascribed to each features in this Lr model.
# The code below shows the feature importance of the model.
# 
# The first is using elif while the second is using permutation importance.

# In[27]:


perm = PermutationImportance(clf_lr2, random_state=1).fit(xtrain, label)
eli5.show_weights(perm, top=50)


# We quickly checked the features with non zero weight. 38 features has weight from all the features.

# In[28]:


(clf_lr2.coef_ != 0).sum()


# We will now use the top features with great importance to train our model and also to predict to see the diffence with the former moddel.

# In[29]:


top_features = [i[1:] for i in eli5.formatters.as_dataframe.explain_weights_df(clf_lr2).feature if 'BIAS' not in i]
xtrain_lr_elif = train[top_features]
xtest_lr_elif = test[top_features]
scaler = StandardScaler()
xtrain_lr_elif = scaler.fit_transform(xtrain_lr_elif)
xtest_lr_elif = scaler.transform(xtest_lr_elif)

forcast_lr4 = np.zeros(len(xtest_lr_elif))
validation_pred_lr4 = np.zeros(len(xtrain_lr_elif))
scores_lr4 = []
valid_lr4 = np.zeros(len(label))
for fold_, (xtrn_indx, xvad_indx) in enumerate(folds.split(xtrain_lr_elif, label)):
    print('Fold {}, started at {}'.format(fold_, time.ctime()))
    x_train, x_valid = xtrain_lr_elif[xtrn_indx], xtrain_lr_elif[xvad_indx]
    y_train, y_valid = label[xtrn_indx], label[xvad_indx]
    
    clf_lr4 = linear_model.LogisticRegression(class_weight='balanced', penalty='l1', C=0.1, solver='liblinear')
    clf_lr4.fit(x_train, y_train)
    pred_valid = clf_lr4.predict(x_valid).reshape(-1,)
    score = accuracy_score(y_valid, pred_valid)
    Pred_Real = clf_lr4.predict_proba(xtest_lr_elif)[:, 1]
    
    validation_pred_lr4[xvad_indx] += (pred_valid).reshape(-1,)
    scores_lr4.append(accuracy_score(y_valid, pred_valid))
    forcast_lr4 += Pred_Real
    valid_lr4[xvad_indx] += (y_valid)

print(c_report(valid_lr4[xvad_indx], validation_pred_lr4[xvad_indx]))
print(c_matrix(valid_lr4[xvad_indx], validation_pred_lr4[xvad_indx]))
print('accuracy is: {}, std: {}.'.format(np.mean(scores_lr4), np.std(scores_lr4)))


# wow we have an increase in the score of the model from 0.708 to 0.74 by using the model to explain the weight......this is a good improvement but yet we need to check on the leaderboard.
# 
# Now let us try permutation importance to explain the weight.

# In[30]:


perm = PermutationImportance(clf_lr2, random_state=1).fit(xtrain, label)
eli5.show_weights(perm, top=50)


# In[32]:


top_features = [i[1:] for i in eli5.formatters.as_dataframe.explain_weights_df(perm).feature if 'BIAS' not in i]
xtrain_lr_perm = train[top_features]
xtest_lr_perm = test[top_features]
scaler = StandardScaler()
xtrain_lr_perm = scaler.fit_transform(xtrain_lr_perm)
xtest_lr_perm = scaler.transform(xtest_lr_perm)

forcast_lr3 = np.zeros(len(xtest_lr_perm))
validation_pred_lr3 = np.zeros(len(xtrain_lr_perm))
scores_lr3 = []
valid_lr3 = np.zeros(len(label))
for fold_, (xtrn_indx, xvad_indx) in enumerate(folds.split(xtrain_lr_perm, label)):
    print('Fold {}, started at {}'.format(fold_, time.ctime()))
    x_train, x_valid = xtrain_lr_perm[xtrn_indx], xtrain_lr_perm[xvad_indx]
    y_train, y_valid = label[xtrn_indx], label[xvad_indx]
    
    clf_lr3 = linear_model.LogisticRegression(class_weight='balanced', penalty='l1', C=0.1, solver='liblinear')
    clf_lr3.fit(x_train, y_train)
    pred_valid = clf_lr3.predict(x_valid).reshape(-1,)
    score = accuracy_score(y_valid, pred_valid)
    Pred_Real = clf_lr3.predict_proba(xtest_lr_perm)[:, 1]
    
    validation_pred_lr3[xvad_indx] += (pred_valid).reshape(-1,)
    scores_lr3.append(accuracy_score(y_valid, pred_valid))
    forcast_lr3 += Pred_Real
    valid_lr3[xvad_indx] += (y_valid)

print(c_report(valid_lr3[xvad_indx], validation_pred_lr3[xvad_indx]))
print(c_matrix(valid_lr3[xvad_indx], validation_pred_lr3[xvad_indx]))
print('accuracy is: {}, std: {}.'.format(np.mean(scores_lr3), np.std(scores_lr3)))


# This reduces the score. I guess it does not work for us.

# **BaggingClassifier**

# In[33]:


from sklearn.ensemble import BaggingClassifier
forcast_bc = np.zeros(len(xtest))
validation_pred_bc = np.zeros(len(xtrain))
scores_bc = []
valid_bc = np.zeros(len(label))
for fold_, (xtrn_indx, xvad_indx) in enumerate(folds.split(xtrain, label)):
    print('Fold {}, started at {}'.format(fold_, time.ctime()))
    x_train, x_valid = xtrain[xtrn_indx], xtrain[xvad_indx]
    y_train, y_valid = label[xtrn_indx], label[xvad_indx]
    
    clf_bc = BaggingClassifier()
    clf_bc.fit(x_train, y_train)
    pred_valid = clf_bc.predict(x_valid).reshape(-1,)
    score = accuracy_score(y_valid, pred_valid)
    Pred_Real = clf_bc.predict_proba(xtest)[:, 1]
    
    validation_pred_bc[xvad_indx] += (pred_valid).reshape(-1,)
    scores_bc.append(accuracy_score(y_valid, pred_valid))
    forcast_bc += Pred_Real
    valid_bc[xvad_indx] += (y_valid)

print(c_report(valid_bc[xvad_indx], validation_pred_bc[xvad_indx]))
print(c_matrix(valid_bc[xvad_indx], validation_pred_bc[xvad_indx]))
print('accuracy is: {}, std: {}.'.format(np.mean(scores_bc), np.std(scores_bc)))


# **AdaBoostClassifier**

# In[34]:


from sklearn.ensemble import AdaBoostClassifier
forcast_adac = np.zeros(len(xtest))
validation_pred_adac = np.zeros(len(xtrain))
scores_adac = []
valid_adac = np.zeros(len(label))
for fold_, (xtrn_indx, xvad_indx) in enumerate(folds.split(xtrain, label)):
    print('Fold {}, started at {}'.format(fold_, time.ctime()))
    x_train, x_valid = xtrain[xtrn_indx], xtrain[xvad_indx]
    y_train, y_valid = label[xtrn_indx], label[xvad_indx]
    
    clf_adac = AdaBoostClassifier()
    clf_adac.fit(x_train, y_train)
    pred_valid = clf_adac.predict(x_valid).reshape(-1,)
    score = accuracy_score(y_valid, pred_valid)
    Pred_Real = clf_adac.predict_proba(xtest)[:, 1]
    
    validation_pred_adac[xvad_indx] += (pred_valid).reshape(-1,)
    scores_adac.append(accuracy_score(y_valid, pred_valid))
    forcast_adac += Pred_Real
    valid_adac[xvad_indx] += (y_valid)

print(c_report(valid_adac[xvad_indx], validation_pred_adac[xvad_indx]))
print(c_matrix(valid_adac[xvad_indx], validation_pred_adac[xvad_indx]))
print('accuracy is: {}, std: {}.'.format(np.mean(scores_adac), np.std(scores_adac)))


# **GradientBoostingClassifier**

# In[35]:


from sklearn.ensemble import GradientBoostingClassifier
forcast_gbc = np.zeros(len(xtest))
validation_pred_gbc = np.zeros(len(xtrain))
scores_gbc = []
valid_gbc = np.zeros(len(label))
for fold_, (xtrn_indx, xvad_indx) in enumerate(folds.split(xtrain, label)):
    print('Fold {}, started at {}'.format(fold_, time.ctime()))
    x_train, x_valid = xtrain[xtrn_indx], xtrain[xvad_indx]
    y_train, y_valid = label[xtrn_indx], label[xvad_indx]
    
    clf_gbc = GradientBoostingClassifier()
    clf_gbc.fit(x_train, y_train)
    pred_valid = clf_gbc.predict(x_valid).reshape(-1,)
    score = accuracy_score(y_valid, pred_valid)
    Pred_Real = clf_gbc.predict_proba(xtest)[:, 1]
    
    validation_pred_gbc[xvad_indx] += (pred_valid).reshape(-1,)
    scores_gbc.append(accuracy_score(y_valid, pred_valid))
    forcast_gbc += Pred_Real
    valid_gbc[xvad_indx] += (y_valid)

print(c_report(valid_gbc[xvad_indx], validation_pred_gbc[xvad_indx]))
print(c_matrix(valid_gbc[xvad_indx], validation_pred_gbc[xvad_indx]))
print('accuracy is: {}, std: {}.'.format(np.mean(scores_gbc), np.std(scores_gbc)))


# **CatBoostingClassifier**

# In[36]:


cat_params = {'learning_rate': 0.02,
              'depth': 5,
              'l2_leaf_reg': 10,
              'bootstrap_type': 'Bernoulli',
              'od_type': 'Iter',
              'od_wait': 50,
              'random_seed': 11,
              'allow_writing_files': False}
forcast_catc = np.zeros(len(xtest))
validation_pred_catc = np.zeros(len(xtrain))
scores_catc = []
valid_catc = np.zeros(len(label))
for fold_, (xtrn_indx, xvad_indx) in enumerate(folds.split(xtrain, label)):
    x_train, x_valid = xtrain[xtrn_indx], xtrain[xvad_indx]
    y_train, y_valid = label[xtrn_indx], label[xvad_indx]
    
    clf_catc = CatBoostClassifier(iterations=400, **cat_params)
    clf_catc.fit(x_train, y_train)
    pred_valid = clf_catc.predict(x_valid).reshape(-1,)
    score = accuracy_score(y_valid, pred_valid)
    Pred_Real = clf_catc.predict_proba(xtest)[:, 1]
    
    validation_pred_catc[xvad_indx] += (pred_valid).reshape(-1,)
    scores_catc.append(accuracy_score(y_valid, pred_valid))
    forcast_catc += Pred_Real
    valid_catc[xvad_indx] += (y_valid)

print(c_report(valid_catc[xvad_indx], validation_pred_catc[xvad_indx]))
print(c_matrix(valid_catc[xvad_indx], validation_pred_catc[xvad_indx]))
print('accuracy is: {}, std: {}.'.format(np.mean(scores_catc), np.std(scores_catc)))


# In[ ]:





# In[37]:


sample_submission['target'] = forcast_lr4
sample_submission.to_csv('Forcasting.csv', index=False)
sample_submission.head(20)


# to be continued...........
