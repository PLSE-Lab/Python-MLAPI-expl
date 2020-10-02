#!/usr/bin/env python
# coding: utf-8

# ## Introduction
# 
# In this notebook I will share main ideas that helped me to get 130th place in LANL earthquake competition. Further I will:
# 
# 1. Give a brief note on how I calculated features;
# 2. Share my CV-strategy;
# 3. Take a look at very simple models which led me to the main insight;
# 4. Outline my model;
# 5. Explain how I selected features for my models.
# 
# To have an idea about the competition itself, please, go and look at this [beautiful kernel](https://www.kaggle.com/allunia/shaking-earth).

# In[ ]:


import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np

from os.path import join
from os import listdir
import os

import matplotlib
import matplotlib.pyplot as plt

from tqdm import tqdm_notebook as tqdm
from sklearn.metrics import roc_curve, auc

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.svm import NuSVR
import lightgbm as lgb

from sklearn.metrics import mean_absolute_error as mae
from itertools import combinations

BLUE = '#3CBEC5'
RED = '#EF758A'
GREEN = '#19B278'

from IPython.display import Image


# ## 1. Feature generation
# 
# All of the features I used were calculated for 150K points intervals, however, I also used overlapping of 20% (which is 30K points):

# In[ ]:


Image("../input/feature-generation-frame/feature_generation_framework.png")


# You can see the groups of intervals, I will explain how to deal with them when building models later on in CV-strategy section. Ideas for the features to generate were taken from the following kernels:
# 
# 1. [Rolling statistics](https://www.kaggle.com/artgor/even-more-features)
# 2. [Magnitude statistics from FFT](https://www.kaggle.com/vettejeep/masters-final-project-model-lb-1-392)
# 3. [Bandpassed feautres](https://www.kaggle.com/vettejeep/masters-final-project-model-lb-1-392)
# 
# To summarise:
# 
# * There were around **700 features** generated.
# * Approximately **20K train records** were obtained from overlapping approach.

# In[ ]:


train = pd.read_csv("../input/processed/train_data.csv")
test = pd.read_csv("../input/processed/test_data.csv")


# Below you can find first rows of the train and test data sets.

# In[ ]:


print("Train data set:", train.shape)
train.head()


# In[ ]:


print("Test data set:")
test.head()


# ## 2. CV-strategy
# 
# As you may remember, I calculated the features for intervals that were overlapping. If one of the overlapping intervals will be used in the model training and the second one in the model validation it may cause overfitting. To remove this effect we will split our intervals in groups of 5 consecutive intervals (<font color="blue">"blue"</font> and <font color="red">"red"</font> colors on the picture from the feature generation section is exactly about this) and take the whole group either for training or for validation. However, we will still have overlapping groups, and for that we will remove all <font color="red">"red"</font> groups which gives us only non-overlapping <font color="blue">"blue"</font> groups.
# 
# This strategy doubles the training data, nevertheless you can take different sizes of <font color="red">"red"</font> and <font color="blue">"blue"</font> groups to increase the size of the training data. Taking 20 consecutive intervals for <font color="blue">"blue"</font> groups and 5 for <font color="red">"red"</font> will give you three times more data than you might have without overlapping technique.
# 
# Next step is to split our groups into folds. I did it randomly to obtain 10 folds, where each fold has some data from every earthquake.
# 
# ***Remark: taking larger blue groups will give us more data, but since we take intervals from one group together, we will have larger time intervals from the same quake. If you take extremely big blue groups you might end up with somewhat close to quake-CV, where each fold is one quake.***
# 

# In[ ]:


# split groups into blue and red
train['fold'] = np.arange(len(train)) // 10
train.loc[(np.arange(len(train)) % 10 > 4), 'fold'] = -1
# remove red groups
train = train[train.fold != -1].reset_index().drop('index', axis=1)

# get 10 folds
N = train.fold.max()
K = 10
k = N // K
np.random.seed(0)
permutation = np.random.choice(N, N, replace=False)
folds = []
for i in range(K):
    fold = permutation[i*k:(i+1)*k]
    folds.append(fold)
    tmp = train[~np.isin(train.fold, fold)]


# ## 3. Simple models + Crucial insight
# 
# 1. Let's start with very simple linear models using 10-fold-CV. I will show only some of them:

# In[ ]:


def build_linear_model(feature):
    train['prediction'] = 0
    # building out-of-fold predictions
    for i in range(K):
        val_fold = folds[i]
        cv_train = train[(~np.isin(train.fold, val_fold))]
        cv_val = train[np.isin(train.fold, val_fold)]
        feature_train = cv_train[feature].values
        target_train = cv_train[TARGET].values
        feature_val = cv_val[feature].values
        target_val = cv_val[TARGET].values
        
        model = LinearRegression()
        model.fit(feature_train.reshape(-1,1), target_train)

        val_predictions = model.predict(feature_val.reshape(-1,1))        
        train.loc[np.isin(train.fold, val_fold), 'prediction'] = val_predictions

    # building model in full training data
    cv_train = train.copy()
    feature_train = cv_train[feature].values
    target_train = cv_train[TARGET].values

    model = LinearRegression()
    model.fit(feature_train.reshape(-1,1), target_train)

    # evaluating quality
    score = mae(train[TARGET].values, train.prediction.values)
    
    # plotting
    fig, ax = plt.subplots(figsize=(18,6))
    
    plt.subplot(121)
    plt.plot(train[TARGET].values, c=RED, label='true')
    plt.plot(train.prediction.values, c=BLUE, alpha=0.5, label='prediction')
    plt.xlabel('time')
    plt.ylabel('time_to_failure')
    plt.legend()
    plt.grid()
    
    plt.subplot(122)
    plt.scatter(y=train[TARGET].values, x=feature_train, c=BLUE, alpha=0.1, label='true')
    a = np.quantile(feature_train, 0.05)
    b = np.quantile(feature_train, 0.95)
    x = np.linspace(a, b, 10)
    features = np.vstack([x]).T
    preds = model.predict(features.reshape(-1,1))
    plt.plot(features, preds, c=RED, alpha=1., label='linear model')
    plt.xlabel(feature)
    plt.ylabel('time_to_failure')
    plt.legend()
    plt.grid()
    
    print("MAE =", score)
    plt.show()
    return score

TARGET = 'time_to_failure'
for feature in train.columns[:-3]:
    build_linear_model(feature)


# Some of the models are very good, for example, with linear model based on feature **roll_100_std_percentile_5** (5% quantile of rolling standard deviation with a window of size 100) we can achieve MAE=2.19. Moreover, if we take a closer look on the right plots, we can see that there are two different patterns: one for **time_to_failure < 0.32** and one for **time_to_failure > 0.32**.
# 
# This separation comes from the fact that nearly at the same time (0.29-0.32 seconds to failure) signal shows extremely high values, and probably, there is a different nature of the experiment before and after this point.

# In[ ]:


# 0.32 seconds separation
fig, ax = plt.subplots(figsize=(20,10))

target = train[TARGET].values[700:900]
feature = train['roll_100_std_percentile_5'].values[700:900]
feature = (feature - min(feature)) / (max(feature) - min(feature)) * 15

plt.subplot(211)
plt.plot(target, color=RED, label='time_to_failure')
plt.plot(feature, color=BLUE, label='roll_100_std_percentile_5 (normalized)')
for split in [115]:
    plt.axvline(x=split, color=GREEN, linestyle='--', label='0.32 seconds to failure')

plt.title('0.32 seconds separation')
plt.xlabel('record')
plt.ylabel('value')
plt.grid()
plt.legend()

plt.show()


# Let's try to build separate models for records with **time_to_failure < 0.32** and for **time_to_failure > 0.32**.

# In[ ]:


feature = 'roll_100_std_percentile_5'

feature_values = train[feature].values
threshold = .32
target = train[TARGET]
index1 = (target > threshold)
index2 = (target <= threshold)
        
# build simple model for points before threshold 
model1 = LinearRegression()
model1.fit(feature_values[index1].reshape(-1,1), train[TARGET].values[index1])
# build simple model for points after threshold 
model2 = LinearRegression()
model2.fit(feature_values[index2].reshape(-1,1), train[TARGET].values[index2])

# evaluate quality
preds1 = model1.predict(feature_values.reshape(-1,1))
preds1[preds1 < 0] = 0
preds2 = model2.predict(feature_values.reshape(-1,1))
preds2[preds2 < 0] = 0
preds = model1.predict(feature_values.reshape(-1,1))
preds[index2] = preds2[index2]
preds[preds < 0] = 0

score = mae(train[TARGET].values, preds)
print("MAE =", score)

# plot
fig, ax = plt.subplots(figsize=(18,6))

plt.subplot(131)
plt.plot(target, c=RED, label='true')
plt.plot(preds, c=BLUE, alpha=0.5, label='prediction')
plt.title('Two linear models')
plt.xlabel('time')
plt.ylabel('time_to_failure')
plt.legend()
plt.grid()

plt.subplot(132)
plt.scatter(y=target[index1], x=feature_values[index1], c=BLUE, alpha=0.1, label='true')
a = np.quantile(feature_values, 0.05)
b = np.quantile(feature_values, 0.95)
x = np.linspace(a, b, 10)
features = np.vstack([x]).T
preds1 = model1.predict(features.reshape(-1,1))
plt.plot(features, preds1, c=RED, alpha=1., label='linear model')
plt.title('More than {} seconds to quake'.format(threshold))
plt.xlabel(feature)
plt.ylabel('time_to_failure')
plt.legend()
plt.grid()

plt.subplot(133)
plt.scatter(y=target[index2], x=feature_values[index2], c=BLUE, alpha=0.1, label='true')
a = np.quantile(feature_values, 0.05)
b = np.quantile(feature_values, 0.95)
x = np.linspace(a, b, 10)
features = np.vstack([x]).T
preds2 = model2.predict(features.reshape(-1,1))
plt.plot(features, preds2, c=RED, alpha=1., label='linear model')
plt.title('Less than {} seconds to quake'.format(threshold))
plt.xlabel(feature)
plt.ylabel('time_to_failure')
plt.legend()
plt.grid()


# Now we can also see that data after 0.32 seconds threshold behaves in non-linear way. Let's try to build a model based on 1/x transformation (in fact, transformation is 0.33-1/x) of the target.

# In[ ]:


def plot_complex_linear_model(feature):
    feature_values = train[feature].values
    threshold = 0.32
    target = train[TARGET]
    index1 = (target > threshold)
    index2 = (target <= threshold)

    # build simple model for points before threshold 
    model1 = LinearRegression()
    model1.fit(feature_values[index1].reshape(-1,1), train[TARGET].values[index1])
    # build simple model for points after threshold 
    model2 = LinearRegression()
    model2.fit(feature_values[index2].reshape(-1,1), 1/(threshold + 0.01 - train[TARGET].values[index2]))
    
    # evaluate quality
    preds1 = model1.predict(feature_values.reshape(-1,1))
    preds1[preds1 < 0] = 0
    preds2 = threshold + 0.01 - 1/model2.predict(feature_values.reshape(-1,1))
    preds2[preds2 < 0] = 0

    preds = model1.predict(feature_values.reshape(-1,1))
    preds[index2] = preds2[index2]
    preds[preds < 0] = 0

    score = mae(target, preds)
    print("MAE =", score)

    # plot
    fig, ax = plt.subplots(figsize=(18,6))

    plt.subplot(131)
    plt.plot(target, c=RED, label='true')
    plt.plot(preds, c=BLUE, alpha=0.5, label='prediction')
    plt.title('Linear and non-linear model')
    plt.xlabel('time')
    plt.ylabel('time_to_failure')
    plt.legend()
    plt.grid()

    plt.subplot(132)
    plt.scatter(y=target[index1], x=feature_values[index1], c=BLUE, alpha=0.1, label='true')
    a = np.quantile(feature_values, 0.01)
    b = np.quantile(feature_values, 0.99)
    x = np.linspace(a, b, 1000)
    features = np.vstack([x]).T
    preds1 = model1.predict(features.reshape(-1,1))
    plt.plot(features, preds1, c=RED, alpha=1., label='linear model')
    plt.title('More than {} seconds to quake'.format(threshold))
    plt.xlabel(feature)
    plt.ylabel('time_to_failure')
    plt.legend()
    plt.grid()

    plt.subplot(133)
    plt.scatter(y=target[index2], x=feature_values[index2], c=BLUE, alpha=0.1, label='true')
    a = np.quantile(feature_values, 0.1)
    b = np.quantile(feature_values, 0.999)
    x = np.linspace(a, b, 1000)
    features = np.vstack([x]).T
    preds2 = threshold + 0.01 - 1/model2.predict(features.reshape(-1,1))
    plt.plot(features, preds2, c=RED, alpha=1., label='linear model')
    plt.title('Less than {} seconds to quake'.format(threshold))
    plt.xlabel(feature)
    plt.ylabel('time_to_failure')
    plt.legend()
    plt.grid()

    plt.show()
    
plot_complex_linear_model(feature)


# Quality decreased a little, but the plot for the data after 0.32 seconds threshold is much nicer.
# 
# All in all, we can see a huge improvement in the model from **MAE=2.19** to **MAE=1.96**. 
# 
# A careful reader will say that we don't know in advance when the signal is before or after this 0.32 seconds threshold. Correct! So we need to build an extra classification model.

# ## 4. Final model outline
# 
# The final model consists of:
# 
# 1. Classifier built by LightGBM to define if the signal is before or after 0.32 seconds threshold.
# 2. LightGBM model built only on data with time_to_failure > 0.32
# 3. NuSVR model based on roll_100_std_percentile_5 feature built only on data with time_to_failure < 0.32
# 
# In the end we select an optimal threshold for classifier to choose wich model to choose LightGBM or NuSVR.

# #### Classifier

# In[ ]:


train['classifier_32'] = 0
train['target'] = train.time_to_failure < 0.32
TARGET = 'target'
features = [
 'mag_freq0_percentile_50',
 'bp5_mad',
 'bp1_interquantile_range_5_95',
 'evol_bp2_interquantile_range_10_90',
 'roll_4096_mean_std',
 'roll_1000_std_percentile_5',
 'bp0_kurtosis',
 'bp1_percentile_10',
 'evol_bp3_percentile_75'
]
params = {
    'num_leaves': 32,
    'max_bin': 63,
    'min_data_in_leaf': 50,
    'learning_rate': 0.05,
    'min_sum_hessian_in_leaf': 0.01,
    'bagging_fraction': 0.8,
    'bagging_freq': 1,
    'feature_fraction': 1,
    'min_gain_to_split': 0.02,
    'max_depth': 6,
    'save_binary': True, 
    'seed': 0,
    'feature_fraction_seed': 0,
    'bagging_seed': 0,
    'drop_seed': 0,
    'data_random_seed': 0,
    'objective': 'binary',
    'boosting_type': 'gbdt',
    'verbose': 1,
    'boost_from_average': True,
    'metric': 'auc',
    'is_unbalance': True
}

num_boost_round=100

for i in range(K):
    val_fold = folds[i]
    cv_train = train[(~np.isin(train.fold, val_fold))]
    cv_val = train[np.isin(train.fold, val_fold)]

    xg_train = lgb.Dataset(cv_train[features].values, label=cv_train[TARGET].values)
    xg_val = lgb.Dataset(cv_val[features].values, label=cv_val[TARGET].values)  

    model = lgb.train(params, xg_train, num_boost_round=num_boost_round, verbose_eval=0)
    val_predictions = model.predict(cv_val[features].values, num_iteration=num_boost_round)
    train.loc[np.isin(train.fold, val_fold), 'classifier_32'] = val_predictions

xg_train = lgb.Dataset(train[features].values, label=train[TARGET].values)
model = lgb.train(params, xg_train, num_boost_round=num_boost_round)
test['classifier_32'] = model.predict(test[features].values, num_iteration=num_boost_round)

fpr, tpr, _ = roc_curve(train['target'], train['classifier_32'])
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8,8))
lw = 2
plt.plot(fpr, tpr, color=BLUE,
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color=RED, lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Classifier ROC curve')
plt.legend(loc="lower right")
plt.show()


# #### LightGBM model

# In[ ]:


features = [
 'roll_2048_std_percentile_10',
 'roll_4096_mean_percentile_95',
 'bp6_percentile_90',
 'bp2_interquantile_range_20_80',
 'bp5_percentile_99',
 'bp7_interquantile_range_10_90',
 'bp4_percentile_80',
 'roll_100_mean_mean',
 'roll_1000_mean_percentile_99',
 'roll_100_std_percentile_5',
 'bp7_mad',
 'evol_bp5_min',
 'evol_bp7_max'
]

params = {'num_leaves': 5,
 'max_bin': 63,
 'min_data_in_leaf': 20,
 'learning_rate': 0.05,
 'min_sum_hessian_in_leaf': 0.008,
 'bagging_fraction': 0.9,
 'bagging_freq': 1,
 'feature_fraction': 1.,
 'min_gain_to_split': 0.5,
 'max_depth': 4,
 'save_binary': True,
 'seed': 0,
 'feature_fraction_seed': 0,
 'bagging_seed': 0,
 'drop_seed': 0,
 'data_random_seed': 0,
 'objective': 'huber',
 'boosting_type': 'gbdt',
 'verbose': 1,
 'metric': 'mae',
 'boost_from_average': True,
}

num_rounds = 2000
early_stopping = 50
TARGET = 'time_to_failure'

scores_train = []
scores_val = []
count = 0
test['lgbm_prediction'] = 0

for i in range(K):
    val_fold = folds[i]

    cv_train = train[(~np.isin(train.fold, val_fold)) & (train.time_to_failure > 0.32)]
    cv_val = train[np.isin(train.fold, val_fold) & (train.time_to_failure > 0.0)]

    xg_train = lgb.Dataset(cv_train[features].values, label=cv_train[TARGET].values)
    model = lgb.train(params, xg_train, num_rounds, verbose_eval=0)

    train_preds = model.predict(cv_train[features].values, num_iteration=num_rounds)
    val_preds = model.predict(cv_val[features].values, num_iteration=num_rounds)

    train.loc[np.isin(train.fold, val_fold), 'lgbm_prediction'] = val_preds

    scores_train.append(mae(cv_train[TARGET].values, train_preds))
    scores_val.append(mae(cv_val[TARGET].values, val_preds))

    count += 1
    test['lgbm_prediction'] += model.predict(test[features].values, num_iteration=num_rounds)

test['lgbm_prediction'] = test['lgbm_prediction'] / count

print("MAE =", mae(train.time_to_failure[:-5], train.lgbm_prediction[:-5]))

# plotting
fig, ax = plt.subplots(figsize=(18,6))

plt.subplot(111)
plt.plot(train[TARGET].values, c=RED, label='true')
plt.plot(train.lgbm_prediction.values, c=BLUE, alpha=0.5, label='prediction')
plt.title('LightGBM predictions')
plt.xlabel('time')
plt.ylabel('time_to_failure')
plt.legend()
plt.grid()

plt.show()


# #### NuCVR model

# In[ ]:


feature = 'roll_100_std_percentile_5'
scores_train = []
scores_val = []
count = 0
test['nusvr_prediction'] = 0
threshold = 0.32
for i in range(K):
    val_fold = folds[i]

    cv_train = train[(~np.isin(train.fold, val_fold)) & (train.time_to_failure <= threshold)]
    cv_val = train[np.isin(train.fold, val_fold) & (train.time_to_failure > 0.0)]
    
    feature_train = cv_train[feature].values
    target_train = cv_train[TARGET].values
    feature_val = cv_val[feature].values
    target_val = cv_val[TARGET].values
    feature_test = test[feature].values
    
    params = {
     'C': 30,
     'gamma': 0.05,
     'kernel': 'rbf',
     'nu': 1,
    }

    model = NuSVR(**params)
    model.fit(feature_train.reshape(-1,1), 1/(threshold + 0.01 - target_train))

    # evaluate quality
    train_preds = threshold + 0.01 - 1/model.predict(feature_train.reshape(-1,1))
    train_preds[train_preds < 0] = 0
    val_preds = threshold + 0.01 - 1/model.predict(feature_val.reshape(-1,1))
    val_preds[val_preds < 0] = 0
    test_preds = threshold + 0.01 - 1/model.predict(feature_test.reshape(-1,1))
    test_preds[test_preds < 0] = 0
    
    train.loc[np.isin(train.fold, val_fold), 'nusvr_prediction'] = val_preds

    scores_train.append(mae(cv_train[TARGET].values, train_preds))
    scores_val.append(mae(cv_val[TARGET].values, val_preds))
    
    count += 1
    test['nusvr_prediction'] += test_preds

test['nusvr_prediction'] = test['nusvr_prediction'] / count

# plotting
fig, ax = plt.subplots(figsize=(18,6))

plt.subplot(111)
plt.plot(train[TARGET].values, c=RED, label='true')
plt.plot(train.nusvr_prediction.values, c=BLUE, alpha=0.5, label='prediction')
plt.title('NuSVR predictions')
plt.xlabel('time')
plt.ylabel('time_to_failure')
plt.legend()
plt.grid()

plt.show()


# #### Mixing models
# 
# Let's define the best threshold for classifier.

# In[ ]:


data = []
for x in np.linspace(0.,1.,21):
    def mix(row):
        t = x
        if row.classifier_32 < t:
            return row.lgbm_prediction
        else:
            return row.nusvr_prediction
    train['final_prediction'] = train.apply(mix, axis=1)
    data.append([np.round(x,2), mae(train.time_to_failure[:-5], train.final_prediction[:-5])])
pd.DataFrame(data=data, columns=['threshold', 'MAE'])


# In[ ]:


def mix(row):
    t = 0.85
    if row.classifier_32 < t:
        return row.lgbm_prediction
    else:
        return row.nusvr_prediction
    
train['final_prediction'] = train.apply(mix, axis=1)
test['time_to_failure'] = test.apply(mix, axis=1)
print("MAE =", mae(train.time_to_failure[:-5], train.final_prediction[:-5]))

# plotting
fig, ax = plt.subplots(figsize=(18,6))

plt.subplot(111)
plt.plot(train[TARGET].values, c=RED, label='true')
plt.plot(train.final_prediction.values, c=BLUE, alpha=0.5, label='prediction')
plt.title('Final predictions')
plt.xlabel('time')
plt.ylabel('time_to_failure')
plt.legend()
plt.grid()

plt.show()


# Finally, we achieved a model with MAE=1.915.

# #### Preparing submission

# In[ ]:


submission = pd.read_csv('../input/LANL-Earthquake-Prediction/sample_submission.csv')
submission = submission.drop('time_to_failure', axis=1)
preds = test[['time_to_failure', 'seg_id']]
submission = submission.merge(preds, how='left')
submission.to_csv('submission_nusvr_lgbm_classifier.csv', index=False)
submission.head()


# ## 5. Small note on feature selection
# 
# To select features for models I used a very greedy forward feature selection method. To select features for LightGBM I used bigger learning rate to evaluate models faster.
# 
# **Thanks for reading!!!**
