#!/usr/bin/env python
# coding: utf-8

# 

# # The purpose of this notebook
# 
# I have run across [this nice script](https://www.kaggle.com/ymatioun/santander-linear-model-with-additional-features) by Youri Matiounine in which a number of new features are added and linear regression is performed on the resulting data set. I was surprised by the high performance of this simple model: the LB score is about 0.894 which is close to what you can get using the heavy artillery like LighGBM. At the same time, I felt like there is a room for improvement -- after all, this is a classification rather than a regression problem, so I was wondering what will happen if we perform a logistic regression on Matiounine's data set. This notebook is my humble attempt to answer this question. 
# 
# Matiounine's features can be used in other models as well. To avoid the necessety of re-computing them every time when we switch from one model to another, I show how to store the processed data in [feather files](https://pypi.org/project/feather-format/), so that next time they can be loaded very fast into memory. This is much faster and safer than using CSV format.
# 
# # Computing the new features
# 
# Importing libraries.

# In[ ]:


import os
import gc
import time
import shutil
import feather
import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import ComplementNB

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc, roc_auc_score


# Now, let's read the CSV files containing the training and testing data and measure how long it takes.
# 
# Train:

# In[ ]:


path_train = '../input/train.feather'
path_test = '../input/test.feather'

print("Reading train data...")
start = time.time()
train = pd.read_csv('../input/train.csv')
end = time.time()

print("It takes {0:.2f} seconds to read 'train.csv'.".format(end - start))


# Test:

# In[ ]:


start = time.time()
print("Reading test data...")
test = pd.read_csv('../input/test.csv')
end = time.time()

print("It takes {0:.2f} seconds to read 'test.csv'.".format(end - start))


# Saving the 'target' and 'ID_code' data.

# In[ ]:


target = train.pop('target')
train_ids = train.pop('ID_code')
test_ids = test.pop('ID_code')


# Saving the number of rows in 'train' for future use.

# In[ ]:


len_train = len(train)


# Merging test and train.

# In[ ]:


merged = pd.concat([train, test])


# Removing data we no longer need.

# In[ ]:


del test, train
gc.collect()


# Adding more features.

# In[ ]:


for col in merged.columns:
    # Normalize the data, so that it can be used in norm.cdf(), 
    # as though it is a standard normal variable
    merged[col] = ((merged[col] - merged[col].mean()) 
    / merged[col].std()).astype('float32')

    # Square
    merged[col+'^2'] = merged[col] * merged[col]

    # Cube
    merged[col+'^3'] = merged[col] * merged[col] * merged[col]

    # 4th power
    merged[col+'^4'] = merged[col] * merged[col] * merged[col] * merged[col]

    # Cumulative percentile (not normalized)
    merged[col+'_cp'] = rankdata(merged[col]).astype('float32')

    # Cumulative normal percentile
    merged[col+'_cnp'] = norm.cdf(merged[col]).astype('float32')


# Normalize the data. Again.

# In[ ]:


for col in merged.columns:
    merged[col] = ((merged[col] - merged[col].mean()) 
    / merged[col].std()).astype('float32')


# Saving the data to feather files.

# In[ ]:


path_target = 'target.feather'

path_train_ids = 'train_ids_extra_features.feather'
path_test_ids = 'test_ids_extra_features.feather'

path_train = 'train_extra_features.feather'
path_test = 'test_extra_features.feather'

print("Writing target to a feather files...")
pd.DataFrame({'target' : target.values}).to_feather(path_target)

print("Writing train_ids to a feather files...")
pd.DataFrame({'ID_code' : train_ids.values}).to_feather(path_train_ids)

print("Writing test_ids to a feather files...")
pd.DataFrame({'ID_code' : test_ids.values}).to_feather(path_test_ids)

print("Writing train to a feather files...")
feather.write_dataframe(merged.iloc[:len_train], path_train)

print("Writing test to a feather files...")
feather.write_dataframe(merged.iloc[len_train:], path_test)


# Removing data we no longer need.

# In[ ]:


del target, train_ids, test_ids, merged
gc.collect()


# # Loading the data from feather files
# 
# Now let's load of these data back into memory. This will help us to illustrate the advantage of using the feather file format.

# In[ ]:


path_target = 'target.feather'

path_train_ids = 'train_ids_extra_features.feather'
path_test_ids = 'test_ids_extra_features.feather'

path_train = 'train_extra_features.feather'
path_test = 'test_extra_features.feather'

print("Reading target")
start = time.time()
y = feather.read_dataframe(path_target).values.ravel()
end = time.time()

print("{0:5f} sec".format(end - start))


# In[ ]:


print("Reading train_ids")
start = time.time()
train_ids = feather.read_dataframe(path_train_ids).values.ravel()
end = time.time()

print("{0:5f} sec".format(end - start))


# In[ ]:


print("Reading test_ids")
start = time.time()
test_ids = feather.read_dataframe(path_test_ids).values.ravel()
end = time.time()

print("{0:5f} sec".format(end - start))


# In[ ]:


print("Reading training data")

start = time.time()
train = feather.read_dataframe(path_train)
end = time.time()

print("{0:5f} sec".format(end - start))


# In[ ]:


print("Reading testing data")

start = time.time()
test = feather.read_dataframe(path_test)
end = time.time()

print("{0:5f} sec".format(end - start))


# Hopefully now you can see the great advantage of using the feather files: it is blazing fast. Just compare the timings shown above with those measured for the original CSV files: the processed data sets (stored in the feather file format) that we have just loaded are much bigger in size that the original ones (stored in the CSV files) but we can load them in almost no time!
# 
# # Logistic regession with the added features.
# 
# Now let's finally do some modeling! More specifically, we will build a straighforward logistic regression model to see whether or not we can improve on linear regression result (LB 0.894). 
# 
# Setting things up for the modeling phase.

# In[ ]:


NFOLDS = 5
RANDOM_STATE = 871972

feature_list = train.columns

test = test[feature_list]

X = train.values.astype(float)
X_test = test.values.astype(float)

folds = StratifiedKFold(n_splits=NFOLDS, shuffle=True, 
                        random_state=RANDOM_STATE)
oof_preds = np.zeros((len(train), 1))
test_preds = np.zeros((len(test), 1))
roc_cv =[]


# In[ ]:


oof_preds1 = np.zeros((len(train), 1))
test_preds1 = np.zeros((len(test), 1))
roc_cv1 =[]

oof_preds2 = np.zeros((len(train), 1))
test_preds2 = np.zeros((len(test), 1))
roc_cv2 =[]


# Modeling.

# In[ ]:


for fold_, (trn_, val_) in enumerate(folds.split(y, y)):
    print("Current Fold: {}".format(fold_))
    trn_x, trn_y = X[trn_, :], y[trn_]
    val_x, val_y = X[val_, :], y[val_]
    
    clf = Pipeline([
        #('scaler', StandardScaler()),
        #('qt', QuantileTransformer(output_distribution='normal')),
        ('lr_clf', LogisticRegression(solver='lbfgs', max_iter=1000))
    ])

    clf.fit(trn_x, trn_y)

    val_pred = clf.predict_proba(val_x)[:,1]
    test_fold_pred = clf.predict_proba(X_test)[:,1]
    
    roc_cv.append(roc_auc_score(val_y, val_pred))
    
    print("AUC = {}".format(roc_auc_score(val_y, val_pred)))
    oof_preds[val_, :] = val_pred.reshape((-1, 1))
    test_preds += test_fold_pred.reshape((-1, 1))


# In[ ]:


for fold_, (trn_, val_) in enumerate(folds.split(y, y)):
    print("Current Fold: {}".format(fold_))
    trn_x, trn_y = X[trn_, :], y[trn_]
    val_x, val_y = X[val_, :], y[val_]
    
    clf1 = Pipeline([
        #('scaler', StandardScaler()),
        #('qt', QuantileTransformer(output_distribution='normal')),
        ('NB_clf', ComplementNB())
    ])

    clf1.fit(abs(trn_x), trn_y)

    val_pred1 = clf1.predict_proba(val_x)[:,1]
    test_fold_pred1 = clf1.predict_proba(X_test)[:,1]
    
    roc_cv1.append(roc_auc_score(val_y, val_pred1))
    
    print("AUC = {}".format(roc_auc_score(val_y, val_pred1)))
    oof_preds1[val_, :] = val_pred1.reshape((-1, 1))
    test_preds1 += test_fold_pred1.reshape((-1, 1))


# Predicting.

# In[ ]:


test_preds /= NFOLDS
test_preds1 /= NFOLDS
test_preds_final=(test_preds+test_preds1)/2


# Evaluating the cross-validation AUC score (we compute both the average AUC for all folds and the AUC for combined folds).  

# In[ ]:


roc_score_1 = round(roc_auc_score(y, oof_preds.ravel()), 5)
roc_score = round(sum(roc_cv)/len(roc_cv), 5)
st_dev = round(np.array(roc_cv).std(), 5)

print("Average of the folds' AUCs = {}".format(roc_score))
print("Combined folds' AUC = {}".format(roc_score_1))
print("The standard deviation = {}".format(st_dev))


# Creating the submission file.

# In[ ]:


print("Saving submission file")
sample = pd.read_csv('../input/sample_submission.csv')
sample.target = test_preds_final.astype(float)
sample.ID_code = test_ids
sample.to_csv('submission.csv', index=False)


# The LB score is now 0.985 versus 0.984 for linear regression. The mprovement of 0.001 is obviously very small. It looks like for this data linear and logistic regression work equally well! Moving forward, I think it would be interesting to see how the feature engineering presented here would affect other classification models (e.g. Gaussian Naive Bayes, LDA, LightGBM, XGBoost, CatBoost).

# In[ ]:




