#!/usr/bin/env python
# coding: utf-8

# # Intro
# There are several strategies to train a model to achieve a similar score on the kernels train data and on the test data where the results are shown on the leaderboard (LB). Commonly used are for instance cross validation (CV) / kfolding or model ensembling.
# We will see that in this competition the train and test data sets are not equaly distributed. Therefore we examine Adversarial Validation as a strategy to generate a validation set that is similar to the test data.
# 
# Disclaimer:
# Please keep in mind that this is a beginners notebook and might include some wrong assumptions or conclusions. We are happy if you discuss with us in the comments section.
# 
# Here are some references where I first stumbled upon Adverserial Validation (AV). Please upvote the kernels if you like them.
# 
# http://fastml.com/adversarial-validation-part-one/
# https://www.kaggle.com/tunguz/adversarial-santander
# 
# Here is another public kernel relevant to this competition concerning AV:
# 
# https://www.kaggle.com/lukeimurfather/adversarial-validation-train-vs-test-an-update
# 
# Discussion about metrics:
# https://www.kaggle.com/joatom/discussing-metrics-on-imbalanced-data
# 
# Some plots are taken from here:
# https://www.kaggle.com/phsheth/forestml-eda-and-stacking-evaluation
# 
# Some basic EDA is taken from here:
# https://www.kaggle.com/mancy7/simple-eda

# # Checking out the data

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import roc_curve, auc, confusion_matrix, accuracy_score, precision_recall_curve,SCORERS
import eli5
from eli5.sklearn import PermutationImportance

import warnings
warnings.filterwarnings('ignore')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        data_path = dirname+'/'
        print(data_path)

# Any results you write to the current directory are saved as output.


# In[ ]:


train = pd.read_csv(data_path+'train.csv')

train.head()


# In[ ]:


train.describe().T


# In[ ]:


test = pd.read_csv(data_path+'test.csv')

test.head()


# In[ ]:


test.describe().T


# There is only one value for Soil_Type7 and Soil_Type15 in train but two in test. Will remove the feature later. (see https://www.kaggle.com/mancy7/simple-eda) 

# ## How many samples per CoverType

# In[ ]:


train.groupby('Cover_Type')['Cover_Type'].count().plot.pie(),train.groupby('Cover_Type')['Cover_Type'].count()


# ==> The classes are equaly distributed

# # Comparing test and train data

# ## General

# In[ ]:


print('Number of train rows:',train.shape[0])
print('Number of test rows:',test.shape[0])
print('Ratio: %0.4f' % (train.shape[0]/test.shape[0]))
plt.pie([train.shape[0],test.shape[0]])


# There are very few train rows compared to test rows. 
# 
# We want to check if the structure of train and test are similar. 

# ## Adversarial Validation

# Set target of train = 0, set target of test = 1. 
# Check if you can differentiate between train and test data with Logistic Regression or Random Forest. 
# If the algorithms can't differentiate between train and test they seem to have similar characteristics.
# 
# We use roc_auc-Score to measure how well data entrys can be classified as test. 1 means it can perfectly classify the data, 0.5 means it can't classify the data. Hence 0.5 means train and test data are similar.
# 
# We examined at a few metrics (https://www.kaggle.com/joatom/discussing-metrics-on-imbalanced-data) to choose a reasonable metric to classify imbalanced data. We also run the experiments with different metrics all showing the same tendecies.
# 

# In[ ]:


#https://www.kaggle.com/tunguz/adversarial-santander

# removed solid type 7 and 15 because of different values an test and train
# (see https://www.kaggle.com/mancy7/simple-eda)
av_features = list(set(train.columns)-set(['Id','Cover_Type','Solid_Type7','Solid_Type15']))

av_X = train[av_features]#train.columns[:]]
av_X['is_test'] = 0
av_X_test = test[av_features]#test.columns[:]]
av_X_test['is_test'] = 1

av_train_test = pd.concat([av_X, av_X_test], axis =0)
av_y = av_train_test['is_test']#.values
av_train_test = av_train_test.drop(['is_test'],axis=1)


# In[ ]:


# setup KFold
splits = 3
#repeats = 2
rskf = StratifiedKFold(n_splits=splits, random_state=2019, shuffle=True)


# In[ ]:



# log regression
scores = cross_val_score(LogisticRegression(random_state=2019, solver='lbfgs',max_iter=1000), av_train_test, av_y, cv=rskf, scoring='roc_auc') #'f1'
print("Log Regression Accuracy (RoC): %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), 'AV LogReg'))

# random forrest
scores = cross_val_score(RandomForestClassifier(n_estimators = 100, n_jobs = -1, random_state=2019), av_train_test, av_y, cv=rskf, scoring='roc_auc') #'f1'
print("Random Forrest Accuracy (RoC): %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), 'AV RandomForestClas'))


# LogReg roc_auc accuracy of 75%. (Also tried on scaled data (Standard- and MinMax-Scaler) with same results. 
# Random Forrest roc_auc accurace of 76% 
# (Also tried f1-score. Got a 0.99% for both.)
# 
# Let's check what features cause the classification.

# In[ ]:


# https://www.kaggle.com/ynouri/random-forest-k-fold-cross-validation
def compute_roc_auc(X, y, index):
    y_predict = clf.predict_proba(X.iloc[index])[:,1]
    print(y_predict)
    fpr, tpr, thresholds = roc_curve(y.iloc[index], y_predict)
    auc_score_roc = auc(fpr, tpr)
    # http://www.davidsbatista.net/blog/2018/08/19/NLP_Metrics/
    precision, recall, thresholds = precision_recall_curve(y.iloc[index], y_predict)
    auc_score_prc = auc(recall, precision)
    
    return y_predict, auc_score_roc, auc_score_prc


# In[ ]:


# http://fastml.com/adversarial-validation-part-one/
clf = RandomForestClassifier(n_estimators = 100, n_jobs = -1, random_state=2019)

fprs, tprs, scores_roc_train, scores_roc_valid, scores_prc_train, scores_prc_valid = [], [], [], [], [], []
# https://github.com/zygmuntz/adversarial-validation/blob/master/numerai/sort_train.py
predictions = np.zeros(av_y.shape[0])

for (i_train, i_valid), i in zip(rskf.split(av_train_test,av_y),range(splits)):
    print('Split', i)
    clf.fit(av_train_test.iloc[i_train], av_y.iloc[i_train])
    
    # score
    _, auc_score_roc_train, auc_score_prc_train = compute_roc_auc(av_train_test, av_y, i_train)
    y_predict, auc_score_roc, auc_score_prc = compute_roc_auc(av_train_test, av_y, i_valid)
    predictions[i_valid] = y_predict
    
    scores_roc_train.append(auc_score_roc_train)
    scores_roc_valid.append(auc_score_roc)
    scores_prc_train.append(auc_score_prc_train)
    scores_prc_valid.append(auc_score_prc)
    
    # Feature Importance
    ## https://towardsdatascience.com/running-random-forests-inspect-the-feature-importances-with-this-code-2b00dd72b92e
    clf.score(av_train_test.iloc[i_valid], av_y.iloc[i_valid])
    rf_feature_importances = pd.DataFrame(clf.feature_importances_,
                                       index = av_train_test.columns,
                                       columns=['importance']).sort_values('importance', ascending=False)
    display(rf_feature_importances.head(10))
    
    # Permutation Importance
    permImp = PermutationImportance(clf, random_state=2021).fit(av_train_test.iloc[i_valid], av_y.iloc[i_valid]) 
    display(eli5.show_weights(permImp, feature_names = av_train_test.columns.tolist()))
    
print('Mean Accuracy roc:', np.mean(scores_roc_valid))
print('Mean Accuracy precision recal:', np.mean(scores_roc_valid))

av_train_test['p'] = predictions


# In[ ]:


av_train_test['is_test']=av_y
av_train_test.groupby(['is_test']).describe()[['p']]


# Seems like RF classifies about 25% of the train data as test data (0.99 at 75%-quartile where *is_test* == 0). We use this 25% as our out-of-adversarial-validation set later.

# ## Rank the "test-likelyness" on train

# We rank the train data on who high the prediction towards test was. In *testalike* wie rank it over the entiry dataset and on *testalike_per_cover_type* we do the ranking per Cover_Type.

# In[ ]:


train['testalike'] = av_train_test.loc[av_train_test.is_test == 0]['p'].rank(method='first').astype(int)
train['testalike_per_cover_type'] = train.groupby(['Cover_Type'])['testalike'].rank(method='first').astype(int)

# check rank distribution per cover type
train.groupby(['Cover_Type']).describe()[['testalike_per_cover_type']]
# => looks good


# ==> high rank means being more *test alike*

# In[ ]:


train.to_csv('train_av.csv', index=False)


# Let's have a look add the most influencial features.
# 

# In[ ]:


# https://www.kaggle.com/phsheth/forestml-eda-and-stacking-evaluation

sns.distplot(train['Elevation'], label = 'train')
sns.distplot(test['Elevation'], label = 'test')
plt.legend()
plt.title('Elevation')
plt.show()


# In[ ]:


sns.distplot(train['Horizontal_Distance_To_Roadways'], label = 'train')
sns.distplot(test['Horizontal_Distance_To_Roadways'], label = 'test')
plt.legend()
plt.title('Horizontal_Distance_To_Roadways')
plt.show()


# In[ ]:


sns.distplot(train['Horizontal_Distance_To_Fire_Points'], label = 'train')
sns.distplot(test['Horizontal_Distance_To_Fire_Points'], label = 'test')
plt.legend()
plt.title('Horizontal_Distance_To_Fire_Points')
plt.show()


# The most influencial features on our AV predictions have differnt distributions on test and train as expected.

# # Probe different validation sets

# We will take x % of the most testalike data from train to build our validation set in experiment no.1 and no.2. 
# In experiment no.3 and no.4 we choose a random sample as validation set. We run a simple Random Forest and predict the classes of the different validation sets.
# Afterwards we predict the classes for the test dataset and probe it against the leaderboard.
# 
# If our assumptions are right no.1 and no.2 will score about the same on the LB as the coresponding validation set.
# 
# 1. RF with 10% validation set of testalike
# 2. RF with 25% validation set of testalike
# 3. RF with simple 10% train-split
# 4. RF with simple 25% train-Split

# Expand to see code for classification and validation set splitting:

# In[ ]:


def testalike_split(X,y, val_size, testalike):
    tr_idx, val_idx = np.split(testalike.argsort(),[-int(len(testalike)*val_size)])
    
    train_X = X[tr_idx]
    train_y = y[tr_idx]
    val_X = X[val_idx]
    val_y = y[val_idx]
    return (train_X, val_X, train_y, val_y), val_size


def simple_split(X,y, val_size):
    return train_test_split(X, y, random_state = 2020, test_size = val_size), val_size


def run_classification(test_spliter):
    (train_X, val_X, train_y, val_y), val_size = test_spliter
    
    clf = RandomForestClassifier(n_estimators = 100, n_jobs = -1, random_state=2020)
    clf.fit(train_X, train_y)

    y_hat_val = clf.predict(val_X)
    y_hat = clf.predict(test_prep)
    
    cm = confusion_matrix(val_y, y_hat_val)
    accuracy = accuracy_score(val_y, y_hat_val)
    
    print('Accuracy:', accuracy)
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=range(1,8), yticklabels=range(1,8))

    return y_hat


# ## 0. preprocessing

# In[ ]:


X = train.drop(['Id','Soil_Type7','Soil_Type15'],axis=1)
y = X.pop('Cover_Type')
testalike = X.pop('testalike')
testalike_group = X.pop('testalike_per_cover_type')

test_prep_ids = test['Id']
test_prep = test.drop(['Id','Soil_Type7','Soil_Type15'],axis=1)


####  Scaling  ####
sc = StandardScaler()
X = sc.fit_transform(X)
test_prep = sc.transform(test_prep)


# ## 1. testalike group split 10%

# In[ ]:


y_hat = run_classification(testalike_split(X,y, 0.1, testalike_group))

output = pd.DataFrame({'Id': test_prep_ids,'Cover_Type': y_hat})
output.to_csv('probe_testalike_split_10.csv', index=False)


# ## 2. testalike group split 25%
# 

# In[ ]:


y_hat = run_classification(testalike_split(X,y, 0.25, testalike_group))

output = pd.DataFrame({'Id': test_prep_ids,'Cover_Type': y_hat})
output.to_csv('probe_testalike_split_25.csv', index=False)


# ## 3. simple split 10%

# In[ ]:


y_hat = run_classification(simple_split(X,y, 0.1))

output = pd.DataFrame({'Id': test_prep_ids,'Cover_Type': y_hat})
output.to_csv('probe_simple_split_10.csv', index=False)


# ## 4. simple split 25 %

# In[ ]:


y_hat = run_classification(simple_split(X,y, 0.25))

output = pd.DataFrame({'Id': test_prep_ids,'Cover_Type': y_hat})
output.to_csv('probe_simple_split_25.csv', index=False)


# # Probing against the LB

# These are the LB results:
# - no. 1: 0.752
# - no. 2: 0.739
# - no. 3: 0.743
# - no. 4: 0.731

# # Conclusions 
# This looks good. The test like validation set no.1 has about the same score on kernel and LB.
# It might be tempting to use no.3 or no.4 because of the high score (0.86) on the kernel. But they are overfit.
# No.1 seams to be a good validation set keep in control while feature and model engineering.
# 
# 
# Thank you for reading sofar. So tell us, what are your favorite validation strategies?
# 
