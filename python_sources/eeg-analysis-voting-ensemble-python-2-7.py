# EEG Confusion Analysis - Voting ensemble classifier (XGboost, Random Forest Classifier and Naive Bayes)

# For Python 2.7

# Label predicted is user-defined label.

# You can find below the results for different sets of features:

# Results

# 0.98985959438377535 > 98.98% > predicting user-defined label with Video ID, Subject ID, Attention and Mediation, and raw signals
# 0.71450858034321374 > 71.45% > predicting user-defined label without Video ID, with Subject ID, Attention and Mediation, and raw signals
# 0.66224648985959433 > 66.22% > predicting user-defined label without Subject ID, with Attention, Mediation, and raw signals
# 0.64040561622464898 > 64.04% > predicting user-defined label without Subject ID, Attention and Mediation, only raw signals

# From my perspective, using Video ID as a feature does not make much sense, but I include the results as a reference.

# Using Subject ID might be interesting as it might suggest that EEG "signatures" in relation to confusion might change with a specific individual's biology. On the other hand, brainwaves aside, reported confusion might also change according to personality characteristics that are totally unrelated to EEG.

# Attention and Mediation propietary measures offer a slight improvement in predictive results vs. only using raw EEG waves.

# Most interesting is trying to predict confusion from the pure EEG recording. Here results achieve around 64%.

# For Python 2.7

import gc
import xgboost as xgb
import numpy as np
import pandas as pd
import random
import time

from sklearn import preprocessing
from sklearn import datasets, metrics

from sklearn.naive_bayes import GaussianNB as gnb
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier as rfc
from sklearn.preprocessing import normalize
from numpy import genfromtxt
from sklearn.ensemble import VotingClassifier
from sklearn import preprocessing
from sklearn.ensemble import AdaBoostClassifier
from sklearn.cross_validation import train_test_split

# Reading the dataset

data = pd.read_csv("EEG data.csv")
print(data.shape)
data.head()

# Here we drop the features we do not want to use

data = data.drop(["predefined_confusion","Subject_ID","Video_ID","Attention_mental focus","Mediation_calmness"],axis=1)
print(data.shape)
data.head()

# Divide in train and predict (called test here, which takes rows without recorded label: it is zero in this dataset)

train = data.loc[~data.user_confusion.isnull()]
test = data.loc[data.user_confusion.isnull()]

print(train.shape, test.shape)

# split data into train and test

test_id = test.ID
test = test.drop(["ID","user_confusion"],axis=1)

X = train.drop(["ID","user_confusion"],axis=1)
y = train.user_confusion.values
X.head()

# Split train set for training

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1) #, random_state=1729)
print(X_train.shape, X_test.shape, test.shape)

# Details of the classifiers and model fit

clf8 = xgb.XGBClassifier(n_estimators=70, learning_rate=0.1, min_child_weight = 2, nthread=-1, max_depth = 9, seed=729, 
                       objective= 'binary:logistic', scale_pos_weight=1)
clf6 = xgb.XGBClassifier(n_estimators=170, learning_rate=0.20, min_child_weight = 3, nthread=-1, max_depth = 12, seed=729, 
                        objective= 'binary:logistic', scale_pos_weight=1)
clf7 = xgb.XGBClassifier(n_estimators=3, learning_rate=0.30, min_child_weight = 4, nthread=-1, max_depth = 6, seed=729, 
                        objective= 'binary:logistic', scale_pos_weight=1)
clf10 = rfc(n_estimators=200, criterion='gini', max_depth=20, random_state=2016)
clf11 = gnb()

eclf2 = VotingClassifier(estimators=[('2', clf6), ('3', clf7), ('4', clf8), ('5', clf10), ('6', clf11)],
                         voting='soft', weights=[1, 1, 1, 1, 1])

eclf2.fit(X_train.values, y_train)

# Prediction and accuracy score

prediction2 = eclf2.predict(X_test.values)
accuracy_score(y_test, prediction2, normalize=True, sample_weight=None)