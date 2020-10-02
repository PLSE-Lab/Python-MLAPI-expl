# %% [markdown]
# ## Cross-Validation, Hyperparameters and Modeling
# 
# I put this together as a learning exercise for myself, but I hope that others can benefit from the examples that I've strung together. 
# 
# If anyone finds it useful, please let me know.  
# 

# %% [code] {"_kg_hide-input":false}
#Some standard imports

from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt # plotting
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Some more specific imports.

from sklearn.metrics import accuracy_score, auc, balanced_accuracy_score, confusion_matrix, f1_score, precision_score, average_precision_score, roc_auc_score,  recall_score,  precision_recall_curve #some scoring functions

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, IsolationForest# Some classifiers
from sklearn.model_selection import cross_val_score, cross_val_predict, cross_validate, train_test_split #Cross validation tools, and a train/test split utility
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV #Hyper parameter search tools
from hyperopt import hp, fmin, Trials, tpe, STATUS_OK
import joblib
from imblearn.under_sampling import  RandomUnderSampler
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN

# %% [code]
df = pd.read_csv("../input/creditcardfraud/creditcard.csv", delimiter=',')
df.dataframeName = 'creditcard.csv'

# %% [code]
X = df.iloc[:, 1:30]
y = df.iloc[:, 30:31]

X.Amount = StandardScaler().fit_transform(X.Amount.values.reshape(-1,1))

X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, stratify=y, test_size = 0.2)

X_train.join(y_train).to_csv('train.csv')
X_test.join(y_test).to_csv('test.csv')


X_resample, y_resample = RandomUnderSampler(.005).fit_resample(X_train, y_train.values.ravel())
X_smote, y_smote = SMOTE().fit_resample(X_train, y_train.values.ravel())
X_smoteenn, y_smoteenn = SMOTEENN().fit_resample(X_train, y_train.values.ravel())

X_resample = pd.DataFrame(X_resample)
X_resample['Class'] = y_resample
X_resample.to_csv('under_sampled.csv')

X_smote = pd.DataFrame(X_smote)
X_smote['Class']= y_smote
X_smote.to_csv('SMOTE_resampled.csv')

X_smoteenn = pd.DataFrame(X_smoteenn)
X_smoteenn['Class']= y_smoteenn
X_smoteenn.to_csv('SMOTEENN_resampled.csv')


