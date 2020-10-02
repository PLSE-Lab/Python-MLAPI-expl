#!/usr/bin/env python
# coding: utf-8

# **Import Packages**

# In[1]:


import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import train_test_split

from sklearn.dummy import DummyClassifier

from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.linear_model import LogisticRegression

from xgboost.sklearn import XGBClassifier

from imblearn.combine import SMOTEENN
from imblearn.pipeline import Pipeline

from keras.models import Sequential
from keras.layers import Dense

from sklearn import metrics

import matplotlib.pyplot as plt
import seaborn as sns


# **Read Data**

# In[2]:


#creditcard = pd.read_csv("creditcard.csv")
creditcard = pd.read_csv("../input/creditcard.csv")


# **EDA**

# In[4]:


creditcard.info()


# In[5]:


creditcard.describe()


# In[6]:


sns.distplot(creditcard["Time"].astype(float))


# In[7]:


sns.countplot(data = creditcard, x = 'Class')


# In[8]:


creditcard['Class'].unique()


# In[9]:


len(creditcard[creditcard['Class']==1])


# In[10]:


len(creditcard[creditcard['Class']==0])


# In[11]:


492/(284315+492)*100


# We can see that the data here is very imbalanced, with only 0.172% of the data as class 1 (i.e 1 for fraudulent transactions)

# ___

# In[12]:


amount = creditcard[creditcard['Class']==1]['Amount']


# In[13]:


sns.distplot(amount)


# We can see that the majority of the fraud data transaction is less than 500

# In[14]:


creditcard.hist(figsize=(30,30))


# From the histogram above we can see that most of the data columns are scaled (apart from amount and time)
# 
# So next we will try to scale the values of amount and time columns
# 
# ___

# In[15]:


# RobustScaler is less prone to outliers.
# https://scikit-learn.org/stable/auto_examples/preprocessing/plot_all_scaling.html

robust = RobustScaler()

creditcard['scaled_amount'] = robust.fit_transform(creditcard['Amount'].values.reshape(-1,1))
creditcard['scaled_time'] = robust.fit_transform(creditcard['Time'].values.reshape(-1,1))

creditcard.drop(['Time','Amount'], axis=1, inplace=True)


# In[16]:


creditcard.info()


# ___

# Now that data values are taken care of, we need to worry about how the dataset will be divided into train and test set.
# 
# We will use StratifiedShuffleSplit with n_split=1 so as to preserve the percentages of original classes in the split as well.
# 
# For more info see this link:
# https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedShuffleSplit.html

# In[17]:


X = creditcard.drop('Class', axis=1)
y = creditcard['Class']


# In[18]:


X.head()


# In[19]:


y.head()


# ___

# We will now divide the dataset to 80/20 train/test with stratified splitting.
# 
# Further, we will divide the train test from above to 80/20 for training and cross-validation

# In[20]:


x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=10, stratify=y)

"""sss1 = StratifiedShuffleSplit(n_splits=5, random_state=10, test_size=0.2)
rest, test = sss1.split(X, y)
print("Train:", rest, "Test:", test)
#X_test, y_test = X.iloc[test], y.iloc[test]"""


# #TestSet has:
# 
# X_test, y_test

# In[21]:


sss = StratifiedShuffleSplit(n_splits=3, random_state=10, test_size=0.2)


# ___

# **Function :**

# In[22]:


from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

def plot_roc(y_test, y_pred, label):
    log_fpr, log_tpr, log_thresold = roc_curve(y_test, y_pred)
    plt.plot(log_fpr, log_tpr, label=label+'{:.4f}'.format(roc_auc_score(y_test, y_pred)))

    plt.plot([0, 1], [0, 1], 'k--')
    plt.axis([-0.01, 1, 0, 1])
    plt.xlabel('False Positive Rate', fontsize=16)
    plt.ylabel('True Positive Rate', fontsize=16)
    plt.legend()


# ___

# **Model 0 :**
# 
# DummyClassifier

# In[ ]:


clf = DummyClassifier()
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)
plot_roc(y_test, y_pred, 'Dummy AUC ')


# ___

# **Model 1 :**
# 
# Logistic Regression with parameter tuning of 'C' and scoring as 'f1_micro' to get proper f1 values for imbalanced dataset classes

# In[ ]:


get_ipython().run_cell_magic('time', '', "clf = LogisticRegression(random_state=10)\nparam_grid={'C': [0.01, 0.1, 1, 10, 100]}\nscoring = 'f1_micro'\n\nGS = GridSearchCV(clf, param_grid, scoring=scoring, cv=sss.split(x_train, y_train), n_jobs=-1)\nLR_GS = GS.fit(x_train, y_train)\n\n\nprint(LR_GS.best_score_)\ny_pred = LR_GS.best_estimator_.predict(x_test)")


# In[ ]:


LR_GS.best_estimator_


# In[ ]:


plot_roc(y_test, y_pred, 'Logistic Regression AUC ')


# ___

# **Model 2 :**
# 
# Logistic Regression with parameter tuning of 'C' and 'class_weight'.
# 
# The 'class_weight' parameter helps in a imbalanced dataset by penalising the wrong classification of smaller classes more.
# 
# Also scoring metric is 'roc_auc'

# In[ ]:


get_ipython().run_cell_magic('time', '', "clf = LogisticRegression(random_state=10, penalty='l1')\nparam_grid={'C': [0.01, 0.1, 1, 10, 100],\n            'class_weight':[{0:.1,1:.9}, {0:.0017,1:0.9983}, {0:.3,1:.7}]}\nscoring = 'roc_auc'\n\nRS = RandomizedSearchCV(clf, param_grid, scoring=scoring, cv=sss.split(x_train, y_train), n_jobs=-1)\nLR_RS = RS.fit(x_train, y_train)\n\nprint(LR_RS.best_score_)\ny_pred = LR_RS.best_estimator_.predict(x_test)")


# In[ ]:


LR_RS.best_estimator_


# In[ ]:


plot_roc(y_test, y_pred, 'Logistic Regression AUC ')


# ___

# **Model 3 :**
# 
# XGBoost

# In[ ]:


params = {
        'min_child_weight': [0.5, 1, 2],
        'gamma': [1, 1.5, 2, 5],
        'subsample': [0.5, 0.8, 1.0],
        'max_depth': [3, 5, 7],
        'scale_pos_weight':[10, 100, 300]
        }

xgb = XGBClassifier(learning_rate=0.02, n_estimators=500, objective='binary:logistic',
                    verbosity=2, nthread=-1)

#disable_default_eval_metric=1, eval_metric = 'auc',

rs_xgb = RandomizedSearchCV(xgb, param_distributions=params, 
                                   scoring='roc_auc', cv=3,
                                   verbose=2, n_jobs=-1,
                                   random_state=10)

#sss.split(x_train, y_train)


# In[ ]:


get_ipython().run_cell_magic('time', '', 'rs_xgb.fit(x_train, y_train)')


# In[ ]:


rs_xgb.grid_scores_


# In[ ]:


print(rs_xgb.best_score_)
print(rs_xgb.best_params_)
y_pred = rs_xgb.best_estimator_.predict(x_test)


# In[ ]:


plot_roc(y_test, y_pred, 'XGBoost AUC ')


# ___

# **Data Sampling**
# 
# Here we will transform the original imbalanced dataset with imblearn python package - https://imbalanced-learn.readthedocs.io/

# In[23]:


get_ipython().run_cell_magic('time', '', 'sm = SMOTEENN(random_state=10)\nx_train_res, y_train_res = sm.fit_sample(x_train, y_train)')


# In[24]:


print(len(x_train_res))
print(len(y_train_res))


# In[25]:


np.unique(y_train_res)


# In[26]:


x_train_res.shape


# ___

# **Model 4 :**
# 
# Keras model to fit and validate train and test set respectively

# In[27]:


model = Sequential()
model.add(Dense(32, input_dim=30, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))


# In[28]:


model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[29]:


get_ipython().run_cell_magic('time', '', 'model.fit(x_train_res, y_train_res, epochs=20, batch_size=5000)')


# In[32]:


y_pred = model.predict(x_test)


# In[33]:


plot_roc(y_test, y_pred, 'Keras model_1 AUC ')


# In[39]:


my_submission = pd.DataFrame(y_pred)
# you could use any filename. We choose submission here
my_submission.to_csv('submission.csv', index=False)


# ___

# **Model 5**
# 
# Lets see if the data sampling improves the original logistic regression model that we used

# In[ ]:


clf = LogisticRegression(random_state=10)
param_grid={'C': [0.01, 0.1, 1, 10, 100]}
scoring = 'f1_micro'

GS = GridSearchCV(clf, param_grid, scoring=scoring, cv=3, n_jobs=-1)
LR_GS_new = GS.fit(x_train_res, y_train_res)

print(LR_GS_new.best_score_)
y_pred = LR_GS_new.best_estimator_.predict(x_test)


# In[ ]:


plot_roc(y_test, y_pred, 'Log_Reg (sampled data) AUC ')


# ___
# ___

# ### Conclusion

# The data in this dataset is extremely imbalanced so my approach to solve this problem can be divided into 2 parts:
# * Trying out varying models on the original data with varying class_weights
# * Resampling the data to have more data for the minority class and then try models

# | Model 	| with class weights 	| with Resampled Data 	| HyperParameter tuned 	| AUC on Testset 	|
# |:----------------------:	|:------------------:	|:-------------------:	|:---------------------------------------------------------------------------:	|:--------------:	|
# | DummyClassifier 	| No 	| No 	| No 	| 0.4991 	|
# | LogisticRegression 	| No 	| No 	| 'C' 	| 0.7805 	|
# | LogisticRegression 	| Yes 	| No 	| 'C', 'class_weight' 	| 0.9029 	|
# | XGBClassifier 	| Yes 	| No 	| 'min_child_weight', 'gamma',  'subsample', 'max_depth',  'scale_pos_weight' 	| 0.9275 	|
# | Keras Sequential Model 	| No 	| Yes 	| No 	| 0.9634 	|
# | LogisticRegression 	| No 	| No 	| 'C' 	| 0.9482 	|

# As we can see from the table, if we have an imbalanced dataset we have to:
# * Give weights for each class, so that the model penalises the misclassification of the class we care about (1 in this case)
# * Or Resample the data to increase or decrease the data of the classes in the dataset

# ___
# ___
