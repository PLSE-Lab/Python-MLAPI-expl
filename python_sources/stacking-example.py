#!/usr/bin/env python
# coding: utf-8

# ## __The importance of insulin__
# 
# - Diabetes is a disease in which your body either can't produce insulin or can't properly use the insulin it produces. Insulin is a hormone produced by your pancreas.
# 
# - Insulin's role is to regulate the amount of glucose (sugar) in the blood. Blood sugar must be carefully regulated to ensure that the body functions properly. Too much blood sugar can cause damage to organs, blood vessels, and nerves. Your body also needs insulin in order to use sugar for energy.
# 
# There are generally 2 types of diabetes:
# 
# 1. Type 1 diabetes is an autoimmune disease and is also known as insulin-dependent diabetes. People with type 1 diabetes aren't able to produce their own insulin (and can't regulate their blood sugar) because their body is attacking the pancreas. Roughly 10 per cent of people living with diabetes have type 1, insulin-dependent diabetes. Type 1 diabetes generally develops in childhood or adolescence, but can also develop in adulthood. People with type 1 need to inject insulin or use an insulin pump to ensure their bodies have the right amount of insulin. 
# 
# 
# 2. People with type 2 diabetes can't properly use the insulin made by their bodies, or their bodies aren't able to produce enough insulin. Roughly 90 per cent of people living with diabetes have type 2 diabetes.Type 2 diabetes is most commonly developed in adulthood, although it can also occur in childhood. Type 2 diabetes can sometimes be managed with healthy eating and regular exercise alone, but may also require medications or insulin therapy.  
# 
# (Source: https://www.diabetes.ca/diabetes-basics/what-is-diabetes)

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import warnings
warnings.filterwarnings('ignore')

# Going to use these 5 base models for the stacking
from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier, 
                              GradientBoostingClassifier, ExtraTreesClassifier)
from sklearn.svm import SVC
from sklearn.model_selection import KFold


# In[ ]:


#Read in the dataset, and giving back their headers
data = pd.read_csv('../input/pima-indians-diabetes.csv', header = None, names = 
                   ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
                    'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Class'])


# In[ ]:


data.head()


# In[ ]:


data.describe()


# In[ ]:


data.info()


# - No missing values, 768 entries, all numbers
# - Class is the target variable

# ## __Heat Map__

# In[ ]:


colormap = plt.cm.inferno
plt.figure(figsize=(14,12))
plt.title('Pearson Correlation of Features', y=1.05, size=15)
sns.heatmap(data.astype(float).corr(),linewidths=0.1,vmax=1.0, 
            square=True, cmap=colormap, linecolor='white', annot=True)


# __Take aways from the heatmap:__
# - Glucose has the highest correlation with our target variable. High glucose would likely cause diabetes
# - BMI, Age and Pregnancies has the secondary highest correlation with target variable
# - However, age is relatively highly correlated with number of times of pregnancies as people ages they will be able to pregenant more times
# - Insulin and skinthickness also has relatively high correlation

# In[ ]:


#Separate the predictors and response variables
y = data['Class'].ravel()
X = data.drop(['Class'], axis=1)


# In[ ]:


#Defining some variables
SEED = 2019 # for reproducibility
NFOLDS = 3
kfold = KFold(n_splits = NFOLDS, random_state = SEED)


# In[ ]:


#Separate training and testing dataset
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = SEED)


# In[ ]:


ntrain = x_train.shape[0]  #Number of rows per training column (recall: train.shape = (3, 4), then shape[0] is 3)
ntest = x_test.shape[0]  #testing column
x_train = x_train.values


# In[ ]:


# Class to extend the Sklearn classifier
class SklearnHelper(object):
    def __init__(self, clf, seed=0, params=None):
        params['random_state'] = seed
        self.clf = clf(**params)

    def train(self, x_train, y_train):
        self.clf.fit(x_train, y_train)

    def predict(self, x):
        return self.clf.predict(x)
    
    def fit(self,x,y):
        return self.clf.fit(x,y)
    
    def feature_importances(self,x,y):
        print(self.clf.fit(x,y).feature_importances_)


# In[ ]:


def get_oof(clf, x_train, y_train, x_test):
    oof_train = np.zeros((ntrain,))  #Establish an array of 0s that has the same length of the training data
    oof_test = np.zeros((ntest,))
    oof_test_skf = np.empty((NFOLDS, ntest))  #Establish an array consist of 5 rows and number of test columns

    for i, (train_index, test_index) in enumerate(kfold.split(x_train)):  #Elocate the splited data into their dataset
        x_tr = x_train[train_index]   #train index of train dataset
        y_tr = y_train[train_index]   #train index of test dataset
        x_te = x_train[test_index]  #test index of train dataset

        clf.train(x_tr, y_tr)

        oof_train[test_index] = clf.predict(x_te)
        oof_test_skf[i, :] = clf.predict(x_test)

    oof_test[:] = oof_test_skf.mean(axis=0)
    return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)


# In[ ]:


# Put in our parameters for said classifiers
# Random Forest parameters
rf_params = {
    'n_jobs': -1,
    'n_estimators': 300,
     'warm_start': True, 
     'max_features': 0.7,
    'max_depth': 6,
    'min_samples_leaf': 2,
    'max_features' : 'sqrt',
    'verbose': 0
}

# Extra Trees Parameters
et_params = {
    'n_jobs': -1,
    'n_estimators':300,
    #'max_features': 0.5,
    'max_depth': 8,
    'min_samples_leaf': 2,
    'verbose': 0
}

# AdaBoost parameters
ada_params = {
    'n_estimators': 500,
    'learning_rate' : 0.1
}

# Gradient Boosting parameters
gb_params = {
    'n_estimators': 500,
     'max_features': 0.7,
    'max_depth': 5,
    'min_samples_leaf': 2,
    'verbose': 0
}

# Support Vector Classifier parameters 
svc_params = {
    'kernel' : 'linear',
    'C' : 0.025
    }


# In[ ]:


# Create 5 objects that represent our 4 models
rf = SklearnHelper(clf=RandomForestClassifier, seed=SEED, params=rf_params)
et = SklearnHelper(clf=ExtraTreesClassifier, seed=SEED, params=et_params)
ada = SklearnHelper(clf=AdaBoostClassifier, seed=SEED, params=ada_params)
gb = SklearnHelper(clf=GradientBoostingClassifier, seed=SEED, params=gb_params)
svc = SklearnHelper(clf=SVC, seed=SEED, params=svc_params)


# In[ ]:


# Create our OOF train and test predictions. These base results will be used as new features
et_oof_train, et_oof_test = get_oof(et, x_train, y_train, x_test) # Extra Trees
rf_oof_train, rf_oof_test = get_oof(rf,x_train, y_train, x_test) # Random Forest
ada_oof_train, ada_oof_test = get_oof(ada, x_train, y_train, x_test) # AdaBoost 
gb_oof_train, gb_oof_test = get_oof(gb,x_train, y_train, x_test) # Gradient Boost
svc_oof_train, svc_oof_test = get_oof(svc,x_train, y_train, x_test) # Support Vector Classifier

print("Training is complete")


# In[ ]:


#Checking feature importance
rf_feature = rf.feature_importances(x_train,y_train)
et_feature = et.feature_importances(x_train, y_train)
ada_feature = ada.feature_importances(x_train, y_train)
gb_feature = gb.feature_importances(x_train,y_train)


# In[ ]:


base_predictions_train = pd.DataFrame( {'RandomForest': rf_oof_train.ravel(),
     'ExtraTrees': et_oof_train.ravel(),
     'AdaBoost': ada_oof_train.ravel(),
      'GradientBoost': gb_oof_train.ravel()
    })
base_predictions_train.head()


# Now plotling the heatmap of our second level models:
# We want them to be as diverse as possible

# In[ ]:


import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls


# In[ ]:


data = [
    go.Heatmap(
        z = base_predictions_train.astype(float).corr().values ,
        x = base_predictions_train.columns.values,
        y = base_predictions_train.columns.values,
          colorscale='Viridis',
            showscale=True,
            reversescale = True
    )
]
py.iplot(data, filename='labelled-heatmap')


# In[ ]:


x_train = np.concatenate(( et_oof_train, rf_oof_train, ada_oof_train, gb_oof_train, svc_oof_train), axis=1)
x_test = np.concatenate(( et_oof_test, rf_oof_test, ada_oof_test, gb_oof_test, svc_oof_test), axis=1)


# In[ ]:


import xgboost as xgb


# In[ ]:


gbm = xgb.XGBClassifier(learning_rate = 0.02,
     n_estimators= 2000,
     max_depth= 4,
     min_child_weight= 2,
     #gamma=1,
     gamma=0.9,                        
     subsample=0.8,
     colsample_bytree=0.8,
     objective= 'binary:logistic',
     nthread= -1,
     scale_pos_weight=1).fit(x_train, y_train)

predictions = gbm.predict(x_test)


# In[ ]:


from sklearn.metrics import accuracy_score

accuracy_score(y_test, predictions)


# In[ ]:





# In[ ]:





# In[ ]:




