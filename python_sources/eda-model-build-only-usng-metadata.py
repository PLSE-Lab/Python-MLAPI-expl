#!/usr/bin/env python
# coding: utf-8

# ### Libraries

# In[ ]:


# Basic numerical and other libraries
import random
import numpy as np
import pandas as pd
from scipy import stats
import sys
import functools

# Display option
from IPython.display import display, HTML
pd.options.display.max_rows = 5000
pd.options.display.max_columns = 5000

# Handle warnings (during execution of code)
import warnings
warnings.filterwarnings('ignore')

# Datetime
import time
from datetime import datetime
from datetime import timedelta

# Visulisation
from pprint import pprint
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# statsmodel
import statsmodels.api as statsm
import statsmodels.discrete.discrete_model as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Imbalanced Data Handling
import imblearn
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

# sklearn
from sklearn.model_selection import StratifiedKFold, KFold, LeaveOneOut, cross_val_score, GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.metrics import auc, roc_auc_score, roc_curve


# xgboost, lightgbm
from xgboost import XGBClassifier


# ### User-Defined Functions

# In[ ]:


def plot_stats(df, feature, target_ftr, label_rotation=False, horizontal_layout=True):
    '''
    This function plot the categorical feature distribution according to target variable
    '''
    temp = df[feature].value_counts()
    df1 = pd.DataFrame({feature: temp.index,'Number of Patients': temp.values})

    # Calculate the percentage of target=1 per category value
    cat_perc = df[[feature, target_ftr]].groupby([feature],as_index=False).mean()
    cat_perc.sort_values(by=target_ftr, ascending=False, inplace=True)
    
    sns.set_color_codes("pastel")
    s = sns.barplot(x = feature, y="Number of Patients",data=df1)
    if(label_rotation):
        s.set_xticklabels(s.get_xticklabels(),rotation=60)

    plt.tick_params(axis='both', which='major', labelsize=10)

    plt.show();


def get_rocauc(model, xTest, yTest): 
    '''
    This function produces the Area under the curve for the model. 
    The 'auto' method calculates this metric by using the roc_auc_score function from sklearn.
    Range: 0 to 1 (0 being the worst predictive model, 0.5 being the random and 1 being the best)
    '''
    predictions = model.predict_proba(xTest)[:, 1]
    roc_auc = roc_auc_score(yTest, predictions)
    print('Model Performance:')
    print('--'*5)
    print('--'*5)
    print('ROC = {:0.2f}%'.format(roc_auc))
    
    return roc_auc


# ### Import Data

# In[ ]:


df_train = pd.read_csv('../input/siim-isic-melanoma-classification/train.csv')
df_test  = pd.read_csv('../input/siim-isic-melanoma-classification/test.csv')
df_sub   = pd.read_csv('../input/siim-isic-melanoma-classification/sample_submission.csv')


# #### 1. #records and #features

# In[ ]:


# shape of training, test and submission data

print(f'training data shape: {df_train.shape}')
print(f'test data shape: {df_test.shape}')
print(f'submission data shape: {df_sub.shape}')


# * Need to analyse how more than 1 feature is less in test data than in training data.

# #### How does each of the datasets look like ?

# #### train dataset

# In[ ]:


df_train.head()


# #### test dataset

# In[ ]:


df_test.head()


# * 'diagnosis' and 'benign_malignant' are missing in test dataset. 
# * So while building any algorithm we will not be able to use these 2 features.

# #### submission dataset

# In[ ]:


df_sub.head()


# * Here, we may need to replace target with the predicted values from the trained model.

# #### Let's check the unique values in the targets of training and submission datasets

# In[ ]:


# training dataset

df_train['target'].unique()


# In[ ]:


# submission dataset

df_sub['target'].unique()


# * target values in training dataset looks good and in submission each and every image is assumed as benign.
# * Our job is to predict the probability of an image being malignant.

# ### EDA

# #### Target Class Distribution

# In[ ]:


sns.countplot(df_train.target)


# In[ ]:


df_train['target'].value_counts()


# * Huge Class imbalance is there.
# * We may use some sampling techniques to treat the class-imbalance. (**Will try after running the baseline model)

# #### Gender Distribution

# In[ ]:


# Gender Distribution

plot_stats(df_train, 'sex', 'target', label_rotation=False, horizontal_layout=True)


# * Train data consists a very balanced gender distribution of patients.
# 

# #### anatom_site_general_challenge Distribution

# In[ ]:


# anatom_site_general_challenge Distribution

plot_stats(df_train, 'anatom_site_general_challenge', 'target', label_rotation=90, horizontal_layout=True)


# #### Age Distribution of Patients according to target

# In[ ]:


# Age Distribution of Patients according to target

fig, axes = plt.subplots(1, 2)

fig.set_size_inches(12, 4)

df_train[df_train['target']==0].hist('age_approx', bins=100, ax=axes[0])
axes[0].set_xlabel('benign')
df_train[df_train['target']==1].hist('age_approx', bins=100, ax=axes[1])
axes[1].set_xlabel('malignant')

plt.show()


# * This can help in distinguishing between benign and malignant as median is less for benign

# ### Data Preprocessing

# In[ ]:


df_train.head(3)


# #### Let's check the null values

# #### Missing Values

# * Training Data

# In[ ]:


# Calculate missing value count and percentage

missing_value_df_train = pd.DataFrame(index = df_train.keys(), data =df_train.isnull().sum(), columns = ['Missing_Value_Count'])
missing_value_df_train['Missing_Value_Percentage'] = ((df_train.isnull().mean())*100)
missing_value_df_train.sort_values('Missing_Value_Count',ascending= False)


# * Test Data

# In[ ]:


# Calculate missing value count and percentage

missing_value_df_test = pd.DataFrame(index = df_test.keys(), data =df_test.isnull().sum(), columns = ['Missing_Value_Count'])
missing_value_df_test['Missing_Value_Percentage'] = ((df_test.isnull().mean())*100)
missing_value_df_test.sort_values('Missing_Value_Count',ascending= False)


# * we need to impute these 3 features    
# 
#   1. anatom_site_general_challenge
#   2. age_approx
#   3. sex

# #### Imputation

# * Training Data

# In[ ]:


# Replace age with median

age_array = df_train[df_train["age_approx"]!=np.nan]["age_approx"]
df_train["age_approx"].replace(np.nan, age_array.median(), inplace=True)


# Replace sex and anatom_site_general_challenge with mode

sex_array = df_train[~(df_train["sex"].isnull())]["sex"]
df_train['sex'].fillna(sex_array.mode().values[0], inplace=True)

anatom_array = df_train[~(df_train["anatom_site_general_challenge"].isnull())]["anatom_site_general_challenge"]
df_train['anatom_site_general_challenge'].fillna(anatom_array.mode().values[0], inplace=True)


# * Test Data

# In[ ]:


# Replace age with median for test data

# age_array = df_test[df_test["age_approx"]!=np.nan]["age_approx"]
# df_test["age_approx"].replace(np.nan, age_array.median(), inplace=True)


# # Replace sex and anatom_site_general_challenge with mode

# sex_array = df_test[~(df_test["sex"].isnull())]["sex"]
# df_test['sex'].fillna(sex_array.mode().values[0], inplace=True)

anatom_array_test = df_test[~(df_test["anatom_site_general_challenge"].isnull())]["anatom_site_general_challenge"]
df_test['anatom_site_general_challenge'].fillna(anatom_array_test.mode().values[0], inplace=True)


# #### Categorical Feature Handling

# In[ ]:


# Unique values for feature 'sex'

print('Unique values for gender:')
print(df_train['sex'].unique())
print('--'*20)
print('--'*20)
print('Unique values for anatom_site_general_challenge:')
print(df_train['anatom_site_general_challenge'].unique())


# * Training Data 

# In[ ]:


## Feature 'sex'

# Need to convert the datatypes of the feature to 'category' before Label encoding.
df_train["sex"] = df_train["sex"].astype('category')

# Label Encoding
df_train["sex_cat"] = df_train["sex"].cat.codes

df_train[["sex", "sex_cat"]].head(3)


# In[ ]:


## Feature 'anatom_site_general_challenge'

# Need to convert the datatypes of the feature to 'category' before Label encoding.
df_train["anatom_site_general_challenge"] = df_train["anatom_site_general_challenge"].astype('category')

# Label Encoding
df_train["anatom_site_general_challenge_cat"] = df_train["anatom_site_general_challenge"].cat.codes

df_train[["anatom_site_general_challenge", "anatom_site_general_challenge_cat"]].head(3)


# * Test Data

# In[ ]:


### Categorical Data Handling for test data


## Feature 'sex'

# Need to convert the datatypes of the feature to 'category' before Label encoding.
df_test["sex"] = df_test["sex"].astype('category')

# Label Encoding
df_test["sex_cat"] = df_test["sex"].cat.codes

df_test[["sex", "sex_cat"]].head(3)


# In[ ]:


### Categorical Data Handling for test data

## Feature 'anatom_site_general_challenge'

# Need to convert the datatypes of the feature to 'category' before Label encoding.
df_test["anatom_site_general_challenge"] = df_test["anatom_site_general_challenge"].astype('category')

# Label Encoding
df_test["anatom_site_general_challenge_cat"] = df_test["anatom_site_general_challenge"].cat.codes

df_test[["anatom_site_general_challenge", "anatom_site_general_challenge_cat"]].head(3)


# ## Feature Selection

# In[ ]:


# feature set

ftr_set = ['sex_cat',
           'age_approx',
           'anatom_site_general_challenge_cat']


# ### Training, Validation & Test Data Preparation

# In[ ]:


# dependent and independent features of training and test datasets

exog_train = df_train[ftr_set]
endog_train = df_train['target']

exog_test = df_test[ftr_set]


# * We need to prepare a validation set for simple baseline accuracy calculation.
# * I will try to make as small as possible the validation set as 33k records are there in training dataset.

# ### Imbalanced Data Handling

# #### Oversampling

# In[ ]:


# define oversampling strategy
oversample = RandomOverSampler(sampling_strategy=0.7)
X_over, y_over = oversample.fit_resample(exog_train, endog_train)

X_over = pd.DataFrame(X_over)
y_over = pd.DataFrame(y_over)


# In[ ]:


# print(X_over.shape)
# print(y_over.shape)


# #### Undersampling

# In[ ]:


# define undersampling strategy
undersample = RandomUnderSampler(sampling_strategy=0.1)
X_under, y_under = undersample.fit_resample(exog_train, endog_train)

X_under = pd.DataFrame(X_under)
y_under = pd.DataFrame(y_under)


# In[ ]:


# print(X_under.shape)
# print(y_under.shape)


# #### Training & Validation data split

# * Oversampled

# In[ ]:


# Oversampled train and validation split

x_train, x_val, y_train, y_val = train_test_split(X_over, y_over, test_size=0.2, stratify=y_over, random_state=42)


# * Undersampled

# In[ ]:


# Undersampled train and validation split

x_train, x_val, y_train, y_val = train_test_split(X_under, y_under, test_size=0.2, stratify=y_under, random_state=42)


# * Insample (Without imbalanced class treatment)

# In[ ]:


# Insample train and validation split

x_train, x_val, y_train, y_val = train_test_split(exog_train, endog_train, test_size=0.1, stratify=endog_train, random_state=42)


# ### Model Build

# #### Random Forest

# In[ ]:


# Random Forest Model

rf = RandomForestClassifier(random_state = 42)


# In[ ]:


# Parameters used by the current forest

print('Parameters currently in use:\n')
pprint(rf.get_params())


# In[ ]:


rf_base_model = RandomForestClassifier(n_estimators = 100, max_depth=5, random_state = 42)
rf_base_model.fit(x_train, y_train)
base_accuracy = get_rocauc(rf_base_model, x_val, y_val)


# #### Cross-Validation****

# In[ ]:


# # Compute cross-validated AUC scores: cv_auc

# cv_auc = cross_val_score(rf_base_model, x_val, y_val, cv=5, scoring = 'roc_auc')

# print("AUC scores computed using 5-fold cross-validation: {}".format(cv_auc))


# #### XGBoost Model

# In[ ]:


# fit model no training data
xgb_model = XGBClassifier()
xgb_model.fit(x_train, y_train)

# # make predictions for test data
# y_pred = xgb_model.predict(X_test)
base_accuracy_xgb = get_rocauc(xgb_model, x_val, y_val)


# #### Cross-validation score

# In[ ]:


# Compute cross-validated AUC scores: cv_auc

xgb_cv_auc = cross_val_score(xgb_model, x_val, y_val, cv=5, scoring = 'roc_auc')

print("AUC scores computed using 5-fold cross-validation: {}".format(xgb_cv_auc))


# ### Predictions

# In[ ]:


rf_pred_test= rf_base_model.predict_proba(exog_test)[:, 1]

print(rf_pred_test)


# In[ ]:


xgb_pred_test= xgb_model.predict_proba(exog_test)[:, 1]

print(xgb_pred_test)


# In[ ]:


# df_sub['target']=list(rf_pred_test)
df_sub['target']=list(xgb_pred_test)


# In[ ]:


df_sub.head()


# In[ ]:


df_sub.to_csv( 'submission.csv', index=False )


# ### Work in Progress !

# #### Future Work:
# 
# 1. Include image data
# 2. Tune model

# ## Upvote if you find this useful !
