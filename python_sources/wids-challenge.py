#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

import warnings
warnings.simplefilter(action = 'ignore')

# Standardization
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler

# Train-Test split
from sklearn.model_selection import train_test_split 

# Importing classification report and confusion matrix from sklearn metrics
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Importing the PCA module
from sklearn.decomposition import PCA

# Importing random forest classifier from sklearn library
from sklearn.ensemble import RandomForestClassifier

# Importing Ridge, Lasso and GridSearch
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV

# Importing XGBoost libraries
from sklearn.ensemble import AdaBoostClassifier

import gc # for deleting unused variables

# Importing the below library and configuring to display all columns in a dataframe
from IPython.display import display
pd.options.display.max_columns = None

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# # Data Understanding and Data Cleaning

# In[ ]:


training_df = pd.read_csv('/kaggle/input/widsdatathon2020/training_v2.csv')
training_df.head()


# In[ ]:


test_df = pd.read_csv('/kaggle/input/widsdatathon2020/unlabeled.csv')
df_unlabel = test_df.copy()
test_df.head()


# In[ ]:


print(training_df.shape)
print(test_df.shape)


# In[ ]:


train_len = len(training_df)


# In[ ]:


training_df = pd.concat(objs = [training_df,test_df], axis = 0)
training_df.shape


# In[ ]:


training_df.info(verbose = True)


# In[ ]:


training_df.isnull().sum()


# In[ ]:


# percentage of missing values in columns greater than 50%
null_cols = training_df.columns[round(training_df.isnull().sum()/len(training_df.index)*100,2) > 50].tolist()
null_cols


# In[ ]:


# deleting cols having missing %age greater than 50%
print(training_df.shape)
training_df = training_df.drop(null_cols,axis = 1)
print(training_df.shape)


# ## Dividing dataframe based on numeric and non-numeric columns 

# ### Analyzing Numeric columns

# In[ ]:


# numeric columns
df_numeric = training_df.select_dtypes(include=['int16', 'int32', 'int64', 'float16', 'float32', 'float64'])
df_numeric.head() 


# ### Analyzing missing values in numeric columns of training set

# In[ ]:


# missing values in columns
num_null_cols = df_numeric.columns[df_numeric.isnull().any()].tolist()
print(num_null_cols)
print(len(num_null_cols))


# #### there are 97 columns having missing columns
# #### dividing them into 5 parts and visualizing them using box plot

# In[ ]:


# dividing the null columns in 5 part 
num_null_cols_1 = num_null_cols[:19]
num_null_cols_2 = num_null_cols[19:38]
num_null_cols_3 = num_null_cols[38:57]
num_null_cols_4 = num_null_cols[57:76]
num_null_cols_5 = num_null_cols[76:]


# In[ ]:


# Visualizing first part of num_null_cols which is num_null_cols_1 using box_plot
plt.figure(figsize=(15,15))
sns.boxplot(x="value", y="variable", data=pd.melt(df_numeric[num_null_cols_1]))
plt.show()


# #### We can impute the missing value with their mean for the above set of numeric columns as there are not much outliers present in the column set 1  

# In[ ]:


# Visualizing first part of num_null_cols which is num_null_cols_2 using box_plot
plt.figure(figsize=(15,15))
sns.boxplot(x="value", y="variable", data=pd.melt(df_numeric[num_null_cols_2]))
plt.show()


# #### In this set of columns, many outliers are present so we can impute the missing values with their median  

# In[ ]:


# Visualizing first part of num_null_cols which is num_null_cols_3 using box_plot
plt.figure(figsize=(15,15))
sns.boxplot(x="value", y="variable", data=pd.melt(df_numeric[num_null_cols_3]))
plt.show()


# #### Here also, we can impute the variables with their median 

# In[ ]:


# Visualizing first part of num_null_cols which is num_null_cols_4 using box_plot
plt.figure(figsize=(15,15))
sns.boxplot(x="value", y="variable", data=pd.melt(df_numeric[num_null_cols_4]))
plt.show()


# In[ ]:


# Visualizing first part of num_null_cols which is num_null_cols_5 using box_plot
plt.figure(figsize=(15,15))
sns.boxplot(x="value", y="variable", data=pd.melt(df_numeric[num_null_cols_5]))
plt.show()


# ### From the above observations we can conclude that we can impute the missing values of all the numeric columns with their median    

# ## Treating missing values

# ### Imputing the missing values of the numeric data frame with their median 

# In[ ]:


# imputing missing values
df_numeric = df_numeric.fillna(df_numeric.median())
print(df_numeric.isnull().sum())


# ### Analyzing Non-Numeric Columns 

# In[ ]:


# non numeric columns
df_non_numeric = training_df.select_dtypes(exclude=['int16', 'int32', 'int64', 'float16', 'float32', 'float64'])
print(df_non_numeric.head())


# ### Analyzing missing values in non-numeric columns of training set

# In[ ]:


# percentage of missing values
round(df_non_numeric.isnull().sum()/len(training_df.index)*100,2)


# In[ ]:


# Visualizing them using countplot
plt.figure(figsize=(10,15))
plt.subplot(4,1,1)
sns.countplot(y = 'ethnicity',data= df_non_numeric)
plt.subplot(4,1,2)
sns.countplot(y = 'gender',data= df_non_numeric)
plt.subplot(4,1,3)
sns.countplot(y = 'hospital_admit_source',data= df_non_numeric)
plt.subplot(4,1,4)
sns.countplot(y = 'icu_admit_source',data= df_non_numeric)
plt.show()


# #### Inference:
# - Mostly ethinicity is 'Caucasian' and gender is 'Male'
# - Most of the hospital admit source is 'Emergency' 
# - And ICU admit score is 'Accident and Emergency'

# In[ ]:


# Visualizing them using countplot
plt.figure(figsize=(10,15))
plt.subplot(4,1,1)
sns.countplot(y = 'icu_stay_type',data= df_non_numeric)
plt.subplot(4,1,2)
sns.countplot(y = 'icu_type',data= df_non_numeric)
plt.subplot(4,1,3)
sns.countplot(y = 'apache_3j_bodysystem',data= df_non_numeric)
plt.subplot(4,1,4)
sns.countplot(y = 'apache_2_bodysystem',data= df_non_numeric)
plt.show()


# #### Inference:
# - icu_stay_type is mostly 'admit'
# - icu_type is 'Med_Surg ICU'
# - apache body stream is 'Cardiovascular'

# #### The above analysis concludes that we can impute the missing values with their respective modes

# ## Treating the missing values

# ### Imputing the missing values of non numeric columns with their modes 

# In[ ]:


for column in df_non_numeric.columns:
    df_non_numeric[column].fillna(df_non_numeric[column].mode()[0], inplace = True)


# In[ ]:


df_non_numeric.isnull().sum()


# ## Merging numeric and non-numeric dataFrame back to original training dataFrame based on their index  

# In[ ]:


# merging df_numeric and df_non_numeric on their index
df_train = pd.concat([df_numeric, df_non_numeric], axis=1)
df_train.head()


# In[ ]:


df_train.shape


# ## Treating Outliers

# In[ ]:


# checking outliers
df_train.describe(percentiles=[.25,.5,.75,.90,.95,.99])


# ### On treating the outliers the large part of data is getting deleted, therefore we are not treating them and moving forward 

# ## Checking death rate

# In[ ]:


df_train['hospital_death'].value_counts().plot('bar')
plt.show()


# In[ ]:


df_train['hospital_death'].sum()/len(df_train['hospital_death'].index)*100


# ### We have almost 5% death rate

# ## Checking correlation of death with every variable 

# In[ ]:


# get correlation of 'hospital_death' with other variables
plt.figure(figsize=(30,9))
df_train.corr()['hospital_death'].sort_values(ascending = False).plot('bar')
plt.show()


# ### Inferences:
# - 'hospital_death' is highly positively correlated to 'apache_4a_hospital_death_prob', 'apache_4a_icu_death_prob' and 'ventilated_apache'
# - 'hospital_death' is highly negatively correlated to 'gcs_eyes_apache', 'gcs_motor_apache' and 'readmission_status' 

# # Data Preparation

# In[ ]:


df_train.head()


# In[ ]:


df_train.shape


# ### Dropping 'encounter_id' and 'patient_id' as they have nothing to do with 'hospital_death' 

# In[ ]:


df_train = df_train.drop(['encounter_id','patient_id'], axis = 1)
df_train.columns


# In[ ]:


import copy
train = copy.copy(df_train[:train_len])
test = copy.copy(df_train[train_len:])
print('train             ', train.shape)
print('test              ', test.shape)


# ## Separating train dataset into independent(X) and dependent(y) variables 

# In[ ]:


X = train.drop('hospital_death',1)
X.head()


# In[ ]:


X.shape


# In[ ]:


y = train['hospital_death']
y.head()


# In[ ]:


test = test.drop('hospital_death', axis = 1)
test.shape


# ## Creating dummy variables for categorical columns of X

# In[ ]:


X.info(verbose=True)


# In[ ]:


X.shape


# In[ ]:


test.shape


# In[ ]:


X_len = len(X)


# In[ ]:


combined_df = pd.concat(objs = [X,test], axis = 0)
combined_df.shape


# #### 'elective_surgery', 'readmission_status', 'apache_post_operative', 'arf_apache', 'gcs_unable_apache', 'intubated_apache', 'ventilated_apache', 'aids', 'cirrhosis', 'diabetes_mellitus', 'hepatic_failure', 'immunosuppression', 'leukemia', 'lymphoma', 'solid_tumor_with_metastasis' are categorical therefore changing their datatype as object  

# In[ ]:


combined_df[['elective_surgery','readmission_status','apache_post_operative','arf_apache','gcs_unable_apache','intubated_apache','ventilated_apache','aids','cirrhosis','diabetes_mellitus','hepatic_failure','immunosuppression','leukemia','lymphoma','solid_tumor_with_metastasis']] = combined_df[['elective_surgery','readmission_status','apache_post_operative','arf_apache','gcs_unable_apache','intubated_apache','ventilated_apache','aids','cirrhosis','diabetes_mellitus','hepatic_failure','immunosuppression','leukemia','lymphoma','solid_tumor_with_metastasis']].astype(object)


# In[ ]:


combined_df_categorical = combined_df[['elective_surgery','readmission_status','apache_post_operative','arf_apache','gcs_unable_apache',
                   'intubated_apache','ventilated_apache','aids','cirrhosis','diabetes_mellitus','hepatic_failure',
                   'immunosuppression','leukemia','lymphoma','solid_tumor_with_metastasis','ethnicity','gender','hospital_admit_source',
                   'icu_stay_type','icu_type','apache_3j_bodysystem','apache_2_bodysystem','icu_admit_source']]

# convert into dummies
combined_dummies = pd.get_dummies(combined_df_categorical, drop_first=True)

# drop cateorical variables from X dataframe
combined_df = combined_df.drop(combined_df_categorical, axis = 1)

# concat dummy variables with X dataframe
combined_df = pd.concat([combined_df, combined_dummies], axis = 1)
print(combined_df.shape)


# In[ ]:


import copy
X = copy.copy(combined_df[:X_len])
test = copy.copy(combined_df[X_len:])
print('X             ', X.shape)
print('test          ', test.shape)


# ## Checking Correlation between variables of X

# In[ ]:


X.corr()


# ### Deleting variables from 'X' data frame having more than 80% correlation[](http://)

# In[ ]:


# create correlation matrix
corr_matrix = X.corr().abs()

# select upper triangle of correlation matrix
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape),k=1).astype(np.bool))

# find index of feature columns with correlation greater than 0.80
to_drop = [column for column in upper.columns if any((upper[column]>0.80))]
to_drop


# In[ ]:


X = X.drop(to_drop,1)
print(X.shape)


# In[ ]:


test = test.drop(to_drop ,1)
print(test.shape)


# In[ ]:


# plotting heat map to see correlation 
plt.figure(figsize=(15,10))
sns.heatmap(data = X.corr())
plt.show()


# ## Feature Scaling 

# In[ ]:


X.info(verbose = True)


# In[ ]:


# standardization of X
scaler = preprocessing.StandardScaler().fit(X)
X = scaler.transform(X)
X = pd.DataFrame(X)

# test
test = scaler.transform(test)
test = pd.DataFrame(test)


# In[ ]:


test.shape


# ## Splitting the data into train and test set

# In[ ]:


# splitting the data into train and test
X_train, X_test, y_train, y_test = train_test_split(X,y, train_size = 0.7, test_size = 0.3, random_state = 100)


# In[ ]:


X_orig = X.copy()
y_orig = y.copy()


# ## Model Building

# ## Modelling using decision tree

# ## Random forest with default parameters

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

X = X_orig.copy()
y = y_orig.copy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100)

model_rf = RandomForestClassifier()
model_rf.fit(X_train, y_train)

# Make predictions
prediction_test = model_rf.predict(X_test)
print(classification_report(y_test,prediction_test))


# In[ ]:


# Printing confusion matrix
print(confusion_matrix(y_test, prediction_test))


# In[ ]:


# Defining a genric function to calculate sensitivity
def sensitivity_score(inp_y_test, inp_y_pred):
    positives = confusion_matrix(inp_y_test,inp_y_pred)[1]
    # print('True Positives: ', positives[1], ' False Positives: ', positives[0])
    return (positives[1]/(positives[1] + positives[0]))


# In[ ]:


print ('Random Forest Accuracy with Default Hyperparameter', metrics.accuracy_score(y_test, prediction_test))
print ('Random Forest Sensitivity with Default Hyperparameter', sensitivity_score(y_test, prediction_test))


# ### With default parameters, we get a 92% accuracy in Random forest model

# In[ ]:


import lightgbm as lgb
from sklearn.model_selection import RandomizedSearchCV
clf = lgb.LGBMClassifier(silent=True, random_state = 304, metric='roc_auc', n_jobs=4)


# In[ ]:


from scipy.stats import randint as sp_randint
from scipy.stats import uniform as sp_uniform
params ={'cat_smooth' : sp_randint(1, 100), 'min_data_per_group': sp_randint(1,1000), 'max_cat_threshold': sp_randint(1,100)}


# In[ ]:


fit_params={"early_stopping_rounds":2, 
            "eval_metric" : 'auc', 
            "eval_set" : [(X_train, y_train),(X_test,y_test)],
            'eval_names': ['train','valid'],
            'verbose': 300,
            'categorical_feature': 'auto'}


# In[ ]:


gs = RandomizedSearchCV( estimator=clf, param_distributions=params, scoring='roc_auc',cv=3, refit=True,random_state=304,verbose=True)


# In[ ]:


gs.fit(X_train, y_train, **fit_params)
print('Best score reached: {} with params: {} '.format(gs.best_score_, gs.best_params_))


# In[ ]:


gs.best_params_, gs.best_score_


# In[ ]:


# clf2 = lgb.LGBMClassifier(random_state=304, metric = 'roc_auc', n_jobs=4)
clf2 = lgb.LGBMClassifier(random_state=304, metric = 'roc_auc', cat_smooth = 32, max_cat_threshold = 75, min_data_per_group = 82, n_jobs=4)


# In[ ]:


params_2 = {'learning_rate': [0.04, 0.05, 0.08],   
            'num_iterations': [1400,1600, 1800]}


# In[ ]:


gs2 = GridSearchCV(clf2,params_2, scoring='roc_auc',cv=3)


# In[ ]:


gs2.fit(X_train, y_train, **fit_params)
print('Best score reached: {} with params: {} '.format(gs2.best_score_, gs2.best_params_))


# In[ ]:


gs2.best_params_, gs2.best_score_


# In[ ]:


params_2 = {
 'bagging_fraction': 0.4,
 'boosting': 'dart',
 'num_iterations': 1400, 
 'learning_rate': 0.04,
 'colsample_bytree': 0.5048747931447324,
 'cat_smooth': 32, 
 'max_cat_threshold':75, 
 'min_data_per_group': 82,
 'max_bin': 1312,
 'max_depth': 12,
 'num_leaves': 4090,
 'min_child_samples': 407,
 'min_child_weight': 0.1,
 'min_data_in_leaf': 2420,
 'reg_alpha': 0.1,
 'reg_lambda': 20,
 'scale_pos_weight': 3,
 'subsample': 0.7340872997512691,
 'subsample_for_bin': 512,
 'scoring': 'roc_auc',
 'metric': 'auc',
 'objective': 'binary'}


# In[ ]:


lgbm_train2 = lgb.Dataset(X_train, y_train)
lgbm_val2 = lgb.Dataset(X_test, y_test)


# In[ ]:


evals_result = {}  # to record eval results for plotting
model_lgbm_2 = lgb.train(params_2,
                lgbm_train2,
                num_boost_round=250,
                valid_sets=[lgbm_train2, lgbm_val2],
                feature_name=['f' + str(i + 1) for i in range(X_train.shape[-1])],
                categorical_feature= [182],
                evals_result=evals_result,
                verbose_eval=100)


# In[ ]:


ax = lgb.plot_metric(evals_result, metric='auc', figsize=(15, 8))
plt.show()


# In[ ]:


df_unlabel["hospital_death"] = model_lgbm_2.predict(test, pred_contrib=False)


# In[ ]:


df_unlabel.shape


# In[ ]:


df_unlabel.head()


# In[ ]:


df_unlabel[['encounter_id','hospital_death']].to_csv('submission.csv',index = False)
df_unlabel[['encounter_id','hospital_death']].head()


# In[ ]:




