#!/usr/bin/env python
# coding: utf-8

# # In this kernel I'll explore data with no cleaning and FE and see areas of improvement.

# ## Imports

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, KFold
import matplotlib.pyplot as plt 
from sklearn.metrics import roc_auc_score
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)

import lightgbm

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        


# Any results you write to the current directory are saved as output.


# In[ ]:


drop_cols = ['encounter_id','patient_id','icu_id', 'hospital_id', 'readmission_status']
drop_cols_test = ['encounter_id','patient_id','icu_id','hospital_death','hospital_id', 'readmission_status']

train = pd.read_csv("/kaggle/input/widsdatathon2020/training_v2.csv").drop(drop_cols, axis=1)
# sample_submission = pd.read_csv("/kaggle/input/widsdatathon2020/samplesubmission.csv")
test = pd.read_csv("/kaggle/input/widsdatathon2020/unlabeled.csv").drop(drop_cols_test, axis=1)
data_dictionary = pd.read_csv("/kaggle/input/widsdatathon2020/WiDS Datathon 2020 Dictionary.csv")
solution_template = pd.read_csv("/kaggle/input/widsdatathon2020/solution_template.csv")

target = 'hospital_death'


# In[ ]:


print(f'Rows in train data : {train.shape[0]} and columns in train data: {train.shape[1]}')
print(f'Rows in test data  : {test.shape[0]} and columns in train data: {test.shape[1]}')


# ### One extra column in train data is hospital_death 

# ## Columns dropped: encounter_id, patient_id, hospital_id and icu_id and readmission status:
# 1. ID columns seem to be noise here.
# 2. Readmission status has only value in both train and test. Not carrying useful information.
#  

# # Check target distribution

# In[ ]:


np.round(train[target].value_counts()*100/len(train[target]),2)


# #### 91.37% of data belong to class 0 and 8.63% of data belong to class 1. Clear imbalance, so it's necessary to setup proper validation scheme. That's why I'll use Stratified K Fold cross validation. 

# In[ ]:


ax = sns.countplot(train[target])
for p in ax.patches:
    ax.annotate('{}'.format(p.get_height()), (p.get_x()+0.15, p.get_height()+1))
plt.show()


# In[ ]:


train.info()


# In[ ]:


train.head()


# ## Percentage of missing values in each column

# ## Train data

# In[ ]:


np.round(train.isna().mean()*100,2)


# ## Test data

# In[ ]:


np.round(test.isna().mean()*100,2)


# ### We can see there are lots of missing values in our data. **Proper filling of missing values can definitely boost up score.**

# ### Data types in our data

# In[ ]:


train.dtypes


# ### Object columns in our data 

# In[ ]:


train.select_dtypes(include='O').columns.values.tolist()


# In[ ]:


test.select_dtypes(include='O').columns.values.tolist()


# # Helper functions

# In[ ]:


def explore_variable(col_name):
    """
    Helper function for categorical variable
    """
    print(f"Unique values in train: {train[col_name].unique()}")
    print(f"Unique values in test:  {test[col_name].unique()}")
    print(f"Number of unique values in train : {train[col_name].nunique()}") 
    print(f"Number of unique values in test: {test[col_name].nunique()}")

def count_plot(col_name, fig_size=(10,10)):
    """
    Helper function for count plot. 
    Here in count plot I have ordered by train[col].value_counts so it is easy compare distribution between train and test
    """
    fig = plt.figure(figsize=fig_size)
    fig.add_subplot(2,1,1)            
    ax1 = sns.countplot(x=col_name, data=train, order = train[col_name].value_counts().index)
    for p in ax1.patches:
        ax1.annotate('{}'.format(p.get_height()), (p.get_x()+0.15, p.get_height()+1))
    ax1.set_title("Train distribution", fontsize='large')
    ax1.set_ylabel(col_name)
    fig.add_subplot(2,1,2)            
    ax2 = sns.countplot(x=col_name, data=test, order = train[col_name].value_counts().index)
    ax2.set_title("Test distribution", fontsize='large')
    ax2.set_ylabel(col_name)
    for p in ax2.patches:
        ax2.annotate('{}'.format(p.get_height()), (p.get_x()+0.15, p.get_height()+1))
    
    plt.show()                        


# # EDA 

# ## 1. Ethnicity

# In[ ]:


explore_variable('ethnicity')


# In[ ]:


count_plot(col_name ='ethnicity', fig_size=(20,10))


# ## 2. Age

# In[ ]:


explore_variable('age')


# In[ ]:


count_plot(col_name ='age', fig_size=(40,15))


# ## 3. hospital_admit_source

# In[ ]:


explore_variable('hospital_admit_source')


# In[ ]:


count_plot(col_name ='hospital_admit_source', fig_size=(30,12))


# ## 4. icu_admit_source

# In[ ]:


explore_variable('icu_admit_source')


# In[ ]:


count_plot(col_name ='icu_admit_source', fig_size=(30,12))


# ## 5. icu_stay_type

# In[ ]:


explore_variable('icu_stay_type')


# In[ ]:


count_plot(col_name ='icu_stay_type', fig_size=(30,10))


# ## 6. icu_type

# In[ ]:


explore_variable('icu_type')


# In[ ]:


count_plot(col_name ='icu_type', fig_size=(20,10))


# ## 7. apache_3j_bodysystem

# In[ ]:


explore_variable('apache_3j_bodysystem')


# In[ ]:


count_plot(col_name ='apache_3j_bodysystem', fig_size=(25,12))


# ## 8. apache_2_bodysystem

# In[ ]:


explore_variable('apache_2_bodysystem')


# In[ ]:


count_plot(col_name ='apache_2_bodysystem', fig_size=(25,12))


# # Key Points 
# 1. Here Undefined diagnoses and Undefined Diagnoses can be combined to one category 
# 2. Some categories can be grouped to create one category to avoid low sample count.

# # Label Encoding 

# In[ ]:


cat_cols =  train.select_dtypes(include='O').columns.values.tolist()
for col in cat_cols: 
    if col in train.columns: 
        le = LabelEncoder() 
        le.fit(list(train[col].astype(str).values) + list(test[col].astype(str).values)) 
        train[col] = le.transform(list(train[col].astype(str).values)) 
        test[col] = le.transform(list(test[col].astype(str).values)) 


# # Model building

# In[ ]:


y=train[target]
train=train.drop(target, axis=1)


# In[ ]:


# Parameters
params = {"objective": "binary", 
          "boosting": "gbdt",
          "metric": "auc",
          "n_jobs":-1,
          "verbose":-1}

num_folds = 10
roc_auc = list()
feature_importances = pd.DataFrame()
feature_importances['feature'] = train.columns
pred_on_test = np.zeros(test.shape[0])


kf = StratifiedKFold(n_splits=num_folds,shuffle=True, random_state=2020)
for index, (train_index, valid_index) in enumerate(kf.split(X=train,y=y)):
    print(f"FOLD {index+1}")

    X_train_fold, y_train_fold = train.iloc[train_index], y.iloc[train_index]
    X_valid_fold, y_valid_fold = train.iloc[valid_index], y.iloc[valid_index]

    dtrain = lightgbm.Dataset(X_train_fold, label=y_train_fold)
    dvalid = lightgbm.Dataset(X_valid_fold, label=y_valid_fold)

    lgb = lightgbm.train(params=params, train_set=dtrain, num_boost_round=2000, 
                         valid_sets=[dtrain, dvalid], verbose_eval=250, early_stopping_rounds=500)

    feature_importances[f'fold_{index + 1}'] = lgb.feature_importance()

    y_valid_pred = (lgb.predict(X_valid_fold,num_iteration=lgb.best_iteration))
    pred_on_test += (lgb.predict(test,num_iteration=lgb.best_iteration)) / num_folds

    # winsorization
    y_valid_pred = np.clip(a=y_valid_pred, a_min=0, a_max=1)
    pred_on_test = np.clip(a=pred_on_test, a_min=0, a_max=1)

    print(f"FOLD {index+1}: ROC_AUC  => {np.round(roc_auc_score(y_true=y_valid_fold, y_score=y_valid_pred),5)}")
    roc_auc.append(roc_auc_score(y_true=y_valid_fold, y_score=y_valid_pred)/num_folds)
    
print(f"Mean roc_auc for {num_folds} folds: {np.round(sum(roc_auc),5)}")


# # Feature Importance

# In[ ]:


def plot_feature_importance(df, k_fold_object):
    df['average_feature_imp'] = df[['fold_{}'.format(fold + 1) for fold in range(k_fold_object.n_splits)]].mean(axis=1)
    plt.figure(figsize=(10, 40))
    sns.barplot(data=df.sort_values(by='average_feature_imp', ascending=False), x='average_feature_imp', y='feature');
    plt.title('Feature importance over {} folds average'.format(k_fold_object.n_splits))
    plt.show()


# In[ ]:


plot_feature_importance(df=feature_importances, k_fold_object=kf)


# In[ ]:


solution_template.hospital_death = pred_on_test
solution_template.to_csv("Version_1.csv", index=0)

