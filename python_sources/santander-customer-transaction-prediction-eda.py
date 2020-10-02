#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# ## 1) Import packages

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


# ## 2) Load the dataset

# In[ ]:


df_train = pd.read_csv("/kaggle/input/santander-customer-transaction-prediction-dataset/train.csv")
df_test = pd.read_csv("/kaggle/input/santander-customer-transaction-prediction-dataset/test.csv")


# In[ ]:


#train data
df_train.head()


# In[ ]:


#test data
df_test.head()


# In[ ]:


df_train.shape,df_test.shape


# In[ ]:


sample_submission = pd.read_csv("/kaggle/input/santander-customer-transaction-prediction/sample_submission.csv")


# In[ ]:


sample_submission.head()


# In[ ]:


sample_submission.shape


# ### Checking Missing values

# In[ ]:


# train data
df_train.isnull().sum().sum()


# In[ ]:


# test data
df_test.isnull().sum().sum()


# #### Hence, there are no missing values in train as well as test data.

# In[ ]:


df_train.info()


# ### Numerical values in train and test data

# In[ ]:


df_train.describe()


# In[ ]:


df_test.describe()


# ## 3) EDA

# ### Target Variable 

# In[ ]:


df_train.target.value_counts(normalize=True)


# In[ ]:


f,ax=plt.subplots(1,2, figsize=(12,4))
df_train.target.value_counts().plot.pie(explode=[0,0.12],autopct='%1.3f%%',ax=ax[0])
sns.countplot('target',data=df_train)
plt.show()


# #### So from above graphs, we can conclude that the number of customers that will not make a transaction is much higher than those that will.

# #### Hence the data is imbalanced with respect to target variable.

# ### Statistical Analysis

# ### Distribution of Mean 

# #### i) Distribution of mean values per row in the train and test dataset

# In[ ]:


feat = df_train.columns.values[2:202]
feat


# In[ ]:


plt.figure(figsize=(15,5))
sns.distplot(df_train[feat].mean(axis=1),color="green", kde=True,bins=100, label='train')
sns.distplot(df_test[feat].mean(axis=1),color="red", kde=True,bins=100, label='test')
plt.title("Distribution of mean values per row in the train and test dataset")
plt.legend()
plt.show()


# #### Thus, we can see that distribution of mean values per row is of Standard Normal Distribution.

# ### ii) Distribution of the mean values per columns in the train and test set.

# In[ ]:


plt.figure(figsize=(15,5))
plt.title("Distribution of mean values per column in the train and test set")
sns.distplot(df_train[feat].mean(axis=0),color="blue",kde=True,bins=100, label='train')
sns.distplot(df_test[feat].mean(axis=0),color="red", kde=True,bins=100, label='test')
plt.legend()
plt.show()


# Thus, we can see that distribution of mean values per column is normally distributed.

# ### Distribution of Standard Deviation

# #### i) Distribution of Standard Deviation values per row in the train and test dataset

# In[ ]:


plt.figure(figsize=(15,5))
sns.distplot(df_train[feat].std(axis=1),color="green", kde=True,bins=100, label='train')
sns.distplot(df_test[feat].std(axis=1),color="red", kde=True,bins=100, label='test')
plt.title("Distribution of Standard Deviation values per row in the train and test dataset")
plt.legend()
plt.show()


# #### Thus, we can see that distribution of Standard Deviation values per row is of Standard Normal Distribution.

# ### ii) Distribution of the Standard Deviation values per columns in the train and test set.

# In[ ]:


plt.figure(figsize=(15,5))
sns.distplot(df_train[feat].std(axis=0),color="blue", kde=True,bins=100, label='train')
sns.distplot(df_test[feat].std(axis=0),color="red", kde=True,bins=100, label='test')
plt.title("Distribution of Standard Deviation values per columns in the train and test dataset")
plt.legend()
plt.show()


# #### Thus, we can see that distribution of Standard Deviation values per column is Positively Skewed.

# ### Distribution of feature grouped by Target variable

# ### i) Distribution of mean values per row in the train set grouped by Target

# In[ ]:


# mean values per row in the train set grouped by Target = 0
df_train.loc[df_train.target == 0][feat].mean(axis=1)


# In[ ]:


# mean values per row in the train set grouped by Target = 1
df_train.loc[df_train.target == 1][feat].mean(axis=1)


# In[ ]:


plt.figure(figsize=(15,5))
sns.distplot(df_train.loc[df_train.target == 0][feat].mean(axis=1),color="red", kde=True,bins=100,label='target = 0')
sns.distplot(df_train.loc[df_train.target == 1][feat].mean(axis=1),color="blue", kde=True,bins=100,label='target = 1')
plt.title("Distribution of mean values per row in the train set grouped by Target")
plt.legend()
plt.show()


# ### ii) Distribution of mean values per column in the train set grouped by Target

# In[ ]:


# mean values per column in the train set grouped by Target = 0
df_train.loc[df_train.target == 0][feat].mean()


# In[ ]:


# mean values per column in the train set grouped by Target = 1
df_train.loc[df_train.target == 1][feat].mean()


# In[ ]:


plt.figure(figsize=(15,5))
sns.distplot(df_train.loc[df_train.target == 0][feat].mean(),color="red", kde=True,bins=100,label='target = 0')
sns.distplot(df_train.loc[df_train.target == 1][feat].mean(),color="green", kde=True,bins=100,label='target = 1')
plt.title("Distribution of mean values per column in the train set grouped by Target")
plt.legend()
plt.show()


# ## Correlation

# ### i) Correlation between features in train dataset

# In[ ]:


df_train.corr()


# In[ ]:


train_cor = df_train.drop(["target"], axis=1).corr()
train_cor = train_cor.values.flatten()
train_cor = train_cor[train_cor != 1]
plt.figure(figsize=(15,5))
sns.distplot(train_cor)
plt.xlabel("Correlation values found in train excluding 1")
plt.ylabel("Density")
plt.title("Correlation between features")
plt.show()


# #### Thus, we can conclude that there is no correlation in train dataset.

# ### ii) Correlation between features in test dataset

# In[ ]:


df_test.corr()


# In[ ]:


test_cor = df_test.corr()
test_cor = test_cor.values.flatten()
test_cor = test_cor[test_cor != 1]
plt.figure(figsize=(15,5))
sns.distplot(test_cor)
plt.xlabel("Correlation values found in test excluding 1")
plt.ylabel("Density")
plt.title("Correlation between features")
plt.show()


# #### Thus, we can conclude that there is no correlation in test dataset.

# ### Import necessary packages

# In[ ]:


import lightgbm as lgb
from sklearn.metrics import f1_score, classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold


# ## 4.) Model Building

# In[ ]:


X= df_train.drop(["target","ID_code"],axis=1)
X.head()


# In[ ]:


Y=df_train.target
Y.head()


# In[ ]:


X_test  = df_test.drop("ID_code",axis=1)
X_test.head()


# ### Light GBM model

# In[ ]:


params = {'objective' : "binary", 
               'boost':"gbdt",
               'metric':"auc",
               'boost_from_average':"false",
               'num_threads':8,
               'learning_rate' : 0.01,
               'num_leaves' : 13,
               'max_depth':-1,
               'tree_learner' : "serial",
               'feature_fraction' : 0.05,
               'bagging_freq' : 5,
               'bagging_fraction' : 0.4,
               'min_data_in_leaf' : 80,
               'min_sum_hessian_in_leaf' : 10.0,
               'verbosity' : 1}


# In[ ]:


num_folds = 10
folds = StratifiedKFold(n_splits=num_folds, shuffle=False, random_state=42)
oof = np.zeros(len(df_train))
y_pred_lgbm = np.zeros(len(df_train.target))

print('Light GBM Model')
for fold_, (trn_idx, val_idx) in enumerate(folds.split(df_train.values,df_train.target.values)):
    print('Fold', fold_)
    
    X_train, Y_train = df_train.iloc[trn_idx][feat], df_train.target.iloc[trn_idx]
    X_valid, Y_valid = df_train.iloc[val_idx][feat], df_train.target.iloc[val_idx]
    
    train_data = lgb.Dataset(X_train, label=Y_train)
    valid_data = lgb.Dataset(X_valid, label=Y_valid)
    
    lgb_model = lgb.train(params, train_data, 1000000, valid_sets = [train_data, valid_data], verbose_eval=5000, early_stopping_rounds = 4000)
    oof[val_idx] = lgb_model.predict(df_train.iloc[val_idx][feat], num_iteration=lgb_model.best_iteration)
    y_pred_lgbm += lgb_model.predict(df_test[feat], num_iteration=lgb_model.best_iteration) / folds.n_splits

print("CV score: {:<8.5f}".format(roc_auc_score(df_train.target, oof)))


# In[ ]:


y_pred_lgbm


# In[ ]:





# ### RandomForest Classifier Model

# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
train_X, val_X, train_y, val_y = train_test_split(X, Y, random_state=42)
rfc_mod = RandomForestClassifier(random_state=42).fit(train_X, train_y)


# In[ ]:


y_pred_rfc = rfc_mod.predict(X_test)


# In[ ]:


y_pred_rfc


# In[ ]:


print("F1 Score :",f1_score(y_pred_rfc,Y,average = "weighted"))
print('Report:\n',classification_report(Y, y_pred_rfc))
print('Confusion Matrix: \n',confusion_matrix(Y, y_pred_rfc))


# ### DecisionTree Classifier Model

# In[ ]:


from sklearn.tree import DecisionTreeClassifier
train_X, val_X, train_y, val_y = train_test_split(X, Y, random_state=42)
dectre_mod = DecisionTreeClassifier(random_state=42, max_depth=5, min_samples_split=5).fit(train_X, train_y)


# In[ ]:


y_pred_dec_tree = dectre_mod.predict(X_test)


# In[ ]:


y_pred_dec_tree


# In[ ]:


print("F1 Score :",f1_score(y_pred_dec_tree,Y,average = "weighted"))
print('Report:\n',classification_report(Y, y_pred_dec_tree))
print('Confusion Matrix: \n',confusion_matrix(Y, y_pred_dec_tree))


# ### LogisticRegression Model

# In[ ]:


from sklearn.linear_model import LogisticRegression
train_X, val_X, train_y, val_y = train_test_split(X,Y,random_state=42)
logreg_mod = LogisticRegression(random_state=42).fit(train_X, train_y)


# In[ ]:


y_pred_logreg = logreg_mod.predict(X_test)


# In[ ]:


y_pred_logreg


# In[ ]:


print("F1 Score :",f1_score(y_pred_logreg,Y,average = "weighted"))
print('Report:\n',classification_report(Y, y_pred_logreg))
print('Confusion Matrix: \n',confusion_matrix(Y, y_pred_logreg))


# ## 5.) Submission File

# In[ ]:


submission_lgbm = pd.DataFrame({
        "ID_code": df_test["ID_code"],
        "target": y_pred_lgbm
    })
submission_lgbm.to_csv('submission_lgbm.csv', index=False)


# In[ ]:


submission_rfc = pd.DataFrame({
        "ID_code": df_test["ID_code"],
        "target": y_pred_rfc
    })
submission_rfc.to_csv('submission_rfc.csv', index=False)


# In[ ]:


submission_dec_tree = pd.DataFrame({
        "ID_code": df_test["ID_code"],
        "target": y_pred_dec_tree
    })
submission_dec_tree.to_csv('submission_dec_tree.csv', index=False)


# In[ ]:


submission_logreg = pd.DataFrame({
        "ID_code": df_test["ID_code"],
        "target": y_pred_logreg
    })
submission_logreg.to_csv('submission_logreg.csv', index=False)


# In[ ]:




