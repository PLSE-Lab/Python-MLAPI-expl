#!/usr/bin/env python
# coding: utf-8

# # Introduction: Home Credit Default Risk
# This notebook is intended for those who are new to machine learning competitions or want a gentle introduction to the problem. I purposely avoid jumping into complicated models or joining together lots of data in order to show the basics of how to get started in machine learning! Any comments or suggestions are much appreciated.
# 
# In this notebook, we will take an initial look at the Home Credit default risk machine learning competition currently hosted on Kaggle. The objective of this competition is to use historical loan application data to predict whether or not an applicant will be able to repay a loan. This is a standard supervised classification task:
# 
# **Supervised:** The labels are included in the training data and the goal is to train a model to learn to predict the labels from the features
# 
# **Classification:** The label is a binary variable, 0 (will repay loan on time), 1 (will have difficulty repaying loan)

# # Data
# The data is provided by Home Credit, a service dedicated to provided lines of credit (loans) to the unbanked population. Predicting whether or not a client will repay a loan or have difficulty is a critical business need, and Home Credit is hosting this competition on Kaggle to see what sort of models the machine learning community can develop to help them in this task.
# 
# **There are 7 different sources of data:**
# 
# 1. ** application_train/application_test:** the main training and testing data with information about each loan application at Home Credit. Every loan has its own row and is identified by the feature SK_ID_CURR. The training application data comes with the TARGET indicating 0: the loan was repaid or 1: the loan was not repaid.
# 2.  **bureau:** data concerning client's previous credits from other financial institutions. Each previous credit has its own row in bureau, but one loan in the application data can have multiple previous credits.
# 3.** bureau_balance:** monthly data about the previous credits in bureau. Each row is one month of a previous credit, and a single previous credit can have multiple rows, one for each month of the credit length.
# 4. **previous_application:** previous applications for loans at Home Credit of clients who have loans in the application data. Each current loan in the application data can have multiple previous loans. Each previous application has one row and is identified by the feature SK_ID_PREV.
# 5. **POS_CASH_BALANCE:** monthly data about previous point of sale or cash loans clients have had with Home Credit. Each row is one month of a previous point of sale or cash loan, and a single previous loan can have many rows.
# 6.** credit_card_balance:** monthly data about previous credit cards clients have had with Home Credit. Each row is one month of a credit card balance, and a single credit card can have many rows.
# 7.** installments_payment:** payment history for previous loans at Home Credit. There is one row for every made payment and one row for every missed payment.

# # Imports Library 
# Most useful library for Data Science: numpy, pandas, matplotlib.

# In[1]:


import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from catboost import CatBoostClassifier
import xgboost as xgb
from sklearn.cross_validation import KFold
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_absolute_error
import gc


# # Load & Read DataSet

# In[2]:


PATH="../input"

application_train = pd.read_csv(PATH+"/application_train.csv")
application_test = pd.read_csv(PATH+"/application_test.csv")
bureau = pd.read_csv(PATH+"/bureau.csv")
bureau_balance = pd.read_csv(PATH+"/bureau_balance.csv")
credit_card_balance = pd.read_csv(PATH+"/credit_card_balance.csv")
installments_payments = pd.read_csv(PATH+"/installments_payments.csv")
previous_application = pd.read_csv(PATH+"/previous_application.csv")
POS_CASH_balance = pd.read_csv(PATH+"/POS_CASH_balance.csv")


# In[3]:


application_train.head()


# In[4]:


application_test.head()


# In[5]:


bureau.head()


# In[6]:


bureau_balance.head()


# In[7]:


credit_card_balance.head()


# In[8]:


installments_payments.head()


# In[9]:


previous_application.head()


# # Lets check the missing values in each file.

# In[10]:


def missing_data(data):
    total = data.isnull().sum().sort_values(ascending = False)
    percent = (data.isnull().sum()/data.isnull().count()*100).sort_values(ascending = False)
    return pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])


# In[11]:


missing_data(application_train).head(10)


# In[12]:


missing_data(application_test).head(10)


# In[13]:


missing_data(bureau)


# In[14]:


missing_data(bureau_balance)


# In[15]:


missing_data(credit_card_balance)


# In[16]:


missing_data(installments_payments)


#  # Explore the data

# ## 1.Categorical features

# In[18]:


import seaborn as sns


# In[19]:


def plot_categorical(data, col, size=[8 ,4], xlabel_angle=0, title=''):
    '''use this for ploting the count of categorical features'''
    plotdata = data[col].value_counts()
    plt.figure(figsize = size)
    sns.barplot(x = plotdata.index, y=plotdata.values)
    plt.title(title)
    if xlabel_angle!=0: 
        plt.xticks(rotation=xlabel_angle)
    plt.show()
plot_categorical(data=application_train, col='TARGET', size=[8 ,4], xlabel_angle=0, title='train set: label')


# In[20]:


plot_categorical(data=application_train, col='OCCUPATION_TYPE', size=[12 ,4], xlabel_angle=30, title='Occupation Type')


# In[21]:


plot_categorical(data=application_train, col='NAME_INCOME_TYPE', size=[12 ,4], xlabel_angle=0, title='Income Type')


# In[22]:


plot_categorical(data=application_train, col='NAME_HOUSING_TYPE', size=[12 ,4], xlabel_angle=0, title='House Type')


# ## 2.Numerical features

# In[23]:


def plot_numerical(data, col, size=[8, 4], bins=50):
    '''use this for ploting the distribution of numercial features'''
    plt.figure(figsize=size)
    plt.title("Distribution of %s" % col)
    sns.distplot(data[col].dropna(), kde=True,bins=bins)
    plt.show()
plot_numerical(application_train, 'AMT_CREDIT')


# In[24]:


plot_numerical(application_train, 'AMT_ANNUITY')


# In[25]:


plot_numerical(application_train, 'DAYS_EMPLOYED')


# ## 3.Categorical features by label

# In[26]:


def plot_categorical_bylabel(data, col, size=[12 ,6], xlabel_angle=0, title=''):
    '''use it to compare the distribution between label 1 and label 0'''
    plt.figure(figsize = size)
    l1 = data.loc[data.TARGET==1, col].value_counts()
    l0 = data.loc[data.TARGET==0, col].value_counts()
    plt.subplot(1,2,1)
    sns.barplot(x = l1.index, y=l1.values)
    plt.title('Default: '+title)
    plt.xticks(rotation=xlabel_angle)
    plt.subplot(1,2,2)
    sns.barplot(x = l0.index, y=l0.values)
    plt.title('Non-default: '+title)
    plt.xticks(rotation=xlabel_angle)
    plt.show()
plot_categorical_bylabel(application_train, 'CODE_GENDER', title='Gender')


# In[27]:


plot_categorical_bylabel(application_train, 'NAME_EDUCATION_TYPE', size=[15 ,6], xlabel_angle=15, title='Education Type')


#  ## 4.Numerical features by label

# In[28]:


def plot_numerical_bylabel(data, col, size=[8, 4], bins=50):
    '''use this to compare the distribution of numercial features'''
    plt.figure(figsize=[12, 6])
    l1 = data.loc[data.TARGET==1, col]
    l0 = data.loc[data.TARGET==0, col]
    plt.subplot(1,2,1)
    sns.distplot(l1.dropna(), kde=True,bins=bins)
    plt.title('Default: Distribution of %s' % col)
    plt.subplot(1,2,2)
    sns.distplot(l0.dropna(), kde=True,bins=bins)
    plt.title('Non-default: Distribution of %s' % col)
    plt.show()
plot_numerical_bylabel(application_train, 'EXT_SOURCE_1', bins=50)


# In[29]:


plot_numerical_bylabel(application_train, 'EXT_SOURCE_2', bins=50)


# In[30]:


plot_numerical_bylabel(application_train, 'EXT_SOURCE_3', bins=50)


#  ## 5.Correlation Matrix

# In[31]:


corr_mat = application_train.corr()
plt.figure(figsize=[15, 15])
sns.heatmap(corr_mat.values, annot=False)
plt.show()


# # Simple LightGBM

# ## 1.Solve imbalance problem 

# In[32]:


application_train = pd.read_csv('../input/application_train.csv')
application_test= pd.read_csv('../input/application_test.csv')


# In[33]:


# get positive sample
n_pos = application_train[application_train.TARGET==1].shape[0]
pos_data = application_train[application_train.TARGET==1]
# get negative sample, and select a subset
n_neg = application_train[application_train.TARGET==0].shape[0]
neg_data = application_train[application_train.TARGET==0].iloc[np.random.randint(1, n_neg, n_pos), :]
# combine them
application_train = pd.concat([pos_data, neg_data], axis=0)
del pos_data, neg_data
gc.collect()


# In[34]:


application_train.shape


# ## 2.Feature

# In[35]:


# use this if you want to convert categorical features to dummies(default)
def cat_to_dummy(train, test):
    train_d = pd.get_dummies(train, drop_first=True)
    test_d = pd.get_dummies(test, drop_first=True)
    # make sure that the number of features in train and test should be same
    for i in train_d.columns:
        if i not in test_d.columns:
            if i!='TARGET':
                train_d = train_d.drop(i, axis=1)
    for j in test_d.columns:
        if j not in train_d.columns:
            if j!='TARGET':
                test_d = test_d.drop(i, axis=1)
    return train_d, test_d
application_train, application_test = cat_to_dummy(application_train, application_test)


# In[38]:


# use this if you want to convert categorical features to numerical ones
def cat_to_num(data):
    '''convert categorical features to numerical features'''
     #find categorical feature list
    cate_feature = [f for f in data.columns if data[f].dtype == 'object']
     #factorize all categorical features
    for feature in cate_feature:data[feature], b = pd.factorize(data[feature])
    return data


# ## 3.Model fitting: only include application features

# In[39]:


from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import cross_val_score
from sklearn.utils import shuffle


# In[40]:


X = application_train.drop(['SK_ID_CURR', 'TARGET'], axis=1)
y = application_train.TARGET
del application_train
gc.collect()


# In[41]:


X, y = shuffle(X, y)
lgbc = LGBMClassifier(n_estimators=720, learning_rate=0.01, num_leaves=6,colsample_bytree=.3,subsample=.8, min_split_gain=.01,
                     silent=-1, verbose=-1)
lgbc.fit(X, y)


# In[42]:


feature_imp=pd.DataFrame({'feature name':X.columns,'feature importance':lgbc.feature_importances_}).sort_values('feature importance', ascending=False).iloc[:, [1,0]]
feature_imp.head()


# ## 4.Feature importance

# In[43]:


n_show=20
plt.figure(figsize = [10, n_show/3])
ax = sns.barplot(x = 'feature importance', y='feature name', data=feature_imp.iloc[:n_show, :], label='Feature Importance')
ax.set_xlabel('feature name')
plt.show()


# ## 5.Prediction

# In[44]:


X_test = application_test.drop(['SK_ID_CURR'], axis=1)
y_pred = lgbc.predict_proba(X_test)[:, 1]


# In[45]:


output = pd.DataFrame({'SK_ID_CURR':application_test.SK_ID_CURR, 'TARGET': y_pred})
output.to_csv('application_train.csv', index=False)

