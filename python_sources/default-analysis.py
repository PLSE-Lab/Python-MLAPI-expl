#!/usr/bin/env python
# coding: utf-8

# ## Imports

# In[ ]:


import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use('ggplot')
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score

from imblearn.over_sampling import SMOTE

import tensorflow as tf


# ## Load & Initial look

# In[ ]:


df = pd.read_csv('../input/loan_final313.csv')


# In[ ]:


df.head()


# In[ ]:


df.info()


# In[ ]:


plt.figure(figsize=(16,8))
sns.heatmap(df.corr(),
           cmap = 'coolwarm')


# ## Define Target Variables

# #### Categorical Default or not based on loan condition

# In[ ]:


def defaulted(x):
    if x == 'Good Loan':
        return 0
    else:
        return 1


# In[ ]:


df['default'] = df['loan_condition'].apply(lambda x: defaulted(x))


# ## Assess each variable individually

# #### id : Identification number for each individual
# 
#     Won't be used for prediction

# In[ ]:


df.drop('id', axis=1, inplace=True)


# #### year: Year the loan was issued
# 
#     Won't be used for prediction

# In[ ]:


df.drop('year', axis=1, inplace=True)


# #### issue_d: Issue Date
# 
#     Won't be used for prediction

# In[ ]:


df.drop('issue_d', axis=1, inplace=True)


# #### final_d: Final Date
# 
#     Won't be used for prediction

# In[ ]:


df.drop('final_d', axis=1, inplace=True)


# #### emp_length_int: Employment length in years
# 
#     Scale to 0 -> 1

# In[ ]:


scaler = MinMaxScaler()


# In[ ]:


df['emp_length_int'] = scaler.fit_transform(df['emp_length_int'].values.reshape(-1,1))


# #### home_ownership: Home Owndership status
#     
#     One-hot encode

# In[ ]:


plt.figure(figsize=(12,6))
sns.countplot(x='home_ownership',data=df, hue='default')


# In[ ]:


df = pd.concat([df, pd.get_dummies(df['home_ownership'])],axis=1).drop(['home_ownership', 'home_ownership_cat'],axis=1)


# In[ ]:


df.drop(['OTHER', 'NONE', 'ANY'],axis=1,inplace=True)


# #### income_category: Low, Medium, or High
# 
#     One-hot encode

# In[ ]:


df = pd.concat([df, pd.get_dummies(df['income_category'])],axis=1).drop(['income_category', 'income_cat'],axis=1)


# #### annual_inc
# 
#     Check for outliers
#     Scale to 0 -> 1

# In[ ]:


plt.figure(figsize=(12,6))
sns.boxplot(x=df['annual_inc'])


# In[ ]:


outliers = df[df['annual_inc'] > df['annual_inc'].quantile(0.99)].index


# In[ ]:


df.loc[outliers,'annual_inc'] = df['annual_inc'].quantile(0.99)


# In[ ]:


plt.figure(figsize=(12,6))
sns.boxplot(x=df['annual_inc'])


# In[ ]:


scaler = MinMaxScaler()
df['annual_inc'] = scaler.fit_transform(df['annual_inc'].values.reshape(-1,1))


# #### loan_amount
# 
#     Scale to 0 -> 1

# In[ ]:


plt.figure(figsize=(12,6))
sns.boxplot(x=df['loan_amount'])


# In[ ]:


scaler = MinMaxScaler()
df['loan_amount'] = scaler.fit_transform(df['loan_amount'].values.reshape(-1,1))


# #### term: Length of the loan
# 
#     One-hot encode

# In[ ]:


df['term'].unique()


# In[ ]:


plt.figure(figsize=(12,6))
sns.countplot(x='term',data=df, hue='default')


# In[ ]:


df = pd.concat([df, pd.get_dummies(df['term_cat'],prefix='term')],axis=1).drop(['term', 'term_cat'],axis=1)


# #### application_type: Individual or Joint
# 
#     Won't be included for prediction

# In[ ]:


plt.figure(figsize=(12,6))
sns.countplot(x='application_type_cat',data=df, hue='default')


# In[ ]:


df.drop(['application_type','application_type_cat'],axis=1,inplace=True)


# #### purpose: Reason for issuing loan
# 
#     One-hot encode

# In[ ]:


df['purpose'].unique()


# In[ ]:


plt.figure(figsize=(12,6))
sns.countplot(x='purpose',data=df, hue='default')


# In[ ]:


df = pd.concat([df, pd.get_dummies(df['purpose'])],axis=1).drop(['purpose', 'purpose_cat'],axis=1)


# In[ ]:


df.drop(['car', 'small_business', 'other', 'wedding', 'home_improvement', 'major_purchase',
       'medical', 'moving', 'vacation', 'house', 'renewable_energy',
       'educational'],axis=1, inplace=True)


# #### interest_payments: Low or High
# 
#     One-hot encode

# In[ ]:


plt.figure(figsize=(12,6))
sns.countplot(x='interest_payments',data=df, hue='default')


# In[ ]:


df = pd.concat([df, pd.get_dummies(df['interest_payments'],prefix='int')],axis=1).drop(['interest_payments', 'interest_payment_cat'],axis=1)


# In[ ]:


df.drop('int_High',axis=1,inplace=True)


# #### loan_condition: Good or bad
# 
#     Target variable
#     Has been converted to 'default', so drop

# In[ ]:


df.drop(['loan_condition', 'loan_condition_cat'],axis=1,inplace=True)


# #### interest_rate
# 
#     Scale 0 -> 1

# In[ ]:


plt.figure(figsize=(12,6))
plt.hist(df[df['default']==0]['interest_rate'],color='orange',alpha=0.5,label='Good')
plt.hist(df[df['default']==1]['interest_rate'],color='blue',alpha=0.5,label='Bad')

plt.legend()


# In[ ]:


plt.figure(figsize=(12,6))
sns.boxplot(x=df['interest_rate'])


# In[ ]:


outliers = df[df['interest_rate'] > df['interest_rate'].quantile(.99)].index


# In[ ]:


df.loc[outliers,'interest_rate'] = df['interest_rate'].quantile(.99)


# In[ ]:


scaler = MinMaxScaler()
df['interest_rate'] = scaler.fit_transform(df['interest_rate'].values.reshape(-1,1))


# #### grade: Loan grade
# 
#     Won't be included for prediction

# In[ ]:


df.info()


# In[ ]:


plt.figure(figsize=(12,6))
sns.countplot(x='grade',data=df, hue='default')


# In[ ]:


df.drop(['grade', 'grade_cat'],axis=1,inplace=True)


# #### dti: Ratio of monthly debt payments to annual income
# 
#     Scale to 0 -> 1

# In[ ]:


plt.figure(figsize=(12,6))
sns.boxplot(x=df['dti'])


# In[ ]:


outliers = df[df['dti'] > df['dti'].quantile(.99)].index
df.loc[outliers,'dti'] = df['dti'].quantile(.99)


# In[ ]:


scaler = MinMaxScaler()
df['dti'] = scaler.fit_transform(df['dti'].values.reshape(-1,1))


# #### total_pymnt
# 
#     Won't be used for prediction

# In[ ]:


df.drop('total_pymnt', axis=1, inplace=True)


# #### total_rec_prncp
# 
#     Won't be used for prediction

# In[ ]:


df.drop('total_rec_prncp', axis=1, inplace=True)


# #### recoveries
# 
#     Won't be used for prediction

# In[ ]:


df.drop('recoveries', axis=1, inplace=True)


# #### installment
# 
#     Won't be used for prediction

# In[ ]:


plt.figure(figsize=(12,6))
plt.hist(df[df['default']==0]['installment'],color='orange',alpha=0.5,label='Good')
plt.hist(df[df['default']==1]['installment'],color='blue',alpha=0.5,label='Bad')

plt.legend()


# In[ ]:


df.drop('installment', axis=1, inplace=True)


# #### region
# 
#     Won't be included for prediction

# In[ ]:


df.drop('region', axis=1, inplace=True)


# ## Train Test Split

# In[ ]:


sum(df['default']) / len(df)


# In[ ]:


X = df.drop(['default'],axis=1)
y = df['default']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100)


# ## Baseline Model Performance

# In[ ]:


rf_base = RandomForestClassifier(n_estimators=100)
ada_base = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(),n_estimators=100)


# In[ ]:


rf_base.fit(X_train,y_train)
ada_base.fit(X_train,y_train)


# In[ ]:


rf_base_pred = rf_base.predict(X_test)
ada_base_pred = ada_base.predict(X_test)


# In[ ]:


print(confusion_matrix(y_test, rf_base_pred))
print('\n')
print(confusion_matrix(y_test, ada_base_pred))


# In[ ]:


print(accuracy_score(y_test,rf_base_pred))
print(accuracy_score(y_test,ada_base_pred))


# In[ ]:


print(recall_score(y_test,rf_base_pred))
print(recall_score(y_test,ada_base_pred))


# ## Undersampling Models

# In[ ]:


ratios = np.array([0.2,0.4,0.6,0.8,1.0])
rf_acc_us = []
ada_acc_us = []
rf_recall_us = []
ada_recall_us = []
default_count = y_train.value_counts()[1]
X_train.reset_index(drop=True, inplace=True)
y_train.reset_index(drop=True, inplace=True)
bad_indices = y_train[y_train==1].index


# In[ ]:


for i in ratios:
    majority_count = int(np.floor(default_count / i))
    good_indices = np.random.choice(y_train[y_train==0].index,size=majority_count)
    indices = np.concatenate((bad_indices,good_indices))
    X_train_us = X_train.iloc[indices]
    y_train_us = y_train.iloc[indices]
    
    rf_ = RandomForestClassifier(n_estimators=100)
    ada_ = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(),n_estimators=100)
    
    rf_.fit(X_train_us,y_train_us)
    ada_.fit(X_train_us,y_train_us)
    
    rf_pred = rf_.predict(X_test)
    ada_pred = ada_.predict(X_test)
    
    rf_acc_us.append(accuracy_score(y_test,rf_pred))
    ada_acc_us.append(accuracy_score(y_test,ada_pred))
    
    rf_recall_us.append(recall_score(y_test,rf_pred))
    ada_recall_us.append(recall_score(y_test,ada_pred))
    


# In[ ]:


ada_acc_us


# In[ ]:


rf_acc_us


# In[ ]:


ada_recall_us


# In[ ]:


rf_recall_us


# In[ ]:


plt.figure(figsize=(12,6))
plt.plot(ratios,ada_acc_us,linestyle='-',marker='o',color='red',label='ada_acc')
plt.plot(ratios,ada_recall_us,linestyle='-',marker='o',color='red',label='ada_rec',alpha=0.5)
plt.plot(ratios,rf_acc_us,linestyle='-',marker='o',color='blue',label='rf_acc')
plt.plot(ratios,rf_recall_us,linestyle='-',marker='o',color='blue',label='rf_rec', alpha=0.5)

plt.legend()


# ## Oversampling Models

# In[ ]:


ratios = np.array([0.2,0.4,0.6,0.8,1.0])
rf_acc_os = []
ada_acc_os = []
rf_recall_os = []
ada_recall_os = []
X_train.reset_index(drop=True, inplace=True)
y_train.reset_index(drop=True, inplace=True)


# In[ ]:


for j in ratios:
    sm = SMOTE(random_state=101, ratio = j)
    X_train_os, y_train_os = sm.fit_sample(X_train, y_train)
    
    X_train_os[:,5:] = np.round(X_train_os[:,5:])
    
    rf_ = RandomForestClassifier(n_estimators=100)
    ada_ = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(),n_estimators=100)
    
    rf_.fit(X_train_os,y_train_os)
    ada_.fit(X_train_os,y_train_os)
    
    rf_pred = rf_.predict(X_test)
    ada_pred = ada_.predict(X_test)
    
    rf_acc_os.append(accuracy_score(y_test,rf_pred))
    ada_acc_os.append(accuracy_score(y_test,ada_pred))
    
    rf_recall_os.append(recall_score(y_test,rf_pred))
    ada_recall_os.append(recall_score(y_test,ada_pred))


# In[ ]:


rf_acc_os


# In[ ]:


ada_acc_os


# In[ ]:


rf_recall_os


# In[ ]:


ada_recall_os


# In[ ]:


plt.figure(figsize=(12,6))
plt.plot(ratios,ada_acc_os,linestyle='-',marker='o',color='red',label='ada_acc')
plt.plot(ratios,ada_recall_os,linestyle='-',marker='o',color='red',label='ada_rec',alpha=0.5)
plt.plot(ratios,rf_acc_os,linestyle='-',marker='o',color='blue',label='rf_acc')
plt.plot(ratios,rf_recall_os,linestyle='-',marker='o',color='blue',label='rf_rec', alpha=0.5)

plt.legend()


# ## Can a neural net do better?

# In[ ]:


default_count = y_train.value_counts()[1]
X_train.reset_index(drop=True, inplace=True)
y_train.reset_index(drop=True, inplace=True)
bad_indices = y_train[y_train==1].index
majority_count = int(np.floor(default_count / 0.6))
good_indices = np.random.choice(y_train[y_train==0].index,size=majority_count) 
indices = np.concatenate((bad_indices,good_indices))
X_train_tf = X_train.iloc[indices]
y_train_tf = y_train.iloc[indices]


# In[ ]:


emp_length = tf.feature_column.numeric_column('emp_length_int')
ann_inc = tf.feature_column.numeric_column('annual_inc')
loan_amt = tf.feature_column.numeric_column('loan_amount')
int_rate = tf.feature_column.numeric_column('interest_rate')
dti = tf.feature_column.numeric_column('dti')


# In[ ]:


mortgage = tf.feature_column.numeric_column('MORTGAGE')
own = tf.feature_column.numeric_column('OWN')
rent = tf.feature_column.numeric_column('RENT')
high = tf.feature_column.numeric_column('High')
low = tf.feature_column.numeric_column('Low')
medium = tf.feature_column.numeric_column('Medium')
short = tf.feature_column.numeric_column('term_1')
long = tf.feature_column.numeric_column('term_2')
credit = tf.feature_column.numeric_column('credit_card')
debt = tf.feature_column.numeric_column('debt_consolidation')
low_int = tf.feature_column.numeric_column('int_Low')


# In[ ]:


feat_cols = [emp_length,ann_inc,loan_amt,int_rate,dti,mortgage,own,rent,high,low,medium,short,long,credit,debt,low_int]


# In[ ]:


input_func = tf.estimator.inputs.pandas_input_fn(X_train_tf,y_train_tf,
                                                batch_size=10000,
                                                num_epochs=1000,
                                                shuffle=True)


# In[ ]:


dnn_model = tf.estimator.DNNClassifier(hidden_units=[10,10,10,10,10],
                                      feature_columns=feat_cols,
                                      n_classes=2)


# In[ ]:


dnn_model.train(input_fn=input_func,steps=5000)


# In[ ]:


eval_input_func = tf.estimator.inputs.pandas_input_fn(x=X_test,
                                                     y=y_test,
                                                     batch_size=10000,
                                                     num_epochs=1,
                                                     shuffle=False)


# In[ ]:


dnn_model.evaluate(eval_input_func)


# In[ ]:




