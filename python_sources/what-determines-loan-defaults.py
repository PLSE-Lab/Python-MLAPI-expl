#!/usr/bin/env python
# coding: utf-8

# ## Exploring the importance of individual features in predicting loan defaults

# In[ ]:


# import libraries

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings('ignore')


# ## Load dataset

# In[ ]:


data = pd.read_csv("../input/loan.csv", low_memory=False)
pd.set_option('display.max_columns', len(data.columns))
data.head(3)


# ## Explore the data

# In[ ]:


data['loan_status'].value_counts()


# In[ ]:


group = data.groupby('loan_status').agg([np.count_nonzero])
grouped = group['id'].reset_index()
grouped


# In[ ]:


sns.set(style="whitegrid")

# Initialize the matplotlib figure
f, ax = plt.subplots(figsize=(12, 6))

sns.set_color_codes("muted")

ax = sns.barplot(x=grouped['loan_status'], y=grouped['count_nonzero'])
ax.set(xlabel = 'loan status', ylabel = 'count', title = 'loan status by occurrence')
ax.set_xticklabels(['charged off','current','default','nc: charged off','nc: fpaid','fpaid','grace period','issued','late <30ds', 'very late >30ds'], rotation=15)


# # Loans distribution

# In[ ]:



ax = sns.distplot(data['loan_amnt'], bins =10, kde=False, color="g", axlabel='loan amount')


# In[ ]:


group2 = data.groupby('grade').agg([np.median])
interest_rate = group2['int_rate'].reset_index()

ax = sns.barplot(x = 'grade', y = 'median', data=interest_rate)
ax.set(xlabel = 'grade', ylabel = 'median interest rate', title = 'median interest rate, by loan grade')


# In[ ]:


group = data.groupby('grade').agg([np.median])
loanamount = group['loan_amnt'].reset_index()

ax = sns.barplot(y = "median", x = 'grade', data=loanamount)
ax.set(xlabel = 'loan grade', ylabel = 'median loan amount', title = 'median loan amount, by loan grade')


# ## Clean data

# In[ ]:


data = data.fillna(0)

loans = pd.get_dummies(data['loan_status'])


# In[ ]:


features_raw = data[['term', 'grade', 'sub_grade', 'emp_title', 'emp_length', 'home_ownership', 'verification_status', 'pymnt_plan', 'title', 'zip_code', 'addr_state', 'application_type', 'verification_status_joint', 'initial_list_status', 'earliest_cr_line', 'last_pymnt_d', 'last_credit_pull_d', 'next_pymnt_d']]
features_raw.head()


# In[ ]:


from sklearn import preprocessing

le = preprocessing.LabelEncoder()

for col in features_raw.columns:
    features_raw[col] = le.fit_transform(data[col])

features_raw.head()


# In[ ]:


numerical = data[['id', 'member_id', 'loan_amnt', 'funded_amnt', 'funded_amnt_inv', 'int_rate', 'installment', 'annual_inc', 'dti', 'delinq_2yrs', 'inq_last_6mths', 'mths_since_last_record', 'open_acc', 'pub_rec', 'revol_bal', 'revol_util', 'total_acc', 'out_prncp', 'out_prncp_inv', 'total_pymnt', 'total_pymnt_inv', 'total_rec_prncp', 'total_rec_int', 'total_rec_late_fee', 'recoveries', 'collection_recovery_fee', 'last_pymnt_amnt', 'collections_12_mths_ex_med', 'mths_since_last_major_derog', 'policy_code', 'annual_inc_joint', 'dti_joint', 'acc_now_delinq', 'tot_coll_amt', 'tot_cur_bal', 'open_acc_6m', 'open_il_6m', 'open_il_12m', 'open_il_24m', 'mths_since_rcnt_il']]

minmax = preprocessing.MinMaxScaler()

for col in numerical.columns:
    numerical[col] = minmax.fit_transform(numerical[col])

numerical.head()


# In[ ]:


#dates=data[['issue_d', 'earliest_cr_line', 'last_pymnt_d', 'last_credit_pull_d', 'next_pymnt_d']]

#dates = pd.to_datetime(dates[['issue_d', 'earliest_cr_line', 'last_pymnt_d', 'last_credit_pull_d', 'next_pymnt_d']], format='%b-%Y', errors='ignore')

dates=pd.to_datetime(data['issue_d'], format='%b-%Y', errors='ignore')
dates.head()


# In[ ]:


data_new = pd.concat([features_raw, numerical, loans, dates], axis=1)


# ## Apply models & check accuracy of predictions

# In[ ]:


# separate X and y

y = data_new['Default']
X = data_new.drop('Default', axis=1)


# In[ ]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)

# Show the results of the split
print "Training set has {} samples.".format(X_train.shape[0])
print "Testing set has {} samples.".format(X_test.shape[0])


# In[ ]:


# TODO: Import two metrics from sklearn - fbeta_score and accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import fbeta_score, accuracy_score
from time import time

beta = 0.5

# TODO: Initialize the models
clf_A = RandomForestClassifier(random_state=101)
clf_B = AdaBoostClassifier(random_state=101)

clf_A.fit(X_train, y_train)
clf_B.fit(X_train, y_train)

pred_A_train = clf_A.predict(X_train)
pred_B_train = clf_B.predict(X_train)

pred_A_test = clf_A.predict(X_test)
pred_B_test = clf_B.predict(X_test)


# In[ ]:


from sklearn.metrics import classification_report

print(classification_report(y_test,pred_A_test))


# ## Confusion matrix: RandomForests

# In[ ]:


# generate confusion matrix
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test,pred_A_test)

df_cm = pd.DataFrame(cm, index = ['True (positive)', 'True (negative)'])
df_cm.columns = ['Predicted (positive)', 'Predicted (negative)']

sns.heatmap(df_cm, annot=True, fmt="d")


# ## Confusion matrix: AdaBoost

# In[ ]:


print(classification_report(y_test,pred_B_test))


# In[ ]:


cm2 = confusion_matrix(y_test,pred_B_test)

df_cm2 = pd.DataFrame(cm2, index = ['True (positive)', 'True (negative)'])
df_cm2.columns = ['Predicted (positive)', 'Predicted (negative)']

sns.heatmap(df_cm2, annot=True, fmt="d")


# ## Factor importances: RandomForests

# In[ ]:


# extract feature importances
import numpy as np

keyfeat_A = clf_A.feature_importances_
df = pd.DataFrame(keyfeat_A)
df.index = np.arange(1, len(df) + 1)

featurenames = data_new.columns
featurenames = pd.DataFrame(data_new.columns)
featurenames.drop(featurenames.head(1).index, inplace=True)

dfnew = pd.concat([featurenames, df], axis=1)
dfnew.columns = ['featurenames', 'weight']
dfsorted = dfnew.sort_values(['weight'], ascending=[False])
dfsorted.head()


# In[ ]:


# plot feature importances

sns.set(style="whitegrid")

# Initialize the matplotlib figure
f, ax = plt.subplots(figsize=(15, 10))

ax = sns.barplot(x=dfsorted['featurenames'].head(7), y=dfsorted['weight'].head(7))

ax.set(xlabel='feature names', ylabel='weight')

ax.set_title('Feature importances')

for item in ax.get_xticklabels():
    item.set_rotation(50)


# ## Feature importances: AdaBoost

# In[ ]:


# extract feature importances
import numpy as np

keyfeat_B = clf_B.feature_importances_
df2 = pd.DataFrame(keyfeat_B)
df2.index = np.arange(1, len(df) + 1)

featurenames_B = data_new.columns
featurenames_B = pd.DataFrame(data_new.columns)
featurenames_B.drop(featurenames_B.head(1).index, inplace=True)

df_B = pd.concat([featurenames, df2], axis=1)
df_B.columns = ['featurenames', 'weight']
dfsorted_B = df_B.sort_values(['weight'], ascending=[False])
dfsorted_B.head()


# In[ ]:


# plot feature importances

sns.set(style="whitegrid")
sns.set(font_scale=3)

# Initialize the matplotlib figure
f, ax = plt.subplots(figsize=(45, 10))

ax = sns.barplot(x=dfsorted_B['featurenames'].head(), y=dfsorted_B['weight'].head())

ax.set(xlabel='feature names', ylabel='weight')

ax.set_title('Feature importances')

for item in ax.get_xticklabels():
    item.set_rotation(50)


# In[ ]:




