#!/usr/bin/env python
# coding: utf-8

# # Ensemble of RFs - further analysis

# ## Model Summary

# In[ ]:


import pandas as pd
import numpy as np

summary = pd.DataFrame({'Model': ['RF', 'RF', 'RF', 'Ensemble', 'Ensemble', 'Ensemble', 'RF', 'RF', 'RF', 'RF', 'Ensemble',
                                 'Ensemble', 'Ensemble'],
                        'Resolution Status': ['Open', 'Open', 'Open', 'Open', 'Open', 'Open', 'All', 'All', 'All', 'All',
                                             'All', 'All', 'All'],
                        'Recovered from Insurer': ['=0', '!=0', '!=0 & nooutliers', '=0 & !=0', '=0 & (!=0 & nooutliers)',
                                                  'all RFs', 'all data', '=0', '!=0', '!=0 & nooutliers','=0 & !=0', 
                                                   '=0 & (!=0 & nooutliers)','all RFs' ],
          'Train_Accuracy': [.8399, 0.98217, 0.98057, 0.86248, 0.85091, 0.86116, 0.8681, 0.84334, 0.60847, 0.9843, 0.86581,
                            0.86709, 0.87985],
          'Validation_Accuracy': [.54051, 0.63936, 0.64259, 0.55766, 0.85076, 0.85957, 0.56724, 0.54831, 0.61998, 0.63736,
                                 0.56124, 0.55739, 0.56574],
          "Train f1-score ['poor', 'average', 'outstanding']": ["[0.7771  0.78983 0.88811]", "[0.97278 0.96527 0.98927]",
                                                               "[0.96941 0.96739 0.98753]", "[0.8166  0.80315 0.9061 ]",
                                                               "[0.79915 0.78934 0.89836]","[0.80431 0.80663 0.90558]",
                                                               "[0.81022 0.81777 0.91077]","[0.78409 0.78979 0.89073]",
                                                               "[0.27404 0.28235 0.77278]","[0.97572 0.97207 0.99008]",
                                                               "[0.8164  0.81175 0.90917]","[0.81563 0.81283 0.91121]",
                                                               "[0.82805 0.83383 0.91856]"],
          "Validation f1-score ['poor', 'average', 'outstanding']": ["0.34549 0.38779 0.68984]", "[0.33129 0.38009 0.78421]",
                                                                    "[0.40571 0.29557 0.78446]", "[0.39659 0.39121 0.69972]",
                                                                    "[0.79849 0.79979 0.89567]", "[0.79421 0.81972 0.90309]",
                                                                    "[0.35744 0.39622 0.71745]","[0.37799 0.39467 0.68947]",
                                                                    "[0.29582 0.25926 0.78027]","[0.39548 0.26804 0.78165]",
                                                                    "[0.39371 0.3921  0.71122]","[0.39647 0.39162 0.70271]",
                                                                    "[0.35685 0.39475 0.71554]"]}
                      )
summary.index = np.arange(1, len(summary)+1)
summary.style.set_properties(subset = ['Recovered from Insurer'], **{'width': '130px'})


# #### RFs are working relatively better on train part of segragated data sets
# #### On an overall perspective the Ensemble model combining 3 RFs on data for Resolution Status = Open is giving the best generalization (Train acc = 0.86116 & Val acc = 0.85957)

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# #### Reading and Preview of the data

# In[ ]:


df_cd = pd.read_csv("../input/Train_Complaints-1564659763354.csv")
df_ind = pd.read_csv("../input/Train-1564659747836.csv")


# In[ ]:


# Combining the dataframes to include the target variable 'DRC' based on key column - 'InsurerID'
df = pd.merge(df_cd, df_ind, how='inner', on = 'InsurerID')


# In[ ]:


df.head()


# In[ ]:


# Converting DateOfRegistration & DateOfResolution to Date
df['DateOfRegistration'] =pd.to_datetime(df.DateOfRegistration, format = '%d-%m-%Y')
df['DateOfResolution'] = pd.to_datetime(df.DateOfResolution, format = '%d-%m-%Y')
df.dtypes


# #### Checking for NAs

# In[ ]:


# Checking column wise NAs

NA_col = pd.DataFrame(df.isna().sum(), columns = ['NA_Count'])
NA_col['%_of_NA'] = (NA_col.NA_Count/len(df))*100
NA_col.sort_values(by = ['%_of_NA'], ascending = False, na_position = 'first').head(7)

# Considering threshold of 20% for columnwise NAs, SubCoverage column will be dropped in further analysis


# In[ ]:


# Checking row wise NAs

NA_row = pd.DataFrame(df.isna().sum(axis=1), columns = ['NA_rw_count'])
NA_row['%_of_rw_NA'] = (NA_row.NA_rw_count/len(df))*100
NA_row.sort_values(by = ['%_of_rw_NA'], ascending = False, na_position = 'first').head(7)

# We are good in terms of rowwise NAs w.r.t. no need of row removals


# #### Checking unique values in columns

# In[ ]:


for col in (df):
    print(col)
    print(len(df[col].unique()))
    
# Column State will be dropped as it just has 1 unique value
# ComplaintID column will also be dropped
# FileNo column will also be dropped
# Bawsed on the Dataset we can drop Company and use InsurerID 


# #### Dropping of columns

# In[ ]:


df = df.drop(['SubCoverage', 'Company','FileNo', 'ComplaintID', 'State'], axis = 1)
df.columns


# #### Segragating 66 Sub Reasons into 5 Sections - Claims, Delay, No SubReason, Service, and Underwriting/Sales

# In[ ]:


df['SubReason'].unique()


# In[ ]:


df['SubReason'] = df['SubReason'].replace({'Other [Enter Sub-Reason]':'Other_Enter_Sub_Reason'})


# In[ ]:


claims = ['Actual Cash Value Dispute', 'Cancellation', 'CPT Code Issue', 
          'CT Continuation 38a-512a', 'Denial of Claim', 'Failed to Remit Premium', 'No Coverage/Premium Paid']
df['New_SubReason'] = ['Claims' if x in claims else x for x in df['SubReason']] 


# In[ ]:


df['New_SubReason'].unique()


# In[ ]:


delay = ["Carrier Never Rec'd Appl", "Carrier Never Rec'd Claim", 'Claim Delays', 'Claim Procedure', 
         'Policy Issue Delay', 'Policy Service', 'Policy Service Delay', 
         'Premium Refund Delay', 'Time Delay', 'Underwriting Delays']
df['New_SubReason'] = ['Delay' if x in delay else x for x in df['New_SubReason']] 


# In[ ]:


nosubreason = ['No Subreason', 'Other_Enter_Sub_Reason']
df['New_SubReason'] = ['No_SubReason' if x in nosubreason else x for x in df['New_SubReason']] 


# In[ ]:


service = ['Audit','Benefit Extension','Case Management','Classification','Comparative Negligence','Contract Provision',
'Coordination of Benefit','Discontinuation & Replmnt','Duplicate Coverage','Loss of Use','Misleading Advertising',
'Mis-Quote','Misrepresentation','Network Adequacy','No Response','Non-Renewal','Provider Contract Issue','Refusal to Insure',
'Steering','Surprise Billing','Unfair Discrimination','Unprofessional Conduct','Unsatisfactory Offer',
           'Unsatisfactory Settlement']
df['New_SubReason'] = ['Service' if x in service else x for x in df['New_SubReason']] 


# In[ ]:


undr_sales = ['After Mrkt Prts/Unsat Set','Diminished Value','Eligibility of Provider','Excessive Charges','Labor Rate',
'Mandated Benefit','Medical Necessity','Other Fees','Pre-Existing Condition','Premium/Notice','Premium/Rate Increase',
'Producer Handling','Rebate','Replacement','Rescission','Service Fees','Storage Fees','Subrogation','Unapproved Form',
'Underwrtng/Credit History','Underwrtng/Waivers/Rated','UR Procedure','Usual and Customary Fees']

df['New_SubReason'] = ['Underwriting/Sales' if x in undr_sales else x for x in df['New_SubReason']]


# In[ ]:


df.head()


# ### Train set with all ResolutionStatus

# In[ ]:


import copy
df_bu = df.copy()


# ## Processing on Train set by dropping 'Re-Opened' & 'Open' ResolutionStatus

# In[ ]:


df = df.drop(df[df.ResolutionStatus.isin(["Re-Opened", "Open"])].index)


# In[ ]:


# Creating column that gives the days taken to resolve complaints
# Going forward DateOfResolution and DateOfRegistration can be dropped
df['DaysTakenforResn'] = (df['DateOfResolution'] - df['DateOfRegistration']).dt.days 


# ### Reading and applying changes to the Test dataset

# In[ ]:


train = df.copy()
train.sort_values(by ='RecoveredFromInsurer', ascending = False).head()


# In[ ]:


# Dropping 'SubReason', 'DateOfRegistration', 'DateOfResolution'
train = train.drop(['SubReason', 'DateOfRegistration', 'DateOfResolution'], axis = 1)
train.columns


# In[ ]:


# Reading Test dataset given
test = pd.read_csv("../input/Test_Complaints-1565162197608.csv")


# In[ ]:


test.head()


# In[ ]:


# Converting DateOfRegistration & DateOfResolution to Date
test['DateOfRegistration'] =pd.to_datetime(test.DateOfRegistration, format = '%Y-%m-%d')
test['DateOfResolution'] = pd.to_datetime(test.DateOfResolution, format = '%Y-%m-%d')


# In[ ]:


test = test.drop(['SubCoverage', 'Company','FileNo', 'ComplaintID', 'State'], axis = 1)
test.columns


# In[ ]:


test['SubReason'] = test['SubReason'].replace({'Other [Enter Sub-Reason]':'Other_Enter_Sub_Reason'})


# In[ ]:


claims = ['Actual Cash Value Dispute', 'Cancellation', 'CPT Code Issue', 
          'CT Continuation 38a-512a', 'Denial of Claim', 'Failed to Remit Premium', 'No Coverage/Premium Paid']
test['New_SubReason'] = ['Claims' if x in claims else x for x in test['SubReason']] 


# In[ ]:


delay = ["Carrier Never Rec'd Appl", "Carrier Never Rec'd Claim", 'Claim Delays', 'Claim Procedure', 
         'Policy Issue Delay', 'Policy Service', 'Policy Service Delay', 
         'Premium Refund Delay', 'Time Delay', 'Underwriting Delays']
test['New_SubReason'] = ['Delay' if x in delay else x for x in test['New_SubReason']] 


# In[ ]:


nosubreason = ['No Subreason', 'Other_Enter_Sub_Reason']
test['New_SubReason'] = ['No_SubReason' if x in nosubreason else x for x in test['New_SubReason']] 


# In[ ]:


service = ['Audit','Benefit Extension','Case Management','Classification','Comparative Negligence','Contract Provision',
'Coordination of Benefit','Discontinuation & Replmnt','Duplicate Coverage','Loss of Use','Misleading Advertising',
'Mis-Quote','Misrepresentation','Network Adequacy','No Response','Non-Renewal','Provider Contract Issue','Refusal to Insure',
'Steering','Surprise Billing','Unfair Discrimination','Unprofessional Conduct','Unsatisfactory Offer',
           'Unsatisfactory Settlement']
test['New_SubReason'] = ['Service' if x in service else x for x in test['New_SubReason']] 


# In[ ]:


undr_sales = ['After Mrkt Prts/Unsat Set','Diminished Value','Eligibility of Provider','Excessive Charges','Labor Rate',
'Mandated Benefit','Medical Necessity','Other Fees','Pre-Existing Condition','Premium/Notice','Premium/Rate Increase',
'Producer Handling','Rebate','Replacement','Rescission','Service Fees','Storage Fees','Subrogation','Unapproved Form',
'Underwrtng/Credit History','Underwrtng/Waivers/Rated','UR Procedure','Usual and Customary Fees']

test['New_SubReason'] = ['Underwriting/Sales' if x in undr_sales else x for x in test['New_SubReason']] 


# ### Test set with all ResolutionStatus

# In[ ]:


test_bu = test.copy()


# ## Dropping 'Re-Opened' & 'Open' for ResolutionStatus from test for analysis continuation

# In[ ]:


test = test.drop(test[test.ResolutionStatus.isin(["Re-Opened", "Open"])].index)


# ## Analysis continuation for only 'Closed' ResolutionStatus

# In[ ]:


test['DaysTakenforResn'] = (test['DateOfResolution'] - test['DateOfRegistration']).dt.days 


# In[ ]:


# Dropping 'SubReason', 'DateOfRegistration', 'DateOfResolution'
test = test.drop(['SubReason', 'DateOfRegistration', 'DateOfResolution'], axis = 1)
test.columns


# In[ ]:


# Separating the target variable
X_train = train.copy().drop('DRC', axis = 1)
y_train = train['DRC']
print(X_train.shape)
print(y_train.shape)


# In[ ]:


test.shape


# In[ ]:


# Checking Train NAs
X_train.isna().sum()


# In[ ]:


# Checking Test NAs
test.isna().sum()


# ## Dummies for Categorical Variables

# In[ ]:


cat_cols = ['Coverage', 'Reason', 'EnforcementAction', 'Conclusion', 'ResolutionStatus', 'New_SubReason']
num_cols = ['RecoveredFromInsurer', 'DaysTakenforResn']


# In[ ]:


X_train_num = len(X_train)
combined_dataset = pd.concat(objs=[X_train, test], axis=0)


# In[ ]:


combined_dataset.head()


# In[ ]:


def impute_with_mode(x):
    max_x = x.value_counts()
    mode = max_x[max_x == max_x.max()].index[0]
    x[x.isna()] = mode
    return x

combined_dataset[cat_cols] = combined_dataset[cat_cols].apply(lambda x: impute_with_mode(x))


# In[ ]:


combined_dataset.isna().sum()


# In[ ]:


combined_dataset = pd.get_dummies(combined_dataset, columns=cat_cols, drop_first=True)


# In[ ]:


combined_dataset.head()


# In[ ]:


X_train = copy.copy(combined_dataset[:X_train_num])
test = copy.copy(combined_dataset[X_train_num:])


# In[ ]:


print(X_train.shape)
print(test.shape)


# In[ ]:


train = pd.concat([X_train, y_train], axis =1)
print(train.shape)


# ## Creating 3 subsets of the data based on RecoveredFromInsurer
# - 1. Subset with RecoveredFromInsurer == 0
# - 2. Subset with RecoveredFromInsurer != 0
# - 3. Subset with RecoveredFromInsurer !=0 & < 400000

# ### 1. For RecoveredFromInsurer == 0

# In[ ]:


df0 = train[train['RecoveredFromInsurer'] == 0]
print(df0.shape)


# In[ ]:


test0 = test[test['RecoveredFromInsurer'] == 0]
print(test0.shape)


# In[ ]:


# Splitting df0 into train0 and val0
from sklearn.model_selection import train_test_split

train0 , val0 = train_test_split(df0, test_size = 0.3, shuffle = True ,random_state = 800)


# In[ ]:


X_Train0 = train0.copy().drop('DRC', axis = 1)
y_Train0 = train0[['InsurerID','DRC']]
X_val0 = val0.copy().drop('DRC', axis = 1)
y_val0 = val0[['InsurerID','DRC']]


# In[ ]:


# Setting InsurerID as the Index
X_Train0.set_index('InsurerID', inplace = True)
y_Train0.set_index('InsurerID', inplace = True)
X_val0.set_index('InsurerID', inplace = True)
y_val0.set_index('InsurerID', inplace = True)


# In[ ]:


# Applying Random Forest 
from sklearn.ensemble import RandomForestClassifier
rfc0 = RandomForestClassifier()
rfc0.fit(X = X_Train0,y = y_Train0)


# In[ ]:


train_pred_rfc0 = rfc0.predict(X_Train0)
test_pred_rfc0 = rfc0.predict(X_val0)


# In[ ]:


from sklearn import metrics
from sklearn.metrics import f1_score
print("\nRF_Train accuracy", metrics.accuracy_score(y_Train0, train_pred_rfc0).round(5))
print("\nRF_Train f1-score for class ['poor', 'average', 'outstanding']", 
      metrics.f1_score(y_Train0, train_pred_rfc0, average = None).round(5))
print("\nRF_Validation accuracy", metrics.accuracy_score(y_val0, test_pred_rfc0).round(5))
print("\nRF_Validation f1-score for class ['poor', 'average', 'outstanding']", 
      metrics.f1_score(y_val0, test_pred_rfc0, average = None).round(5))


# In[ ]:


# Applying on Test data
test0_new = test0.copy()


# In[ ]:


test0_new.set_index('InsurerID', inplace = True)


# In[ ]:


y_pred_rfc0 = rfc0.predict(test0_new)


# In[ ]:


y_pred_rfc0 = pd.DataFrame(y_pred_rfc0, columns = ['DRC'])


# In[ ]:


y_pred_rfc0['InsurerID'] = test0['InsurerID']


# In[ ]:


y_pred_rfc0.DRC.replace(('poor', 'average', 'outstanding'), (1, 2, 3), inplace=True)


# In[ ]:


group_RF0_pred = pd.DataFrame(y_pred_rfc0.groupby(['InsurerID'])['DRC'].mean())
group_RF0_pred = group_RF0_pred.round().astype(int)
group_RF0_pred.head(10)


# ### 2. For RecoveredFromInsurer != 0

# In[ ]:


dfnot0 = train[train['RecoveredFromInsurer'] != 0]
print(dfnot0.shape)


# In[ ]:


dfnot0.sort_values(by = 'RecoveredFromInsurer', ascending = False).head()


# In[ ]:


testnot0 = test[test['RecoveredFromInsurer'] != 0]
print(testnot0.shape)


# In[ ]:


Xno0 = dfnot0.copy().drop('DRC', axis = 1)
yno0 = dfnot0[['DRC']]


# In[ ]:


Xno0_num = len(Xno0)
combined_dataset_no0 = pd.concat(objs=[Xno0, testnot0], axis=0)
print(combined_dataset_no0.shape)


# In[ ]:


# Standardization of Numerical Columns
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(combined_dataset_no0.loc[:,['RecoveredFromInsurer', 'DaysTakenforResn']])

combined_dataset_no0.loc[:,['RecoveredFromInsurer', 'DaysTakenforResn']]=scaler.transform(
    combined_dataset_no0.loc[:,['RecoveredFromInsurer', 'DaysTakenforResn']])


# In[ ]:


Xno0 = copy.copy(combined_dataset_no0[:Xno0_num])
testnot0 = copy.copy(combined_dataset_no0[Xno0_num:])


# In[ ]:


dfnot0 = pd.concat([Xno0, yno0], axis =1)


# In[ ]:


print(dfnot0.shape)
print(testnot0.shape)


# In[ ]:


# Splitting dfnot0 into trainnot0 and valnot0
from sklearn.model_selection import train_test_split

trainnot0 , valnot0 = train_test_split(dfnot0, test_size = 0.3, shuffle = True ,random_state = 800)


# In[ ]:


X_Trainnot0 = trainnot0.copy().drop('DRC', axis = 1)
y_Trainnot0 = trainnot0[['InsurerID','DRC']]
X_valnot0 = valnot0.copy().drop('DRC', axis = 1)
y_valnot0 = valnot0[['InsurerID','DRC']]


# In[ ]:


# Setting InsurerID as the Index
X_Trainnot0.set_index('InsurerID', inplace = True)
y_Trainnot0.set_index('InsurerID', inplace = True)
X_valnot0.set_index('InsurerID', inplace = True)
y_valnot0.set_index('InsurerID', inplace = True)


# In[ ]:


# Applying Random Forest
rfcnot0 = RandomForestClassifier()
rfcnot0.fit(X = X_Trainnot0,y = y_Trainnot0)


# In[ ]:


train_pred_rfcnot0 = rfcnot0.predict(X_Trainnot0)
test_pred_rfcnot0 = rfcnot0.predict(X_valnot0)


# In[ ]:


from sklearn import metrics
from sklearn.metrics import f1_score
print("\nRF_not0_Train accuracy", metrics.accuracy_score(y_Trainnot0, train_pred_rfcnot0).round(5))
print("\nRF_not0_Train f1-score for class ['poor', 'average', 'outstanding']", 
      metrics.f1_score(y_Trainnot0, train_pred_rfcnot0, average = None).round(5))
print("\nRF_not0_Validation accuracy", metrics.accuracy_score(y_valnot0, test_pred_rfcnot0).round(5))
print("\nRF_not0_Validation f1-score for class ['poor', 'average', 'outstanding']", 
      metrics.f1_score(y_valnot0, test_pred_rfcnot0, average = None).round(5))


# In[ ]:


# Applying on Test data
testnot0_new = testnot0.copy()


# In[ ]:


testnot0_new.set_index('InsurerID', inplace = True)


# In[ ]:


y_pred_not0 = rfcnot0.predict(testnot0_new)


# In[ ]:


y_pred_not0 = pd.DataFrame(y_pred_not0, columns = ['DRC'])


# In[ ]:


y_pred_not0['InsurerID'] = testnot0['InsurerID']


# In[ ]:


y_pred_not0.DRC.replace(('poor', 'average', 'outstanding'), (1, 2, 3), inplace=True)


# In[ ]:


group_RF_not0_pred = pd.DataFrame(y_pred_not0.groupby(['InsurerID'])['DRC'].mean())
group_RF_not0_pred = group_RF_not0_pred.round().astype(int)
group_RF_not0_pred.head(10)


# ### 3. For RecoveredFromInsurer != 0 removing outliers (RecoveredFromInsurer >= 400000)

# In[ ]:


dfnot0_noout = train[(train['RecoveredFromInsurer'] != 0) & (train['RecoveredFromInsurer'] < 400000)]
print(dfnot0_noout.shape)


# In[ ]:


testnot0_noout = test[(test['RecoveredFromInsurer'] != 0) & (test['RecoveredFromInsurer'] < 400000)]
print(testnot0_noout.shape)


# In[ ]:


Xno0_noout = dfnot0_noout.copy().drop('DRC', axis = 1)
yno0_noout = dfnot0_noout[['DRC']]


# In[ ]:


Xno0_noout_num = len(Xno0_noout)
combined_dataset_no0_noout = pd.concat(objs=[Xno0_noout, testnot0_noout], axis=0)
print(combined_dataset_no0_noout.shape)


# In[ ]:


# Standardization of Numerical Columns
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(combined_dataset_no0_noout.loc[:,['RecoveredFromInsurer', 'DaysTakenforResn']])

combined_dataset_no0_noout.loc[:,['RecoveredFromInsurer', 'DaysTakenforResn']]=scaler.transform(
    combined_dataset_no0_noout.loc[:,['RecoveredFromInsurer', 'DaysTakenforResn']])


# In[ ]:


Xno0_noout = copy.copy(combined_dataset_no0_noout[:Xno0_noout_num])
testnot0_noout = copy.copy(combined_dataset_no0_noout[Xno0_noout_num:])


# In[ ]:


dfnot0_noout = pd.concat([Xno0_noout, yno0_noout], axis =1)
print(dfnot0_noout.shape)
print(testnot0_noout.shape)


# In[ ]:


# Splitting dfnot0 into trainnot0 and valnot0
from sklearn.model_selection import train_test_split

trainnot0_noout , valnot0_noout = train_test_split(dfnot0_noout, test_size = 0.3, shuffle = True ,random_state = 800)


# In[ ]:


X_Trainnot0_noout = trainnot0_noout.copy().drop('DRC', axis = 1)
y_Trainnot0_noout = trainnot0_noout[['InsurerID','DRC']]
X_valnot0_noout = valnot0_noout.copy().drop('DRC', axis = 1)
y_valnot0_noout = valnot0_noout[['InsurerID','DRC']]


# In[ ]:


# Setting InsurerID as the Index
X_Trainnot0_noout.set_index('InsurerID', inplace = True)
y_Trainnot0_noout.set_index('InsurerID', inplace = True)
X_valnot0_noout.set_index('InsurerID', inplace = True)
y_valnot0_noout.set_index('InsurerID', inplace = True)


# In[ ]:


# Applying Random Forest
rfcnot0_noout = RandomForestClassifier()
rfcnot0_noout.fit(X = X_Trainnot0_noout,y = y_Trainnot0_noout)


# In[ ]:


train_pred_rfcnot0_noout = rfcnot0_noout.predict(X_Trainnot0_noout)
test_pred_rfcnot0_noout = rfcnot0_noout.predict(X_valnot0_noout)


# In[ ]:


from sklearn import metrics
from sklearn.metrics import f1_score
print("\nRF_not0_noout_Train accuracy", metrics.accuracy_score(y_Trainnot0_noout, train_pred_rfcnot0_noout).round(5))
print("\nRF_not0_noout_Train f1-score for class ['poor', 'average', 'outstanding']", 
      metrics.f1_score(y_Trainnot0_noout, train_pred_rfcnot0_noout, average = None).round(5))
print("\nRF_not0_noout_Validation accuracy", metrics.accuracy_score(y_valnot0_noout, test_pred_rfcnot0_noout).round(5))
print("\nRF_not0_noout_Validation f1-score for class ['poor', 'average', 'outstanding']", 
      metrics.f1_score(y_valnot0_noout, test_pred_rfcnot0_noout, average = None).round(5))


# In[ ]:


# Applying on Test data
testnot0_noout_new = testnot0_noout.copy()


# In[ ]:


testnot0_noout_new.set_index('InsurerID', inplace = True)


# In[ ]:


y_pred_not0_noout = rfcnot0_noout.predict(testnot0_noout_new)


# In[ ]:


y_pred_not0_noout = pd.DataFrame(y_pred_not0_noout, columns = ['DRC'])


# In[ ]:


y_pred_not0_noout['InsurerID'] = testnot0_noout['InsurerID']


# In[ ]:


y_pred_not0_noout.DRC.replace(('poor', 'average', 'outstanding'), (1, 2, 3), inplace=True)


# In[ ]:


group_RF_not0_noout_pred = pd.DataFrame(y_pred_not0_noout.groupby(['InsurerID'])['DRC'].mean())
group_RF_not0_noout_pred = group_RF_not0_noout_pred.round().astype(int)
group_RF_not0_noout_pred.head(10)


# ## Ensemble with RF for RecoveredFromInsurer == 0 and RF for RecoveredFromInsurer != 0

# In[ ]:


from sklearn.ensemble import VotingClassifier


# In[ ]:


model = VotingClassifier(estimators = [('rf1', rfc0), ('rf2', rfcnot0)], voting = 'hard')


# In[ ]:


Train , val = train_test_split(train, test_size = 0.3, shuffle = True ,random_state = 800)


# In[ ]:


X_Train = Train.copy().drop('DRC', axis = 1)
y_Train = Train[['InsurerID','DRC']]
X_val = val.copy().drop('DRC', axis = 1)
y_val = val[['InsurerID','DRC']]


# In[ ]:


X_Train.set_index('InsurerID', inplace = True)
y_Train.set_index('InsurerID', inplace = True)
X_val.set_index('InsurerID', inplace = True)
y_val.set_index('InsurerID', inplace = True)


# In[ ]:


model.fit(X_Train, y_Train)


# In[ ]:


train_model_pred = model.predict(X_Train)
test_model_pred = model.predict(X_val)


# In[ ]:


print("\nEn_Train accuracy", metrics.accuracy_score(y_Train, train_model_pred).round(5))
print("\nEn_Train f1-score for class ['poor', 'average', 'outstanding']", 
      metrics.f1_score(y_Train, train_model_pred, average = None).round(5))
print("\nEn_Validation accuracy", metrics.accuracy_score(y_val, test_model_pred).round(5))
print("\nEn_Validation f1-score for class ['poor', 'average', 'outstanding']", 
      metrics.f1_score(y_val, test_model_pred, average = None).round(5))


# In[ ]:


test_en1 = test.copy()


# In[ ]:


test_en1.set_index('InsurerID', inplace = True)


# In[ ]:


y_pred_en_model1 = model.predict(test_en1)


# In[ ]:


y_pred_en_model1 = pd.DataFrame(y_pred_en_model1, columns = ['DRC'])


# In[ ]:


y_pred_en_model1['InsurerID'] = test['InsurerID']


# In[ ]:


y_pred_en_model1.DRC.replace(('poor', 'average', 'outstanding'), (1, 2, 3), inplace=True)


# In[ ]:


group_ensemble1_pred = pd.DataFrame(y_pred_en_model1.groupby(['InsurerID'])['DRC'].mean())
group_ensemble1_pred = group_ensemble1_pred.round().astype(int)
group_ensemble1_pred.head(10)


# In[ ]:


pd.DataFrame(group_ensemble1_pred, columns=['DRC']).to_csv('Prediction_DRC_ensemble1.csv')


# ## Ensemble with RF for RecoveredFromInsurer == 0 and RF for RecoveredFromInsurer (!= 0 & <400000)

# In[ ]:


model_0_noout = VotingClassifier(estimators = [('rf1', rfc0), ('rf3', rfcnot0_noout)], voting = 'hard')


# In[ ]:


model_0_noout.fit(X_Train, y_Train)


# In[ ]:


train_modelnoout_pred = model_0_noout.predict(X_Train)
test_modelnoout_pred = model_0_noout.predict(X_val)


# In[ ]:


print("\nEn2_Train accuracy", metrics.accuracy_score(y_Train, train_modelnoout_pred).round(5))
print("\nEn2_Train f1-score for class ['poor', 'average', 'outstanding']", 
      metrics.f1_score(y_Train, train_modelnoout_pred, average = None).round(5))
print("\nEn2_Validation accuracy", metrics.accuracy_score(y_val, test_modelnoout_pred).round(5))
print("\nEn2_Validation f1-score for class ['poor', 'average', 'outstanding']", 
      metrics.f1_score(y_val, test_modelnoout_pred, average = None).round(5))


# ## Ensemble for all RFs

# In[ ]:


model_all = VotingClassifier(estimators = [('rf1', rfc0), ('rf2', rfcnot0),('rf3', rfcnot0_noout)], voting = 'hard')


# In[ ]:


model_all.fit(X_Train, y_Train)


# In[ ]:


train_model_all = model_all.predict(X_Train)
test_model_all = model_all.predict(X_val)


# In[ ]:


print("\nEn_all_Train accuracy", metrics.accuracy_score(y_Train, train_model_all).round(5))
print("\nEn_all_Train f1-score for class ['poor', 'average', 'outstanding']", 
      metrics.f1_score(y_Train, train_model_all, average = None).round(5))
print("\nEn_all_Validation accuracy", metrics.accuracy_score(y_val, test_model_all).round(5))
print("\nEn_all_Validation f1-score for class ['poor', 'average', 'outstanding']", 
      metrics.f1_score(y_val, test_model_all, average = None).round(5))


# In[ ]:


y_pred_en_model_all = model_all.predict(test_en1)


# In[ ]:


y_pred_en_model_all = pd.DataFrame(y_pred_en_model_all, columns = ['DRC'])


# In[ ]:


y_pred_en_model_all['InsurerID'] = test['InsurerID']


# In[ ]:


y_pred_en_model_all.DRC.replace(('poor', 'average', 'outstanding'), (1, 2, 3), inplace=True)


# In[ ]:


group_ensemble_predall = pd.DataFrame(y_pred_en_model_all.groupby(['InsurerID'])['DRC'].mean())
group_ensemble_predall = group_ensemble_predall.round().astype(int)
group_ensemble_predall.head(10)


# In[ ]:


# Writing prediction for upload


# In[ ]:


y_pred_all_new = y_pred_en_model_all.groupby('InsurerID')['DRC'].value_counts()
# Grouping to count the occurrences for different DRCs


# In[ ]:


arf_df = y_pred_all_new.unstack()
# Reshaping data to stack the columns row-wise


# In[ ]:


arf_df_pred = arf_df.idxmax(axis=1, skipna = True)
# Taking the max value based on index


# In[ ]:


arf_df_pred = pd.DataFrame(arf_df_pred, columns = ['DRC'])


# In[ ]:


pd.DataFrame(arf_df_pred).to_csv('Ensemble_AllRFs.csv')
arf_df_pred.head()


# # Analysis with all types of Resolution Status

# In[ ]:


df_bu.ResolutionStatus.unique()


# In[ ]:


test_bu.ResolutionStatus.unique()


# In[ ]:


df_bu.isna().sum()


# In[ ]:


test_bu.isna().sum()


# In[ ]:


import datetime
df_bu.loc[df_bu['DateOfResolution'].isna(), 'DateOfResolution'] = datetime.datetime.now()
test_bu.loc[test_bu['DateOfResolution'].isna(), 'DateOfResolution'] = datetime.datetime.now()


# In[ ]:


df_bu.loc[df_bu['DateOfResolution'].isna(), 'DateOfResolution'] = datetime.datetime.now()


# In[ ]:


df_bu.isna().sum()


# In[ ]:


test_bu.isna().sum()


# In[ ]:


df_bu['DaysTakenforResn'] = (df_bu['DateOfResolution'] - df_bu['DateOfRegistration']).dt.days
df_bu.head()


# In[ ]:


test_bu['DaysTakenforResn'] = (test_bu['DateOfResolution'] - test_bu['DateOfRegistration']).dt.days
test_bu.head()


# In[ ]:


# Dropping 'SubReason', 'DateOfRegistration', 'DateOfResolution'
df_bu = df_bu.drop(['SubReason', 'DateOfRegistration', 'DateOfResolution'], axis = 1)
print(df_bu.columns)
test_bu = test_bu.drop(['SubReason', 'DateOfRegistration', 'DateOfResolution'], axis = 1)
print(test_bu.columns)


# In[ ]:


# Separating the target variable
X_train_new = df_bu.copy().drop('DRC', axis = 1)
y_train_new = df_bu['DRC']
print(X_train_new.shape)
print(y_train_new.shape)


# In[ ]:


X_train_new_num = len(X_train_new)
combined_dataset_new = pd.concat(objs=[X_train_new, test_bu], axis=0)


# In[ ]:


combined_dataset_new[cat_cols] = combined_dataset_new[cat_cols].apply(lambda x: impute_with_mode(x))


# In[ ]:


combined_dataset_new = pd.get_dummies(combined_dataset_new, columns=cat_cols, drop_first=True)


# In[ ]:


X_train_new = copy.copy(combined_dataset_new[:X_train_new_num])
test_bu = copy.copy(combined_dataset_new[X_train_new_num:])


# In[ ]:


train_new = pd.concat([X_train_new, y_train_new], axis =1)
print(train_new.shape)
print(test_bu.shape)


# ### Basic RF - applying on all the data

# In[ ]:


train_brf , val_brf = train_test_split(train_new, test_size = 0.3, shuffle = True ,random_state = 800)


# In[ ]:


X_Train_brf = train_brf.copy().drop('DRC', axis = 1)
y_Train_brf = train_brf[['InsurerID','DRC']]
X_val_brf = val_brf.copy().drop('DRC', axis = 1)
y_val_brf = val_brf[['InsurerID','DRC']]


# In[ ]:


X_Train_brf.set_index('InsurerID', inplace = True)
y_Train_brf.set_index('InsurerID', inplace = True)
X_val_brf.set_index('InsurerID', inplace = True)
y_val_brf.set_index('InsurerID', inplace = True)


# In[ ]:


# Applying Random Forest
rf_brf = RandomForestClassifier()
rf_brf.fit(X = X_Train_brf,y = y_Train_brf)


# In[ ]:


train_pred_brf = rf_brf.predict(X_Train_brf)
test_pred_brf = rf_brf.predict(X_val_brf)


# In[ ]:


print("\nRF_new_Train accuracy", metrics.accuracy_score(y_Train_brf, train_pred_brf).round(5))
print("\nRF_new_Train f1-score for class ['poor', 'average', 'outstanding']", 
      metrics.f1_score(y_Train_brf, train_pred_brf, average = None).round(5))
print("\nRF_new_Validation accuracy", metrics.accuracy_score(y_val_brf, test_pred_brf).round(5))
print("\nRF_new_Validation f1-score for class ['poor', 'average', 'outstanding']", 
      metrics.f1_score(y_val_brf, test_pred_brf, average = None).round(5))


# ## Creating 3 subsets of the data based on RecoveredFromInsurer for all type of Resolution Status
# - 1. Subset with RecoveredFromInsurer == 0
# - 2. Subset with RecoveredFromInsurer != 0
# - 3. Subset with RecoveredFromInsurer !=0 & < 400000

# ### 1. For RecoveredFromInsurer == 0

# In[ ]:


dfnew0 = train_new[train_new['RecoveredFromInsurer'] == 0]
testnew0 = test_bu[test_bu['RecoveredFromInsurer'] == 0]
print(dfnew0.shape)
print(testnew0.shape)


# In[ ]:


train_new0, val_new0 = train_test_split(dfnew0, test_size = 0.3, shuffle = True, random_state = 800)


# In[ ]:


X_Train_new0 = train_new0.copy().drop('DRC', axis = 1)
y_Train_new0 = train_new0[['InsurerID', 'DRC']]
X_val_new0 = val_new0.copy().drop('DRC', axis = 1)
y_val_new0 = val_new0[['InsurerID', 'DRC']]


# In[ ]:


X_Train_new0.set_index('InsurerID', inplace = True)
y_Train_new0.set_index('InsurerID', inplace = True)
X_val_new0.set_index('InsurerID', inplace = True)
y_val_new0.set_index('InsurerID', inplace = True)


# In[ ]:


rf_0_new = RandomForestClassifier()
rf_0_new.fit(X_Train_new0, y_Train_new0)


# In[ ]:


train_pred_rf_0_new = rf_0_new.predict(X_Train_new0)
test_pred_rf_0_new = rf_0_new.predict(X_val_new0)


# In[ ]:


print("\nRF_0_new_Train accuracy", metrics.accuracy_score(y_Train_new0, train_pred_rf_0_new).round(5))
print("\nRF_0_new_Train f1-score for class ['poor', 'average', 'outstanding']", 
      metrics.f1_score(y_Train_new0, train_pred_rf_0_new, average = None).round(5))
print("\nRF_0_new_Validation accuracy", metrics.accuracy_score(y_val_new0, test_pred_rf_0_new).round(5))
print("\nRF_0_new_Validation f1-score for class ['poor', 'average', 'outstanding']", 
      metrics.f1_score(y_val_new0, test_pred_rf_0_new, average = None).round(5))


# ### 2. For RecoveredFromInsurer != 0

# In[ ]:


dfnewnot0 = train_new[train_new['RecoveredFromInsurer'] != 0]
testnewnot0 = test_bu[test_bu['RecoveredFromInsurer'] != 0]
print(dfnewnot0.shape)
print(testnewnot0.shape)


# In[ ]:


X = dfnewnot0.copy().drop('DRC', axis = 1)
y = dfnewnot0[['DRC']]


# In[ ]:


X_num = len(X)
combined_df = pd.concat(objs = [X, testnewnot0], axis = 0)
print(combined_df.shape)


# In[ ]:


scaler = StandardScaler()
scaler.fit(combined_df.loc[:,['RecoveredFromInsurer', 'DaysTakenforResn']])
combined_df.loc[:,['RecoveredFromInsurer', 'DaysTakenforResn']]=scaler.transform(
    combined_df.loc[:,['RecoveredFromInsurer', 'DaysTakenforResn']])


# In[ ]:


X = copy.copy(combined_df[:X_num])
testnewnot0 = copy.copy(combined_df[X_num:])


# In[ ]:


dfnewnot0 = pd.concat([X, y], axis=1)
print(dfnewnot0.shape)
print(testnewnot0.shape)


# In[ ]:


train_newnot0, val_newnot0 = train_test_split(dfnewnot0, test_size = 0.3, shuffle = True, random_state = 800)


# In[ ]:


X_Train_newnot0 = train_newnot0.copy().drop('DRC', axis = 1)
y_Train_newnot0 = train_newnot0[['InsurerID', 'DRC']]
X_val_newnot0 = val_newnot0.copy().drop('DRC', axis = 1)
y_val_newnot0 = val_newnot0[['InsurerID', 'DRC']]


# In[ ]:


X_Train_newnot0.set_index('InsurerID', inplace = True)
y_Train_newnot0.set_index('InsurerID', inplace = True)
X_val_newnot0.set_index('InsurerID', inplace = True)
y_val_newnot0.set_index('InsurerID', inplace = True)


# In[ ]:


rf_newnot0 = RandomForestClassifier()
rf_newnot0.fit(X_Train_newnot0, y_Train_newnot0)


# In[ ]:


train_pred_rf_newnot0 = rf_0_new.predict(X_Train_newnot0)
test_pred_rf_newnot0 = rf_0_new.predict(X_val_newnot0)


# In[ ]:


print("\nRF_new_not0_Train accuracy", metrics.accuracy_score(y_Train_newnot0, train_pred_rf_newnot0).round(5))
print("\nRF_new_not0_Train f1-score for class ['poor', 'average', 'outstanding']", 
      metrics.f1_score(y_Train_newnot0, train_pred_rf_newnot0, average = None).round(5))
print("\nRF_new_not0_Validation accuracy", metrics.accuracy_score(y_val_newnot0, test_pred_rf_newnot0).round(5))
print("\nRF_new_not0_Validation f1-score for class ['poor', 'average', 'outstanding']", 
      metrics.f1_score(y_val_newnot0, test_pred_rf_newnot0, average = None).round(5))


# ### 3. For RecoveredFromInsurer != 0 removing outliers (RecoveredFromInsurer >= 400000)

# In[ ]:


dfnew_no0_noout = train_new[(train_new['RecoveredFromInsurer'] != 0) & (train_new['RecoveredFromInsurer'] < 400000)]
testnew_no0_noout = test_bu[(test_bu['RecoveredFromInsurer'] != 0) &(test_bu['RecoveredFromInsurer'] < 400000)]
print(dfnew_no0_noout.shape)
print(testnew_no0_noout.shape)


# In[ ]:


X = dfnew_no0_noout.copy().drop('DRC', axis =1)
y = dfnew_no0_noout[['DRC']]


# In[ ]:


X_num = len(X)
combined_df = pd.concat(objs = [X, testnew_no0_noout], axis=0)
print(combined_df.shape)


# In[ ]:


scaler = StandardScaler()
scaler.fit(combined_df.loc[:,['RecoveredFromInsurer', 'DaysTakenforResn']])

combined_df.loc[:,['RecoveredFromInsurer', 'DaysTakenforResn']]=scaler.transform(
    combined_df.loc[:,['RecoveredFromInsurer', 'DaysTakenforResn']])


# In[ ]:


X = copy.copy(combined_df[:X_num])
testnew_no0_noout = copy.copy(combined_df[X_num:])


# In[ ]:


dfnew_no0_noout = pd.concat([X,y],axis =1)
print(dfnew_no0_noout.shape)
print(testnew_no0_noout.shape)


# In[ ]:


t, v = train_test_split(dfnew_no0_noout, test_size = 0.3, shuffle = True, random_state = 800)


# In[ ]:


Xt = t.copy().drop('DRC', axis = 1)
yt = t[['InsurerID', 'DRC']]
Xv = v.copy().drop('DRC', axis = 1)
yv = v[['InsurerID', 'DRC']]


# In[ ]:


Xt.set_index('InsurerID', inplace = True)
yt.set_index('InsurerID', inplace = True)
Xv.set_index('InsurerID', inplace = True)
yv.set_index('InsurerID', inplace = True)


# In[ ]:


RF = RandomForestClassifier()
RF.fit(X=Xt, y= yt)


# In[ ]:


tp = RF.predict(Xt)
vp = RF.predict(Xv)


# In[ ]:


print("\nRF_not0_noout_Train accuracy", metrics.accuracy_score(yt, tp).round(5))
print("\nRF_not0_noout_Train f1-score for class ['poor', 'average', 'outstanding']", 
      metrics.f1_score(yt, tp, average = None).round(5))
print("\nRF_not0_noout_Validation accuracy", metrics.accuracy_score(yv, vp).round(5))
print("\nRF_not0_noout_Validation f1-score for class ['poor', 'average', 'outstanding']", 
      metrics.f1_score(yv, vp, average = None).round(5))


# ## Ensemble with RF for RecoveredFromInsurer == 0 and RF for RecoveredFromInsurer != 0

# In[ ]:


model1 = VotingClassifier(estimators = [('rf1', rf_0_new), ('rf2', rf_newnot0)], voting = 'hard')


# In[ ]:


Train , val = train_test_split(train_new, test_size = 0.3, shuffle = True ,random_state = 800)


# In[ ]:


X_Train = Train.copy().drop('DRC', axis = 1)
y_Train = Train[['InsurerID','DRC']]
X_val = val.copy().drop('DRC', axis = 1)
y_val = val[['InsurerID','DRC']]


# In[ ]:


X_Train.set_index('InsurerID', inplace = True)
y_Train.set_index('InsurerID', inplace = True)
X_val.set_index('InsurerID', inplace = True)
y_val.set_index('InsurerID', inplace = True)


# In[ ]:


model1.fit(X_Train, y_Train)


# In[ ]:


train_model1_pred = model1.predict(X_Train)
test_model1_pred = model1.predict(X_val)


# In[ ]:


print("\nEnnew1_Train accuracy", metrics.accuracy_score(y_Train, train_model1_pred).round(5))
print("\nEnnew1_Train f1-score for class ['poor', 'average', 'outstanding']", 
      metrics.f1_score(y_Train, train_model1_pred, average = None).round(5))
print("\nEnnew1_Validation accuracy", metrics.accuracy_score(y_val, test_model1_pred).round(5))
print("\nEnnew1_Validation f1-score for class ['poor', 'average', 'outstanding']", 
      metrics.f1_score(y_val, test_model1_pred, average = None).round(5))


# ## Ensemble with RF for RecoveredFromInsurer == 0 and RF for RecoveredFromInsurer (!= 0 & <400000)

# In[ ]:


model2 = VotingClassifier(estimators = [('rf1', rf_0_new), ('rf2', RF)], voting = 'hard')


# In[ ]:


model2.fit(X_Train, y_Train)


# In[ ]:


train_model2_pred = model2.predict(X_Train)
test_model2_pred = model2.predict(X_val)


# In[ ]:


print("\nEnnew2_Train accuracy", metrics.accuracy_score(y_Train, train_model2_pred).round(5))
print("\nEnnew2_Train f1-score for class ['poor', 'average', 'outstanding']", 
      metrics.f1_score(y_Train, train_model2_pred, average = None).round(5))
print("\nEnnew2_Validation accuracy", metrics.accuracy_score(y_val, test_model2_pred).round(5))
print("\nEnnew2_Validation f1-score for class ['poor', 'average', 'outstanding']", 
      metrics.f1_score(y_val, test_model2_pred, average = None).round(5))


# ## Ensemble with all RFs

# In[ ]:


model3 = VotingClassifier(estimators = [('rf1', rf_0_new), ('rf2', rf_newnot0), ('rf3', RF)], voting = 'hard')


# In[ ]:


model3.fit(X_Train, y_Train)


# In[ ]:


train_model3_pred = model3.predict(X_Train)
test_model3_pred = model3.predict(X_val)


# In[ ]:


print("\nEnnew2_Train accuracy", metrics.accuracy_score(y_Train, train_model3_pred).round(5))
print("\nEnnew2_Train f1-score for class ['poor', 'average', 'outstanding']", 
      metrics.f1_score(y_Train, train_model3_pred, average = None).round(5))
print("\nEnnew2_Validation accuracy", metrics.accuracy_score(y_val, test_model3_pred).round(5))
print("\nEnnew2_Validation f1-score for class ['poor', 'average', 'outstanding']", 
      metrics.f1_score(y_val, test_model3_pred, average = None).round(5))

