#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd


# # Reading data from xlsx

# In[ ]:


bank=pd.read_excel('../input/loan.xlsx')


# In[ ]:


print(bank.head())
print(bank.shape)


# # Deleting columns of no use and columns having all value as NaN

# In[ ]:


del bank['id']
del bank['member_id']
del bank['emp_title']
del bank['issue_d']
del bank['url']
del bank['desc']
del bank['title']
del bank['zip_code']
del bank['addr_state']
del bank['earliest_cr_line']
del bank['last_pymnt_d']
del bank['next_pymnt_d']
del bank['last_credit_pull_d']
del bank['mths_since_last_major_derog']
del bank['annual_inc_joint']
del bank['dti_joint']
del bank['verification_status_joint']
del bank['tot_coll_amt']
del bank['tot_cur_bal']
del bank['open_acc_6m']
del bank['open_il_6m']
del bank['open_il_12m']
del bank['open_il_24m']
del bank['mths_since_rcnt_il']
del bank['total_bal_il']
del bank['il_util']
del bank['open_rv_12m']
del bank['open_rv_24m']
del bank['max_bal_bc']
del bank['all_util']
del bank['total_rev_hi_lim']
del bank['inq_fi']
del bank['total_cu_tl']
del bank['inq_last_12m']
del bank['acc_open_past_24mths']
del bank['avg_cur_bal']
del bank['bc_open_to_buy']
del bank['bc_util']
del bank['mo_sin_old_il_acct']
del bank['mo_sin_old_rev_tl_op']
del bank['mo_sin_rcnt_rev_tl_op']
del bank['mo_sin_rcnt_tl']
del bank['mort_acc']
del bank['mths_since_recent_bc']
del bank['mths_since_recent_bc_dlq']
del bank['mths_since_recent_inq']
del bank['mths_since_recent_revol_delinq']
del bank['num_accts_ever_120_pd']
del bank['num_actv_bc_tl']
del bank['num_actv_rev_tl']
del bank['num_bc_sats']
del bank['num_bc_tl']
del bank['num_il_tl']
del bank['num_op_rev_tl']
del bank['num_rev_accts']
del bank['num_rev_tl_bal_gt_0']
del bank['num_sats']
del bank['num_tl_120dpd_2m']
del bank['num_tl_30dpd']
del bank['num_tl_90g_dpd_24m']
del bank['num_tl_op_past_12m']
del bank['pct_tl_nvr_dlq']
del bank['percent_bc_gt_75']
del bank['tot_hi_cred_lim']
del bank['total_bal_ex_mort']
del bank['total_bc_limit']
del bank['total_il_high_credit_limit']


# In[ ]:


bank.shape


# # Describing the bank details

# In[ ]:


bank.describe()


# # Deleting the columns with 0 mean

# In[ ]:


del bank['collections_12_mths_ex_med']
del bank['acc_now_delinq']
del bank['chargeoff_within_12_mths']
del bank['delinq_amnt']
del bank['tax_liens']


# In[ ]:


bank.shape


# In[ ]:


bank.head()


# # Seeing the NaN values in bank columns

# In[ ]:


bank.isna().sum()


# # Dealing with all NaN Values

# In[ ]:


def classify(s):
    if(s=='10+ years'):
        return 10
    elif(s=='1 year'):
        return 1
    elif(s=='2 years'):
        return 2
    elif(s=='3 years'):
        return 3
    elif(s=='4 years'):
        return 4
    elif(s=='5 years'):
        return 5
    elif(s=='6 years'):
        return 6
    elif(s=='7 years'):
        return 7
    elif(s=='8 years'):
        return 8
    elif(s=='9 years'):
        return 9
    else:
        return 0


# In[ ]:


bank['emp_length']=bank.emp_length.apply(classify)


# In[ ]:


bank.mths_since_last_delinq.fillna(0,inplace=True)


# In[ ]:


del bank['mths_since_last_record']


# In[ ]:


bank.pub_rec_bankruptcies.fillna(bank.pub_rec_bankruptcies.mean(),inplace=True)


# In[ ]:


bank.revol_util.fillna(0,inplace=True)


# # Taking a look at data

# In[ ]:


bank


# # Dealing wth string type of data which is important for classification but string type of data can't be used in classification

# In[ ]:


bank.term.unique()


# In[ ]:


def term(s):
    if(s=='36 months'):
        return 36
    else:
        return 60


# In[ ]:


bank['term']=bank.term.apply(term)


# In[ ]:


bank.grade.unique()


# In[ ]:


def grade(s):
    if(s=='A'):
        return 1
    elif(s=='B'):
        return 2
    elif(s=='C'):
        return 3
    elif(s=='D'):
        return 4
    elif(s=='E'):
        return 5
    elif(s=='F'):
        return 6
    else:
        return 7


# In[ ]:


bank['grade']=bank.grade.apply(grade)


# In[ ]:


del bank['sub_grade']


# In[ ]:


bank.home_ownership.unique()


# In[ ]:


def home(s):
    if(s=='RENT'):
        return 1
    elif(s=='OWN'):
        return 2
    elif(s=='MORTGAGE'):
        return 3
    elif(s=='OTHER'):
        return 4
    else:
        return 5


# In[ ]:


bank['home_ownership']=bank.home_ownership.apply(home)


# In[ ]:


bank.application_type.unique()


# # All application type are of type INDIVIDUAL so this feature is not going to have any effect on the preictions hence deleting it is best option

# In[ ]:


del bank['application_type']


# In[ ]:


bank.columns


# In[ ]:


bank.verification_status.unique()


# In[ ]:


def verification(s):
    if(s=='Verified'):
        return 1
    elif(s=='Source Verified'):
        return 2
    else:
        return 3


# In[ ]:


bank['verification_status']=bank.verification_status.apply(verification)


# # Creating our Y for the data which is loan status

# In[ ]:


y=bank['loan_status']


# In[ ]:


y.unique()


# In[ ]:


del bank['loan_status']


# In[ ]:


def loan_status(s):
    if(s=='Fully Paid'):
        return 1
    elif(s=='Charged Off'):
        return 2
    else:
        return 3


# In[ ]:


y=y.apply(loan_status)


# In[ ]:


y.describe()


# # Again converting string data to int binary data

# In[ ]:


bank.pymnt_plan.unique()


# In[ ]:


del bank['pymnt_plan']


# In[ ]:


bank.purpose.unique()


# # According to me the company has the most danger from the persons who doesn't gives any purpose of loans so i will be classifying it in binary type

# In[ ]:


def purpose(s):
    if(s=='other'):
        return 2
    else:
        return 1


# In[ ]:


bank['purpose']=bank.purpose.apply(purpose)


# In[ ]:


bank.initial_list_status.unique()


# In[ ]:


del bank['initial_list_status']


# In[ ]:


bank.describe()


# In[ ]:


bank.shape


# # Till now all NaN has been removed and x and y are also there so task of data cleaning is done

# In[ ]:


from sklearn import model_selection


# # Dividing Data into training , Development and testing set

# In[ ]:


x_train,x_test,y_train,y_test=model_selection.train_test_split(bank,y,test_size=0.3)


# In[ ]:


print(x_train.shape)
print(x_test.shape)


# In[ ]:


x_dev,x_test1,y_dev,y_test1=model_selection.train_test_split(x_test,y_test,test_size=0.5)


# In[ ]:


print(x_dev.shape)
print(x_test1.shape)


# # Applying algorithms to see which one fits best

# In[ ]:


from xgboost.sklearn import XGBClassifier


# In[ ]:


clf=XGBClassifier(random_state=1,n_jobs=8)


# In[ ]:


clf.fit(x_train,y_train)


# In[ ]:


y_predict=clf.predict(x_test1)


# # Without Using development set to optimize my test accuracy is 99.5%

# In[ ]:


clf.score(x_test1,y_test1)


# # Using Development Set to see if i can improve my model accuracy

# In[ ]:


learning=[0.1,0.5,0.01,0.05,0.001]
for i in range(0,5):
    clf1=XGBClassifier(random_state=1,learning_rate=learning[i])
    clf1.fit(x_train,y_train)
    print(clf.score(x_dev,y_dev))


# In[ ]:


learning=[0.1,0.5,0.01,0.05,0.001]
for i in range(0,5):
    clf1=XGBClassifier(random_state=1,learning_rate=learning[i],booster='gblinear')
    clf1.fit(x_train,y_train)
    print(clf.score(x_dev,y_dev))


# In[ ]:


print('Test accuracy score')
clf.score(x_test,y_test)


# # Our XGBClassifier is giving 99.5% accuracy in every case of learning_rate and using both algo of xgtree as well as xglinear

# In[ ]:


print('Train_accuracy_score')
clf.score(x_train,y_train)


# In[ ]:


x_train


# In[ ]:


columns=x_test.columns


# In[ ]:


import matplotlib.pyplot as plt


# In[ ]:


x=x_test.values
y=y_test.values


# In[ ]:


for i in range(0,33):
    plt.scatter(x[0:1000,i],y[0:1000,])
    plt.xlabel(columns[i])
    plt.ylabel("y")
    plt.show()
    plt.scatter(x[2000:3000,i],y[2000:3000,])
    plt.xlabel(columns[i])
    plt.ylabel("y")
    plt.show()
    plt.scatter(x[3000:4000,i],y[3000:4000,])
    plt.xlabel(columns[i])
    plt.ylabel("y")
    plt.show()


# In[ ]:




