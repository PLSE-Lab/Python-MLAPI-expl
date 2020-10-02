# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

loan = pd.read_csv('../input/loan.csv', low_memory=False)

#This script checks out which features have missing values and some ways to impute them.

#First feature with missing values is emp_title.
#I believe a good way to impute these values is with "Job title not given"
loan['emp_title'] = np.where(loan['emp_title'].isnull(), 'Job title not given', loan['emp_title'])
    
#annual_inc has only 4 missing observations so I will do a median value imputation with this feature.
loan['annual_inc'] = loan['annual_inc'].fillna(loan['annual_inc'].median())

#For title, i will impute 'title not given' since there are so few missing.
loan['title'] = np.where(loan['title'].isnull(), 0, loan['title'])

#delinq_2yrs has 29 missing observations and I think we can replace those with zero, giving lendors the benefit of the doubt they wouldn't forget someone deliquent.
loan['delinq_2yrs'] = np.where(loan['delinq_2yrs'].isnull(), 0, loan['delinq_2yrs'])

#inq_last_6mths will be fixed in a similar manner.
loan['inq_last_6mths'] = np.where(loan['inq_last_6mths'].isnull(), 0, loan['inq_last_6mths'])

#mths_since_last_delinq missing values might need to be changed to a really high number.
#the reason for this is that we need to punish borrowers with small numbers in this feature.
#Missing values most likely mean the borrower has no delinquencies.

loan['mths_since_last_delinq'].max() #max is 188 so this will be our imputed value.
loan['mths_since_last_delinq'] = np.where(loan['mths_since_last_delinq'].isnull(), 188, loan['mths_since_last_delinq'])

#open_acc missing values will be imputed with 0.
loan['open_acc'] = np.where(loan['open_acc'].isnull(), 0, loan['open_acc'])

#pub_rec missing values will be replaced with 0.
loan['pub_rec'] = np.where(loan['pub_rec'].isnull(), 0, loan['pub_rec'])

#Revol_util will involve a median value imputation.
loan['revol_util'] = loan['revol_util'].fillna(loan['revol_util'].median())

#total_acc missing values will be replaced with 0.
loan['total_acc'] = np.where(loan['total_acc'].isnull(), 0, loan['total_acc'])

#collections_12_mths_ex_med missing valued will be replaced with 0.
loan['collections_12_mths_ex_med'] = np.where(loan['collections_12_mths_ex_med'].isnull(), 0, loan['collections_12_mths_ex_med'])

#mths_since_last_major_derog will be changed to a new variable where missing values = 0 for no derogs and non-missing = 1 for atleast 1 derog.
#feature will be named 90day_worse_rating.
loan['90day_worse_rating'] = np.where(loan['mths_since_last_major_derog'].isnull(), 0, 1)


#acc_now_delinq missing values will be replaced with 0.
loan['acc_now_delinq'] = np.where(loan['acc_now_delinq'].isnull(), 0, loan['acc_now_delinq'])

#tot_coll_amt will involve a median value imputation. I think regression imputation might work better but since this is initial stuff we will keep it simple.
loan['tot_coll_amt'] = loan['tot_coll_amt'].fillna(loan['tot_coll_amt'].median())

#tot_cur_bal will be fixed in similar manner.
loan['tot_cur_bal'] = loan['tot_cur_bal'].fillna(loan['tot_cur_bal'].median())

#total_rev_hi_lim will also contain median imputation
loan['total_rev_hi_lim'] = loan['total_rev_hi_lim'].fillna(loan['total_rev_hi_lim'].median())

#features below are being dropped due to their significantly high proportion of missing values or they are date values.

loan = loan.drop(['earliest_cr_line', 'last_pymnt_d', 'next_pymnt_d', 'last_credit_pull_d', 'annual_inc_joint', 'dti_joint', 'verification_status_joint', 'open_acc_6m', 'open_il_6m', 'open_il_12m', 'open_il_24m', 'mths_since_rcnt_il', 'total_bal_il', 'il_util', 'open_rv_24m', 'open_rv_12m', 'max_bal_bc', 'all_util', 'inq_fi', 'total_cu_tl', 'inq_last_12m', 'desc', 'mths_since_last_record', 'mths_since_last_major_derog'], axis=1)

#All suggestions of code or missing value imputations are welcomed! 