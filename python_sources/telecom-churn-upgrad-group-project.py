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
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# #### Importing the Dataset

# In[ ]:



telecom_data = pd.read_csv("../input/telecom_churn_data.csv")


# #### Intial Data Analysis

# In[ ]:


telecom_data.describe()


# In[ ]:


list(telecom_data.columns)


# In[ ]:


pd.set_option('display.max_columns', None)
telecom_data.head()


# #### Scrolled through all the columns and their values and tried to understand the meaning of each column with the help of the excel sheet which has all the meaings of the acronyms used in column names.

# In[ ]:


telecom_data.info(verbose=True, null_counts=True)


# In[ ]:


telecom_data.mobile_number	.nunique()


# #### mobile_number is  the unique column and is not duplicated in the data frame, which is like the primary key of this dataset

# #### Before moving on to further steps like null value analysis, deriving of the key features etc, lets first filter only the required high value customers for the analysis, as mentioned in the problem statement.

# #### High-value customers : Those who have recharged with an amount more than or equal to X, where X is the 70th percentile of the average recharge amount in the first two months (the good phase).
# #### After filtering the high-value customers, you should get about 29.9k rows

# #### We will use below 2 columns to get this info, last 2 columns contain null values which we will impute with 0.
# 
# total_rech_amt_6            99999 non-null int64
# 
# total_rech_amt_7            99999 non-null int64
# 
# total_rech_amt_8            99999 non-null int64
# 
# total_rech_amt_9            99999 non-null int64
# 
# total_rech_data_6           25153 non-null float64
# 
# total_rech_data_7           25571 non-null float64

# In[ ]:


#avg_first_2_months_rchg = (sum(telecom_data.total_rech_amt_6)+sum(telecom_data.total_rech_amt_7))/(len(telecom_data)*2)
telecom_data['avg_first_2_months_rchg'] = (telecom_data.total_rech_amt_6 + telecom_data.total_rech_amt_7 + telecom_data.total_rech_data_6.fillna(0) + telecom_data.total_rech_data_7.fillna(0))/4


# In[ ]:


#temp_df = telecom_data.loc[(telecom_data.total_rech_amt_6 + telecom_data.total_rech_amt_7) <= avg_first_2_months_rchg, ['total_rech_amt_6','total_rech_amt_7']]


# In[ ]:


#X_70th_percentile = np.percentile(temp_df.total_rech_amt_6 + temp_df.total_rech_amt_7, 70)
X_70th_percentile = np.percentile(telecom_data.avg_first_2_months_rchg, 70)


# In[ ]:


X_70th_percentile


# In[ ]:


#telecom_data_hvc =  telecom_data.loc[(telecom_data.total_rech_amt_6 + telecom_data.total_rech_amt_7 + telecom_data.total_rech_amt_8 + telecom_data.total_rech_amt_9) >= X_70th_percentile]
telecom_data_hvc =  telecom_data.loc[telecom_data.avg_first_2_months_rchg >= X_70th_percentile]


# In[ ]:


len(telecom_data_hvc)


# #### If we take > X_70th_percentile, it is giving 29991, which is what is mentioned in the problem statement(about 29.9k rows).
# #### But since problem statement also mentiones >= X, we will go by that which gives a litle higher i.e. 30019 rows.

# In[ ]:


telecom_data_hvc.info(verbose=True, null_counts=True)


# #### Tagging of churners needs to be done based on following columns
# total_ic_mou_9
# 
# total_og_mou_9
# 
# vol_2g_mb_9
# 
# vol_3g_mb_9
# 
# #### Just confirmed that these columns dont have any null values

# In[ ]:


telecom_data_hvc['Churned'] =  telecom_data_hvc.total_ic_mou_9 + telecom_data_hvc.total_og_mou_9 + telecom_data_hvc.vol_2g_mb_9 + telecom_data_hvc.vol_3g_mb_9


# In[ ]:


telecom_data_hvc.Churned = telecom_data_hvc.Churned.apply(lambda x: 1 if x==0 else 0 )


# #### Checking % of churns against nonchurns

# In[ ]:


telecom_data_hvc.Churned.value_counts()/len(telecom_data_hvc)*100


# #### So overall there are around 8.6 % Churners and around 91.3% Non Churners

# In[ ]:


telecom_data_hvc.Churned.value_counts()


# #### In terms of no of customers, 2590 mout ofm the 30019 high mmvalue customers have churned i.e switched to avail services of another telecom company.

# #### This is the actual Churned Customers column. Now we need to drop all the columns related to month 9 i.e. September. 
# #### And based on first 3 months columns, we need to predict the churned customers.

# #### Fetching all the _9 columns

# In[ ]:


drop_cols = list(telecom_data_hvc.filter(like='_9').columns)
telecom_data_hvc.drop(drop_cols, inplace=True, axis=1)


# In[ ]:


telecom_data_hvc.columns


# In[ ]:


telecom_data_hvc.info(verbose=True, null_counts=True)


# #### As this is the final dataset, now we will target null values.

# #### Before that, we will check there are any columns having only 1 unique value, and drop such columns, as they wont add any value to our analysis.

# In[ ]:


pd.set_option('display.max_rows', None)
pd.DataFrame(telecom_data_hvc.nunique())
#telecom_data_hvc.filter(telecom_data_hvc.nunique()==1).columns


# In[ ]:


pd.set_option('display.max_rows', 60)


# #### Dropping the ones that have a single unique value first.

# In[ ]:


telecom_data_hvc.drop(['circle_id','loc_og_t2o_mou','std_og_t2o_mou','loc_ic_t2o_mou','last_date_of_month_6','last_date_of_month_7','last_date_of_month_8','std_og_t2c_mou_6','std_og_t2c_mou_7','std_og_t2c_mou_8','std_ic_t2o_mou_6','std_ic_t2o_mou_7','std_ic_t2o_mou_8'],inplace=True,axis=1),


# In[ ]:


pd.set_option('display.max_rows', None)
pd.DataFrame(telecom_data_hvc.nunique())
#telecom_data_hvc.filter(telecom_data_hvc.nunique()==1).columns


# In[ ]:


pd.set_option('display.max_rows', 60)


# #### Now we will check for null/missing values in columns.

# In[ ]:


# Checking the percentage of missing values
col_list = telecom_data_hvc.columns

for col_name in telecom_data_hvc.columns:
    missing_percent = round(100* ((telecom_data_hvc[col_name].isnull()) | (telecom_data_hvc[col_name].astype(str) == 'Select')).sum() /len(telecom_data_hvc.index) , 2)
    print(col_name + " - " + str(missing_percent))


# #### We will drop the columns having more than 60% missing values, imputing them might introduce bias towards the values with which they are updated.
# 
# #### The columns having more than 60% missing values are as below:
# 
# date_of_last_rech_data_6 - 61.88
# 
# date_of_last_rech_data_7 - 60.99
# 
# date_of_last_rech_data_8 - 60.72
# 
# total_rech_data_6 - 61.88
# 
# total_rech_data_7 - 60.99
# 
# total_rech_data_8 - 60.72
# 
# max_rech_data_6 - 61.88
# 
# max_rech_data_7 - 60.99
# 
# max_rech_data_8 - 60.72
# 
# count_rech_2g_6 - 61.88
# 
# count_rech_2g_7 - 60.99
# 
# count_rech_2g_8 - 60.72
# 
# count_rech_3g_6 - 61.88
# 
# count_rech_3g_7 - 60.99
# 
# count_rech_3g_8 - 60.72
# 
# av_rech_amt_data_6 - 61.88
# 
# av_rech_amt_data_7 - 60.99
# 
# av_rech_amt_data_8 - 60.72
# 
# arpu_3g_6 - 61.88
# 
# arpu_3g_7 - 60.99
# 
# arpu_3g_8 - 60.72
# 
# arpu_2g_6 - 61.88
# 
# arpu_2g_7 - 60.99
# 
# arpu_2g_8 - 60.72
# 
# night_pck_user_6 - 61.88
# 
# night_pck_user_7 - 60.99
# 
# night_pck_user_8 - 60.72
# 
# fb_user_6 - 61.88
# 
# fb_user_7 - 60.99
# 
# fb_user_8 - 60.72
# 

# In[ ]:


telecom_data_hvc.drop(['date_of_last_rech_data_6','date_of_last_rech_data_7','date_of_last_rech_data_8','total_rech_data_6','total_rech_data_7',
 'total_rech_data_8','max_rech_data_6','max_rech_data_7','max_rech_data_8','count_rech_2g_6','count_rech_2g_7',
 'count_rech_2g_8','count_rech_3g_6','count_rech_3g_7','count_rech_3g_8','av_rech_amt_data_6','av_rech_amt_data_7',
 'av_rech_amt_data_8','arpu_3g_6','arpu_3g_7','arpu_3g_8','arpu_2g_6','arpu_2g_7','arpu_2g_8','night_pck_user_6',
 'night_pck_user_7','night_pck_user_8','fb_user_6','fb_user_7','fb_user_8'],inplace=True,axis=1)


# #### Checking the percentage of missing values again after the deletion of columns.

# In[ ]:


# Checking the percentage of missing values
col_list = telecom_data_hvc.columns

for col_name in telecom_data_hvc.columns:
    missing_percent = round(100* ((telecom_data_hvc[col_name].isnull()) | (telecom_data_hvc[col_name].astype(str) == 'Select')).sum() /len(telecom_data_hvc.index) , 2)
    print(col_name + " - " + str(missing_percent))


# Now there remain only columns with upto 3.5 % missing vaues, we will delete the rows having missing values for these columns.

# In[ ]:


telecom_data_hvc = telecom_data_hvc[~np.isnan(telecom_data_hvc['onnet_mou_6'])]
telecom_data_hvc = telecom_data_hvc[~np.isnan(telecom_data_hvc['onnet_mou_7'])]
telecom_data_hvc = telecom_data_hvc[~np.isnan(telecom_data_hvc['onnet_mou_8'])]
telecom_data_hvc = telecom_data_hvc[~np.isnan(telecom_data_hvc['offnet_mou_6'])]
telecom_data_hvc = telecom_data_hvc[~np.isnan(telecom_data_hvc['offnet_mou_7'])]
telecom_data_hvc = telecom_data_hvc[~np.isnan(telecom_data_hvc['offnet_mou_8'])]
telecom_data_hvc = telecom_data_hvc[~np.isnan(telecom_data_hvc['roam_ic_mou_6'])]
telecom_data_hvc = telecom_data_hvc[~np.isnan(telecom_data_hvc['roam_ic_mou_7'])]
telecom_data_hvc = telecom_data_hvc[~np.isnan(telecom_data_hvc['roam_ic_mou_8'])]
telecom_data_hvc = telecom_data_hvc[~np.isnan(telecom_data_hvc['roam_og_mou_6'])]
telecom_data_hvc = telecom_data_hvc[~np.isnan(telecom_data_hvc['roam_og_mou_7'])]
telecom_data_hvc = telecom_data_hvc[~np.isnan(telecom_data_hvc['roam_og_mou_8'])]
telecom_data_hvc = telecom_data_hvc[~np.isnan(telecom_data_hvc['loc_og_t2t_mou_6'])]
telecom_data_hvc = telecom_data_hvc[~np.isnan(telecom_data_hvc['loc_og_t2t_mou_7'])]
telecom_data_hvc = telecom_data_hvc[~np.isnan(telecom_data_hvc['loc_og_t2t_mou_8'])]
telecom_data_hvc = telecom_data_hvc[~np.isnan(telecom_data_hvc['loc_og_t2m_mou_6'])]
telecom_data_hvc = telecom_data_hvc[~np.isnan(telecom_data_hvc['loc_og_t2m_mou_7'])]
telecom_data_hvc = telecom_data_hvc[~np.isnan(telecom_data_hvc['loc_og_t2m_mou_8'])]
telecom_data_hvc = telecom_data_hvc[~np.isnan(telecom_data_hvc['loc_og_t2f_mou_6'])]
telecom_data_hvc = telecom_data_hvc[~np.isnan(telecom_data_hvc['loc_og_t2f_mou_7'])]
telecom_data_hvc = telecom_data_hvc[~np.isnan(telecom_data_hvc['loc_og_t2f_mou_8'])]
telecom_data_hvc = telecom_data_hvc[~np.isnan(telecom_data_hvc['loc_og_t2c_mou_6'])]
telecom_data_hvc = telecom_data_hvc[~np.isnan(telecom_data_hvc['loc_og_t2c_mou_7'])]
telecom_data_hvc = telecom_data_hvc[~np.isnan(telecom_data_hvc['loc_og_t2c_mou_8'])]
telecom_data_hvc = telecom_data_hvc[~np.isnan(telecom_data_hvc['loc_og_mou_6'])]
telecom_data_hvc = telecom_data_hvc[~np.isnan(telecom_data_hvc['loc_og_mou_7'])]
telecom_data_hvc = telecom_data_hvc[~np.isnan(telecom_data_hvc['loc_og_mou_8'])]
telecom_data_hvc = telecom_data_hvc[~np.isnan(telecom_data_hvc['std_og_t2t_mou_6'])]
telecom_data_hvc = telecom_data_hvc[~np.isnan(telecom_data_hvc['std_og_t2t_mou_7'])]
telecom_data_hvc = telecom_data_hvc[~np.isnan(telecom_data_hvc['std_og_t2t_mou_8'])]
telecom_data_hvc = telecom_data_hvc[~np.isnan(telecom_data_hvc['std_og_t2m_mou_6'])]
telecom_data_hvc = telecom_data_hvc[~np.isnan(telecom_data_hvc['std_og_t2m_mou_7'])]
telecom_data_hvc = telecom_data_hvc[~np.isnan(telecom_data_hvc['std_og_t2m_mou_8'])]
telecom_data_hvc = telecom_data_hvc[~np.isnan(telecom_data_hvc['std_og_t2f_mou_6'])]
telecom_data_hvc = telecom_data_hvc[~np.isnan(telecom_data_hvc['std_og_t2f_mou_7'])]
telecom_data_hvc = telecom_data_hvc[~np.isnan(telecom_data_hvc['std_og_t2f_mou_8'])]
telecom_data_hvc = telecom_data_hvc[~np.isnan(telecom_data_hvc['std_og_mou_6'])]
telecom_data_hvc = telecom_data_hvc[~np.isnan(telecom_data_hvc['std_og_mou_7'])]
telecom_data_hvc = telecom_data_hvc[~np.isnan(telecom_data_hvc['std_og_mou_8'])]
telecom_data_hvc = telecom_data_hvc[~np.isnan(telecom_data_hvc['isd_og_mou_6'])]
telecom_data_hvc = telecom_data_hvc[~np.isnan(telecom_data_hvc['isd_og_mou_7'])]
telecom_data_hvc = telecom_data_hvc[~np.isnan(telecom_data_hvc['isd_og_mou_8'])]
telecom_data_hvc = telecom_data_hvc[~np.isnan(telecom_data_hvc['spl_og_mou_6'])]
telecom_data_hvc = telecom_data_hvc[~np.isnan(telecom_data_hvc['spl_og_mou_7'])]
telecom_data_hvc = telecom_data_hvc[~np.isnan(telecom_data_hvc['spl_og_mou_8'])]
telecom_data_hvc = telecom_data_hvc[~np.isnan(telecom_data_hvc['og_others_6'])]
telecom_data_hvc = telecom_data_hvc[~np.isnan(telecom_data_hvc['og_others_7'])]
telecom_data_hvc = telecom_data_hvc[~np.isnan(telecom_data_hvc['og_others_8'])]
telecom_data_hvc = telecom_data_hvc[~np.isnan(telecom_data_hvc['loc_ic_t2t_mou_6'])]
telecom_data_hvc = telecom_data_hvc[~np.isnan(telecom_data_hvc['loc_ic_t2t_mou_7'])]
telecom_data_hvc = telecom_data_hvc[~np.isnan(telecom_data_hvc['loc_ic_t2t_mou_8'])]
telecom_data_hvc = telecom_data_hvc[~np.isnan(telecom_data_hvc['loc_ic_t2m_mou_6'])]
telecom_data_hvc = telecom_data_hvc[~np.isnan(telecom_data_hvc['loc_ic_t2m_mou_7'])]
telecom_data_hvc = telecom_data_hvc[~np.isnan(telecom_data_hvc['loc_ic_t2m_mou_8'])]
telecom_data_hvc = telecom_data_hvc[~np.isnan(telecom_data_hvc['loc_ic_t2f_mou_6'])]
telecom_data_hvc = telecom_data_hvc[~np.isnan(telecom_data_hvc['loc_ic_t2f_mou_7'])]
telecom_data_hvc = telecom_data_hvc[~np.isnan(telecom_data_hvc['loc_ic_t2f_mou_8'])]
telecom_data_hvc = telecom_data_hvc[~np.isnan(telecom_data_hvc['loc_ic_mou_6'])]
telecom_data_hvc = telecom_data_hvc[~np.isnan(telecom_data_hvc['loc_ic_mou_7'])]
telecom_data_hvc = telecom_data_hvc[~np.isnan(telecom_data_hvc['loc_ic_mou_8'])]
telecom_data_hvc = telecom_data_hvc[~np.isnan(telecom_data_hvc['std_ic_t2t_mou_6'])]
telecom_data_hvc = telecom_data_hvc[~np.isnan(telecom_data_hvc['std_ic_t2t_mou_7'])]
telecom_data_hvc = telecom_data_hvc[~np.isnan(telecom_data_hvc['std_ic_t2t_mou_8'])]
telecom_data_hvc = telecom_data_hvc[~np.isnan(telecom_data_hvc['std_ic_t2m_mou_6'])]
telecom_data_hvc = telecom_data_hvc[~np.isnan(telecom_data_hvc['std_ic_t2m_mou_7'])]
telecom_data_hvc = telecom_data_hvc[~np.isnan(telecom_data_hvc['std_ic_t2m_mou_8'])]
telecom_data_hvc = telecom_data_hvc[~np.isnan(telecom_data_hvc['std_ic_t2f_mou_6'])]
telecom_data_hvc = telecom_data_hvc[~np.isnan(telecom_data_hvc['std_ic_t2f_mou_7'])]
telecom_data_hvc = telecom_data_hvc[~np.isnan(telecom_data_hvc['std_ic_t2f_mou_8'])]
telecom_data_hvc = telecom_data_hvc[~np.isnan(telecom_data_hvc['std_ic_mou_6'])]
telecom_data_hvc = telecom_data_hvc[~np.isnan(telecom_data_hvc['std_ic_mou_7'])]
telecom_data_hvc = telecom_data_hvc[~np.isnan(telecom_data_hvc['std_ic_mou_8'])]
telecom_data_hvc = telecom_data_hvc[~np.isnan(telecom_data_hvc['spl_ic_mou_6'])]
telecom_data_hvc = telecom_data_hvc[~np.isnan(telecom_data_hvc['spl_ic_mou_7'])]
telecom_data_hvc = telecom_data_hvc[~np.isnan(telecom_data_hvc['spl_ic_mou_8'])]
telecom_data_hvc = telecom_data_hvc[~np.isnan(telecom_data_hvc['isd_ic_mou_6'])]
telecom_data_hvc = telecom_data_hvc[~np.isnan(telecom_data_hvc['isd_ic_mou_7'])]
telecom_data_hvc = telecom_data_hvc[~np.isnan(telecom_data_hvc['isd_ic_mou_8'])]
telecom_data_hvc = telecom_data_hvc[~np.isnan(telecom_data_hvc['ic_others_6'])]
telecom_data_hvc = telecom_data_hvc[~np.isnan(telecom_data_hvc['ic_others_7'])]
telecom_data_hvc = telecom_data_hvc[~np.isnan(telecom_data_hvc['ic_others_8'])]
telecom_data_hvc = telecom_data_hvc[~telecom_data_hvc.date_of_last_rech_6.isnull()]
telecom_data_hvc = telecom_data_hvc[~telecom_data_hvc.date_of_last_rech_7.isnull()]
telecom_data_hvc = telecom_data_hvc[~telecom_data_hvc.date_of_last_rech_8.isnull()]


# In[ ]:


# Checking the percentage of missing values
col_list = telecom_data_hvc.columns

for col_name in telecom_data_hvc.columns:
    missing_percent = round(100* ((telecom_data_hvc[col_name].isnull()) | (telecom_data_hvc[col_name].astype(str) == 'Select')).sum() /len(telecom_data_hvc.index) , 2)
    print(col_name + " - " + str(missing_percent))


# In[ ]:


len(telecom_data_hvc)


# In[ ]:


(30019-28493)*100/30019


# #### From 30019 rows we have reduced to 28493 during null  value treatment.
# 
# #### We have deleted around 5.08% rows of high value customers.

# In[ ]:


telecom_data_hvc.Churned.value_counts()/len(telecom_data_hvc)*100


# #### After doing the null value treatment, the Churned Customers % has reduced from 8.6% to 6.2%.

# #### We will create a copy of this dataframe and do further Outlier treatment as PCA would get effected by the presence of outliers.
# #### For the prediction part we are doing PCA, for the 2nd part wherein  we need to find the driver variables for Chured, we are not doing PCA, and would use tree cased model so outlier treatment will not be required in that case.

# In[ ]:


telecom_data_hvc_for_pca =  telecom_data_hvc


# In[ ]:


list(telecom_data_hvc_for_pca.columns)


# #### Since these are a lot many columns we will go for a bulk outlier treatment. :-)

# In[ ]:


telecom_data_hvc_for_pca_num_cols = telecom_data_hvc_for_pca[['arpu_6',
 'arpu_7',
 'arpu_8',
 'onnet_mou_6',
 'onnet_mou_7',
 'onnet_mou_8',
 'offnet_mou_6',
 'offnet_mou_7',
 'offnet_mou_8',
 'roam_ic_mou_6',
 'roam_ic_mou_7',
 'roam_ic_mou_8',
 'roam_og_mou_6',
 'roam_og_mou_7',
 'roam_og_mou_8',
 'loc_og_t2t_mou_6',
 'loc_og_t2t_mou_7',
 'loc_og_t2t_mou_8',
 'loc_og_t2m_mou_6',
 'loc_og_t2m_mou_7',
 'loc_og_t2m_mou_8',
 'loc_og_t2f_mou_6',
 'loc_og_t2f_mou_7',
 'loc_og_t2f_mou_8',
 'loc_og_t2c_mou_6',
 'loc_og_t2c_mou_7',
 'loc_og_t2c_mou_8',
 'loc_og_mou_6',
 'loc_og_mou_7',
 'loc_og_mou_8',
 'std_og_t2t_mou_6',
 'std_og_t2t_mou_7',
 'std_og_t2t_mou_8',
 'std_og_t2m_mou_6',
 'std_og_t2m_mou_7',
 'std_og_t2m_mou_8',
 'std_og_t2f_mou_6',
 'std_og_t2f_mou_7',
 'std_og_t2f_mou_8',
 'std_og_mou_6',
 'std_og_mou_7',
 'std_og_mou_8',
 'isd_og_mou_6',
 'isd_og_mou_7',
 'isd_og_mou_8',
 'spl_og_mou_6',
 'spl_og_mou_7',
 'spl_og_mou_8',
 'og_others_6',
 'og_others_7',
 'og_others_8',
 'total_og_mou_6',
 'total_og_mou_7',
 'total_og_mou_8',
 'loc_ic_t2t_mou_6',
 'loc_ic_t2t_mou_7',
 'loc_ic_t2t_mou_8',
 'loc_ic_t2m_mou_6',
 'loc_ic_t2m_mou_7',
 'loc_ic_t2m_mou_8',
 'loc_ic_t2f_mou_6',
 'loc_ic_t2f_mou_7',
 'loc_ic_t2f_mou_8',
 'loc_ic_mou_6',
 'loc_ic_mou_7',
 'loc_ic_mou_8',
 'std_ic_t2t_mou_6',
 'std_ic_t2t_mou_7',
 'std_ic_t2t_mou_8',
 'std_ic_t2m_mou_6',
 'std_ic_t2m_mou_7',
 'std_ic_t2m_mou_8',
 'std_ic_t2f_mou_6',
 'std_ic_t2f_mou_7',
 'std_ic_t2f_mou_8',
 'std_ic_mou_6',
 'std_ic_mou_7',
 'std_ic_mou_8',
 'total_ic_mou_6',
 'total_ic_mou_7',
 'total_ic_mou_8',
 'spl_ic_mou_6',
 'spl_ic_mou_7',
 'spl_ic_mou_8',
 'isd_ic_mou_6',
 'isd_ic_mou_7',
 'isd_ic_mou_8',
 'ic_others_6',
 'ic_others_7',
 'ic_others_8',
 'total_rech_num_6',
 'total_rech_num_7',
 'total_rech_num_8',
 'total_rech_amt_6',
 'total_rech_amt_7',
 'total_rech_amt_8',
 'max_rech_amt_6',
 'max_rech_amt_7',
 'max_rech_amt_8',
 'last_day_rch_amt_6',
 'last_day_rch_amt_7',
 'last_day_rch_amt_8',
 'vol_2g_mb_6',
 'vol_2g_mb_7',
 'vol_2g_mb_8',
 'vol_3g_mb_6',
 'vol_3g_mb_7',
 'vol_3g_mb_8',
 'aon',
 'aug_vbc_3g',
 'jul_vbc_3g',
 'jun_vbc_3g',
 'sep_vbc_3g',
 'avg_first_2_months_rchg']]


# ### Outlier rows Removal

# In[ ]:


from scipy import stats
z = np.abs(stats.zscore(telecom_data_hvc_for_pca_num_cols))
print(z)


# In[ ]:


threshold = 3
print(np.where(z > 3))


# In[ ]:


telecom_data_hvc_for_pca_num_cols_o = telecom_data_hvc_for_pca_num_cols[(z >= 3).all(axis=1)]
telecom_data_hvc_for_pca_num_cols = telecom_data_hvc_for_pca_num_cols[(z < 3).all(axis=1)]


# In[ ]:


len(telecom_data_hvc_for_pca_num_cols)


# In[ ]:


len(telecom_data_hvc_for_pca)


# In[ ]:


(30019-15084)*100/30019


# In[ ]:


telecom_data_hvc_for_pca.columns


# In[ ]:


#telecom_data_hvc_for_pca.join(telecom_data_hvc_for_pca_num_cols, how='inner')
#col_lst = telecom_data_hvc_for_pca.columns
telecom_data_hvc_for_pca = pd.concat([telecom_data_hvc_for_pca, telecom_data_hvc_for_pca_num_cols],axis=1, join = 'inner')


# #### Deleting the duplicate columns formed due to concat

# In[ ]:


telecom_data_hvc_for_pca = telecom_data_hvc_for_pca.loc[:,~telecom_data_hvc_for_pca.columns.duplicated()]


# In[ ]:


list(telecom_data_hvc_for_pca.columns)


# In[ ]:


len(telecom_data_hvc_for_pca_num_cols)


# In[ ]:


len(telecom_data_hvc_for_pca)


# In[ ]:


telecom_data_hvc_for_pca_num_cols.head()


# In[ ]:


telecom_data_hvc_for_pca.head()


# In[ ]:


telecom_data_hvc.Churned.value_counts()/len(telecom_data_hvc)*100


#  #### Since there is a lot of class imbalance with Churned Customers comprising only 6.13% of total customers, We will use SMOTE which is an oversampling technique.
#  What smote does is simple. First it finds the n-nearest neighbors in the minority class for each of the samples in the class . Then it draws a line between the the neighbors an generates random points on the lines.
#  
#  Referred https://medium.com/coinmonks/smote-and-adasyn-handling-imbalanced-data-set-34f5223e167 

# In[ ]:


import datetime
from dateutil.parser import parse
def dt2epoch(value):
    d = parse(value)
    epoch = (d - datetime.datetime(1970,1,1)).total_seconds()
    return epoch 

telecom_data_hvc_for_pca.date_of_last_rech_6=telecom_data_hvc_for_pca.date_of_last_rech_6.astype('|S')
telecom_data_hvc_for_pca.date_of_last_rech_6 = telecom_data_hvc_for_pca.date_of_last_rech_6.apply(dt2epoch)
telecom_data_hvc_for_pca.date_of_last_rech_7=telecom_data_hvc_for_pca.date_of_last_rech_7.astype('|S')
telecom_data_hvc_for_pca.date_of_last_rech_7 = telecom_data_hvc_for_pca.date_of_last_rech_7.apply(dt2epoch)
telecom_data_hvc_for_pca.date_of_last_rech_8=telecom_data_hvc_for_pca.date_of_last_rech_8.astype('|S')
telecom_data_hvc_for_pca.date_of_last_rech_8 = telecom_data_hvc_for_pca.date_of_last_rech_8.apply(dt2epoch)


# In[ ]:


X = telecom_data_hvc_for_pca.drop(['mobile_number','Churned'], axis=1)
y = telecom_data_hvc_for_pca['Churned']


# In[ ]:


col_lst = X.columns


# In[ ]:


from imblearn.over_sampling import SMOTE
sm = SMOTE()
X, y = sm.fit_sample(X, y)


# In[ ]:


X.shape


# In[ ]:


y.shape


# In[ ]:


y= pd.DataFrame(y,columns=['Churned'])
y.Churned.value_counts()


# **So this has taken care of the Class Imbalance.

# In[ ]:


X = pd.DataFrame(X, columns = col_lst)


# In[ ]:


X.head()


# Since we will perform PCA on this data, we will check for colinearity between independent variables before and after PCA.

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
# Let's see the correlation matrix 
plt.figure(figsize = (100,100))     # Size of the figure
sns.heatmap(X.corr(),annot = True)


# Since there are a large no of columns, the image is not very redeable but we do see some lighter patches indicating colinearity.

# In[ ]:


#creating correlation matrix for the given data
corrmat = np.corrcoef(X.transpose())


# In[ ]:


#Make a diagonal matrix with diagonal entry of Matrix corrmat
p=np.diagflat(corrmat.diagonal())


# In[ ]:


# subtract diagonal entries making all diagonals 0
corrmat_diag_zero = corrmat - p
print("max corr:",corrmat_diag_zero.max(), ", min corr: ", corrmat_diag_zero.min(),)


# #### Standardizing the numerical data with StandardScaler as a preparation step for performing PCA

# In[ ]:


from sklearn.preprocessing import StandardScaler
standard_scaler = StandardScaler()
X = standard_scaler.fit_transform(X)


# #### Performing PCA on the dataframe containing standardized numerical variables.

# In[ ]:


from sklearn.decomposition import PCA


# In[ ]:


pca = PCA(random_state=100)


# In[ ]:


telecom_data_hvc_pca = pca.fit_transform(X)


# In[ ]:


telecom_data_hvc_pca_df = pd.DataFrame(telecom_data_hvc_pca)


# In[ ]:


telecom_data_hvc_pca_df.shape


# In[ ]:


X.shape


# In[ ]:


telecom_data_hvc_pca_df.columns = ['PC1','PC2','PC3','PC4','PC5','PC6','PC7','PC8','PC9','PC10','PC11','PC12','PC13','PC14','PC15','PC16','PC17','PC18','PC19','PC20','PC21','PC22','PC23','PC24','PC25','PC26','PC27','PC28','PC29','PC30','PC31','PC32','PC33','PC34','PC35','PC36','PC37','PC38','PC39','PC40','PC41','PC42','PC43','PC44','PC45','PC46','PC47','PC48','PC49','PC50','PC51','PC52','PC53','PC54','PC55','PC56','PC57','PC58','PC59','PC60','PC61','PC62','PC63','PC64','PC65','PC66','PC67','PC68','PC69','PC70','PC71','PC72','PC73','PC74','PC75','PC76','PC77','PC78','PC79','PC80','PC81','PC82','PC83','PC84','PC85','PC86','PC87','PC88','PC89','PC90','PC91','PC92','PC93','PC94','PC95','PC96','PC97','PC98','PC99','PC100','PC101','PC102','PC103','PC104','PC105','PC106','PC107','PC108','PC109','PC110','PC111','PC112','PC113','PC114','PC115','PC116','PC117','PC118','PC119','PC120','PC121','PC122','PC123','PC124','PC125','PC126','PC127','PC128','PC129']


# #### There are 129 principal components formed as there were total 129 numerical variables on which PCA was applied.

# #### Checking for collinearity in the data after performing PCA.

# In[ ]:


#creating correlation matrix for the principal components
corrmat = np.corrcoef(telecom_data_hvc_pca_df.transpose())


# In[ ]:


#plotting the correlation matrix
get_ipython().run_line_magic('matplotlib', 'inline')
plt.figure(figsize = (20,10))
sns.heatmap(corrmat,annot = True)


# In[ ]:


# 1s -> 0s in diagonals
corrmat_nodiag = corrmat - np.diagflat(corrmat.diagonal())
print("max corr:",corrmat_nodiag.max(), ", min corr: ", corrmat_nodiag.min(),)
# we see that correlations are indeed very close to 0


# #### As we can see in the heat map and the values for min and max correlation coefficients values, the values are almost 0 which means there is no collinearity between the variables post PCA.

# #### Now comes the part to decide how many principal components will be required based on how much variance is needed to be described.

# In[ ]:


pca.explained_variance_


# In[ ]:


print("pca.explained_variance_ratio_: ",pca.explained_variance_ratio_.round(3)*100)


# In[ ]:


print (pca.explained_variance_ratio_.cumsum())


# #### So we have around 74 variables explaing 95.26074 % variace, we will go ahead with selectiing first 74 Principle Components.

# #### SCREE plot to support our decision below:

# In[ ]:


#Making the screeplot - plotting the cumulative variance against the number of components
get_ipython().run_line_magic('matplotlib', 'inline')
fig = plt.figure(figsize = (12,8))
sns.set(font_scale = 2)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')
plt.show()


# #### Since all the principal components are in orthogonal directions to each other, their dot products would be 0, the same is found below

# In[ ]:


product = np.dot(pca.components_[0],pca.components_[1])
product.round(5)


# In[ ]:


product = np.dot(pca.components_[1],pca.components_[12])
product.round(5)


# #### Plotting Scatterplots  between some principle components to see if there are visible differentiations.

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
fig = plt.figure(figsize = (8,8))
sns.set(font_scale = 1)
plt.scatter(telecom_data_hvc_pca_df.PC1, telecom_data_hvc_pca_df.PC2)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
fig = plt.figure(figsize = (8,8))
sns.set(font_scale = 1)
plt.scatter(telecom_data_hvc_pca_df.PC1, telecom_data_hvc_pca_df.PC2)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 3')


# In[ ]:


components = pd.DataFrame({'PC1':pca.components_[0],'PC2':pca.components_[1],'Feature':col_lst })
components


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
fig = plt.figure(figsize = (25,25))
sns.set(font_scale = 1)
plt.scatter(components.PC1, components.PC2)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
for i, txt in enumerate(components.Feature):
    plt.annotate(txt, (components.PC1[i],components.PC2[i]))
plt.tight_layout()
plt.show()


# #### We will perform Logistic Regression, SVM and Random Forests with CART decision trees and check which model provides the best results.

# #### Here it is more important to identify churners than the non-churners accurately.
# #### So Churners being our positive class, Non-Churners identified as Churners(False Positive FP) is atleast not as bad as identifying Churners as Non Churners(False Negative)
# #### So we need a metric which has FN in the denominator, so that it is as low as possible and our metric value becomes higher.
# #### Also True Positive is in the numerator so that Churners are identified accurately, so the metric to go for would be Sensitivity which is the same as Recall.
# #### As Sensitivity/Recall = TP / (TP+FN)

# ### Logistic Regression with GridSearchCV

# Since post PCA we have decided to preserve first 74 PCs, copying them into another dataframe.

# In[ ]:


X1 = pd.DataFrame(X).iloc[:,0:74]


# In[ ]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(X1,y,test_size=0.3)
y_train = np.array(y_train).astype(int)
y_test = np.array(y_test).astype(int)


# In[ ]:




# creating a KFold object with 5 splits 
folds = KFold(n_splits = 5, shuffle = True, random_state = 4)

# Create regularization penalty space
penalty = ['l1', 'l2']  # l1 lasso l2 ridge

# Create regularization hyperparameter space
C = np.logspace(-3,3,7) 

# Create hyperparameter options
hyperparameters = dict(C=C, penalty=penalty)

logistic=LogisticRegression()

# Create grid search using 5-fold cross validation
grid = GridSearchCV(logistic, hyperparameters, cv=folds, verbose=1, scoring='recall',n_jobs=-1)


# In[ ]:


grid_result = grid.fit(x_train, y_train.ravel())
# Summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))


# In[ ]:



# model with optimal hyperparameters

# model
model = LogisticRegression(C=0.001, penalty='l1')

model.fit(x_train, y_train.ravel())
y_pred = model.predict(x_test)


# In[ ]:


confusion = metrics.confusion_matrix(y_test, y_pred )
confusion


# In[ ]:


from sklearn.metrics import precision_score, recall_score
recall_score(y_test, y_pred)


# Logistic Regression gave us a Recall/Sensitivity score of 83.61 on test data.

# ### SVM 

# We will try each of linear, polynial and rbf SVMs to see which one gives the best score.

# #### Linear SVM with GridSearchCV to find best values of hyperparameters for a higher recall score

# After multiple runs, found that SVM is taking way too much time to fit the model, hence as a workaround will try deleting some data while maintaing class balance, do PCA again and then try.

# In[ ]:


telecom_data_hvc_for_pca.Churned.value_counts()


# In[ ]:


telecom_data_hvc_for_pca_nc_ld = telecom_data_hvc_for_pca[telecom_data_hvc_for_pca.Churned==0]
telecom_data_hvc_for_pca_nc_ld = telecom_data_hvc_for_pca_nc_ld.sample(n=4000)
telecom_data_hvc_for_pca_ld = pd.concat([telecom_data_hvc_for_pca_nc_ld, (telecom_data_hvc_for_pca[telecom_data_hvc_for_pca.Churned==1])], axis = 0)


# In[ ]:


X = telecom_data_hvc_for_pca_ld.drop(['mobile_number','Churned'], axis=1)
y = telecom_data_hvc_for_pca_ld['Churned']


# In[ ]:


col_lst = X.columns
from imblearn.over_sampling import SMOTE
sm = SMOTE()
X, y = sm.fit_sample(X, y)
y= pd.DataFrame(y,columns=['Churned'])
X = pd.DataFrame(X, columns = col_lst)


# In[ ]:


y.Churned.value_counts()


# In[ ]:


X.shape


# In[ ]:


y.shape


# In[ ]:


from sklearn.preprocessing import StandardScaler
standard_scaler = StandardScaler()
X = standard_scaler.fit_transform(X)


# In[ ]:


pca = PCA(random_state=100)
telecom_data_hvc_pca = pca.fit_transform(X)
telecom_data_hvc_pca_df = pd.DataFrame(telecom_data_hvc_pca)
telecom_data_hvc_pca_df.shape


# In[ ]:


telecom_data_hvc_pca_df.columns = ['PC1','PC2','PC3','PC4','PC5','PC6','PC7','PC8','PC9','PC10','PC11','PC12','PC13','PC14','PC15','PC16','PC17','PC18','PC19','PC20','PC21','PC22','PC23','PC24','PC25','PC26','PC27','PC28','PC29','PC30','PC31','PC32','PC33','PC34','PC35','PC36','PC37','PC38','PC39','PC40','PC41','PC42','PC43','PC44','PC45','PC46','PC47','PC48','PC49','PC50','PC51','PC52','PC53','PC54','PC55','PC56','PC57','PC58','PC59','PC60','PC61','PC62','PC63','PC64','PC65','PC66','PC67','PC68','PC69','PC70','PC71','PC72','PC73','PC74','PC75','PC76','PC77','PC78','PC79','PC80','PC81','PC82','PC83','PC84','PC85','PC86','PC87','PC88','PC89','PC90','PC91','PC92','PC93','PC94','PC95','PC96','PC97','PC98','PC99','PC100','PC101','PC102','PC103','PC104','PC105','PC106','PC107','PC108','PC109','PC110','PC111','PC112','PC113','PC114','PC115','PC116','PC117','PC118','PC119','PC120','PC121','PC122','PC123','PC124','PC125','PC126','PC127','PC128','PC129']

# 1s -> 0s in diagonals
corrmat_nodiag = corrmat - np.diagflat(corrmat.diagonal())
print("max corr:",corrmat_nodiag.max(), ", min corr: ", corrmat_nodiag.min(),)


# In[ ]:


print (pca.explained_variance_ratio_.cumsum())


# So again first 74 PCs explain 95.308896% variance and we will extract only those for our further analysis.

# In[ ]:


X1 = pd.DataFrame(X).iloc[:,0:74]
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(X1,y,test_size=0.3)
y_train = np.array(y_train).astype(int)
y_test = np.array(y_test).astype(int)


# In[ ]:


x_train.shape


# In[ ]:


x_test.shape


# In[ ]:


y_train.shape


# In[ ]:


y_test.shape


# In[ ]:


from sklearn.svm import SVC

# creating a KFold object with 5 splits 
folds = KFold(n_splits = 5, random_state = 4)

# specify range of parameters (C) as a list
params = {"C": [10, 100, 1000]}

model_linear = SVC(kernel='linear', cache_size=10000)

# set up grid search scheme
# note that we are still using the 5 fold CV scheme we set up earlier
model_cv = GridSearchCV(estimator = model_linear, 
                        param_grid = params, 
                        scoring= 'recall', 
                        cv = folds, 
                        verbose = 1,
                        return_train_score=True,
                        n_jobs=-1)      


# In[ ]:


# fit the model - it will fit 5 folds across all values of C
model_cv.fit(x_train, y_train.ravel())


# In[ ]:


# results of grid search CV
cv_results = pd.DataFrame(model_cv.cv_results_)   
cv_results
 


# In[ ]:


# plot of C versus train and test scores

plt.figure(figsize=(8, 6))
plt.plot(cv_results['param_C'], cv_results['mean_test_score'])
plt.plot(cv_results['param_C'], cv_results['mean_train_score'])
plt.xlabel('C')
plt.ylabel('Recall')
plt.legend(['test recall', 'train recall'], loc='upper left')
plt.xscale('log')


# In[ ]:


# printing the optimal accuracy score and hyperparameters
best_score = model_cv.best_score_
best_hyperparams = model_cv.best_params_

print("The best test score is {0} corresponding to hyperparameters {1}".format(best_score, best_hyperparams))


# In[ ]:


# model with optimal hyperparameters

# model
model = SVC(C=10,  kernel="linear")

model.fit(x_train, y_train.ravel())
y_pred = model.predict(x_test)


# In[ ]:


confusion = metrics.confusion_matrix(y_test, y_pred )
confusion


# In[ ]:


recall_score(y_test, y_pred)


# Linear SVM gave a Recall/Sensitivity score of 82.54 omn test data.

# #### Polynomial SVM with GridSearchCV to find best values of hyperparameters for a higher recall score

# In[ ]:


# creating a KFold object with 5 splits 
folds = KFold(n_splits = 5, shuffle = True, random_state = 4)

# specify range of parameters (C) as a list
hyper_params = [ {'gamma': [1e-1, 1e-2],
                      'C': [10, 100],
                 'degree': [5,6,7,8]
                 }]


model_poly = SVC(kernel='poly', cache_size=10000)

# set up grid search scheme
# note that we are still using the 5 fold CV scheme we set up earlier
model_cv = GridSearchCV( estimator = model_poly, 
                         param_grid = hyper_params, 
                         scoring= 'recall', 
                         cv = folds, 
                         verbose = 1,
                         return_train_score=True,
                         n_jobs=-1)      


# In[ ]:


# fit the model - it will fit 5 folds across all values of gamma, C and degree
model_cv.fit(x_train, y_train.ravel())


# In[ ]:


# results of grid search CV
cv_results = pd.DataFrame(model_cv.cv_results_)
cv_results


# In[ ]:


# # plotting
plt.figure(figsize=(25,20))

# subplot 4/1
plt.subplot(221)
gamma_1_degree_2 = cv_results.loc[(cv_results.param_gamma==0.1) & (cv_results.param_degree==5)]

plt.plot(gamma_1_degree_2["param_C"], gamma_1_degree_2["mean_test_score"])
plt.plot(gamma_1_degree_2["param_C"], gamma_1_degree_2["mean_train_score"])
plt.xlabel('C')
plt.ylabel('recall')
plt.title("Gamma=0.1 Degree=5")
plt.ylim([0.60, 1])
plt.legend(['test recall', 'train recall'], loc='upper left')
plt.xscale('log')

# subplot 4/2
plt.subplot(222)
gamma_1_degree_3 = cv_results.loc[(cv_results.param_gamma==0.1) & (cv_results.param_degree==6)]

plt.plot(gamma_1_degree_3["param_C"], gamma_1_degree_3["mean_test_score"])
plt.plot(gamma_1_degree_3["param_C"], gamma_1_degree_3["mean_train_score"])
plt.xlabel('C')
plt.ylabel('recall')
plt.title("Gamma=0.1 Degree=6")
plt.ylim([0.60, 1])
plt.legend(['test recall', 'train recall'], loc='upper left')
plt.xscale('log')

# subplot 4/3
plt.subplot(223)
gamma_01_degree_2 = cv_results.loc[(cv_results.param_gamma==0.01) & (cv_results.param_degree==7)]

plt.plot(gamma_01_degree_2["param_C"], gamma_01_degree_2["mean_test_score"])
plt.plot(gamma_01_degree_2["param_C"], gamma_01_degree_2["mean_train_score"])
plt.xlabel('C')
plt.ylabel('recall')
plt.title("Gamma=0.01 Degree=7")
plt.ylim([0.60, 1])
plt.legend(['test recall', 'train recall'], loc='upper left')
plt.xscale('log')

# subplot 4/4
plt.subplot(224)
gamma_01_degree_3 = cv_results.loc[(cv_results.param_gamma==0.01) & (cv_results.param_degree==8)]

plt.plot(gamma_01_degree_3["param_C"], gamma_01_degree_3["mean_test_score"])
plt.plot(gamma_01_degree_3["param_C"], gamma_01_degree_3["mean_train_score"])
plt.xlabel('C')
plt.ylabel('recall')
plt.title("Gamma=0.01 Degree=8")
plt.ylim([0.60, 1])
plt.legend(['test recall', 'train recall'], loc='upper left')
plt.xscale('log')


# In[ ]:


# printing the optimal accuracy score and hyperparameters
best_score = model_cv.best_score_
best_hyperparams = model_cv.best_params_

print("The best test score is {0} corresponding to hyperparameters {1}".format(best_score, best_hyperparams))


# In[ ]:


# model with optimal hyperparameters

# model
model = SVC(C=10, degree=7, gamma=0.01, kernel="poly")

model.fit(x_train, y_train.ravel())
y_pred = model.predict(x_test)
y_pred = pd.DataFrame(y_pred)


# In[ ]:


confusion = metrics.confusion_matrix(y_test, y_pred )
confusion


# In[ ]:


from sklearn.metrics import precision_score, recall_score
recall_score(y_test, y_pred)


# So far Recall Score is the highest 98.20 with Polynomial SVM.

# #### RBF SVM with GridSearchCV to find best values of hyperparameters for a higher recall score

# In[ ]:


# creating a KFold object with 5 splits 
folds = KFold(n_splits = 5, shuffle = True, random_state = 4)

# specify range of parameters (C) as a list
hyper_params = [ {'gamma': [1e-1, 1e-2, 1e-3],
                      'C': [0.1, 1, 10]
                 }]


model_poly = SVC(kernel='rbf', cache_size=10000)

# set up grid search scheme
# note that we are still using the 5 fold CV scheme we set up earlier
model_cv = GridSearchCV( estimator = model_poly, 
                         param_grid = hyper_params, 
                         scoring= 'recall', 
                         cv = folds, 
                         verbose = 1,
                         return_train_score=True,
                         n_jobs=-1)      


# In[ ]:


# fit the model - it will fit 5 folds across all values of C and gamma
model_cv.fit(x_train, y_train.ravel()) 


# In[ ]:


# results of grid search CV
cv_results = pd.DataFrame(model_cv.cv_results_)
cv_results


# In[ ]:


# # plotting
plt.figure(figsize=(25,8))

# subplot 3/1
plt.subplot(131)
gamma_1 = cv_results.loc[(cv_results.param_gamma==0.1)]

plt.plot(gamma_1["param_C"], gamma_1["mean_test_score"])
plt.plot(gamma_1["param_C"], gamma_1["mean_train_score"])
plt.xlabel('C')
plt.ylabel('Recall')
plt.title("Gamma=0.1")
plt.ylim([0.60, 1])
plt.legend(['test recall', 'train recall'], loc='upper left')
plt.xscale('log')

# subplot 3/2
plt.subplot(132)
gamma_01 = cv_results.loc[(cv_results.param_gamma==0.01)]

plt.plot(gamma_01["param_C"], gamma_01["mean_test_score"])
plt.plot(gamma_01["param_C"], gamma_01["mean_train_score"])
plt.xlabel('C')
plt.ylabel('Recall')
plt.title("Gamma=0.01")
plt.ylim([0.60, 1])
plt.legend(['test recall', 'train recall'], loc='upper left')
plt.xscale('log')

# subplot 3/3
plt.subplot(133)
gamma_001 = cv_results.loc[(cv_results.param_gamma==0.001)]

plt.plot(gamma_001["param_C"], gamma_001["mean_test_score"])
plt.plot(gamma_001["param_C"], gamma_001["mean_train_score"])
plt.xlabel('C')
plt.ylabel('Recall')
plt.title("Gamma=0.001")
plt.ylim([0.60, 1])
plt.legend(['test recall', 'train recall'], loc='upper left')
plt.xscale('log')


# In[ ]:


# printing the optimal accuracy score and hyperparameters
best_score = model_cv.best_score_
best_hyperparams = model_cv.best_params_

print("The best test score is {0} corresponding to hyperparameters {1}".format(best_score, best_hyperparams))


# In[ ]:


# model with optimal hyperparameters

# model
model = SVC(C=10, gamma=0.01, kernel="rbf")

model.fit(x_train, y_train.ravel())
y_pred = model.predict(x_test)
y_pred = pd.DataFrame(y_pred)


# In[ ]:


confusion = metrics.confusion_matrix(y_test, y_pred )
confusion


# In[ ]:


recall_score(y_test, y_pred)


# This Recall Score of 93.28 is not as good as Recall score with Polynomial SVM with degree 7 of 98.20.

# #### Random Forests

# In[ ]:


# Create the parameter grid based on the results of random search 
param_grid = {
    'max_depth': [4,8,10],
    'min_samples_leaf': range(100, 400, 200),
    'min_samples_split': range(200, 500, 200),
    'n_estimators': [100,200, 300], 
    'max_features': [5, 10]
}
# Create a based model
rf = RandomForestClassifier()
# Instantiate the grid search model
grid_search = GridSearchCV(estimator = rf, 
                           param_grid = param_grid, 
                           cv = 3, 
                           n_jobs = -1,
                           verbose = 1)

# Fit the grid search to the data
grid_search.fit(x_train, y_train)


# In[ ]:


# printing the optimal accuracy score and hyperparameters
print('We can get accuracy of',grid_search.best_score_,'using',grid_search.best_params_)


# In[ ]:


# model with the best hyperparameters
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(bootstrap=True,
                             max_depth=10,
                             min_samples_leaf=100, 
                             min_samples_split=200,
                             max_features=5,
                             n_estimators=200)
# fit
rfc.fit(x_train,y_train.ravel())


# In[ ]:


y_pred = rfc.predict(x_test)
y_pred = pd.DataFrame(y_pred)


# In[ ]:


confusion = metrics.confusion_matrix(y_test, y_pred )
confusion


# In[ ]:


recall_score(y_test, y_pred)


# The recall score of 84.73 is nowhere near the recall score of 98.20 derived through Polynomial SVM of degree 7.
# 
# So for Prediction, Polynomial SVM with degree 7 gave us the best model in terms of recall score metric.

# #### Driver Variables Identification

# For the driver variable identification, 2 choices would be Decision tree and a Logistic Regression Model.
# Because of the large no of variables, we will go with decision tree first.

# For this part we will use the non PCA data. 
# 
# Also for class imbalance we will use ADASYN this time.
# 
# ADASYN:
# Its a improved version of Smote. What it does is same as SMOTE just with a minor improvement. After creating those sample it adds a random small values to the points thus making it more realistic. In other words instead of all the sample being linearly correlated to the parent they have a little more variance in them i.e they are bit scattered.
# 
# https://medium.com/coinmonks/smote-and-adasyn-handling-imbalanced-data-set-34f5223e167

# In[ ]:


pd.options.display.float_format='{:.0f}'.format
telecom_data_hvc_for_chaid_tree_nc_ld = telecom_data_hvc[telecom_data_hvc.Churned==0]
telecom_data_hvc_for_chaid_tree_nc_ld = telecom_data_hvc_for_chaid_tree_nc_ld.sample(n=2500, random_state=4)
telecom_data_hvc_for_chaid_tree = pd.concat([telecom_data_hvc_for_chaid_tree_nc_ld, (telecom_data_hvc[telecom_data_hvc.Churned==1])], axis = 0)


# In[ ]:


list(telecom_data_hvc_for_chaid_tree.columns)


# Converting date columns to numeric i.e. seconds since epoch, so that ADASYN does not run into any problemms.

# In[ ]:


import datetime
from dateutil.parser import parse
def dt2epoch(value):
    d = parse(value)
    epoch = (d - datetime.datetime(1970,1,1)).total_seconds()
    return epoch 

telecom_data_hvc_for_chaid_tree.date_of_last_rech_6=telecom_data_hvc_for_chaid_tree.date_of_last_rech_6.astype('|S')
telecom_data_hvc_for_chaid_tree.date_of_last_rech_6 = telecom_data_hvc_for_chaid_tree.date_of_last_rech_6.apply(dt2epoch)
telecom_data_hvc_for_chaid_tree.date_of_last_rech_7=telecom_data_hvc_for_chaid_tree.date_of_last_rech_7.astype('|S')
telecom_data_hvc_for_chaid_tree.date_of_last_rech_7 = telecom_data_hvc_for_chaid_tree.date_of_last_rech_7.apply(dt2epoch)
telecom_data_hvc_for_chaid_tree.date_of_last_rech_8=telecom_data_hvc_for_chaid_tree.date_of_last_rech_8.astype('|S')
telecom_data_hvc_for_chaid_tree.date_of_last_rech_8 = telecom_data_hvc_for_chaid_tree.date_of_last_rech_8.apply(dt2epoch)


# In[ ]:


X = telecom_data_hvc_for_chaid_tree.drop(['mobile_number','Churned'], axis=1)
y = telecom_data_hvc_for_chaid_tree['Churned']
X_col_lst = X.columns


# In[ ]:


y.value_counts()


# In[ ]:


from imblearn.over_sampling import ADASYN
sm = ADASYN()
X, y = sm.fit_sample(X, y)


# In[ ]:


y= pd.DataFrame(y,columns=['Churned'])
y.Churned.value_counts()


# This has corrected the class Imbalance.

# In[ ]:


X = pd.DataFrame(X,columns=X_col_lst)


# In[ ]:


X.head()


# CHAID model code below took lot of time to run, hence removing outliers and normalizing data to run logistic regression.

# In[ ]:


from scipy import stats
z = np.abs(stats.zscore(X))
print(z)


# In[ ]:


threshold = 3
print(np.where(z > 3))


# In[ ]:


X_o = X[(z >= 3).all(axis=1)]
X = X[(z < 3).all(axis=1)]


# In[ ]:


y = y.iloc[X.index]


# In[ ]:


from sklearn.preprocessing import StandardScaler
standard_scaler = StandardScaler()
X = standard_scaler.fit_transform(X)


# In[ ]:


X = pd.DataFrame(X,columns=X_col_lst)
X.head()


# In[ ]:


y.reset_index(inplace=True, drop =True)
y.head()


# In[ ]:


#!pip install CHAID


# In[ ]:


#from CHAID import Tree
#df = pd.concat([X, y], axis = 1)
#independent_variable_columns = X_col_lst
#dep_variable = 'Churned'

## create the Tree via pandas
#tree = Tree.from_pandas_df(df, dict(zip(independent_variable_columns, ['nominal'] * 129)), dep_variable, 
#                           dep_variable_type='continuous')

## print the tree (though not enough power to split)
#tree.print_tree()


# Commented the above CHAID tree code as it was taking too much time to run, trying Logistic regression instead.

# Looking at Correlations

# In[ ]:


# Let's see the correlation matrix 
plt.figure(figsize = (20,10))        # Size of the figure
sns.heatmap(X.iloc[:,0:24].corr(),annot = True)
plt.show()


# In[ ]:


# Let's see the correlation matrix 
plt.figure(figsize = (20,10))        # Size of the figure
sns.heatmap(X.iloc[:,25:49].corr(),annot = True)
plt.show()


# In[ ]:


# Let's see the correlation matrix 
plt.figure(figsize = (20,10))        # Size of the figure
sns.heatmap(X.iloc[:,50:74].corr(),annot = True)
plt.show()


# In[ ]:


# Let's see the correlation matrix 
plt.figure(figsize = (20,10))        # Size of the figure
sns.heatmap(X.iloc[:,75:99].corr(),annot = True)
plt.show()


# In[ ]:


# Let's see the correlation matrix 
plt.figure(figsize = (20,10))        # Size of the figure
sns.heatmap(X.iloc[:,100:129].corr(),annot = True)
plt.show()


# Some very high correlations are between variables loc_ic_mou_6/7/8 and loc_ic_t2m_mou_6/7/8.
# 
# Since loc_ic_mou_6/7/8 is summation of t2m, t2t, t2o etc, it is already covered in them, hence dropping these columns.

# In[ ]:


X.drop(['loc_ic_mou_6','loc_ic_mou_7','loc_ic_mou_8'],axis=1,inplace=True)


# In[ ]:


X.info()


# In[ ]:


import statsmodels.api as sm


# Since we are only interested in finding driver variables here, not performing the trIN, TEST SPLIT.

# In[ ]:


# Logistic regression model
logm1 = sm.GLM(y,(sm.add_constant(X)), family = sm.families.Binomial())
logm1.fit().summary()


# In[ ]:


# Check for the VIF values of the feature variables. 
from statsmodels.stats.outliers_influence import variance_inflation_factor
# Create a dataframe that will contain the names of all the feature variables and their respective VIFs
vif = pd.DataFrame()
vif['Features'] = X.columns
vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# Using RFE to get top 20 features

# In[ ]:


from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()


# In[ ]:


from sklearn.feature_selection import RFE
rfe = RFE(logreg, 20)             # running RFE with 20 variables as output
rfe = rfe.fit(X, y)


# In[ ]:


rfe.support_


# In[ ]:


list(zip(X.columns, rfe.support_, rfe.ranking_))


# In[ ]:


col = X.columns[rfe.support_]


# In[ ]:


X_sm = sm.add_constant(X[col])
logm2 = sm.GLM(y,X_sm, family = sm.families.Binomial())
res = logm2.fit()
res.summary()


# In[ ]:


# Create a dataframe that will contain the names of all the feature variables and their respective VIFs
vif = pd.DataFrame()
vif['Features'] = X[col].columns
vif['VIF'] = [variance_inflation_factor(X[col].values, i) for i in range(X[col].shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# In[ ]:


X = X[col] 
X.drop('offnet_mou_6',inplace=True,axis=1)


# In[ ]:


X_sm = sm.add_constant(X)
logm2 = sm.GLM(y,X_sm, family = sm.families.Binomial())
res = logm2.fit()
res.summary()


# In[ ]:


# Create a dataframe that will contain the names of all the feature variables and their respective VIFs
vif = pd.DataFrame()
vif['Features'] = X.columns
vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# In[ ]:


X.drop('total_rech_amt_8',axis=1,inplace=True)


# In[ ]:


X_sm = sm.add_constant(X)
logm2 = sm.GLM(y,X_sm, family = sm.families.Binomial())
res = logm2.fit()
res.summary()


# In[ ]:


# Create a dataframe that will contain the names of all the feature variables and their respective VIFs
vif = pd.DataFrame()
vif['Features'] = X.columns
vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# In[ ]:


X.drop('og_others_8',axis=1,inplace=True)


# In[ ]:


X_sm = sm.add_constant(X)
logm2 = sm.GLM(y,X_sm, family = sm.families.Binomial())
res = logm2.fit()
res.summary()


# In[ ]:


# Create a dataframe that will contain the names of all the feature variables and their respective VIFs
vif = pd.DataFrame()
vif['Features'] = X.columns
vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# In[ ]:


X.drop('monthly_2g_6',axis=1,inplace=True)


# In[ ]:


X_sm = sm.add_constant(X)
logm2 = sm.GLM(y,X_sm, family = sm.families.Binomial())
res = logm2.fit()
res.summary()


# In[ ]:


# Create a dataframe that will contain the names of all the feature variables and their respective VIFs
vif = pd.DataFrame()
vif['Features'] = X.columns
vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# In[ ]:


X.drop('monthly_2g_8',axis=1,inplace=True)


# In[ ]:


X_sm = sm.add_constant(X)
logm2 = sm.GLM(y,X_sm, family = sm.families.Binomial())
res = logm2.fit()
res.summary()


# In[ ]:


# Create a dataframe that will contain the names of all the feature variables and their respective VIFs
vif = pd.DataFrame()
vif['Features'] = X.columns
vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# In[ ]:


X.drop('og_others_7',axis=1,inplace=True)


# In[ ]:


X_sm = sm.add_constant(X)
logm2 = sm.GLM(y,X_sm, family = sm.families.Binomial())
res = logm2.fit()
res.summary()


# In[ ]:


# Create a dataframe that will contain the names of all the feature variables and their respective VIFs
vif = pd.DataFrame()
vif['Features'] = X.columns
vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# In[ ]:


X.drop('std_og_t2t_mou_6',axis=1,inplace=True)


# In[ ]:


X_sm = sm.add_constant(X)
logm2 = sm.GLM(y,X_sm, family = sm.families.Binomial())
res = logm2.fit()
res.summary()


# In[ ]:


X.drop('offnet_mou_8',axis=1,inplace=True)


# In[ ]:


X_sm = sm.add_constant(X)
logm2 = sm.GLM(y,X_sm, family = sm.families.Binomial())
res = logm2.fit()
res.summary()


# In[ ]:


X.drop('std_og_t2m_mou_6',axis=1,inplace=True)


# In[ ]:


X_sm = sm.add_constant(X)
logm2 = sm.GLM(y,X_sm, family = sm.families.Binomial())
res = logm2.fit()
res.summary()


# In[ ]:


After RFE and later manual dropping of columns to achieve optimal values of VIF and p_values for all columns, above are the set of driver variables obtained.


# After RFE and later manual dropping of columns to achieve optimal values of VIF and p_values for all columns, above are the set of driver variables obtained.
