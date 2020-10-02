#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np 
import pandas as pd 
from sklearn.metrics import fbeta_score, classification_report, log_loss


# In[ ]:


employees = pd.read_csv("/kaggle/input/softserve-ds-hackathon-2020/employees.csv")

history = pd.read_csv("/kaggle/input/softserve-ds-hackathon-2020/history.csv")
history['Date'] = pd.to_datetime(history['Date'])
history['month'] = history['Date'].dt.month
history = history.merge(employees, how='inner', on='EmployeeID')
history['HiringDate'] = pd.to_datetime(history['HiringDate'])
history['DismissalDate'] = pd.to_datetime(history['DismissalDate'], errors='coerce')
history['set'] = np.where((history['Date'] < '2018-12-01'), "train", "test")
history['validation'] = history['Date'] >= '2018-09-01'
history['days_to_dismissal'] = (history['DismissalDate'] - history['Date'])/ np.timedelta64(1, 'D')#.astype(int)
history['HiringDate'] = (history['Date'] - history['HiringDate'])/ np.timedelta64(1, 'D')#.days#.astype(int)
history.loc[history.ProjectID.isnull(),'ProjectID'] = 'noprj'
mad_emp = list(history[history.days_to_dismissal<0]['EmployeeID'].unique())
history = history[(history['days_to_dismissal']!=0) & (~history.EmployeeID.isin(mad_emp))]


# In[ ]:


#df_dis_prj = history.loc[~history.DismissalDate.isnull(),['ProjectID', 'EmployeeID','Date','DevCenterID']].groupby(['DevCenterID','ProjectID','Date']).count().reset_index()
df_dis_prj = history.loc[~history.DismissalDate.isnull(),['ProjectID', 'EmployeeID','Date']].groupby(['ProjectID','Date']).count().reset_index()

df_emp_on_pos = history[['PositionID', 'Date','HiringDate' ]].groupby(['PositionID', 'Date']).agg({'median', 'count'})
df_emp_on_pos.columns =df_emp_on_pos.columns.map('{0[0]}_{0[1]}'.format) 
df_emp_on_pos = df_emp_on_pos.rename(columns={'HiringDate_count': 'emp_on', 'HiringDate_median':'ltime_median'}).add_suffix('_pos').reset_index()
#
df_emp_on_prj = history[['ProjectID', 'Date','HiringDate' ]].groupby(['ProjectID', 'Date']).agg({'median', 'count'})
df_emp_on_prj.columns =df_emp_on_prj.columns.map('{0[0]}_{0[1]}'.format) 
df_emp_on_prj = df_emp_on_prj.rename(columns={'HiringDate_count': 'emp_on', 'HiringDate_median':'ltime_median'}).add_suffix('_prj').reset_index()
#
df_emp_on_dev = history[['DevCenterID', 'Date','HiringDate' ]].groupby(['DevCenterID', 'Date']).agg({'median', 'count'})
df_emp_on_dev.columns =df_emp_on_dev.columns.map('{0[0]}_{0[1]}'.format) 
df_emp_on_dev = df_emp_on_dev.rename(columns={'HiringDate_count': 'emp_on', 'HiringDate_median':'ltime_median'}).add_suffix('_dev').reset_index()
#
df_emp_on_sub = history[['SBUID', 'Date','HiringDate' ]].groupby(['SBUID', 'Date']).agg({'median', 'count'})
df_emp_on_sub.columns =df_emp_on_sub.columns.map('{0[0]}_{0[1]}'.format) 
df_emp_on_sub = df_emp_on_sub.rename(columns={'HiringDate_count': 'emp_on', 'HiringDate_median':'ltime_median'}).add_suffix('_sub').reset_index()


# In[ ]:


#WageGross
df_wage_pr_pos = history[['ProjectID', 'PositionID', 'Date','WageGross' ]].    groupby(['ProjectID', 'PositionID', 'Date']).median().rename(columns={'WageGross': 'WageGross_prj_pos'}).reset_index()
df_wage_pos = history[[ 'PositionID', 'Date','WageGross' ]].    groupby(['PositionID', 'Date']).median().rename(columns={'WageGross': 'WageGross_pos'}).reset_index()

#
df_emp_on_prj_pos = history[['ProjectID', 'PositionID', 'Date','HiringDate' ]].groupby(['ProjectID', 'PositionID', 'Date']).agg({'median', 'count'})
df_emp_on_prj_pos.columns =df_emp_on_prj_pos.columns.map('{0[0]}_{0[1]}'.format) 
df_emp_on_prj_pos = df_emp_on_prj_pos.rename(columns={'HiringDate_count': 'emp_on', 'HiringDate_median':'ltime_median'}).add_suffix('_prj_pos').reset_index()
#
df_emp_on_dev_pos = history[['DevCenterID', 'PositionID', 'Date','HiringDate' ]].groupby(['DevCenterID', 'PositionID', 'Date']).agg({'median', 'count'})
df_emp_on_dev_pos.columns =df_emp_on_dev_pos.columns.map('{0[0]}_{0[1]}'.format) 
df_emp_on_dev_pos = df_emp_on_dev_pos.rename(columns={'HiringDate_count': 'emp_on', 'HiringDate_median':'ltime_median'}).add_suffix('_dev_pos').reset_index()
df_emp_on_dev_pos.sample(5)


# In[ ]:


history = history.merge(df_emp_on_pos, on = ['PositionID', 'Date'], how = 'left').    merge(df_emp_on_prj, on = ['ProjectID', 'Date'], how = 'left').        merge(df_emp_on_dev, on = ['DevCenterID', 'Date'], how = 'left').            merge(df_emp_on_sub, on = ['SBUID', 'Date'], how = 'left').                merge(df_emp_on_dev_pos, on = ['DevCenterID', 'PositionID', 'Date'], how = 'left').                merge(df_emp_on_prj_pos, on = ['ProjectID', 'PositionID', 'Date'], how = 'left').                merge(df_wage_pr_pos, on = ['ProjectID', 'PositionID', 'Date'], how = 'left').                merge(df_wage_pos, on = ['PositionID', 'Date'], how = 'left')


# In[ ]:


for i in ['_pos','_prj','_dev','_sub' ,'_prj_pos','_dev_pos']:
    history['ltime_ratio' + i] = history['HiringDate']/history['ltime_median'+i]

for i in ['_pos','_prj_pos']:
    history['WageGross_ratio' + i] = history['WageGross']/history['WageGross'+i]


# In[ ]:


cols_prj = list(history.loc[:,history.columns.str.endswith('_prj') | history.columns.str.endswith('_prj_pos')].columns) 
for i in cols_prj :
    history.loc[(history.ProjectID=='noprj'), i] = history.loc[(history.ProjectID!='noprj'), i].median()


# In[ ]:


history['emp_on_pos_change'] = history.groupby('EmployeeID')['emp_on_pos'].shift(1) - history['emp_on_pos']
history['emp_on_prj_change'] = history.groupby('EmployeeID')['emp_on_prj'].shift(1) - history['emp_on_prj']

history['Wage_plus'] = (history['WageGross'] - history.groupby('EmployeeID')['WageGross'].shift(1)).fillna(0)
history['ProjectID'] = np.where(history['ProjectID']=='noprj', 1, 0)
history['Date_up_wage'] = pd.to_datetime('2017-01-01')
history.loc[history['Wage_plus'] != 0, 'Date_up_wage'] =  history.loc[history['Wage_plus'] != 0, 'Date']
history['Date_up_wage'] = history.groupby('EmployeeID')['Date_up_wage'].cummax()
history['Date_up_wage'] = (history['Date'] - history['Date_up_wage']).dt.days
#history['wage_date'] = history['WageGross'] / history['HiringDate'] #????????

history['PositionLevel'] = history.groupby('PositionLevel')['WageGross'].transform('median')
history['PositionLevel_Wage'] = history['WageGross'] - history.groupby(['PositionLevel', 'Date'])['WageGross'].transform('median')
history['PositionID'] = history.groupby('PositionID')['WageGross'].transform('median')
history['Position_Wage'] = history['WageGross'] - history.groupby(['PositionID', 'Date'])['WageGross'].transform('median')
history['Position_count'] = history.groupby(['PositionID', 'Date'])['EmployeeID'].transform('count')
history['LanguageLevelID'] = history.groupby('LanguageLevelID')['WageGross'].transform('median')       
history['CompetenceGroupID'] = history.groupby('CompetenceGroupID')['WageGross'].transform('median')
history['FunctionalOfficeID'] = history.groupby('FunctionalOfficeID')['WageGross'].transform('median')
history['PaymentTypeId'] = history.groupby('PaymentTypeId')['WageGross'].transform('median')
history['SBUID_count'] = history.groupby(['SBUID', 'Date'])['EmployeeID'].transform('count')
history['SBUID'] = history.groupby('SBUID')['WageGross'].transform('median')
history['SBUID_Wage'] = history['WageGross'] - history.groupby(['SBUID', 'Date'])['WageGross'].transform('median')
history['DevCenter_count'] = history.groupby(['DevCenterID', 'Date'])['EmployeeID'].transform('count')
history['DevCenterID'] = history.groupby('DevCenterID')['WageGross'].transform('median')

#history['PositionLevel'] = history.groupby('PositionLevel')['WageGross'].transform('median')
#history['PositionID'] = history.groupby('PositionID')['WageGross'].transform('median')
#history['LanguageLevelID'] = history.groupby('LanguageLevelID')['WageGross'].transform('median')       
history['CompetenceGroupID_wage'] = history.groupby('CompetenceGroupID')['WageGross'].transform('median')
#history['FunctionalOfficeID'] = history.groupby('FunctionalOfficeID')['WageGross'].transform('median')
#history['PaymentTypeId'] = history.groupby('PaymentTypeId')['WageGross'].transform('median')
#
history['SBUID_wage'] = history.groupby('SBUID')['WageGross'].transform('median')
#history['DevCenterID'] = history.groupby('DevCenterID')['WageGross'].transform('median')
history.loc[history['Date_up_wage'] > 1000000, 'Date_up_wage'] = 0
history['target'] = np.where(history['days_to_dismissal'] <= 92, 1, 0) 
history = history.loc[~(history['Date'] <= '2017-08-01'), :]
#history['target'] = np.where(history['days_to_dismissal'] < 92, 1, 0) 

history = history.drop(columns=['days_to_dismissal'])
history.head(2)


# In[ ]:


history.groupby('target').size()


# In[ ]:


history['validation'].isna().sum()#.size()


# In[ ]:


em = list(history.loc[history.ltime_ratio_pos.isnull(),'EmployeeID'].unique())


# In[ ]:


history.info()


# In[ ]:


history.replace([np.inf, -np.inf], np.nan,inplace=True)
history = history.fillna(0)


# In[ ]:


history.shape


# In[ ]:


ones_from_train = history.loc[(history['set'] == "train") & (~history['validation']) & (history['target'] == 1), 'EmployeeID']

validation = history.loc[(history['set'] == "train") & (history['validation']) & (~history['EmployeeID'].isin(ones_from_train)),:]
validation.groupby('target').size()


# In[ ]:


x = [x.select_dtypes(include=['int', 'float']) for _, x in validation.groupby('Date')]
len(x)


# In[ ]:


validation.groupby('Date').size()


# In[ ]:


train = history.loc[(history['set'] == "train") & (history['validation'] == False),:]
train = train.select_dtypes(include=['int', 'float'])
train.groupby('target').size()


# In[ ]:


train.columns


# In[ ]:


nulls = train.loc[train['target'] == 0, :]
ones = train.loc[train['target'] == 1, :]


# In[ ]:


train_list = [pd.concat([i.head(2268), ones]).sample(frac=1.0, random_state=666) for i in np.array_split(nulls.sample(frac=1.0, random_state=666), 16)]

train_splits = [(train.iloc[:, :(train.shape[1]-1)].values, train.iloc[:, (train.shape[1]-1)].values) for train in train_list]


# In[ ]:


from sklearn.metrics import fbeta_score
from imblearn.over_sampling import RandomOverSampler
from xgboost import XGBRegressor, XGBClassifier

import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import auc, accuracy_score, roc_auc_score, roc_curve
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt


# In[ ]:


classifiers = [XGBClassifier(
    objective='binary:logistic',
    booster= 'dart',
     learning_rate =0.01,
     n_estimators=200,
     max_depth=7,
     min_child_weight=2,
     #gamma=0,
     subsample=0.9,
     colsample_bytree=0.5,
    reg_lambda=0.1,
    alpha=0.1,
    #rate_drop= 0.1,
    #skip_drop= 0.1,
     #eval_metric='logloss', # aucpr
     nthread=-4,
     seed=27).fit(X_train, y_train) for X_train, y_train in train_splits]


# In[ ]:


r = []
for i in x:
    X_test = i.iloc[:, :(train.shape[1]-1)].values
    y_test = i.iloc[:, (train.shape[1]-1)].values
    p = np.median(np.array([clf.predict_proba(X_test)[:,1] for clf in classifiers]), 0)
    r.append(fbeta_score(y_test, (p > 0.5).astype('int'), beta=1.7))
    
r, np.array(r).mean()


# In[ ]:


pd.Series(p).hist()


# In[ ]:


pd.DataFrame({
    'column': train.columns[:-1],
    'imp': classifiers[0].feature_importances_
}).sort_values('imp', ascending=False)


# In[ ]:


test = history.loc[history['set'] == 'test', :].groupby('EmployeeID').tail(1)
test


# In[ ]:


(classifiers[0].predict_proba(X_test)[:,1] > 0.5).mean()


# In[ ]:


vr = {i:fbeta_score(y_test, (classifiers[0].predict_proba(X_test)[:, 1] > i).astype('int'), beta=1.7) for i in np.array(list(range(10, 1000, 10)))/1000}
vr


# In[ ]:


pd.DataFrame({
    'threshhold': np.array(list(range(10, 1000, 10)))/1000,
    'fbeta': list(vr.values())
}).plot.line(x='threshhold', y='fbeta')


# In[ ]:


tt = test.select_dtypes(include=['int', 'float']).iloc[:, :(train.shape[1]-1)].values

pr_submit = np.array([clf.predict_proba(tt)[:,1] for clf in classifiers]).mean(0)

pr_submit


# In[ ]:


pr_valid = np.array([clf.predict_proba(X_test)[:,1] for clf in classifiers]).mean(0)


# In[ ]:


pd.Series(pr_valid).plot(kind='density')
pd.Series(pr_submit).plot(kind='density')


# In[ ]:


pr_valid.mean(), pr_submit.mean()


# In[ ]:


for th in np.arange(0.48, 0.7, 0.02):
    print(th)
    df_pr_test['pred'] =  df_pr_test['proba'].apply(lambda x: 1 if x>=th else 0)
    df_pr_test_12['pred'] =  df_pr_test_12['proba'].apply(lambda x: 1 if x>=th else 0)
    df_pr_test_2['pred'] =  df_pr_test_2['proba'].apply(lambda x: 1 if x>=th else 0)
    #print(df_pr_test['pred'].sum(),df_pr_test['pred'].mean() ,df_pr_test['target'].mean())
    print(fbeta_score(df_pr_test['target'], df_pr_test['pred'], beta=1.7), fbeta_score(df_pr_test_12['target'], df_pr_test_12['pred'], beta=1.7)) 
    #print('2018-12')
    print('mean')
    print(df_pr_test['target'].mean(), df_pr_test['pred'].mean(),df_pr_test_12['pred'].mean(),df_pr_test_2['pred'].mean())
    #print(fbeta_score(df_pr_test_12['target'], df_pr_test_12['pred'], beta=1.7))
    print(classification_report(df_pr_test['target'], df_pr_test['pred'], target_names=['0','1']))
    print('----')


# In[ ]:


(pr_valid > 0.5).astype('int').mean(), (pr_submit > 0.5).astype('int').mean()


# In[ ]:


pr_binary = (pr_submit > 0.48).astype('int') #
pr_binary.mean()


# In[ ]:


sub = pd.read_csv("/kaggle/input/softserve-ds-hackathon-2020/submission.csv")

sub.loc[:,['EmployeeID']].merge(pd.DataFrame({
    'EmployeeID': test['EmployeeID'].values,
    'target': pr_binary#.astype('int')
}), on='EmployeeID', how='left').to_csv('submission.csv', index=False)


# In[ ]:


sub.loc[:,['EmployeeID']].merge(pd.DataFrame({
    'EmployeeID': test['EmployeeID'].values,
    'target': pr_submit#.astype('int')
}), on='EmployeeID', how='left').to_csv('scores.csv', index=False)


# In[ ]:




