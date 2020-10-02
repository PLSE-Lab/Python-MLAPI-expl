#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np 
import pandas as pd 


# In[ ]:


employees = pd.read_csv("/kaggle/input/softserve-ds-hackathon-2020/employees.csv")
employees.head()


# In[ ]:


sub = pd.read_csv("/kaggle/input/softserve-ds-hackathon-2020/submission.csv")
sub.head()


# In[ ]:


import numpy as np 
import pandas as pd 

employees = pd.read_csv("/kaggle/input/softserve-ds-hackathon-2020/employees.csv")
sub = pd.read_csv("/kaggle/input/softserve-ds-hackathon-2020/submission.csv")
history = pd.read_csv("/kaggle/input/softserve-ds-hackathon-2020/history.csv")

history['Date'] = pd.to_datetime(history['Date'])
history['emp_on_pos'] = history.groupby(['PositionID', 'Date'])['EmployeeID'].transform('count')
history['emp_on_pos'] = history['emp_on_pos'] / history.groupby('Date')['emp_on_pos'].transform('mean')
history['emp_on_pos_change'] = (history.groupby('EmployeeID')['emp_on_pos'].shift(1) - history['emp_on_pos']).fillna(0)
history['emp_on_prj'] = (history.groupby(['ProjectID', 'Date'])['EmployeeID'].transform('count')).fillna(0)
history['emp_on_prj_change'] = (history.groupby('EmployeeID')['emp_on_prj'].shift(1) - history['emp_on_prj']).fillna(0)
history['emp_on_cus'] = history.groupby(['CustomerID', 'Date'])['EmployeeID'].transform('count')
history['emp_on_cus'] = history['emp_on_cus'] / history.groupby('Date')['emp_on_cus'].transform('mean')
history['emp_on_dev'] = history.groupby(['DevCenterID', 'Date'])['EmployeeID'].transform('count')
history['emp_on_dev'] = history['emp_on_dev'] / history.groupby('Date')['emp_on_dev'].transform('mean')
#history['month'] = history['Date'].dt.month
history['Utilization_change'] = (history.groupby('EmployeeID')['Utilization'].shift(1) - history['Utilization']).fillna(0)
history['HourVacation_change'] = (history.groupby('EmployeeID')['HourVacation'].shift(1) - history['HourVacation']).fillna(0)
history = history.merge(employees, how='inner', on='EmployeeID')
history['HiringDate'] = pd.to_datetime(history['HiringDate'])
history['DismissalDate'] = pd.to_datetime(history['DismissalDate'], errors='coerce')
history['set'] = np.where((history['Date'] < '2018-12-01'), "train", "test") #(history['DismissalDate'].isna() == False) |
#history['validation'] = history['EmployeeID'].isin(pd.Series(history['EmployeeID'].unique()).sample(frac=0.2, random_state=1))
history['validation'] = history['Date'] >= '2018-09-01'
#history = history.loc[(history['DevCenterID'].isin(np.array([-1, 0, 6, 14, 30])) & (history['validation']) & (history['set'] == 'train')) == False, :]
#history = history.loc[(history['SBUID'].isin(np.array([ 24,  42,  47, 109, 122, 126, 153, 222, 226, 250, 274, 275, 292,
#       293, 297, 298, 299, 300, 301, 311, 313, 316, 319, 355, 357, 405, 422])) & (history['validation']) & (history['set'] == 'train')) == False, :]
#history = history.loc[(history['CustomerID'].isin(np.array(['015A39EF-7BD8-4B8A-8C20-2A3B89833AC7',
#       '017FBF2E-7B5A-4E80-A4EF-7D64AA276FA6',
#       '069A19C4-7CB1-47B1-B6AE-C5076E47AB69',
#       '07BA6D7C-947F-41B0-AB88-51E60526F706',
#       '116E18F6-78FC-44A1-8187-2E1FCB1904C5',
#       '14D0CB70-B8F8-4008-8CCF-97D3E5607809',
#       '172113B0-3B6B-4A74-A069-97FC0B852CFA',
#       '1A58DE58-8742-4561-965D-BF4066E483AF',
#       '1BC39ED9-F2FC-466F-8ECE-AA524AF1A88F',
#       '1E2092D2-AE10-4572-967E-5F2471C2EA23',
#       '2294EE92-2273-4B8C-8B66-8B31C3F161C2',
#       '284191DB-5EBA-4FEF-8D9F-D122C08863C2',
#       '2B696DC1-F578-4994-9C0D-784F31C2F4D4',
#       '2E693522-A8CB-4A93-957C-6C7A00AEFED8',
#       '354B6660-A20A-4E2E-B444-6D58D6C98F82',
#       '37ED4142-B3BE-4E75-8F26-857F88D09E9B',
#       '3B6BC1F4-743A-4160-9F30-827AB5581959',
#       '3C8A7325-A8DD-447F-A0E6-5E467752CE40',
#       '3ED9C16F-8C85-44E0-9A25-33B66D6F844D',
#       '4018F3C3-FAB6-4ED9-BB61-AA08262C3492',
#       '46E885C4-C2C1-4291-A1D5-6032B4B2C654',
#       '484DD72C-2D3A-4A44-A0EA-4DABDB342FB6',
#       '4D6B497B-5F1F-480D-9B6F-B0B8FCEE8539',
#       '4DD946D2-B357-4B16-B60A-39C17B7E315D',
#       '4FA4D1A4-9A19-4945-B90E-17643CEE5A90',
#       '50896DFD-DEC3-4EA5-8578-2E7BA7B72313',
#       '5270AE84-4495-4DB4-B385-6D10C6B03388',
#       '5760B361-A7C2-4879-B83E-9167A6F4A796',
#       '5EBB9332-4BD9-4176-AB5C-CA711BB5033D',
#       '5F5FD4A1-CF42-4DFF-A7ED-5B162BBD98E0',
#       '6115BB68-C966-40F8-BDCA-24434153AA53',
#       '6EA5880B-6860-4BF3-B30F-D6AFF3312AE4',
#       '7149A3D8-3FB3-40B2-9195-141F13941BD3',
#       '72F37A79-F4CB-4B1C-A81F-9DCAEF9F77CB',
#       '742227C1-4885-4730-962A-9D8E27474E15',
#       '7474C3CB-0189-4061-957B-586433B91E0E',
#       '768847D2-BC5F-4E1F-B3A3-F3C57FCEAD3D',
#       '7724EA76-F17C-4E0C-8A8B-2A0F02315CFA',
#       '7B763088-CD35-4D7B-91BA-F200A0A5A9A1',
#       '7B9A6C4C-38C8-47B9-85AB-D1A8C8ABA53D',
#       '7BBD8523-24E9-4D41-A1F4-66B416F736DF',
#       '7D4BFEDD-2640-458B-81B3-7B69AB1D753A',
#       '7F461E52-1E28-44EF-902C-718BC1AD1DFE',
#       '85B2F195-512A-4B8A-A868-74F2E9F77B46',
#       '86448D10-E733-4EC1-A11D-723F44C84E3D',
#       '8729DBF8-FB75-43C4-A4ED-A7093809CDC3',
#       '872C2E11-6A3C-4501-9585-A0FEBB395386',
#       '89FCDA4E-0CC3-4AEA-9A07-29CB468452B0',
#       '8B85C566-A256-4060-848E-D17BA6A39E22',
#       '8C8CF6BE-6B63-48A4-8983-C2EFC5D11971',
#       '8D8C176C-9415-4950-96D9-783D4A1D215A',
#       '8DDEA1A2-246C-48F3-99EB-DF28F55FBC1D',
#       '8FE689BB-1909-47B8-87AE-DAE0CD7C0F9A',
#       '909269AB-62DD-42F3-AF16-EB78BDAC09C4',
#       '9180973E-D20B-4E8C-8881-74C8300E718B',
#       '9393E803-D873-4F36-801B-2DAC0A662A9D',
#       '9EC3C5D1-D6B5-4459-93BF-EA938A095731',
#       'A00D255A-6787-4885-B954-27D31B8F553D',
#       'A2089880-08AD-48DB-8C1A-FC8377079C63',
#       'A36CF132-07F9-471A-840D-522AEECF54D7',
#       'A4771E0F-B827-4748-AD9D-F7076D91A1CA',
#       'A51F9FDE-7218-40D8-B58F-A255D5291EF1',
#       'A6E4A4B4-2643-4B3D-BAD5-76F288232745',
#       'B199F6F1-846A-408D-854C-0F3D596F0C4D',
#       'B235B7F5-A071-4B29-9AB4-B22FE09A50F3',
#       'B28E7A2F-1F75-4C66-967D-35DC7219B558',
#       'B74C7C7C-C8E2-4389-8AAD-3E46485B8215',
#       'BB90C3BF-D862-4342-9044-A70A098D08F7',
#       'BBEE244C-0ABB-4AA9-AAA9-8510DE529356',
#       'BF15AE51-95EE-4E06-BEA7-E1891A4B63C2',
#       'C0225BAC-BA7D-432F-8D91-966ACE0B58C4',
#       'C2D2FDCB-B2FC-40E6-B990-CCBB3E93D100',
#       'C4CF2209-8536-49EA-A7E1-A1FBC516D8CA',
#       'C5653814-8197-40EF-9046-C63E1F3E7ADC',
#       'CFF364A1-68E8-4AEE-BA4C-61834B85ACAD',
#       'D1F447A6-1623-4773-9F3E-9E11E79D9F02',
#       'D5BE4AB2-4398-41DC-9327-2F426E86D6A7',
#       'D9C4184C-F2BC-4BB3-B0EC-C6A89DFD6FE9',
#       'DBFD3DD1-27BA-43AE-BF49-BE1D07BA2F4E',
#       'DD1B3AB0-48A9-4243-BCAF-32F94B62323C',
#       'DD265C71-783A-4E56-BBFC-9B1E0E022A5B',
#       'E06DCD4A-7A1A-473E-9D19-A48FA458F4A8',
#       'E94F4117-A69A-49AB-9721-5050DD431011',
#       'EBB2E646-5434-4507-B4E6-C52C93F63BBE',
#       'ED7722A3-72F4-43CE-86F7-8E8355F177BB',
#       'F495A029-EE32-494F-9EFD-C22996F07484',
#       'F69AECAC-2267-405F-A85C-B9A449E36D5A',
#       'F7BCC5D9-7848-40B7-B64A-AD13963D2C6D',
#       'F8997A90-6D0A-4B40-86AA-64E25CAE2978',
#       'F9DA595D-D4CC-41EA-B22A-F5A9B0133F5A',
#       'FF88F846-DE9C-48E8-B984-818023092ACA'], dtype=object)) & (history['validation']) & (history['set'] == 'train')) == False, :]
#history = history.loc[(history['PositionID'].isin(np.array([   3,   32,   60,   79,   84,   91,  107,  108,  115,  122,  127,
#        137,  147,  186,  233,  234,  250,  283,  295,  298,  301,  313,
#        314,  345,  373,  377,  395,  399,  418,  419,  420,  422,  423,
#        424,  430,  434,  445,  461,  472,  505,  507,  515,  521,  523,
#        524,  529,  590,  594,  604,  608,  623,  625,  626,  628,  630,
#        636,  684,  703,  730,  732,  743,  748,  763,  768,  781,  782,
#        789,  793,  804,  805,  810,  811,  823,  825,  840,  844,  847,
#        848,  851,  852,  859,  860,  861,  877,  892,  893,  910,  911,
#        913,  914,  918,  932,  935,  951,  965,  966,  972,  975,  985,
#       1001, 1002, 1003, 1004, 1005, 1006, 1009, 1012, 1013, 1014, 1017,
#       1030, 1052, 1053, 1070, 1072, 1073, 1074, 1075, 1076, 1077, 1078,
#       1080, 1082, 1083, 1084, 1085, 1086, 1087, 1089, 1090, 1091, 1096,
#       1099, 1102, 1105, 1110, 1114, 1118, 1126, 1139, 1143, 1146, 1147,
#       1149, 1150, 1152, 1158, 1162, 1163, 1165, 1166, 1186, 1187, 1189,
#       1190, 1192, 1193, 1201, 1216, 1218, 1223, 1224, 1226, 1230, 1246,
#       1254, 1263])) & (history['validation']) & (history['set'] == 'train')) == False, :]
history['days_to_dismissal'] = (history['DismissalDate'] - history['Date'])/ np.timedelta64(1, 'D')#.astype(int)
mad_emp = list(history[history.days_to_dismissal<0]['EmployeeID'].unique())
history = history[(history['days_to_dismissal']!=0) & (~history.EmployeeID.isin(mad_emp))]
#history = history[(~(history['days_to_dismissal'] < 65))]
history['HiringDate'] = (history['Date'] - history['HiringDate'])/ np.timedelta64(1, 'D')#.days#.astype(int)
history['Wage_plus'] = (history['WageGross'] - history.groupby('EmployeeID')['WageGross'].shift(1)).fillna(0)
#history['ProjectID'] = np.where(history['ProjectID'].isna(), 1, 0)
history.loc[history.ProjectID.isnull(),'ProjectID'] = 'noprj'
history['Date_up_wage'] = pd.to_datetime('2017-01-01')
history.loc[history['Wage_plus'] != 0, 'Date_up_wage'] =  history.loc[history['Wage_plus'] != 0, 'Date']
history['Date_up_wage'] = history.groupby('EmployeeID')['Date_up_wage'].cummax()
history['Date_up_wage'] = (history['Date'] - history['Date_up_wage']).dt.days#/ np.timedelta64(1, 'D')
history['New_project'] = (history['CustomerID'] != history.groupby('EmployeeID')['CustomerID'].shift(1)).astype('int')
history['projects'] = history.groupby('EmployeeID')['New_project'].cumsum()
history['Date_np'] = pd.to_datetime('2017-01-01')
history.loc[history['New_project'] != 0, 'Date_np'] =  history.loc[history['New_project'] != 0, 'Date']
history['Date_np'] = history.groupby('EmployeeID')['Date_np'].cummax()
history['Date_np'] = (history['Date'] - history['Date_np']).dt.days
history['wage_date'] = history['WageGross'] / np.log(history['HiringDate']+2)
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
history.loc[history['Date_up_wage'] > 1000000, 'Date_up_wage'] = 0
df_new_prj = history.loc[history.Date==history.HiringDate,['ProjectID', 'EmployeeID','Date']].groupby(['ProjectID','Date']).            count().reset_index().rename(columns={'EmployeeID': 'emp_new_on_prj'})

df_dis_prj = history.loc[~history.DismissalDate.isnull(),['ProjectID','DismissalDate']]
#p['dis'] = p['DismissalDate'] < p['Date']
df_dis_prj = df_dis_prj.groupby(['ProjectID','DismissalDate']).size().reset_index()
df_dis_prj['Date'] = df_dis_prj['DismissalDate'] #+ pd.DateOffset(months=1)
df_dis_prj = df_dis_prj.loc[:, ['ProjectID', 'Date', 0]].rename(columns={0:'emp_dis_on_prj'})

history = history.merge(history.loc[:,['ProjectID', 'EmployeeID','Date']].groupby(['ProjectID','Date']).            count().reset_index().rename(columns={'EmployeeID': 'emp_on_prj'}).merge(df_new_prj, how='outer', on=['Date', 'ProjectID'])     .merge(df_dis_prj, how='outer', on=['Date', 'ProjectID']).fillna(0), on=['Date', 'ProjectID'], how='left')

history['pr_ch'] = (history['emp_on_prj_y'] - history['emp_dis_on_prj']) / history['emp_on_prj_y'] 
history['pr_ch_p'] = (history['emp_on_prj_y'] - history['emp_new_on_prj']) / history['emp_on_prj_y'] 
history['target'] = np.where(history['days_to_dismissal'] <= 92, 1, 0) 
#history['PositionLevel'] = history['PositionLevel'] / history.groupby('Date')['PositionLevel'].transform('mean')
#history['BonusOneTime'] = history['BonusOneTime'] / history.groupby('Date')['BonusOneTime'].transform('mean')
#history['LanguageLevelID'] = history['LanguageLevelID'] / history.groupby('Date')['LanguageLevelID'].transform('mean')
#history['wage_date'] = history['wage_date'] / history.groupby('Date')['wage_date'].transform('mean')
#history['Date_up_wage'] = history['Date_up_wage'] / history.groupby('Date')['Date_up_wage'].transform('mean')
#history['WageGross'] = history['WageGross'] / history.groupby('Date')['WageGross'].transform('mean')
history['HiringDate'] = history['HiringDate'] / history.groupby('Date')['HiringDate'].transform('mean')
#history['Utilization'] = history['Utilization'] / history.groupby('Date')['Utilization'].transform('mean')
#history['HourVacation'] = history['HourVacation'] / history.groupby('Date')['HourVacation'].transform('mean')
#history['HourMobileReserve'] = history['HourMobileReserve'] / history.groupby('Date')['HourMobileReserve'].transform('mean')
#history['HourLockedReserve'] = history['HourLockedReserve'] / history.groupby('Date')['HourLockedReserve'].transform('mean')
#history['APM'] = history['APM'] / history.groupby('Date')['APM'].transform('mean')
#history['MonthOnPosition'] = history['MonthOnPosition'] / history.groupby('Date')['MonthOnPosition'].transform('mean')
#history['MonthOnSalary'] = history['MonthOnSalary'] / history.groupby('Date')['MonthOnSalary'].transform('mean')
history = history.loc[~(history['Date'] <= '2017-08-01'), :]

history = history.drop(columns = ['OnSite', 'days_to_dismissal', 'New_project', 'IsTrainee', 'IsInternalProject'])
history.head(40)


# In[ ]:





# In[ ]:


history 


# In[ ]:


ones_from_train = history.loc[(history['set'] == "train") & (~history['validation']) & (history['target'] == 1), 'EmployeeID']


# In[ ]:


validation = history.loc[(history['set'] == "train") & (history['validation']) & (~history['EmployeeID'].isin(ones_from_train)),:]
validation.groupby('target').size()


# In[ ]:


x = [x.select_dtypes(include=['int', 'float']) for _, x in validation.groupby('Date')]
len(x)


# In[ ]:


train = history.loc[(history['set'] == "train") & (history['validation'] == False),:] #\
    #.sample(frac=1.0).groupby('target').head(2911)

train = train.select_dtypes(include=['int', 'float'])
train.groupby('target').size()


# In[ ]:


nulls = train.loc[train['target'] == 0, :]
ones = train.loc[train['target'] == 1, :]


# In[ ]:


train_list = [pd.concat([i.head(2268), ones]).sample(frac=1.0, random_state=666) for i in np.array_split(nulls.sample(frac=1.0, random_state=666), 16)]


# In[ ]:


train_splits = [(train.iloc[:, :(train.shape[1]-1)].values, train.iloc[:, (train.shape[1]-1)].values) for train in train_list]


# In[ ]:


len(train_splits)


# $$F_\beta = (1 + \beta^2) \cdot \frac{\mathrm{precision} \cdot \mathrm{recall}}{(\beta^2 \cdot \mathrm{precision}) + \mathrm{recall}}.$$

# In[ ]:


#from autograd import grad, hessian, jacobian
#from autograd import hessian_vector_product as hvp

def f(y, pred_y, beta=1.7):
    pred_y = np.round(pred_y) #+ np.random.normal(0, 0.001, len(pred_y))# + np.random.rand(len(pred_y))
    tp = (y*pred_y).sum() #+ np.random.normal(0, 0.001, 1)
    tn = ((1-y)*(1-pred_y)).sum()
    fp = ((1-y)*pred_y).sum()
    fn = (y*(1-pred_y)).sum()

    p = tp / (tp + fp + 1e-07) #+ 1e-07
    r = tp / (tp + fn + 1e-07) #+ 1e-07

    f1 = (1+beta**2)*(p*r / ((beta**2)*p+r+1e-07))
    
    return f1

#fg = grad(f)
#fh = hessian(f)
#fv = hvp(f)
#
#a = np.array([1.0, 0.0, 1, 1, 0], dtype=float)
#b = np.array([0.0, 1.0, 0, 0.0, 0.0], dtype=float)
#
#fg(a, b), fh(a, b),fh(a, b).diagonal(axis1=0, axis2=1)
#fv(a, b)


# In[ ]:


from sklearn.metrics import fbeta_score

#def fbeta_train(y, pred):
#    
#    grad = fg(y, pred)
#    hess = fh(y, pred)
#    
#    return grad, hess[0].diagonal(axis1=0, axis2=1)#np.diag(hess)
#
#def fbeta_train(y, pred):
#    beta = 3
#    p    = 1. / (1 + np.exp(-pred))
#    grad = p * ((beta -1) * y + 1) - beta * y
#    hess = ((beta - 1) * y + 1) * p * (1.0 - p)
#    
#    return grad, hess

def fbeta_valid(y_true, y_pred):
    loss = f(y_true, y_pred)
    return "f_beta", loss, True #np.mean(loss), True


# In[ ]:


#X_train = train.iloc[:, :(train.shape[1]-1)].values#[:10000,:]
#y_train = train.iloc[:, (train.shape[1]-1)].values#[:10000]
#
#X_test = x[2].iloc[:, :(train.shape[1]-1)].values
#y_test = x[2].iloc[:, (train.shape[1]-1)].values


# In[ ]:


import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import auc, accuracy_score, roc_auc_score, roc_curve
from sklearn.model_selection import GridSearchCV
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt


estimators = [lgb.LGBMClassifier(
    objective='binary', #fbeta_train, #
                          n_jobs=-1,
                          #is_unbalance=True,
                          #bagging_fraction=0.9,
                          #bagging_freq=10,
    boosting='goss',
    #max_depth=3,
                          #boosting_type='gbdt',
                          feature_fraction=0.7,
                          learning_rate=0.01,
                          min_data_in_leaf=5,
                          #min_split_gain=0.005,
                          n_estimators=180,
                          num_leaves=50,
                          #reg_alpha=0.0001,
                          #reg_lambda=0.01,
                          subsample=0.9).fit(X_train, y_train) for X_train, y_train in train_splits]

#param_grid = {
    #'n_estimators': [x for x in range(24,2000,20)],
    #'learning_rate': [0.01, 0.05, 0.02, 0.1]
    #'min_data_in_leaf': [20, 5, 10],
    #'feature_fraction': [0.6, 0.7, 0.8],
    #'num_leaves': [40, 35, 50, 45],
#}
#gridsearch = GridSearchCV(estimator, param_grid)
#
#gridsearch.fit(X_train, y_train, #init_score = np.random.rand(100),
#        eval_set = [(X_test, y_test)],
#        eval_metric = fbeta_valid, #'binary_logloss',# #['binary_logloss'],#,#['binary_logloss'], #'auc',  # 
#        early_stopping_rounds = 50)


# In[ ]:


r = []
for i in x:
    X_test = i.iloc[:, :(train.shape[1]-1)].values
    y_test = i.iloc[:, (train.shape[1]-1)].values
    p = np.median(np.array([clf.predict_proba(X_test)[:,1] for clf in estimators]), 0)
    r.append(fbeta_score(y_test, (p > 0.5).astype('int'), beta=1.7))
    
r, np.array(r).mean()


# In[ ]:


pd.DataFrame({
    'column': train.columns[:-1],
    'imp': estimators[0].feature_importances_
}).sort_values('imp', ascending=False)


# In[ ]:


test = history.loc[history['set'] == 'test', :].groupby('EmployeeID').tail(1)
test


# In[ ]:


(estimators[0].predict_proba(X_test)[:,1] > 0.51).mean()


# In[ ]:


vr = {i:fbeta_score(y_test, (estimators[0].predict_proba(X_test)[:, 1] > i).astype('int'), beta=1.7) for i in np.array(list(range(10, 1000, 10)))/1000}


# In[ ]:


vr


# In[ ]:


pd.DataFrame({
    'threshhold': np.array(list(range(10, 1000, 10)))/1000,
    'fbeta': list(vr.values())
}).plot.line(x='threshhold', y='fbeta')


# In[ ]:


tt = test.select_dtypes(include=['int', 'float']).iloc[:, :(train.shape[1]-1)]

tt = tt.values

pr_submit = np.array([clf.predict_proba(tt)[:,1] for clf in estimators]).mean(0)


# In[ ]:


pr_submit


# In[ ]:


pr_valid = np.array([clf.predict_proba(X_test)[:,1] for clf in estimators]).mean(0)


# In[ ]:


pd.Series(pr_valid).plot(kind='density')
pd.Series(pr_submit).plot(kind='density')


# In[ ]:


pr_valid.mean(), pr_submit.mean()


# In[ ]:


(pr_valid > 0.5).astype('int').mean(), (pr_submit > 0.51).astype('int').mean()


# In[ ]:


pr_binary = (pr_submit > 0.51).astype('int')#.mean() 


# In[ ]:


sub.loc[:,['EmployeeID']].merge(pd.DataFrame({
    'EmployeeID': test['EmployeeID'].values,
    'target': pr_binary#.astype('int')
}), on='EmployeeID', how='left').to_csv('submission.csv', index=False)


# In[ ]:


sub.loc[:,['EmployeeID']].merge(pd.DataFrame({
    'EmployeeID': test['EmployeeID'].values,
    'target': pr_submit#.astype('int')
}), on='EmployeeID', how='left').to_csv('scores.csv', index=False)

