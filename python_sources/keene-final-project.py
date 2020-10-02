# DongKyu Kim

import numpy as np
import pandas as pd
import math
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

# Feature Engineering

# exampl from https://www.kaggle.com/dansbecker/xgboost/notebook
train_X = pd.read_csv('../input/trainFeatures.csv')
train_Y = pd.read_csv('../input/trainLabels.csv')
test_X = pd.read_csv('../input/testFeatures.csv')
train_X = train_X.drop(['ids','RatingID','erkey','AccountabilityID','RatingInterval','CN_Priority','Creation_Date','date_calculated','publish_date','exclude','ExcludeFromLists','previousratingid','DataEntryAnalyst','Rpt_Comp_Date','Rpt_Comp_Emp','Reader2_Date','EmployeeID','Rpt_Ap_Date','Rpt_Ap_Emp','Rpt_Ver_Date','Incomplete','StatusID','DonorAdvisoryDate','DonorAdvisoryText','IRSControlID','ResultsID','DonorAdvisoryCategoryID','RatingTableID','CauseID','CNVersion','Direct_Support','Indirect_Support','Int_Expense','Depreciation','Assets_45','Assets_46','Assets_47c','Assets_48c','Assets_49','Assets_54','Liability_60'],axis=1)
test_X = test_X.drop(['ids','RatingID','erkey','AccountabilityID','RatingInterval','CN_Priority','Creation_Date','date_calculated','publish_date','exclude','ExcludeFromLists','previousratingid','DataEntryAnalyst','Rpt_Comp_Date','Rpt_Comp_Emp','Reader2_Date','EmployeeID','Rpt_Ap_Date','Rpt_Ap_Emp','Rpt_Ver_Date','Incomplete','StatusID','DonorAdvisoryDate','DonorAdvisoryText','IRSControlID','ResultsID','DonorAdvisoryCategoryID','RatingTableID','CauseID','CNVersion','Direct_Support','Indirect_Support','Int_Expense','Depreciation','Assets_45','Assets_46','Assets_47c','Assets_48c','Assets_49','Assets_54','Liability_60'],axis=1)
train_X = train_X.drop(['MemDues','Pymt_Affiliates'],axis = 1)
test_X = test_X.drop(['MemDues','Pymt_Affiliates'],axis = 1)
# Commit this and commit with different values of XGBRegressor
train_X = train_X.drop(['RatingYear','BaseYear'],axis = 1)
test_X = test_X.drop(['RatingYear','BaseYear'],axis = 1)
train_Y = train_Y.drop(['ids','ATScore'],axis=1)
train_Y = train_Y.drop([5448],axis=0)
train_X = train_X.drop([5448],axis=0)
my_imputer = SimpleImputer(missing_values=np.nan)
train_X = my_imputer.fit_transform(train_X)
test_X = my_imputer.transform(test_X)

# Random Forrest
#my_model = RandomForestRegressor(n_estimators=200, max_depth=5)
# XGBoost Tree
my_model = XGBRegressor(max_depth=8, learning_rate=0.05, n_estimators=1000, silent=True,
    objective='reg:linear', booster='gbtree', n_jobs=10, nthread=None, gamma=0, 
    min_child_weight=0, max_delta_step=0, subsample=0.8, colsample_bytree=0.8, 
    colsample_bylevel=0.8, reg_alpha=0, reg_lambda=0, scale_pos_weight=1,
    base_score=0.5, random_state=1, seed=None, missing=np.nan, importance_type='gain')

#my_model.fit(train_X[100:],train_Y[100:],eval_set = [(train_X[0:100],train_Y[0:100])],verbose=True,early_stopping_rounds=10) # For consistent result / comparison
my_model.fit(train_X,train_Y)

# Output
predictions = my_model.predict(test_X)
predictions[np.isnan(predictions)]= np.nanmean(predictions)
#predictions += np.array((np.random.randn(2126)*0.5)) #lol
out = open('submission.csv',"w")
out.write("Id,OverallScore\n")
for num in range(1,2127):
    out.write(str(num)+','+str(round(predictions[num-1],2))+'\n')
