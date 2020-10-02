#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


df_train=pd.read_csv("../input/Train.csv")
df_test=pd.read_csv("../input/Test.csv")


# In[ ]:


print("The shape of the train dataset is"+str(df_train.shape))
print("The shape of the test dataset is"+str(df_test.shape))


# In[ ]:


#lets do some pre processing first
df_train.describe()


# In[ ]:


#df_train.drop_duplicates(keep='first',subset=["date_time","traffic_volume"],inplace=True)#
df_train.shape
#df_train.drop(["date_time"],inplace=True,axis=1)


# In[ ]:


#df_train["weather_description"]=df_train["weather_description"].str.replace("Sky is Clear","sky is clear")#


# In[ ]:


#df_train["weather_description"].unique()
#df_test["is_holiday"].unique()


# In[ ]:


df_train.info()


# In[ ]:


#df_train["weather_description"].unique()#


# In[ ]:


from sklearn.preprocessing import LabelEncoder
LE=LabelEncoder()
#train
df_train["is_holiday"]=LE.fit_transform(df_train["is_holiday"])
#df_train["weather_description"] = LE.fit_transform(df_train["weather_description"])
df_train["weather_type"] = LE.fit_transform(df_train["weather_type"])
#test
df_test["is_holiday"]=LE.fit_transform(df_test["is_holiday"])
#df_test["weather_description"] = LE.fit_transform(df_test["weather_description"])
df_test["weather_type"] = LE.fit_transform(df_test["weather_type"])


# In[ ]:


from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error,mean_squared_log_error
from sklearn.model_selection import train_test_split


# In[ ]:


#df_train.head(10)


# In[ ]:





# In[ ]:


#splitting the test and train dataset
train=df_train[0:29650]
val=df_train[29650:]
train.reset_index(inplace=True,drop=True)
val.reset_index(inplace=True,drop=True)


# In[ ]:


#train.tail(10)
#val.head(5)


# In[ ]:


#train.head(5)


# In[ ]:


x_train=train.drop(["traffic_volume","date_time","weather_description"],axis=1)
y_train=train["traffic_volume"]
x_val=val.drop(["traffic_volume","date_time","weather_description"],axis=1)
y_val=val["traffic_volume"]
x_test=df_test.drop(["date_time","weather_description"],axis=1)
x_train.shape,y_train.shape,x_val.shape,y_val.shape


# In[ ]:


import math as m


# In[ ]:


#train.isna().sum(),val.isna().sum()


# In[ ]:


#rfmodel=RandomForestRegressor(n_estimators= 10, min_samples_split= 2, min_samples_leaf= 5, max_features= 4, max_depth= 8, bootstrap= False)
#rfmodel.fit(x_train,y_train)


# In[ ]:


#predicted=rfmodel.predict(x_test)


# In[ ]:


#rmse=m.sqrt(mean_squared_log_error(y_val,predicted))
#rmse


# In[ ]:


#predicted.shape##


# In[ ]:


#submission#


# In[ ]:


def report(result,n_top):
    for i in range(1,n_top+1):
        candidates=np.flatnonzero(result["rank_test_score"]==i)
        for cand in candidates:
            print("Model with rank {0}".format(i))
            print("The mean test score is {0:.5f} and the std_dev is{1:.5f}".format(result["mean_test_score"][cand],result["std_test_score"][cand]))
            print("the parameter of the {0}".format(result["params"][cand]))
            print("")
        


# In[ ]:


#rfreg=RandomForestRegressor(n_jobs=-1,verbose=2)


# In[ ]:


#random_grid = {'n_estimators': [2,4,6,8,10,12],
 #              'max_features': [2,4,6,8,10],
  #             'max_depth': [None,2,4,6,8,10],
   #            'min_samples_split': [2,3,4,6,8],
    #           "min_samples_leaf":[2,3,4,5,6],
     #         'bootstrap': [True,False]}


# In[ ]:


#n_iterations=30
#random_search=RandomizedSearchCV(rfreg,param_distributions=random_grid,cv=10,
#                                 n_iter=n_iterations)


# In[ ]:


#random_search.fit(x_train,y_train)


# In[ ]:


#report(random_search.cv_results_,10)


# In[ ]:


#xgboost parameters tuning 
#grid_search={ 
 #         
  ##          
    #           "n_estimators":[2,4,6,8,10],
     #        'min_child_weight':[1,3,5,6,7]}#


# In[ ]:


import math as m


# In[ ]:


#df_train["is_holiday"]=LE.fit_transform(df_train["is_holiday"])
#df_train["weather_description"] = LE.fit_transform(df_train["weather_description"])
#df_train["weather_type"] = LE.fit_transform(df_train["weather_type"])
#test
#df_test["is_holiday"]=LE.fit_transform(df_test["is_holiday"])
#df_test["weather_description"] = LE.fit_transform(df_test["weather_description"])
#df_test["weather_type"] = LE.fit_transform(df_test["weather_type"])


# In[ ]:


df_train.head()


# In[ ]:


tr_final=df_train.drop(["date_time","weather_description"],axis=1)


# In[ ]:


#tr_final.head(5)
#x_test.head()


# In[ ]:


tr_final_x=tr_final.drop(["traffic_volume"],axis=1)
tr_final_y=tr_final["traffic_volume"]


# In[ ]:


tr_final_x.reset_index(drop=True,inplace=True)
tr_final_y.reset_index(drop=True,inplace=True)


# In[ ]:


from xgboost import XGBRegressor
import xgboost as xgb

#xgmodel=XGBRegressor(booster='dart', colsample_bytree= 0.9, gamma= 0.0, learning_rate= 0.3,   
  #                   min_child_weight=3,n_estimators=10,
 #                    reg_alpha= 0.01, subsample= 0.9)
#xgmodel.fit(tr_final_x,tr_final_y)

#predicted=xgmodel.predict(x_test)
#rmse=m.sqrt(mean_squared_log_error(y_val,predicted))
#rmse


# In[ ]:


#gsearch1 = GridSearchCV(estimator=xgmodel,verbose=2,param_grid=grid_search
 #                       ,n_jobs=4, cv=5)
#gsearch1.fit(x_train,y_train)


# In[ ]:


#report(gsearch1.cv_results_,10)


# In[ ]:


#report(gsearch1.cv_results_,10)


# In[ ]:


#from sklearn.linear_model import LinearRegression

#lmodel=LinearRegression()
#lmodel.fit(x_train,y_train)
#predicted=lmodel.predict(x_val)
#rmse=m.sqrt(mean_squared_log_error(y_val,predicted))
#rmse


# In[ ]:


from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from xgboost import XGBRFRegressor
from sklearn.ensemble import ExtraTreesRegressor


# In[ ]:


#lets try stcking here now
reg1=RandomForestRegressor()
reg2=DecisionTreeRegressor()
reg3=AdaBoostRegressor()
reg4=XGBRFRegressor()
reg5=ExtraTreesRegressor()


# In[ ]:


#making the list of all the regressors


# In[ ]:


rows=tr_final_x.shape[0]
rows


# In[ ]:


layer1=pd.DataFrame({"reg1":np.zeros(rows),"reg2":np.zeros(rows),
                    "reg3":np.zeros(rows),"reg4":np.zeros(rows),"reg5":np.zeros(rows)})

#layer1=pd.DataFrame({'clf1':np.zeros(rows),'clf2':np.zeros(rows),'clf3':np.zeros(rows),
 #                   'clf4':np.zeros(rows),'clf5':np.zeros(rows)})


# In[ ]:


layer1


# In[ ]:


from sklearn.model_selection import KFold


# In[ ]:


kf=KFold(n_splits=10)


# In[ ]:


#fold=1
#for train,left_out_chunk in kf.split(tr_final_x):
 #   print('fold number : ', fold)
    
  #  for i,clf in enumerate(Algo):
   #     print('Algo number :',i+1)
        
    #    x_train_train=tr_final_x.loc[train]
     #   y_train_train=tr_final_y[train]
      #  x_train_left_out_chunk=tr_final_x.loc[left_out_chunk]
        
       # clf.fit(x_train_train,y_train_train)
       # p=clf.predict(x_train_left_out_chunk)[:,1]
        
        #layer1.iloc[left_out_chunk,i]=p
        
    #fold+=1  
    


# In[ ]:


reg1.fit(tr_final_x,tr_final_y)
reg2.fit(tr_final_x,tr_final_y)
reg3.fit(tr_final_x,tr_final_y)
reg4.fit(tr_final_x,tr_final_y)
reg5.fit(tr_final_x,tr_final_y)


# In[ ]:


pre1=reg1.predict(x_val)
pre2=reg2.predict(x_val)
pre3=reg3.predict(x_val)
pre4=reg4.predict(x_val)
pre5=reg5.predict(x_val)


# In[ ]:


pred1=reg1.predict(x_test)
pred2=reg2.predict(x_test)
pred3=reg3.predict(x_test)
pred4=reg4.predict(x_test)
pred5=reg5.predict(x_test)


# In[ ]:


stacked_pred=np.column_stack((pre1,pre2,pre3,pre4,pre5))
stacked_test=np.column_stack((pred1,pred2,pred3,pred4,pred5))


# In[ ]:


stacked_pred.shape,stacked_test.shape


# In[ ]:


from sklearn.linear_model import LinearRegression


# In[ ]:


meta_model=LinearRegression()


# In[ ]:


meta_model.fit(stacked_pred,y_val)


# In[ ]:


predicted=meta_model.predict(stacked_test)


# In[ ]:


col=df_test["date_time"]
submission=pd.DataFrame([col,predicted]).T
#submission.rename({date_time:"date_time",0:"traffic_volume"})
submission.to_csv("submission_metro.csv",header=["date_time","traffic_volume"],index=False)

