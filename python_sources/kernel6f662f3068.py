#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error
from sklearn import preprocessing, feature_selection
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression,LassoCV
from sklearn.model_selection import train_test_split,GridSearchCV
import math
from scipy import stats
sns.set()
UCI_data=pd.read_csv("/kaggle/input/UCI_data.csv",index_col=0,parse_dates=True)
UCI_data.head()


# In[ ]:


UCI_data.info()


# In[ ]:


UCI_data.describe()


# In[ ]:


plt.figure()
UCI_data.hist(figsize=(20,20))
plt.show()


# In[ ]:


UCI_data.plot(kind='box',subplots=True,figsize=(20,30))
plt.show()


# 

# In[ ]:


UCI_data['Hours']=UCI_data.index.hour
UCI_data['Weekday']=UCI_data.index.weekday
UCI_data['Month']=UCI_data.index.month
UCI_data['Day']=UCI_data.index.day
UCI_data['Minutes']=UCI_data.index.minute
UCI_data['Years']=UCI_data.index.year


# In[ ]:



sns.boxplot(x=UCI_data['Weekday'], y=UCI_data['TARGET_energy'],data=UCI_data)
plt.title("Plot of energy fluctuation  with Weekdays")
plt.ylabel("TARGET_energy")
plt.xlabel("Weekdays")
plt.xticks(np.arange(7))
plt.show()


# In[ ]:


"Hourly distribution of energy use",
sns.boxplot(x=UCI_data['Hours'], y=UCI_data['TARGET_energy'],data=UCI_data)
plt.title("Plot of energy fluctuation  with Hours")
plt.ylabel("TARGET_energy")
plt.xlabel("Hours")
plt.show()


# In[ ]:


"Correlation matrix to study the nature of the relationship \n",
fig,ax=plt.subplots(figsize=(20,20))
sns.heatmap(UCI_data.corr(),vmin=-1, vmax=1, center=0,cmap="YlOrRd",annot=True, fmt='.2f')
plt.show()


# In[ ]:


"Monthy peak of energy usage",
figure,ax=plt.subplots(figsize=(5,5))
UCI_data['TARGET_energy'].resample('M').sum().plot(kind='bar',color='orange')
x=np.arange(5)
plt.title("Plot of energy use through out the time periods on monthly basis")
plt.ylabel("Energy Consumption in Wh")
plt.xlabel("Months")
ax.set_xticks(x)
ax.set_xticklabels(['Jan','Feb','Mar','Apr','May'])
plt.show()


# In[ ]:


#"Separation of target and predictors\n",
target=UCI_data['TARGET_energy']
predictors=UCI_data.drop('TARGET_energy',axis=1)


# In[ ]:


#"Study of peak hours of usage of energy"
plt.bar(predictors['Hours'],target,color="red")
plt.title("Plot of energy fluctuation  with Hours of Day")
plt.ylabel("TARGET_energy")
plt.xlabel("Hours")
plt.show()


# In[ ]:


#Relationship study between target variable and predictors\n",
for i in predictors.columns:
 plt.scatter(predictors[i],target,color="red")
 plt.title("Plot for"+i+"with Target Energy")
 plt.ylabel("TARGET_energy")
 plt.xlabel(i)
 plt.show()


# In[ ]:


UCI_data.reset_index(inplace=True)# Dropping the date feature\n",
UCI_data.drop('date',axis=1,inplace=True)


# In[ ]:


UCI_predictors_train_valid,UCI_predictors_test,UCI_target_energy_train_valid,UCI_target_energy_test=train_test_split(predictors,target,test_size=0.1,random_state=42)


# In[ ]:


#Separation of training and Validation sets.\n",
UCI_predictors_train,UCI_predictors_valid,UCI_target_energy_train,UCI_target_energy_valid=train_test_split(UCI_predictors_train_valid,UCI_target_energy_train_valid,test_size=0.2,random_state=42)


# In[ ]:


#Normalization process done for feature normalization",
mx=MinMaxScaler()
X_train_transformed=mx.fit_transform(UCI_predictors_train)
X_valid_transformed=mx.transform(UCI_predictors_valid)
X_test_transformed=mx.transform(UCI_predictors_test)


# In[ ]:


#Feature selection",
selector=feature_selection.SelectKBest(feature_selection.f_regression,k=20)
X_train_new=selector.fit_transform(X_train_transformed,UCI_target_energy_train)
X_valid_new=selector.transform(X_valid_transformed)
X_test_new=selector.transform(X_test_transformed)
print("Shape of training features: ",X_train_new.shape)
print("Shape of validation features: ",X_valid_new.shape)
skb_mask=selector.get_support()
print(skb_mask)
out_list=[]
skb_features = []
for bool,feature in zip(skb_mask, UCI_predictors_train.columns):
 if bool:
  skb_features.append(feature)
print('Optimal number of features :',len(skb_features))
print('Best features :',skb_features)
for col in UCI_predictors_train.columns:
 skb_pvalues=stats.pearsonr(UCI_predictors_train[col],UCI_target_energy_train)
 out_list.append([col,skb_pvalues[0],skb_pvalues[1]])
 p_value_df=pd.DataFrame(out_list,columns=["Features","Correlation","P-values"])
 print("Dataframe for p-values of features:","\n",p_value_df.head())
fig, ax = plt.subplots(figsize=(15,8))
plt.bar(p_value_df["Features"],p_value_df["P-values"],color="green")
plt.xticks(range(len( UCI_predictors_train.columns)))
plt.xticks(rotation=45, ha='right')
plt.title("Plot of Features against their p-values from Select K Best Method")
plt.ylabel("p-values")
plt.xlabel("Features")
plt.show()


# In[ ]:


#Performance of Linear Regression with all features
   
lasso = LassoCV(alphas=[0.006, 0.01, 0.03, 0.06,0.25,0.5, 0.1,0.3, 0.6, 1],max_iter=100000, cv=5)
lasso.fit(X_train_transformed,UCI_target_energy_train)
y_predicted = lasso.predict(X_valid_transformed)
plt.figure(figsize=(10, 5))
plt.scatter(UCI_target_energy_valid, y_predicted, s=20)
plt.title("Actual Vs Predicted value without feature selection")
plt.xlabel('Actual Value')
plt.ylabel('Predicted Value')


# In[ ]:


#Performance of Linear Regression with selected features
lasso = LassoCV(alphas=[0.006, 0.01, 0.03, 0.06,0.25,0.5, 0.1,0.3, 0.6, 1],max_iter=100000, cv=5)
lasso.fit(X_train_new,UCI_target_energy_train)
y_predicted = lasso.predict(X_valid_new)
plt.figure(figsize=(10, 5))
plt.scatter(UCI_target_energy_valid, y_predicted, s=20)
plt.title("Actual Vs Predicted value with feature selection")
plt.xlabel('Actual Value')
plt.ylabel('Predicted Value')


# In[ ]:


#Function for Random Forest with Grid Search cross-validation\n",
def modelselection_forest(model,model_param):
 model_design=GridSearchCV(model,model_param,cv=5,n_jobs=-1)
 model_design.fit(X_train_new,UCI_target_energy_train)
 result_predict=model_design.predict(X_valid_new)
 print("RMSE score for RF",math.sqrt(mean_squared_error(UCI_target_energy_valid,result_predict)))
 print("Train data score is ",model_design.score(X_train_new,UCI_target_energy_train))
 print("Validation data score is ",model_design.score(X_valid_new,UCI_target_energy_valid))
 print("The best parameter for the model is")
 print(model_design.best_params_)


# In[ ]:


"#Function for Gradient Boosting Model with Grid Search cross-validation\n",
def modelselection_gbm(model,model_param):
 model_design=GridSearchCV(model,model_param,cv=5,n_jobs=-1)
 model_design.fit(X_train_new,UCI_target_energy_train)
 result_predict=model_design.predict(X_valid_new)
 print("RMSE score for GBM",math.sqrt(mean_squared_error(UCI_target_energy_valid,result_predict)))
 print("Train data score is ",model_design.score(X_train_new,UCI_target_energy_train))
 print("Validation data score is ",model_design.score(X_valid_new,UCI_target_energy_valid))
 print("The best parameter for the model is")
 print(model_design.best_params_)


# In[ ]:


"#Function for KNN model with Grid Search cross-validation\n",
def modelselection_knn(model,model_param):
 model_design=GridSearchCV(model,model_param,cv=5,n_jobs=-1)
 model_design.fit(X_train_new,UCI_target_energy_train)
 result_predict=model_design.predict(X_valid_new)
 print("RMSE score for KNN",math.sqrt(mean_squared_error(UCI_target_energy_valid,result_predict)))
 print("Train data score is ",model_design.score(X_train_new,UCI_target_energy_train))
 print("Validation data score is ",model_design.score(X_valid_new,UCI_target_energy_valid))
 print("The best parameter for the model is")
 print(model_design.best_params_)


# In[ ]:


#Parameter defining for models and calling for cross validations\n",
parameter_for_gradient_boost={'n_estimators':[80,100,200],'loss':['ls','lad','huber','quantile'],'learning_rate':[0.1,0.5],'max_features':['auto','sqrt'],'criterion':['friedman_mse','mse']}
parameter_for_knn={'n_neighbors':[2,3,4,5,6,7],'weights':['uniform','distance'],'algorithm':['auto', 'ball_tree', 'kd_tree', 'brute'],'p':[1,2]}
parameter_for_forest={'max_features':['auto','sqrt'],'criterion':['mse']}
neigbour_regressor=KNeighborsRegressor()
forest_regressor=RandomForestRegressor()
gradient_regressor=GradientBoostingRegressor()
print("Result for Random Forest Regressor")
modelselection_forest(forest_regressor,parameter_for_forest)
print("Result for GradientBoost Regressor")
modelselection_gbm(gradient_regressor,parameter_for_gradient_boost)
print("Result for KNN Regressor"),
modelselection_knn(neigbour_regressor,parameter_for_knn)


# In the above result it can be seen that Random Forest and KNN does reflect much of overfitting.

# In[ ]:


print("Prediction for the rest unseen data")
print("******** For GBM Model*********")
model=GradientBoostingRegressor(n_estimators=200,criterion='friedman_mse',learning_rate=0.5,loss='ls',max_features='sqrt')
model.fit(X_train_new,UCI_target_energy_train)
y_pred_new=model.predict(X_test_new)
print("RMSE score for the unseen test dataset",math.sqrt(mean_squared_error(UCI_target_energy_test,y_pred_new)))
print("R2 score the unseen test dataset :",r2_score(UCI_target_energy_test,y_pred_new))


# In[ ]:


#Prediction study over unseen data\n",
plt.scatter(UCI_target_energy_test, y_pred_new, s=20)
plt.title("Actual Vs Predicted value for GBM")
plt.xlabel('Actual Value')
plt.ylabel('Predicted Value')


# In[ ]:


print("Prediction for the rest unseen data")
print("******** For Random Forest Model*********")
model=RandomForestRegressor(criterion='mse',max_features='sqrt')
model.fit(X_train_new,UCI_target_energy_train)
y_pred_new_rf=model.predict(X_test_new)
print("RMSE score for the unseen test dataset",math.sqrt(mean_squared_error(UCI_target_energy_test,y_pred_new_rf)))
print("R2 score the unseen test dataset :",r2_score(UCI_target_energy_test,y_pred_new_rf))


# In[ ]:


#Prediction over unseen data\n",
plt.scatter(UCI_target_energy_test, y_pred_new_rf, s=20)
plt.title("Actual Vs Predicted value for Random Forest")
plt.xlabel('Actual Value')
plt.ylabel('Predicted Value')

